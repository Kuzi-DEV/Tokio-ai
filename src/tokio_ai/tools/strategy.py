"""Strategies described in plain English, backtested in a Lab.

The agent turns a description ("buy when the 20-day average crosses above
the 100-day, sell when it crosses back") into a small `signal(d, **params)`
function and calls `backtest_strategy`. This module checks that code before
running it, keeps one `Lab` per symbol for the whole conversation (so every
variant the agent tries is counted, exactly as if you'd run them yourself),
and returns the lab's verdict as text.

The code check is a guard against a model writing something other than a
signal, not a security sandbox: no imports, no dunder access, no file,
process or network calls, and only `pd`, `np` and a few plain builtins in
scope. It runs on your machine, with your data, as part of your own agent.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "range": range, "len": len, "float": float,
    "int": int, "bool": bool, "round": round, "sum": sum, "zip": zip, "enumerate": enumerate,
    "list": list, "dict": dict, "tuple": tuple, "isinstance": isinstance, "sorted": sorted,
    "True": True, "False": False, "None": None,
}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "open", "input", "getattr", "setattr", "delattr",
                   "globals", "locals", "vars", "__import__", "breakpoint", "help", "memoryview"}
FORBIDDEN_ATTRS = {"to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_sql", "to_hdf",
                   "to_feather", "to_stata", "to_html", "to_xml", "to_clipboard", "to_markdown",
                   "load", "loadtxt", "genfromtxt", "fromfile", "tofile", "save", "savez",
                   "savez_compressed", "savetxt", "memmap", "system", "popen", "eval", "query",
                   "io", "os", "sys", "subprocess", "ctypeslib", "load_library", "testing", "f2py",
                   "distutils", "api", "plotting", "options", "set_option", "builtins", "pickle"}
MAX_CODE_CHARS = 4000


class UnsafeCode(ValueError):
    """The strategy code does something other than compute a signal."""


def compile_signal(code: str, name: str = "signal") -> Callable:
    """Check and compile code that defines `def signal(d, **params)`; return that function."""
    if len(code) > MAX_CODE_CHARS:
        raise UnsafeCode(f"strategy code is {len(code)} characters; keep it under {MAX_CODE_CHARS}")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"strategy code doesn't parse: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode("no imports: pd (pandas) and np (numpy) are already available")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
                             ast.Await, ast.Yield, ast.YieldFrom, ast.ClassDef, ast.Try)):
            raise UnsafeCode(f"{type(node).__name__} isn't allowed in a signal")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_CALLS):
            raise UnsafeCode(f"{node.id!r} isn't allowed in a signal")
        if isinstance(node, ast.Attribute) and (node.attr.startswith(("_", "read_"))
                                                 or node.attr in FORBIDDEN_ATTRS):
            raise UnsafeCode(f"attribute {node.attr!r} isn't allowed in a signal")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(f.name == "signal" for f in funcs):
        raise UnsafeCode("define `def signal(d, **params)` returning a position per bar (-1..1)")
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Expr, ast.Assign))]
    if others:
        raise UnsafeCode("only function definitions and constants at the top level")

    import numpy as np
    import pandas as pd

    scope: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    exec(compile(tree, f"<strategy {name}>", "exec"), scope)  # noqa: S102 - checked above
    fn = scope["signal"]
    fn.__name__ = name
    return fn


@dataclass
class StrategyBook:
    """One Lab per symbol for a conversation, and each strategy compiled once (so re-runs aren't new trials)."""

    holdout: float = 0.25
    labs: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)  # (symbol, name, code hash) -> function
    loader: Callable | None = None  # for tests: symbol -> DataFrame

    def lab(self, symbol: str, vs: str = "cash"):
        from ..lab import Lab

        key = (symbol.upper(), vs)
        if key not in self.labs:
            data = self.loader(symbol) if self.loader else symbol
            self.labs[key] = Lab(data, holdout=self.holdout, vs=vs)
        return self.labs[key]

    def signal(self, symbol: str, name: str, code: str) -> Callable:
        h = hashlib.sha1(code.encode()).hexdigest()[:10]
        key = (symbol.upper(), name, h)
        if key not in self.compiled:
            # same name, different code: a new strategy, so give it its own name
            clash = any(k[0] == key[0] and k[1] == name and k[2] != h for k in self.compiled)
            self.compiled[key] = compile_signal(code, f"{name}_{h[:4]}" if clash else name)
        return self.compiled[key]


def backtest_strategy(book: StrategyBook, symbol: str, name: str, code: str,
                      params: dict | None = None, vs: str = "cash") -> str:
    """Run a strategy (with an optional parameter grid) in the symbol's Lab and report on everything run so far."""
    lab = book.lab(symbol, vs)
    fn = book.signal(symbol, name, code)
    grid = {k: (v if isinstance(v, list) else [v]) for k, v in (params or {}).items()}
    runs = lab.sweep(fn, **grid) if grid else [lab.run(fn)]
    first, last = lab.in_sample.index[0], lab.in_sample.index[-1]
    head = [f"{symbol.upper()}: in-sample {first.date()} to {last.date()} ({len(lab.in_sample)} bars); "
            f"the last {len(lab.data) - lab.split} bars are held out, unseen.",
            f"Ran {len(runs)} variant(s) of {fn.__name__}; {len(lab.runs)} variant(s) run on "
            f"{symbol.upper()} in this conversation, all counted below.", ""]
    return "\n".join(head) + str(lab.check())


def final_test_strategy(book: StrategyBook, symbol: str, name: str, code: str,
                        params: dict | None = None, vs: str = "cash") -> str:
    """The single look at the held-out data for one chosen variant."""
    lab = book.lab(symbol, vs)
    fn = book.signal(symbol, name, code)
    return str(lab.final_test(fn, **(params or {})))
