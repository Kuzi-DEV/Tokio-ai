"""`tokio-ai-backtest`: run check_backtest on a CSV, no Python required.

    tokio-ai-backtest pnl.csv --trials 40
    tokio-ai-backtest equity.csv --equity --periods-per-year 52
    tokio-ai-backtest trades.csv --contracts price won --sizes count --fees 0.003

The CSV needs a header row. Every column that parses as numbers is a
strategy variant (per-bar returns, or with --equity an equity curve / price
series that gets converted to returns); a non-numeric column such as a date
is skipped. Empty cells are missing. "1.5%" is read as 0.015.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

from ._stdio import force_utf8_stdio


_MISSING = {"", "-", "na", "n/a", "nan", "null", "none", "#n/a"}


def _num(cell: str) -> float | None:
    """A float, NaN for an empty or missing-data cell, or None if the cell isn't a number."""
    c = cell.strip().replace(",", "")
    if c.lower() in _MISSING:
        return math.nan
    pct = c.endswith("%")
    try:
        v = float(c[:-1] if pct else c)
    except ValueError:
        return None
    return v / 100 if pct else v


def read_columns(path: str) -> dict[str, list[float]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path}: needs a header row and at least one data row")
    header, body = rows[0], rows[1:]
    cols: dict[str, list[float]] = {}
    for j, name in enumerate(header):
        cells = [r[j] if j < len(r) else "" for r in body]
        vals = [_num(c) for c in cells]
        bad = [c for c, v in zip(cells, vals) if v is None]
        numeric = sum(1 for v in vals if v is not None and not math.isnan(v))
        label = name.strip() or f"col{j}"
        if bad and numeric > len(bad):
            # Mostly numbers with a few stray cells: almost certainly a strategy
            # column with odd missing-data markers. Refuse rather than silently
            # dropping a variant, which would also undercount the trials.
            raise ValueError(
                f"column {label!r} is numeric except {len(bad)} cell(s) such as {bad[0]!r}; "
                f"blank them or use NaN"
            )
        if bad or not numeric:
            continue  # a date, a label, or an empty column
        cols[label] = vals
    if not cols:
        raise ValueError(f"{path}: no numeric columns found")
    return cols


def read_raw(path: str) -> dict[str, list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path}: needs a header row and at least one data row")
    header = [h.strip() for h in rows[0]]
    return {h: [r[j].strip() if j < len(r) else "" for r in rows[1:]] for j, h in enumerate(header)}


_TRUE = {"1", "true", "yes", "won", "win", "y", "t"}
_FALSE = {"0", "false", "no", "lost", "loss", "n", "f"}


def _contracts(a) -> str:
    from .contracts import check_contracts

    raw = read_raw(a.csv[0])
    price_col, outcome_col = a.contracts

    def col(name):
        if name not in raw:
            raise ValueError(f"column {name!r} not found; have {list(raw)}")
        return raw[name]

    def nums(name):
        out = []
        for i, c in enumerate(col(name)):
            v = _num(c)
            if v is None or math.isnan(v):
                raise ValueError(f"column {name!r}, row {i + 2}: {c!r} is not a number")
            out.append(v)
        return out

    prices = nums(price_col)
    if a.cents or max(prices) > 1:
        prices = [p / 100 for p in prices]
    outcomes = []
    for i, c in enumerate(col(outcome_col)):
        lc = c.lower()
        if lc not in _TRUE | _FALSE:
            raise ValueError(f"column {outcome_col!r}, row {i + 2}: {c!r} is not a win/loss value")
        outcomes.append(lc in _TRUE)
    sizes = nums(a.sizes) if a.sizes else 1.0
    fees: float | list[float] = 0.0
    if a.fees is not None:
        v = _num(a.fees)
        fees = v if v is not None and not math.isnan(v) else nums(a.fees)
    groups = col(a.groups) if a.groups else None
    exits = None
    if a.exits:
        vals = []
        for i, c in enumerate(col(a.exits)):
            v = _num(c)
            if v is None:
                raise ValueError(f"column {a.exits!r}, row {i + 2}: {c!r} is not a number")
            vals.append(v)
        # One decision for the whole column, like the prices: cents if asked
        # for, or if any exit is above 1.
        present = [v for v in vals if not math.isnan(v)]
        scale = 100 if (a.cents or (present and max(present) > 1)) else 1
        exits = [None if math.isnan(v) else v / scale for v in vals]
    return str(check_contracts(prices, outcomes, sizes=sizes, fees=fees, groups=groups,
                               exit_values=exits, alpha=a.alpha))


def to_returns(curve: list[float]) -> list[float]:
    out = [math.nan]
    for prev, cur in zip(curve, curve[1:]):
        out.append(cur / prev - 1 if prev and not math.isnan(prev) and not math.isnan(cur) else math.nan)
    return out


def _stem(path: str) -> str:
    import os

    return os.path.splitext(os.path.basename(path))[0]


def _tradingview_files(paths: list[str]) -> bool:
    """True if every file is a TradingView trade export; an error if only some are."""
    from .trades import is_tradingview_header

    flags = []
    for p in paths:
        if p.lower().endswith((".xlsx", ".xls")):
            flags.append(True)
            continue
        with open(p, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
        flags.append(is_tradingview_header(header))
    if any(flags) and not all(flags):
        raise ValueError("mix of TradingView exports and plain CSVs; pass one kind")
    return all(flags)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser(
        prog="tokio-ai-backtest",
        description="Is this backtest distinguishable from luck, given how many variants you tried?",
    )
    ap.add_argument("csv", nargs="+",
                    help="CSV with a header row and one numeric column per strategy variant; or one "
                         "or more TradingView Strategy Tester 'List of trades' exports (.csv/.xlsx), "
                         "recognised by their columns")
    ap.add_argument("--trials", type=int, help="variants tried in total (default: the columns given)")
    ap.add_argument("--columns", nargs="+", help="only these columns are variants")
    ap.add_argument("--benchmark", help="column to subtract from every variant first")
    ap.add_argument("--equity", action="store_true",
                    help="columns are equity curves or prices, not per-bar returns")
    ap.add_argument("--periods-per-year", type=float, default=None,
                    help="bars per year, for the annualized Sharpe (default 252 daily; 52 weekly, 12 "
                         "monthly; for trade lists, trades per year from the dates)")
    ap.add_argument("--basis", choices=("position", "equity"), default="position",
                    help="TradingView exports: each trade's return on the position's value (default) "
                         "or on account equity before the trade")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--contracts", nargs=2, metavar=("PRICE", "OUTCOME"),
                    help="binary-contract mode: the price paid and whether that side won "
                         "(1/0, true/false, won/lost); runs the exact check_contracts test")
    ap.add_argument("--sizes", help="--contracts: column of contracts per bet")
    ap.add_argument("--fees", help="--contracts: fee per contract, a number or a column, always "
                                    "as a fraction of $1 (0.003 = 0.3c), even with --cents")
    ap.add_argument("--groups", help="--contracts: column naming the contract, to merge fills")
    ap.add_argument("--exits", help="--contracts: column of what each early exit (stop-loss, "
                                     "take-profit) returned per contract, net of fees; blank if "
                                     "held to settlement")
    ap.add_argument("--cents", action="store_true",
                    help="--contracts: prices are in cents (assumed if any price is above 1)")
    a = ap.parse_args(argv)

    if a.contracts:
        try:
            print(_contracts(a))
        except (OSError, ValueError) as e:
            print(f"tokio-ai-backtest: {e}", file=sys.stderr)
            return 2
        return 0

    from .backtest import check_backtest  # numpy import deferred until arguments are valid

    try:
        if _tradingview_files(a.csv):
            from .trades import read_tradingview

            lists = {_stem(f): read_tradingview(f, basis=a.basis) for f in a.csv}
            if len(lists) != len(a.csv):
                raise ValueError("two exports have the same file name; rename one")
            res = check_backtest(lists if len(lists) > 1 else next(iter(lists.values())),
                                 trials=a.trials, periods_per_year=a.periods_per_year, alpha=a.alpha)
            print(res)
            return 0
        if len(a.csv) > 1:
            raise ValueError("several files are only for TradingView exports; put variants in columns")
        cols = read_columns(a.csv[0])
        if a.equity:
            cols = {k: to_returns(v) for k, v in cols.items()}
        bench = None
        if a.benchmark:
            if a.benchmark not in cols:
                raise ValueError(f"benchmark column {a.benchmark!r} not found; have {list(cols)}")
            bench = cols.pop(a.benchmark)
        if a.columns:
            missing = [c for c in a.columns if c not in cols]
            if missing:
                raise ValueError(f"columns not found: {missing}; have {list(cols)}")
            cols = {c: cols[c] for c in a.columns}
        if not cols:
            raise ValueError("no strategy columns left to test")
        variants = cols if len(cols) > 1 else next(iter(cols.values()))
        res = check_backtest(variants, trials=a.trials, benchmark=bench,
                             periods_per_year=a.periods_per_year, alpha=a.alpha)
    except (OSError, ValueError) as e:
        print(f"tokio-ai-backtest: {e}", file=sys.stderr)
        return 2
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
