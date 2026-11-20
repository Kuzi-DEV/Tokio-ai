"""v3 trajectories: v2, plus planted-error turns marked train=False, and NOT REPORTABLE
explained by what the positions actually were (flat-zero vs constant exposure).

    python training/agent/make_data_v3.py --per-pair 2 --workers 4

Same guarantees as make_data.py: every trajectory is executed through the web
terminal's own Python shim, every tool output is real, every final answer is
built from the Lab's result, and a trajectory is kept only if the Lab counted
exactly the variants asked for with the right baseline. Writes data/v3/.

An assistant turn that submits planted buggy code carries "train": False: the
model sees it as context (so it learns to read the error and fix it) but is never
trained to write it. v2 trained on those turns and the model learned the bugs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "evals" / "agent"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import families as TEST_F  # noqa: E402
import families_v2 as V  # noqa: E402
import train_families as T1  # noqa: E402  (ticker lists)
from make_data import CASH_SUFFIX, call, clean, fmt_list, pct  # noqa: E402

assert not set(V.F) & set(TEST_F.FAMILIES)
assert not (set(T1.TRAIN_SYMBOLS) | set(T1.DEV_SYMBOLS)) & T1.TEST_SYMBOLS

FN_NAMES = ["signal", "strategy", "rule", "{fam}", "{fam}", "{fam}_signal", "pos_{fam}"]
WRAP = ["{q}", "{q}", "{q}", "Quick question: {q}", "Can you check this? {q}", "{q} Keep it short.", "{q} Be honest if it's nothing."]


# ---------------------------------------------------------------- errors
# Each mutates working code into code that fails for a real reason; the fix turn
# explains the cause in one line and resubmits corrected code.
def _close_key(code):
    return code.replace("d.close", 'd["Close"]', 1) if "d.close" in code else None


def _close_attr(code):
    return code.replace("d.close", "d.Close", 1) if "d.close" in code else None


def _talib(code):
    return "import talib\n" + code


def _from_lab(code):
    return "from lab import lab\n" + code


def _lookahead(code):
    lines = code.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"^(    return )(.+)$", lines[i])
        if m:
            lines[i] = f"{m.group(1)}({m.group(2)}).shift(-1).fillna(0.0)"
            return "\n".join(lines)
    return None


def _bad_param(code):
    m = re.search(r"L\.(sweep|run)\((\w+), (\w+)=", code)
    if not m:
        return None
    return code[: m.start(3)] + m.group(3) + "s" + code[m.end(3):]


ERRORS = {
    "close_key": (_close_key, "The price columns are lowercase (`close`, not `Close`). Fixing that."),
    "close_attr": (_close_attr, "The column is `close` in lowercase; `d.Close` doesn't exist. Rerunning with the right name."),
    "talib": (_talib, "TA-Lib isn't installed here, and I don't need it: the indicator is a few lines of pandas."),
    "from_lab": (_from_lab, "`lab` is already defined in the namespace; there's nothing to import."),
    "lookahead": (_lookahead, "The Lab caught lookahead: `.shift(-1)` reads tomorrow's value. Positions are already filled at the next open, so the signal needs no shift."),
    "bad_param": (_bad_param, "I passed the wrong parameter name; the signal's argument is spelled differently. Fixing the call."),
}


def build_code(src, fn, sym, vs, grid, lab_kw=True):
    lab_args = f"'{sym}', vs='cash'" if (vs == "cash" and lab_kw) else f"'{sym}'"
    combos = 1
    for v in grid.values():
        combos *= len(v)
    if combos > 1:
        run = f"L.sweep({fn}, " + ", ".join(f"{k}={v!r}" for k, v in grid.items()) + ")"
    else:
        run = f"L.run({fn}, " + ", ".join(f"{k}={v[0]!r}" for k, v in grid.items()) + ")"
    return f"{src}\n\nL = lab({lab_args})\n{run}\nprint(L.check())", combos


def headline(rng, verdict, sym, vs, n_var, name):
    base = f"holding {sym} at the same exposure" if vs == "market" else "cash"
    if verdict == "SIGNIFICANT":
        return f"**SIGNIFICANT.** {name} beats {base} " + (f"even after correcting for the {n_var} variants tested." if n_var > 1 else "in its single test.")
    if verdict == "NOT SIGNIFICANT":
        return rng.choice([f"**NOT SIGNIFICANT.** No variant beats {base} once the {n_var} variant{'s' if n_var > 1 else ''} tested are accounted for.",
                           f"**NOT SIGNIFICANT.** The evidence doesn't support this on {sym}.",
                           f"**NOT SIGNIFICANT.** Nothing here you can tell apart from luck against {base}."])
    return "**NOT REPORTABLE.**"


def answer(rng, sym, vs, n_var, L, r, holdout, note, check_text):
    from dataclasses import replace
    from tokio_ai.lab import _label
    best = r.best
    if n_var == 1:
        best = replace(best, name=_label(L.runs[0]))
    idx = L.runs[0].returns.index
    sample = (f"Sample: {sym} daily bars {idx[0].date()} to {idx[-1].date()} ({best.n_bars} bars, the first 75% of the history); "
              f"{n_var} variant{'s' if n_var > 1 else ''} tried, tested {'against holding it at the same exposure' if vs == 'market' else 'against cash'}, "
              "next-open fills, 10 bp costs.")
    parts = [headline(rng, r.verdict, sym, vs, n_var, best.name)]
    if r.verdict == "NOT REPORTABLE":
        why = check_text.strip().split("\n")[0]
        pos = L.runs[0].positions.dropna()
        never = float(pos.abs().max()) < 1e-12
        if vs == "market" and not never and float(pos.std()) < 1e-9:
            reason = (f"A constant position has no timing to test: against holding {sym} at the same exposure its returns are identical, "
                      "so there is nothing to measure.")
        else:
            reason = (f"The position was zero on every bar: the rule never triggered on {sym} in this sample, "
                      "so there are no returns to test.")
        parts += [f"The Lab says: \"{why}\"", reason, sample,
                  ("This isn't evidence for or against the idea; the test simply has nothing to work with. A shallower threshold or a longer history would." if never else "This isn't evidence for or against the idea; the test simply has nothing to work with.")]
        return "\n\n".join(p for p in parts if p)
    if note:
        parts.append(note)
    nums = [f"corrected p = {pct(r.p_value)}",
            f"best variant {best.name}: Sharpe {best.sharpe:.2f} {'vs the market' if vs == 'market' else 'absolute'}",
            f"haircut Sharpe {best.haircut_sharpe:.2f}"]
    pbo = getattr(r.overfitting, "pbo", None) if r.overfitting is not None else None
    if n_var > 1 and pbo is not None:
        nums.append(f"probability of overfitting {pbo:.2f}")
    parts += ["- " + "\n- ".join(nums), sample]
    if holdout is not None:
        parts.append(f"On the untouched last 25% ({holdout.best.n_bars} bars), {best.name} came out **{holdout.verdict.lower()}** "
                     f"(p = {pct(holdout.p_value)}, Sharpe {holdout.best.sharpe:.2f}). "
                     + ("It held up out of sample." if holdout.verdict == "SIGNIFICANT" else "The in-sample result did not carry into the holdout; treat it as fragile."))
    if r.verdict == "SIGNIFICANT":
        parts.append("This is a research result on past data, not a recommendation to trade it.")
    elif vs == "market":
        parts.append(rng.choice([f"Any gains came from {sym}'s own move, not from the timing.",
                                 "The timing rule doesn't add anything you can tell apart from luck.",
                                 "Whatever it earned, holding the asset at the same exposure earned too."]))
    else:
        parts.append("Whatever it earned is not distinguishable from zero on this sample.")
    return "\n\n".join(parts)


def build(task):
    shim = build.shim
    rng = random.Random(task["seed"])
    sym, vs, grid, fam = task["symbol"], task["vs"], task["grid"], task["family"]
    fn = task["fn"]
    src = task["src"].replace("{fn}", fn)
    shim["reset_agent"]()
    msgs = [{"role": "system", "content": common.system_prompt()}, {"role": "user", "content": task["question"]}]
    n = 0

    def tool(cid, out):
        msgs.append({"role": "tool", "tool_call_id": cid, "content": clean(out)})

    def py(code, content="", train=True):
        nonlocal n
        n += 1
        m = {"role": "assistant", "content": content, "tool_calls": [call(f"c{n}", "run_python", {"code": code})]}
        if not train:
            m["train"] = False
        msgs.append(m)
        out = json.loads(shim["run_code"](code, 0))
        tool(f"c{n}", out["output"])
        return out

    n += 1
    msgs.append({"role": "assistant", "content": rng.choice(["", "", f"Loading {sym}."]), "tool_calls": [call(f"c{n}", "get_prices", {"symbol": sym})]})
    tool(f"c{n}", shim["load_chart"](sym, common.chart_payload(sym)))

    good, n_var = build_code(src, fn, sym, vs, grid)
    lab_made = False
    content = task["note"] or ""
    for err in task["errors"]:
        bad = ERRORS[err][0](good if not lab_made else build_code(src, fn, sym, vs, grid, lab_kw=False)[0])
        if bad is None:
            continue
        out = py(bad, content, train=False)
        if out["ok"]:
            return None
        content = ERRORS[err][1]
        lab_made = sym in shim["LABS"]
    final_code = build_code(src, fn, sym, vs, grid, lab_kw=not lab_made)[0]
    out = py(final_code, content)
    if not out["ok"]:
        return None
    L = shim["LABS"].get(sym)
    if L is None or len(L.runs) != n_var or L.vs != vs:
        return None
    r = L.check()
    holdout = None
    if r.verdict == "SIGNIFICANT":
        o = py("ho = L.final_test(L.best())\nprint(ho)", "It passes in-sample. Running the chosen variant once on the untouched holdout.")
        if not o["ok"]:
            return None
        holdout = getattr(shim["NS"]["ho"], "report", shim["NS"]["ho"])
    msgs.append({"role": "assistant", "content": answer(rng, sym, vs, n_var, L, r, holdout, task["note"], out["output"])})
    meta = {k: task[k] for k in ("family", "symbol", "vs", "grid", "vague", "errors", "split", "fn", "impl")}
    meta.update(verdict=r.verdict, p=round(float(r.p_value), 4), n_variants=n_var, holdout=None if holdout is None else holdout.verdict)
    return {"messages": msgs, "tools": common.TOOLS, "meta": meta}


def _init():
    build.shim = common.load_shim()


def _safe(task):
    try:
        return build(task)
    except Exception as e:
        print("drop", task["family"], task["symbol"], type(e).__name__, str(e)[:160], flush=True)
        return None


def tasks(per_pair, seed):
    rng = random.Random(seed)
    out = []
    for split, syms, k in (("train", T1.TRAIN_SYMBOLS, per_pair), ("dev", T1.DEV_SYMBOLS, 1)):
        for sym, fam in itertools.product(syms, V.FAMILY_NAMES):
            spec = V.F[fam]
            for j in range(k):
                vague = spec["vague"] is not None and j == 0 and rng.random() < 0.35
                if vague:
                    q, grid, note = spec["vague"]
                    question = q.format(sym=sym)
                else:
                    grid, note = rng.choice(spec["grids"]), None
                    question = rng.choice(spec["q"]).format(sym=sym, **{a: fmt_list(v) for a, v in grid.items()})
                vs = spec["vs"]
                if vs == "market" and spec["cash_ok"] and not vague and rng.random() < 0.15:
                    vs, question = "cash", question + rng.choice(CASH_SUFFIX)
                question = rng.choice(WRAP).format(q=question)
                impl = rng.randrange(len(spec["impls"]))
                fn = rng.choice(FN_NAMES).format(fam=fam)
                errors = []
                if rng.random() < 0.3:
                    errors.append(rng.choice(list(ERRORS)))
                    if rng.random() < 0.25:
                        errors.append(rng.choice([e for e in ERRORS if e != errors[0]]))
                out.append(dict(seed=rng.randrange(1 << 30), family=fam, symbol=sym, vs=vs, grid=grid, question=question,
                                vague=vague, note=note, src=spec["impls"][impl], impl=impl, fn=fn, errors=errors, split=split))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    todo = tasks(args.per_pair, args.seed)
    if args.limit:
        random.Random(1).shuffle(todo)
        todo = todo[: args.limit]
    for sym in set(T1.TRAIN_SYMBOLS) | set(T1.DEV_SYMBOLS):
        common.chart_payload(sym)
    outdir = HERE / "data" / "v3"
    outdir.mkdir(parents=True, exist_ok=True)
    files = {s: (outdir / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "dev")}
    stats, rejected = {}, {}
    from multiprocessing import Pool
    with Pool(args.workers, initializer=_init) as pool:
        for i, (task, rec) in enumerate(zip(todo, pool.imap(_safe, todo)), 1):
            if rec is None:
                rejected[task["family"]] = rejected.get(task["family"], 0) + 1
                continue
            m = rec["meta"]
            files[m["split"]].write(json.dumps(rec) + "\n")
            stats[(m["split"], m["verdict"])] = stats.get((m["split"], m["verdict"]), 0) + 1
            if i % 100 == 0:
                print(i, "/", len(todo), stats, flush=True)
    for f in files.values():
        f.close()
    print("done", stats, "rejected", rejected)


if __name__ == "__main__":
    main()
