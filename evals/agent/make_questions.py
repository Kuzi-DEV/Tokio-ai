"""Build questions.jsonl: 100 agent questions with reference answers.

    python evals/agent/make_questions.py

For each question, the reference grid is run through a fresh tokio_ai Lab
(vs='market' unless the question asks about cash) on prices frozen at
common.END_DATE, exactly as the agent's `lab()` would. The file records the
expected number of variants and the reference verdict.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import families as F  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GLD", "TLT", "EEM", "XLE", "BTC-USD", "NVDA"]


def fmt_list(xs):
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def specs(rng: random.Random):
    """Yield (family, symbol, fixed params, grid dict, question) tuples."""
    out = []
    for sym in SYMBOLS:
        fast = rng.choice([[10, 20], [20, 50], [10, 20, 50]])
        slow = rng.choice([[100, 200], [150, 200], [100]])
        out.append(("sma_cross", sym, {}, {"fast": fast, "slow": slow},
                    F.FAMILIES["sma_cross"]["q"].format(sym=sym, fast_list=fmt_list(fast), slow_list=fmt_list(slow))))
        ns = rng.choice([[50, 100, 200], [100, 200], [20, 50, 100, 200]])
        out.append(("above_sma", sym, {}, {"n": ns}, F.FAMILIES["above_sma"]["q"].format(sym=sym, n_list=fmt_list(ns))))
        n, sell = rng.choice([(2, 70), (2, 60), (3, 70)])
        buys = rng.choice([[5, 10, 15], [10, 20], [5, 10]])
        out.append(("rsi_revert", sym, {"n": n, "sell": sell}, {"buy": buys},
                    F.FAMILIES["rsi_revert"]["q"].format(sym=sym, n=n, buy_list=fmt_list(buys), sell=sell)))
        lbs = rng.choice([[20, 60, 120], [60, 120, 250], [20, 250]])
        out.append(("momentum", sym, {}, {"lookback": lbs}, F.FAMILIES["momentum"]["q"].format(sym=sym, lookback_list=fmt_list(lbs))))
        dn = rng.choice([[20, 55], [20, 50, 100], [55]])
        out.append(("donchian", sym, {}, {"n": dn}, F.FAMILIES["donchian"]["q"].format(sym=sym, n_list=fmt_list(dn))))
        days = rng.choice([[0, 4], [0, 1, 2, 3, 4], [0]])
        out.append(("weekday", sym, {}, {"day": days},
                    F.FAMILIES["weekday"]["q"].format(sym=sym, day_list=fmt_list([F.DAYS[d] for d in days]))))
        afters = rng.choice([[3], [1, 3], [2, 3, 4]])
        out.append(("turn_of_month", sym, {"before": 1}, {"after": afters},
                    F.FAMILIES["turn_of_month"]["q"].format(sym=sym, before=1, after_list=fmt_list(afters))))
        ks = rng.choice([[2.0], [1.5, 2.0], [1.5, 2.0, 2.5]])
        out.append(("bollinger", sym, {"n": 20}, {"k": ks},
                    F.FAMILIES["bollinger"]["q"].format(sym=sym, n=20, k_list=fmt_list(ks))))
        out.append(("buy_hold", sym, {}, {}, F.FAMILIES["buy_hold"]["q"].format(sym=sym)))
    return out


def reference(shim, fam, sym, fixed, grid):
    shim["PRICES"].clear()
    shim["LABS"].clear()
    shim["load_chart"](sym, common.chart_payload(sym))
    spec = F.FAMILIES[fam]
    L = shim["lab"](sym, vs=spec["vs"])
    names = list(grid)
    combos = list(itertools.product(*grid.values())) if grid else [()]
    sig = spec["signal"]
    for combo in combos:
        L.run(sig, **fixed, **dict(zip(names, combo)))
    res = L.check()
    return {"n_variants": len(combos), "verdict": res.verdict, "p_adjusted": round(res.p_value, 4),
            "best": res.best.name, "sharpe": round(res.best.sharpe, 3), "vs": spec["vs"]}


def main():
    rng = random.Random(20261118)
    shim = common.load_shim()
    items = specs(rng)
    rng.shuffle(items)
    items = items[:100]
    path = common.HERE / "questions.jsonl"
    counts = {}
    with path.open("w", encoding="utf-8") as fh:
        for i, (fam, sym, fixed, grid, q) in enumerate(items, 1):
            ref = reference(shim, fam, sym, fixed, grid)
            counts[ref["verdict"]] = counts.get(ref["verdict"], 0) + 1
            fh.write(json.dumps({"id": f"q{i:03d}", "family": fam, "symbol": sym, "question": q,
                                 "fixed": fixed, "grid": grid, "reference": ref}) + "\n")
            print(f"q{i:03d} {fam:14s} {sym:8s} n={ref['n_variants']} {ref['verdict']} p={ref['p_adjusted']}")
    print(counts, "->", path)


if __name__ == "__main__":
    main()
