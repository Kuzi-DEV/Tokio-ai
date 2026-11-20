"""Build questions_b.jsonl: test set B, 25 families x 4 tickers = 100 questions.

    python evals/agent/make_questions_b.py

Same reference procedure as set A (make_questions.reference): each grid runs
through a fresh Lab on prices frozen at common.END_DATE. Each family gets 4
different test tickers, so a family is at most 4 questions and the score rests
on 25 independent families instead of 9.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import families as FA  # noqa: E402
import families_b as F  # noqa: E402
from make_questions import SYMBOLS, fmt_list  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "agent"))
import families_v2 as TRAIN  # noqa: E402

assert not set(F.FAMILIES) & set(FA.FAMILIES)
assert not set(F.FAMILIES) & set(TRAIN.F)

# family -> (fixed params, [grid options]); a grid key named in the question as {key_list}
GRIDS = {
    "dual_sma_filter": ({"a": 50}, [{"b": [100, 200]}, {"b": [150, 200]}, {"b": [200]}]),
    "triple_ma": ({"a": 10, "b": 50}, [{"c": [100, 200]}, {"c": [200]}, {"c": [100, 150, 200]}]),
    "regression_slope": ({}, [{"n": [20, 60]}, {"n": [50, 100, 200]}, {"n": [100]}]),
    "trix": ({}, [{"n": [15]}, {"n": [10, 20]}, {"n": [15, 30, 45]}]),
    "adx_trend": ({"n": 14}, [{"thr": [20, 25]}, {"thr": [25]}, {"thr": [15, 20, 30]}]),
    "channel_mid": ({}, [{"n": [20, 50]}, {"n": [55]}, {"n": [20, 55, 100]}]),
    "skip_month_momentum": ({}, [{"n": [126, 252]}, {"n": [252]}, {"n": [63, 126, 252]}]),
    "near_high": ({}, [{"pct": [0.05, 0.1]}, {"pct": [0.1]}, {"pct": [0.03, 0.05, 0.1, 0.2]}]),
    "new_low_avoid": ({"hold": 10}, [{"n": [50, 100]}, {"n": [100]}, {"n": [20, 50, 252]}]),
    "half_trend": ({}, [{"n": [50, 200]}, {"n": [200]}, {"n": [100, 150, 200]}]),
    "trend_vs_cash": ({}, [{"n": [200]}, {"n": [50, 200]}, {"n": [100]}]),
    "volume_spike_up": ({}, [{"k": [1.5, 2.0]}, {"k": [2.0]}, {"k": [1.5, 2.0, 3.0]}]),
    "volume_dryup": ({}, [{"k": [0.8]}, {"k": [0.7, 0.9]}, {"k": [0.6, 0.8, 1.0]}]),
    "vwap_trend": ({}, [{"n": [20, 50]}, {"n": [50]}, {"n": [10, 50, 200]}]),
    "obv_trend": ({}, [{"n": [20, 50]}, {"n": [50]}, {"n": [20, 100, 200]}]),
    "cmf_positive": ({}, [{"n": [20]}, {"n": [10, 20]}, {"n": [20, 50]}]),
    "inside_day": ({}, [{}]),
    "outside_down_day": ({}, [{}]),
    "range_expansion": ({}, [{"k": [1.5, 2.0]}, {"k": [2.0]}, {"k": [1.5, 2.0, 2.5]}]),
    "big_drop": ({}, [{"x": [0.02, 0.03]}, {"x": [0.03]}, {"x": [0.01, 0.02, 0.04]}]),
    "big_up": ({}, [{"x": [0.02, 0.03]}, {"x": [0.02]}, {"x": [0.01, 0.02, 0.04]}]),
    "higher_lows": ({}, [{"n": [3]}, {"n": [2, 3, 4]}, {"n": [3, 5]}]),
    "bullish_candle": ({}, [{}]),
    "above_prior_high": ({}, [{}]),
    "year_start": ({}, [{"n": [5]}, {"n": [5, 10, 20]}, {"n": [10]}]),
}
assert set(GRIDS) == set(F.FAMILIES)


def question(fam, sym, fixed, grid):
    kw = dict(sym=sym, **{k: v for k, v in fixed.items()})
    kw.update({f"{k}_list": fmt_list(v) for k, v in grid.items()})
    return F.FAMILIES[fam]["q"].format(**kw)


def reference(shim, fam, sym, fixed, grid):
    shim["PRICES"].clear()
    shim["LABS"].clear()
    shim["load_chart"](sym, common.chart_payload(sym))
    spec = F.FAMILIES[fam]
    L = shim["lab"](sym, vs=spec["vs"])
    names = list(grid)
    combos = list(itertools.product(*grid.values())) if grid else [()]
    for combo in combos:
        L.run(spec["signal"], **fixed, **dict(zip(names, combo)))
    res = L.check()
    return {"n_variants": len(combos), "verdict": res.verdict, "p_adjusted": round(res.p_value, 4),
            "best": res.best.name, "sharpe": round(res.best.sharpe, 3), "vs": spec["vs"]}


def main():
    rng = random.Random(20261120)
    shim = common.load_shim()
    items = []
    for fam in sorted(F.FAMILIES):
        fixed, options = GRIDS[fam]
        for sym in rng.sample(SYMBOLS, 4):
            grid = rng.choice(options)
            items.append((fam, sym, fixed, grid, question(fam, sym, fixed, grid)))
    rng.shuffle(items)
    path = common.HERE / "questions_b.jsonl"
    counts = {}
    with path.open("w", encoding="utf-8") as fh:
        for i, (fam, sym, fixed, grid, q) in enumerate(items, 1):
            ref = reference(shim, fam, sym, fixed, grid)
            counts[ref["verdict"]] = counts.get(ref["verdict"], 0) + 1
            fh.write(json.dumps({"id": f"b{i:03d}", "family": fam, "symbol": sym, "question": q,
                                 "fixed": fixed, "grid": grid, "reference": ref}) + "\n")
            print(f"b{i:03d} {fam:20s} {sym:8s} n={ref['n_variants']} {ref['verdict']} p={ref['p_adjusted']}", flush=True)
    print(counts, "->", path)


if __name__ == "__main__":
    main()
