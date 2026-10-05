"""Does `check_many` control the family-wise error rate on a realistic grid?

A grid search is a family of highly correlated tests: "20-day momentum" and
"40-day momentum" agree on most days, a 1% drop threshold contains every 2%
drop. The grid here is 30 variants -- momentum lookbacks 5/10/20/40/60 and
drop thresholds 1/1.5/2/2.5/3%, each at horizons 1, 5 and 20 bars.

Two measurements per method:

  family-wise error  On markets with NO edge, how often is ANY variant in
                     the grid flagged? Should be ~5%. This is the number
                     that matters when you report "the best of my grid".
  power              With one real edge planted in one variant, how often
                     is that variant flagged?

Methods: no correction (flag anything with p <= 0.05 alone); Bonferroni and
Holm (valid, assume nothing about dependence); Benjamini-Hochberg (the
TokIO ledger's correction; controls FDR, which equals FWER when nothing is
real); and check_many. All of them start from the same per-variant p-values
(check()'s), so the comparison isolates the correction.

The autocorrelated-return generators are left out on purpose: conditions
computed from past returns genuinely predict AR returns, so that would not
be a null.

Run:  python scripts/calibration_family.py [--trials N] [--workers N]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_check import generate  # noqa: E402

from tokio_ai.family import check_many  # noqa: E402
from tokio_ai.rigor.stats import benjamini_hochberg  # noqa: E402

ALPHA = 0.05
LOOKBACKS = (5, 10, 20, 40, 60)
THRESHOLDS = (0.01, 0.015, 0.02, 0.025, 0.03)
HORIZONS = (1, 5, 20)
METHODS = ("uncorrected", "bonferroni", "holm", "bh", "tokio")
# (condition, horizon) that carries the planted edge in the power runs
PLANTS = (("mom20", 5), ("drop0.02", 1), ("mom60", 20))


def grid(r):
    n = len(r)
    growth = [1.0]
    for x in r:
        growth.append(growth[-1] * (1 + x))
    conds = {}
    for L in LOOKBACKS:
        conds[f"mom{L}"] = [None if i + 1 < L else growth[i + 1] / growth[i + 1 - L] > 1 for i in range(n)]
    for x in THRESHOLDS:
        conds[f"drop{x}"] = [v < -x for v in r]
    return conds


def flagged(res):
    """Per method: the set of member names flagged."""
    members = [m for m in res.members if m.reportable]
    p = [m.p_raw for m in members]
    k = len(p)
    out = {"uncorrected": {m.name for m in members if m.p_raw <= ALPHA},
           "bonferroni": {m.name for m in members if m.p_raw <= ALPHA / k}}
    holm, order = set(), sorted(range(k), key=lambda i: p[i])
    for rank, i in enumerate(order):
        if p[i] > ALPHA / (k - rank):
            break
        holm.add(members[i].name)
    out["holm"] = holm
    out["bh"] = {m.name for m, s in zip(members, benjamini_hochberg(p, ALPHA)) if s}
    return out


def run(job):
    gen, offset, trials, effect = job
    fwer = dict.fromkeys(METHODS, 0)
    power = {(plant, m): 0 for plant in PLANTS for m in METHODS}
    used = 0
    for t in range(trials):
        r = generate(gen, 1000, seed=30_000 + offset + t)
        conds = grid(r)
        used += 1
        res = check_many(r, conds, horizon=list(HORIZONS), seed=t)
        f = flagged(res)
        f["tokio"] = {m.name for m in res.significant}
        for m in METHODS:
            fwer[m] += bool(f[m])
        for (cname, h) in PLANTS:
            lab = conds[cname]
            nt = sum(1 for c in lab if c is True)
            nf = sum(1 for c in lab if c is False)
            sigma = (sum(v * v for v in r) / len(r)) ** 0.5
            drift = effect * sigma * math.sqrt(h) * math.sqrt(h * (1 / nt + 1 / nf)) / h
            pl = list(r)
            for i, c in enumerate(lab):
                if c:
                    for j in range(i + 1, min(i + 1 + h, len(r))):
                        pl[j] += drift
            # labels stay those of the unplanted path, as in benchmark.py
            res_p = check_many(pl, conds, horizon=list(HORIZONS), seed=t)
            fp = flagged(res_p)
            fp["tokio"] = {m.name for m in res_p.significant}
            target = f"{cname} @h{h}"
            for m in METHODS:
                power[((cname, h), m)] += target in fp[m]
    return gen, used, fwer, power


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--effect", type=float, default=3.0,
                        help="planted edge in standard errors (see benchmark.py)")
    args = parser.parse_args()
    gens = ("garch", "garch_t4", "regime")
    started = time.time()
    per = max(1, args.trials // args.workers)
    jobs = [(g, w * per, per, args.effect) for g in gens for w in range(args.workers)]
    agg = {g: [0, dict.fromkeys(METHODS, 0), {}] for g in gens}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for g, used, fwer, power in pool.map(run, jobs):
            a = agg[g]
            a[0] += used
            for m in METHODS:
                a[1][m] += fwer[m]
            for k, v in power.items():
                a[2][k] = a[2].get(k, 0) + v

    print(f"check_many calibration -- 30-variant grid, {args.trials} paths per generator.\n")
    print("Family-wise error (any variant flagged, no edge anywhere):\n")
    print("| generator | " + " | ".join(METHODS) + " |")
    print("|---|" + "---:|" * len(METHODS))
    for g in gens:
        used, fwer, _ = agg[g]
        print(f"| {g} | " + " | ".join(f"{fwer[m] / used:.1%}" for m in METHODS) + " |")
    print(f"\nPower (planted edge of {args.effect} SE, that variant flagged):\n")
    print("| generator | planted in | " + " | ".join(METHODS) + " |")
    print("|---|---|" + "---:|" * len(METHODS))
    for g in gens:
        used, _, power = agg[g]
        for plant in PLANTS:
            print(f"| {g} | {plant[0]} @h{plant[1]} | "
                  + " | ".join(f"{power[(plant, m)] / used:.1%}" for m in METHODS) + " |")
    print(f"\nelapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
