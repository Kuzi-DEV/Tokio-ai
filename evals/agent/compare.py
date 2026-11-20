"""Paired comparison of two eval runs, with a confidence interval that resamples families.

    python evals/agent/compare.py ft2-qwen3-1.7b ft3-qwen3-1.7b

A model writes the same code across a family's questions, so questions within a
family are not independent. The interval resamples whole families (paired: the
same families for both runs), which is the honest unit.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

RES = Path(__file__).parent / "results"


def load(tag):
    return {r["id"]: r for r in map(json.loads, (RES / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines())}


def main(a, b, key="all_ok", draws=20000):
    A, B = load(a), load(b)
    assert A.keys() == B.keys(), "runs cover different questions"
    fams = collections.defaultdict(list)
    for k in A:
        fams[A[k]["family"]].append((int(A[k][key]), int(B[k][key])))
    print(f"{'family':22s} {a[:14]:>14s} {b[:14]:>14s}")
    for f, v in sorted(fams.items()):
        print(f"{f:22s} {sum(x for x, _ in v):10d}/{len(v):<3d} {sum(y for _, y in v):10d}/{len(v):<3d}")
    groups = list(fams.values())
    n = sum(map(len, groups))
    diff = sum(y - x for g in groups for x, y in g) / n * 100
    rng = random.Random(0)
    ds = []
    for _ in range(draws):
        s = [rng.choice(groups) for _ in groups]
        ds.append(sum(y - x for g in s for x, y in g) / sum(map(len, s)) * 100)
    ds.sort()
    lo, hi = ds[int(0.025 * draws)], ds[int(0.975 * draws)]
    print(f"\n{key}: {a} {sum(int(r[key]) for r in A.values())}/{n}, {b} {sum(int(r[key]) for r in B.values())}/{n}")
    print(f"difference {diff:+.1f} pts per 100, 95% CI over {len(groups)} families [{lo:+.1f}, {hi:+.1f}], "
          f"share of resamples <= 0: {sum(d <= 0 for d in ds) / draws:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3], **({"key": sys.argv[3]} if len(sys.argv) > 3 else {}))
