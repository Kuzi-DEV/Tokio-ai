"""Family-wise error of check_many on REAL market history.

The real-data counterpart of calibration_family.py. Each placebo is a whole
grid of 15 signals built from one hidden random process that is generated
independently of the prices -- so every signal in the grid predicts
nothing, while the signals are strongly correlated with each other the way
a real parameter grid is (the process smoothed over 5/20/60 bars, cut at
several thresholds). Each grid is tested at horizons 1, 5 and 20, which
makes 45 variants per family.

The question is the one a researcher faces after a grid search: in a market
where nothing works, how often does SOMETHING in the grid come out
significant?

Run:  python scripts/placebo_family_real.py [--placebos N] [--workers N]
Needs network (Yahoo daily bars) and numpy.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_family import METHODS, flagged  # noqa: E402
from placebo_real import ASSETS  # noqa: E402

from tokio_ai.family import check_many  # noqa: E402
from tokio_ai.tools.prices import fetch_daily_bars  # noqa: E402

HORIZONS = (1, 5, 20)
WINDOWS = (5, 20, 60)
QUANTILES = (0.2, 0.35, 0.5, 0.65, 0.8)


def placebo_grid(n: int, rng: random.Random) -> dict[str, list[bool | None]]:
    """15 correlated signals from one hidden AR(1) process, independent of prices."""
    x, level = [], 0.0
    for _ in range(n):
        level = 0.97 * level + rng.gauss(0, 1)
        x.append(level)
    grid = {}
    for w in WINDOWS:
        sm, acc = [], 0.0
        for i, v in enumerate(x):
            acc += v - (x[i - w] if i >= w else 0.0)
            sm.append(acc / w if i + 1 >= w else None)
        known = sorted(v for v in sm if v is not None)
        for q in QUANTILES:
            cut = known[int(q * (len(known) - 1))]
            grid[f"w{w}_q{q}"] = [None if v is None else v > cut for v in sm]
    return grid


def run_asset(job):
    symbol, returns, placebos = job
    rng = random.Random(f"family-placebo:{symbol}")
    hits = dict.fromkeys(METHODS, 0)
    for k in range(placebos):
        res = check_many(returns, placebo_grid(len(returns), rng), horizon=list(HORIZONS), seed=k)
        f = flagged(res)
        f["tokio"] = {m.name for m in res.significant}
        for m in METHODS:
            hits[m] += bool(f[m])
    return symbol, placebos, hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placebos", type=int, default=200, help="grids per asset")
    parser.add_argument("--range", default="20y")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    jobs = []
    for symbol in ASSETS:
        closes = [b.adj_close for b in fetch_daily_bars(symbol, args.range)]
        returns = [None] + [(c / p - 1) if p else None for p, c in zip(closes, closes[1:])]
        jobs.append((symbol, returns, args.placebos))

    print(f"Real-data family placebo -- {args.placebos} random 45-variant grids per asset, "
          f"{args.range} of daily bars. Family-wise error = share of grids with ANY variant "
          f"flagged; a calibrated correction gives ~5%.\n")
    print("| asset | " + " | ".join(METHODS) + " |")
    print("|---|" + "---:|" * len(METHODS))
    total = dict.fromkeys(METHODS, 0)
    count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for symbol, n, hits in pool.map(run_asset, jobs):
            count += n
            for m in METHODS:
                total[m] += hits[m]
            print(f"| {symbol} | " + " | ".join(f"{hits[m] / n:.1%}" for m in METHODS) + " |",
                  flush=True)
    print(f"| **pooled** | " + " | ".join(f"**{total[m] / count:.1%}**" for m in METHODS) + " |")
    print(f"\nelapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
