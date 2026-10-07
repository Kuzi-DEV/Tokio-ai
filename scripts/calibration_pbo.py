"""How does the probability of backtest overfitting behave with and without a real edge?

PBO has no alpha, so there is no false-positive rate to hold. What matters
is whether its reading can be trusted in both directions:

  no edge    every variant is noise. PBO should sit near 0.5. The number to
             watch is how often it comes out LOW (< 0.2), which would tell
             a user "the in-sample winner persists" about pure noise.
  real edge  one variant carries a true Sharpe of 0.5 / 1.0 in a grid of
             noise. PBO should be low, and the planted variant should be
             the one picked most often.

Grids as in calibration_backtest.py: 50 independent random-position
strategies, or one trend rule on an unrelated random walk at 50 lookbacks
(neighbours nearly identical), on GARCH-t4 returns. 2,520 bars.

Run:  python scripts/calibration_pbo.py [--paths N] [--workers N]
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

import numpy as np  # noqa: E402

from calibration_backtest import grid  # noqa: E402
from tokio_ai.overfit import probability_of_overfitting  # noqa: E402


def run(args):
    kind, sharpe, paths, seed = args
    rng = np.random.default_rng(seed)
    pbos, picked = [], 0
    for _ in range(paths):
        g = grid(kind, rng, 2520, 50)
        target = list(g)[len(g) // 2]
        if sharpe:
            x = g[target]
            g[target] = x + sharpe / math.sqrt(252) * x.std()
        res = probability_of_overfitting(g)
        pbos.append(res.pbo)
        picked += res.most_selected[0][0] == target
    return kind, sharpe, paths, np.array(pbos), picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(kind, sr, a.paths, 104729 * i + int(sr * 10))
            for i, kind in enumerate(("independent", "lookback")) for sr in (0.0, 0.5, 1.0)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(run, jobs))
    print(f"50-variant grids, 2520 bars ({time.time() - t0:.0f}s)\n")
    print(f"{'grid':<12} {'edge SR':>7} {'paths':>5}  {'mean PBO':>8}  {'median':>6}  "
          f"{'PBO<0.2':>7}  {'PBO>=0.4':>8}  {'planted picked most':>19}")
    for kind, sr, paths, p, picked in out:
        pk = f"{100 * picked / paths:.0f}%" if sr else "-"
        print(f"{kind:<12} {sr:>7.1f} {paths:>5}  {p.mean():>8.2f}  {np.median(p):>6.2f}  "
              f"{100 * np.mean(p < 0.2):>6.1f}%  {100 * np.mean(p >= 0.4):>7.1f}%  {pk:>19}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
