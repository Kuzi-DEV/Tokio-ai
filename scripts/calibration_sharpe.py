"""How often do the Probabilistic and Deflated Sharpe Ratios fire on P&L with no edge?

The same null P&L shapes and grids as calibration_backtest.py. A strategy
"passes" when PSR/DSR >= 0.95, the threshold the papers use, so a
calibrated statistic passes 5% of the time.

  psr / dsr        as published: the Sharpe's variance assumes independent bars
  psr_dep / dsr_dep  the same with the variance scaled by the P&L's long-run
                   variance (dependence=True)
  rw               check_backtest's Romano-Wolf, for reference (grids only)

Run:  python scripts/calibration_sharpe.py [--paths N] [--workers N]
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

from calibration_backtest import grid, single_pnl  # noqa: E402
from tokio_ai.backtest import check_backtest  # noqa: E402
from tokio_ai.sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio  # noqa: E402

PASS = 0.95


def run_single(args):
    kind, n, paths, seed = args
    rng = np.random.default_rng(seed)
    hits = {"psr": 0, "psr_dep": 0}
    for _ in range(paths):
        x = single_pnl(kind, rng, n)
        hits["psr"] += probabilistic_sharpe_ratio(x).psr >= PASS
        hits["psr_dep"] += probabilistic_sharpe_ratio(x, dependence=True).psr >= PASS
    return ("single", kind, n, paths, hits)


def run_grid(args):
    kind, n, k, paths, planted, seed = args
    rng = np.random.default_rng(seed)
    hits = {"dsr": 0, "dsr_dep": 0, "rw": 0}
    for _ in range(paths):
        g = grid(kind, rng, n, k)
        target = None
        if planted:
            target = list(g)[len(g) // 2]
            x = g[target]
            g[target] = x + 1.0 / math.sqrt(252) * x.std()
        names = list(g)
        srs = [g[nm].mean() / g[nm].std(ddof=1) for nm in names]
        best = names[int(np.argmax(srs))]
        d = deflated_sharpe_ratio(g)
        dd = deflated_sharpe_ratio(g, dependence=True)
        res = check_backtest(g)
        if planted:
            # found = the planted variant is the one tested, and it passes
            hits["dsr"] += best == target and d.psr >= PASS
            hits["dsr_dep"] += best == target and dd.psr >= PASS
            hits["rw"] += {s.name: s for s in res.strategies}[target].p_adjusted <= 0.05
        else:
            hits["dsr"] += d.psr >= PASS
            hits["dsr_dep"] += dd.psr >= PASS
            hits["rw"] += res.best.p_adjusted <= 0.05
    return ("power" if planted else "selection", f"{kind} k={k}", n, paths, hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs_s = [(kind, n, a.paths, 1000 * i + n)
              for i, kind in enumerate(("pos1", "pos20", "pos120", "ar+0.2", "ar-0.2",
                                        "tranche5", "tranche20", "regime"))
              for n in (500, 2520)]
    jobs_g = [(kind, 2520, k, a.paths // 2, planted, 7919 * (k + planted) + len(kind))
              for kind in ("independent", "lookback") for k in (20, 100)
              for planted in (False, True)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_single, jobs_s)) + list(ex.map(run_grid, jobs_g))
    print(f"pass = PSR/DSR >= {PASS}; rates in %  ({time.time() - t0:.0f}s)\n")
    for part in ("single", "selection", "power"):
        rows = [r for r in results if r[0] == part]
        cols = list(rows[0][4])
        print(f"== {part} ==")
        print(f"{'config':<22} {'n':>5} {'paths':>5}  " + "  ".join(f"{c:>7}" for c in cols))
        for _, kind, n, paths, hits in rows:
            print(f"{kind:<22} {n:>5} {paths:>5}  " + "  ".join(f"{100 * hits[c] / paths:>7.1f}" for c in cols))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
