"""`check_backtest` on REAL market history with placebo strategies.

Twenty years of daily bars for ten assets (the same set as placebo_real.py).
Each placebo strategy takes random long/short positions generated
independently of the prices, so its true expected P&L is zero -- even on an
asset that rose, because long and short are equally likely. Anything called
profitable is a false positive, measured against the real return path:
fat tails, volatility clustering, 2008, 2020, USO's months-long trends.

  single     one placebo strategy. Positions flip on average every 1, 20
             or 120 bars; "trades20" lists each day's 20-bar trade return,
             so consecutive entries overlap (a common way P&L gets logged).
  +pos       the same test with the placebo's positions passed in
             (`positions=`), which widens the lag window to the holding run.
  selection  50 placebo strategies on the same asset, best one reported.

Run:  python scripts/placebo_backtest_real.py [--placebos N] [--workers N]
Needs network (Yahoo daily bars).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from calibration_backtest import positions, t_p  # noqa: E402
from tokio_ai.backtest import check_backtest  # noqa: E402
from tokio_ai.tools.prices import fetch_daily_bars  # noqa: E402

ALPHA = 0.05
ASSETS = ("SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "AAPL", "JPM", "XOM", "EURUSD=X")
KINDS = ("pos1", "pos20", "pos120", "trades20")
GRID_K = 50


def placebo(kind, r, rng):
    """(P&L, positions or None)."""
    n = len(r)
    if kind.startswith("pos"):
        p = positions(rng, n, int(kind[3:]))
        return p * r, p
    h = 20
    c = np.concatenate([[0.0], np.cumsum(np.log1p(r))])
    fwd = np.expm1(c[h + 1:] - c[1:-h])  # trade entered at close of t, held h bars
    return positions(rng, len(fwd), 20) * fwd, None


def run_asset(job):
    symbol, r, placebos = job
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))  # hash() is salted per process
    out = {}
    for kind in KINDS:
        t = h = hp = 0
        for _ in range(placebos):
            x, p = placebo(kind, r, rng)
            t += t_p(x) <= ALPHA
            h += check_backtest(x).best.p_alone <= ALPHA
            hp += check_backtest(x, positions=p).best.p_alone <= ALPHA
        out[kind] = (t, h, hp, placebos)
    g = {"t": 0, "hac": 0, "rw": 0}
    grids = max(placebos // 5, 1)
    for _ in range(grids):
        fam = {f"s{j}": positions(rng, len(r), 20) * r for j in range(GRID_K)}
        res = check_backtest(fam)
        g["t"] += min(s.p_naive for s in res.strategies) <= ALPHA
        g["hac"] += min(s.p_alone for s in res.strategies) <= ALPHA
        g["rw"] += min(s.p_adjusted for s in res.strategies) <= ALPHA
    return symbol, len(r), out, g, grids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebos", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--range", default="20y")
    a = ap.parse_args()
    t0 = time.time()
    jobs = []
    for sym in ASSETS:
        closes = np.array([b.adj_close for b in fetch_daily_bars(sym, a.range)], dtype=float)
        r = closes[1:] / closes[:-1] - 1
        r = r[np.isfinite(r)]
        jobs.append((sym, r, a.placebos))
    tot = {k: [0, 0, 0, 0] for k in KINDS}
    gt = {"t": 0, "hac": 0, "rw": 0, "n": 0}
    print(f"alpha={ALPHA}; false-positive rates in %\n")
    print(f"{'asset':<9} {'bars':>5}  " + "  ".join(f"{k + ' t/hac':>16}" for k in KINDS)
          + f"  {'best of 50 t/hac/rw':>20}")
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, n, out, g, grids in ex.map(run_asset, jobs):
            cells = []
            for k in KINDS:
                t, h, hp, p = out[k]
                for i, v in enumerate((t, h, hp, p)):
                    tot[k][i] += v
                cells.append(f"{100 * t / p:.1f}/{100 * h / p:.1f}/{100 * hp / p:.1f}")
            for key in ("t", "hac", "rw"):
                gt[key] += g[key]
            gt["n"] += grids
            print(f"{sym:<9} {n:>5}  " + "  ".join(f"{c:>22}" for c in cells)
                  + f"  {100 * g['t'] / grids:>6.1f}/{100 * g['hac'] / grids:.1f}/{100 * g['rw'] / grids:.1f}")
    cells = [f"{100 * tot[k][0] / tot[k][3]:.1f}/{100 * tot[k][1] / tot[k][3]:.1f}/"
             f"{100 * tot[k][2] / tot[k][3]:.1f}" for k in KINDS]
    print(f"{'POOLED':<9} {'':>5}  " + "  ".join(f"{c:>22}" for c in cells)
          + f"  {100 * gt['t'] / gt['n']:>6.1f}/{100 * gt['hac'] / gt['n']:.1f}/{100 * gt['rw'] / gt['n']:.1f}")
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
