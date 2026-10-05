"""Does `check_backtest` hold its false-positive rate on strategy P&L with no edge?

Three measurements, each on simulated P&L whose true mean is exactly zero:

  single     One strategy, no search. How often does each test call it
             profitable at alpha=0.05? P&L shapes: random positions (held
             1 / 20 / 120 bars) on GARCH-t4 returns, P&L with its own
             autocorrelation (+/-0.2), overlapping tranches (the average of
             h staggered holdings, which makes the P&L a moving average),
             and a volatility-regime switch.
  selection  k strategies run on the same market; the best one is reported.
             How often is it called significant? Independent random
             positions, and a correlated "lookback grid" (one idea, many
             parameters). This is the number that matters.
  power      The same grids with a real Sharpe of 1.0 planted in one
             variant: how often is that variant found?

Tests: a plain t-test on the mean; the HAC test alone (check_backtest with
trials=1); Sidak on the HAC p-value; check_backtest with every variant
(Romano-Wolf).

Run:  python scripts/calibration_backtest.py [--paths N] [--workers N]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from tokio_ai.backtest import check_backtest  # noqa: E402

ALPHA = 0.05


def garch_t4(rng, n, burn=500):
    omega, a, b = 2.0e-6, 0.09, 0.90
    var = omega / (1 - a - b)
    z = rng.standard_t(4, n + burn) / math.sqrt(2.0)
    out = np.empty(n + burn)
    for t in range(n + burn):
        r = math.sqrt(var) * z[t]
        var = omega + a * r * r + b * var
        out[t] = r
    return out[burn:]


def positions(rng, n, run):
    if run == 1:
        return np.where(rng.random(n) < 0.5, -1.0, 1.0)
    flips = rng.random(n) < 1 / run
    start = 1.0 if rng.random() < 0.5 else -1.0
    return start * np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)


def single_pnl(kind, rng, n):
    r = garch_t4(rng, n)
    if kind.startswith("pos"):
        return positions(rng, n, int(kind[3:])) * r
    if kind in ("ar+0.2", "ar-0.2"):
        phi = float(kind[2:])
        out, prev = np.empty(n), 0.0
        for t in range(n):
            prev = phi * prev + r[t]
            out[t] = prev
        return out
    if kind.startswith("tranche"):
        h = int(kind[7:])
        c = np.concatenate([[0.0], np.cumsum(r)])
        idx = np.arange(n)
        return (c[idx + 1] - c[np.maximum(idx + 1 - h, 0)]) / h
    if kind == "regime":
        calm = np.cumsum(rng.random(n) < 0.01) % 2 == 0
        return rng.standard_normal(n) * np.where(calm, 0.008, 0.03)
    raise ValueError(kind)


def t_p(x):
    m, s = x.mean(), x.std(ddof=1)
    return 0.5 * math.erfc(m / (s / math.sqrt(len(x))) / math.sqrt(2)) if s > 0 else 1.0


def grid(kind, rng, n, k):
    r = garch_t4(rng, n)
    if kind == "independent":
        return {f"s{j}": positions(rng, n, 20) * r for j in range(k)}
    # One idea, k parameters: trend on an unrelated random walk at lookbacks
    # 5..200. Neighbouring lookbacks hold the same position most days.
    w = np.cumsum(rng.standard_normal(n))
    out = {}
    for L in np.unique(np.geomspace(5, 200, k).astype(int)):
        sig = np.ones(n)
        sig[L:] = np.sign(w[L:] - w[:-L])
        sig[:L] = 1.0
        pos = np.concatenate([[1.0], sig[:-1]])  # trade on yesterday's signal
        out[f"lb{L}"] = pos * r
    return out


def run_single(args):
    kind, n, paths, seed = args
    rng = np.random.default_rng(seed)
    hits = {"t": 0, "hac": 0}
    for _ in range(paths):
        x = single_pnl(kind, rng, n)
        hits["t"] += t_p(x) <= ALPHA
        hits["hac"] += check_backtest(x).best.p_alone <= ALPHA
    return ("single", kind, n, paths, hits)


def run_grid(args):
    kind, n, k, paths, planted, seed = args
    rng = np.random.default_rng(seed)
    hits = {"t": 0, "hac": 0, "sidak": 0, "rw": 0}
    for _ in range(paths):
        g = grid(kind, rng, n, k)
        target = None
        if planted:
            target = list(g)[len(g) // 2]
            x = g[target]
            g[target] = x + 1.0 / math.sqrt(252) * x.std()
        res = check_backtest(g)
        by = {s.name: s for s in res.strategies}
        kk = len(g)
        if planted:
            s = by[target]
            hits["t"] += s.p_naive <= ALPHA
            hits["hac"] += s.p_alone <= ALPHA
            hits["sidak"] += -math.expm1(kk * math.log1p(-s.p_alone)) <= ALPHA
            hits["rw"] += s.p_adjusted <= ALPHA
        else:
            hits["t"] += min(s.p_naive for s in res.strategies) <= ALPHA
            hits["hac"] += min(s.p_alone for s in res.strategies) <= ALPHA
            hits["sidak"] += -math.expm1(kk * math.log1p(-min(s.p_alone for s in res.strategies))) <= ALPHA
            hits["rw"] += min(s.p_adjusted for s in res.strategies) <= ALPHA
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
    print(f"alpha={ALPHA}; rates in %  ({time.time() - t0:.0f}s)\n")
    for part in ("single", "selection", "power"):
        rows = [r for r in results if r[0] == part]
        cols = list(rows[0][4])
        print(f"== {part} ==")
        print(f"{'config':<22} {'n':>5} {'paths':>5}  " + "  ".join(f"{c:>6}" for c in cols))
        for _, kind, n, paths, hits in rows:
            print(f"{kind:<22} {n:>5} {paths:>5}  " + "  ".join(f"{100 * hits[c] / paths:>6.1f}" for c in cols))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
