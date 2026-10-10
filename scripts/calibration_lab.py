"""Does a Lab sweep keep its false-positive rate on markets with no edge?

Each path is a GARCH-t4 random walk with zero drift and realistic opens
(an overnight gap plus a session). A researcher sweeps a grid of strategies
(SMA crossovers, long/short, and momentum lookbacks) through a Lab, picks
the best, and asks whether it's significant. Counted:

  naive   the best variant's own p-value (what you'd report without a lab)
  lab     lab.check(): corrected for every variant run (Romano-Wolf)

Run:  python scripts/calibration_lab.py [--paths N] [--workers N]
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
import pandas as pd  # noqa: E402

from tokio_ai import Lab  # noqa: E402

ALPHA = 0.05


def market(rng, n):
    omega, a, b = 2.0e-6, 0.09, 0.90
    var = omega / (1 - a - b)
    z = rng.standard_t(4, n + 300) / math.sqrt(2.0)
    r = np.empty(n + 300)
    for t in range(n + 300):
        r[t] = math.sqrt(var) * z[t]
        var = omega + a * r[t] ** 2 + b * var
    r = r[300:]
    gap_share = 0.3  # part of each day's move that happens overnight
    close = 100 * np.cumprod(1 + r)
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * (1 + gap_share * r)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"open": open_, "close": close}, index=idx)


def sma_ls(d, fast=10, slow=50):
    f, s = d.close.rolling(fast).mean(), d.close.rolling(slow).mean()
    return np.sign(f - s).fillna(0.0)


def momentum(d, k=20):
    return np.sign(d.close / d.close.shift(k) - 1).fillna(0.0)


def one(seed_n):
    seed, n = seed_n
    rng = np.random.default_rng(seed)
    lab = Lab(market(rng, n), holdout=0.25)
    lab.sweep(sma_ls, fast=[5, 10, 20], slow=[50, 100, 200])
    lab.sweep(momentum, k=[5, 10, 20, 60, 120])
    res = lab.check()
    naive = min(s.p_alone for s in res.strategies)
    return naive <= ALPHA, res.significant, len(lab.runs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bars", type=int, default=2520)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(one, [(1000 + i, a.bars) for i in range(a.paths)]))
    naive = sum(o[0] for o in out) / len(out)
    lab = sum(o[1] for o in out) / len(out)
    print(f"{a.paths} no-edge markets x {out[0][2]} variants, {a.bars} bars ({time.time() - t0:.0f}s)")
    print(f"best variant's own p <= {ALPHA}: {naive:.1%}")
    print(f"lab.check() significant:      {lab:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
