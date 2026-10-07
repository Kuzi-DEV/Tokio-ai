"""Does check_contracts hold its false-positive rate on books with stop-losses?

Fair markets, simulated the way 15-minute binary contracts behave: the
underlying is a random walk, and the contract's price at every moment is
its true probability of finishing in the money, so the price is a
martingale and no strategy has an edge. A favourite is bought 5 minutes
before close at a price drawn from 85-98c (plus a 0.3c fee), with a stop a
fixed distance below entry. The exits go to check_contracts as
`exit_values`; the stop level itself isn't needed. The stop is checked once a minute (as a replay on
1-minute candles would) or every second (nearly continuous); a check that
finds the price below the stop sells at that price, so a gap costs.

Run:  python scripts/calibration_stops.py [--books N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from tokio_ai.contracts import check_contracts  # noqa: E402

ALPHA = 0.05
FEE = 0.003
N = NormalDist()


def book(rng, n, stop_dist, steps_per_min, edge=0.0):
    p0 = rng.uniform(0.85, 0.98, n)
    minutes, steps = 5, 5 * steps_per_min
    dt = minutes / steps
    x = np.array([N.inv_cdf(v) for v in p0]) * np.sqrt(minutes)  # distance to strike, vol 1/min
    L = np.maximum(p0 - stop_dist, 0.01) if stop_dist else np.zeros(n)
    exit_v = np.full(n, np.nan)
    alive = np.ones(n, bool)
    for k in range(1, steps + 1):
        x = x + rng.standard_normal(n) * np.sqrt(dt)
        tau = minutes - k * dt
        if tau <= 1e-12:
            break
        if stop_dist and k % steps_per_min == 0 or (stop_dist and steps_per_min == 1):
            price = np.array([N.cdf(v) for v in x / np.sqrt(tau)])
            hit = alive & (price <= L)
            exit_v[hit] = np.maximum(price[hit] - FEE, 0.0)
            alive &= ~hit
    won = (x > 0).astype(float)
    # Pay p0 - edge for a contract truly worth p0. With edge 0 the entry fee
    # makes the book slightly worse than fair.
    return check_contracts(p0 - edge, won, fees=FEE,
                           exit_values=exit_v if stop_dist else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", type=int, default=600)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    t0 = time.time()
    print(f"alpha={ALPHA}; false-positive rate (p_value <= alpha) on fair markets, {a.books} books per row\n")
    print(f"{'stop':>10} {'checks':>9} {'bets':>5}  {'FPR':>6}  {'mean stopped':>12}")
    for stop in (0.0, 0.15, 0.30):
        for spm in ((1,) if stop == 0 else (1, 60)):
            for n in (100, 500):
                hits, stopped = 0, 0
                for _ in range(a.books):
                    r = book(rng, n, stop, spm)
                    hits += r.p_value <= ALPHA
                    stopped += r.n_exited
                label = "none" if not stop else f"-{int(stop * 100)}c"
                chk = "-" if not stop else ("1/min" if spm == 1 else "1/sec")
                print(f"{label:>10} {chk:>9} {n:>5}  {100 * hits / a.books:>5.1f}%  "
                      f"{stopped / a.books:>12.1f}")
    print("\npower: the favourite is bought 2c below what it is truly worth, 500 bets")
    for stop in (0.0, 0.15, 0.30):
        hits = sum(book(rng, 500, stop, 1, edge=0.02).p_value <= ALPHA for _ in range(a.books))
        label = "none" if not stop else f"-{int(stop * 100)}c"
        print(f"{label:>10}  found {100 * hits / a.books:.1f}%")
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
