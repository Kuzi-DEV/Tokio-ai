"""False-positive rate on binary-contract bets with no edge.

Every contract is fairly priced: it wins with probability equal to its
breakeven price (price + fee). P&L per contract is outcome - breakeven, so
the true expected P&L is exactly zero; anything called profitable is a
false positive. Rows: favourites at fixed prices (the shape where
variance-estimating tests break), a mixed book of 85-99c prices, and
mixed sizes (1-50 contracts).

Compared: a plain one-sided t-test on per-bet P&L, check_backtest's HAC
(same P&L series), and check_contracts.

Run:  python scripts/calibration_contracts.py [--paths N]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from tokio_ai.backtest import check_backtest  # noqa: E402
from tokio_ai.contracts import check_contracts  # noqa: E402

ALPHA = 0.05


def t_p(x):
    s = x.std(ddof=1)
    if s == 0:
        return 0.0 if x.mean() > 0 else 1.0
    return 0.5 * math.erfc(x.mean() / (s / math.sqrt(len(x))) / math.sqrt(2))


def book(kind, n, rng):
    if kind.startswith("fixed"):
        q = np.full(n, float(kind[5:]))
        w = np.ones(n)
    elif kind == "mixed":
        q = rng.uniform(0.85, 0.99, n)
        w = np.ones(n)
    else:  # mixed prices and sizes
        q = rng.uniform(0.85, 0.99, n)
        w = rng.integers(1, 51, n).astype(float)
    return q, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    t0 = time.time()
    print(f"alpha={ALPHA}; false-positive rates in %, {a.paths} books per row\n")
    print(f"{'book':<12} {'n':>5}  {'t-test':>7}  {'HAC':>6}  {'contracts':>9}")
    for kind in ("fixed0.80", "fixed0.90", "fixed0.95", "fixed0.98", "mixed", "mixed_sizes"):
        for n in (30, 100, 300, 1000):
            t = h = c = 0
            for _ in range(a.paths):
                q, w = book(kind, n, rng)
                o = (rng.random(n) < q).astype(float)
                pnl = w * (o - q)
                t += t_p(pnl) <= ALPHA
                if n >= 60 and pnl.std() > 0:
                    h += check_backtest(pnl, periods_per_year=n).best.p_alone <= ALPHA
                elif n >= 60:
                    h += 1  # every bet won: a zero-variance "sure thing"
                c += check_contracts(q, o, sizes=w).p_value <= ALPHA
            hs = f"{100 * h / a.paths:>5.1f}%" if n >= 60 else "   n/a"
            print(f"{kind:<12} {n:>5}  {100 * t / a.paths:>6.1f}%  {hs}  {100 * c / a.paths:>8.1f}%")
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
