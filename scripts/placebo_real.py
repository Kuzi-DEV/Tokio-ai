"""False-positive rates on REAL market history, not simulated markets.

The simulation studies answer "does the test hold on data that behaves like
a market?". The obvious objection is that GARCH is not a market. This study
removes the simulation from the returns entirely: it uses 20 years of real
daily prices for ten assets -- equity indices, single stocks, bonds, gold,
oil and FX, through 2008, 2020 and 2022 -- and simulates only the signal.

Each placebo condition is a random label series generated independently of
the prices, so it carries no information about them by construction. Any
"significant" verdict is a false positive, measured against the real
return path with every one of its real features: fat tails, volatility
clustering, crashes, regime changes, holiday gaps.

Placebo labels come in four persistence levels, from independent coin
flips to runs of months, because persistence is what breaks naive tests.

Run:  python scripts/placebo_real.py [--placebos N] [--workers N]
Needs network (Yahoo daily bars) and `pip install -e ".[calibration]"`.
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

from benchmark import _pairs, p_newey_west  # noqa: E402
from calibration_check import welch_p  # noqa: E402

from tokio_ai.check import check, forward_returns  # noqa: E402
from tokio_ai.tools.prices import fetch_daily_bars  # noqa: E402

ALPHA = 0.05
ASSETS = ("SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "AAPL", "JPM", "XOM", "EURUSD=X")
HORIZONS = (1, 5, 20)
# Mean run length of the placebo label, in bars. 1 = independent coin flips.
PERSISTENCE = (1, 5, 30, 120)
TESTS = ("welch_t", "newey_west", "tokio_hod", "tokio_rot")


def placebo_labels(n: int, run: int, rng: random.Random) -> list[bool]:
    if run == 1:
        return [rng.random() < 0.3 for _ in range(n)]
    state, out = rng.random() < 0.5, []
    for _ in range(n):
        if rng.random() < 1 / run:
            state = not state
        out.append(state)
    return out


def run_asset(job):
    symbol, returns, placebos = job
    n = len(returns)
    counts = {(run, h, t): 0 for run in PERSISTENCE for h in HORIZONS for t in TESTS}
    usable = {(run, h): 0 for run in PERSISTENCE for h in HORIZONS}
    # Seeded per asset so each asset's placebos are reproducible on their own.
    rng = random.Random(f"placebo:{symbol}")
    for run in PERSISTENCE:
        for _ in range(placebos):
            labels = placebo_labels(n, run, rng)
            for h in HORIZONS:
                res = check(returns, labels, horizon=h, iters=2000)
                if res.verdict == "NOT REPORTABLE":
                    continue
                usable[(run, h)] += 1
                fwd = forward_returns(returns, h)
                x, y = _pairs(labels, fwd)
                ps = {
                    "welch_t": welch_p(labels, fwd),
                    "newey_west": p_newey_west(x, y, h),
                    "tokio_hod": res.p_hodrick,
                    "tokio_rot": res.p_rotation,
                }
                for t in TESTS:
                    counts[(run, h, t)] += ps[t] is not None and ps[t] <= ALPHA
    return symbol, n, counts, usable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placebos", type=int, default=300, help="per asset x persistence")
    parser.add_argument("--range", default="20y")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    jobs, windows = [], {}
    for symbol in ASSETS:
        bars = fetch_daily_bars(symbol, args.range)
        closes = [b.adj_close for b in bars]
        returns = [None] + [(c / p - 1) if p else None for p, c in zip(closes, closes[1:])]
        windows[symbol] = (bars[0].date, bars[-1].date)
        jobs.append((symbol, returns, args.placebos))

    print(f"Real-data placebo study -- {args.placebos} random independent signals per asset "
          f"per persistence level, {args.range} of daily bars. A calibrated test fires at "
          f"{ALPHA:.0%}.\n")
    for symbol, (a, b) in windows.items():
        print(f"- {symbol}: {a} to {b}")
    print()

    totals = {(run, h, t): [0, 0] for run in PERSISTENCE for h in HORIZONS for t in TESTS}
    per_asset_worst = {t: (0.0, "") for t in TESTS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for symbol, n, counts, usable in pool.map(run_asset, jobs):
            for (run, h, t), c in counts.items():
                u = usable[(run, h)]
                totals[(run, h, t)][0] += c
                totals[(run, h, t)][1] += u
                if u and c / u > per_asset_worst[t][0]:
                    per_asset_worst[t] = (c / u, f"{symbol}, run {run}, h={h}")

    print("Pooled across all ten assets:\n")
    print("| signal run length | horizon | t-test | Newey-West | TokIO hodrick (default) | TokIO rotation |")
    print("|---:|---:|---:|---:|---:|---:|")
    worst = {t: 0.0 for t in TESTS}
    for run in PERSISTENCE:
        for h in HORIZONS:
            cells = []
            for t in TESTS:
                c, u = totals[(run, h, t)]
                rate = c / u if u else float("nan")
                worst[t] = max(worst[t], rate)
                cells.append(f"{rate:.1%}")
            print(f"| {run} | {h} | " + " | ".join(cells) + " |")
    print("\n| test | worst pooled rate | worst single asset |")
    print("|---|---:|---|")
    for t in TESTS:
        rate, where = per_asset_worst[t]
        print(f"| {t} | {worst[t]:.1%} | {rate:.1%} ({where}) |")
    print(f"\nelapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
