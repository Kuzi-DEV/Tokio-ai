"""Common technical-pattern testing: derive a simple feature from daily bars,
split days into two groups by comparing the feature to a threshold, and run
the split through the rigor engine's permutation test.

This exists because asking an LLM to manually bucket hundreds of raw price
bars inline (as tool_hypothesis alone requires) is unreliable and
token-expensive -- classic case for "let code do the arithmetic, let the
model decide what to ask for."
"""

from __future__ import annotations

from ..check import check
from ..family import check_many
from ..rigor.ledger import TestLedger
from ..rigor.stats import PermutationResult
from .prices import DailyBar, fetch_daily_bars

FEATURES = ("daily_return", "gap_pct", "volume_ratio")
OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}
VOLUME_LOOKBACK = 20


def compute_feature(bars: list[DailyBar], feature: str) -> list[float | None]:
    """One value per bar, aligned by index. None where undefined (not enough
    history yet, e.g. the first bar has no prior close to diff against)."""
    if feature == "daily_return":
        out: list[float | None] = [None]
        for prev, cur in zip(bars, bars[1:]):
            out.append((cur.close - prev.close) / prev.close if prev.close else None)
        return out
    if feature == "gap_pct":
        out = [None]
        for prev, cur in zip(bars, bars[1:]):
            out.append((cur.open - prev.close) / prev.close if prev.close else None)
        return out
    if feature == "volume_ratio":
        out = [None] * len(bars)
        for i in range(VOLUME_LOOKBACK, len(bars)):
            window = bars[i - VOLUME_LOOKBACK : i]
            avg_vol = sum(b.volume for b in window) / len(window)
            out[i] = (bars[i].volume / avg_vol) if avg_vol else None
        return out
    raise ValueError(f"unknown feature {feature!r}, must be one of {FEATURES}")


def paired_forward_returns(
    bars: list[DailyBar], feature: str, op: str, threshold: float, horizon_days: int
) -> tuple[list[bool], list[float]]:
    """Time-ordered (condition_met, forward_return) series, aligned by index.

    Returns two parallel lists rather than two buckets, because the time
    ORDER is not incidental here -- it is what makes an honest test
    possible. Forward returns over a multi-day horizon come from
    overlapping windows and are therefore autocorrelated, and conditions
    cluster in time. `circular_shift_test` needs both series intact to
    account for that; a pre-split pair of buckets has already thrown the
    information away. See `bucket_forward_returns` for the bucketed view.
    """
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}, must be one of {list(OPS)}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    values = compute_feature(bars, feature)
    cmp = OPS[op]
    labels: list[bool] = []
    forward_returns: list[float] = []
    for i, v in enumerate(values):
        exit_i = i + horizon_days
        if v is None or exit_i >= len(bars) or bars[i].close == 0:
            continue
        labels.append(bool(cmp(v, threshold)))
        forward_returns.append(bars[exit_i].close / bars[i].close - 1)
    return labels, forward_returns


def bucket_forward_returns(
    bars: list[DailyBar], feature: str, op: str, threshold: float, horizon_days: int
) -> tuple[list[float], list[float]]:
    """Split days into (condition-met, condition-not-met) groups and return
    each day's forward return from that day's close to horizon_days later.

    Kept as the bucketed view of `paired_forward_returns`. Note that testing
    these two groups against each other with a plain shuffled permutation
    test is NOT sound for horizon_days > 1 -- the windows overlap. Use
    `paired_forward_returns` + `circular_shift_test` for a verdict.
    """
    labels, forward_returns = paired_forward_returns(bars, feature, op, threshold, horizon_days)
    group_a = [r for flag, r in zip(labels, forward_returns) if flag]
    group_b = [r for flag, r in zip(labels, forward_returns) if not flag]
    return group_a, group_b


def test_return_pattern(
    ledger: TestLedger,
    symbol: str,
    feature: str,
    op: str,
    threshold: float,
    horizon_days: int,
    range_: str = "10y",
) -> str:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}, must be one of {list(OPS)}")
    bars = fetch_daily_bars(symbol, range_)
    # Route through check() so the agent and the library give the same
    # answer to the same question. Bar returns close-to-close, and the
    # condition at each bar's close; check() compounds bars i+1..i+h,
    # which is exactly close[i+h] / close[i] - 1.
    returns: list[float | None] = [None] + [
        (cur.close / prev.close - 1) if prev.close else None for prev, cur in zip(bars, bars[1:])
    ]
    cmp = OPS[op]
    condition = [None if v is None else bool(cmp(v, threshold)) for v in compute_feature(bars, feature)]
    name = f"{symbol}_{feature}_{op}{threshold}_{horizon_days}d"
    result = check(returns, condition, horizon=horizon_days, ledger=ledger, name=name)
    verdict = ledger.verdict(name)
    second = result._second_opinion()
    if second:
        verdict += " " + second
    # The model has no reliable notion of "today" or the actual data window
    # fetched -- hand it the real dates so it reports facts, not a guess.
    data_window = f"data window: {bars[0].date} to {bars[-1].date}" if bars else "no data"
    return f"{verdict} ({data_window})"


def test_pattern_grid(
    ledger: TestLedger,
    symbol: str,
    feature: str,
    op: str,
    thresholds: list[float],
    horizons: list[int],
    range_: str = "10y",
) -> str:
    """Every threshold at every horizon as one family (tokio_ai.check_many)."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}, must be one of {list(OPS)}")
    if not thresholds or not horizons:
        raise ValueError("thresholds and horizons must each list at least one value")
    bars = fetch_daily_bars(symbol, range_)
    returns: list[float | None] = [None] + [
        (cur.close / prev.close - 1) if prev.close else None for prev, cur in zip(bars, bars[1:])
    ]
    values = compute_feature(bars, feature)
    cmp = OPS[op]
    grid = {
        f"{feature}{op}{t}": [None if v is None else bool(cmp(v, t)) for v in values]
        for t in thresholds
    }
    result = check_many(returns, grid, horizon=[int(h) for h in horizons])
    # Every variant goes into the session ledger too. Otherwise the grid is a
    # loophole: a variant that fails the grid correction could be re-tested
    # alone as the session's "first" test and come back significant.
    for m in result.members:
        if m.reportable:
            ledger.record(
                f"{symbol}_{m.name}".replace(" ", ""),
                PermutationResult(m.gap, m.p_raw, m.n_condition, m.n_other,
                                  iters=0, seed=None, statistic="hodrick_hac"),
            )
    data_window = f"data window: {bars[0].date} to {bars[-1].date}" if bars else "no data"
    return f"{symbol} ({data_window})\n{result}"
