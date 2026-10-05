from tokio_ai.tools.patterns import bucket_forward_returns, compute_feature
from tokio_ai.tools.prices import DailyBar


def _bar(date: str, open_: float, close: float, volume: int = 1000) -> DailyBar:
    return DailyBar(date=date, open=open_, high=max(open_, close), low=min(open_, close),
                     close=close, adj_close=close, volume=volume)


def test_daily_return_first_bar_is_none():
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110)]
    values = compute_feature(bars, "daily_return")
    assert values[0] is None
    assert abs(values[1] - 0.10) < 1e-9


def test_gap_pct_uses_open_vs_prior_close():
    bars = [_bar("2024-01-01", 100, 105), _bar("2024-01-02", 110, 108)]
    values = compute_feature(bars, "gap_pct")
    assert values[0] is None
    assert abs(values[1] - ((110 - 105) / 105)) < 1e-9


def test_volume_ratio_needs_lookback_window():
    bars = [_bar(f"2024-01-{i:02d}", 100, 100, volume=1000) for i in range(1, 25)]
    values = compute_feature(bars, "volume_ratio")
    assert all(v is None for v in values[:20])
    assert values[20] is not None


def test_unknown_feature_raises():
    bars = [_bar("2024-01-01", 100, 100)]
    try:
        compute_feature(bars, "nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bucket_forward_returns_condition_met_goes_to_group_a():
    # Exactly one bar (index 1) has both a defined feature and a valid
    # lookahead: index 0 has no prior bar (feature=None), index 2 has no bar
    # to look ahead to at horizon=1. Its +10% return clears the 5% threshold.
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110), _bar("2024-01-03", 110, 105)]
    group_a, group_b = bucket_forward_returns(bars, "daily_return", ">", 0.05, horizon_days=1)
    assert group_b == []
    assert len(group_a) == 1
    assert abs(group_a[0] - (105 / 110 - 1)) < 1e-9


def test_bucket_forward_returns_condition_not_met_goes_to_group_b():
    # Same shape, but index 1's return (+2%) falls below the 5% threshold.
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 102), _bar("2024-01-03", 102, 105)]
    group_a, group_b = bucket_forward_returns(bars, "daily_return", ">", 0.05, horizon_days=1)
    assert group_a == []
    assert len(group_b) == 1
    assert abs(group_b[0] - (105 / 102 - 1)) < 1e-9


def test_bucket_forward_returns_drops_bars_without_enough_lookahead():
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110)]
    group_a, group_b = bucket_forward_returns(bars, "daily_return", ">", 0.05, horizon_days=5)
    assert group_a == []
    assert group_b == []


def test_invalid_op_raises():
    bars = [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 100, 110)]
    try:
        bucket_forward_returns(bars, "daily_return", "==", 0.0, horizon_days=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def _random_bars(n, seed):
    import random

    rng = random.Random(seed)
    price, bars = 100.0, []
    for i in range(n):
        prev, price = price, price * (1 + rng.gauss(0, 0.012))
        bars.append(_bar(f"d{i:05d}", prev, price))
    return bars


def test_grid_variants_count_in_the_session_ledger(monkeypatch):
    # The loophole this closes: run a grid, see a variant fail the grid
    # correction, then re-test it alone as the session's "first" test. Every
    # grid variant must already be in the ledger, so the single test is
    # corrected for all of them.
    import pytest

    pytest.importorskip("numpy")
    import tokio_ai.tools.patterns as patterns
    from tokio_ai.rigor.ledger import TestLedger

    bars = _random_bars(1500, 1)
    monkeypatch.setattr(patterns, "fetch_daily_bars", lambda symbol, range_: bars)
    ledger = TestLedger()
    out = patterns.test_pattern_grid(ledger, "FAKE", "daily_return", "<",
                                     [-0.01, -0.02], [1, 5, 20])
    assert "6 variants tested as one family" in out
    assert len(ledger.tests) == 6
    patterns.test_return_pattern(ledger, "FAKE", "daily_return", "<", -0.01, 5)
    assert len(ledger.tests) == 7
    assert "correcting for 7 hypothesis test(s)" in ledger.verdict("FAKE_daily_return_<-0.01_5d")
