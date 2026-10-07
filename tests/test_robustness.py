import math

import pytest

np = pytest.importorskip("numpy")

from tokio_ai import check_backtest  # noqa: E402
from tokio_ai.robustness import diagnose, min_backtest_length  # noqa: E402


def _market(n=2520, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * 0.01


def test_lookahead_is_flagged_and_an_honest_strategy_is_not():
    r = _market()
    cheat = np.sign(r)                       # knows this bar's return before it happens
    honest = np.concatenate([[1.0], np.sign(np.convolve(r, np.ones(20), "full")[:len(r) - 1])])
    bad = check_backtest(cheat * r, positions=cheat, asset_returns=r)
    assert bad.robustness.sharpe_delayed < 0.5 * bad.robustness.sharpe_rebuilt
    assert "lookahead" in str(bad)
    ok = check_backtest(honest * r + 0.0004, positions=honest, asset_returns=r + 0.0004)
    assert "lookahead" not in str(ok)


def test_no_delay_check_without_asset_returns():
    r = _market()
    res = check_backtest(r + 0.001, positions=np.ones(len(r)))
    assert res.robustness.sharpe_delayed is None
    assert "one bar later" not in str(res)


def test_symmetric_trimming_keeps_a_zero_edge_near_zero():
    # fat tails: dropping only the best days would push this far negative
    x = np.random.default_rng(1).standard_t(3, 20000) * 0.01
    x -= x.mean()
    d = diagnose(np, x, 252, "bars", "s", 1)
    assert abs(d.sharpe_trimmed) < 0.3
    assert d.removed == 200


def test_trades_trim_five_at_each_end():
    d = diagnose(np, _market(300), 50, "trades", "s", 1)
    assert d.removed == 5


def test_consistency_counts_positive_blocks():
    x = np.concatenate([np.full(700, 0.01), np.full(100, -0.001)])
    d = diagnose(np, x + _market(800) * 0.001, 252, "bars", "s", 1)
    assert d.positive_blocks == 7 and d.blocks == 8


def test_min_backtest_length():
    assert min_backtest_length(1, 1.0) is None
    assert min_backtest_length(10, -0.5) is None
    # Bailey et al. 2014: ~45 independent trials can reach an in-sample Sharpe of 1 within 5 years
    years = min_backtest_length(45, 1.0)
    assert 4 < years < 6
    assert min_backtest_length(1000, 1.0) > years


def test_asset_returns_must_match_length():
    r = _market(500)
    with pytest.raises(ValueError, match="asset_returns"):
        check_backtest(r + 0.002, positions=np.ones(500), asset_returns=r[:-1])
