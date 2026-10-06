import math
from statistics import NormalDist

import pytest

np = pytest.importorskip("numpy")

from tokio_ai import check_backtest  # noqa: E402
from tokio_ai.sharpe import (  # noqa: E402
    deflated_sharpe_ratio, expected_max_sharpe, min_track_record_length, probabilistic_sharpe_ratio)

N = NormalDist()


def _x(n=1000, seed=0, edge=0.0):
    return np.random.default_rng(seed).standard_normal(n) * 0.01 + edge


def test_psr_matches_the_published_formula():
    x = _x(edge=0.0005)
    m, sd = x.mean(), x.std(ddof=1)
    d = x - m
    g3 = (d ** 3).mean() / (d ** 2).mean() ** 1.5
    g4 = (d ** 4).mean() / (d ** 2).mean() ** 2
    sr = m / sd
    want = N.cdf(sr * math.sqrt(len(x) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    assert probabilistic_sharpe_ratio(x).psr == pytest.approx(want)


def test_expected_max_sharpe_grows_with_trials_and_is_zero_for_one():
    assert expected_max_sharpe(1, 0.01) == 0
    a, b = expected_max_sharpe(10, 0.01), expected_max_sharpe(1000, 0.01)
    assert 0 < a < b
    # BLdP 2014's own example: 100 trials, Var(SR)=1/2 (annualised) -> E[max] ~ 1.77... scaled
    assert expected_max_sharpe(100, 0.5) == pytest.approx(math.sqrt(0.5) * (
        (1 - 0.5772156649015329) * N.inv_cdf(0.99) + 0.5772156649015329 * N.inv_cdf(1 - 1 / (100 * math.e))))


def test_dsr_is_stricter_than_psr():
    x = _x(edge=0.0008)
    psr = probabilistic_sharpe_ratio(x).psr
    dsr = deflated_sharpe_ratio(x, trials=50).psr
    assert dsr < psr


def test_dsr_on_a_grid_tests_the_highest_sharpe_variant():
    g = {f"s{i}": _x(seed=i) for i in range(8)}
    g["s3"] = g["s3"] + 0.002
    d = deflated_sharpe_ratio(g)
    assert d.trials == 8 and d.sharpe == pytest.approx(g["s3"].mean() / g["s3"].std(ddof=1))


def test_dependence_lowers_psr_on_overlapping_returns():
    x = _x(3000, 4)
    c = np.concatenate([[0], np.cumsum(x)])
    tranche = (c[20:] - c[:-20]) / 20 + 0.0003
    plain = probabilistic_sharpe_ratio(tranche)
    dep = probabilistic_sharpe_ratio(tranche, dependence=True)
    assert dep.variance_ratio > 5
    assert dep.psr < plain.psr


def test_min_track_record_length():
    x = _x(edge=0.001)
    n = min_track_record_length(x)
    assert n is not None and n < len(x)  # a strong edge needs fewer bars than it has
    assert min_track_record_length(_x(edge=-0.001)) is None


def test_check_backtest_reports_both_versions():
    res = check_backtest(_x(edge=0.0005))
    assert res.deflated is not None and res.deflated_published is not None
    assert "Probabilistic Sharpe Ratio" in str(res)
    res = check_backtest({f"s{i}": _x(seed=i) for i in range(5)})
    assert "Deflated Sharpe Ratio" in str(res)
