import math

import pytest

np = pytest.importorskip("numpy")

import tokio_ai  # noqa: E402
from tokio_ai import OverfitResult, check_backtest, probability_of_overfitting  # noqa: E402


def _grid(seed, k=30, n=2000, planted=None):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * 0.01
    g = {f"s{j}": np.where(rng.random(n) < 0.5, -1.0, 1.0) * r for j in range(k)}
    if planted:
        g["real"] = rng.standard_normal(n) * 0.01 + planted
    return g


def test_exported():
    assert tokio_ai.probability_of_overfitting is probability_of_overfitting
    assert tokio_ai.OverfitResult is OverfitResult


def test_noise_grids_average_near_one_half():
    pbos = [probability_of_overfitting(_grid(s)).pbo for s in range(12)]
    assert 0.3 < float(np.mean(pbos)) < 0.7


def test_a_real_edge_is_found_and_persists():
    res = probability_of_overfitting(_grid(1, planted=0.0015))
    assert res.pbo < 0.1
    assert res.most_selected[0][0] == "real"
    assert res.median_oos_sharpe > 1.0


def test_split_count_is_all_half_combinations():
    res = probability_of_overfitting(_grid(2, k=5), blocks=8)
    assert res.n_splits == math.comb(8, 4)
    assert res.bars_used == (2000 // 8) * 8


def test_blocks_validation():
    with pytest.raises(ValueError, match="even"):
        probability_of_overfitting(_grid(3, k=3), blocks=7)
    with pytest.raises(ValueError, match="too few"):
        probability_of_overfitting({"a": np.zeros(100) + 1e-3, "b": np.ones(100) * 1e-3})


def test_needs_two_variants():
    with pytest.raises(ValueError, match="two variants"):
        probability_of_overfitting({"a": np.ones(500)})


def test_identical_variants_sit_at_the_median():
    x = np.random.default_rng(4).standard_normal(1600) * 0.01
    res = probability_of_overfitting({"a": x, "b": x.copy(), "c": x.copy()})
    assert res.pbo == 1.0  # ties share the median rank, which counts as "not above it"
    assert res.median_oos_percentile == pytest.approx(0.5)


def test_check_backtest_includes_pbo_for_grids_only():
    g = _grid(5, k=6)
    res = check_backtest(g)
    assert res.overfitting is not None
    assert "Probability of backtest overfitting" in str(res)
    assert check_backtest({"a": g["s0"], "b": g["s1"]}).overfitting is None


def test_pbo_in_check_backtest_is_net_of_costs():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(2000) * 0.01
    pos = {f"s{j}": np.where(rng.random(2000) < 0.5, -1.0, 1.0) for j in range(5)}
    pnl = {k: p * r for k, p in pos.items()}
    net = {k: pnl[k] - 0.001 * np.abs(np.diff(p, prepend=0.0)) for k, p in pos.items()}
    a = check_backtest(pnl, positions=pos, costs=0.001).overfitting
    b = probability_of_overfitting(net)
    assert a.pbo == pytest.approx(b.pbo)


def test_str_and_verdict_tiers():
    s = str(probability_of_overfitting(_grid(7, planted=0.0015)))
    assert "Probability of backtest overfitting" in s and "Not a significance test" in s
