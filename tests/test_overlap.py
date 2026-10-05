import math
import random

import pytest

from tokio_ai.rigor.overlap import _hodrick_python, default_bandwidth, hodrick_test


def _series(n, seed):
    rng = random.Random(seed)
    r = [rng.gauss(0, 0.01) for _ in range(n)]
    lab = [rng.random() < 0.3 for _ in range(n)]
    return r, lab


def test_regrouping_identity_holds_exactly():
    # sum_i a_i * y_i (by observation) must equal sum_t r_t * B_t (by bar):
    # the whole method rests on this. Checked with bandwidth 0, where
    # z = S / sqrt(sum u^2), by rebuilding S the direct way.
    r, lab = _series(300, 1)
    h = 7
    logs = [math.log1p(x) for x in r]
    valid = list(range(300 - h))
    abar = sum(lab[i] for i in valid) / len(valid)
    rbar = sum(logs) / len(logs)
    direct = sum((lab[i] - abar) * sum(logs[i + 1 : i + 1 + h]) for i in valid)
    a = [(lab[i] - abar) if i in set(valid) else 0.0 for i in range(300)]
    B = [sum(a[max(t - h, 0) : t]) for t in range(300)]
    V = sum(((logs[t] - rbar) * B[t]) ** 2 for t in range(300))
    res = hodrick_test(r, lab, h, bandwidth=0)
    assert res.z == pytest.approx(direct / math.sqrt(V), rel=1e-9)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("h", [1, 5, 20])
def test_python_and_numpy_paths_agree(seed, h):
    np = pytest.importorskip("numpy")
    from tokio_ai.rigor.overlap import _hodrick_numpy

    r, lab = _series(700, seed)
    rng = random.Random(seed + 50)
    for i in rng.sample(range(700), 15):
        r[i] = None
    for i in rng.sample(range(700), 15):
        lab[i] = None
    a = _hodrick_python(r, lab, h, None)
    b = _hodrick_numpy(np, r, lab, h, None)
    assert a.z == pytest.approx(b.z, rel=1e-9)
    assert a.bandwidth == b.bandwidth


def test_windows_touching_a_missing_bar_are_excluded():
    # A missing bar drops the labels whose windows contain it; the result
    # must change, and stay finite rather than propagating NaN.
    r, lab = _series(400, 3)
    r2 = list(r)
    r2[200] = None
    res_full = hodrick_test(r, lab, 5, bandwidth=0)
    res_gap = hodrick_test(r2, lab, 5, bandwidth=0)
    assert res_full.z != res_gap.z
    assert math.isfinite(res_gap.z)


def test_default_bandwidth_is_the_newey_west_rule():
    assert default_bandwidth(100) == 4
    assert default_bandwidth(1000) == 6
    assert default_bandwidth(1_000_000) == 30


def test_total_loss_bar_falls_back_to_simple_returns():
    r, lab = _series(300, 4)
    r[100] = -1.0
    res = hodrick_test(r, lab, 3)
    assert math.isfinite(res.z)


def test_no_usable_labels_gives_no_evidence():
    res = hodrick_test([0.01] * 50, [None] * 50, 5)
    assert res.p_value == 1.0


def test_null_rejection_rate_is_near_nominal():
    # Small in-suite calibration: iid returns, independent labels.
    hits = 0
    for seed in range(200):
        r, lab = _series(400, 1000 + seed)
        hits += hodrick_test(r, lab, 5).p_value <= 0.05
    assert 2 <= hits <= 20  # 1%-10% of 200


def test_bandwidth_covers_the_conditions_own_time_scale():
    from tokio_ai.rigor.overlap import hac_bandwidth, mean_run_length

    assert mean_run_length([True] * 10 + [False] * 10) == 10
    assert mean_run_length([True, False] * 10) == 1
    # Short runs: the Newey-West rule of thumb wins.
    assert hac_bandwidth(1000, [True, False] * 500, 1) == default_bandwidth(1000)
    # Long runs: run length + horizon wins.
    assert hac_bandwidth(1000, [True] * 100 + [False] * 100, 5) == 105


def test_slow_mean_drift_does_not_fool_a_persistent_placebo():
    # The USO failure, reproduced: returns whose MEAN wanders over months,
    # and a random condition that persists about as long. With only the
    # rule-of-thumb bandwidth this fired well above 5%.
    np = pytest.importorskip("numpy")
    hits = hits_rule = 0
    trials = 150
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        n = 2500
        drift = np.zeros(n)
        level = 0.0
        for t in range(n):
            if rng.random() < 1 / 150:
                level = rng.normal(0, 0.003)
            drift[t] = level
        r = list(drift + rng.normal(0, 0.02, n))
        state, lab = bool(rng.random() < 0.5), []
        for _ in range(n):
            if rng.random() < 1 / 120:
                state = not state
            lab.append(state)
        hits += hodrick_test(r, lab, 20).p_value <= 0.05
        hits_rule += hodrick_test(r, lab, 20, bandwidth=default_bandwidth(n)).p_value <= 0.05
    assert hits / trials <= 0.10
    assert hits < hits_rule


@pytest.mark.parametrize("n,k,L", [(50, 3, 0), (50, 3, 7), (301, 5, 40), (64, 2, 63), (64, 2, 500)])
def test_long_run_cov_matches_the_direct_lag_sum(n, k, L):
    # The frequency-domain shortcut must equal the definition exactly,
    # including zero lags, every lag, and a bandwidth past the series end.
    np = pytest.importorskip("numpy")
    from tokio_ai.rigor.overlap import bartlett_long_run_cov

    rng = np.random.default_rng(n + k + L)
    U = rng.normal(size=(k, n))
    U[1] += np.roll(U[0], 3)  # cross-dependence at a nonzero lag
    Lc = min(L, n - 1)
    ref = np.zeros((k, k))
    for lag in range(-Lc, Lc + 1):
        w = 1 - abs(lag) / (Lc + 1)
        for j in range(k):
            for q in range(k):
                if lag >= 0:
                    ref[j, q] += w * (U[j, lag:] @ U[q, : n - lag])
                else:
                    ref[j, q] += w * (U[j, : n + lag] @ U[q, -lag:])
    got = bartlett_long_run_cov(np, U, L)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))
    assert np.allclose(got, got.T)
