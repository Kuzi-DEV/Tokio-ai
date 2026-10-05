import math
import random

import pytest

np = pytest.importorskip("numpy")

import tokio_ai  # noqa: E402
from tokio_ai import FamilyResult, check, check_many  # noqa: E402
from tokio_ai.family import _romano_wolf  # noqa: E402


def _noise(n, seed, sd=0.01):
    rng = random.Random(seed)
    return [rng.gauss(0, sd) for _ in range(n)]


def _momentum(r, lookback):
    growth = [1.0]
    for x in r:
        growth.append(growth[-1] * (1 + x))
    return [None if i + 1 < lookback else growth[i + 1] / growth[i + 1 - lookback] > 1
            for i in range(len(r))]


def test_exported_from_the_package():
    assert tokio_ai.check_many is check_many


def test_each_variants_raw_p_is_exactly_what_check_gives_it():
    r = _noise(800, 1)
    conds = {"m10": _momentum(r, 10), "m40": _momentum(r, 40), "down": [x < 0 for x in r]}
    res = check_many(r, conds, horizon=[1, 5])
    for m in res.members:
        name, h = m.name.split(" @h")
        alone = check(r, conds[name], horizon=int(h))
        assert m.p_raw == alone.p_value
        assert m.p_rotation == alone.p_rotation
        assert m.gap == pytest.approx(alone.gap)


def test_single_horizon_names_have_no_suffix():
    r = _noise(500, 2)
    res = check_many(r, {"a": [x > 0 for x in r]}, horizon=5)
    assert [m.name for m in res.members] == ["a"]


def test_adjusted_p_is_never_below_raw_and_never_above_holm():
    rng = random.Random(9003)  # distinct from the noise seed
    r = _noise(900, 3)
    conds = {f"c{k}": [rng.random() < 0.3 for _ in range(900)] for k in range(8)}
    res = check_many(r, conds, horizon=[1, 5, 20])
    members = [m for m in res.members if m.reportable]
    k = len(members)
    ps = sorted(m.p_raw for m in members)
    holm = {}
    running = 0.0
    for rank, p in enumerate(ps):
        running = max(running, min(1.0, p * (k - rank)))
        holm.setdefault(p, running)
    for m in members:
        assert m.p_adjusted >= m.p_raw - 1e-12
        assert m.p_adjusted <= holm[m.p_raw] + 1e-12


def test_near_duplicate_variants_are_not_punished_like_independent_ones():
    # Ten copies of the same condition are one test, not ten. Holm would
    # multiply the p-value by ten; the joint correction should barely move it.
    # (Moderate p on purpose: far in the tail, 20,000 draws can't resolve the
    # max-|z| probability and the Holm cap takes over.)
    rng = random.Random(9004)  # distinct from the noise seed
    n = 1500
    r = _noise(n, 4)
    base = [rng.random() < 0.3 for _ in range(n)]
    for i in range(n - 1):
        if base[i]:
            r[i + 1] += 0.0018
    conds = {f"copy{k}": list(base) for k in range(10)}
    res = check_many(r, conds, horizon=1)
    m = res.members[0]
    assert m.p_raw < 0.02
    assert m.p_adjusted < 3 * m.p_raw
    assert m.p_adjusted < m.p_raw * 10


def test_independent_variants_are_corrected_like_holm():
    # With genuinely unrelated conditions there is no dependence to exploit,
    # so the smallest adjusted p should be close to Bonferroni's k * p.
    rng = random.Random(9005)  # distinct from the noise seed
    n = 2000
    r = _noise(n, 5)
    conds = {f"c{k}": [rng.random() < 0.4 for _ in range(n)] for k in range(10)}
    res = check_many(r, conds, horizon=1)
    best = min((m for m in res.members if m.reportable), key=lambda m: m.p_raw)
    sidak = 1 - (1 - best.p_raw) ** 10
    assert best.p_adjusted == pytest.approx(sidak, abs=0.03)


def test_planted_edge_survives_and_noise_does_not():
    rng = random.Random(9006)  # distinct from the noise seed
    n = 2500
    r = _noise(n, 6)
    real = [rng.random() < 0.3 for _ in range(n)]
    for i in range(n - 1):
        if real[i]:
            r[i + 1] += 0.004
    conds = {"real": real}
    conds.update({f"noise{k}": [rng.random() < 0.3 for _ in range(n)] for k in range(6)})
    res = check_many(r, conds, horizon=1)
    sig = {m.name for m in res.significant}
    assert "real" in sig
    assert not any(name.startswith("noise") for name in sig)


def test_not_reportable_variants_are_excluded_from_the_family():
    r = _noise(600, 7)
    rare = [i < 10 for i in range(600)]
    res = check_many(r, {"rare": rare, "half": [x > 0 for x in r]}, horizon=1)
    by = {m.name: m for m in res.members}
    assert not by["rare"].reportable
    assert by["rare"].p_adjusted == 1.0
    assert "NOT REPORTABLE" in str(res)


def test_reproducible_with_a_seed():
    r = _noise(700, 8)
    conds = {"a": _momentum(r, 10), "b": _momentum(r, 30)}
    a = check_many(r, conds, horizon=[1, 5], seed=3)
    b = check_many(r, conds, horizon=[1, 5], seed=3)
    assert [m.p_adjusted for m in a.members] == [m.p_adjusted for m in b.members]


def test_str_is_a_readable_table():
    r = _noise(700, 9)
    res = check_many(r, {"a": _momentum(r, 10), "b": [x < 0 for x in r]}, horizon=[1, 5])
    text = str(res)
    assert isinstance(res, FamilyResult)
    assert "4 variants tested as one family" in text
    assert "p alone" in text and "p grid" in text
    assert "romano_wolf" in text


@pytest.mark.parametrize("bad", [
    dict(conditions={}),
    dict(horizon=[]),
    dict(horizon=[0, 5]),
    dict(alpha=2),
])
def test_bad_arguments_raise(bad):
    r = _noise(100, 10)
    kwargs = dict(conditions={"a": [x > 0 for x in r]}, horizon=1)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        check_many(r, **kwargs)


def test_length_mismatch_names_the_condition():
    with pytest.raises(ValueError, match="'short'"):
        check_many([0.0] * 50, {"short": [True] * 49})


def test_romano_wolf_with_identity_correlation_matches_sidak_stepdown():
    # Independent statistics: the step-down max-|z| is Sidak's step-down,
    # which has a closed form to check the Monte Carlo against.
    z = np.array([3.0, 2.5, 1.0, 0.2])
    p_adj = _romano_wolf(np, z, np.eye(4), seed=0)
    p = [math.erfc(abs(v) / math.sqrt(2)) for v in z]
    expected, running = [], 0.0
    for j, pj in enumerate(p):
        running = max(running, 1 - (1 - pj) ** (4 - j))
        expected.append(running)
    assert p_adj == pytest.approx(expected, abs=0.005)


def test_second_opinion_is_always_stated():
    # The grid output must say whether the rotation test agrees, so a reader
    # (or the agent) never has to guess -- the agent once claimed agreement
    # the output didn't contain.
    r = _noise(700, 11)
    res = check_many(r, {"a": _momentum(r, 10), "b": [x < 0 for x in r]}, horizon=[1, 5])
    assert "Second opinion" in str(res)
