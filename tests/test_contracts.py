import math

import pytest

np = pytest.importorskip("numpy")

import tokio_ai  # noqa: E402
from tokio_ai import ContractsResult, check_contracts  # noqa: E402
from tokio_ai.contracts import (  # noqa: E402
    _clopper_pearson_upper,
    _lattice_upper_tail_exact,
    _pb_upper_tail_exact,
    _saddlepoint_upper,
)


def test_exported():
    assert tokio_ai.check_contracts is check_contracts
    assert tokio_ai.ContractsResult is ContractsResult


def test_perfect_short_record_at_98c_is_not_evidence():
    res = check_contracts([0.98] * 30, [1] * 30)
    assert res.p_value == pytest.approx(0.98 ** 30)
    assert not res.significant
    assert res.bets_for_perfect_record == math.ceil(math.log(0.05) / math.log(0.98))
    assert "even a perfect record" in str(res)


def test_exact_matches_brute_force_enumeration():
    q = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
    for k in range(6):
        brute = 0.0
        for mask in range(32):
            bits = [(mask >> i) & 1 for i in range(5)]
            if sum(bits) >= k:
                brute += math.prod(qi if b else 1 - qi for qi, b in zip(q, bits))
        assert _pb_upper_tail_exact(np, q, k) == pytest.approx(brute)


def test_lattice_exact_matches_brute_force_with_sizes():
    q = np.array([0.7, 0.9, 0.95, 0.8])
    w = np.array([3, 1, 5, 2])
    for s in range(12):
        brute = 0.0
        for mask in range(16):
            bits = [(mask >> i) & 1 for i in range(4)]
            if sum(wi * b for wi, b in zip(w, bits)) >= s:
                brute += math.prod(qi if b else 1 - qi for qi, b in zip(q, bits))
        assert _lattice_upper_tail_exact(np, w, q, s) == pytest.approx(brute)


def test_saddlepoint_matches_exact_for_equal_sizes():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.9, 0.99, 800)
    k = int(q.sum()) + 5
    assert _saddlepoint_upper(np, np.ones(800), q, float(k), 1.0) == pytest.approx(
        _pb_upper_tail_exact(np, q, k), rel=0.01)


def test_fees_raise_the_breakeven():
    a = check_contracts([0.9] * 200, [1] * 190 + [0] * 10)
    b = check_contracts([0.9] * 200, [1] * 190 + [0] * 10, fees=0.02)
    assert b.p_value > a.p_value
    assert b.pnl == pytest.approx(a.pnl - 0.02 * 200)


def test_real_edge_is_found_and_worse_than_fair_is_flagged():
    rng = np.random.default_rng(1)
    q = np.full(2000, 0.9)
    good = check_contracts(q, rng.random(2000) < 0.93)
    bad = check_contracts(q, rng.random(2000) < 0.87)
    assert good.significant and good.p_worse > 0.5
    assert not bad.significant and bad.p_worse < 0.05
    assert "Worse than fair" in str(bad)


def test_groups_merge_fills_on_the_same_contract():
    res = check_contracts([0.9, 0.92, 0.95], [1, 1, 0], sizes=[2, 3, 1], groups=["a", "a", "b"])
    assert res.n_bets == 2 and res.merged == 1
    with pytest.raises(ValueError, match="different outcomes"):
        check_contracts([0.9, 0.9], [1, 0], groups=["a", "a"])


def test_input_validation():
    with pytest.raises(ValueError, match="0.95, not 95"):
        check_contracts([95], [1])
    with pytest.raises(ValueError, match="0/1"):
        check_contracts([0.9], [2])
    with pytest.raises(ValueError, match="reaches 1"):
        check_contracts([0.99], [1], fees=0.02)
    with pytest.raises(ValueError, match="same length"):
        check_contracts([0.9, 0.9], [1])


def test_clopper_pearson_upper():
    # 0 of n: the exact bound is 1 - 0.05**(1/n)
    assert _clopper_pearson_upper(np, 0, 100) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)
    assert _clopper_pearson_upper(np, 10, 892) == pytest.approx(0.0189417339, rel=1e-8)  # scipy beta.ppf(.95, 11, 882)


def test_calibrated_on_fair_favourites():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(400):
        q = rng.uniform(0.9, 0.99, 300)
        hits += check_contracts(q, rng.random(300) < q).p_value <= 0.05
    assert hits / 400 <= 0.07


def test_cli_contracts_mode(tmp_path, capsys):
    from tokio_ai.backtest_cli import main

    p = tmp_path / "trades.csv"
    rows = ["ticker,price,won,count"] + [f"m{i},96,{'lost' if i % 20 == 0 else 'won'},{1 + i % 5}"
                                         for i in range(200)]
    p.write_text("\n".join(rows), encoding="utf-8")
    assert main([str(p), "--contracts", "price", "won", "--sizes", "count", "--fees", "0.003",
                 "--groups", "ticker"]) == 0
    out = capsys.readouterr().out
    assert "200 bets" in out and "Losses: 10" in out
    bad = tmp_path / "bad.csv"
    bad.write_text("price,won\n0.9,maybe\n", encoding="utf-8")
    assert main([str(bad), "--contracts", "price", "won"]) == 2
    assert "win/loss" in capsys.readouterr().err


def test_mixed_sizes_report_the_contract_weighted_loss_rate():
    res = check_contracts([0.5] * 21, [1] + [0] * 20, sizes=[1000] + [1] * 20)
    text = str(res)
    assert res.contract_loss_rate == pytest.approx(20 / 1020)
    assert "Weighted by size" in text and "per bet" in text
    assert "rule out a losing book" not in text


def test_saddlepoint_on_the_mean_and_below_zero():
    w, q = np.ones(50), np.full(50, 0.95)
    assert _saddlepoint_upper(np, w, q, 48.0, 1.0) == pytest.approx(_pb_upper_tail_exact(np, q, 48), rel=0.02)
    assert _saddlepoint_upper(np, w, q, 0.001, 1.0) == 1.0


def test_clopper_pearson_is_fast_on_big_books():
    import time

    t = time.time()
    _clopper_pearson_upper(np, 10000, 20000)
    assert time.time() - t < 0.5


def test_exit_values_replace_the_settlement_value():
    # 3 bets at 0.9: one stopped at 0.6 (it would have lost), one stopped at
    # 0.5 (it would have won), one held and won.
    res = check_contracts([0.9, 0.9, 0.9], [0, 1, 1], exit_values=[0.6, 0.5, None])
    assert res.n_exited == 2
    assert res.pnl == pytest.approx((0.6 - 0.9) + (0.5 - 0.9) + (1 - 0.9))
    assert res.exit_pnl_vs_settlement == pytest.approx((0.6 - 0) + (0.5 - 1))
    assert res.n_losses == 2
    assert "closed early" in str(res)


def test_exits_tested_against_the_bounded_null():
    # The p-value is P(sum of Bernoulli(q) >= realized total value).
    q = np.full(40, 0.9)
    won = np.ones(40)
    ex = [None] * 36 + [0.7] * 4
    res = check_contracts(q, won, exit_values=ex)
    total = 36 + 4 * 0.7  # 38.8 -> needs 39+ wins
    assert res.p_value == pytest.approx(_pb_upper_tail_exact(np, q, math.ceil(total)))


def test_exit_values_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        check_contracts([0.9], [1], exit_values=[1.2])
    with pytest.raises(ValueError, match="length"):
        check_contracts([0.9, 0.9], [1, 1], exit_values=[0.5])
    with pytest.raises(ValueError, match="mixes exits"):
        check_contracts([0.9, 0.9], [1, 1], groups=["a", "a"], exit_values=[0.5, None])


def test_stopped_books_hold_size_on_fair_markets():
    from statistics import NormalDist

    N = NormalDist()
    rng = np.random.default_rng(9)
    hits = 0
    for _ in range(150):
        n = 200
        p0 = rng.uniform(0.85, 0.98, n)
        x = np.array([N.inv_cdf(v) for v in p0]) * math.sqrt(5)
        ex = np.full(n, np.nan)
        alive = np.ones(n, bool)
        for k in range(1, 5):
            x = x + rng.standard_normal(n)
            price = np.array([N.cdf(v) for v in x / math.sqrt(5 - k)])
            hit = alive & (price <= p0 - 0.15)
            ex[hit] = price[hit]
            alive &= ~hit
        x = x + rng.standard_normal(n)
        hits += check_contracts(p0, (x > 0).astype(float), exit_values=ex).p_value <= 0.05
    assert hits / 150 <= 0.06


def test_cost_margin_is_the_flip_point():
    q = np.full(2000, 0.9)
    won = np.random.default_rng(12).random(2000) < 0.93
    res = check_contracts(q, won)
    m = res.cost_margin_edge
    assert res.significant and m is not None and m > 0
    assert check_contracts(q, won, fees=m * 0.95).significant
    assert not check_contracts(q, won, fees=m * 1.05).significant
    assert "Robustness" in str(res)


def test_worse_margin_and_fragile_warning():
    won = np.random.default_rng(13).random(1500) < 0.86
    res = check_contracts(np.full(1500, 0.9), won)
    m = res.cost_margin_worse
    assert res.p_worse <= 0.05 and m is not None and m > 0
    assert check_contracts(np.full(1500, 0.9 - m * 0.95), won).p_worse <= 0.05
    assert check_contracts(np.full(1500, 0.9 - m * 1.05), won).p_worse > 0.05
    # Move the prices most of the way to the flip point: what's left is < 1c.
    near = check_contracts(np.full(1500, 0.9 - (m - 0.005)), won)
    assert near.cost_margin_worse < 0.01 and "recording error" in str(near)


def test_no_margin_when_nothing_is_significant():
    res = check_contracts([0.9] * 100, [1] * 90 + [0] * 10)
    assert res.cost_margin_edge is None and res.cost_margin_worse is None
