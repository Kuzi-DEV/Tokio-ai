import math

import pytest

np = pytest.importorskip("numpy")

import tokio_ai  # noqa: E402
from tokio_ai import BacktestResult, check_backtest  # noqa: E402
from tokio_ai.backtest import andrews_bandwidth, fixed_b_scale  # noqa: E402
from tokio_ai.backtest_cli import main as cli_main  # noqa: E402
from tokio_ai.backtest_cli import read_columns, to_returns  # noqa: E402
from tokio_ai.family import _romano_wolf  # noqa: E402


def _noise(n, seed, sd=0.01):
    return np.random.default_rng(seed).standard_normal(n) * sd


def test_exported_from_the_package():
    assert tokio_ai.check_backtest is check_backtest
    assert tokio_ai.BacktestResult is BacktestResult


def test_real_edge_is_significant_and_zero_edge_is_not():
    r = _noise(2520, 1)
    assert check_backtest(r + 0.001).significant
    assert not check_backtest(r - r.mean()).significant


def test_accepts_lists_and_pandas():
    pd = pytest.importorskip("pandas")
    r = _noise(500, 2) + 0.001
    a = check_backtest(list(r))
    b = check_backtest(pd.Series(r))
    assert a.p_value == pytest.approx(b.p_value)


def test_sharpe_is_annualized():
    r = _noise(1000, 3) + 0.0005
    res = check_backtest(r, periods_per_year=252)
    assert res.best.sharpe == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))


def test_p_alone_matches_between_single_and_family():
    rng = np.random.default_rng(4)
    fam = {f"v{j}": rng.standard_normal(800) * 0.01 + 0.0003 * j for j in range(4)}
    res = check_backtest(fam)
    for s in res.strategies:
        assert s.p_alone == pytest.approx(check_backtest(fam[s.name]).best.p_alone)


def test_family_adjustment_never_below_alone_and_never_above_bonferroni():
    rng = np.random.default_rng(5)
    base = rng.standard_normal(1500) * 0.01
    fam = {f"v{j}": base * rng.choice([-1, 1], 1500) + 0.0002 * j for j in range(10)}
    res = check_backtest(fam)
    for s in res.strategies:
        assert s.p_adjusted >= s.p_alone - 1e-12
        assert s.p_adjusted <= min(1.0, 10 * s.p_alone) + 1e-9


def test_best_of_many_zero_edge_strategies_is_not_significant_after_correction():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(2520) * 0.01
    fam = {f"s{j}": np.where(rng.random(2520) < 0.5, -1.0, 1.0) * r for j in range(100)}
    res = check_backtest(fam)
    assert min(s.p_naive for s in res.strategies) < 0.05  # the search found a "winner"
    assert not res.significant


def test_headline_is_the_most_significant_variant():
    rng = np.random.default_rng(22)
    fam = {f"s{j}": rng.standard_normal(1000) * 0.01 for j in range(5)}
    fam["real"] = rng.standard_normal(1000) * 0.01 + 0.002
    res = check_backtest(fam)
    assert res.best.name == "real"
    assert res.significant


def test_extra_trials_use_sidak():
    r = _noise(2520, 7) + 0.0006
    alone = check_backtest(r)
    res = check_backtest(r, trials=20)
    assert res.correction == "sidak"
    assert res.p_value == pytest.approx(1 - (1 - alone.p_value) ** 20)
    assert res.best.haircut_sharpe < alone.best.sharpe


def test_trials_below_variant_count_raises():
    with pytest.raises(ValueError, match="trials"):
        check_backtest({"a": _noise(100, 1), "b": _noise(100, 2)}, trials=1)


def test_haircut_is_zero_when_nothing_survives():
    r = _noise(500, 8)
    res = check_backtest(r - r.mean() + 1e-5, trials=100)
    assert res.best.haircut_sharpe == 0.0


def test_survives_trials_matches_sidak_boundary():
    r = _noise(2520, 9) + 0.0007
    s = check_backtest(r).best
    n = s.survives_trials(0.05)
    assert n >= 1
    assert 1 - (1 - s.p_alone) ** n <= 0.05 < 1 - (1 - s.p_alone) ** (n + 1)


def test_benchmark_is_subtracted():
    r = _noise(1000, 10)
    res = check_backtest(r + 0.002, benchmark=np.full(1000, 0.002))
    assert res.best.mean == pytest.approx(r.mean())


def test_missing_bars_dropped_jointly():
    a = _noise(300, 11)
    b = _noise(300, 12)
    a[5] = np.nan
    b[7] = np.nan
    res = check_backtest({"a": a, "b": b})
    assert res.best.n_bars == 298
    assert any("2 bars" in n for n in res.notes)


def test_short_sample_not_reportable():
    res = check_backtest(_noise(30, 13) + 0.01)
    assert res.verdict == "NOT REPORTABLE"
    assert "NOT REPORTABLE" in str(res)


def test_flat_pnl_not_reportable():
    assert check_backtest(np.zeros(500)).verdict == "NOT REPORTABLE"


def test_mismatched_pandas_indexes_raise():
    pd = pytest.importorskip("pandas")
    r = pd.Series(_noise(200, 14), index=range(200))
    bench = pd.Series(_noise(200, 15), index=range(1, 201))
    with pytest.raises(ValueError, match="indexes"):
        check_backtest(r, benchmark=bench)


def test_dataframe_columns_are_variants():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(16)
    df = pd.DataFrame({"x": rng.standard_normal(400) * 0.01, "y": rng.standard_normal(400) * 0.01})
    res = check_backtest(df)
    assert {s.name for s in res.strategies} == {"x", "y"}
    assert res.trials == 2


def test_andrews_bandwidth_grows_with_persistence():
    rng = np.random.default_rng(17)
    e = rng.standard_normal(3000)
    ma = np.convolve(e, np.ones(20) / 20, "same")
    assert andrews_bandwidth(np, e - e.mean()) < 5
    assert andrews_bandwidth(np, ma - ma.mean()) > 30


def test_overlapping_trade_returns_get_a_wider_interval():
    # 20-bar trade returns logged daily are a strong moving average; a plain
    # t-test treats them as independent and is far too confident.
    rng = np.random.default_rng(18)
    e = rng.standard_normal(2540) * 0.01
    ma = np.convolve(e, np.ones(20), "valid")
    ma = ma - ma.mean() + 0.0005
    res = check_backtest(ma)
    assert res.best.p_alone > res.best.p_naive


def test_positions_widen_the_window_to_the_holding_run():
    rng = np.random.default_rng(19)
    r = rng.standard_normal(2000) * 0.01
    pos = np.repeat(np.where(rng.random(20) < 0.5, -1.0, 1.0), 100)
    a = check_backtest(pos * r)
    b = check_backtest(pos * r, positions=pos)
    assert b.bandwidth >= 100 > a.bandwidth


def test_positions_for_family_must_be_a_dict():
    with pytest.raises(ValueError, match="dict"):
        check_backtest({"a": _noise(100, 1), "b": _noise(100, 2)}, positions=np.ones(100))


def test_fixed_b_scale():
    assert fixed_b_scale(0.0) == 1.0
    assert 0.8 < fixed_b_scale(0.1) < 0.9
    assert fixed_b_scale(0.3) < fixed_b_scale(0.1)


def test_one_sided_romano_wolf_ignores_large_negative_z():
    p = _romano_wolf(np, np.array([-4.0, 2.0, 0.0]), np.eye(3), 0, one_sided=True)
    assert p[0] > 0.99
    assert p[1] < 0.1


def test_str_mentions_plain_t_test_and_haircut():
    rng = np.random.default_rng(20)
    fam = {f"s{j}": rng.standard_normal(1000) * 0.01 for j in range(5)}
    text = str(check_backtest(fam))
    assert "plain t-test" in text and "Haircut Sharpe after 5 trials" in text
    assert "romano_wolf" in text


def test_cli_reads_csv(tmp_path, capsys):
    rng = np.random.default_rng(21)
    p = tmp_path / "pnl.csv"
    lines = ["date,a,b,spy"] + [
        f"d{i},{rng.normal(0, 1):.4f}%,{rng.normal(0, 0.01):.6f},{rng.normal(0, 0.01):.6f}"
        for i in range(300)
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    assert cli_main([str(p), "--benchmark", "spy", "--trials", "10"]) == 0
    out = capsys.readouterr().out
    assert "after correcting for 10 trials" in out
    assert "in excess of the benchmark" in out
    cols = read_columns(str(p))
    assert set(cols) == {"a", "b", "spy"}
    assert abs(cols["a"][0]) < 0.1  # "0.1234%" -> 0.001234


def test_cli_bad_input_exits_2(tmp_path, capsys):
    p = tmp_path / "x.csv"
    p.write_text("date\n2020-01-01\n", encoding="utf-8")
    assert cli_main([str(p)]) == 2
    assert "no numeric columns" in capsys.readouterr().err


def test_to_returns():
    out = to_returns([100.0, 110.0, 99.0])
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(-0.10)


def test_cli_missing_markers_are_nan_and_stray_text_refuses(tmp_path, capsys):
    rng = np.random.default_rng(23)
    rows = [f"d{i},{rng.normal(0, 0.01):.5f},{rng.normal(0, 0.01):.5f}" for i in range(100)]
    rows[3] = "d3,-,N/A"
    p = tmp_path / "m.csv"
    p.write_text(chr(10).join(["date,a,b"] + rows), encoding="utf-8")
    cols = read_columns(str(p))
    assert set(cols) == {"a", "b"} and math.isnan(cols["a"][3]) and math.isnan(cols["b"][3])
    rows[5] = "d5,oops,0.01"
    p.write_text(chr(10).join(["date,a,b"] + rows), encoding="utf-8")
    assert cli_main([str(p)]) == 2
    assert "column 'a'" in capsys.readouterr().err


def _edge_with_positions(seed=30, n=2520, run=10, edge=0.0015):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * 0.01
    pos = np.repeat(np.where(rng.random(n // run + 1) < 0.5, -1.0, 1.0), run)[:n]
    return pos * r + edge * np.abs(pos), pos


def test_costs_need_positions():
    with pytest.raises(ValueError, match="positions"):
        check_backtest(_noise(500, 1), costs=0.001)
    with pytest.raises(ValueError, match="non-negative"):
        check_backtest(_noise(500, 1), positions=np.ones(500), costs=-1)


def test_costs_are_charged_on_position_changes():
    pnl, pos = _edge_with_positions()
    c = 0.0005
    res = check_backtest(pnl, positions=pos, costs=c)
    turn = np.abs(np.diff(pos, prepend=0.0))
    assert res.best.mean == pytest.approx(pnl.mean() - c * turn.mean())
    assert res.turnover_per_year == pytest.approx(turn.mean() * 252)


def test_breakeven_is_the_significance_boundary():
    pnl, pos = _edge_with_positions()
    res = check_backtest(pnl, positions=pos)
    be = res.breakeven_cost
    assert be is not None and be > 0
    assert check_backtest(pnl, positions=pos, costs=be * 0.98).significant
    assert not check_backtest(pnl, positions=pos, costs=be * 1.05).significant
    assert "Breakeven" in str(res)


def test_costs_that_sink_a_result_are_named():
    pnl, pos = _edge_with_positions()
    be = check_backtest(pnl, positions=pos).breakeven_cost
    res = check_backtest(pnl, positions=pos, costs=be * 3)
    assert not res.significant and res.p_before_costs <= 0.05
    assert "costs are what sink it" in str(res)


def test_no_breakeven_when_not_significant_before_costs():
    rng = np.random.default_rng(31)
    pos = np.where(rng.random(1000) < 0.5, -1.0, 1.0)
    res = check_backtest(pos * rng.standard_normal(1000) * 0.01, positions=pos)
    assert res.breakeven_cost is None


def test_huge_edge_does_not_crash_the_haircut():
    rng = np.random.default_rng(40)
    fam = {"a": rng.standard_normal(5000) * 0.01 + 0.01, "b": rng.standard_normal(5000) * 0.01}
    res = check_backtest(fam)
    assert res.significant and math.isfinite(res.best.haircut_sharpe)


def test_before_costs_line_is_about_the_reported_variant():
    rng = np.random.default_rng(41)
    n = 2520
    r = rng.standard_normal(n) * 0.01
    pos_b = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # flips every bar: expensive
    pos_a = np.ones(n)
    fam = {"a": pos_a * rng.standard_normal(n) * 0.01, "b": pos_b * r + 0.002}
    res = check_backtest(fam, positions={"a": pos_a, "b": pos_b}, costs=0.002)
    gross = check_backtest(fam, positions={"a": pos_a, "b": pos_b})
    assert gross.best.name == "b" and gross.significant
    assert res.best.name == "a"  # costs flipped the strongest variant
    assert res.p_before_costs == pytest.approx(
        next(s.p_adjusted for s in gross.strategies if s.name == "a"))
    assert "costs are what sink it" not in str(res)


def test_partial_positions_still_report_turnover_for_the_reported_variant():
    rng = np.random.default_rng(42)
    pos = np.repeat([1.0, -1.0] * 50, 20)
    fam = {"a": pos * rng.standard_normal(2000) * 0.01 + 0.002, "b": rng.standard_normal(2000) * 0.01}
    res = check_backtest(fam, positions={"a": pos})
    assert res.best.name == "a"
    assert res.turnover_per_year is not None and res.breakeven_cost is None
    assert any("every variant" in n for n in res.notes)


def test_a_variant_that_never_trades_is_explained_not_silently_dropped():
    r = _noise(3000, 11) + 0.0006
    res = check_backtest({"live": r, "never_trades": np.zeros(3000)})
    assert "never traded" in str(res)
