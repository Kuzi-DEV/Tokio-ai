import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from tokio_ai import check_backtest  # noqa: E402
from tokio_ai.adapters import from_backtesting, is_backtest_object, unpack  # noqa: E402


class FakePortfolio:
    """Shaped like a vectorbt Portfolio: returns(), asset_value(), value()."""

    def __init__(self, returns, exposure):
        self._r = returns
        self._e = exposure

    def returns(self):
        return self._r

    def value(self):
        return (1 + self._r).cumprod() * 100

    def asset_value(self):
        return self.value() * self._e


FakePortfolio.__module__ = "vectorbt.portfolio.base"


def _rets(n, seed, edge=0.0):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(np.random.default_rng(seed).standard_normal(n) * 0.01 + edge, index=idx)


def _stats(equity, trades):
    """Shaped like backtesting.py's Backtest.run() result."""
    curve = pd.DataFrame({"Equity": equity})
    return pd.Series({"Sharpe Ratio": 1.0, "_equity_curve": curve, "_trades": pd.DataFrame(trades)}, dtype=object)


def test_plain_inputs_are_not_mistaken_for_backtests():
    assert not is_backtest_object([0.01, 0.02])
    assert not is_backtest_object(np.zeros(5))
    assert not is_backtest_object(pd.Series([0.01, 0.02]))
    assert not is_backtest_object({"a": [0.01]})
    assert not is_backtest_object({})


def test_vectorbt_single_column_matches_passing_returns_and_lagged_exposure():
    r = _rets(800, 1, edge=0.0008)
    expo = pd.Series(np.where(np.arange(800) // 40 % 2 == 0, 1.0, 0.0), index=r.index)
    pf = FakePortfolio(r, expo)
    assert is_backtest_object(pf)
    got = check_backtest(pf)
    want = check_backtest(r, positions=expo.shift(1).fillna(0.0))
    assert got.p_value == pytest.approx(want.p_value)


def test_vectorbt_grid_becomes_variants_with_tuple_names():
    cols = pd.MultiIndex.from_tuples([(10, 50), (20, 100), (5, 30)], names=["fast", "slow"])
    r = pd.DataFrame({c: _rets(600, i) for i, c in enumerate(cols)})
    r.columns = cols
    expo = pd.DataFrame(1.0, index=r.index, columns=cols)
    res = check_backtest(FakePortfolio(r, expo))
    assert sorted(s.name for s in res.strategies) == ["10,50", "20,100", "5,30"]
    assert res.trials == 3


def test_backtesting_stats_rebuilds_positions_from_trades():
    eq = pd.Series(100 * np.cumprod(1 + np.random.default_rng(3).standard_normal(300) * 0.01))
    trades = {"Size": [10, -5], "EntryPrice": [10.0, 20.0], "EntryBar": [5, 100], "ExitBar": [49, 199]}
    rets, pos, _ = from_backtesting(_stats(eq, trades))
    assert rets.iloc[1:].to_numpy() == pytest.approx(eq.pct_change().iloc[1:].to_numpy())
    assert pos.iloc[4] == 0 and pos.iloc[50] == 0 and pos.iloc[250] == 0
    assert pos.iloc[5] == pytest.approx(10 * 10.0 / eq.iloc[5])
    assert pos.iloc[49] == pos.iloc[5]
    assert pos.iloc[100] == pytest.approx(-5 * 20.0 / eq.iloc[100])


def test_backtesting_stats_with_no_trades_is_flat():
    eq = pd.Series(np.full(100, 100.0))
    _, pos, _ = from_backtesting(_stats(eq, {"Size": [], "EntryPrice": [], "EntryBar": [], "ExitBar": []}))
    assert (pos == 0).all()


def test_dict_of_backtests_and_explicit_positions_win():
    a, b = _rets(400, 4), _rets(400, 5)
    ones = pd.Series(1.0, index=a.index)
    rets, pos, _ = unpack({"a": FakePortfolio(a, ones), "b": FakePortfolio(b, ones)})
    assert set(rets) == {"a", "b"} and set(pos) == {"a", "b"}
    mine = {"a": ones * 0.5, "b": ones * 0.5}
    res = check_backtest({"a": FakePortfolio(a, ones), "b": FakePortfolio(b, ones)}, positions=mine, costs=0.001)
    assert res.trials == 2


def test_real_vectorbt_if_installed():
    vbt = pytest.importorskip("vectorbt")
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(6).standard_normal(1000) * 0.01),
                      index=pd.date_range("2018-01-01", periods=1000, freq="D"))
    # every pair from 4 windows: 6 crossover variants in one portfolio
    fast, slow = vbt.MA.run_combs(close, window=[5, 10, 20, 50], r=2, short_names=["fast", "slow"])
    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)
    pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.0005, freq="1D")
    res = check_backtest(pf)
    assert res.trials == 6


def test_real_backtesting_py_if_installed():
    bt = pytest.importorskip("backtesting")
    from backtesting.test import GOOG, SMA

    class Cross(bt.Strategy):
        def init(self):
            self.f = self.I(SMA, self.data.Close, 10)
            self.s = self.I(SMA, self.data.Close, 30)

        def next(self):
            if self.f[-1] > self.s[-1] and not self.position:
                self.buy()
            elif self.f[-1] < self.s[-1] and self.position:
                self.position.close()

    stats = bt.Backtest(GOOG, Cross, commission=0.001).run()
    rets, pos, _ = from_backtesting(stats)
    assert len(rets) == len(pos) == len(GOOG)
    assert pos.max() > 0.5
    open_trades = stats["_strategy"].trades
    if open_trades:  # held to the end, not dropped
        assert pos.iloc[-1] > 0.5
    res = check_backtest(stats)
    assert res.strategies[0].n_bars > 1000


def test_mostly_long_significant_result_warns_about_drift():
    r = _rets(2520, 7, edge=0.0)
    drift = r + 0.0015                       # a rising market
    # long nearly always, flat one bar in 50: pure beta, no timing
    pos = pd.Series(np.where(np.arange(len(r)) % 50 == 0, 0.0, 1.0), index=r.index)
    res = check_backtest(drift, positions=pos)
    assert res.significant
    assert "drift" in str(res) and "benchmark=" in str(res)
    quiet = check_backtest(drift, positions=pos, benchmark=drift)
    assert "market's own drift" not in str(quiet)
    hedged = check_backtest(drift, positions=pos * 0.1)
    assert "market's own drift" not in str(hedged)


def test_flattened_grid_names_cannot_collide_silently():
    cols = pd.Index(["x", "y"])
    r = pd.DataFrame({c: _rets(300, i) for i, c in enumerate(cols)})
    grid = FakePortfolio(r, pd.DataFrame(1.0, index=r.index, columns=cols))
    single = FakePortfolio(_rets(300, 9), pd.Series(1.0, index=r.index))
    with pytest.raises(ValueError, match="both named"):
        check_backtest({"a/x": single, "a": grid})
