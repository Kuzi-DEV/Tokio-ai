import math

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import tokio_ai  # noqa: E402
from tokio_ai import Lab, LookaheadError  # noqa: E402


def _bars(n=1500, seed=0, drift=0.0003):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01 + drift)
    gap = rng.standard_normal(n) * 0.003
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + gap)
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                         "low": np.minimum(open_, close) * 0.998, "close": close,
                         "volume": 1e6}, index=idx)


def sma_cross(d, fast=10, slow=50):
    f, s = d.close.rolling(fast).mean(), d.close.rolling(slow).mean()
    return (f > s).astype(float)


def test_exported():
    assert tokio_ai.Lab is Lab


def test_always_long_matches_buy_and_hold_after_one_entry_cost():
    d = _bars()
    lab = Lab(d, holdout=0, commission=0.001, slippage=0.0)
    run = lab.run(lambda x: pd.Series(1.0, index=x.index))
    c = d.close.to_numpy()
    # decided at bar 0's close, filled at bar 1's open: session of bar 1, then close-to-close
    want = (1 - 0.001) * (c[1] / d.open.iloc[1]) * (c[-1] / c[1])
    assert np.prod(1 + run.returns.to_numpy()) == pytest.approx(want, rel=1e-9)
    assert run.trades == 1


def test_fills_at_the_next_open_not_the_signal_close():
    d = _bars()
    lab = Lab(d, holdout=0, commission=0, slippage=0)
    # long only on bar 100's decision: bought at bar 101's open, sold at bar 102's
    # open, so it earns bar 101's session and the overnight gap into bar 102
    sig = lambda x: pd.Series(np.where(np.arange(len(x)) == 100, 1.0, 0.0), index=x.index)
    run = lab.run(sig)
    r = run.returns.to_numpy()
    assert r[101] == pytest.approx(d.close.iloc[101] / d.open.iloc[101] - 1)
    assert r[102] == pytest.approx(d.open.iloc[102] / d.close.iloc[101] - 1)
    assert np.count_nonzero(r) == 2


def test_short_and_fractional_positions():
    d = _bars()
    lab = Lab(d, holdout=0, commission=0, slippage=0)
    run = lab.run(lambda x: pd.Series(-0.5, index=x.index))
    # bar 2: gap and session both at -0.5
    g = d.open.iloc[2] / d.close.iloc[1] - 1
    s = d.close.iloc[2] / d.open.iloc[2] - 1
    assert run.returns.iloc[2] == pytest.approx((1 - 0.5 * g) * (1 - 0.5 * s) - 1)


@pytest.mark.parametrize("cheat", [
    lambda d: (d.close.shift(-1) > d.close).astype(float),                       # tomorrow's close
    lambda d: (d.close > d.close.rolling(21, center=True).mean()).astype(float),  # centered window
    lambda d: ((d.close - d.close.mean()) / d.close.std() > 0).astype(float),    # full-sample z-score
])
def test_lookahead_is_caught(cheat):
    lab = Lab(_bars(), holdout=0)
    with pytest.raises(LookaheadError, match="uses future data"):
        lab.run(cheat)
    assert lab.runs == []  # a rejected run isn't counted or kept


def test_honest_signal_passes_the_lookahead_check():
    lab = Lab(_bars(), holdout=0)
    run = lab.run(sma_cross, fast=10, slow=50)
    assert run.trades > 5


def test_holdout_is_invisible_until_final_test():
    d = _bars(2000)
    seen = []

    def spy(x, k=1):
        seen.append(len(x))
        return (x.close > x.close.shift(k)).astype(float)

    lab = Lab(d, holdout=0.25)
    lab.run(spy, k=1)
    assert max(seen) == lab.split == 1500
    final = lab.final_test(spy, k=1)
    assert max(seen) == 2000
    assert len(final.returns) == 500 and final.segment == "holdout"
    with pytest.raises(RuntimeError, match="already been used"):
        lab.final_test(spy, k=1)
    again = lab.final_test(spy, k=1, force=True)
    assert "not an out-of-sample test" in str(again)


def test_every_variant_is_counted_once():
    lab = Lab(_bars(), holdout=0.2)
    runs = lab.sweep(sma_cross, fast=[5, 10], slow=[50, 100, 150])
    assert len(runs) == 6 and len(lab.runs) == 6
    lab.run(sma_cross, fast=5, slow=50)  # identical variant: not a new trial
    assert len(lab.runs) == 6
    res = lab.check()
    assert res.trials == 6 and len(res.strategies) == 6
    assert "6 variant(s) counted" in str(res)


def test_bad_positions_are_refused():
    lab = Lab(_bars(), holdout=0)
    with pytest.raises(ValueError, match="beyond -1..1"):
        lab.run(lambda x: pd.Series(2.0, index=x.index))
    with pytest.raises(ValueError, match="positions for"):
        lab.run(lambda x: pd.Series([1.0, 0.0]))


def test_dataframe_columns_case_insensitive_and_ppy_inferred():
    d = _bars().rename(columns=str.title)
    lab = Lab(d, holdout=0)
    assert 240 < lab.periods_per_year < 270


def test_vs_market_removes_drift():
    d = _bars(3000, seed=3, drift=0.0008)  # a market that rises
    always = lambda x: pd.Series(1.0, index=x.index)
    cash = Lab(d, holdout=0, vs="cash").run(always)
    mkt = Lab(d, holdout=0, vs="market").run(always)
    assert cash.report.significant  # holding a rising market "passes" against zero
    assert not mkt.report.significant  # but it is exactly the market: no timing
    assert "Tested against the market" in str(mkt.report)
    with pytest.raises(ValueError, match="vs must be"):
        Lab(d, vs="spy")


def test_costs_note_says_they_are_already_charged():
    run = Lab(_bars(), holdout=0).run(sma_cross)
    assert "already charged in these returns" in str(run.report)


def test_different_signals_with_the_same_name_are_different_variants():
    lab = Lab(_bars(), holdout=0)
    a = lab.run(lambda d: (d.close > d.close.rolling(10).mean()).astype(float))
    b = lab.run(lambda d: (d.close < d.close.rolling(10).mean()).astype(float))
    assert a is not b and len(lab.runs) == 2
    assert a.name == "lambda" and b.name == "lambda#2"
    assert len(lab.check().strategies) == 2
    again = lab.run(a and lab._signals[0][0])  # the same function object: not a new trial
    assert again is a and len(lab.runs) == 2


def test_lab_never_blames_lookahead_it_has_ruled_out():
    d = _bars(3000, seed=9)
    # a real but one-bar edge: plant next-session returns on yesterday's up days
    r = np.random.default_rng(1).standard_normal(len(d)) * 0.01
    sess = d.close / d.open - 1
    up = (d.close > d.open).astype(float)
    d2 = d.copy()
    d2["close"] = d2["open"] * (1 + sess + 0.004 * up.shift(1).fillna(0) * np.sign(r + 2))
    run = Lab(d2, holdout=0, commission=0, slippage=0).run(lambda x: (x.close > x.open).astype(float))
    text = str(run.report)
    assert "check for lookahead" not in text
    assert run.report.robustness.lookahead_tested


def test_vs_market_has_no_raw_rebuilt_line():
    run = Lab(_bars(), holdout=0, vs="market").run(sma_cross)
    assert run.report.robustness.sharpe_rebuilt is None


def test_best_and_final_test_of_a_run():
    lab = Lab(_bars(2000), holdout=0.25)
    lab.sweep(sma_cross, fast=[5, 10], slow=[50, 100])
    b = lab.best()
    assert b in lab.runs and b.signal is sma_cross
    final = lab.final_test(b)
    assert final.params == b.params and final.segment == "holdout"
