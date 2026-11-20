"""Strategy families for agent test set B (questions_b.jsonl).

Set A has 9 families. The model writes the same code across a family's
questions, so set A's 100 questions are really ~9 independent units, which is
too few to tell one model from another (v2 vs v3: +12, 95% CI [-16, +38]).
Set B has 25 more families, none of them a set-A family or a training family
(training/agent/families_v2.py), on the same test tickers.

Conventions, same as set A: a signal is decided at the close and filled at the
next open, so "go long the next day after X" means the signal is 1.0 on the
day X happens. Every rule is long/flat unless the question says otherwise.
Frozen 2026-11-20, before any model was scored on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------ trend / momentum
def dual_sma_filter(d, a, b):
    return ((d.close > d.close.rolling(a).mean()) & (d.close > d.close.rolling(b).mean())).astype(float)


def triple_ma(d, a, b, c):
    x, y, z = (d.close.rolling(n).mean() for n in (a, b, c))
    return ((x > y) & (y > z)).astype(float)


def regression_slope(d, n):
    t = np.arange(n, dtype=float)
    t -= t.mean()
    slope = d.close.rolling(n).apply(lambda y: float(np.dot(t, y - y.mean()) / np.dot(t, t)), raw=True)
    return (slope > 0).astype(float)


def trix(d, n):
    e = d.close
    for _ in range(3):
        e = e.ewm(span=n, adjust=False).mean()
    return (e.pct_change() > 0).astype(float)


def adx_trend(d, n, thr):
    up, dn = d.high.diff(), -d.low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift(1)).abs(), (d.low - d.close.shift(1)).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    pdi, mdi = 100 * _wilder(plus, n) / atr, 100 * _wilder(minus, n) / atr
    adx = _wilder(100 * (pdi - mdi).abs() / (pdi + mdi), n)
    return ((adx > thr) & (pdi > mdi)).astype(float)


def channel_mid(d, n):
    mid = (d.high.rolling(n).max().shift(1) + d.low.rolling(n).min().shift(1)) / 2
    return (d.close > mid).astype(float)


def skip_month_momentum(d, n):
    return (d.close.shift(21) / d.close.shift(n) - 1 > 0).astype(float)


def near_high(d, pct):
    return (d.close >= (1 - pct) * d.close.rolling(252).max()).astype(float)


def new_low_avoid(d, n, hold):
    low = d.close < d.close.rolling(n).min().shift(1)
    out = low.astype(float).rolling(hold, min_periods=1).max()
    return 1.0 - out


def half_trend(d, n):
    return pd.Series(np.where(d.close > d.close.rolling(n).mean(), 1.0, 0.5), index=d.index)


def trend_vs_cash(d, n):
    return (d.close > d.close.rolling(n).mean()).astype(float)


# ------------------------------------------------------------ volume
def volume_spike_up(d, k):
    return ((d.volume > k * d.volume.rolling(20).mean().shift(1)) & (d.close > d.close.shift(1))).astype(float)


def volume_dryup(d, k):
    return (d.volume.rolling(5).mean() < k * d.volume.rolling(50).mean()).astype(float)


def vwap_trend(d, n):
    vwap = (d.close * d.volume).rolling(n).sum() / d.volume.rolling(n).sum()
    return (d.close > vwap).astype(float)


def obv_trend(d, n):
    obv = (np.sign(d.close.diff()).fillna(0.0) * d.volume).cumsum()
    return (obv > obv.rolling(n).mean()).astype(float)


def cmf_positive(d, n):
    rng = (d.high - d.low).replace(0, np.nan)
    mfv = (((d.close - d.low) - (d.high - d.close)) / rng).fillna(0.0) * d.volume
    return (mfv.rolling(n).sum() / d.volume.rolling(n).sum() > 0).astype(float)


# ------------------------------------------------------------ one-day patterns (hold the next day)
def inside_day(d):
    return ((d.high < d.high.shift(1)) & (d.low > d.low.shift(1))).astype(float)


def outside_down_day(d):
    return ((d.high > d.high.shift(1)) & (d.low < d.low.shift(1)) & (d.close < d.open)).astype(float)


def range_expansion(d, k):
    r = d.high - d.low
    return (r > k * r.rolling(20).mean().shift(1)).astype(float)


def big_drop(d, x):
    return (d.close.pct_change() < -x).astype(float)


def big_up(d, x):
    return (d.close.pct_change() > x).astype(float)


def higher_lows(d, n):
    hl = (d.low > d.low.shift(1)).astype(int)
    return (hl.rolling(n).sum() == n).astype(float)


def bullish_candle(d):
    return (d.close > d.open).astype(float)


def above_prior_high(d):
    return (d.close > d.high.shift(1)).astype(float)


# ------------------------------------------------------------ calendar
def year_start(d, n):
    # The position is held on the next bar, so ask whether the NEXT trading day is among
    # the first n of its calendar year. The trading calendar is known in advance.
    nxt = d.index[1:].append(pd.DatetimeIndex([d.index[-1] + pd.offsets.BDay(1)]))
    yr = pd.Series(nxt.year, index=d.index)
    rank = yr.groupby(yr.values).cumcount()  # 0 = first trading day of the year
    return (rank < n).astype(float)


Q = "Does it beat holding {sym} at the same exposure?"
FAMILIES = {
    "dual_sma_filter": dict(signal=dual_sma_filter, vs="market",
        q="Trend filter on {sym}: long only when the close is above BOTH its {a}-day and its {b_list}-day simple moving averages, flat otherwise. Test each long average. " + Q),
    "triple_ma": dict(signal=triple_ma, vs="market",
        q="Triple moving-average alignment on {sym}: long when SMA({a}) > SMA({b}) > SMA({c_list}), flat otherwise. Test each slow length. " + Q),
    "regression_slope": dict(signal=regression_slope, vs="market",
        q="On {sym}, be long when the slope of a least-squares line fitted to the last N closes is positive, flat otherwise, for N = {n_list}. " + Q),
    "trix": dict(signal=trix, vs="market",
        q="TRIX on {sym}: take a triple-smoothed EMA of the close (span {n_list}, three EMAs in a row) and be long when it rose on the day, flat otherwise. Test each span. " + Q),
    "adx_trend": dict(signal=adx_trend, vs="market",
        q="ADX trend filter on {sym}: with Wilder's {n}-day ADX, be long when ADX is above {thr_list} and +DI is above -DI, flat otherwise. Test each threshold. " + Q),
    "channel_mid": dict(signal=channel_mid, vs="market",
        q="On {sym}, be long when the close is above the midpoint of the prior N-day high-low channel (prior N days' highest high and lowest low, excluding today), for N = {n_list}. " + Q),
    "skip_month_momentum": dict(signal=skip_month_momentum, vs="market",
        q="Skip-a-month momentum on {sym}: long when the return from N trading days ago to 21 trading days ago is positive, flat otherwise, for N = {n_list}. " + Q),
    "near_high": dict(signal=near_high, vs="market",
        q="Stay-near-the-highs rule on {sym}: be long while the close is within {pct_list} of its 252-day closing high (as a fraction, e.g. 0.05 = 5%), flat otherwise. Test each band. " + Q),
    "new_low_avoid": dict(signal=new_low_avoid, vs="market",
        q="On {sym}, be long by default but go flat for {hold} trading days after any close below the prior {n_list}-day closing low. Test each N. " + Q),
    "half_trend": dict(signal=half_trend, vs="market",
        q="Position sizing by trend on {sym}: hold 100% when the close is above its {n_list}-day SMA and 50% when it's below. Test each N. " + Q),
    "trend_vs_cash": dict(signal=trend_vs_cash, vs="cash",
        q="Did a simple trend rule on {sym} (long when the close is above its {n_list}-day SMA, in cash otherwise) make money in a statistically meaningful way? Test it against cash, not against the market."),
    "volume_spike_up": dict(signal=volume_spike_up, vs="market",
        q="On {sym}, go long for the next day after any up day whose volume is more than {k_list} times the average volume of the prior 20 days. Test each multiple. " + Q),
    "volume_dryup": dict(signal=volume_dryup, vs="market",
        q="Quiet-volume rule on {sym}: long when the 5-day average volume is below {k_list} times the 50-day average volume, flat otherwise. Test each ratio. " + Q),
    "vwap_trend": dict(signal=vwap_trend, vs="market",
        q="On {sym}, be long when the close is above its N-day volume-weighted average price (sum of close times volume over sum of volume), for N = {n_list}. " + Q),
    "obv_trend": dict(signal=obv_trend, vs="market",
        q="On-balance volume on {sym}: be long when OBV is above its own {n_list}-day simple average, flat otherwise. Test each N. " + Q),
    "cmf_positive": dict(signal=cmf_positive, vs="market",
        q="Chaikin money flow on {sym}: be long when the {n_list}-day CMF is above zero, flat otherwise. Test each N. " + Q),
    "inside_day": dict(signal=inside_day, vs="market",
        q="Inside days on {sym}: after a day whose high is below the previous high and whose low is above the previous low, hold {sym} for the next day. " + Q),
    "outside_down_day": dict(signal=outside_down_day, vs="market",
        q="On {sym}, after an outside day that closes below its open (higher high and lower low than the day before), hold for the next day. " + Q),
    "range_expansion": dict(signal=range_expansion, vs="market",
        q="Range expansion on {sym}: after a day whose high-low range is more than {k_list} times the average range of the prior 20 days, hold for the next day. Test each multiple. " + Q),
    "big_drop": dict(signal=big_drop, vs="market",
        q="Does {sym} bounce after a big down day? Hold for the next day after any close-to-close drop larger than {x_list} (as a fraction). Test each size. " + Q),
    "big_up": dict(signal=big_up, vs="market",
        q="Do big up days in {sym} follow through? Hold for the next day after any close-to-close gain larger than {x_list} (as a fraction). Test each size. " + Q),
    "higher_lows": dict(signal=higher_lows, vs="market",
        q="On {sym}, after {n_list} days in a row of higher lows (each day's low above the day before's), hold for the next day. Test each streak length. " + Q),
    "bullish_candle": dict(signal=bullish_candle, vs="market",
        q="On {sym}, after any day that closes above its open, hold for the next day. " + Q),
    "above_prior_high": dict(signal=above_prior_high, vs="market",
        q="On {sym}, after any day that closes above the previous day's high, hold for the next day. " + Q),
    "year_start": dict(signal=year_start, vs="market",
        q="Start-of-year effect on {sym}: hold only during the first N trading days of each calendar year, for N = {n_list}. " + Q),
}
