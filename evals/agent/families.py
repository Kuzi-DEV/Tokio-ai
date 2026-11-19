"""Strategy families for the agent test set.

Each family turns (symbol, params) into a natural-language question and a
reference signal grid. The reference verdict is what TokIO's Lab says about
that grid on the cached prices; an agent answering the question should test
the same number of variants and reach the same verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def sma_cross(d, fast, slow):
    f, s = d.close.rolling(fast).mean(), d.close.rolling(slow).mean()
    return (f > s).astype(float)


def above_sma(d, n):
    return (d.close > d.close.rolling(n).mean()).astype(float)


def rsi_revert(d, n, buy, sell):
    r = _rsi(d.close, n)
    pos = pd.Series(np.nan, index=d.index)
    pos[r < buy] = 1.0
    pos[r > sell] = 0.0
    return pos.ffill().fillna(0.0)


def momentum(d, lookback):
    return (d.close.pct_change(lookback) > 0).astype(float)


def donchian(d, n):
    hi = d.high.rolling(n).max().shift(1)
    lo = d.low.rolling(n // 2).min().shift(1)
    pos = pd.Series(np.nan, index=d.index)
    pos[d.close > hi] = 1.0
    pos[d.close < lo] = 0.0
    return pos.ffill().fillna(0.0)


def _next_bday(idx):
    # The calendar is known in advance, so the next business day is not lookahead.
    return idx + pd.offsets.BDay(1)


def weekday(d, day):
    # Positions are decided at the close and filled at the next open, so hold
    # going into `day`: the signal is on when the NEXT business day is `day`.
    return pd.Series((_next_bday(d.index).dayofweek == day).astype(float), index=d.index)


def turn_of_month(d, before, after):
    nxt = _next_bday(d.index)
    days = nxt.values.astype("datetime64[D]")
    month_start = nxt.to_period("M").to_timestamp().values.astype("datetime64[D]")
    month_end = (nxt.to_period("M").to_timestamp(how="end")).values.astype("datetime64[D]")
    from_start = np.busday_count(month_start, days)          # 0 = first business day
    to_end = np.busday_count(days, month_end + 1) - 1        # 0 = last business day
    return pd.Series(((to_end < before) | (from_start < after)).astype(float), index=d.index)


def bollinger(d, n, k):
    m, s = d.close.rolling(n).mean(), d.close.rolling(n).std()
    pos = pd.Series(np.nan, index=d.index)
    pos[d.close < m - k * s] = 1.0
    pos[d.close > m] = 0.0
    return pos.ffill().fillna(0.0)


def buy_hold(d):
    return pd.Series(1.0, index=d.index)


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# family -> (question template, signal, grid builder, vs)
FAMILIES = {
    "sma_cross": dict(
        signal=sma_cross, vs="market",
        q="Does an SMA crossover strategy on {sym} (long when the {fast_list}-day average is above the {slow_list}-day average, flat otherwise) beat holding {sym}? Test every combination.",
    ),
    "above_sma": dict(
        signal=above_sma, vs="market",
        q="Test a trend filter on {sym}: long when the close is above its N-day simple moving average, flat otherwise, for N = {n_list}. Does it beat holding {sym}?",
    ),
    "rsi_revert": dict(
        signal=rsi_revert, vs="market",
        q="Test RSI mean reversion on {sym}: buy when RSI({n}) drops below {buy_list}, sell when it rises above {sell}. Try each buy threshold. Does it beat holding {sym} at the same exposure?",
    ),
    "momentum": dict(
        signal=momentum, vs="market",
        q="Time-series momentum on {sym}: be long when the trailing {lookback_list}-day return is positive, flat otherwise. Test all lookbacks. Is any of them better than holding {sym}?",
    ),
    "donchian": dict(
        signal=donchian, vs="market",
        q="Breakout test on {sym}: buy when the close breaks above the prior N-day high, exit when it falls below the prior N/2-day low, for N = {n_list}. Does it beat holding {sym}?",
    ),
    "weekday": dict(
        signal=weekday, vs="market",
        q="Is there a day-of-week effect in {sym}? Test holding {sym} only on {day_list} (one strategy per day). Does any day beat holding at the same exposure?",
    ),
    "turn_of_month": dict(
        signal=turn_of_month, vs="market",
        q="Turn-of-the-month effect in {sym}: hold only the last {before} trading day(s) of each month and the first {after_list} trading day(s) of the next. Test each window. Does it beat holding {sym} at the same exposure?",
    ),
    "bollinger": dict(
        signal=bollinger, vs="market",
        q="Bollinger band mean reversion on {sym}: buy when the close drops below the {n}-day average minus K standard deviations, exit when it gets back above the average, for K = {k_list}. Does it beat holding {sym}?",
    ),
    "buy_hold": dict(
        signal=buy_hold, vs="cash",
        q="Has simply buying and holding {sym} made money over the available history, in a statistically meaningful way? Test it against cash, not against the market.",
    ),
}
