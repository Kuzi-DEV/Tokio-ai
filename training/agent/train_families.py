"""Strategy families for TRAINING data only.

None of these appear in evals/agent/families.py, and none of the tickers in
TRAIN_SYMBOLS / DEV_SYMBOLS appear in the test set, so a model trained on this
data is scored on strategies and tickers it has never seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEST_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GLD", "TLT", "EEM", "XLE", "BTC-USD", "NVDA"}
# Index proxies of test tickers are left out too (^GSPC ~ SPY, ^NDX ~ QQQ, ...).
TRAIN_SYMBOLS = ["AMZN", "GOOGL", "META", "JPM", "XOM", "PG", "JNJ", "WMT", "XLF", "XLK",
                 "XLV", "SLV", "USO", "HYG", "LQD", "EFA", "SMH", "INTC", "CAT", "DBC"]
DEV_SYMBOLS = ["KO", "XLU", "ETH-USD", "VNQ"]


def ema_cross(d, fast, slow):
    f = d.close.ewm(span=fast, adjust=False).mean()
    s = d.close.ewm(span=slow, adjust=False).mean()
    return (f > s).astype(float)


def macd(d, signal_span, fast=12, slow=26):
    line = d.close.ewm(span=fast, adjust=False).mean() - d.close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal_span, adjust=False).mean()
    return (line > sig).astype(float)


def low_vol(d, n, lookback=252):
    vol = d.close.pct_change().rolling(n).std()
    return (vol < vol.rolling(lookback).median()).astype(float)


def down_streak(d, k):
    down = (d.close.diff() < 0).astype(int)
    streak = down.rolling(k).sum()
    return (streak == k).astype(float)  # hold for one day after k down closes in a row


def near_high(d, pct, n=252):
    return (d.close >= (1 - pct) * d.high.rolling(n).max()).astype(float)


def stochastic(d, buy, n=14, sell=80):
    lo, hi = d.low.rolling(n).min(), d.high.rolling(n).max()
    k = 100 * (d.close - lo) / (hi - lo)
    pos = pd.Series(np.nan, index=d.index)
    pos[k < buy] = 1.0
    pos[k > sell] = 0.0
    return pos.ffill().fillna(0.0)


def seasonal(d, start, end=4):
    # Decided at the close, filled at the next open: use the next business day's month.
    m = (d.index + pd.offsets.BDay(1)).month
    on = (m >= start) | (m <= end)
    return pd.Series(on.astype(float), index=d.index)


def dip_buy(d, dd, n=252):
    return (d.close <= (1 - dd) * d.close.rolling(n, min_periods=20).max()).astype(float)


def long_short_trend(d, n):
    sma = d.close.rolling(n).mean()
    return np.sign(d.close - sma).fillna(0.0)


# Each family: signal, grids to draw from, explicit question templates, and one vague
# question the agent has to interpret (rule 5: pick a standard version and say so).
FAMILIES = {
    "ema_cross": dict(
        signal=ema_cross, grids=[{"fast": [10, 20], "slow": [50, 100]}, {"fast": [12], "slow": [26, 50, 100]}, {"fast": [20, 50], "slow": [150, 200]}],
        q=["Does an EMA crossover on {sym} work? Long when the {fast}-day EMA is above the {slow}-day EMA, flat otherwise. Try every pair.",
           "Backtest exponential moving average crossovers on {sym}: fast span {fast}, slow span {slow}, all combinations. Better than buy and hold?"],
        vague="Do moving average crossovers work on {sym}?", vague_grid={"fast": [20, 50], "slow": [100, 200]},
        vague_note="I'll use exponential moving averages, fast spans 20 and 50 against slow spans 100 and 200 (4 variants)."),
    "macd": dict(
        signal=macd, grids=[{"signal_span": [5, 9, 12]}, {"signal_span": [9]}, {"signal_span": [6, 9]}],
        q=["Test MACD(12,26) on {sym}: long when the MACD line is above its signal line, for signal spans {signal_span}. Does it beat holding?",
           "Is the MACD crossover worth anything on {sym}? Use 12/26 with a signal line of {signal_span} days."],
        vague="Is MACD any good on {sym}?", vague_grid={"signal_span": [9]},
        vague_note="I'll test the standard MACD(12, 26, 9): long when the MACD line is above the signal line, flat otherwise."),
    "low_vol": dict(
        signal=low_vol, grids=[{"n": [10, 20]}, {"n": [20]}, {"n": [10, 20, 60]}],
        q=["Volatility filter on {sym}: hold only when {n}-day realized volatility is below its 1-year median. Does that beat holding at the same exposure?",
           "Does sitting out high-volatility periods help on {sym}? Long when the {n}-day return volatility is under its rolling 252-day median, flat otherwise."],
        vague="Should I avoid {sym} when it's volatile?", vague_grid={"n": [20]},
        vague_note="I'll test a standard version: hold when 20-day realized volatility is below its 1-year median, flat otherwise."),
    "down_streak": dict(
        signal=down_streak, grids=[{"k": [2, 3, 4]}, {"k": [3]}, {"k": [2, 3]}],
        q=["Buy {sym} after {k} down days in a row and hold one day. Test each streak length. Is there a bounce?",
           "Short-term reversal on {sym}: after {k} consecutive lower closes, go long for the next day. Does it beat holding at the same exposure?"],
        vague="Does {sym} bounce after a losing streak?", vague_grid={"k": [2, 3, 4]},
        vague_note="I'll go long for one day after 2, 3 or 4 consecutive down closes (3 variants)."),
    "near_high": dict(
        signal=near_high, grids=[{"pct": [0.02, 0.05, 0.10]}, {"pct": [0.05]}, {"pct": [0.03, 0.10]}],
        q=["52-week-high strategy on {sym}: hold only when the close is within {pct} of the 252-day high. Try each threshold. Better than holding?",
           "Test buying strength in {sym}: long when price is within {pct} (as a fraction) of its one-year high, flat otherwise."],
        vague="Is it a good idea to only own {sym} near its highs?", vague_grid={"pct": [0.05, 0.10]},
        vague_note="I'll hold only when the close is within 5% or 10% of its 252-day high (2 variants)."),
    "stochastic": dict(
        signal=stochastic, grids=[{"buy": [10, 20, 30]}, {"buy": [20]}, {"buy": [10, 20]}],
        q=["Stochastic oscillator on {sym}: buy when %K(14) drops below {buy}, sell when it goes above 80. Test each buy level.",
           "Mean reversion with the 14-day stochastic on {sym}: enter under {buy}, exit over 80. Does it beat holding at the same exposure?"],
        vague="Does buying {sym} when the stochastic is oversold work?", vague_grid={"buy": [20]},
        vague_note="I'll use the standard 14-day %K: buy below 20, exit above 80."),
    "seasonal": dict(
        signal=seasonal, grids=[{"start": [10, 11]}, {"start": [11]}, {"start": [9, 10, 11]}],
        q=["\"Sell in May\" on {sym}: hold only from month {start} through April, cash otherwise. Test each start month. Any edge over holding at the same exposure?",
           "Seasonality test on {sym}: invested from month {start} to April each year, flat in the summer. Does it beat holding?"],
        vague="Does 'sell in May and go away' work for {sym}?", vague_grid={"start": [11]},
        vague_note="I'll test the classic version: hold November through April, flat May through October."),
    "dip_buy": dict(
        signal=dip_buy, grids=[{"dd": [0.10, 0.20]}, {"dd": [0.15]}, {"dd": [0.05, 0.10, 0.20]}],
        q=["Buy the dip on {sym}: hold only while it is at least {dd} (as a fraction) below its 1-year high. Test each depth.",
           "Does owning {sym} only during drawdowns of {dd} or more from the 252-day high beat holding at the same exposure?"],
        vague="Is buying {sym} on big dips a good strategy?", vague_grid={"dd": [0.10, 0.20]},
        vague_note="I'll hold only while it's 10% or 20% below its 252-day high (2 variants)."),
    "long_short_trend": dict(
        signal=long_short_trend, grids=[{"n": [50, 100, 200]}, {"n": [200]}, {"n": [100, 200]}],
        q=["Long/short trend on {sym}: long above the {n}-day SMA, short below it. Test each length.",
           "Test going long {sym} above its {n}-day average and short below. Does the timing add anything over holding?"],
        vague="Would shorting {sym} in downtrends have helped?", vague_grid={"n": [200]},
        vague_note="I'll test long above the 200-day SMA and short below it, a common definition of the trend."),
}
