"""Training strategy families, v2: 30 families, several implementations each.

v1 had 9 families with one fixed implementation, and the model memorized them
(it answered a weekday question with the training `seasonal` code). v2 gives
the model many more ways a rule is described and written:

- each family has 1-3 implementations in different styles (pandas ops,
  np.where, explicit state, helper functions), as source text with `{fn}` for
  the function name, which is drawn per trajectory;
- several phrasings per question, plus a vague version where it makes sense.

None of the test families (evals/agent/families.py) is reused by name or rule.
Test tickers stay out (train_families.TEST_SYMBOLS).
"""

from __future__ import annotations

# family -> dict(impls=[source...], grids=[grid...], q=[template...], vague=(question, grid, note) | None,
#                vs="market"|"cash", cash_ok=bool)
F = {}


def fam(name, impls, grids, q, vague=None, vs="market", cash_ok=True):
    F[name] = dict(impls=impls, grids=grids, q=q, vague=vague, vs=vs, cash_ok=cash_ok)


# ---------------------------------------------------------------- trend
fam("ema_cross", [
    '''def {fn}(d, fast, slow):
    f = d.close.ewm(span=fast, adjust=False).mean()
    s = d.close.ewm(span=slow, adjust=False).mean()
    return (f > s).astype(float)''',
    '''def {fn}(d, fast, slow):
    ema = lambda n: d.close.ewm(span=n, adjust=False).mean()
    return pd.Series(np.where(ema(fast) > ema(slow), 1.0, 0.0), index=d.index)'''],
    [{"fast": [10, 20], "slow": [50, 100]}, {"fast": [12], "slow": [26, 50]}, {"fast": [20, 50], "slow": [150, 200]}],
    ["Does an EMA crossover on {sym} work? Long when the {fast}-day EMA is above the {slow}-day EMA, flat otherwise. Try every pair.",
     "Backtest exponential moving average crossovers on {sym}: fast span {fast}, slow span {slow}, all combinations. Better than buy and hold?",
     "EMA({fast}) over EMA({slow}) as a long/flat filter on {sym} - test each combo and tell me if any of it is real."],
    ("Do moving average crossovers work on {sym}?", {"fast": [20, 50], "slow": [100, 200]},
     "I'll use exponential moving averages, fast spans 20 and 50 against slow spans 100 and 200 (4 variants)."))

fam("above_ema", [
    '''def {fn}(d, n):
    return (d.close > d.close.ewm(span=n, adjust=False).mean()).astype(float)''',
    '''def {fn}(d, n):
    trend = d.close.ewm(span=n, adjust=False).mean()
    pos = pd.Series(0.0, index=d.index)
    pos[d.close > trend] = 1.0
    return pos'''],
    [{"n": [50, 100]}, {"n": [21, 63, 126]}, {"n": [200]}],
    ["Hold {sym} only while it closes above its {n}-day exponential moving average. Does that beat holding?",
     "Trend filter test on {sym} using an EMA instead of an SMA: long above EMA({n}), cash below.",
     "Is staying in {sym} only above the {n}-day EMA any better than just owning it at the same exposure?"])

fam("sma_slope", [
    '''def {fn}(d, n, k):
    sma = d.close.rolling(n).mean()
    return (sma > sma.shift(k)).astype(float)''',
    '''def {fn}(d, n, k):
    sma = d.close.rolling(n).mean()
    rising = sma.diff(k) > 0
    return rising.astype(float)'''],
    [{"n": [50, 100], "k": [5, 20]}, {"n": [200], "k": [10]}, {"n": [50], "k": [5, 10, 20]}],
    ["Long {sym} when the {n}-day moving average is higher than it was {k} days ago, flat otherwise. Test every combination.",
     "Moving-average slope filter on {sym}: is the {n}-day SMA rising over the last {k} days? Hold only then.",
     "Does the slope of the {n}-day average (measured over {k} days) time {sym} better than holding it?"])

fam("macd", [
    '''def {fn}(d, signal_span, fast=12, slow=26):
    line = d.close.ewm(span=fast, adjust=False).mean() - d.close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal_span, adjust=False).mean()
    return (line > sig).astype(float)''',
    '''def {fn}(d, signal_span, fast=12, slow=26):
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    macd = ema(d.close, fast) - ema(d.close, slow)
    hist = macd - ema(macd, signal_span)
    return (hist > 0).astype(float)'''],
    [{"signal_span": [5, 9, 12]}, {"signal_span": [9]}, {"signal_span": [6, 9]}],
    ["Test MACD(12,26) on {sym}: long when the MACD line is above its signal line, for signal spans {signal_span}. Does it beat holding?",
     "Is the MACD crossover worth anything on {sym}? Use 12/26 with a signal line of {signal_span} days.",
     "MACD histogram above zero = long, otherwise flat, on {sym}. Signal span {signal_span}. Real or noise?"],
    ("Is MACD any good on {sym}?", {"signal_span": [9]},
     "I'll test the standard MACD(12, 26, 9): long when the MACD line is above the signal line, flat otherwise."))

fam("band_trend", [
    '''def {fn}(d, n, band):
    sma = d.close.rolling(n).mean()
    pos = pd.Series(np.nan, index=d.index)
    pos[d.close > sma * (1 + band)] = 1.0
    pos[d.close < sma * (1 - band)] = 0.0
    return pos.ffill().fillna(0.0)'''],
    [{"n": [100, 200], "band": [0.02, 0.05]}, {"n": [200], "band": [0.03]}, {"n": [50], "band": [0.01, 0.03]}],
    ["Trend with a buffer on {sym}: buy when the close is {band} (fraction) above the {n}-day SMA, sell when it is that far below. Test each combination.",
     "To cut whipsaws, only switch {sym} in or out when it's more than {band} away from its {n}-day average. Does it beat holding?"])

fam("long_short_trend", [
    '''def {fn}(d, n):
    sma = d.close.rolling(n).mean()
    return np.sign(d.close - sma).fillna(0.0)''',
    '''def {fn}(d, n):
    above = d.close > d.close.rolling(n).mean()
    return pd.Series(np.where(above, 1.0, -1.0), index=d.index).where(d.close.rolling(n).mean().notna(), 0.0)'''],
    [{"n": [50, 100, 200]}, {"n": [200]}, {"n": [100, 200]}],
    ["Long/short trend on {sym}: long above the {n}-day SMA, short below it. Test each length.",
     "Test going long {sym} above its {n}-day average and short below. Does the timing add anything over holding?",
     "Flip between long and short {sym} on the {n}-day moving average. Any edge?"],
    ("Would shorting {sym} in downtrends have helped?", {"n": [200]},
     "I'll test long above the 200-day SMA and short below it, a common definition of the trend."), cash_ok=False)

fam("roc_threshold", [
    '''def {fn}(d, n, thr):
    return (d.close / d.close.shift(n) - 1 > thr).astype(float)''',
    '''def {fn}(d, n, thr):
    roc = d.close.pct_change(n)
    return pd.Series(np.where(roc > thr, 1.0, 0.0), index=d.index)'''],
    [{"n": [20, 60], "thr": [0.03, 0.05]}, {"n": [120], "thr": [0.05, 0.10]}, {"n": [10], "thr": [0.02]}],
    ["Rate-of-change filter on {sym}: long only when the {n}-day return is above {thr}. Test each combination.",
     "Hold {sym} only after a strong run: {n}-day gain greater than {thr} (as a fraction). Better than holding?"])

fam("keltner_breakout", [
    '''def {fn}(d, n, k):
    mid = d.close.ewm(span=n, adjust=False).mean()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    pos = pd.Series(np.nan, index=d.index)
    pos[d.close > mid + k * atr] = 1.0
    pos[d.close < mid] = 0.0
    return pos.ffill().fillna(0.0)''',
    '''def atr(d, n):
    prev = d.close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - prev).abs(), (d.low - prev).abs()))
    return tr.rolling(n).mean()

def {fn}(d, n, k):
    mid = d.close.ewm(span=n, adjust=False).mean()
    upper = mid + k * atr(d, n)
    pos = pd.Series(np.nan, index=d.index)
    pos[d.close > upper] = 1.0
    pos[d.close < mid] = 0.0
    return pos.ffill().fillna(0.0)'''],
    [{"n": [20], "k": [1.5, 2.0]}, {"n": [20, 50], "k": [1.0]}, {"n": [20], "k": [2.0]}],
    ["Keltner channel breakout on {sym}: buy when the close goes above the {n}-day EMA plus {k} ATRs, exit back below the EMA. Test each setting.",
     "ATR channel test on {sym}: entry above EMA({n}) + {k}*ATR({n}), exit under the EMA. Does it beat holding at the same exposure?"])

fam("new_high_hold", [
    '''def {fn}(d, n, hold):
    trigger = d.close >= d.close.rolling(n).max()
    return trigger.astype(float).rolling(hold, min_periods=1).max().fillna(0.0)''',
    '''def {fn}(d, n, hold):
    hit = (d.close == d.close.rolling(n).max())
    pos = pd.Series(0.0, index=d.index)
    for lag in range(hold):
        pos = np.maximum(pos, hit.shift(lag, fill_value=False).astype(float))
    return pos''',
    '''def {fn}(d, n, hold):
    # a new n-day high = above the prior n-1 closes; shift(1) keeps today out of its own window
    prior_high = d.close.rolling(n - 1).max().shift(1)
    trigger = d.close > prior_high
    return trigger.astype(float).rolling(hold, min_periods=1).max().fillna(0.0)'''],
    [{"n": [50, 252], "hold": [5, 20]}, {"n": [100], "hold": [10]}, {"n": [252], "hold": [20, 60]}],
    ["When {sym} makes a new {n}-day closing high, buy and hold for {hold} days. Test each combination.",
     "Do new {n}-day highs in {sym} keep going? Long for {hold} trading days after each one."])

fam("aroon", [
    '''def {fn}(d, n, thr):
    up = d.high.rolling(n + 1).apply(lambda x: x.argmax(), raw=True) / n * 100
    dn = d.low.rolling(n + 1).apply(lambda x: x.argmin(), raw=True) / n * 100
    return ((up - dn) > thr).astype(float)'''],
    [{"n": [25], "thr": [0, 50]}, {"n": [14, 25], "thr": [50]}, {"n": [25], "thr": [30]}],
    ["Aroon oscillator on {sym}: long when Aroon({n}) up minus down is above {thr}. Test each setting.",
     "Test an Aroon trend filter on {sym} (period {n}, threshold {thr}). Does it beat holding at the same exposure?"])

# ---------------------------------------------------------------- mean reversion
fam("stochastic", [
    '''def {fn}(d, buy, n=14, sell=80):
    lo, hi = d.low.rolling(n).min(), d.high.rolling(n).max()
    k = 100 * (d.close - lo) / (hi - lo)
    pos = pd.Series(np.nan, index=d.index)
    pos[k < buy] = 1.0
    pos[k > sell] = 0.0
    return pos.ffill().fillna(0.0)''',
    '''def {fn}(d, buy, n=14, sell=80):
    k = (d.close - d.low.rolling(n).min()) / (d.high.rolling(n).max() - d.low.rolling(n).min()) * 100
    state, out = 0.0, []
    for v in k:
        if v < buy:
            state = 1.0
        elif v > sell:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=d.index)'''],
    [{"buy": [10, 20, 30]}, {"buy": [20]}, {"buy": [10, 20]}],
    ["Stochastic oscillator on {sym}: buy when %K(14) drops below {buy}, sell when it goes above 80. Test each buy level.",
     "Mean reversion with the 14-day stochastic on {sym}: enter under {buy}, exit over 80. Does it beat holding at the same exposure?"],
    ("Does buying {sym} when the stochastic is oversold work?", {"buy": [20]},
     "I'll use the standard 14-day %K: buy below 20, exit above 80."))

fam("williams_r", [
    '''def {fn}(d, n, level):
    hh, ll = d.high.rolling(n).max(), d.low.rolling(n).min()
    wr = -100 * (hh - d.close) / (hh - ll)
    pos = pd.Series(np.nan, index=d.index)
    pos[wr < level] = 1.0
    pos[wr > -50] = 0.0
    return pos.ffill().fillna(0.0)'''],
    [{"n": [10], "level": [-90, -80]}, {"n": [14], "level": [-80]}, {"n": [5, 10], "level": [-90]}],
    ["Williams %R({n}) on {sym}: buy when it drops below {level}, exit when it rises above -50. Test each setting.",
     "Test Williams %R mean reversion on {sym}, lookback {n}, entry under {level}, exit above -50."])

fam("cci", [
    '''def {fn}(d, n, entry):
    tp = (d.high + d.low + d.close) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - ma) / (0.015 * md)
    pos = pd.Series(np.nan, index=d.index)
    pos[cci < -entry] = 1.0
    pos[cci > 0] = 0.0
    return pos.ffill().fillna(0.0)'''],
    [{"n": [20], "entry": [100, 150]}, {"n": [14, 20], "entry": [100]}, {"n": [20], "entry": [200]}],
    ["CCI({n}) mean reversion on {sym}: buy when CCI falls below -{entry}, exit when it crosses back above 0. Test each setting.",
     "Commodity Channel Index on {sym}: long under -{entry}, out above zero. Is there anything there?"])

fam("down_streak", [
    '''def {fn}(d, k):
    down = (d.close.diff() < 0).astype(int)
    return (down.rolling(k).sum() == k).astype(float)''',
    '''def {fn}(d, k):
    down = d.close < d.close.shift(1)
    streak = down.groupby((~down).cumsum()).cumsum()
    return (streak >= k).astype(float)'''],
    [{"k": [2, 3, 4]}, {"k": [3]}, {"k": [2, 3]}],
    ["Buy {sym} after {k} down days in a row and hold one day. Test each streak length. Is there a bounce?",
     "Short-term reversal on {sym}: after {k} consecutive lower closes, go long for the next day. Does it beat holding at the same exposure?"],
    ("Does {sym} bounce after a losing streak?", {"k": [2, 3, 4]},
     "I'll go long for one day after 2, 3 or 4 consecutive down closes (3 variants)."))

fam("gap_down", [
    '''def {fn}(d, thr):
    gap = d.open / d.close.shift(1) - 1
    return (gap < -thr).astype(float)''',
    '''def {fn}(d, thr):
    gapped = d.open < d.close.shift(1) * (1 - thr)
    return pd.Series(np.where(gapped, 1.0, 0.0), index=d.index)'''],
    [{"thr": [0.01, 0.02]}, {"thr": [0.005, 0.01, 0.02]}, {"thr": [0.03]}],
    ["After {sym} gaps down more than {thr} (fraction) at the open, hold it the next day. Test each gap size.",
     "Do big down-gaps in {sym} get bought? A day that opened at least {thr} below the prior close triggers one day long."])

fam("ibs", [
    '''def {fn}(d, thr):
    ibs = (d.close - d.low) / (d.high - d.low)
    return (ibs < thr).astype(float)''',
    '''def {fn}(d, thr):
    rng = (d.high - d.low).replace(0, np.nan)
    return ((d.close - d.low) / rng < thr).astype(float)'''],
    [{"thr": [0.1, 0.2, 0.3]}, {"thr": [0.2]}, {"thr": [0.15, 0.25]}],
    ["Internal bar strength on {sym}: if the close is in the bottom {thr} of the day's range, go long for the next day. Test each threshold.",
     "IBS strategy on {sym}: long the day after (close - low) / (high - low) is below {thr}. Any edge?"],
    ("Does buying {sym} after weak closes work?", {"thr": [0.2]},
     "I'll measure weakness with internal bar strength: long for a day after the close lands in the bottom 20% of the day's range."))

fam("dip_buy", [
    '''def {fn}(d, dd, n=252):
    return (d.close <= (1 - dd) * d.close.rolling(n, min_periods=20).max()).astype(float)''',
    '''def {fn}(d, dd, n=252):
    peak = d.close.rolling(n, min_periods=20).max()
    drawdown = d.close / peak - 1
    return (drawdown <= -dd).astype(float)'''],
    [{"dd": [0.10, 0.20]}, {"dd": [0.15]}, {"dd": [0.05, 0.10, 0.20]}],
    ["Buy the dip on {sym}: hold only while it is at least {dd} (as a fraction) below its 1-year high. Test each depth.",
     "Does owning {sym} only during drawdowns of {dd} or more from the 252-day high beat holding at the same exposure?"],
    ("Is buying {sym} on big dips a good strategy?", {"dd": [0.10, 0.20]},
     "I'll hold only while it's 10% or 20% below its 252-day high (2 variants)."))

fam("crash_buy", [  # deep thresholds: often never triggers, which the Lab reports as NOT REPORTABLE
    '''def {fn}(d, dd):
    return (d.close <= (1 - dd) * d.close.cummax()).astype(float)''',
    '''def {fn}(d, dd):
    drawdown = d.close / d.close.cummax() - 1
    return (drawdown <= -dd).astype(float)'''],
    [{"dd": [0.5, 0.6]}, {"dd": [0.7]}, {"dd": [0.8]}, {"dd": [0.6, 0.8]}],
    ["Only buy {sym} after a crash: hold while it's at least {dd} below its all-time high in this data. Test each depth.",
     "Crash-buying test on {sym}: long whenever the drawdown from the running peak is {dd} or worse."])

# ---------------------------------------------------------------- volatility
fam("low_vol", [
    '''def {fn}(d, n, lookback=252):
    vol = d.close.pct_change().rolling(n).std()
    return (vol < vol.rolling(lookback).median()).astype(float)''',
    '''def {fn}(d, n, lookback=252):
    rv = np.log(d.close).diff().rolling(n).std()
    calm = rv < rv.rolling(lookback).median()
    return calm.astype(float)'''],
    [{"n": [10, 20]}, {"n": [20]}, {"n": [10, 20, 60]}],
    ["Volatility filter on {sym}: hold only when {n}-day realized volatility is below its 1-year median. Does that beat holding at the same exposure?",
     "Does sitting out high-volatility periods help on {sym}? Long when the {n}-day return volatility is under its rolling 252-day median, flat otherwise."],
    ("Should I avoid {sym} when it's volatile?", {"n": [20]},
     "I'll test a standard version: hold when 20-day realized volatility is below its 1-year median, flat otherwise."))

fam("vol_target", [
    '''def {fn}(d, n, target):
    vol = d.close.pct_change().rolling(n).std() * np.sqrt(252)
    return (target / vol).clip(upper=1.0).fillna(0.0)''',
    '''def {fn}(d, n, target):
    ann_vol = d.close.pct_change().rolling(n).std() * 252 ** 0.5
    w = target / ann_vol
    return w.where(w < 1.0, 1.0).fillna(0.0)'''],
    [{"n": [20], "target": [0.10, 0.15]}, {"n": [20, 60], "target": [0.15]}, {"n": [60], "target": [0.10]}],
    ["Volatility targeting on {sym}: position = {target} / annualized {n}-day vol, capped at 100%. Test each setting.",
     "Scale {sym} exposure to hit {target} annual volatility using {n}-day realized vol (max 1x). Better than holding at the same average exposure?"],
    ("Does volatility targeting improve {sym}?", {"n": [20], "target": [0.15]},
     "I'll target 15% annualized volatility from 20-day realized vol, capped at 100% invested."))

fam("atr_filter", [
    '''def {fn}(d, n):
    prev = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - prev).abs(), (d.low - prev).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(n).mean() / d.close
    return (atr_pct < atr_pct.rolling(252).median()).astype(float)'''],
    [{"n": [14]}, {"n": [14, 50]}, {"n": [7, 14]}],
    ["Hold {sym} only when ATR({n}) as a percent of price is below its one-year median. Does that beat holding at the same exposure?",
     "ATR-based calm filter on {sym}, ATR period {n}. Real edge or not?"])

fam("narrow_range", [
    '''def {fn}(d, n):
    rng = d.high - d.low
    return (rng == rng.rolling(n).min()).astype(float)''',
    '''def {fn}(d, n):
    rng = d.high - d.low
    # narrower than each of the prior n-1 days (shift(1): today isn't compared with itself)
    prior_min = rng.rolling(n - 1).min().shift(1)
    return (rng < prior_min).astype(float)'''],
    [{"n": [4, 7]}, {"n": [7]}, {"n": [5, 7, 10]}],
    ["NR{n} setup on {sym}: when today's high-low range is the narrowest of the last {n} days, go long the next day. Test each n.",
     "Narrowest-range-in-{n}-days days on {sym}: is the next day better than average?"])

# ---------------------------------------------------------------- calendar
fam("seasonal", [
    '''def {fn}(d, start, end=4):
    m = (d.index + pd.offsets.BDay(1)).month  # decided at the close, filled at the next open
    on = (m >= start) | (m <= end)
    return pd.Series(on.astype(float), index=d.index)'''],
    [{"start": [10, 11]}, {"start": [11]}, {"start": [9, 10, 11]}],
    ["\"Sell in May\" on {sym}: hold only from month {start} through April, cash otherwise. Test each start month. Any edge over holding at the same exposure?",
     "Seasonality test on {sym}: invested from month {start} to April each year, flat in the summer. Does it beat holding?"],
    ("Does 'sell in May and go away' work for {sym}?", {"start": [11]},
     "I'll test the classic version: hold November through April, flat May through October."))

fam("month_half", [
    '''def {fn}(d, half):
    day = (d.index + pd.offsets.BDay(1)).day  # the day the position is actually held
    on = day <= 15 if half == 1 else day > 15
    return pd.Series(np.asarray(on, dtype=float), index=d.index)'''],
    [{"half": [1, 2]}, {"half": [1]}, {"half": [2]}],
    ["Is the first or second half of the month better for {sym}? Hold only in half {half} (1 = days 1-15, 2 = days 16-end).",
     "Test holding {sym} only during half {half} of each calendar month (1 = first 15 days)."])

fam("quarter_end", [
    '''def {fn}(d, days):
    nxt = d.index + pd.offsets.BDay(1)
    q_end = (nxt + pd.offsets.BQuarterEnd(0))
    left = np.busday_count(nxt.values.astype("datetime64[D]"), q_end.values.astype("datetime64[D]"))
    return pd.Series((left < days).astype(float), index=d.index)'''],
    [{"days": [3, 5]}, {"days": [5]}, {"days": [2, 5, 10]}],
    ["Quarter-end effect in {sym}: hold only the last {days} trading days of each quarter. Test each window.",
     "Window dressing test on {sym}: long during the final {days} business days of every quarter, flat otherwise."])

fam("one_month", [
    '''def {fn}(d, month):
    m = (d.index + pd.offsets.BDay(1)).month
    return pd.Series((m == month).astype(float), index=d.index)'''],
    [{"month": [1]}, {"month": [1, 12]}, {"month": [9, 10]}],
    ["Is month {month} special for {sym}? Hold it only during that month (one strategy per month).",
     "Calendar test on {sym}: invested only in month number {month}, cash the rest of the year."],
    ("Is there a January effect in {sym}?", {"month": [1]},
     "I'll hold only during January and compare with holding at the same exposure."))

fam("week_of_month", [
    '''def {fn}(d, week):
    day = (d.index + pd.offsets.BDay(1)).day
    return pd.Series(((day - 1) // 7 + 1 == week).astype(float), index=d.index)'''],
    [{"week": [1, 2]}, {"week": [1, 2, 3, 4]}, {"week": [3]}],
    ["Week-of-month test on {sym}: hold only during week {week} of each month (week 1 = days 1-7). One strategy per week.",
     "Does {sym} do better in calendar week {week} of the month than at other times?"])

# ---------------------------------------------------------------- exposure / absolute
fam("constant_exposure", [  # vs market this is the benchmark itself: no timing to test
    '''def {fn}(d, frac):
    return pd.Series(frac, index=d.index)'''],
    [{"frac": [0.5]}, {"frac": [0.5, 1.0]}, {"frac": [0.25]}],
    ["Has holding {sym} at a constant {frac} of capital made money, in a statistically meaningful way? Compare against cash.",
     "Test a fixed {frac} allocation to {sym}, always on. Is the return distinguishable from zero? Use cash as the baseline."],
    vs="cash")

fam("constant_exposure_vs_market", [
    '''def {fn}(d, frac):
    return pd.Series(frac, index=d.index)'''],
    [{"frac": [0.5]}, {"frac": [0.75]}],
    ["Does keeping a constant {frac} position in {sym} beat holding {sym} at the same exposure?",
     "Compare an always-on {frac} allocation to {sym} against holding it at matched exposure."], cash_ok=False)

fam("trend_cash", [  # trend rules asked about absolute profit
    '''def {fn}(d, n):
    return (d.close > d.close.ewm(span=n, adjust=False).mean()).astype(float)'''],
    [{"n": [100, 200]}, {"n": [200]}],
    ["Did a {n}-day exponential moving-average filter on {sym} make money at all? I don't care about beating buy-and-hold, just whether it's profitable versus cash.",
     "Absolute-return check: hold {sym} above its {n}-day EMA, cash below. Is the profit statistically real?"], vs="cash")

FAMILY_NAMES = sorted(F)
