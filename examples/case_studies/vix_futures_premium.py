"""Case study 2: selling VIX futures (the volatility risk premium).  (docs/case-studies.md)

Builds a constant 30-day VIX futures position from Cboe's free per-contract settlement files
(each day's change measured on the same contracts: no roll artefact), then asks:
is there a premium, does any timing signal help, and does it add anything to holding SPY?
First run downloads ~160 contract files to ~/.tokio_ai/cache/vx.
"""
import io
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tokio_ai

CACHE = Path(os.environ.get("TOKIO_AI_CACHE", Path.home() / ".tokio_ai" / "cache")) / "vx"
CACHE.mkdir(parents=True, exist_ok=True)
VX_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{d}.csv"
IDX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{n}_History.csv"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()


def monthly_expiry(y, m):  # Wednesday 30 days before the third Friday of the next month
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    first = date(ny, nm, 1)
    return first + timedelta((4 - first.weekday()) % 7 + 14) - timedelta(30)


def contract(y, m):
    exp = monthly_expiry(y, m)
    for d in (exp, exp - timedelta(1), exp + timedelta(1), exp - timedelta(2)):  # holiday shifts
        p = CACHE / f"VX_{d}.csv"
        if not p.exists():
            try:
                body = get(VX_URL.format(d=d))
            except Exception:
                continue
            if not body.startswith(b"Trade Date"):
                continue
            p.write_bytes(body)
        df = pd.read_csv(p)
        s = df.set_index(pd.to_datetime(df["Trade Date"]))["Settle"].astype(float)
        return pd.Timestamp(d), s.where(s > 0)
    return None


def index(name):
    d = pd.read_csv(io.BytesIO(get(IDX_URL.format(n=name))))
    return d.set_index(pd.to_datetime(d["DATE"], format="%m/%d/%Y"))["CLOSE"].astype(float)


today = date.today()
px = dict(c for y in range(2013, today.year + 2) for m in range(1, 13) if (c := contract(y, m)))
px = pd.DataFrame(px).sort_index(axis=1).loc["2014":]
exps = np.array(px.columns, dtype="datetime64[ns]")

rows = []
for t0, t1 in zip(px.index[:-1], px.index[1:]):
    dte = (exps - np.datetime64(t0)) / np.timedelta64(1, "D")
    live = np.where(dte >= 1)[0]
    j = live[np.searchsorted(dte[live], 30)] if dte[live].max() >= 30 else live[1]
    a, b = (live[0], live[1]) if j == live[0] else (j - 1, j)
    w1 = min(max((dte[b] - 30) / (dte[b] - dte[a]), 0.0), 1.0)
    f0 = w1 * px.iat[px.index.get_loc(t0), a] + (1 - w1) * px.iat[px.index.get_loc(t0), b]
    df_ = w1 * (px.at[t1, px.columns[a]] - px.at[t0, px.columns[a]]) + (1 - w1) * (px.at[t1, px.columns[b]] - px.at[t0, px.columns[b]])
    roll_units = 2.0 / (dte[b] - dte[a])
    rows.append((t1, -df_ / f0, f0, roll_units))
s = pd.DataFrame(rows, columns=["date", "short", "level_prev", "roll_units"]).set_index("date")
s["level"] = s.level_prev.shift(-1)
vix, vix3m, vix9d = index("VIX"), index("VIX3M"), index("VIX9D")
s["roll"] = s.level / vix.reindex(s.index) - 1
s["ts"] = vix.reindex(s.index) / vix3m.reindex(s.index)
s["ts9"] = vix9d.reindex(s.index) / vix.reindex(s.index)

print(f"Worst day for a 1x short: {s.short.min():.1%} on {s.short.idxmin().date()}")

disc = s.loc[:"2020"]
print("\n=== 1. Is there a premium? (short at 1x, no costs, 2014-2020) ===")
print(tokio_ai.check_backtest(disc.short.dropna()))

lag = disc[["roll", "ts", "ts9"]].shift(1)
ok = lag.notna().all(axis=1)
conds = {"contango>10%": (lag.roll > 0.10).where(ok), "backwardation": (lag.roll < 0).where(ok),
         "VIX/VIX3M<0.9": (lag.ts < 0.9).where(ok), "VIX/VIX3M>1": (lag.ts > 1).where(ok),
         "VIX9D/VIX>1": (lag.ts9 > 1).where(ok)}
print("\n=== 2. Does any timing signal help? (2014-2020) ===")
print(tokio_ai.check_many(disc.short, conds, horizon=[1, 5]))

frac = 0.20  # a 1x short does not survive 2018-02-05
cost = (frac * s.roll_units.shift(1) + (frac * s.short.abs() * (1 + frac)).shift(1).fillna(0)) * 0.03 / s.level_prev
sized = (frac * s.short - cost).dropna()
hold = sized.loc["2021":]
print("\n=== 3. Sized at 0.2x with costs, holdout 2021 on ===")
print(tokio_ai.check_backtest(hold))

spy = tokio_ai.load_prices("SPY")["close"].pct_change()
both = pd.concat({"short_vol": sized, "spy": spy}, axis=1).dropna()
beta = np.polyfit(both.spy, both.short_vol, 1)[0]
alpha = (both.short_vol - beta * both.spy).mean() * 252
print("\n=== 4. Does it add anything to holding SPY? (2014 on) ===")
print(f"correlation {both.corr().iloc[0, 1]:.2f}, beta {beta:.2f}, alpha {alpha:+.2%}/yr")
for w in (0, 0.25, 0.5):
    m = (1 - w) * both.spy + w * both.short_vol
    print(f"  {int(w * 100):>2}% short vol + rest SPY: Sharpe {m.mean() / m.std() * np.sqrt(252):.2f}")
