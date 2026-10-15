"""Case study 1: does the Cboe put/call ratio predict SPY?  (docs/case-studies.md)

Fixed before the first run: 3 ratios x {fear, calm} x {1, 5, 20} days = 18 variants,
one family. Discovery 2006-2019 (Cboe CSV era); 2020-2026 shown as a second look.
"""
import numpy as np
import pandas as pd
import tokio_ai

WIN, Z = 252, 1.5
BREAKS = [pd.Timestamp("2012-06-11"), pd.Timestamp("2019-10-07")]  # definitional break, source switch


def zscore(x):
    parts, edges = [], [x.index[0]] + BREAKS + [x.index[-1] + pd.Timedelta(days=1)]
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[(x.index >= a) & (x.index < b)]
        past = seg.shift(1).rolling(WIN, min_periods=WIN)
        parts.append((seg - past.mean()) / past.std())
    return pd.concat(parts)


r = tokio_ai.load_prices("SPY")["close"].pct_change()
conds = {}
for kind in ("equity", "index", "total"):
    pcr = tokio_ai.load_put_call(kind)
    z = zscore(np.log(pcr["ratio"])).reindex(r.index).shift(1)  # published after the close: use it next day
    conds[f"fear_{kind}"] = (z > Z).where(z.notna())
    conds[f"calm_{kind}"] = (z < -Z).where(z.notna())

for name, lo, hi in [("2007-2019 (discovery)", "2006-11-01", "2019-10-04"), ("2020-2026 (second look)", "2019-10-07", None)]:
    rr = r.loc[lo:hi]
    print(f"\n=== {name} ===")
    print(tokio_ai.check_many(rr, {k: c.loc[lo:hi] for k, c in conds.items()}, horizon=[1, 5, 20]))
