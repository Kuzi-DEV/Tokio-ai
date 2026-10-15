"""Case study 3: SPY's overnight premium.  (docs/case-studies.md)

Hold SPY from the close to the next open only, 1 bp per round trip.
Discovery 2000-2018, holdout 2019 on; intraday half as the control; buy-and-hold for scale.
"""
import numpy as np
import pandas as pd
import tokio_ai

spy = tokio_ai.load_prices("SPY")  # split- and dividend-adjusted
legs = pd.DataFrame({
    "overnight (net 1bp)": spy.open / spy.close.shift(1) - 1 - 1e-4,
    "intraday": spy.close / spy.open - 1,
    "buy and hold": spy.close.pct_change(),
}).dropna()


def table(d):
    eq = (1 + d).cumprod()
    return pd.DataFrame({"return/yr": eq.iloc[-1] ** (252 / len(d)) - 1, "Sharpe": d.mean() / d.std() * np.sqrt(252),
                         "max drawdown": (eq / eq.cummax() - 1).min()}).round(3)


for name, lo, hi in [("2000-2018 (discovery)", "2000", "2018"), ("2019-2026 (holdout)", "2019", None)]:
    d = legs.loc[lo:hi]
    print(f"\n=== {name} ===\n{table(d)}")
    print(tokio_ai.check_backtest(d["overnight (net 1bp)"]))
