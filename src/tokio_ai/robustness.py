"""Robustness diagnostics for a backtest: is the profit broad, steady, and executable?

The significance test says whether the mean is above zero. These say what
kind of profit it is, which a p-value can't:

- **Concentration.** The Sharpe after removing the best and the worst 1% of
  bars (or 5 trades at each end). Trimming both tails keeps a zero-edge
  strategy at zero -- removing only the best days sinks even buy-and-hold,
  which says nothing -- so a result that disappears here was living in a
  few outliers, not in the body of its trades.
- **Consistency.** The share of equal time blocks in which the strategy made
  money. Without an edge about half are positive; one great block can carry
  a significant mean while most of the history loses.
- **One-bar delay.** The Sharpe if every position were taken one bar later.
  An edge that vanishes with one bar of delay is either very fast (and needs
  execution to match) or is using information it didn't have yet: lookahead
  is the commonest backtest bug. Needs the positions and the asset's own
  returns, which the vectorbt and backtesting.py adapters supply.
- **Minimum backtest length** (Bailey, Borwein, Lopez de Prado & Zhu 2014):
  the years of data needed before the best of N zero-edge trials would be
  expected to show a Sharpe below the one observed. Shorter history than
  that, and the search alone can explain the result.

These are diagnostics, not tests: none has a false-positive rate, and the
output never calls one significant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

_N = NormalDist()
BLOCKS = 8
TOP_FRACTION = 0.01
TOP_TRADES = 5


@dataclass(frozen=True)
class Robustness:
    name: str
    unit: str
    sharpe: float  # annualized, as tested
    sharpe_trimmed: float  # annualized, `removed` best and `removed` worst dropped
    removed: int
    positive_blocks: int
    blocks: int
    sharpe_rebuilt: float | None = None  # annualized, positions x asset returns, no costs
    sharpe_delayed: float | None = None  # the same with every position one bar later
    min_backtest_years: float | None = None  # for the trials searched
    trials: int = 1

    def lines(self) -> list[str]:
        out = []
        what = f"best and worst {self.removed} {self.unit}"
        if self.sharpe > 0:
            verdict = (": still positive" if self.sharpe_trimmed > 0 else
                       ": gone, so the profit lives in a few outliers")
        else:
            verdict = ""
        out.append(f"Without its {what}, the Sharpe is {self.sharpe_trimmed:.2f} "
                   f"(from {self.sharpe:.2f}){verdict}.")
        steady = self.positive_blocks / self.blocks
        tail = (" Most of the history lost money; one stretch carries the mean." if steady < 0.5 else
                " About what a coin flip gives." if steady <= 0.625 else "")
        out.append(f"Made money in {self.positive_blocks} of {self.blocks} equal time blocks.{tail}")
        if self.sharpe_delayed is not None and self.sharpe_rebuilt is not None:
            line = (f"Rebuilt from its positions (no costs) the Sharpe is {self.sharpe_rebuilt:.2f}; "
                    f"with every position taken one bar later, {self.sharpe_delayed:.2f}.")
            if self.sharpe_rebuilt > 0.3 and self.sharpe_delayed < 0.5 * self.sharpe_rebuilt:
                line += (" Most of the edge needs same-bar execution: check for lookahead (a signal "
                         "using the bar's own close or later) before trusting it.")
            out.append(line)
        if self.min_backtest_years is not None:
            out.append(f"Minimum backtest length for {self.trials} trials at this Sharpe: "
                       f"{self.min_backtest_years:.1f} years.")
        return out


def _sharpe(np, x, ppy) -> float:
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0


def min_backtest_length(trials: int, sharpe: float) -> float | None:
    """Years of data before `trials` zero-edge strategies' best is expected below `sharpe` (annualized).

    Bailey et al. (2014): MinBTL ~ (E[max_N] / SR)^2 with E[max_N] the
    expected maximum of N standard normals.
    """
    if trials <= 1 or sharpe <= 0:
        return None
    g = 0.5772156649015329
    e_max = (1 - g) * _N.inv_cdf(1 - 1 / trials) + g * _N.inv_cdf(1 - 1 / (trials * math.e))
    return (e_max / sharpe) ** 2


def diagnose(np, x, ppy, unit, name, trials, positions=None, asset_returns=None) -> Robustness:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if unit == "trades":
        k = min(TOP_TRADES, max(1, n // 20))
    else:
        k = max(1, int(round(n * TOP_FRACTION)))
    trimmed = np.sort(x)[k: n - k] if n > 2 * k + 1 else x
    blocks = np.array_split(x, BLOCKS)
    pos_blocks = sum(1 for b in blocks if len(b) and b.mean() > 0)
    sr = _sharpe(np, x, ppy)
    rebuilt = delayed = None
    if positions is not None and asset_returns is not None:
        p = np.nan_to_num(np.asarray(positions, dtype=float), nan=0.0)
        r = np.nan_to_num(np.asarray(asset_returns, dtype=float), nan=0.0)
        if p.shape == r.shape == (n,) and n > 2:
            rebuilt = _sharpe(np, (p * r)[1:], ppy)
            delayed = _sharpe(np, (p[:-1] * r[1:]), ppy)
    return Robustness(name=name, unit=unit, sharpe=sr, sharpe_trimmed=_sharpe(np, trimmed, ppy),
                      removed=k, positive_blocks=pos_blocks, blocks=BLOCKS, sharpe_rebuilt=rebuilt,
                      sharpe_delayed=delayed, min_backtest_years=min_backtest_length(trials, sr),
                      trials=trials)
