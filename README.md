# TokIO AI

[![tests](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml/badge.svg)](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/tokio-ai.svg)](https://pypi.org/project/tokio-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![calibrated](https://img.shields.io/badge/false%20positive%20rate-measured-brightgreen.svg)](docs/calibration.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jordanahern2009-svg/Tokio-ai/blob/main/examples/check_your_backtest.ipynb)

**Is your backtest real?** TokIO tells you whether a strategy's profit is
distinguishable from luck, once you allow for the two things a Sharpe ratio
hides: P&L that isn't independent from bar to bar, and the other variants
you tried before this one. Every test it runs has a measured false-positive
rate, published and reproducible.

```bash
pip install tokio-ai
```

```python
import tokio_ai

print(tokio_ai.check_backtest(pf))      # a vectorbt Portfolio, a whole parameter grid included
print(tokio_ai.check_backtest(stats))   # what backtesting.py's Backtest(...).run() returns
print(tokio_ai.check_backtest(tokio_ai.read_tradingview("List of trades.csv")))  # TradingView
print(tokio_ai.check_backtest(pnl, trials=40))  # or plain per-bar returns, from anywhere
```

Every result reports the **Deflated Sharpe Ratio** and **Probabilistic Sharpe
Ratio** (Bailey & López de Prado), the **minimum track record length**, the
**probability of backtest overfitting** (CSCV), a **haircut Sharpe**, the
**breakeven transaction cost**, and a **lookahead check**, next to a
significance test corrected for every variant you tried (a one-sided
Romano-Wolf step-down, the stepwise form of White's Reality Check). Each one's
false-positive rate is [measured](docs/calibration.md), including the
published Deflated Sharpe Ratio's, which [fails on overlapping
trades](#deflated-sharpe-ratio-psr-and-minimum-track-record-length).

Nineteen SMA crossovers on 20 years of SPY. The best one, tested alone:
**p = 0.038.** The same backtest with the other eighteen it was picked from:

```
NOT SIGNIFICANT after correcting for 19 trials (Romano-Wolf, using their
correlation) (p=0.1176, alpha=0.05). Strongest variant: sma10/100.
Haircut Sharpe after 19 trials: 0.21.
Probability of backtest overfitting (CSCV, 12870 splits): 0.90. Picking by
backtest did worse than picking at random: the in-sample winner finished at
or below the median out of sample in 90% of splits.
```

**[Run it yourself in Colab, then upload your own backtest →](https://colab.research.google.com/github/jordanahern2009-svg/Tokio-ai/blob/main/examples/check_your_backtest.ipynb)**
No API key, no account. There's also a CLI for a CSV export:
`tokio-ai-backtest pnl.csv --trials 40`.

TokIO also ships an LLM research agent (price history, SEC filings, plain-English
hypotheses) that routes every claim through the same tests;
[see below](#quickstart).

## We tested the tests, and they failed

Most tools that promise statistical rigor never check whether their own
statistics work. We checked ours, by running it on thousands of simulated
price series containing **no predictable pattern at all** and counting how
often it claimed to find one. A test that reports `p < 0.05` should be wrong
about 5% of the time.

TokIO v0.2.0 was wrong up to **45%** of the time.

Two compounding bugs, both specific to the questions this tool exists to
answer. Conditions like "days that dropped more than 2%" select volatile
days *by construction*, and a permutation test on a raw mean difference
isn't valid when one group is far noisier than the other. On top of that,
multi-day forward returns come from overlapping windows, so shuffling them
pretends there is far more independent data than there is.

v0.3.0 fixes both — a studentized statistic (Chung & Romano 2013) and a
circular-shift randomization that preserves the time structure instead of
destroying it. Worst-case false-positive rate went from **45% to 7%**, and the
engine got about 7x faster, because rotations are cheap enough to
enumerate exactly rather than sample.

The most useful thing the study found was in our own README. The showcase
example here used to be "AAPL gaps above 2% fade over the next 5 days,
p = 0.0042." Re-run with the corrected test on the same 10 years of real
data: **p = 0.089.** Not significant. The headline result was a false
positive produced by the bug.

**[Read the full study →](docs/calibration.md)** — or reproduce it yourself,
no API key or network needed:

```bash
python scripts/calibration_study.py
```

## Test a condition on your own data

No agent, no API key, no network. You bring returns and a condition you
think predicts them; `check()` tells you whether that's distinguishable
from noise.

```python
import tokio_ai

r = prices.pct_change()                     # your data: list, numpy or pandas
past = prices.shift(20)
momentum = (prices > past).where(past.notna())  # anything known at the bar's close
print(tokio_ai.check(r, momentum, horizon=20))
```

(The `.where(...)` is there because pandas evaluates `NaN > x` as `False`,
not NaN. Without it, the first 20 bars, where momentum is unknown, get
counted as "momentum down". `check()` skips None and NaN, but it can't
recover a NaN that pandas has already turned into `False`.)

Here is that exact check on 10 years of real SPY closes (2016-09 to 2026-09):

```
NOT SIGNIFICANT (p=0.3274, alpha=0.05). Over the next 20 bars, the 1742
condition bars averaged +1.003% vs +1.890% on the other 731 (gap -0.887%).
```

(The rotation engine, as a second opinion: p = 0.31.)

A Welch t-test on the same two groups returns **p = 0.0001**. That's how
you end up "discovering" a mean-reversion edge. The trap is that
consecutive 20-day returns share 19 of their 20 days, so 2,500 bars carry
nowhere near 2,500 independent observations. The t-test doesn't know that.

What `check()` handles for you:

- **Lookahead.** The outcome for bar *i* starts at bar *i+1*, so a
  condition can never predict its own bar. Two pandas Series with
  different indexes raise an error instead of being silently paired by
  position.
- **Overlapping windows and persistent conditions**, handled exactly by
  the default Hodrick engine, and checked by a rotation test that keeps
  the time structure of both series intact.
- **Volatility-selecting conditions**, via heteroskedasticity-robust
  statistics, plus a note telling you which way a naive test would have
  been wrong.
- **Testing many ideas.** Pass `ledger=tokio_ai.TestLedger()` to every
  call, and each verdict is Benjamini-Hochberg corrected against all of
  them.
- **Tiny samples.** Fewer than 30 bars on either side returns
  `NOT REPORTABLE`, not a p-value.

**Two engines on every call.** The verdict comes from Hodrick (1992)
standard errors, which handle the overlap between multi-bar windows
*exactly* instead of estimating it, plus a short HAC for the returns' own
autocorrelation. A circular-shift randomization test runs alongside as an
assumption-free second opinion, and the result tells you when they
disagree. With numpy installed both are fast: the rotation test evaluates
every rotation through an FFT, in 1.2 s at a million bars.

**How it compares to the tools quants already use**, on 51 simulated
markets with no edge (size) and with a planted one (power):

| test | worst false-positive rate | mean power |
|---|---:|---:|
| Welch t-test | 64.0% | 74.0% |
| Newey-West (HAC, lags = h) | 15.0% | 74.2% |
| stationary bootstrap (`arch`) | 15.0% | 74.3% |
| **TokIO `check()`** | **8.3%** | 73.5% |

Same power as Newey-West, within a point, at about half its worst-case
false-positive rate. The default engine was picked by that benchmark,
after two other designs lost.

**And on real markets, not just simulated ones:** on 20 years of daily
data for ten assets (equity indices, single stocks, bonds, gold, oil, FX),
20,000 random signals that by construction predict nothing were flagged
by a t-test up to 64% of the time, by Newey-West up to 10%, and by
`check()` at most 5.1%. [Real-data placebo study
→](docs/calibration.md#real-market-data-placebo-signals-on-20-years-of-ten-assets) [The full head-to-head, what we tried, and
where each engine is weaker
→](docs/calibration.md#head-to-head-tokio-vs-newey-west-vs-the-stationary-bootstrap)

## Check a whole grid without fooling yourself

The most common way to fool yourself isn't testing one idea badly. It's
testing thirty variants (lookback 5, 20 or 60; threshold 1%, 2% or 3%;
horizon 1, 5 or 20) and reporting the one that worked. On markets with **no
edge at all**, a 30-variant grid produces at least one p < 0.05 about
**45% of the time**.

```python
grid = {f"mom{L}": (prices > prices.shift(L)).where(prices.shift(L).notna())
        for L in (5, 10, 20, 60, 120, 250)}
grid.update({f"drop{int(x * 100)}%": r < -x for x in (0.01, 0.02, 0.03)})
print(tokio_ai.check_many(r, grid, horizon=[1, 5, 20]))
```

That exact grid on 20 years of real SPY data, run 2026-09-25 (2006-09-25
to 2026-09-25; the numbers move slightly as new days arrive):

```
27 variants tested as one family (alpha=0.05, family-wise). 0 survive the
correction; 2 would have looked significant alone.

variant        h        gap   p alone    p grid  verdict
mom5 @h5       5    -0.332%    0.0109    0.1350  not significant
drop1% @h1     1    +0.148%    0.0490    0.4187  not significant
drop2% @h1     1    +0.313%    0.0586    0.4533  not significant
...
```

Five-day momentum followed by a five-day reversal, p = 0.011 on its own,
is a textbook short-term-reversal "edge". Found as the best of 27 tries,
it's p = 0.135. Holm would have said about 0.29; `check_many` is half as
harsh because it knows the variants overlap, and still doesn't let the
search fool you.

`check_many` tests every variant at every horizon as one family and reports
which ones survive the search. The chance of even one false discovery
anywhere in the grid stays at 5%. Bonferroni and Holm do that too, but they
treat 30 near-copies of one idea as 30 unrelated ideas, and throw away real
edges. `check_many` measures how correlated the variants are, the same
overlap-exact way `check()` works, and uses it (Romano-Wolf step-down):

| on markets with no edge | any variant flagged |
|---|---:|
| no correction | 42.5-46.0% |
| Bonferroni / Holm | 1.5-2.5% |
| **`check_many`** | **3.5-4.0%** |

And with a real edge planted in one rare-event variant, `check_many` finds
it 30-100% more often than Holm, at the same error guarantee. On real
market data (2,000 random 45-variant grids on 20 years of ten assets),
no correction flagged something in 40.2% of grids, Holm in 0.5%, and
`check_many` in 2.6%. [Full grid study
→](docs/calibration.md#grids-check_many-and-the-cost-of-searching)

## Check a finished backtest

Most people don't arrive with a condition. They arrive with a column of
daily strategy returns and a Sharpe ratio they like. `check_backtest` asks
whether that profit is distinguishable from luck, once two things are
allowed for: the P&L's serial dependence, and the variants you tried before
this one.

```python
import tokio_ai

print(tokio_ai.check_backtest(pnl, trials=40))            # you tried 40 things
print(tokio_ai.check_backtest({"sma10/100": pnl_a, ...}))  # or pass them all
```

From vectorbt or backtesting.py, pass the object itself. Returns and the
position over each bar are read from it, and a multi-column vectorbt
Portfolio counts every column as a variant you tried:

```python
fast, slow = vbt.MA.run_combs(close, window=[5, 10, 20, 50], r=2)
pf = vbt.Portfolio.from_signals(close, fast.ma_crossed_above(slow), fast.ma_crossed_below(slow), fees=0.0005)
print(tokio_ai.check_backtest(pf))                     # all 6 pairs, corrected for each other

stats = Backtest(data, SmaCross, commission=0.001).run()
print(tokio_ai.check_backtest(stats))
```

That vectorbt grid is long-only, and on 20 years of SPY four of the six pairs
come out significant. Read the note under the verdict: they were long about
65% of the time in a market that rose, and against a zero-profit null that
drift is enough. Pass `benchmark=` buy-and-hold scaled to their average
exposure and every one of them fails. That's the question you usually mean.

Or without writing any Python, on a CSV exported from wherever you
backtest (one column per variant; a date column is skipped):

```bash
tokio-ai-backtest pnl.csv --trials 40
```

Nineteen SMA crossovers (fast 5/10/20/50, slow 50-250, long/short) on 20
years of real SPY, run 2026-09-29. The best of them, alone:

```
SIGNIFICANT (p=0.0379, alpha=0.05).
Sharpe 0.36 annualized over 4779 bars (~19.0 years at 252/year). A plain
t-test would say p=0.0577; allowing for serial dependence in the P&L,
p=0.0379 before any correction for trials.
It would pass as the best of at most 1 independent trial. If you tried
more than that, it's not evidence.
The P&L's lag-1 autocorrelation is -0.10: a test that treats bars as
independent overstates the uncertainty. This one doesn't.
```

(Yes, here the honest test is *less* strict than the t-test: SPY's daily
returns mean-revert slightly, and that makes a long-horizon average steadier
than independent bars would. The correction for dependence runs both ways.
The positions were passed as `positions=`, which sizes the window to the
~250-bar holding runs.)

All nineteen passed together, as they were actually found (run
2026-09-30, so the window has moved one day):

```
NOT SIGNIFICANT after correcting for 19 trials (Romano-Wolf, using their
correlation) (p=0.1181, alpha=0.05). Strongest variant: sma10/100.
Haircut Sharpe after 19 trials: 0.21.

No extra costs charged (returns are taken as already net of any fees); sma10/100 turns over 6.1x a year.
Probability of backtest overfitting (CSCV, 12870 splits): 0.91. Picking by
backtest did worse than picking at random: the in-sample winner finished at
or below the median out of sample in 91% of splits.
The in-sample winner's median Sharpe: 0.48 in sample, 0.20 out of sample.
```

That last part is the one to sit with. Cut the 19 years into 16 blocks and
pick the best crossover on any half of them: on the other half, it lands in
the bottom half of the nineteen 91% of the time. The backtest isn't telling
you which crossover is good.

What it reports that a Sharpe ratio doesn't:

- **An honest p-value for the mean.** The variance comes from a HAC
  estimator whose window adapts to how persistent the P&L is (Andrews'
  plug-in, with Kiefer-Vogelsang fixed-b critical values for long windows
  on short samples), so listing overlapping 20-day trade returns daily,
  or holding positions for months, can't pass as independent evidence.
- **The correction for your search.** Pass every variant and it uses how
  correlated they are (one-sided Romano-Wolf step-down), so 19 near-copies
  of one idea are charged less than 19 unrelated ideas. Pass only a count
  and it applies Šidák's worst case.
- **A haircut Sharpe** (the Sharpe your result is worth after the search),
  and **how many trials it could survive**: the number to compare against
  how many you really ran.
- **Costs and the breakeven cost.** Pass `positions=` and `costs=` (0.0005
  = 5 bps per unit traded) and the P&L is charged for every position
  change. With positions, it also reports the most you could pay per unit
  traded and still have a significant result, corrected for the search.
  Run alone, sma10/100 above breaks even at 9.7 bps and turns over 6.4x a
  year.
- **The probability of backtest overfitting** for any grid of four or more
  variants (`tokio_ai.probability_of_overfitting` on its own): does the
  variant that wins on part of the history keep winning on the rest?
  Bailey, Borwein, López de Prado & Zhu's combinatorially symmetric
  cross-validation, over all 12,870 half-splits of 16 blocks.

On P&L with no edge at all:

| | plain t-test | `check_backtest` |
|---|---:|---:|
| one strategy, worst of 16 simulated P&L shapes | 38.0% | **8.0%** |
| overlapping 20-day trades, 20y of 10 real assets | 30.5% | **5.4%** |
| best of 50 strategies, 20y of 10 real assets | 83.5% | **4.2%** |
| best of 100 strategies, simulated | 99.0% | **6.0%** |

PBO isn't a test, so it has no false-positive rate. On 50-variant grids of
pure noise it averages 0.50, as it should, but reads a falsely reassuring
"below 0.2" in 3-10% of them. With a real Sharpe of 1.0 in one of the 50, it
averages 0.11-0.26, and that variant is the one picked most often in 82-98%
of grids.

[Full study, including what broke on the way →](docs/calibration.md#finished-backtests-check_backtest)

## Deflated Sharpe Ratio, PSR and minimum track record length

The Deflated Sharpe Ratio (Bailey & López de Prado 2014) is the most-cited
answer to "is my Sharpe luck?": the probability that the true Sharpe beats
the best you'd expect from N zero-edge trials, allowing for skew and fat
tails. TokIO computes it, the Probabilistic Sharpe Ratio and the minimum
track record length, on every `check_backtest` result and on their own:

```python
from tokio_ai import deflated_sharpe_ratio, probabilistic_sharpe_ratio

print(deflated_sharpe_ratio(variants))                 # {name: returns}: tests the best, deflated for all
print(deflated_sharpe_ratio(pnl, trials=40))           # one series picked from 40
print(probabilistic_sharpe_ratio(pnl, dependence=True))
```

We measured the published formulas the way we measure everything else: on
P&L with no edge, how often do they pass (PSR or DSR ≥ 0.95)? They should
pass 5% of the time.

| P&L with no edge | PSR as published | PSR, `dependence=True` |
|---|---:|---:|
| random positions held 1-120 bars | 3.5-6.0% | 3.0-6.0% |
| autocorrelated P&L (+0.2) | 9.5-11.0% | 6.0-7.2% |
| overlapping 5-day trades | **18.8-20.5%** | 4.0-4.8% |
| overlapping 20-day trades | **37.8-38.5%** | 5.0-7.8% |

The published formula's variance assumes the bars are independent, so a
trade log where each day's P&L averages 20 overlapping holdings passes a
strategy with no edge **38% of the time**. `dependence=True` scales that
variance by the P&L's long-run variance (the same HAC estimator
`check_backtest` uses) and holds it to 4-8%. `check_backtest` reports both,
and says so when only the published one passes.

On grids, the DSR has the opposite problem. Picking the best of 20-100
variants, it almost never passes noise (0-0.5%), but with a real Sharpe of
1.0 planted in one variant it finds it **10-37%** of the time, against
**40-77%** for the Romano-Wolf correction `check_backtest` uses for its
verdict. The DSR treats the trials as independent; Romano-Wolf uses how
correlated they actually are. That's why the verdict comes from Romano-Wolf
and the DSR is reported alongside it.

## Is it robust? Concentration, consistency and lookahead

A p-value says the mean is above zero. It doesn't say what kind of profit
it is. Every `check_backtest` result adds:

- **Concentration:** the Sharpe without the best and worst 1% of bars (5
  trades at each end). Trimming both tails keeps a zero-edge strategy at
  zero; trimming only the best days sinks even buy-and-hold, which says
  nothing.
- **Consistency:** how many of 8 equal time blocks made money.
- **Lookahead:** the Sharpe if every position were taken one bar later.
  Given positions and the asset's own returns (automatic from vectorbt and
  backtesting.py, or `asset_returns=`), TokIO rebuilds the P&L and shifts
  it. A strategy that peeks at the bar's own close collapses.
- **Minimum backtest length** (Bailey, Borwein, López de Prado & Zhu 2014):
  the years of history before N trials can't produce this Sharpe by chance.

A deliberately cheating SPY strategy (it goes long on days that close up,
deciding at that day's close) passes every significance test, p=0.0000 with
a Sharpe of 8.2. The robustness section catches it:

```
Rebuilt from its positions (no costs) the Sharpe is 8.74; with every position
taken one bar later, 0.16. Most of the edge needs same-bar execution: check
for lookahead (a signal using the bar's own close or later) before trusting it.
```

An honest 10/50 crossover on the same data: 0.61, and 0.61 a bar later.

## TradingView

**Strategy Tester export.** Download the "List of trades" (CSV, or the XLSX
report) and test it per closed trade:

```python
trades = tokio_ai.read_tradingview("List of trades.csv")
print(tokio_ai.check_backtest(trades, trials=30))
```

```bash
tokio-ai-backtest "List of trades.csv" --trials 30
tokio-ai-backtest v1.csv v2.csv v3.csv          # several exports = several variants
```

Each trade's return is its P&L over the position's value (TradingView's "Net
P&L %"), so sizing doesn't move the verdict; `basis="equity"` uses P&L over
account equity. Old and new export formats, either row order, open trades
skipped. (TradingView's export needs a paid plan.)

**On the chart, free.** [`pine/tokio_check.pine`](pine/tokio_check.pine) is a
block you paste under any Pine v6 strategy. It reads the strategy's closed
trades and draws the verdict as a table: the dependence-corrected test,
Šidák's correction for the variants you say you tried, the Deflated /
Probabilistic Sharpe Ratio (corrected and as published), the minimum track
record length and the trimmed Sharpe. Verified in TradingView: it compiles
with no warnings (even under a strategy that reuses 25 common variable
names), and on a 469-trade SPY backtest every number in its table matches
the Python package's on the same trades. [`pine/demo_sma_cross.pine`](pine/demo_sma_cross.pine)
is a ready-made example. Grids, PBO, costs and the lookahead check need the
Python package.

## Bets on binary contracts: `check_contracts`

Prediction markets, binary options, bets at known odds. Each bet pays 1 or
0 and you know what you paid, and that makes an exact test possible:

```python
tokio_ai.check_contracts(prices, won, sizes=contracts, fees=0.003)
```

```bash
tokio-ai-backtest trades.csv --contracts price won --sizes count --fees 0.003
```

It matters most for favourites. Buy at 98c and you win 2c forty-nine times
for every 98c loss. A short record with no loss yet looks like a steady
stream of wins with almost no variance, and every test that estimates the
variance from the P&L calls it significant. On contracts priced exactly
fairly (no edge at all):

| book | plain t-test | `check_backtest` | `check_contracts` |
|---|---:|---:|---:|
| 98c favourites, 30 bets | 54.4% | n/a | 0.0% |
| 98c favourites, 300 bets | 16.6% | 16.2% | 2.2% |
| 95c favourites, 100 bets | 13.7% | 12.8% | 4.4% |
| 85-99c, sizes 1-50, 100 bets | 10.9% | 10.5% | 5.1% |
| worst of 24 rows | 54.4% | 16.2% | **5.6%** |

Nothing is estimated. The null is that each contract wins with
probability exactly equal to its breakeven (price plus fees), so the P&L's
distribution is known, and the p-value is computed exactly (a
Poisson-binomial recursion over contracts won). It also reports the other
tail, whether the book did significantly *worse* than fair, which is how
you find out that your entries are overpaying. And it reports an exact
upper bound on your true loss rate next to the breakeven one. At 98c,
even a perfect 30-for-30 record happens 55% of the time with no edge; the
output tells you how many bets a perfect record would need (149).

**Every verdict says how fragile it is.** A significant result comes with
the cost error that would flip it: "the edge disappears if the true cost per
contract is 0.6c higher than the prices and fees given". Under 1c, the
output warns that a bookkeeping error (limit price recorded instead of fill,
fee rounding, a price in the other side's terms) could explain it. That
warning exists because it happened to us: a p = 0.0003 "worse than fair"
result on a real book came from limit prices that were about 1c above the
fills. Its margin was 0.91c.

**Stop-losses and other early exits.** Pass what each early exit returned,
net of fees, and `None` for bets held to the end:

```python
tokio_ai.check_contracts(prices, won, fees=0.003, exit_values=exits)
```

```bash
tokio-ai-backtest trades.csv --contracts price won --exits exit_net
```

Any exit rule works: fixed stops, trailing stops, take-profits, momentum
reversals. If the market is fair, a position's price is a martingale, so
whatever the rule, the position is worth its breakeven on average and ends
somewhere between 0 and 1. The plain win/lose bet is the most spread-out way
to do that, which makes it a conservative reference. On simulated fair
15-minute markets with stops checked every minute or every second, the
false-positive rate stayed at or below 3.0%. A real 2c edge was still found
28-29% of the time over 500 stopped bets (36.5% without stops). A sharper
null (a stop at L ends at exactly L or 1) was tried first and fired 7-13%,
because real stops fill below their level and prices jump through them near
expiry.

`check_backtest` now warns when a P&L is shaped like this (skewness below
−2) and points here.

## Why this exists

Generic LLM agents are commoditized -- anyone can wrap an LLM in a chat loop
and call it an agent. What isn't commoditized is discipline: most retail
(and plenty of professional) research fails because someone eyeballs a mean,
sees a gap, and calls it an edge without asking how likely that gap was to
appear by chance. The rigor layer here (`tokio_ai.rigor`) generalizes a
hypothesis-testing discipline actually used across real trading research
projects -- see `rigor/stats.py` and `rigor/ledger.py` for the
randomization tests, minimum-sample gate, and Bonferroni/Benjamini-Hochberg
multiple-testing correction that every claim has to pass through.

Discipline you can't verify is just a claim, though, which is why
[`docs/calibration.md`](docs/calibration.md) measures whether any of it
actually holds -- and reports where it didn't.

## What it can do today (v0)

- Pull daily OHLCV price history for any ticker (Yahoo Finance, no key)
- Pull recent SEC filings for any ticker (EDGAR, no key)
- Rank the real S&P 500 by trailing return, with optional GICS sector
  filtering -- handles open-ended asks like "what are the best performing
  stocks" without requiring you to already know a ticker or sector
- Test whether a simple technical condition (a big daily move, a gap at the
  open, unusual volume) actually predicts what happens next -- fetches,
  buckets, and runs the test in one call, not via the model eyeballing
  raw numbers -- using Hodrick standard errors that handle overlapping
  forward windows exactly, with a circular-shift randomization test as a
  second opinion
- Compare a whole grid of thresholds and horizons in one call ("which drop
  size works best?"), corrected for having searched the grid, so the agent
  can't p-hack by testing variants one at a time and reporting the winner
- Run a two-sided permutation test comparing any two groups of numbers you
  already have -- studentized, so it stays honest when one group is much
  noisier than the other -- with automatic multiple-testing correction
  across everything tested in the session
- Chat with it via a full-screen terminal UI with a persistent sidebar (new chat, chat list, usage, settings), or a plain-text REPL (`tokio-ai-plain`); it decides when to call which tool
- Multiple named, disk-persisted chats -- start a new one from the sidebar, click back into old ones, each keeps its own multiple-testing correction history so resuming picks up exactly where you left off
- A usage view showing real token/request counts for the current chat (not a dollar cost -- the free tier has none)
- A settings screen for the model (free-text override; only the default is verified to support tool-calling) and tool-call permissions (auto-approve, or confirm every call)

## TUI layout

Sidebar on the left (new chat, chat list, usage/settings buttons), chat on
the right -- click-driven like a normal chat app, not a keybindings-only
REPL. A few shortcuts still exist for the same actions:

| Key | Action |
|---|---|
| `Ctrl+N` | New chat |
| `Ctrl+U` | Usage (tokens, requests, this chat) |
| `Ctrl+O` | Settings (model, tool permissions) |
| `Ctrl+C` | Quit |

Textual's built-in command palette is intentionally disabled -- it defaults
to `Ctrl+P`, which collided with this app's own bindings, and its "change
theme" command has no visible effect here since the CSS uses fixed colors
rather than Textual's theme-variable system (a real, working theme switcher
is planned, not built yet).

## What it explicitly does not do

- Give investment advice or pick stocks
- Execute trades
- Pretend a small or cherry-picked sample proves anything

## Quickstart

```bash
pip install tokio-ai
cp .env.example .env   # or just set the env vars directly
# fill in OPENAI_API_KEY (a free key from https://build.nvidia.com works out of the box)
# and TOKIO_AI_USER_AGENT in .env
tokio-ai
```

```
> Pull AAPL's price history and tell me the most recent closing price.
> Get NVDA's recent 10-K and 10-Q filings.
```

### Developing locally

```bash
git clone https://github.com/jordanahern2009-svg/Tokio-ai
cd Tokio-ai
pip install -e ".[dev]"
python -m pytest       # no API key needed, no network calls
python -m tokio_ai.cli # if the tokio-ai console script isn't on PATH
```

## Architecture

- `tokio_ai/rigor/` -- pure-Python statistics engine (studentized
  permutation testing, circular-shift randomization for time-ordered data,
  multiple-testing correction, session-level test ledger). Fully unit
  tested, zero dependencies beyond the standard library, and its
  false-positive rate is measured rather than assumed
  ([`docs/calibration.md`](docs/calibration.md), reproducible via
  `scripts/calibration_study.py`).
- `tokio_ai/tools/` -- data ingest (Yahoo price history, SEC EDGAR filings,
  a bundled real S&P 500 + GICS sector snapshot) and the agent-facing
  screening/pattern-testing/hypothesis-testing tools.
- `tokio_ai/agent/` -- the OpenAI-compatible tool-use loop (works against
  any provider with that API shape; defaults to NVIDIA's free NIM catalog),
  system prompt, and tool schemas. Pure logic, no I/O or presentation
  concerns -- both entry points below are just views over the same `Agent`.
- `tokio_ai/tui.py` -- the default full-screen terminal UI (`tokio-ai`),
  built with [Textual](https://textual.textualize.io/): a fixed banner,
  scrollable chat log, an input box, and modal screens for chat
  browsing/usage/settings, all dark-themed.
- `tokio_ai/cli.py` -- plain-text REPL fallback (`tokio-ai-plain`), for
  scripting, piping, or terminals that don't support a full-screen TUI.
- `tokio_ai/chat_store.py` -- local chat persistence, one JSON file per
  chat under `~/.tokio_ai/chats/` (no database dependency). Stores the raw
  message history plus the `TestLedger` state, so resuming a chat resumes
  its multiple-testing correction too, not just the transcript.

**On language choice:** this is pure Python for now. The rigor engine (many
permutation-test iterations over numeric arrays) is the one part of this
codebase that's a plausible candidate for a Rust extension if it ever
becomes an actual measured bottleneck -- but the agent loop is I/O-bound on
LLM API calls, not local compute, so a polyglot rewrite ahead of a real
performance problem would just be added build complexity for no benefit.
Python first, optimize what's proven slow, not what looks slow.

## Status

Early and under active development. The rigor engine and data-ingest tools
are tested against live sources. The agent loop has been verified
end-to-end against NVIDIA's free NIM catalog (`nvidia/nemotron-3-super-120b-a12b`
by default) -- real tool calls, real data, correct multi-turn answers.
Free-tier models do get retired: the previous default went dark on
2026-08-26. If you get an HTTP 410 "end of life" error, set `TOKIO_AI_MODEL`
to another tool-calling model from https://build.nvidia.com.

**Known limitation:** the free tier has inconsistent latency (observed
anywhere from ~5s to 90s+ for the same model/prompt shape). That's the
tradeoff for "runs with zero-cost credentials out of the box." If you have
a paid OpenAI-compatible key with better SLAs, point `OPENAI_BASE_URL` /
`OPENAI_API_KEY` / `TOKIO_AI_MODEL` at it and nothing else changes.

Earlier versions asked the agent to manually crunch raw price history inline
for "does X predict Y"-style questions, which was unreliable on any LLM
backend (not specific to this one) -- that kind of bucketing belongs in a
Python tool, not the model's own token-by-token reasoning over a big JSON
blob. `test_return_pattern` and `top_performing_stocks` now do that
fetch+compute work in Python for the common cases (a technical condition
predicting forward returns; ranking stocks by trailing performance). If you
ask something shaped differently enough that neither tool fits, the model
may still fall back to reasoning over raw data by hand -- treat that path
as unreliable until there's a dedicated tool for it.

### Example

```
> Test whether AAPL days that gap up more than 2% at the open tend to keep
  drifting up over the next 5 trading days, using 10 years of history.

Not significant (p=0.089). Over 2016-08-18 to 2026-08-17 there were 76 days
with a gap above 2%, and they averaged -1.29% over the next 5 trading days
versus baseline -- so the point estimate does lean toward gap-fade rather
than continuation, but not by enough to separate from chance.

Worth flagging: those 76 days have 2.68x the return variance of the other
2,430. That is the condition selecting volatile days, and it is exactly the
case where a naive test overstates significance -- this one reported
p=0.0042 on the same data before v0.3.0.
```

That second paragraph is the whole point of the project. The interesting
answer was not the pattern; it was the reason to distrust the pattern.

## License

MIT -- see [LICENSE](LICENSE).
