# Does the rigor engine actually work?

TokIO's whole pitch is that it won't call noise a signal. That is a
falsifiable claim, so this is the file where it gets falsified.

A test that reports "significant at p < 0.05" is promising something
specific: run it on data where nothing is there, and it should say
"significant" about 5% of the time. Not 20%. Not 45%. You can check this,
and until v0.3.0 nobody had.

**When we checked, TokIO failed.** Badly, and on exactly the questions it was
built to answer. This document is the study, the diagnosis, the fix, and the
same study re-run afterwards.

Reproduce all of it:

```bash
python scripts/calibration_study.py --trials 200
```

That is the exact invocation behind every number below. No API key and no
network; about 9 minutes on a laptop, nearly all of it spent running the
*old* method for comparison. The default (`--trials 120`) is quicker.

## How you test a test

Generate price data with **no predictable structure at all**, then ask TokIO
to find some. Every "significant" answer it returns is, by construction, a
false positive. Count them.

The generator is a GARCH(1,1) process. Returns are conditionally mean-zero,
so nothing computed from the past can forecast the sign of anything in the
future — there is no edge to find, at any horizon, with any feature. But it
reproduces the one property of real markets that matters here: **volatility
clusters.** Quiet stretches and violent stretches arrive in runs, exactly as
they do in real price series. Volume is modeled as persistent for the same
reason.

That detail turns out to be the whole story. An earlier version of this
study used plain Gaussian random walks and TokIO passed cleanly — 4.2%,
3.3%, 8.3%, 5.8% across horizons, all within noise of the promised 5%. The
bugs below are invisible on data that is too well-behaved, which is a decent
argument for why they survived a full test suite and a prior bug hunt.

## What we found

Two independent defects, which compound.

### 1. Selecting on volatility breaks a raw-difference permutation test

A permutation test shuffles group labels and asks how often chance produces
a gap this large. That is valid when the two groups are *exchangeable* under
the null — informally, when the only thing that could differ between them is
the mean.

Every condition worth asking TokIO about violates this. "Days that fell more
than 2%." "Days with unusual volume." These select high-volatility days **by
construction**, and their forward returns are genuinely more spread out than
baseline days are. Measured on the study data, the condition group had
**1.75x the variance** of the baseline group — and was 15x smaller (n≈54 vs
n≈844).

Pooling a small, noisy group with a large, quiet one and shuffling makes the
small group's mean look far more stable than it is. The null distribution
comes out too narrow, and ordinary noise clears the bar. This is the
Behrens–Fisher problem, and it shows up even at a one-day horizon where
nothing overlaps.

### 2. Overlapping forward windows are not independent observations

Ask about a 20-day forward return and consecutive observations share 19 of
their 20 days. They are nearly the same number. Shuffling labels treats them
as 800 independent draws when the sample holds nothing like 800 independent
pieces of information — and because volatility clusters, the *labels* arrive
in runs too, so the real overlap is worse than random.

Here is that defect in one line, on synthetic data with no relationship
whatsoever between the condition and the outcome:

| method | p-value |
|---|---|
| shuffled raw difference | **0.00033** |
| studentized circular shift | 0.49 |

Same data. One of these is wrong by a factor of about 1,500. It is a
regression test now
(`test_circular_shift_is_not_fooled_by_clustered_labels_on_autocorrelated_values`).

## The fix

**Studentize the statistic.** Instead of permuting the raw mean difference,
permute a Welch-style statistic — the difference divided by its own standard
error, with each group's variance recomputed on every relabeling. Permuting
a studentized statistic stays asymptotically valid when the groups have
different variances; permuting a raw difference does not. See Chung & Romano
(2013), *Exact and asymptotically robust permutation tests*.

**Rotate instead of shuffling.** For time-ordered data, don't scramble the
labels — slide the whole label series along the value series and wrap it
around. Every rotation preserves the autocorrelation of both series exactly,
so overlapping windows and clustered conditions are built into the null
rather than assumed away. The hypothesis being tested becomes the right one:
*are these two series related, beyond what each one's own internal structure
already explains?*

A rotation is also cheap. Only the condition-met days move, and they are
typically a small slice of all days, so TokIO enumerates **every distinct
rotation** rather than sampling — the p-value is exact rather than Monte
Carlo, and the whole thing got roughly 7x faster.

## Results

False-positive rate on data containing no edge. Nominal is 5%.

| condition | horizon | before (v0.2.0) | after (v0.3.0) |
|---|---|---|---|
| `daily_return < -2%` | 1d | 15.8% | **7.0%** |
| `daily_return < -2%` | 5d | 21.6% | **5.8%** |
| `daily_return < -2%` | 20d | 30.2% | **4.1%** |
| `daily_return < -2%` | 60d | 45.1% | **6.8%** |
| `volume_ratio > 2` | 1d | 4.0% | **4.5%** |
| `volume_ratio > 2` | 5d | 21.5% | **4.0%** |
| `volume_ratio > 2` | 20d | 26.0% | **3.5%** |
| `volume_ratio > 2` | 60d | 21.5% | **5.5%** |

Both columns come from the same run on the same simulated data, so the
comparison is paired rather than two studies stitched together. 200 paths per
configuration; a few are skipped where the condition group falls below the
`MIN_SAMPLE` floor, so the effective count per row is 162-200 and the
standard error on each figure is roughly 1.6 points.

Worst case went from **45.1% to 7.0%**. Every corrected figure lands
between 3.5% and 7.0%, all within sampling noise of the promised 5%.

Calibration did not come for free in the other direction: a planted, real
effect is still detected (`test_circular_shift_finds_a_real_planted_effect`).
A test that never fires would be perfectly "calibrated" and perfectly
useless.

## The example in our own README was a false positive

The uncomfortable part. Until v0.3.0 the README's showcase result was AAPL
gapping up more than 2% at the open, tested against the next 5 days, over 10
years of real data. It reported **p = 0.0042** and a confident story about
gap-fade.

Re-run on the same real data with the corrected test:

| method | p-value | verdict |
|---|---|---|
| shuffled raw difference (v0.2.0) | 0.0042 | "significant gap-fade" |
| studentized circular shift (v0.3.0) | **0.089** | not significant |

The variance ratio between the two groups is **2.68x** — those 76 gap-up
days really are far more volatile than the other 2,430, precisely the
condition that broke the old test. The pattern may well be real; the honest
statement is that 10 years of AAPL history is not enough to establish it.

A tool built to stop people fooling themselves with statistics had, in its
own front-page example, fooled itself with statistics. Finding that is the
point of writing the check.

## `check()` on harsher nulls

`check()` runs the engine on any condition a user writes, so the engine has
to hold on more than the two conditions and the one generator above. This
second study adds:

- **fat tails**: GARCH with Student-t(4) shocks (`garch_t4`)
- **volatility regimes**: a calm (0.8%/day) and a crisis (3%/day) state,
  each lasting about 100 bars (`regime`)
- **a persistent condition**: "20-bar momentum is positive" stays true or
  false for weeks at a time
- **autocorrelated returns**: AR(1) at −0.25, like intraday bid-ask
  bounce (`ar_neg`), and at +0.15, like trend (`ar_pos`). These are tested
  with a persistent condition that is *independent* of the returns. A
  condition built from past returns genuinely predicts AR returns, so it
  wouldn't be a null.

In every row, the condition carries no information about future returns.
Both of `check()`'s engines are reported (see the head-to-head below for
why there are two), next to a Welch t-test
(`scipy.stats.ttest_ind(equal_var=False)`).
300 null paths × 1,000 bars per row.

```bash
python scripts/calibration_check.py --trials 300   # ~1 min on 4 cores
```

| generator | condition | horizon | t-test | TokIO `hodrick` (default) | TokIO `rotation` |
|---|---|---:|---:|---:|---:|
| garch | drop 2% | 1 | 4.6% | 4.3% | 4.3% |
| garch | drop 2% | 5 | 7.5% | 3.9% | 3.2% |
| garch | drop 2% | 20 | **21.2%** | 2.2% | 2.5% |
| garch | up day | 1 | 3.3% | 3.0% | 3.3% |
| garch | up day | 5 | 5.7% | 5.0% | 5.3% |
| garch | up day | 20 | 6.3% | 7.7% | 5.0% |
| garch | momentum 20 | 1 | 6.7% | 7.7% | 4.3% |
| garch | momentum 20 | 5 | **35.3%** | 5.7% | 5.3% |
| garch | momentum 20 | 20 | **56.7%** | 4.7% | 2.3% |
| garch | vol shock | 1 | 5.7% | 4.7% | 5.7% |
| garch | vol shock | 5 | 3.0% | 1.0% | 2.7% |
| garch | vol shock | 20 | 3.7% | 3.3% | 4.0% |
| garch | independent, persistent | 1 | 3.0% | 2.7% | 3.7% |
| garch | independent, persistent | 5 | **34.3%** | 4.0% | 6.0% |
| garch | independent, persistent | 20 | **61.0%** | 3.7% | 5.0% |
| garch_t4 | drop 2% | 1 | 5.3% | 3.6% | 4.7% |
| garch_t4 | drop 2% | 5 | 6.5% | 3.0% | 4.2% |
| garch_t4 | drop 2% | 20 | **25.6%** | 2.4% | 3.7% |
| garch_t4 | up day | 1 | 4.7% | 3.7% | 4.0% |
| garch_t4 | up day | 5 | 6.3% | 7.3% | 6.0% |
| garch_t4 | up day | 20 | 6.7% | 7.0% | 5.3% |
| garch_t4 | momentum 20 | 1 | 5.0% | 5.7% | 4.7% |
| garch_t4 | momentum 20 | 5 | **33.0%** | 4.3% | 4.3% |
| garch_t4 | momentum 20 | 20 | **51.7%** | 6.3% | 3.3% |
| garch_t4 | vol shock | 1 | 4.3% | 3.3% | 3.7% |
| garch_t4 | vol shock | 5 | 4.3% | 3.3% | 3.0% |
| garch_t4 | vol shock | 20 | 4.0% | 3.0% | 3.7% |
| garch_t4 | independent, persistent | 1 | 4.7% | 3.3% | 4.3% |
| garch_t4 | independent, persistent | 5 | **36.3%** | 6.0% | 7.0% |
| garch_t4 | independent, persistent | 20 | **57.7%** | 3.0% | 4.3% |
| regime | drop 2% | 1 | 4.0% | 3.7% | 4.0% |
| regime | drop 2% | 5 | 10.7% | 7.7% | 5.0% |
| regime | drop 2% | 20 | **24.1%** | 4.7% | 3.3% |
| regime | up day | 1 | 6.3% | 6.3% | 6.7% |
| regime | up day | 5 | 3.7% | 4.7% | 4.7% |
| regime | up day | 20 | 5.3% | 5.3% | 4.3% |
| regime | momentum 20 | 1 | 7.7% | 7.0% | 3.7% |
| regime | momentum 20 | 5 | **32.3%** | 5.0% | 2.7% |
| regime | momentum 20 | 20 | **54.7%** | 3.7% | 3.3% |
| regime | vol shock | 1 | 4.0% | 4.0% | 3.7% |
| regime | vol shock | 5 | 7.3% | 4.7% | 5.3% |
| regime | vol shock | 20 | 9.0% | 3.7% | 6.3% |
| regime | independent, persistent | 1 | 5.0% | 4.3% | 3.7% |
| regime | independent, persistent | 5 | **36.7%** | 3.7% | 4.3% |
| regime | independent, persistent | 20 | **60.7%** | 2.3% | 3.3% |
| ar_neg | independent, persistent | 1 | 0.3% | 2.7% | 4.7% |
| ar_neg | independent, persistent | 5 | **33.3%** | 3.7% | 5.7% |
| ar_neg | independent, persistent | 20 | **60.0%** | 3.7% | 5.0% |
| ar_pos | independent, persistent | 1 | 8.0% | 2.7% | 3.7% |
| ar_pos | independent, persistent | 5 | **36.3%** | 4.0% | 6.0% |
| ar_pos | independent, persistent | 20 | **60.7%** | 4.0% | 5.0% |

**Worst case: t-test 61.0%, TokIO `hodrick` 7.7%, TokIO `rotation` 7.0%.**
With 300 paths per row, one standard error is about 1.3 points, so both
engines are within noise of 5%. The t-test's failures sit where the time
structure is strongest: persistent conditions at long horizons, where it
"finds" an edge in 55–61% of markets that have none.

This is not hypothetical on real data. On 10 years of SPY, "20-day
momentum up → next 20 days" gets p = 0.0001 from the t-test, p = 0.33 from
`check()`'s default engine, and p = 0.31 from the rotation engine.

## Head-to-head: TokIO vs Newey-West vs the stationary bootstrap

Beating a naive t-test is a low bar. The real question a quant would ask is
how this compares to the tools they already trust:

- **Newey-West**: OLS of the forward return on a condition dummy, HAC
  standard errors with `maxlags = h` (statsmodels). This is the textbook fix
  for overlapping horizons.
- **Stationary block bootstrap** of the (condition, outcome) pairs, with
  block length from `arch`'s `optimal_block_length` and a studentized,
  centered statistic. This is what a careful quant does.

Every test answers the same question on the same paths, and we measure two
things:

- **Size:** how often it fires with no edge in the data (should be ~5%).
- **Power:** how often it fires with a real edge planted, sized to about
  1.5 standard errors so that power lands mid-range.

200 paths x 1,000 bars per row, on the same 51 null configurations as
above.

```bash
pip install -e ".[calibration]"
python scripts/benchmark.py --trials 200    # ~7 min on 4 cores
```

| test | mean size | **worst size** | mean power |
|---|---:|---:|---:|
| Welch t-test | 19.3% | **64.0%** | 74.0% |
| Newey-West (HAC, lags = h) | 7.0% | **15.0%** | 74.2% |
| stationary bootstrap (arch) | 7.8% | **15.0%** | 74.3% |
| **TokIO `hodrick` (default)** | **4.4%** | **8.3%** | 73.5% |
| TokIO `rotation` | 4.6% | 8.3% | 72.7% |

Both TokIO worst cases (8.3%) come from the fat-tailed "drop 2%" rows,
which have only 60 usable paths (one standard error ≈ 2.8 points). Across
rows with about 200 paths (196-200), the default engine's worst is 7.7%.

Where they differ (size / power):

| setup | Newey-West | bootstrap | TokIO `hodrick` | TokIO `rotation` |
|---|---:|---:|---:|---:|
| GARCH, momentum 20 → next 20 | 12.5% / 100.0% | 13.5% / 100.0% | 6.5% / 100.0% | 3.0% / 100.0% |
| regime, momentum 20 → next 20 | 12.0% / 100.0% | 13.0% / 100.0% | 6.0% / 100.0% | 3.5% / 100.0% |
| AR(+0.15) returns, independent condition → next 20 | 10.0% / 100.0% | 11.0% / 100.0% | 5.0% / 100.0% | 7.5% / 100.0% |
| GARCH-t4, drop 2% → next 20 | 15.0% / 100.0% | 15.0% / 98.3% | 0.0% / 93.3% | 8.3% / 85.0% |
| **GARCH, momentum 20 → next 1** | 6.0% / 26.0% | 7.5% / 31.0% | 5.0% / **25.0%** | 4.5% / **12.5%** |
| **regime, momentum 20 → next 1** | 6.5% / 26.0% | 8.5% / 30.5% | 6.0% / **25.0%** | 6.0% / **16.0%** |

(The GARCH-t4 rows have 60 usable paths, since 2% drops are rarer under
those tails. One standard error there is about 2.8 points.)

### How the default engine was chosen

The first version of this benchmark had only the rotation test. It held
its size everywhere, but lost to Newey-West in one clear place: a
persistent condition at a one-bar horizon, where it caught half as many
real edges. The mechanism: when a persistent condition carries a real edge,
the edge makes the return series persistent too. A rotation test keeps
that persistence in its null, so rotations a few bars off the truth look
nearly as extreme as the truth.

What we tried, in order:

1. **A minimum-shift guard band** (exclude near-identity rotations, as
   circular-shift tests in spatial statistics do). Power went from 12.5% to
   ~20%, but size rose to 6.5-8%. Rejected.
2. **Hodrick (1992) "1B" standard errors.** Regroup the statistic by bar
   instead of by observation, `sum_i a_i y_i = sum_t r_t B_t` with `B_t`
   the rolling sum of conditions whose window contains bar t, so the
   overlap is handled exactly instead of estimated. This matched
   Newey-West's power with far better size, but leaked to 10% when the
   one-bar returns are themselves autocorrelated (AR +0.15). That's
   expected: 1B assumes they aren't.
3. **Cauchy combination** of the rotation and 1B p-values (Liu & Xie 2020).
   Worst case 10%, worse than either alone. Rejected.
4. **1B plus a short Bartlett HAC** over the one-bar terms
   (`rigor/overlap.py`). The long-range overlap stays exact inside `B_t`;
   the kernel only has to cover the returns' own short memory, so it uses
   the Newey-West rule-of-thumb bandwidth, 4(n/100)^(2/9), about 6 at a
   thousand bars, not h. Shipped as the default -- then revised by
   step 5.
5. **Widen the kernel to the condition's own time scale.** The real-data
   placebo study below found one failure the simulations missed: on USO,
   whose returns trend over months, a random condition persisting ~120
   bars fired 11-14% of the time. A significance-test switch (Lo-MacKinlay
   variance ratio, then use the rotation engine) never triggered: 20 years
   of data can't establish that memory at the 5% level. What worked is a
   bandwidth of max(rule of thumb, the condition's mean run length + h).
   USO fell to ~6%, power and simulated size were unchanged within noise,
   and the bandwidth depends only on the condition's persistence, so it
   can't be steered toward a result. **This is the current default.**

The rotation test still runs on every `check()`, as a second opinion that
assumes nothing about return autocorrelation. When the two land on
opposite sides of alpha, the result says so, and which assumption the
verdict depends on.

**Neither engine is uniformly best, and we say so.** The Hodrick engine
relies on the one-bar returns having *short-range* autocorrelation, which is
true of daily bars and of bid-ask bounce, but not of returns with slow mean
drift. The rotation engine relies on stationarity but assumes nothing
about autocorrelation, and pays for that in power on persistent conditions
at short horizons. `method="rotation"` makes the rotation test the verdict.

## Real market data: placebo signals on 20 years of ten assets

Every study above simulates the returns, and the fair objection is that
GARCH is not a market. This one uses real returns and simulates only the
signal: 20 years of daily prices (2006-09 to 2026-09) for SPY, QQQ, IWM,
TLT, GLD, USO, AAPL, JPM, XOM and EUR/USD -- through 2008, 2020 and 2022,
with every real crash, regime and gap.

Each placebo is a random signal generated independently of the prices, so
it carries no information about them. A "significant" verdict is a false
positive, measured on the real return path. Signals come in four
persistence levels (mean run length 1, 5, 30 and 120 bars), 500 per asset
per level: 20,000 placebos, each tested at three horizons.

```bash
python scripts/placebo_real.py --placebos 500    # ~6 min, needs network
```

Pooled across all ten assets (5,000 placebos per cell; one standard error
≈ 0.3 points):

| signal run length | horizon | t-test | Newey-West | TokIO hodrick (default) | TokIO rotation |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5.1% | 5.2% | 5.1% | 5.1% |
| 1 | 5 | 4.6% | 4.6% | 4.1% | 4.6% |
| 1 | 20 | 4.5% | 4.6% | 4.4% | 4.7% |
| 5 | 1 | 3.6% | 4.3% | 4.6% | 4.9% |
| 5 | 5 | 22.2% | 7.6% | 5.1% | 5.7% |
| 5 | 20 | 30.2% | 6.0% | 4.5% | 5.0% |
| 30 | 1 | 3.2% | 3.6% | 4.7% | 5.1% |
| 30 | 5 | 34.5% | 8.4% | 4.7% | 4.9% |
| 30 | 20 | 59.2% | 9.1% | 4.6% | 4.8% |
| 120 | 1 | 3.0% | 3.4% | 4.3% | 4.9% |
| 120 | 5 | 34.5% | 8.7% | 4.3% | 4.8% |
| 120 | 20 | 64.1% | 10.4% | 4.7% | 5.4% |

| test | worst pooled rate | worst single asset |
|---|---:|---|
| Welch t-test | 64.1% | 67.6% (USO, run 120, h=20) |
| Newey-West | 10.4% | 16.4% (USO, run 120, h=5) |
| TokIO `hodrick` (default) | 5.1% | 7.8% (GLD, run 5, h=5) |
| TokIO `rotation` | 5.7% | 7.8% (GLD, run 5, h=5) |

The same pattern as the simulations, now on real returns. The t-test fails
on persistent signals (64% at a 20-day horizon), and Newey-West drifts to
8-10% pooled and 16% on USO. Both TokIO engines stay at 4.3-5.7% pooled.
Each single-asset figure is the worst of 120 cells at 500 placebos each,
so a value near 8% is about what chance alone produces at the maximum.

**This study changed the engine.** Its first run found the default engine
at 11% on USO (12% in a separate 400-placebo check; see step 5 above). USO's returns trend over months:
its 120-day variance ratio is 1.63, where the other nine range from 0.50
(JPM) to 0.99 (AAPL). The simulations had no generator with that
property. After the bandwidth fix, USO's worst cell for the default engine
is 6.4%, in the independent-coin-flip row rather than a persistent one,
and all numbers above are from the post-fix run.

## Grids: `check_many` and the cost of searching

Everything above tests one condition. Research doesn't work that way:
people try a grid of variants and keep the best. `check_many` tests the
grid as one family and controls the family-wise error rate, the
probability of even one false discovery anywhere in it.

**How.** Each variant gets exactly the statistic `check()` gives it alone
(a Hodrick z). Every such z is a sum of one-bar terms, so the covariance
between any two variants' statistics can be estimated from those terms,
with the overlap handled exactly, at a common Bartlett bandwidth that keeps
the matrix positive semi-definite. The correction is the Romano & Wolf
(2005) step-down max-|z| against that joint normal, by Monte Carlo (20,000
draws, seeded). By the union bound it can never be more conservative than
Holm, whatever the correlation.

**The grid** here: momentum lookbacks 5/10/20/40/60 and drop thresholds
1/1.5/2/2.5/3%, each at horizons 1, 5 and 20, for 30 variants that are
heavily correlated. All methods start from the same per-variant p-values,
so the comparison isolates the correction. 400 paths per generator.

```bash
python scripts/calibration_family.py --trials 400   # ~8 min on 4 cores
```

Family-wise error, with no edge anywhere (target 5%):

| generator | no correction | Bonferroni | Holm | Benjamini-Hochberg | **TokIO `check_many`** |
|---|---:|---:|---:|---:|---:|
| garch | 43.8% | 2.5% | 2.5% | 2.8% | 4.0% |
| garch_t4 | 42.5% | 2.2% | 2.2% | 2.5% | 3.5% |
| regime | 46.0% | 1.5% | 1.5% | 1.8% | 3.5% |

With no correction, a grid search "finds" something almost half the time.
Every correction holds the line, but Bonferroni and Holm hold it at
1.5-2.5%, well inside 5%. That slack is power they give away. `check_many`
runs closer to 5% because it knows the 30 variants are a few ideas tried
several ways.

Power: one real edge planted in one variant (3 standard errors), and how
often that variant is flagged:

| generator | planted in | no correction | Bonferroni | Holm | Benjamini-Hochberg | **TokIO `check_many`** |
|---|---|---:|---:|---:|---:|---:|
| garch | mom20 @h5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| garch | drop0.02 @h1 | 54.8% | 11.8% | 12.0% | 15.8% | 15.8% |
| garch | mom60 @h20 | 100.0% | 98.0% | 99.2% | 100.0% | 100.0% |
| garch_t4 | mom20 @h5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| garch_t4 | drop0.02 @h1 | 26.8% | 2.0% | 2.5% | 5.0% | 5.2% |
| garch_t4 | mom60 @h20 | 100.0% | 99.0% | 99.2% | 100.0% | 99.8% |
| regime | mom20 @h5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| regime | drop0.02 @h1 | 58.5% | 16.2% | 16.5% | 23.2% | 23.0% |
| regime | mom60 @h20 | 100.0% | 98.5% | 98.5% | 99.8% | 99.8% |

The momentum edges are found about 100% of the time by every method, so
they don't separate the methods. The rare-event variant does. `check_many`
finds it 30-100% more often than Holm (15.8% vs 12.0%, 5.2% vs 2.5%, 23.0%
vs 16.5%) at the same family-wise guarantee, and matches
Benjamini-Hochberg, which only controls the weaker false-discovery rate.

**What didn't work.** The first family engine rotated every variant
jointly against the returns (min-P step-down over joint rotations). It held
its error rate, but found a planted persistent edge **0%** of the time where
every other method found it 100%: the rotation test's weakness on
persistent conditions, amplified by the correction. Removed. A second
version gave every variant one common bandwidth for its own statistic too,
sized for the most persistent variant. That drowned the short-lived
variants in noise and put power below Bonferroni's. Now each variant keeps
its own statistic, and the common bandwidth is used only for the
correlations.

**On real data.** The same question on 20 years of daily bars for the ten
assets of the placebo study above. Each placebo is a whole grid of 15
signals built from one hidden random process, generated independently of
the prices (smoothed over 5/20/60 bars and cut at five thresholds), so the
signals are correlated with each other the way a real grid is and predict
nothing. Tested at horizons 1, 5 and 20, that's 45 variants per grid, and
200 grids per asset.

```bash
python scripts/placebo_family_real.py --placebos 200   # ~15 min, needs network
```

| asset | no correction | Bonferroni | Holm | Benjamini-Hochberg | **TokIO `check_many`** |
|---|---:|---:|---:|---:|---:|
| SPY | 35.0% | 0.0% | 0.0% | 1.5% | 2.5% |
| QQQ | 39.5% | 0.5% | 0.5% | 1.0% | 1.5% |
| IWM | 42.0% | 1.5% | 1.5% | 1.5% | 2.0% |
| TLT | 42.0% | 0.0% | 0.0% | 0.0% | 1.5% |
| GLD | 41.5% | 1.0% | 1.0% | 2.0% | 3.0% |
| USO | 44.0% | 0.0% | 0.0% | 0.0% | 2.5% |
| AAPL | 38.5% | 0.0% | 0.0% | 2.0% | 5.0% |
| JPM | 43.5% | 0.0% | 0.0% | 1.0% | 2.0% |
| XOM | 34.5% | 0.5% | 0.5% | 1.0% | 1.5% |
| EURUSD=X | 41.0% | 1.5% | 1.5% | 2.0% | 4.5% |
| **pooled** | **40.2%** | **0.5%** | **0.5%** | **1.2%** | **2.6%** |

On real returns the gap between the corrections widens. Holm, valid at
5%, actually runs at 0.5%, ten times stricter than promised; that is power
thrown away on every grid. `check_many` runs at 2.6% pooled and never
above 5% on any asset. With no correction, four grids in ten come out with
a "significant" signal that predicts nothing.

## Finished backtests: `check_backtest`

`check_backtest(pnl, trials=...)` tests a strategy's per-bar returns for a
positive mean, with a Bartlett HAC variance, and corrects for the variants
tried: one-sided Romano-Wolf when every variant is passed, Šidák when only a
count is. Two studies, reproducible with `scripts/calibration_backtest.py`
and `scripts/placebo_backtest_real.py`.

### Simulated P&L with no edge

GARCH(1,1) returns with Student-t(4) shocks. P&L shapes: random ±1
positions held ~1, 20 or 120 bars; P&L with its own AR(1) of ±0.2; and
"tranches", the average of the last h bars' returns (what a trade log looks
like when each day's h-bar trade return is listed daily); plus a
volatility-regime switch. 400 paths per row, at 500 and 2,520 bars.

| P&L shape | n | plain t-test | `check_backtest` |
|---|---:|---:|---:|
| positions, run 1 / 20 / 120 | 500 | 4.5 / 4.8 / 5.2% | 4.8 / 4.5 / 5.2% |
| positions, run 1 / 20 / 120 | 2520 | 5.8 / 3.5 / 6.0% | 5.5 / 3.2 / 5.8% |
| AR +0.2 / −0.2 | 500 | 11.0 / 1.2% | 7.0 / 3.8% |
| AR +0.2 / −0.2 | 2520 | 9.8 / 1.2% | 6.0 / 5.2% |
| tranche 5 / 20 | 500 | 20.2 / 37.8% | 4.5 / **8.0%** |
| tranche 5 / 20 | 2520 | 18.8 / 38.0% | 3.8 / 4.0% |
| volatility regimes | 500 / 2520 | 4.5 / 4.8% | 4.2 / 5.0% |

The worst row, 8.0%, is a 20-bar moving average observed for only 500 bars:
about 25 independent blocks. It's disclosed rather than tuned away.

**What broke on the way.** The first version used the Newey-West
rule-of-thumb window (8 lags at 2,520 bars). Tranche-20 P&L then passed
**15-25%** of the time. The window now also takes Andrews' (1991) AR(1)
plug-in, which reads persistence from the P&L's own lag-1 autocorrelation:
15% → 5.5% at 2,520 bars, but still 10.5% at 500 bars, where the window is a
large share of the sample and normal critical values are too small. Kiefer &
Vogelsang's (2005) fixed-b critical value (a cubic in b = window/n,
checked here by simulation at b = 0.02-0.3) brought that to 8.0%.

**Selection.** k strategies on one simulated market, best one reported.
"Independent" = unrelated random positions; "lookback" = one trend rule on
an unrelated random walk at k lookbacks from 5 to 200 (neighbours hold the
same position most days). 200 paths per row. Power = a real Sharpe of 1.0
planted in the middle variant.

| grid | no correction | Šidák | `check_backtest` | power: Šidák | power: `check_backtest` |
|---|---:|---:|---:|---:|---:|
| independent, k=20 | 68.0% | 6.0% | 6.0% | 64.0% | 64.0% |
| independent, k=100 | 99.0% | 6.0% | 6.0% | 40.0% | 40.0% |
| lookback, k=20 | 30.0% | 2.0% | 3.5% | 69.0% | **77.0%** |
| lookback, k=100 | 53.5% | 1.5% | 6.0% | 54.5% | **64.5%** |

On unrelated variants the two corrections agree, as they should. On one
idea tried at many settings, Šidák runs well under 5% and gives up power;
the correlation-aware step-down finds the real edge 8-10 points more often.

### Real market data

Ten assets, 20 years of daily bars (the same set as the `check()` placebo
study). Placebo strategies take random long/short positions independent of
prices, so their expected P&L is zero even on an asset that rose. 500
placebos per asset per shape; 100 grids of 50 per asset.

| placebo | plain t-test | `check_backtest` | with `positions=` |
|---|---:|---:|---:|
| positions, run 1 | 5.3% | 5.4% | 5.4% |
| positions, run 20 | 3.8% | 5.0% | 4.9% |
| positions, run 120 | 4.4% | 5.9% | 5.4% |
| overlapping 20-day trades | 30.5% | 5.4% | — |
| best of 50 | 83.5% | 4.2% (Romano-Wolf) | — |

Pooled across the ten assets. The weak spot is single stocks and oil with
120-bar positions: AAPL 9.6%, USO 8.2%, QQQ 8.2% without `positions=`. A
long block that happens to sit through a rally makes the P&L persistent in
a way its lag-1 autocorrelation barely shows. Passing the positions sizes
the window to the holding run (the same guard `check()` uses for a
persistent condition) and brings those to 6.4%, 4.2% and 7.0%. Worst single
asset with `positions=` on any single-strategy placebo: 7.8% (IWM, run 1,
where the t-test is also at 8.0%, so it's that asset's path, not the
window). Worst asset for best-of-50: USO at 8.0% of 100 grids.

### Costs

`costs=` changes the P&L, not the test: each bar is charged `costs x
|position change|` and the calibrated test runs on what's left, so none of
the rates above move. The breakeven cost is found by bisection on the full
corrected verdict. It's a boundary rather than a guaranteed maximum,
because the lag window is data-driven and the verdict isn't strictly
monotone in the cost.

## Overfitting: the probability that the in-sample winner doesn't persist

`probability_of_overfitting` (and `check_backtest` on any grid of 4+
variants) runs Bailey, Borwein, López de Prado & Zhu's combinatorially
symmetric cross-validation: 16 contiguous blocks, every one of the 12,870
ways to call half of them in-sample, and in each, where the in-sample winner
ranks out of sample. PBO is the share of splits where it lands at or below
the median.

It has no alpha, so the question isn't a false-positive rate but whether
its reading can be trusted in both directions. `scripts/calibration_pbo.py`,
300 grids of 50 variants per row, 2,520 bars of GARCH-t4 returns:

| grid | planted Sharpe | mean PBO | PBO < 0.2 | PBO ≥ 0.4 | planted variant picked most |
|---|---:|---:|---:|---:|---:|
| independent | 0 | 0.50 | 3.3% | 67.7% | — |
| independent | 0.5 | 0.43 | 10.0% | 52.7% | 31% |
| independent | 1.0 | 0.26 | 43.3% | 19.0% | 82% |
| lookback | 0 | 0.50 | 9.7% | 67.3% | — |
| lookback | 0.5 | 0.35 | 24.3% | 36.0% | 62% |
| lookback | 1.0 | 0.11 | 81.3% | 2.0% | 98% |

On noise it centres on 0.5, as it should. But it is a noisy instrument.
On grids of near-duplicate variants it reads a falsely reassuring "below
0.2" about one time in ten, and a real Sharpe of 0.5 barely moves it. So
TokIO reports it next to the p-value, never instead of it, and says so in
the output when it reads low.

**What was left out.** The paper also regresses the winner's out-of-sample
Sharpe on its in-sample Sharpe ("performance degradation"). Each split's
two halves are complements, so any one strategy's two half-sample Sharpes
average to its full-sample Sharpe, and the slope is pushed towards −1
whatever happened. On a grid whose winner had a real edge in every split it
came out at −0.33, which reads as severe degradation of an edge that didn't
degrade. TokIO reports the winner's median in-sample and out-of-sample
Sharpe instead.

## The Deflated and Probabilistic Sharpe Ratios

`scripts/calibration_sharpe.py`, 400 paths per row (200 for grids), the same
null P&L shapes and grids as `check_backtest`'s study above. A strategy
passes when PSR/DSR ≥ 0.95, the papers' threshold; a calibrated statistic
passes 5% of the time.

One strategy, no search (PSR):

| P&L shape | n | as published | `dependence=True` |
|---|---:|---:|---:|
| random positions, 1 bar | 500 / 2520 | 4.5 / 5.8 | 4.8 / 5.8 |
| random positions, 20 bars | 500 / 2520 | 4.8 / 3.5 | 5.0 / 3.0 |
| random positions, 120 bars | 500 / 2520 | 5.5 / 6.0 | 5.2 / 6.0 |
| AR(+0.2) | 500 / 2520 | 11.0 / 9.5 | 7.2 / 6.0 |
| AR(−0.2) | 500 / 2520 | 1.5 / 1.2 | 3.8 / 5.2 |
| overlapping 5-bar tranches | 500 / 2520 | **20.5 / 18.8** | 4.8 / 4.0 |
| overlapping 20-bar tranches | 500 / 2520 | **37.8 / 38.5** | 7.8 / 5.0 |
| volatility regimes | 500 / 2520 | 4.0 / 4.8 | 4.2 / 5.0 |

The published PSR's variance, (1 − γ₃·SR + (γ₄−1)/4·SR²)/(n−1), corrects
for skew and kurtosis but assumes independent observations. A P&L that is a
moving average of overlapping holdings has a long-run variance many times its
plain variance (15 to 30 times, measured, for 20-bar tranches), and the PSR passes noise 38%
of the time. `dependence=True` multiplies the variance by the Bartlett HAC
long-run variance ratio, with the same window and fixed-b widening as
`check_backtest`; its worst row is 7.8%, at n=500, where `check_backtest`'s
own HAC test also reads 8.0%.

Best of k variants, 2520 bars (DSR; Romano-Wolf for comparison):

| grid | false positives: DSR / dep. / RW | power, Sharpe 1.0 planted: DSR / dep. / RW |
|---|---:|---:|
| independent, k=20 | 0.0 / 0.0 / 6.0 | 21.5 / 21.0 / **64.0** |
| independent, k=100 | 0.5 / 0.5 / 6.0 | 10.0 / 11.5 / **40.0** |
| lookback grid, k=20 | 0.5 / 0.5 / 3.5 | 36.0 / 35.0 / **77.0** |
| lookback grid, k=100 | 0.5 / 0.5 / 6.0 | 36.5 / 37.0 / **64.5** |

"Power" counts the planted variant only when it is the one tested and it
passes. The DSR's expected-maximum benchmark grows with the number of trials
and the spread of their Sharpes; on these grids it sits well above where a
5% test would, so the DSR is safe but blunt. Romano-Wolf's critical value
comes from the joint distribution of the variants' statistics, so it charges
correlated variants less and finds the real edge two to four times as
often. `check_backtest` therefore takes its verdict from Romano-Wolf and
reports the DSR beside it.

## Binary contracts: `check_contracts`

`scripts/calibration_contracts.py`. Books of contracts priced exactly at
breakeven, so the expected P&L is zero. 1,000 books per row.

| book | n | t-test | `check_backtest` (HAC) | `check_contracts` |
|---|---:|---:|---:|---:|
| 80c | 30 / 100 / 300 / 1000 | 13.4 / 8.8 / 6.9 / 4.7% | n/a / 6.9 / 5.9 / 4.2% | 5.6 / 5.5 / 4.4 / 3.6% |
| 90c | 30 / 100 / 300 / 1000 | 17.9 / 11.2 / 6.5 / 7.1% | n/a / 8.0 / 6.6 / 7.2% | 4.8 / 2.4 / 4.3 / 5.3% |
| 95c | 30 / 100 / 300 / 1000 | 23.3 / 13.7 / 7.1 / 6.3% | n/a / 12.8 / 9.6 / 6.7% | 0.0 / 4.4 / 3.7 / 3.9% |
| 98c | 30 / 100 / 300 / 1000 | 54.4 / 13.9 / 16.6 / 5.3% | n/a / 13.9 / 16.2 / 6.2% | 0.0 / 0.0 / 2.2 / 3.1% |
| 85-99c | 30 / 100 / 300 / 1000 | 10.3 / 10.3 / 6.5 / 5.1% | n/a / 9.3 / 5.9 / 5.3% | 0.0 / 3.7 / 2.9 / 4.2% |
| 85-99c, sizes 1-50 | 30 / 100 / 300 / 1000 | 22.3 / 10.9 / 7.6 / 7.0% | n/a / 10.5 / 7.1 / 6.7% | 0.4 / 5.1 / 4.7 / 5.4% |

The exact test can't exceed 5% except by Monte Carlo noise in the table
(±0.7 points at 1,000 books). It runs below 5% where the outcome is coarse:
at 98c over 100 bets, no record at all is significant, because even 100 wins
out of 100 happens 13% of the time with no edge. That's not conservatism to
be tuned away; it's what that many bets at that price can tell you.

**What broke on the way.** Sizes differ bet to bet, so the first version
used a saddlepoint approximation for mixed sizes. It matched the exact
answer to 3-4 digits for equal sizes, down to p = 1e-62, and then came out
**36% too small** on 40 bets with sizes 1-30 (0.071 against 0.111
simulated, 400,000 draws): with few bets and uneven sizes the P&L is lumpy,
and the approximation smooths over the gaps. It errs toward significance,
the dangerous direction. Integer sizes now go through an exact recursion
over contracts won (checked against brute-force enumeration and against
simulation); the saddlepoint is kept only for books too large for that,
where it is accurate.

**Stop-losses and early exits.** `scripts/calibration_stops.py`. Fair
15-minute markets: the underlying is a random walk and the contract's price
is its true probability of finishing in the money, so the price is a
martingale. A favourite is bought 5 minutes before close at 85-98c plus a
0.3c fee, with a stop 15c or 30c below entry, checked once a minute or once
a second; a check that finds the price below the stop sells there. 400 books
per row.

The first design used the fact that a stop at L on a continuous martingale
ends at exactly L or 1, reaching 1 with probability (q - L)/(1 - L): a
two-point bet, and the exact recursion still applies. It failed:

| stop | checks | bets | two-point null | bounded null (shipped) |
|---|---|---:|---:|---:|
| −15c | 1/min | 100 / 500 | 12.0 / 8.7% | 0.7 / 0.0% |
| −15c | 1/sec | 100 / 500 | 12.7 / 8.3% | 0.3 / 0.0% |
| −30c | 1/min | 100 / 500 | 13.0 / 7.0% | 3.0 / 0.7% |
| −30c | 1/sec | 100 / 500 | 13.0 / 6.7% | 1.7 / 1.3% |

(Both columns on the same 300 books per row. A second run of the shipped
version, 400 books per row, gave 0.0-2.0%; with no stops, 2.0-2.8%.)

Even checked every second, real exits land below their stop, and near
expiry the price jumps straight through it to 0. The exits spread between
0, sub-stop prices and 1, and that spread is wider than two points. A gap
costs money on average and still inflates the tail.

What ships makes no assumption about the exit rule. If the market is fair,
every position is worth its breakeven on average whatever the rule, and its
final value is between 0 and 1. The win/lose bet with that mean dominates
every such value in convex order (Hoeffding 1963), so the realized values
are tested against the no-exit null. It is conservative: over 500 bets, a
real 2c edge was found 36.5% of the time without stops and 28.2% / 29.0%
with stops at −15c / −30c. A stop that really cuts risk gets no credit.

**Rare-loss P&L without prices.** The same shape breaks `check_backtest`,
and without prices there is no exact null. It now warns when the P&L's
skewness is below −2. On fairly priced books the warning fired on 97.5% of
90c books, all 95c and 98c books, and 1% of 80c books; on symmetric
fat-tailed (t4) P&L it fired on 3%.

## What is still not handled

- **Multiple testing you don't tell it about.** `check_many` and
  `check_backtest` correct for the grid you pass them (or the `trials`
  count you give), and the `TestLedger` for every hypothesis in one
  conversation. None of them can see the variants you tried and quietly
  dropped. Pass the whole grid, including the ones that looked bad.
- **Multiple testing across sessions.** The `TestLedger` corrects for every
  hypothesis in one conversation. Start a new chat and the counter resets,
  so testing 20 ideas across 20 chats dodges the correction entirely.
- **Universe and selection bias.** Testing one condition on a ticker you
  chose *because* you already noticed something there is not corrected by
  anything here, and cannot be.
- **The rotation null assumes stationarity.** A structural break mid-sample
  (a regime change, a company that behaves like two different companies
  before and after) violates it.
- **Artificial junctions.** A circular shift joins the end of the series
  to its start. `check()` also drops bars with missing data, which joins the
  bars on either side of every gap. A quick test with one 50-bar gap
  (GARCH, momentum condition, 200 paths) measured 6% vs 7% without the gap,
  so the effect looks negligible. But many gaps haven't been tested.
- **Correlated contracts.** `check_contracts` assumes outcomes are
  independent across bets. Several fills on one contract can be merged
  (`groups=`); different contracts that resolve on the same underlying
  move can't be, so test them separately or keep one bet per window.
- **Backtest bugs.** `check_backtest` takes the P&L as given. Lookahead,
  survivorship, or fills you couldn't have got are upstream of any test; a
  HAC can't see them. `costs=` charges a flat cost per unit traded; market
  impact that grows with size isn't modelled.
- **Ten assets are not every market.** The real-data placebo study covers
  daily equities, bonds, gold, oil and FX. Intraday bars, crypto and very
  illiquid assets are untested. USO is the reminder: one asset with an
  unusual property broke an engine that passed every simulation.
