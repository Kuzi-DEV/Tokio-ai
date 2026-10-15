# Case studies: famous market claims, tested honestly

Each study fixes its tests before the first run, counts every variant it tries, and
holds out years nobody looked at while designing it. Every number below comes from a
script in [`examples/case_studies/`](../examples/case_studies/) using only free data;
run it and you get the same table.

The point isn't to find a trade. It's to show what each claim looks like once the
statistics stop flattering it. Most don't survive, and that's the useful result.

| Claim | Verdict |
|---|---|
| Put/call ratio spikes predict the market | **No.** 0 of 18 variants survive, in either era |
| Selling VIX futures harvests a risk premium | **It does earn, but it's just stock exposure in disguise:** beta 0.66 to SPY, negative alpha |
| Most of the stock market's return happens overnight | **True,** but after costs it trails simply holding SPY |

---

## 1. The put/call ratio as a contrarian signal

`python examples/case_studies/putcall_sentiment.py`

**The claim.** When traders pile into puts, fear peaks and the market rebounds; when
nobody buys puts, complacency precedes a drop. The ratio is on every sentiment
dashboard.

**The test.** Cboe's daily equity, index and total put/call volume
(`tokio_ai.load_put_call`), 2006 to today. "Fear" = the log ratio more than 1.5
standard deviations above its own past year, "calm" = 1.5 below. Each day's ratio is
used the following day, because it's published after the close. Forward SPY returns are
checked over 1, 5 and 20 days: 3 ratios x 2 conditions x 3 horizons = **18 variants,
one family** (`tokio_ai.check_many`).

**Result, 2007-2019:** 0 of 18 survive. Two look significant alone, which is what 18
tries produce by chance:

| variant | gap vs all days | p alone | p for the family |
|---|---|---|---|
| index fear, 20 days | +1.19% | 0.013 | 0.17 |
| index calm, 20 days | -1.41% | 0.036 | 0.39 |

**2020-2026, same 18 variants:** 0 survive. The two leaders faded: +0.88% (p 0.25) and
-0.04% (the sign is gone).

**Takeaway.** The one variant someone would have published, "index fear predicts a 20-day
rebound, p = 0.013", is exactly the kind of result a grid search manufactures.

---

## 2. Selling VIX futures (the volatility risk premium)

`python examples/case_studies/vix_futures_premium.py`

**The claim.** Investors overpay for crash insurance, so whoever sells it (VIX futures,
puts) collects a premium. This one has a real mechanism and a real literature behind it.

**The data.** Cboe publishes every VX contract's daily settlement price for free. The
script builds a constant 30-day VIX futures position from the two contracts around 30
days to expiry. Each day's change is measured on the same contracts, so there are no
false jumps from rolling. Checked against history: on 2018-02-05 the position rose 97%
and a 1x short lost **99.6%**, which is the day the XIV fund ended.

**Results.**

1. *Is there a premium?* A 1x short, before costs, 2014-2020: Sharpe 0.26, p = 0.24.
   Not distinguishable from luck, and at 1x it doesn't survive February 2018 at all.
2. *Does timing help?* Five well-known signals (contango, VIX/VIX3M, VIX9D/VIX) at 1
   and 5 days: 0 of 10 survive.
3. *Sized small (0.2x) with costs, held-out 2021-2026:* +8.9%/yr, Sharpe 0.66,
   max drawdown -22%, p = 0.036. It looks like a strategy.
4. *But does it add anything to holding SPY?* Correlation 0.72, beta 0.66, **alpha
   -3.3%/yr**. Mixing it into an SPY portfolio lowers the Sharpe: 0.84 -> 0.78 at 25%,
   0.68 at 50%.

**Takeaway.** "Profitable" and "worth doing" are different questions. Short vol earns
mostly because it's long the stock market, and it does so with worse tails than just
buying the stock market.

---

## 3. The overnight premium

`python examples/case_studies/overnight_premium.py`

**The claim.** Nearly all of the stock market's gains happen between the close and the
next open; the trading day itself earns roughly nothing.

**The test.** Hold SPY only from the close to the next open, at 1 bp per round trip,
with the intraday half as the control. Discovery 2000-2018, holdout 2019 on.

| | return/yr | Sharpe | max drawdown |
|---|---|---|---|
| **2000-2018** overnight, net | +3.4% | 0.37 (p = 0.025) | -36% |
| intraday | -1.2% | 0.01 | -57% |
| buy and hold | +4.8% | 0.34 | -55% |
| **2019-2026** overnight, net | +7.2% | 0.62 (p = 0.043) | -30% |
| intraday | +6.7% | 0.53 | -14% |
| buy and hold | +17.3% | 0.93 | -34% |

**Takeaway.** The effect is real out of sample. In 2000-2018 the intraday half really
did earn nothing. Costs ate 43% of it, though, and since 2019 the trading day has
earned too, so simply holding SPY did far better. It's useful only if you need the
money free during the day.

---

## What went wrong, so you can discount accordingly

- Studies 2 and 3 were first run under a stricter plan whose VIX cost assumption was
  about 5x too high. The page reports the corrected version, which was judged only on
  years the first run never touched.
- While checking data quality for study 3, per-year averages for the holdout years were
  printed before the holdout was formally run. The pass rule was already fixed, so it
  couldn't move, but you should know.
- Judged together with Holm's correction for two tests, neither held-out result in
  studies 2 and 3 clears 5% (0.036 and 0.043).
