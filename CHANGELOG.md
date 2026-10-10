# Changelog

## Unreleased

- Model-written strategies (the agent's `backtest_strategy` tool) run in an
  isolated process: empty environment, OS memory/process limits, and an
  audit-hook allowlist (docs/sandbox.md). **Held back from release until an
  independent security review.**
- A result now says when variants never traded and so don't tighten the
  correction.

## 0.6.0 — 2026-10-09

- **`tokio_ai.Lab`: backtest here, honestly.** Write `signal(data, **params)`
  returning positions; the lab fills them at the next bar's open with
  commission and slippage, refuses signals that use future data (a prefix
  re-run test), counts every variant run and corrects for all of them in
  `lab.check()`, keeps a holdout locked until a single `final_test()`, and can
  test against exposure-matched buy-and-hold (`vs="market"`). Matches
  vectorbt to 1e-16 per bar without fees. On 400 no-edge markets x 14
  variants: best variant's own p < 0.05 in 20.5%, `lab.check()` 2.8%
  (`scripts/calibration_lab.py`).
- `lab.best()` and `lab.final_test(run)`: spend the holdout on the strongest variant.
- Colab notebook: a "let TokIO run it" section (sweep vs the market, a
  lookahead caught, one holdout look).
- `tokio_ai.load_prices(symbol)`: full daily history from Yahoo (SPY from
  1993), split- and dividend-adjusted. `range=max` could come back silently
  downsampled; an explicit date span doesn't.
- Clearer wording when a result's returns already include their costs.

## 0.5.1 — 2026-10-08

- The project moved to github.com/Kuzi-DEV/Tokio-ai (homepage kuzi-dev.github.io/Tokio-ai).
  Old links redirect. No code changes.

## 0.5.0 — 2026-10-07

0.4.0 was committed to git but never tagged or published to PyPI, so 0.5.0 is the first
release to carry everything below and everything in 0.4.0.

- **Lighter install.** `pip install tokio-ai` now installs only numpy; the
  research agent's openai and textual moved to the `agent` extra
  (`pip install "tokio-ai[agent]"`). `tokio-ai` / `tokio-ai-plain` say so if
  they're missing. Result stamps no longer list the openai version except on
  the agent's own output.
- **Deflated Sharpe Ratio, Probabilistic Sharpe Ratio, minimum track record
  length** (`tokio_ai.sharpe`), as published and with `dependence=True`, and
  on every `check_backtest` result. Measured: the published PSR passes a
  zero-edge strategy 38% of the time on overlapping 20-bar trades (4-8% with
  the dependence correction); the DSR on grids finds a planted Sharpe-1.0
  edge 10-37% of the time vs Romano-Wolf's 40-77%.
  `scripts/calibration_sharpe.py`.
- **Robustness section** on every result: Sharpe with both tails trimmed,
  money-making share of 8 time blocks, a one-bar-delay lookahead check (needs
  positions and `asset_returns=`, automatic from the adapters), and the
  minimum backtest length.
- **TradingView.** `tokio_ai.read_tradingview()` reads the Strategy Tester's
  "List of trades" export (CSV or XLSX, old and new formats); the CLI
  recognises exports and treats several as variants. `pine/tokio_check.pine`
  draws the per-trade verdict on the chart, pasted under any v6 strategy.
- **`check_backtest` takes the backtest object itself**: a vectorbt
  `Portfolio` (a multi-column one is a grid of variants, corrected for each
  other) or the stats from backtesting.py's `Backtest.run()`, or a dict of
  them. Returns and per-bar positions are read from it (`tokio_ai.adapters`;
  neither library is a dependency). Tested against vectorbt 1.1.1 and
  backtesting.py 0.6.6.
- **`examples/check_your_backtest.ipynb`**, a Colab notebook: 19 SMA
  crossovers on 20 years of SPY, alone vs corrected, with costs, then
  upload your own CSV. Saved with its outputs, so GitHub's preview shows the
  result without running anything.
- **`tokio_ai.check_backtest(pnl, trials=...)`**: is a finished
  backtest's profit distinguishable from luck? Takes per-bar strategy
  returns (or every variant tried, as a dict / DataFrame) and tests for a
  positive mean with a Bartlett HAC whose window adapts to the P&L's
  persistence (Andrews 1991 plug-in, Kiefer-Vogelsang fixed-b critical
  values), then corrects for the search: one-sided Romano-Wolf across the
  variants passed, or Sidak for a bare `trials` count. Reports the plain
  t-test's p-value alongside, a haircut Sharpe, and how many independent
  trials the result could survive. Optional `benchmark=` (test the excess)
  and `positions=` (sizes the window to the holding run). Worst false-positive
  rate on 16 simulated no-edge P&L shapes: 8.0% (plain t-test 38.0%). On 20
  years of ten real assets: overlapping 20-day trade logs 5.4% (t-test
  30.5%), best of 50 placebo strategies 4.2% (t-test 83.5%).
  `scripts/calibration_backtest.py` and `scripts/placebo_backtest_real.py`
  reproduce it; docs/calibration.md has the account, including the two
  windows that failed first.
- **Costs and the breakeven cost.** `check_backtest(..., positions=,
  costs=0.0005)` charges every position change (5 bps per unit traded here)
  before testing, reports turnover and the annual cost drag, says so when
  costs are what sink a result that was significant gross, and reports the
  breakeven cost: the most you could pay per unit traded and still pass,
  corrected for every trial.
- **`tokio_ai.probability_of_overfitting(variants)`**: the probability of
  backtest overfitting by combinatorially symmetric cross-validation
  (Bailey, Borwein, López de Prado & Zhu 2017), over all 12,870 half-splits
  of 16 blocks. Also reported automatically by `check_backtest` for any
  grid of four or more variants, net of costs. On 20 years of SPY, 19 SMA
  crossovers score 0.91: the in-sample winner is worse than a random pick
  out of sample. Measured on noise grids (mean 0.50, but a falsely low
  reading below 0.2 in 3-10%) and with a planted edge
  (`scripts/calibration_pbo.py`). The paper's degradation-slope statistic
  is deliberately omitted: complementary halves force it towards -1, and it
  read -0.33 on an edge that never degraded.
- **`tokio_ai.check_contracts(prices, outcomes, sizes=, fees=, groups=)`**:
  an exact test for bets on binary contracts (prediction markets, binary
  options, fixed-odds bets). Null: every bet wins with probability equal to
  its breakeven (price plus fee); the p-value is exact (Poisson-binomial
  recursion, over contracts won for integer sizes), with a saddlepoint
  approximation only for books too large for that. Also reports whether
  the book did significantly worse than fair, an exact upper bound on the
  loss rate, and how many bets a perfect record would need. Worst
  false-positive rate on 24 fairly priced books: 5.6% (t-test 54.4%,
  `check_backtest` 16.2%). `scripts/calibration_contracts.py`.
  `tokio-ai-backtest --contracts PRICE OUTCOME [--sizes] [--fees] [--groups]`
  runs it on a CSV.
- **Robustness margin on every `check_contracts` verdict**: how far the
  true cost per contract could be from the prices and fees given before a
  significant result (either tail) flips, with a warning under 1c. Found by
  using the tool on a real book: a "significantly worse than fair" finding
  (p = 0.0003) came from recording limit prices instead of fill prices, which
  differed by about 1c. The margin on that wrong analysis was 0.91c.
- **Early exits in `check_contracts`**: `exit_values=` takes what each
  stopped-out (or otherwise closed) bet returned, net of fees. Works for any
  exit rule. Tested against the no-exit win/lose null, which bounds every
  exit rule in convex order: false-positive rate at most 3.0% on simulated
  fair 15-minute markets with stops (`scripts/calibration_stops.py`). A
  sharper two-point stop null was tried and dropped at 7-13%. CLI:
  `--exits COLUMN`.
- `check_backtest` warns when the P&L's skewness is below -2 (favourites,
  sold options): estimated-variance tests over-fire there.
- **`tokio-ai-backtest` command**: the same test on a CSV, no Python:
  `tokio-ai-backtest pnl.csv --trials 40` (`--equity` for equity curves,
  `--benchmark COL`, `--columns ...`). A numeric column with stray text
  cells is refused rather than silently dropped, since dropping it would
  also undercount the trials.
- **numpy is now a required dependency.** `check_many` already needed it
  and failed with an ImportError on a plain `pip install tokio-ai`.
- `check_many`'s adjusted p-value can no longer come out a hair below the
  variant's own p-value through Monte Carlo noise.
- **`tokio_ai.check_many(returns, conditions, horizon=[...])`**: test a whole
  grid of variants as one family, controlling the chance of even one false
  discovery anywhere in it. Uses the correlation between variants
  (Romano-Wolf step-down against the joint normal of their Hodrick
  statistics), so near-duplicate variants aren't punished like unrelated
  ones. It's never more conservative than Holm. On a 30-variant grid with no
  edge, no correction flags something 42.5-46.0% of the time; `check_many`
  3.5-4.0%, Holm 1.5-2.5%, with 30-100% more power than Holm on the
  variant that separates them. `scripts/calibration_family.py` and
  `scripts/placebo_family_real.py` reproduce it.
- **Agent tool `test_pattern_grid`**: the chat agent tests every
  threshold x horizon of a condition as one `check_many` family, and its
  system prompt now requires one grid call instead of a series of single
  tests when the question is "which works best". Every grid variant is
  also recorded in the session ledger, so a variant that fails the grid
  can't be re-tested alone as a fresh "first" test. The grid output always
  states whether the rotation second opinion agrees; in a live run the
  agent had claimed an agreement that the output didn't contain.
- **Faster long-run covariance.** `bartlett_long_run_cov` now works in the
  frequency domain (the Bartlett-weighted cross-covariance equals the
  cross-spectrum weighted by the Fejér kernel): one FFT per series and one
  matrix product, instead of an inverse FFT per pair. Tested equal to the
  direct lag sum to 1e-12. A 30-variant grid takes roughly 0.3-0.5 s at
  5,000 bars and 4-5 s at 100,000, depending on hardware.
- **Agent system prompt:** the unequal-variance rule said the condition
  selects "volatile" days, even when it selects calm ones (the same bug
  fixed in the verdict text in 0.4.0).
- **Refactor:** `rigor/overlap.py` exposes `hodrick_terms` and
  `bartlett_long_run_cov`, shared by `check` and `check_many`.

## 0.4.0 — 2026-09-24 (git only)

- **`tokio_ai.check(returns, condition, horizon)`**: the calibrated
  engines as a plain library call on your own data (lists, numpy or
  pandas). No agent, API key or network, and it doesn't import the LLM
  client or the TUI. Outcomes start at bar i+1, so a condition can't
  predict its own bar. Pandas inputs with mismatched indexes raise an
  error instead of pairing by position. An optional `ledger=` applies
  Benjamini-Hochberg correction across checks.
- **`scripts/calibration_check.py`**: false-positive rates on fat tails,
  volatility regimes, a persistent momentum condition and autocorrelated
  returns, next to a Welch t-test. Worst case across 51 configurations:
  TokIO 7.7% (default engine) and 7.0% (rotation), t-test 61.0%. See
  [docs/calibration.md](docs/calibration.md#check-on-harsher-nulls).
- **Exact and fast with numpy.** The circular-shift test now evaluates
  every rotation at once through an FFT (a circular cross-correlation,
  zero-padded to a power of two so arbitrary lengths stay fast). It used to
  sample 5,000 rotations. It's exact at any n: 0.07 s at 100k bars
  (previously 20.6 s) and 1.2 s at 1M. `check()` has a matching vectorized
  preparation path. Without numpy, the pure-Python path still runs and is
  the reference that the fast path is tested against.
- **New default engine: Hodrick (1992) standard errors + a short HAC**
  (`rigor/overlap.py`). Overlapping windows are handled exactly by
  regrouping the statistic by bar; a Bartlett kernel with the Newey-West
  rule-of-thumb bandwidth covers the returns' own short-range
  autocorrelation. The rotation test still runs on every call as a second
  opinion, and `__str__` flags it when they disagree. `method="rotation"`
  makes it the verdict. The agent's `test_return_pattern` tool now goes
  through `check()`, so the agent and the library agree.
- **`scripts/benchmark.py`**: head-to-head size and power against
  Newey-West (statsmodels) and a stationary block bootstrap (arch) on 51
  null configurations, including autocorrelated returns. Worst-case size:
  TokIO 8.3% (a 60-path row; 7.7% where rows have ~200 paths), Newey-West
  15.0%, bootstrap 15.0%, t-test 64.0%, at equal power (73.5% vs 74.2%).
  The default engine was chosen by this benchmark, after a minimum-shift
  rotation variant and a Cauchy combination both lost. docs/calibration.md
  has the full account.
- **`scripts/placebo_real.py`**: false-positive rates on real market data.
  20,000 random signals, independent of prices by construction, tested on
  20 years of daily bars for ten assets. Pooled worst: t-test 64.1%,
  Newey-West 10.4%, TokIO 5.1%.
- **HAC bandwidth covers the condition's time scale.** The placebo study
  found the default engine at 11-12% on USO, whose returns trend over months.
  The bandwidth is now max(Newey-West rule of thumb, mean run length of the
  condition + h), which brought USO to ~6% with no measurable cost in the
  simulations. The autocovariances are computed by FFT, so long bandwidths
  stay fast.
- **`.github/workflows/publish.yml`**: PyPI Trusted Publishing on version
  tags, gated on the test suite and a tag/version match. No API token.
- **Fixed: the agent was dead for every new user.** NVIDIA retired the
  default model (`llama-3.3-nemotron-super-49b-v1.5`) on 2026-08-26, and
  every request returned HTTP 410. The new default is
  `nvidia/nemotron-3-super-120b-a12b`, picked by running the same real
  tasks on 7 free-tier candidates (details in `agent/loop.py`).
- **Fixed:** the unequal-variance note said a condition selected
  "volatile" days even when it selected calm ones. It also claimed a
  naive test would always overstate significance. Pooling overstates only
  when the smaller group is the noisier one. Both messages now report the
  real direction.

## 0.3.0 — 2026-08-17

The rigor engine was measured against its own promise for the first time,
failed, and was rewritten. See [docs/calibration.md](docs/calibration.md)
for the full study; `scripts/calibration_study.py` reproduces it with no API
key and no network.

A test that reports `p < 0.05` should be wrong 5% of the time on data with
nothing in it. On simulated price series with no predictable structure
whatsoever, v0.2.0 was wrong up to **45%** of the time (worst case now 7.0%). Two compounding
causes, both specific to what this tool is for:

- **Studentized permutation statistic** (breaking change to p-values).
  `permutation_test` permuted a raw mean difference, which is only valid
  when the two groups are exchangeable. Every condition worth testing here
  ("days that fell 2%", "days with unusual volume") selects volatile days by
  construction, so the condition group is small and much noisier than
  baseline — measured at 1.75x the variance and 15x fewer observations.
  Pooling those makes the small group's mean look far more stable than it
  is. Now permutes a Welch-style studentized statistic, which stays valid
  under unequal variances (Chung & Romano 2013).
- **New `circular_shift_test` for time-ordered data.** Forward returns over
  a multi-day horizon come from overlapping windows and are heavily
  autocorrelated; conditions cluster in time as well. Shuffling labels
  assumes neither is true. Rotating the label series against the value
  series preserves both structures exactly. On a synthetic case with no
  relationship at all, the old approach returned p=0.00033 and the new one
  p=0.49.
- `test_return_pattern` now uses the rotation test. Because only the
  condition-met days move under a rotation, every distinct rotation is
  enumerated rather than sampled — p-values are **exact**, not Monte Carlo,
  and the test suite got ~2.7x faster.
- **The README's own showcase example was a false positive.** "AAPL gaps
  above 2% fade over the next 5 days, p=0.0042" re-runs at **p=0.089** on
  the same real 10-year window. Replaced with the corrected result.
- Verdicts now report which method produced them, and warn explicitly when
  the two groups' variances differ enough that a naive test would have
  overstated the result.
- `PermutationResult` gains `statistic` and `variance_ratio`. Both are
  persisted; older saved chats load unchanged.

Also fixed, unrelated to the above:

- **The plain REPL crashed on Windows whenever a reply contained a
  character outside the console codepage** — an arrow, an en dash, a
  "greater than or equal" sign, all routine in model output. `print()`
  raised `UnicodeEncodeError`, and in `cli.py` that call sits outside the
  try/except around the API call, so one punctuation mark ended the session
  and lost the conversation. Both entry points now force UTF-8 stdio with
  `errors="replace"`. Found by running the agent for real, not by reading
  the code; this is the third time this project's default-codepage
  assumption has caused a bug.
- **The provenance stamp reported the installed version, not the running
  one.** With a source tree ahead of the last `pip install` — the normal
  state while developing — verdicts were stamped with a version of the code
  that did not produce them, which defeats the point of stamping them. Now
  reports the running `__version__`, noting the installed version when they
  disagree.

- **Chat saves are now atomic.** `save_chat` truncated the real file before
  writing, so a crash or Ctrl+C mid-save destroyed the conversation.
  Writes to a temp file and `os.replace`s it. (`list_chats` already had to
  skip unparseable files — that was the symptom.)
- Loading a chat tolerates unknown fields in a saved result instead of
  raising, so a chat written by a newer TokIO still opens in an older one.
- Persisted result fields derive from the dataclass, so a new field can no
  longer be silently dropped from saved chats.
- Chat listing iterates in sorted order, for stable output.
- Packaging: homepage now points at the docs site, added a Documentation
  URL, Python 3.14 and an Information Analysis classifier.

## 0.2.0 — 2026-08-02

New feature release: a full-screen terminal UI, now the default `tokio-ai`
experience.

- **New `tokio_ai.tui` module**, built with [Textual](https://textual.textualize.io/):
  a dark-themed screen with a "TOKIO AI" banner, a scrollable bordered chat
  log, and an input box pinned at the bottom. Agent calls run in a
  background worker so the UI stays responsive during slower (free-tier)
  LLM round trips.
- `tokio-ai` now launches the TUI by default. The previous plain-text REPL
  moved to `tokio-ai-plain` (still reachable via `python -m tokio_ai.cli`)
  for scripting, piping, or terminals that can't render a full-screen app.
  Pure presentation layer -- both entry points share the same underlying
  `Agent` class, no logic duplicated.
- Verified with Textual's headless pilot test framework (no real terminal
  or network needed) plus a rendered SVG snapshot to visually confirm the
  layout.
- Fixed a real legibility bug caught in review: the initial hand-drawn
  ASCII-art banner rendered the letter "I" as a plain vertical bar,
  indistinguishable from a "T" at a glance. Replaced with a proper figlet
  font where I renders as a distinct slanted stroke.

## 0.1.1 — 2026-08-02

Bug-fix release. A deep-dive audit (manual pass + an independent code-review
pass) found and fixed 7 real bugs, two of which shipped in 0.1.0:

- **`permutation_test` could report `p=0.0` exactly.** With finite Monte
  Carlo resampling that overclaims certainty -- fixed with the standard
  plus-one correction, `(hits+1)/(iters+1)` (Phipson & Smyth 2010).
- **Windows encoding bug, live in 0.1.0**: reading the bundled S&P 500 CSV
  without `encoding="utf-8"` corrupted names like "Brown-Forman" into
  mojibake under Windows' default cp1252 codepage. CI now also runs on
  `windows-latest` (previously Linux-only, which is why this shipped
  unnoticed).
- `TestLedger.verdict()` returned the *first* match by hypothesis name, not
  the latest -- reusing a name silently reported a stale result forever.
- `recent_filings(limit=0)` returned 1 result instead of 0.
- `top_performers(top_n=-N)` silently returned "all but the last N" via
  Python slice semantics instead of erroring.
- Agent conversation history could end up inconsistent after an API
  failure mid-turn, or after exhausting the tool-call round limit.
- `.env` parsing broke on quoted values (`KEY="value"`) and had the same
  missing-encoding bug as the CSV reader.

Also: `top_performing_stocks` now surfaces an explicit warning when data
yield drops below 90%, instead of relying on the caller to notice by
comparing two numbers. Test suite grew from 44 to 57 tests, all with
regression coverage for the bugs above.

## 0.1.0 — 2026-08-02

Initial release: price history and SEC filings lookups, a real S&P 500
screener with GICS sector filtering, technical-pattern hypothesis testing,
and the core rigor engine (permutation testing + Bonferroni/
Benjamini-Hochberg multiple-testing correction). Agent runs against any
OpenAI-compatible endpoint, defaulting to NVIDIA's free NIM catalog.
