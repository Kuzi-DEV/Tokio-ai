"""Is a backtest's profit distinguishable from luck, given how many you tried?

`check` and `check_many` test a *condition*. Most people arrive with a
*backtest*: a column of per-bar strategy returns and a Sharpe ratio they
like. Two things make that Sharpe look better than it is:

1. **The plain t-test on the mean.** It assumes the bars are independent.
   Strategy P&L rarely is: positions held for weeks, returns that trend or
   mean-revert, and fat tails. The variance of the mean is estimated here
   with a Bartlett HAC (the same long-run covariance `check_many` uses), so
   serial dependence in the P&L widens the interval instead of being
   ignored.

2. **Selection.** The Sharpe on the screen is the best of however many
   variants were run. The best of twenty zero-edge strategies has a
   respectable Sharpe by construction. If you pass every variant you tried,
   the correction is the Romano & Wolf (2005) step-down against the joint
   normal of their statistics -- one-sided, since a backtest claims a
   profit -- which credits near-duplicate variants for being near-duplicates.
   If you can only say how many you tried, the correction is Sidak's, which
   treats the trials as independent: the worst case, and the right default
   when the other variants aren't on hand.

The output puts the honest p-value next to the plain t-test's, turns the
corrected p-value into a haircut Sharpe (Harvey & Liu 2015: the Sharpe whose
plain p-value would equal the corrected one), and says how many independent
trials this backtest could have survived.

Requires numpy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from statistics import NormalDist
from typing import Any

from .adapters import is_backtest_object, unpack
from .robustness import diagnose
from .trades import TradeList, on_calendar
from .check import _aligned
from .family import DRAWS, _romano_wolf
from .rigor.overlap import bartlett_long_run_cov, default_bandwidth
from .rigor.provenance import stamp

# Fewer bars than this and a Sharpe estimate is too noisy for any verdict to
# mean much (about three months of daily bars).
MIN_BARS = 60
MIN_PBO_VARIANTS = 4
# Below this skewness the P&L looks like a book of favourites or short
# options: many small wins, rare large losses. Estimated-variance tests
# over-fire there (HAC 12-16% at 98c, 9% at 95c, on fairly priced
# contracts), so the result says so. Ordinary strategy P&L sits near
# -0.5..+0.5; 90c favourites are about -2.7, 98c about -7.
SKEW_WARN = -2.0
# A significant result from a strategy whose average position is at least
# this one-sided gets a note: against a zero-profit null, being long a rising
# market passes on the drift alone. Long-only SMA crossovers on 20 years of
# SPY (mean position ~0.65) pass as a grid, and every one of them fails
# against exposure-matched buy-and-hold.
NET_EXPOSURE_WARN = 0.25

_N = NormalDist()


def _p_greater(z: float) -> float:
    """One-sided p-value for H1: mean > 0."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def andrews_bandwidth(np, d) -> int:
    """Andrews (1991) AR(1) plug-in lag window for a Bartlett kernel.

    The rule of thumb sizes the window from the sample length alone, which
    is fine for returns with short memory and badly wrong for P&L with long
    memory. Measured: a strategy that averages 20 staggered holdings has a
    P&L that is a 20-bar moving average, and at the rule-of-thumb window the
    test fired 15-25% of the time at a nominal 5%. The plug-in reads the
    persistence off the series' own first-order autocorrelation (clipped
    at 0.97 so a near-unit root can't ask for the whole sample).
    """
    m = len(d)
    ss = float(d @ d)
    if m < 3 or ss <= 0:
        return 0
    rho = max(-0.97, min(0.97, float(d[1:] @ d[:-1]) / ss))
    a1 = 4 * rho * rho / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(1.1447 * (a1 * m) ** (1 / 3))


def fixed_b_scale(b: float) -> float:
    """Shrink factor for a Bartlett HAC z-statistic whose window is a fraction b of the sample.

    Standard normal critical values assume the window is a vanishing share
    of the data. When it isn't -- a long-memory P&L over a short sample --
    the HAC variance is itself noisy and the statistic has fatter tails.
    Kiefer & Vogelsang (2005) give the 95% critical value as a cubic in b =
    (L+1)/n; dividing z by cv(b)/1.645 makes the normal p-value exact at
    one-sided 5% and close elsewhere (checked by simulation at the 90%, 95%
    and 99% points for b up to 0.3). At daily data with a short window, b is
    about 0.004 and the factor is 0.99.
    """
    cv = 1.6449 + 2.1859 * b + 0.3142 * b * b - 0.3427 * b ** 3
    return 1.6449 / cv


def _sidak(p: float, trials: int) -> float:
    # 1-(1-p)^N, computed without losing a tiny p to rounding.
    return -math.expm1(trials * math.log1p(-p)) if p < 1 else 1.0


@dataclass(frozen=True)
class StrategyResult:
    name: str
    n_bars: int
    mean: float  # per bar
    sharpe: float  # annualized
    p_naive: float  # plain one-sided t-test, bars treated as independent
    p_alone: float  # HAC, this strategy alone
    p_adjusted: float  # corrected for every trial
    haircut_sharpe: float
    lag1_autocorr: float
    reportable: bool
    skewness: float = 0.0

    def significant(self, alpha: float) -> bool:
        return self.reportable and self.p_adjusted <= alpha

    def survives_trials(self, alpha: float) -> int:
        """How many independent trials this could be the best of and still pass."""
        if not self.reportable or self.p_alone > alpha:
            return 0
        if self.p_alone <= 0:
            return 10**9
        return max(1, int(math.log1p(-alpha) / math.log1p(-self.p_alone)))


@dataclass(frozen=True)
class BacktestResult:
    strategies: list[StrategyResult]
    trials: int
    correction: str  # "none", "romano_wolf" or "sidak"
    alpha: float
    periods_per_year: float
    bandwidth: int
    provenance: str
    notes: list[str] = field(default_factory=list)
    costs: float = 0.0  # per unit of position traded, as a return (0.0005 = 5 bps)
    # The largest cost per unit traded at which the verdict is still
    # SIGNIFICANT, corrected for every trial. None when it can't be computed
    # (no positions) or the result isn't significant even before costs.
    breakeven_cost: float | None = None
    turnover_per_year: float | None = None  # of the strongest variant
    p_before_costs: float | None = None
    # Probability of backtest overfitting (CSCV), computed when at least
    # MIN_PBO_VARIANTS variants and enough bars for 16 blocks are passed.
    overfitting: Any = None
    # Bailey & Lopez de Prado's Deflated Sharpe Ratio (Probabilistic, for a
    # single trial) of the highest-Sharpe variant, net of costs: with the
    # variance scaled for serial dependence, and as published (independent
    # bars). tokio_ai.sharpe.SharpeStats; None when not reportable.
    deflated: Any = None
    deflated_published: Any = None
    bar_unit: str = "bars"
    robustness: Any = None  # tokio_ai.robustness.Robustness of the strongest variant

    @property
    def best(self) -> StrategyResult:
        """The strongest variant: lowest corrected p-value, then highest Sharpe.

        Not simply the highest Sharpe -- with serial dependence the ranking can
        differ, and the verdict must be about the variant closest to passing.
        """
        rep = [s for s in self.strategies if s.reportable] or self.strategies
        return min(rep, key=lambda s: (s.p_adjusted, s.p_alone,
                                       -(s.sharpe if not math.isnan(s.sharpe) else -math.inf)))

    @property
    def significant(self) -> bool:
        return self.best.significant(self.alpha)

    @property
    def p_value(self) -> float:
        return self.best.p_adjusted

    @property
    def verdict(self) -> str:
        if not self.best.reportable:
            return "NOT REPORTABLE"
        return "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"

    def _correction_text(self) -> str:
        if self.trials <= 1:
            return ""
        if self.correction == "sidak":
            return f" after correcting for {self.trials} trials (Sidak, as if independent)"
        return f" after correcting for {self.trials} trials (Romano-Wolf, using their correlation)"

    def _cost_lines(self) -> list[str]:
        b = self.best
        out: list[str] = []
        if not b.reportable:
            return out
        if self.turnover_per_year is not None:
            who = b.name if len(self.strategies) > 1 else "the strategy"
            if self.costs:
                out.append(
                    f"Costs: {self.costs * 1e4:.1f} bps per unit traded; {who} turns over "
                    f"{self.turnover_per_year:.1f}x a year, a drag of "
                    f"{self.costs * self.turnover_per_year:.2%} a year."
                )
            else:
                out.append(f"No extra costs charged (returns are taken as already net of any fees); "
                           f"{who} turns over {self.turnover_per_year:.1f}x a year.")
        if self.p_before_costs is not None and self.p_before_costs <= self.alpha < b.p_adjusted:
            out.append(
                f"Before costs it was significant (p={self.p_before_costs:.4f}). The costs are "
                f"what sink it."
            )
        if self.breakeven_cost is not None:
            corrected = " (corrected for every trial)" if self.trials > 1 else ""
            out.append(
                f"Breakeven: it stays significant up to {self.breakeven_cost * 1e4:.1f} bps per "
                f"unit traded{corrected}. If your real costs, slippage included, are higher, "
                f"it's not evidence."
            )
        return out

    def _sharpe_lines(self) -> list[str]:
        d, pub = self.deflated, self.deflated_published
        if d is None or pub is None:
            return []
        name = "Deflated Sharpe Ratio" if self.trials > 1 else "Probabilistic Sharpe Ratio"
        top = max((s for s in self.strategies if s.reportable), key=lambda s: s.sharpe, default=None)
        of = ""
        if len(self.strategies) > 1 and top is not None:
            of = (f" of {top.name}" if top.name == self.best.name else
                  f" of {top.name} (the DSR tests the highest Sharpe; the verdict's variant is the one "
                  f"closest to passing once dependence is allowed for)")
        line = (f"{name}{of}: {d.psr:.2f} allowing for serial dependence; {pub.psr:.2f} as "
                f"published, which treats {self.bar_unit} as independent (0.95 passes).")
        if pub.psr >= 0.95 > d.psr:
            line += (" Only the published version passes: on overlapping-trade P&L it passes "
                     "a zero-edge strategy 38% of the time instead of 5%.")
        out = [line]
        if d.min_track_record is not None and self.trials <= 1:
            out.append(f"Minimum track record length at 95%: {math.ceil(d.min_track_record)} "
                       f"{self.bar_unit} (have {d.n}).")
        return out

    def __str__(self) -> str:
        b = self.best
        if not b.reportable:
            lines = [f"NOT REPORTABLE: {b.n_bars} usable {self.bar_unit}; a Sharpe ratio needs at least "
                     f"{MIN_BARS} before any verdict means anything."]
        else:
            years = b.n_bars / self.periods_per_year
            who = "" if len(self.strategies) == 1 else f" Strongest variant: {b.name}."
            lines = [
                f"{self.verdict}{self._correction_text()} (p={b.p_adjusted:.4f}, "
                f"alpha={self.alpha}).{who}",
                f"Sharpe {b.sharpe:.2f} annualized over {b.n_bars} {self.bar_unit} (~{years:.1f} years at "
                f"{self.periods_per_year:g}/year). A plain t-test would say p={b.p_naive:.4f}; "
                f"allowing for serial dependence in the P&L, p={b.p_alone:.4f} before any "
                f"correction for trials.",
            ]
            if self.trials > 1:
                lines.append(f"Haircut Sharpe after {self.trials} trials: {b.haircut_sharpe:.2f}.")
            lines.extend(self._sharpe_lines())
            n_ok = b.survives_trials(self.alpha)
            if n_ok == 0:
                lines.append("It would not pass even as the only strategy ever tried.")
            elif n_ok >= 10**6:
                lines.append("It would pass as the best of over a million independent trials.")
            else:
                lines.append(
                    f"It would pass as the best of at most {n_ok} independent "
                    f"trial{'s' if n_ok != 1 else ''}. If you tried more than that, it's not evidence."
                )
        if len(self.strategies) > 1:
            lines.append("")
            width = max([len(s.name) for s in self.strategies] + [7])
            lines.append(f"{'variant':<{width}}  {'Sharpe':>7}  {'p naive':>8}  {'p alone':>8}  "
                         f"{'p adj':>8}  {'haircut':>7}  verdict")
            order = sorted(self.strategies, key=lambda s: (not s.reportable, s.p_adjusted, -s.sharpe))
            for s in order:
                if not s.reportable:
                    lines.append(f"{s.name:<{width}}  NOT REPORTABLE ({s.n_bars} bars)")
                    continue
                v = "SIGNIFICANT" if s.significant(self.alpha) else "not significant"
                lines.append(f"{s.name:<{width}}  {s.sharpe:>7.2f}  {s.p_naive:>8.4f}  "
                             f"{s.p_alone:>8.4f}  {s.p_adjusted:>8.4f}  {s.haircut_sharpe:>7.2f}  {v}")
        extra = self._cost_lines()
        if self.robustness is not None and b.reportable:
            extra = extra + self.robustness.lines()
        if extra or self.overfitting is not None:
            lines.append("")
        lines.extend(extra)
        of = self.overfitting
        if of is not None:
            lines.append(
                f"Probability of backtest overfitting (CSCV, {of.n_splits} splits): {of.pbo:.2f}. "
                + of.verdict()
            )
            lines.append(
                f"The in-sample winner's median Sharpe: {of.median_is_sharpe:.2f} in sample, "
                f"{of.median_oos_sharpe:.2f} out of sample."
            )
        lines.extend(self.notes)
        engine = f"HAC Bartlett fixed-b L={self.bandwidth}{' (max over variants)' if len(self.strategies) > 1 else ''}, one-sided"
        if self.correction == "romano_wolf":
            engine += f"; romano_wolf step-down, {DRAWS} draws"
        elif self.correction == "sidak":
            engine += "; sidak"
        lines.append(f"[{engine} | {self.provenance}]")
        return "\n".join(lines)


def _as_columns(returns: Any) -> tuple[list[str], list[Any]]:
    """Split the input into named columns: one strategy, or one per variant."""
    if isinstance(returns, Mapping):
        if not returns:
            raise ValueError("returns is empty")
        return [str(k) for k in returns], list(returns.values())
    columns = getattr(returns, "columns", None)
    if columns is not None and hasattr(returns, "__getitem__"):  # pandas DataFrame
        return [str(c) for c in columns], [returns[c] for c in columns]
    ndim = getattr(returns, "ndim", 1)
    if ndim == 2:
        return [str(j) for j in range(returns.shape[1])], [returns[:, j] for j in range(returns.shape[1])]
    return ["strategy"], [returns]


def check_backtest(
    returns: Any,
    *,
    trials: int | None = None,
    benchmark: Iterable[Any] | None = None,
    positions: Any = None,
    costs: float = 0.0,
    asset_returns: Any = None,
    periods_per_year: float | None = None,
    alpha: float = 0.05,
    bandwidth: int | None = None,
    seed: int = 0,
) -> BacktestResult:
    """Is this backtest's mean return above zero once dependence and selection are allowed for?

    `returns` is the strategy's per-bar returns (net of costs) in time order:
    a list, numpy array or pandas Series. It can also be the backtest
    itself: a vectorbt Portfolio (a multi-column one is a grid of variants),
    the stats from backtesting.py's ``Backtest.run()``, a trade list from
    `tokio_ai.read_tradingview` (tested per closed trade; periods_per_year
    defaults to trades per year), or a dict of those;
    returns and positions are then read from it (see `tokio_ai.adapters`). To correct for a search, pass every
    variant you tried instead, as a dict {name: returns}, a pandas DataFrame
    or a 2-D array with one column per variant; the verdict is then about the
    best of them, corrected for all of them.

    `trials`: how many variants you tried in total. Defaults to the number
    passed in. If it's larger (you tried 200 and kept 5), the variants that
    aren't on hand can't be modelled, so every p-value gets Sidak's
    worst-case correction for `trials`.

    `benchmark`: optional per-bar returns to subtract first, so the question
    becomes "does it beat this?" -- e.g. buy-and-hold scaled to the
    strategy's average exposure. Without it, the null is a mean return of
    zero, and a strategy that is mostly long a rising market passes because
    the market rose.

    `positions`: optional, the position held over each bar (+1 long, -1
    short, 0 flat, or sizes), for one strategy, or a dict {name: positions}.
    Pass it when positions are held for long stretches. A long block that
    happens to sit through a rally makes the P&L persistent in a way its own
    autocorrelation barely shows; measured on 20 years of AAPL, random
    positions held ~120 bars were called profitable 9.6% of the time
    without it. With it, the lag window also covers the positions' mean
    holding run, the same guard `check` applies to a persistent condition.

    `asset_returns`: optional, the traded instrument's own per-bar returns
    (a dict per variant if they differ). With `positions`, the result adds a
    one-bar-delay check: the Sharpe if every position were taken a bar
    later, the usual giveaway of lookahead. The vectorbt and backtesting.py
    adapters fill this in.

    `costs`: cost per unit of position traded, as a return: 0.0005 is 5 bps
    for trading 100% of capital. Needs `positions` for every variant; each
    bar is charged `costs * |position change|` (entering from flat counts;
    NaN positions count as flat). Whenever positions are given, the result
    also reports the breakeven cost -- the most you could pay per unit
    traded and still have a significant result, corrected for every trial.

    Bars where any series is missing (None/NaN) are dropped from all of
    them. `bandwidth` overrides the HAC lag window (default: the largest of
    the Newey-West rule of thumb, Andrews' AR(1) plug-in and the positions'
    mean holding run, per variant).
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover - exercised only without numpy
        raise ImportError("check_backtest needs numpy (pip install numpy)") from e

    if is_backtest_object(returns):
        returns, unpacked_positions, unpacked_assets = unpack(returns)
        if positions is None:
            positions = unpacked_positions
        if asset_returns is None:
            asset_returns = unpacked_assets
    trade_notes: list[str] = []
    bar_unit = "bars"
    if isinstance(returns, TradeList) or (
            isinstance(returns, Mapping) and returns and all(isinstance(v, TradeList) for v in returns.values())):
        bar_unit = "trades" if isinstance(returns, TradeList) else "days"
        returns, ppy, trade_notes = _from_trade_lists(returns)
        if periods_per_year is None:
            periods_per_year = ppy
    if periods_per_year is None:
        periods_per_year = 252

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")

    names, cols = _as_columns(returns)
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    n = None
    mats = []
    for name, c in zip(names, cols):
        if benchmark is not None:
            _aligned(c, benchmark)
        try:
            arr = np.asarray(c, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"returns for {name!r} are not numeric") from e
        if arr.ndim != 1:
            raise ValueError(f"returns for {name!r} must be one-dimensional")
        if n is None:
            n = len(arr)
        elif len(arr) != n:
            raise ValueError(f"variant {name!r} has length {len(arr)}, expected {n}")
        mats.append(arr)
    X = np.vstack(mats)
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)
        if b.shape != (n,):
            raise ValueError(f"benchmark has length {len(b)}, returns has {n}")
        X = X - b[None, :]

    k = len(names)
    if trials is None:
        trials = k
    if trials < k:
        raise ValueError(f"trials={trials} but {k} variants were passed; trials counts all of them")

    if costs < 0:
        raise ValueError(f"costs must be non-negative, got {costs!r}")
    runs = [0] * k
    turnover = [None] * k
    net = [None] * k  # mean signed position, NaN counted as flat
    held = [None] * k  # positions as given
    if positions is not None:
        pos_map = positions if isinstance(positions, Mapping) else (
            {names[0]: positions} if k == 1 else None)
        if pos_map is None:
            raise ValueError("with several variants, pass positions as a dict {name: positions}")
        unknown = set(map(str, pos_map)) - set(names)
        if unknown:
            raise ValueError(f"positions given for unknown variants: {sorted(unknown)}")
        for j, nm in enumerate(names):
            if nm not in pos_map:
                continue
            pz = np.asarray(pos_map[nm], dtype=float)
            if pz.shape != (n,):
                raise ValueError(f"positions for {nm!r} have length {len(pz)}, returns have {n}")
            sg = np.sign(pz[~np.isnan(pz)])
            if len(sg):
                runs[j] = max(1, round(len(sg) / (1 + int(np.count_nonzero(sg[1:] != sg[:-1])))))
            turnover[j] = np.abs(np.diff(np.nan_to_num(pz, nan=0.0), prepend=0.0))
            net[j] = np.nan_to_num(pz, nan=0.0)
            held[j] = pz
    if costs > 0 and any(t is None for t in turnover):
        missing = [nm for nm, t in zip(names, turnover) if t is None]
        raise ValueError(f"costs need positions for every variant; missing for {missing}")
    T = np.vstack(turnover) if all(t is not None for t in turnover) else None
    if positions is not None and T is None:
        partial_note = ("Positions weren't given for every variant, so no breakeven cost: costs "
                        "can't be charged to the variants without them.")
    else:
        partial_note = None

    keep = ~np.isnan(X).any(axis=0)
    dropped = int(n - keep.sum())
    X = X[:, keep]
    turnover = [t[keep] if t is not None else None for t in turnover]
    net = [float(v[keep].mean()) if v is not None and keep.any() else None for v in net]
    if T is not None:
        T = T[:, keep]

    notes = list(trade_notes)
    if dropped:
        notes.append(f"{dropped} bars with a missing value in some series were dropped from all of them.")
    if benchmark is not None:
        notes.append("Returns are measured in excess of the benchmark.")

    def run(c, notes):
        Xc = X - c * T if c else X
        return _score(np, Xc, names, runs, trials, alpha, periods_per_year, bandwidth, seed, notes)

    res = run(costs, notes)
    if k >= MIN_PBO_VARIANTS and res.best.reportable and X.shape[1] >= 16 * 10:
        from .overfit import probability_of_overfitting

        Xn = X - costs * T if costs else X
        res = replace(res, overfitting=probability_of_overfitting(
            {nm: Xn[j] for j, nm in enumerate(names)}, periods_per_year=periods_per_year))
    if res.best.reportable:
        Xn = X - costs * T if costs else X
        res = replace(res, bar_unit=bar_unit, **_deflated(Xn, names, trials, periods_per_year, bandwidth))
        jb = names.index(res.best.name)
        pos_b = held[jb][keep] if held[jb] is not None else None
        res = replace(res, robustness=diagnose(
            np, Xn[jb], periods_per_year, bar_unit, res.best.name, trials, pos_b,
            _asset_for(np, asset_returns, res.best.name, n, keep)))
    else:
        res = replace(res, bar_unit=bar_unit)
    if not res.best.reportable:
        return res
    j = names.index(res.best.name)
    if res.significant and benchmark is None and net[j] is not None and abs(net[j]) >= NET_EXPOSURE_WARN:
        side = "long" if net[j] > 0 else "short"
        notes.append(
            f"{res.best.name} was net {side} on average (mean position {net[j]:+.2f}), and the null here "
            f"is zero profit, so part of this may be the market's own drift rather than timing. To test "
            f"the timing, pass benchmark= the asset's per-bar returns times {net[j]:.2f}.")
    extra = {}
    if turnover[j] is not None:
        extra["turnover_per_year"] = float(turnover[j].mean() * periods_per_year)
    if T is None:
        if partial_note:
            notes.append(partial_note)
        return replace(res, **extra)
    extra["costs"] = costs
    gross = run(0.0, []) if costs else res
    if costs:
        # The same variant's p before costs -- costs can change which variant
        # is strongest, and the message is about the one being reported.
        extra["p_before_costs"] = next(
            st.p_adjusted for st in gross.strategies if st.name == res.best.name)
    if gross.significant:
        extra["breakeven_cost"] = _breakeven(run, X, T)
    return replace(res, **extra)


def _asset_for(np, asset_returns, name, n, keep):
    """The traded asset's returns for one variant, on the kept bars, or None."""
    if asset_returns is None:
        return None
    a = asset_returns.get(name) if isinstance(asset_returns, Mapping) else asset_returns
    if a is None:
        return None
    arr = np.asarray(a, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"asset_returns have length {len(arr)}, returns have {n}")
    return arr[keep]


def _deflated(X, names, trials, periods_per_year, bandwidth) -> dict:
    """DSR/PSR of the highest-Sharpe variant, with and without the dependence correction."""
    from .sharpe import deflated_sharpe_ratio

    try:
        arg = {nm: X[j] for j, nm in enumerate(names)} if len(names) > 1 else X[0]
        kw = dict(trials=trials, periods_per_year=periods_per_year, bandwidth=bandwidth)
        return {"deflated": deflated_sharpe_ratio(arg, dependence=True, **kw),
                "deflated_published": deflated_sharpe_ratio(arg, **kw)}
    except ValueError:  # e.g. a flat series
        return {}


def _from_trade_lists(returns):
    """A TradeList (or {name: TradeList}) as returns, periods per year, and notes."""
    if isinstance(returns, TradeList):
        notes = list(returns.notes)
        basis = "position value" if returns.basis == "position" else "account equity"
        notes.append(f"Tested per closed trade ({len(returns)} trades, returns on {basis}); "
                     "the Sharpe is per trade, annualized by trades per year.")
        if returns.skipped_open:
            notes.append(f"{returns.skipped_open} trade(s) still open at the end were left out.")
        ppy = returns.periods_per_year
        if ppy is None:
            notes.append("No usable dates, so periods_per_year defaults to 252; pass the real trades per year.")
        return list(returns.returns), ppy, notes
    series, ppy = on_calendar(returns)
    notes = [f"{len(returns)} trade lists put on a common calendar: each trade's return is booked on "
             "its exit day, summed per day."]
    return series, ppy, notes


def _breakeven(run, X, T) -> float | None:
    """Largest cost per unit traded at which run(cost) is still significant.

    Bisection, to 0.01 bps. Above max_j mean_j / turnover_j every variant
    loses money net of costs, so nothing can be significant there. The
    verdict isn't strictly monotone in the cost (the HAC window is
    data-driven), so this finds a boundary rather than a guaranteed maximum.
    """
    mean_t = T.mean(axis=1)
    traded = mean_t > 0
    if not traded.any():
        return None
    hi = float((X.mean(axis=1)[traded] / mean_t[traded]).max())
    if hi <= 0:
        return None
    lo = 0.0
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        if run(mid, []).significant:
            lo = mid
        else:
            hi = mid
    return lo


def _score(np, X, names, runs, trials, alpha, periods_per_year, bandwidth, seed, notes) -> BacktestResult:
    k = len(names)
    m = X.shape[1]
    cap = max(m - 1, 0)

    if m < MIN_BARS:
        strategies = [
            StrategyResult(nm, m, math.nan, math.nan, 1.0, 1.0, 1.0, 0.0, math.nan, False)
            for nm in names
        ]
        return BacktestResult(strategies, trials, "none", alpha, periods_per_year, 0, stamp(), notes)

    mean = X.mean(axis=1)
    D = X - mean[:, None]
    sd = D.std(axis=1, ddof=1)
    # Each variant's own statistic uses its own window, so its p_alone is
    # exactly what it gets passed in alone. The correlation between variants
    # uses one common window (the longest), which keeps the matrix PSD.
    if bandwidth is None:
        own_L = [min(cap, max(default_bandwidth(m), andrews_bandwidth(np, D[j]), runs[j]))
                 for j in range(k)]
    else:
        own_L = [min(cap, max(0, int(bandwidth)))] * k
    L = max(own_L)
    lrv = np.array([
        max(float(bartlett_long_run_cov(np, D[j][None, :], own_L[j])[0, 0]), 0.0) for j in range(k)
    ])
    ann = math.sqrt(periods_per_year)

    z = np.zeros(k)
    rows = []
    for j in range(k):
        flat = sd[j] <= 0 or lrv[j] <= 0
        z[j] = 0.0 if flat else mean[j] * m / math.sqrt(lrv[j]) * fixed_b_scale((own_L[j] + 1) / m)
        z_naive = 0.0 if sd[j] <= 0 else mean[j] / (sd[j] / math.sqrt(m))
        ac1 = (float(D[j, 1:] @ D[j, :-1] / (D[j] @ D[j])) if D[j] @ D[j] > 0 else 0.0)
        m2 = float((D[j] ** 2).mean())
        skew = float((D[j] ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0
        sharpe = 0.0 if sd[j] <= 0 else float(mean[j] / sd[j] * ann)
        rows.append((sharpe, _p_greater(z_naive), _p_greater(float(z[j])), ac1, flat, skew))

    if trials == 1:
        correction = "none"
        p_adj = np.array([r[2] for r in rows])
    elif trials > k:
        correction = "sidak"
        p_adj = np.array([_sidak(r[2], trials) for r in rows])
        notes.append(
            f"{trials - k} of the {trials} trials weren't passed in, so the correction treats "
            f"all {trials} as independent: the worst case. Pass every variant to get credit for "
            f"how similar they are."
        )
    else:
        correction = "romano_wolf"
        C = bartlett_long_run_cov(np, D, L)
        sdl = np.sqrt(np.maximum(np.diag(C), 1e-300))
        R = C / np.outer(sdl, sdl)
        p_adj = _romano_wolf(np, z, R, seed, one_sided=True)

    strategies = []
    for j, nm in enumerate(names):
        sharpe, p_naive, p_alone, ac1, flat, skew = rows[j]
        pa = float(p_adj[j])
        if z[j] > 0 and pa < 0.5 and not flat:
            # -inv_cdf(p), not inv_cdf(1 - p): below ~1e-16, 1 - p rounds to 1.0
            haircut = sharpe * -_N.inv_cdf(max(pa, 1e-300)) / z[j]
        else:
            haircut = 0.0
        strategies.append(StrategyResult(
            name=nm, n_bars=m, mean=float(mean[j]), sharpe=sharpe, p_naive=p_naive,
            p_alone=p_alone, p_adjusted=pa, haircut_sharpe=max(haircut, 0.0),
            lag1_autocorr=ac1, reportable=not flat, skewness=skew,
        ))

    res = BacktestResult(strategies, trials, correction, alpha, periods_per_year, L, stamp(), notes)
    best = res.best
    if best.reportable and abs(best.lag1_autocorr) >= 0.1:
        direction = "understates" if best.lag1_autocorr > 0 else "overstates"
        notes.append(
            f"The P&L's lag-1 autocorrelation is {best.lag1_autocorr:+.2f}: a test that treats "
            f"bars as independent {direction} the uncertainty. This one doesn't."
        )
    if best.reportable and best.skewness <= SKEW_WARN:
        notes.append(
            f"Warning: the P&L's skewness is {best.skewness:.1f} -- many small wins, rare large "
            f"losses, like a book of favourites or sold options. Any test that estimates the "
            f"variance from the P&L, this one included, is too generous here until the losses "
            f"have shown up often enough (measured: 9-16% false positives at a nominal 5% on "
            f"fairly priced 95-98c contracts). If these are bets on binary contracts at known "
            f"prices, tokio_ai.check_contracts tests them exactly."
        )
    if best.reportable and best.p_naive <= alpha < best.p_adjusted:
        notes.append("The plain t-test calls this significant; the corrected test doesn't.")
    return res
