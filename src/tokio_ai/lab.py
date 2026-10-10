"""A backtesting lab that keeps the backtest honest.

`check_backtest` judges a backtest you bring. `Lab` runs it, and removes the
usual ways a backtest flatters itself:

- **Lookahead is tested, not trusted.** You write `signal(data, **params)`
  returning a target position per bar (-1..1, fraction of equity). After
  running it, the lab re-runs it on the data cut off at several random bars;
  if any position before a cut changes when later bars are removed, the
  signal used information it didn't have yet, and the lab refuses with the
  first bar that moved. That catches `shift(-1)`, centered rolling windows,
  and scaling by full-sample statistics.
- **Realistic execution by default.** A position decided at bar t's close is
  filled at bar t+1's open, after the overnight gap, which is held at the old
  position. Commission and slippage are charged on every change of position.
- **Every variant you run is counted.** `lab.check()` tests the best of
  everything you tried, corrected for all of it (Romano-Wolf across the
  variants' actual returns, plus the probability of backtest overfitting).
  There is no trial count to remember.
- **A locked holdout.** The last `holdout` share of the data is invisible to
  your signal until `lab.final_test(signal, **params)`, which works once.

    lab = tokio_ai.Lab("SPY")                       # free daily bars, ~20 years
    def sma_cross(d, fast=10, slow=50):
        f, s = d.close.rolling(fast).mean(), d.close.rolling(slow).mean()
        return (f > s).astype(float)
    lab.sweep(sma_cross, fast=[5, 10, 20], slow=[50, 100, 200])
    print(lab.check())
    print(lab.final_test(sma_cross, fast=10, slow=100))

Requires numpy and pandas.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
LOOKAHEAD_CUTS = 6


class LookaheadError(ValueError):
    """The signal's past positions changed when future bars were removed."""


@dataclass
class Run:
    name: str
    params: dict
    returns: Any  # pandas Series of per-bar net returns
    positions: Any  # pandas Series: position held over each bar's open..close
    asset_returns: Any
    trades: int
    variant: int  # 1-based index among this lab's variants
    segment: str  # "in-sample" or "holdout"
    report: Any = None  # BacktestResult
    signal: Any = field(default=None, repr=False)  # the function that produced it

    def __str__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        head = (f"{self.name}({p}) on {self.segment} data: {len(self.returns)} bars, "
                f"{self.trades} position changes.")
        return head + "\n" + str(self.report)


@dataclass
class Lab:
    data: Any
    holdout: float = 0.25
    commission: float = 0.0005  # per unit of position traded, as a return (5 bps)
    slippage: float = 0.0005
    periods_per_year: float | None = None
    range_: str = "max"
    # "cash": is the profit above zero? "market": did it beat holding the asset
    # at the same average exposure (so a long-only strategy can't pass on the
    # market's own rise)?
    vs: str = "cash"
    runs: list[Run] = field(default_factory=list, repr=False)
    _keys: dict = field(default_factory=dict, repr=False)
    _signals: list = field(default_factory=list, repr=False)  # (function, display name), by identity
    _final_done: bool = field(default=False, repr=False)

    def __post_init__(self):
        pd = _pandas()
        if isinstance(self.data, str):
            self.symbol = self.data.upper()
            self.data = load_prices(self.symbol, self.range_)
        else:
            self.symbol = None
            self.data = _normalise(pd, self.data)
        if not 0 <= self.holdout < 0.9:
            raise ValueError(f"holdout must be in [0, 0.9), got {self.holdout!r}")
        if self.vs not in ("cash", "market"):
            raise ValueError(f"vs must be 'cash' or 'market', got {self.vs!r}")
        if self.commission < 0 or self.slippage < 0:
            raise ValueError("commission and slippage must be non-negative")
        n = len(self.data)
        self.split = n - int(round(n * self.holdout))
        if self.split < 100:
            raise ValueError(f"only {self.split} in-sample bars; need at least 100")
        if self.periods_per_year is None:
            self.periods_per_year = _infer_ppy(pd, self.data.index)

    # ------------------------------------------------------------------ running

    @property
    def in_sample(self):
        return self.data.iloc[: self.split]

    def run(self, signal: Callable, **params) -> Run:
        """Backtest `signal(data, **params)` on the in-sample data; counted as a variant."""
        name = self._name(signal)
        key = _key(name, params)
        if key in self._keys:  # re-running an identical variant isn't a new trial
            return self.runs[self._keys[key]]
        d = self.in_sample
        _prefetch(signal, name, d, [params])
        pos = _positions(_pandas(), signal, d, params, name)
        _assert_no_lookahead(_pandas(), signal, d, params, pos, name)
        r, held, a, trades = _simulate(d, pos, self.commission + self.slippage)
        run = Run(name, dict(params), r, held, a, trades, len(self.runs) + 1, "in-sample", signal=signal)
        self._keys[key] = len(self.runs)
        self.runs.append(run)
        run.report = self._report([run], trials=len(self.runs))
        return run

    def _name(self, signal) -> str:
        """A display name unique to this function object: two different signals never share one.

        Keyed on identity, so a lambda, a closure from a factory, or a
        function redefined in a notebook cell is its own strategy (and its
        own trials), never mistaken for an earlier one with the same name.
        """
        for fn, nm in self._signals:
            if fn is signal:
                return nm
        base = getattr(signal, "__name__", "signal")
        if base == "<lambda>":
            base = "lambda"
        taken = {nm for _, nm in self._signals}
        nm, k = base, 2
        while nm in taken:
            nm, k = f"{base}#{k}", k + 1
        self._signals.append((signal, nm))
        return nm

    def sweep(self, signal: Callable, **grid) -> list[Run]:
        """Run every combination of the parameter lists given, e.g. fast=[5, 10], slow=[50, 100]."""
        names = list(grid)
        values = [v if isinstance(v, (list, tuple, range)) else [v] for v in grid.values()]
        combos = [dict(zip(names, combo)) for combo in itertools.product(*values)]
        _prefetch(signal, self._name(signal), self.in_sample, combos)
        return [self.run(signal, **p) for p in combos]

    def check(self, **kwargs):
        """Every variant run so far, tested together: the honest verdict on the search."""
        if not self.runs:
            raise ValueError("nothing has been run yet")
        return self._report(self.runs, trials=len(self.runs), **kwargs)

    def best(self) -> Run:
        """The variant `check()` reports as strongest (lowest corrected p-value)."""
        label = self.check().best.name
        return next(r for r in self.runs if _label(r) == label)

    def final_test(self, signal: Callable | Run, *, force: bool = False, **params) -> Run:
        """The one look at the holdout: the chosen variant on data it has never seen.

        Tested alone (trials=1): the search happened in-sample, and this is
        the single pre-registered test of its winner. A second call raises,
        because a holdout looked at twice is just more in-sample data;
        `force=True` runs it anyway and says so in the report.
        """
        if isinstance(signal, Run):  # lab.final_test(lab.best())
            params = {**signal.params, **params}
            signal = signal.signal
        if self.holdout == 0:
            raise ValueError("this lab has no holdout (holdout=0)")
        if self._final_done and not force:
            raise RuntimeError("the holdout has already been used; a second look makes it in-sample. "
                               "Pass force=True to run anyway (the report will say so).")
        pd = _pandas()
        name = self._name(signal)
        full = self.data
        _prefetch(signal, name, full, [params])
        pos = _positions(pd, signal, full, params, name)
        _assert_no_lookahead(pd, signal, full, params, pos, name)
        r, held, a, _ = _simulate(full, pos, self.commission + self.slippage)
        h = slice(self.split, None)
        changes = held.diff().abs().fillna(held.abs()) > 0
        run = Run(name, dict(params), r.iloc[h], held.iloc[h], a.iloc[h],
                  int(changes.iloc[h].sum()), 0, "holdout")
        repeat = self._final_done
        self._final_done = True
        run.report = self._report([run], trials=1)
        if repeat:
            run.report.notes.append("WARNING: the holdout had already been used; this second look is not "
                                    "an out-of-sample test.")
        return run

    def _tested(self, run):
        """The return series actually tested: the run's, or its excess over exposure-matched holding."""
        if self.vs == "cash":
            return run.returns
        w = float(run.positions.mean())
        return run.returns - w * run.asset_returns.fillna(0.0)

    def _report(self, runs, trials, **kwargs):
        from dataclasses import replace

        from .backtest import check_backtest

        # Under vs="market" the tested series is an excess return; a rebuilt
        # long-only Sharpe beside it would mix the two, so the delay check is
        # left out (the lab has tested lookahead directly anyway).
        assets = (lambda r: r.asset_returns) if self.vs == "cash" else (lambda r: None)
        if len(runs) == 1:
            r = runs[0]
            res = check_backtest(self._tested(r), positions=r.positions, asset_returns=assets(r),
                                 trials=trials, periods_per_year=self.periods_per_year, **kwargs)
        else:
            labels = [_label(r) for r in runs]
            per = {lb: assets(r) for lb, r in zip(labels, runs)}
            res = check_backtest({lb: self._tested(r) for lb, r in zip(labels, runs)},
                                 positions={lb: r.positions for lb, r in zip(labels, runs)},
                                 asset_returns=per if self.vs == "cash" else None,
                                 trials=trials, periods_per_year=self.periods_per_year, **kwargs)
        if res.robustness is not None:
            res = replace(res, robustness=replace(res.robustness, lookahead_tested=True))
        cost = (self.commission + self.slippage) * 1e4
        res.notes.append(f"Lab: fills at the next bar's open; {cost:.1f} bps per unit traded "
                         f"(commission + slippage) is already charged in these returns; {trials} "
                         f"variant(s) counted"
                         + ("" if runs[0].segment == "holdout" else
                            f"; holdout of {len(self.data) - self.split} bars kept unseen") + ".")
        if self.vs == "market":
            res.notes.append("Tested against the market: each variant's returns minus holding the asset "
                             "at that variant's average exposure, so the question is timing, not the "
                             "market's own rise.")
            res.notes[:] = [n for n in res.notes if "market's own drift" not in n]
        return res


# ---------------------------------------------------------------------- engine

def _simulate(d, pos, cost: float):
    """Per-bar net returns for target weights decided at each close, filled at the next open.

    Bar t: the overnight gap (close[t-1] -> open[t]) is earned at the
    position held coming in, w[t-1]; at the open the book is rebalanced to the
    new target (paying `cost` per unit traded); the session (open[t] ->
    close[t]) is earned at the new position. The target decided at bar t-1's
    close is what is traded at bar t's open.
    """
    import numpy as np

    o = d["open"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    target = np.nan_to_num(pos.to_numpy(dtype=float), nan=0.0)
    held = np.concatenate([[0.0], target[:-1]])  # position over bar t's session
    before = np.concatenate([[0.0], held[:-1]])  # position coming into bar t (through the gap)
    gap = np.zeros(len(c))
    gap[1:] = o[1:] / c[:-1] - 1
    session = c / o - 1
    traded = np.abs(held - before)
    r = (1 + before * gap) * (1 - cost * traded) * (1 + held * session) - 1
    pd = _pandas()
    idx = d.index
    asset = pd.Series(np.concatenate([[np.nan], c[1:] / c[:-1] - 1]), index=idx)
    return (pd.Series(r, index=idx), pd.Series(held, index=idx), asset,
            int(np.count_nonzero(traded)))


def _positions(pd, signal, d, params, name):
    import numpy as np

    out = signal(d.copy(), **params)
    s = pd.Series(np.asarray(out, dtype=float), index=d.index) if not isinstance(out, pd.Series) else out
    if len(s) != len(d):
        raise ValueError(f"{name} returned {len(s)} positions for {len(d)} bars")
    if not s.index.equals(d.index):
        s = s.reindex(d.index)
    s = s.astype(float)
    if np.nanmax(np.abs(s.to_numpy())) > 1 + 1e-9 if s.notna().any() else False:
        raise ValueError(f"{name} returned positions beyond -1..1 (fraction of equity); scale it down")
    return s


def _cut_points(name, params, n, cuts: int = LOOKAHEAD_CUTS) -> list[int]:
    """Deterministic bars at which the lookahead test cuts the data, spread across the sample."""
    import numpy as np

    lo = max(20, n // 10)
    if n - lo < 2:
        return []
    seed = int(hashlib.sha1(f"{name}{sorted(params.items())}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return sorted(set(rng.integers(lo, n - 1, size=cuts).tolist()) | {n - 2})


def _prefetch(signal, name, d, param_sets) -> None:
    """Tell a sandboxed signal every (params, length) a run will need, so one process serves them all."""
    pre = getattr(signal, "prefetch", None)
    if pre is None:
        return
    reqs = []
    for params in param_sets:
        reqs.append((params, len(d)))
        reqs.extend((params, c + 1) for c in _cut_points(name, params, len(d)))
    pre(d, reqs)


def _assert_no_lookahead(pd, signal, d, params, pos, name, cuts: int = LOOKAHEAD_CUTS):
    """Re-run on the data cut at several bars; earlier positions must not move."""
    import numpy as np

    n = len(d)
    points = _cut_points(name, params, n, cuts)
    if not points:
        return
    full = pos.to_numpy(dtype=float)
    for cut in points:
        part = _positions(pd, signal, d.iloc[: cut + 1], params, name).to_numpy(dtype=float)
        a, b = full[: cut + 1], part
        same = (np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b)))
        if not same.all():
            bad = int(np.argmin(same))
            raise LookaheadError(
                f"{name}{_fmt(params)} uses future data: its position at {d.index[bad]} was "
                f"{a[bad]:.4g} on the full data but {b[bad]:.4g} when the data stops at "
                f"{d.index[cut]}. A signal may only use bars up to the one it's decided on "
                f"(look for shift(-k), center=True, or statistics over the whole series).")


# ------------------------------------------------------------------------ data

def load_prices(symbol: str, range_: str = "max"):
    """Daily OHLCV for `symbol` from Yahoo Finance, adjusted for splits and dividends."""
    from .tools.prices import fetch_daily_bars, fetch_full_daily_bars

    pd = _pandas()
    bars = fetch_full_daily_bars(symbol) if range_ == "max" else fetch_daily_bars(symbol, range_)
    df = pd.DataFrame([b.to_dict() for b in bars])
    df.index = pd.to_datetime(df.pop("date"))
    adj = (df["adj_close"] / df["close"]).where(df["close"] > 0)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * adj
    df = df.drop(columns=["adj_close"]).dropna(subset=["open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    return df[list(PRICE_COLUMNS)]


def _normalise(pd, df):
    if not hasattr(df, "columns"):
        raise TypeError("data must be a symbol or a DataFrame with open and close columns")
    cols = {str(c).lower(): c for c in df.columns}
    if "open" not in cols or "close" not in cols:
        raise ValueError(f"data needs open and close columns; has {list(df.columns)}")
    out = pd.DataFrame({k: df[cols[k]] for k in PRICE_COLUMNS if k in cols})
    out = out.dropna(subset=["open", "close"])
    if (out[["open", "close"]] <= 0).any().any():
        raise ValueError("open and close must be positive")
    return out


def _infer_ppy(pd, index) -> float:
    if len(index) > 2 and hasattr(index, "to_series"):
        try:
            span = (pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).total_seconds() / (365.25 * 86400)
            if span > 0:
                return len(index) / span
        except (TypeError, ValueError):
            pass
    return 252.0


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("tokio_ai.Lab needs pandas (pip install pandas)") from e
    return pd


def _key(name, params) -> str:
    return name + repr(sorted(params.items()))


def _fmt(params) -> str:
    return "(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


def _label(run) -> str:
    return run.name + _fmt(run.params)
