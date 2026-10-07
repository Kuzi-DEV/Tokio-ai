"""The Sharpe-ratio statistics of Bailey & López de Prado.

- **Probabilistic Sharpe Ratio** (PSR, Bailey & López de Prado 2012): the
  probability that the true Sharpe exceeds a benchmark, allowing for the
  sample length and the returns' skewness and kurtosis.
- **Deflated Sharpe Ratio** (DSR, Bailey & López de Prado 2014): the PSR
  against the Sharpe you'd expect from the best of N zero-edge trials, so
  the search is charged for.
- **Minimum Track Record Length** (MinTRL): how many bars it takes before
  the observed Sharpe is significantly above the benchmark.

All three are computed exactly as published, by default: the variance of
the Sharpe estimate assumes the bars are independent. Pass
``dependence=True`` for the same statistics with that variance scaled by
the P&L's long-run variance (the Bartlett HAC `check_backtest` uses), so
autocorrelated P&L can't pass as more evidence than it is. How often each
version fires on P&L with no edge is measured in
``scripts/calibration_sharpe.py``.

Sharpe ratios here are per bar, as in the papers; ``annualized`` fields
multiply by sqrt(periods_per_year).

Requires numpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class SharpeStats:
    sharpe: float  # per bar
    annualized: float
    n: int
    skewness: float
    kurtosis: float  # not excess: 3 for normal
    psr: float  # P(true Sharpe > benchmark)
    benchmark: float  # per bar; for DSR, the expected max Sharpe of `trials` zero-edge trials
    trials: int
    min_track_record: float | None  # bars; None if the Sharpe doesn't beat the benchmark
    dependence: bool
    variance_ratio: float  # long-run / plain variance used (1.0 when dependence=False)

    @property
    def dsr(self) -> float:
        """The PSR against the deflated benchmark; equal to `psr` (named for the search)."""
        return self.psr

    def significant(self, confidence: float = 0.95) -> bool:
        return self.psr >= confidence

    def __str__(self) -> str:
        kind = "Deflated Sharpe Ratio" if self.trials > 1 else "Probabilistic Sharpe Ratio"
        lines = [
            f"{kind}: {self.psr:.4f} (the probability the true Sharpe beats "
            f"{self.benchmark:+.4f}/bar"
            + (f", the best you'd expect from {self.trials} zero-edge trials" if self.trials > 1 else "")
            + ").",
            f"Sharpe {self.sharpe:+.4f}/bar ({self.annualized:+.2f} annualized) over {self.n} bars; "
            f"skewness {self.skewness:+.2f}, kurtosis {self.kurtosis:.2f}.",
        ]
        if self.min_track_record is None:
            lines.append("Minimum track record length: never -- the Sharpe doesn't beat the benchmark.")
        else:
            lines.append(f"Minimum track record length at 95%: {math.ceil(self.min_track_record)} bars.")
        if self.dependence:
            lines.append(f"Variance scaled by {self.variance_ratio:.2f} for the P&L's serial dependence.")
        return "\n".join(lines)


def _moments(np, x):
    m = x.mean()
    d = x - m
    s2 = (d * d).mean()
    if s2 <= 0:
        raise ValueError("returns have zero variance")
    s = math.sqrt(s2)
    return m, x.std(ddof=1), float((d ** 3).mean() / s ** 3), float((d ** 4).mean() / s2 ** 2)


def _variance_ratio(np, x, runs=0, bandwidth=None) -> float:
    """Long-run variance / plain variance, with the fixed-b widening -- what `check_backtest` uses."""
    from .backtest import andrews_bandwidth, fixed_b_scale
    from .rigor.overlap import bartlett_long_run_cov, default_bandwidth

    d = x - x.mean()
    m = len(d)
    cap = max(m - 1, 0)
    if bandwidth is None:
        L = min(cap, max(default_bandwidth(m), andrews_bandwidth(np, d), runs))
    else:
        L = min(cap, max(0, int(bandwidth)))
    lrv = float(bartlett_long_run_cov(np, d[None, :], L)[0, 0]) / m
    v = float((d * d).mean())
    if v <= 0 or lrv <= 0:
        return 1.0
    return (lrv / v) / fixed_b_scale((L + 1) / m) ** 2


def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """E[max] of `trials` zero-mean Sharpe estimates with this variance (BLdP 2014, eq. 2)."""
    if trials <= 1:
        return 0.0
    return math.sqrt(sharpe_variance) * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / trials)
        + EULER_GAMMA * _N.inv_cdf(1 - 1 / (trials * math.e)))


def _stats(np, x, benchmark, trials, dependence, periods_per_year, bandwidth):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        raise ValueError("need at least 3 returns")
    m, sd, g3, g4 = _moments(np, x)
    sr = m / sd
    vr = _variance_ratio(np, x, 0, bandwidth) if dependence else 1.0
    n = len(x)
    # BLdP: Var(SR) ~ (1 - g3*SR + (g4-1)/4*SR^2) / (n-1), scaled by vr
    core = max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-12) * vr
    z = (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(core)
    psr = _N.cdf(z)
    if sr > benchmark:
        mintrl = 1 + core * (_N.inv_cdf(0.95) / (sr - benchmark)) ** 2
    else:
        mintrl = None
    return SharpeStats(sharpe=sr, annualized=sr * math.sqrt(periods_per_year), n=n, skewness=g3,
                       kurtosis=g4, psr=psr, benchmark=benchmark, trials=trials,
                       min_track_record=mintrl, dependence=dependence, variance_ratio=vr)


def probabilistic_sharpe_ratio(returns: Any, benchmark: float = 0.0, *, dependence: bool = False,
                               periods_per_year: float = 252, bandwidth: int | None = None) -> SharpeStats:
    """PSR: P(true per-bar Sharpe > `benchmark`), with the minimum track record length."""
    import numpy as np

    return _stats(np, returns, benchmark, 1, dependence, periods_per_year, bandwidth)


def deflated_sharpe_ratio(returns: Any, trials: int | None = None, *, sharpe_variance: float | None = None,
                          dependence: bool = False, periods_per_year: float = 252,
                          bandwidth: int | None = None) -> SharpeStats:
    """DSR of the best of several variants, or of one strategy picked from `trials`.

    `returns`: every variant as a dict {name: returns} / DataFrame / 2-D
    array (the best by Sharpe is tested and the variance of their Sharpes
    sets the benchmark), or a single series with `trials` and
    `sharpe_variance` -- the variance of the per-bar Sharpes across all the
    trials, which the method needs and a single series can't supply. If
    `sharpe_variance` is omitted for a single series, the zero-edge
    sampling variance 1/(n-1) is used.
    """
    import numpy as np

    from .backtest import _as_columns

    names, cols = _as_columns(returns)
    arrs = [np.asarray(c, dtype=float) for c in cols]
    if len(arrs) > 1:
        keep = ~np.isnan(np.vstack(arrs)).any(axis=0)
        arrs = [a[keep] for a in arrs]
        k = len(arrs)
        trials = trials or k
        if trials < k:
            raise ValueError(f"trials={trials} but {k} variants were passed")
        # A flat variant has no Sharpe; it still counts as a trial, but can't
        # be the best or inform the spread of Sharpes.
        live = [a for a in arrs if len(a) > 2 and a.std(ddof=1) > 0]
        if not live:
            raise ValueError("every variant has zero variance")
        srs = [a.mean() / a.std(ddof=1) for a in live]
        if sharpe_variance is not None:
            var = sharpe_variance
        elif len(srs) > 1:
            var = float(np.var(srs, ddof=1))
        else:
            var = 1 / (len(live[0]) - 1)
        x = live[int(np.argmax(srs))]
    else:
        x = arrs[0]
        x = x[~np.isnan(x)]
        trials = trials or 1
        var = sharpe_variance if sharpe_variance is not None else 1 / (len(x) - 1)
    return _stats(np, x, expected_max_sharpe(trials, var), trials, dependence, periods_per_year, bandwidth)


def min_track_record_length(returns: Any, benchmark: float = 0.0, confidence: float = 0.95, *,
                            dependence: bool = False) -> float | None:
    """Bars needed for the observed Sharpe to beat `benchmark` at `confidence`; None if it never will."""
    import numpy as np

    s = _stats(np, returns, benchmark, 1, dependence, 252, None)
    if s.sharpe <= benchmark:
        return None
    core = (1 - s.skewness * s.sharpe + (s.kurtosis - 1) / 4 * s.sharpe ** 2) * s.variance_ratio
    return 1 + core * (_N.inv_cdf(confidence) / (s.sharpe - benchmark)) ** 2
