"""Test a whole grid of conditions at once, correcting for having searched it.

The most common way to fool yourself with a backtest is not a bad test. It
is running twenty variants -- lookback 10, 20 or 60; threshold 1%, 2% or 3%;
horizon 1, 5 or 20 -- and reporting whichever one cleared p < 0.05. With
twenty tries, something usually does.

The standard corrections are a poor fit for grids. Bonferroni and Holm
divide alpha among the variants as if they were unrelated; when they are
near-copies of each other ("20-day momentum" and "40-day momentum" agree
on most days), that is far stricter than it needs to be, and real edges get
thrown away. Benjamini-Hochberg controls a weaker error rate.

`check_many` uses the correlation between the variants instead. Each
variant gets exactly the statistic `check` would give it alone -- a Hodrick
z-statistic (see `rigor/overlap.py`). What the family adds is their joint
distribution: every z is a sum of one-bar terms, so the covariance between
any two variants' statistics can be estimated from those terms directly,
with the same exact treatment of overlapping windows. The correction is the
Romano & Wolf (2005) step-down max-|z| against that joint normal: for each
variant in order of strength, how often would the largest |z| among the
variants not yet rejected be at least this large by chance?

That controls the family-wise error rate -- the probability of even one
false discovery anywhere in the grid -- while crediting the grid for being
one idea tried several ways rather than thirty independent ideas.

Requires numpy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .check import _aligned, check
from .rigor.overlap import bartlett_long_run_cov, hodrick_terms
from .rigor.provenance import stamp

DRAWS = 20000
_TIE_RTOL = 1e-9


@dataclass(frozen=True)
class FamilyMember:
    name: str
    horizon: int
    n_condition: int
    n_other: int
    gap: float
    p_raw: float  # this variant alone: exactly check()'s p-value
    p_rotation: float  # this variant alone: check()'s rotation second opinion
    p_adjusted: float  # corrected for the whole grid (Romano-Wolf)
    reportable: bool

    def significant(self, alpha: float) -> bool:
        return self.reportable and self.p_adjusted <= alpha


@dataclass(frozen=True)
class FamilyResult:
    members: list[FamilyMember]
    alpha: float
    n_bars: int
    provenance: str
    uncorrected_hits: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> list[FamilyMember]:
        return [m for m in self.members if m.significant(self.alpha)]

    def __str__(self) -> str:
        tested = [m for m in self.members if m.reportable]
        head = (
            f"{len(tested)} variants tested as one family (alpha={self.alpha}, "
            f"family-wise). {len(self.significant)} survive the correction"
        )
        if self.uncorrected_hits:
            head += f"; {self.uncorrected_hits} would have looked significant alone"
        lines = [head + ".", ""]
        width = max([len(m.name) for m in self.members] + [7])
        lines.append(f"{'variant':<{width}}  {'h':>3}  {'gap':>9}  {'p alone':>8}  {'p grid':>8}  verdict")
        order = sorted(self.members, key=lambda m: (not m.reportable, m.p_adjusted, m.p_raw))
        for m in order:
            if not m.reportable:
                lines.append(f"{m.name:<{width}}  {m.horizon:>3}  {'':>9}  {'':>8}  {'':>8}  "
                             f"NOT REPORTABLE (n={m.n_condition}/{m.n_other})")
                continue
            verdict = "SIGNIFICANT" if m.significant(self.alpha) else "not significant"
            lines.append(
                f"{m.name:<{width}}  {m.horizon:>3}  {m.gap:>+9.3%}  {m.p_raw:>8.4f}  "
                f"{m.p_adjusted:>8.4f}  {verdict}"
            )
        lines.extend(self.notes)
        lines.append(f"[romano_wolf step-down, joint normal of Hodrick statistics, {DRAWS} draws "
                     f"| {self.provenance}]")
        return "\n".join(lines)


def _romano_wolf(np, z, R, seed, one_sided=False):
    """Step-down max-|z| p-values for statistics z with correlation R.

    With `one_sided`, the alternative is z > 0 for every member (a backtest
    claims a profit, not merely a difference): the step-down runs on max z
    rather than max |z|, and the Holm cap uses the one-sided tail.
    """
    k = len(z)
    vals, vecs = np.linalg.eigh(R)
    A = vecs * np.sqrt(np.clip(vals, 0.0, None))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((DRAWS, k)) @ A.T
    obs = np.asarray(z, dtype=float)
    if not one_sided:
        Z = np.abs(Z)
        obs = np.abs(obs)
    order = np.argsort(-obs, kind="stable")
    remaining = list(range(k))
    p_adj = np.ones(k)
    running = 0.0
    for j in order:
        mx = Z[:, remaining].max(axis=1)
        adj = (np.count_nonzero(mx >= obs[j] * (1 - _TIE_RTOL)) + 1) / (DRAWS + 1)
        # By the union bound, the max-|z| tail over the remaining variants is
        # never above Holm's for ANY correlation, so this cap changes nothing
        # except Monte Carlo noise near the boundary.
        tail = math.erfc(obs[j] / math.sqrt(2))
        if one_sided:
            tail /= 2
        holm = min(1.0, tail * len(remaining))
        # And never below the variant's own p-value: the max over a set that
        # includes it can't be smaller. Only Monte Carlo noise could say so.
        running = max(running, min(adj, holm), min(tail, 1.0))
        p_adj[j] = min(running, 1.0)
        if running >= 1.0:
            break  # step-down p-values only rise; everything left is 1 (p_adj's default)
        remaining.remove(j)
    return p_adj


def check_many(
    returns: Iterable[Any],
    conditions: Mapping[str, Iterable[Any]],
    horizon: int | Sequence[int] = 1,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> FamilyResult:
    """Test every condition (at every horizon) as one family.

    `conditions` maps a name to a boolean series aligned with `returns`,
    with the same conventions as `check`: condition[i] known at the close of
    bar i, None/NaN for unknown. `horizon` may be a list, in which case every
    condition is tested at every horizon and all of them form the family.

    Each variant's `p_raw` is exactly what `check` reports for it alone;
    `p_adjusted` is its Romano-Wolf step-down p-value, corrected for the
    whole grid. A variant is SIGNIFICANT only if `p_adjusted <= alpha`: the
    probability of even one false discovery anywhere in the family is held
    at alpha. `p_adjusted` is never larger than Holm's correction would give.
    `seed` fixes the Monte Carlo draws, so results are reproducible.
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover - exercised only without numpy
        raise ImportError("check_many needs numpy (pip install numpy)") from e

    if not conditions:
        raise ValueError("conditions is empty")
    horizons = [horizon] if isinstance(horizon, int) else list(horizon)
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError(f"horizons must be positive, got {horizons!r}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    if not hasattr(returns, "__len__"):
        returns = list(returns)
    n = len(returns)
    conds = {}
    for name, c in conditions.items():
        _aligned(returns, c)
        if not hasattr(c, "__len__"):
            c = list(c)
        if len(c) != n:
            raise ValueError(f"condition {name!r} has length {len(c)}, returns has {n}")
        conds[name] = c

    meta = []  # (label, horizon, single CheckResult, z, reportable)
    u_rows = []
    bandwidths = []
    for name, c in conds.items():
        for h in horizons:
            single = check(returns, c, horizon=h, alpha=alpha)
            terms = hodrick_terms(np, returns, c, h)
            reportable = single.verdict != "NOT REPORTABLE" and terms is not None
            z = 0.0
            if reportable:
                # The same z check() computed, rebuilt from the terms already
                # in hand rather than running the engine a third time.
                u, own_L = terms
                v = float(bartlett_long_run_cov(np, u[None, :], own_L)[0, 0])
                z = float(u.sum()) / math.sqrt(v) if v > 0 else 0.0
            label = name if len(horizons) == 1 else f"{name} @h{h}"
            meta.append((label, h, single, z, reportable))
            if reportable:
                u_rows.append(terms[0])
                bandwidths.append(terms[1])

    p_adj_all = np.ones(len(meta))
    rep_idx = [i for i, m in enumerate(meta) if m[4]]
    if rep_idx:
        # The correlation between variants, from their one-bar terms at one
        # common bandwidth (the longest any variant needs), which keeps the
        # matrix positive semi-definite. Each variant's own z -- and so its
        # marginal p-value -- stays exactly what check() gives it.
        C = bartlett_long_run_cov(np, np.vstack(u_rows), max(bandwidths))
        sd = np.sqrt(np.maximum(np.diag(C), 1e-300))
        R = C / np.outer(sd, sd)
        z = np.array([meta[i][3] for i in rep_idx])
        p_adj = _romano_wolf(np, z, R, seed)
        for k, i in enumerate(rep_idx):
            p_adj_all[i] = p_adj[k]

    members = [
        FamilyMember(
            name=label, horizon=h, n_condition=single.n_condition, n_other=single.n_other,
            gap=single.gap, p_raw=single.p_value, p_rotation=single.p_rotation,
            p_adjusted=float(p_adj_all[i]), reportable=reportable,
        )
        for i, (label, h, single, _, reportable) in enumerate(meta)
    ]
    uncorrected = sum(1 for m in members if m.reportable and m.p_raw <= alpha)
    notes = []
    split = [m.name for m in members
             if m.reportable and (m.p_raw <= alpha) != (m.p_rotation <= alpha)]
    if split:
        notes.append(
            f"Second opinion (rotation test, per variant alone) disagrees on {len(split)} of "
            f"{sum(m.reportable for m in members)} variants: {', '.join(split)}."
        )
    else:
        notes.append("Second opinion (rotation test, per variant alone) agrees on every variant.")
    if uncorrected > sum(1 for m in members if m.significant(alpha)):
        notes.append(
            "Variants that clear alpha alone but not as a family are what a grid search "
            "produces by chance; treat them as unconfirmed."
        )
    return FamilyResult(members, alpha, n, stamp(), uncorrected, notes)
