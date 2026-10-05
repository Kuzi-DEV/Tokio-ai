"""Probability of backtest overfitting (PBO), by combinatorially symmetric cross-validation.

`check_backtest` asks whether the best variant's profit survives the search.
This asks a different question about the search itself: when you pick the
variant that did best on one part of the history, does it keep doing well on
the rest? If picking by backtest is no better than picking at random, the
backtest is not telling you which variant is good -- whatever its Sharpe.

The method is Bailey, Borwein, Lopez de Prado & Zhu (2017), "The
probability of backtest overfitting". Split the history into S contiguous
blocks. For every way of choosing S/2 of them as the in-sample half (S=16
gives 12,870 splits, each paired with its mirror image):

  1. find the variant with the best in-sample Sharpe;
  2. rank that variant's out-of-sample Sharpe among all variants;
  3. record whether it landed at or below the median.

PBO is the share of splits where it did. About 0.5 means the in-sample
winner is a random pick out of sample; near 0 means the search found
something that persists.

Two more numbers from the same splits: how often the in-sample winner lost
money out of sample, and its median Sharpe in sample vs out of sample.

Not reported: the "performance degradation" regression of out-of-sample on
in-sample Sharpe from the paper. The two halves of every split are
complements, so for any one strategy they average to its full-sample
Sharpe, and the slope is pushed towards -1 whether or not anything was
overfit. Measured here: -0.33 on a grid whose winner had a real edge in
every split.

This is a diagnostic, not a significance test: it has no alpha and makes
no promise about false-positive rates. It complements check_backtest's
p-value rather than replacing it. See docs/calibration.md for how it
behaves on grids with and without a real edge.

Requires numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .rigor.provenance import stamp


@dataclass(frozen=True)
class OverfitResult:
    pbo: float
    prob_oos_loss: float  # share of splits where the in-sample winner lost money out of sample
    median_is_sharpe: float  # the winner's, annualized
    median_oos_sharpe: float
    median_oos_percentile: float  # the winner's median out-of-sample rank, 0 = worst, 1 = best
    most_selected: list[tuple[str, float]]  # variants most often picked in-sample, with share
    n_variants: int
    n_splits: int
    blocks: int
    bars_used: int
    provenance: str

    def verdict(self) -> str:
        if self.pbo >= 0.6:
            return (f"Picking by backtest did worse than picking at random: the in-sample winner "
                    f"finished at or below the median out of sample in {self.pbo:.0%} of splits.")
        if self.pbo >= 0.4:
            return (f"Picking by backtest is close to a coin flip: the in-sample winner finished at "
                    f"or below the median out of sample in {self.pbo:.0%} of splits.")
        if self.pbo >= 0.2:
            return (f"Picking by backtest helps only somewhat: the in-sample winner finished at or "
                    f"below the median out of sample in {self.pbo:.0%} of splits.")
        return (f"The in-sample winner usually stays good: it finished at or below the median out "
                f"of sample in only {self.pbo:.0%} of splits. (Not a significance test: on grids "
                f"of pure noise, PBO read below 0.2 in 3-10% of runs.)")

    def __str__(self) -> str:
        top = ", ".join(f"{n} ({s:.0%})" for n, s in self.most_selected)
        lines = [
            f"Probability of backtest overfitting: {self.pbo:.2f} ({self.n_variants} variants, "
            f"{self.n_splits} in/out-of-sample splits of {self.blocks} blocks).",
            self.verdict(),
            f"The in-sample winner's median Sharpe was {self.median_is_sharpe:.2f} in sample and "
            f"{self.median_oos_sharpe:.2f} out of sample; it lost money out of sample in "
            f"{self.prob_oos_loss:.0%} of splits, and its median out-of-sample rank was the "
            f"{self.median_oos_percentile:.0%} percentile.",
            f"Picked most often in-sample: {top}.",
            f"[CSCV, Bailey et al. 2017 | {self.provenance}]",
        ]
        return "\n".join(lines)


def _matrix(np, variants: Any):
    from .backtest import _as_columns

    names, cols = _as_columns(variants)
    if len(names) < 2:
        raise ValueError("probability_of_overfitting needs at least two variants")
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    try:
        X = np.vstack([np.asarray(c, dtype=float) for c in cols])
    except ValueError as e:
        raise ValueError("every variant must be a numeric series of the same length") from e
    return names, X[:, ~np.isnan(X).any(axis=0)]


def probability_of_overfitting(
    variants: Any, *, blocks: int = 16, periods_per_year: float = 252
) -> OverfitResult:
    """PBO by combinatorially symmetric cross-validation.

    `variants`: every variant you tried, as a dict {name: per-bar returns},
    a pandas DataFrame or a 2-D array with one column per variant -- the
    same input `check_backtest` takes. Bars missing in any variant are
    dropped. `blocks` must be even; the history is cut into that many
    contiguous blocks (the first few bars are dropped so they divide evenly).
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("probability_of_overfitting needs numpy (pip install numpy)") from e
    if blocks < 4 or blocks % 2:
        raise ValueError(f"blocks must be an even number of at least 4, got {blocks!r}")
    names, X = _matrix(np, variants)
    k, m = X.shape
    b = m // blocks
    if b < 10:
        raise ValueError(f"{m} usable bars is too few for {blocks} blocks of at least 10 bars")
    X = X[:, m - b * blocks:]
    used = b * blocks

    blk = X.reshape(k, blocks, b)
    S1 = blk.sum(axis=2).T  # blocks x k
    S2 = (blk ** 2).sum(axis=2).T
    tot1, tot2 = S1.sum(axis=0), S2.sum(axis=0)

    half = blocks // 2
    I = np.zeros((math.comb(blocks, half), blocks))
    for row, combo in enumerate(combinations(range(blocks), half)):
        I[row, list(combo)] = 1.0
    nh = half * b

    def sharpe(s1, s2):
        mean = s1 / nh
        var = np.maximum(s2 / nh - mean ** 2, 0.0) * nh / (nh - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, mean / np.sqrt(var), 0.0)

    is1, is2 = I @ S1, I @ S2
    is_sr = sharpe(is1, is2)
    oos_sr = sharpe(tot1 - is1, tot2 - is2)

    rows = np.arange(len(I))
    star = np.argmax(is_sr, axis=1)
    oos_star = oos_sr[rows, star]
    # Rank 1 = worst. Ties share the average rank, so a winner tied with
    # everything sits at the median rather than at an arbitrary end.
    below = (oos_sr < oos_star[:, None]).sum(axis=1)
    ties = (oos_sr == oos_star[:, None]).sum(axis=1)
    rank = below + (ties + 1) / 2
    omega = rank / (k + 1)
    pbo = float(np.mean(omega <= 0.5))

    is_star = is_sr[rows, star]
    ann = math.sqrt(periods_per_year)
    counts = np.bincount(star, minlength=k) / len(I)
    top = [(names[j], float(counts[j])) for j in np.argsort(-counts, kind="stable")[:3] if counts[j] > 0]

    return OverfitResult(
        pbo=pbo,
        prob_oos_loss=float(np.mean(oos_star < 0)),
        median_is_sharpe=float(np.median(is_star) * ann),
        median_oos_sharpe=float(np.median(oos_star) * ann),
        median_oos_percentile=float(np.median((rank - 1) / (k - 1))),
        most_selected=top,
        n_variants=k,
        n_splits=len(I),
        blocks=blocks,
        bars_used=used,
        provenance=stamp(),
    )
