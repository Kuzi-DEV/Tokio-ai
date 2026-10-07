"""Exact test for bets on binary contracts: do you win more often than the price says?

Prediction-market contracts, binary options, sports bets at known odds: each
trade pays 1 if its side wins and 0 if not, and you paid a known price. That
knowledge makes a far better test possible than anything that has to
estimate a variance from the P&L.

It matters most for favourites. Buying at 95c wins 5c nineteen times and
loses 95c once, on average. A short record with no loss yet looks like a
steady stream of small wins with almost no variance, and any test that
estimates the variance from the P&L -- a t-test, a HAC, a bootstrap --
calls that hugely significant. Measured on fairly priced contracts (no
edge at all): at 98c, a t-test fired 59% of the time over 30 trades, and
check_backtest's HAC 12-16% over 100-300 trades.

Here nothing is estimated. The null hypothesis is the least favourable one
for a claimed edge: each contract wins with probability exactly equal to its
breakeven price (price plus fees), so the expected P&L is zero. Under that
null the distribution of the P&L is known exactly -- a sum of independent
weighted Bernoulli outcomes -- and the p-value is the probability of doing
at least as well as you did.

- Equal sizes: exact, by the Poisson-binomial recursion.
- Mixed integer sizes (contract counts): exact, by the same recursion over
  contracts won.
- Books too large for either recursion, or fractional sizes: Lugannani-Rice
  saddlepoint approximation with Daniels' lattice continuity correction.
  It matches the exact answer to 3-4 digits for equal sizes, down to
  p = 1e-62, but it is NOT used on small books with uneven sizes, where the
  P&L is lumpy and it came out 36% too small (0.071 vs 0.111 simulated).

It also reports an exact (Clopper-Pearson) upper bound on the true loss
rate next to the breakeven loss rate, for books of favourites where the loss
rate is the whole business.

Assumption: outcomes are independent across entries. Several fills on the
same contract are one bet, not several -- pass `groups=` (e.g. the market
ticker) and they are merged. Different contracts that resolve on the same
underlying move (BTC, SOL and XRP in the same 15 minutes) are correlated in
a way no argument here can know; test them per asset, or keep one entry per
window.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from .rigor.provenance import stamp

EXACT_MAX_N = 20000
EXACT_LATTICE_MAX = 3e8  # n x total contracts (time)
EXACT_LATTICE_MAX_TOTAL = 5e7  # total contracts (memory: one float each)
_N = NormalDist()


@dataclass(frozen=True)
class ContractsResult:
    verdict: str
    p_value: float
    # The other tail: how often a fairly priced book does this badly or
    # worse. Small means the entries are overpaying -- losing to the price.
    p_worse: float
    method: str  # "exact" or "saddlepoint"
    n_bets: int
    n_losses: int
    pnl: float  # net, in price units x size (dollars if prices are in dollars)
    expected_losses: float  # under the null: sum of (1 - breakeven)
    loss_rate: float
    breakeven_loss_rate: float  # mean over bets of 1 - breakeven
    loss_rate_upper95: float  # Clopper-Pearson one-sided 95% upper bound
    alpha: float
    merged: int  # entries merged into other bets by `groups`
    # Share of CONTRACTS lost and its breakeven, which is what the p-value and
    # P&L weigh. Equal to the per-bet figures when every bet is the same size.
    contract_loss_rate: float
    contract_breakeven_loss_rate: float
    equal_sizes: bool
    # Bets at this book's mean breakeven price for an all-wins record to
    # reach alpha: below it, no outcome at all could be significant.
    bets_for_perfect_record: int
    provenance: str
    # How far the cost per contract (price + fee) could be off, in price
    # units, before the verdict flips: up for a significant edge, down for a
    # significant "worse than fair". None when there is nothing to flip.
    cost_margin_edge: float | None = None
    cost_margin_worse: float | None = None
    n_exited: int = 0  # bets closed before settlement (stop-loss, take-profit, any rule)
    mean_exit_value: float = math.nan  # what those exits returned per contract, on average
    exit_pnl_vs_settlement: float = math.nan  # exits' P&L minus what holding would have made

    @property
    def significant(self) -> bool:
        return self.verdict == "SIGNIFICANT"

    def __str__(self) -> str:
        head = (
            f"{self.verdict} (p={self.p_value:.4f}, alpha={self.alpha}, {self.method}). "
            f"{self.n_bets} bets, net P&L {self.pnl:+.4f}."
        )
        per = "" if self.equal_sizes else " per bet"
        what = "Lost money (settled at 0 or exited below breakeven)" if self.n_exited else "Losses"
        lines = [
            head,
            f"{what}: {self.n_losses} of {self.n_bets} bets ({self.loss_rate:.2%}). If every "
            f"contract were fairly priced at its breakeven (price plus fees) you'd expect "
            f"{self.expected_losses:.1f} settled losses ({self.breakeven_loss_rate:.2%}).",
        ]
        if self.n_exited:
            lines.append(
                f"{self.n_exited} bets were closed early, returning {self.mean_exit_value:.4f} per "
                f"contract on average. Against holding them to settlement, the exits made "
                f"{self.exit_pnl_vs_settlement:+.4f}. (Early exits are tested conservatively: the "
                f"p-value treats every position as if it could still end at 0 or 1, so a stop "
                f"that genuinely cuts risk gets no credit for it.)"
            )
        if not self.equal_sizes:
            lines.append(
                f"Weighted by size (what the P&L and p-value count): {self.contract_loss_rate:.2%} "
                f"of contracts lost vs {self.contract_breakeven_loss_rate:.2%} at breakeven."
            )
        if self.loss_rate_upper95 < self.breakeven_loss_rate:
            tail = " -- still below breakeven."
        elif self.p_worse <= self.alpha or not self.equal_sizes:
            tail = f" -- above breakeven ({self.breakeven_loss_rate:.2%})."
        else:
            tail = (f" -- above breakeven ({self.breakeven_loss_rate:.2%}), so the data can't yet "
                    f"rule out a losing book.")
        lines.append(f"95% upper bound on the true loss rate{per}: {self.loss_rate_upper95:.2%}" + tail)
        if self.p_worse <= self.alpha:
            lines.append(
                f"Worse than fair: a book priced exactly at breakeven does this badly only "
                f"{self.p_worse:.2%} of the time. The entries are paying more than their win rate "
                f"is worth."
            )
        for margin, what, way in ((self.cost_margin_edge, "edge", "higher"),
                                  (self.cost_margin_worse, "'worse than fair' finding", "lower")):
            if margin is None:
                continue
            lines.append(
                f"Robustness: the {what} disappears if the true cost per contract is "
                f"{margin * 100:.2f}c {way} than the prices and fees given."
                + (" That is within a typical recording error (limit vs fill price, rounding, "
                   "fees) -- check the prices before trusting it." if margin < 0.01 else "")
            )
        if self.n_losses == 0:
            lines.append(
                "No losses yet. Every variance estimated from this P&L would be near zero, "
                "which is exactly when a t-test is most wrong; this test doesn't estimate one."
            )
        if self.p_value > self.alpha and self.bets_for_perfect_record > self.n_bets:
            lines.append(
                f"At these prices even a perfect record needs about {self.bets_for_perfect_record} "
                f"bets to be significant; no result from {self.n_bets} can be."
            )
        if self.merged:
            lines.append(f"{self.merged} entries were merged into the bet on the same contract (groups=).")
        null = "each position worth breakeven on average, bounded in [0, 1]" if self.n_exited \
            else "win prob equals breakeven"
        lines.append(f"[poisson-binomial {self.method}, null = {null} | {self.provenance}]")
        return "\n".join(lines)


def _pb_upper_tail_exact(np, q, k):
    """P(K >= k) for K a sum of independent Bernoulli(q_i)."""
    pmf = np.zeros(len(q) + 1)
    pmf[0] = 1.0
    for i, qi in enumerate(q):
        top = i + 2
        pmf[1:top] = pmf[1:top] * (1 - qi) + pmf[: top - 1] * qi
        pmf[0] *= 1 - qi
    return float(min(1.0, pmf[k:].sum()))


def _lattice_upper_tail_exact(np, w, q, s):
    """P(S >= s) for S = sum w_i * Bernoulli(q_i), w_i positive integers.

    The same recursion as the equal-size case, over total contracts won
    instead of bets won. Exact; costs n * sum(w).
    """
    total = int(w.sum())
    pmf = np.zeros(total + 1)
    pmf[0] = 1.0
    top = 0
    for wi, qi in zip(w.astype(np.int64), q):
        top += int(wi)
        new = pmf[: top + 1] * (1 - qi)
        new[wi:] += pmf[: top + 1 - wi] * qi
        pmf[: top + 1] = new
    return float(min(1.0, pmf[int(round(s)):].sum()))


def _saddlepoint_upper(np, w, q, s, span):
    """P(S >= s) for S = sum w_i * Bernoulli(q_i), Lugannani-Rice.

    `span`: lattice span of S (gcd of integer sizes), 0 if not a lattice.
    """
    lo_s, hi_s = 0.0, float(w.sum())
    if s <= lo_s:
        return 1.0
    if s > hi_s + 1e-12 * max(1.0, hi_s):
        return 0.0
    if span and s >= hi_s - span / 2:
        return float(np.exp(np.log(q).sum()))  # only "every bet won" reaches the top
    x = s - span / 2 if span else s
    if x <= 0:
        return 1.0
    mean = float((w * q).sum())
    var = float((w * w * q * (1 - q)).sum())
    if var <= 0:
        return 1.0 if x <= mean else 0.0

    def k1(t):  # K'(t) and K''(t)
        e = q * np.exp(np.clip(w * t, -700, 700))
        pt = e / (1 - q + e)
        return float((w * pt).sum()), float((w * w * pt * (1 - pt)).sum())

    # K' is increasing: bracket, then bisect in t.
    a, b = -1.0, 1.0
    while k1(a)[0] > x:
        a *= 2
    while k1(b)[0] < x:
        b *= 2
        if b > 1e6:
            return 0.0
    t = 0.0
    for _ in range(200):
        t = (a + b) / 2
        m1, _ = k1(t)
        if m1 < x:
            a = t
        else:
            b = t
        if b - a < 1e-12 * max(1.0, abs(t)):
            break
    m1, m2 = k1(t)
    K = float(np.log(1 - q + q * np.exp(np.clip(w * t, -700, 700))).sum())
    if abs(t) < 1e-10:
        # x sits on the mean. Lugannani-Rice's limit there is not 1/2 but
        # 1/2 - kappa3 / (6 sqrt(2 pi) kappa2^1.5). (The lattice correction
        # has already moved x by half a span; it adds nothing more here.)
        k3 = float((w ** 3 * q * (1 - q) * (1 - 2 * q)).sum())
        p = 0.5 - k3 / (6 * math.sqrt(2 * math.pi) * var ** 1.5)
        return float(min(1.0, max(0.0, p)))
    r = math.copysign(math.sqrt(max(2 * (t * x - K), 0.0)), t)
    # Daniels (1987) second continuity correction on a lattice of span h:
    # solve at s - h/2 and replace t by 2 sinh(t h / 2) / h.
    tt = 2 * math.sinh(t * span / 2) / span if span else t
    u = tt * math.sqrt(m2)
    if r == 0 or u == 0:
        return 0.5
    p = 1 - _N.cdf(r) + _N.pdf(r) * (1 / u - 1 / r)
    return float(min(1.0, max(0.0, p)))


def _clopper_pearson_upper(np, losses: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper confidence bound on a binomial proportion (exact)."""
    if losses >= n:
        return 1.0
    i = np.arange(losses + 1)
    logc = np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in i])

    def cdf(p):
        return float(np.exp(logc + i * math.log(p) + (n - i) * math.log1p(-p)).sum())

    # Solve P(Binom(n, p) <= losses) = 1 - conf for p; the cdf falls in p.
    lo, hi = losses / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid >= 1.0 or mid <= 0.0:
            break
        if cdf(mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def _upper_tail(np, w, q, s):
    """(method, P(S >= s)) for S = sum w_i * Bernoulli(q_i).

    Weights that are integers after scaling by 100 or 10,000 go through the
    exact lattice recursion on that scale.
    """
    for scale in (1, 100, 10000):
        ws = w * scale
        if (np.all(np.abs(ws - np.round(ws)) < 1e-6 * scale) and len(q) * ws.sum() <= EXACT_LATTICE_MAX
                and ws.sum() <= EXACT_LATTICE_MAX_TOTAL):
            return _upper_tail_scaled(np, np.round(ws), q, s * scale)
    return _upper_tail_scaled(np, w, q, s)


def _upper_tail_scaled(np, w, q, s):
    n = len(q)
    equal = bool(np.all(w == w[0]))
    integer = bool(np.all(np.abs(w - np.round(w)) < 1e-9))
    if equal and n <= EXACT_MAX_N:
        # S >= s  <=>  bets won >= ceil(s / w): thresholds between lattice
        # points (a stop filled below its level) round up, never down.
        return "exact", _pb_upper_tail_exact(np, q, max(0, math.ceil(s / w[0] - 1e-9)))
    if integer and n * w.sum() <= EXACT_LATTICE_MAX and w.sum() <= EXACT_LATTICE_MAX_TOTAL:
        wi = np.round(w).astype(np.int64)
        g = int(np.gcd.reduce(wi))
        return "exact", _lattice_upper_tail_exact(np, wi // g, q, max(0, math.ceil(s / g - 1e-9)))
    # Only for books too big for the exact recursions, where the sum is
    # smooth enough for the approximation. On 40 bets with sizes 1-30 it was
    # 36% too small against simulation; on 200 bets, within 5%.
    span = float(np.gcd.reduce(np.round(w).astype(np.int64))) if integer else 0.0
    return "saddlepoint", _saddlepoint_upper(np, w, q, s, span)


def check_contracts(
    prices: Iterable[Any],
    outcomes: Iterable[Any],
    *,
    sizes: Iterable[Any] | float = 1.0,
    fees: Iterable[Any] | float = 0.0,
    groups: Iterable[Any] | None = None,
    exit_values: Iterable[Any] | None = None,
    alpha: float = 0.05,
) -> ContractsResult:
    """Do these bets win more often than their prices imply, net of fees?

    `prices`: what you paid per contract for the side you bought, as a
    probability (0.95, not 95 cents). `outcomes`: 1/True if that side won,
    0/False if not. `sizes`: contracts per bet (default 1). `fees`: cost per
    contract in the same units as the price (0.003 = 0.3c), a number or one
    per bet. `groups`: optional label per bet; bets with the same label are
    the same contract and are merged into one (their outcomes must agree).

    The null is that every bet wins with probability price + fee exactly, so
    its expected P&L is zero. SIGNIFICANT means doing this well would be
    rarer than `alpha` under that null.

    Early exits (stop-losses, take-profits, any rule at all): `exit_values`
    gives, for each bet closed before settlement, what the exit returned
    per contract net of the exit fee; None/NaN for bets held to the end.
    Each position's final value then lies between 0 and 1, and if the
    market is fair its expected value is the breakeven whatever the exit
    rule, since a fair price is a martingale. Among all ways of ending
    between 0 and 1 with that mean, the plain win/lose bet is the most
    spread out (Hoeffding 1963: it dominates in convex order), so testing
    the realized values against the no-exit null is conservative.

    Every significant verdict comes with a robustness margin: how much the
    true cost per contract could differ from the prices and fees given before
    it flips. A verdict that flips on a 1c error rests on the bookkeeping, not
    the market -- recording the limit price instead of the fill, a fee
    schedule that rounds, a price quoted in the wrong side's terms.

    A sharper null was tried and dropped: a stop at level L ends at L or 1,
    a two-point bet with a smaller spread. On simulated fair 15-minute
    markets it fired 7-13% of the time at a nominal 5%, because real exits
    fill below their stops and prices jump through them near expiry. The
    bound here fired at most 3.0%. The cost is power: a stop that really
    does cut risk gets no credit for it.
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("check_contracts needs numpy (pip install numpy)") from e
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    p = np.asarray(list(prices) if not hasattr(prices, "__len__") else prices, dtype=float)
    o = np.asarray(list(outcomes) if not hasattr(outcomes, "__len__") else outcomes, dtype=float)
    n0 = len(p)
    if p.ndim != 1 or o.shape != p.shape:
        raise ValueError(f"prices and outcomes must be 1-D and the same length, got {p.shape}, {o.shape}")
    w = np.broadcast_to(np.asarray(sizes, dtype=float), p.shape).astype(float)
    f = np.broadcast_to(np.asarray(fees, dtype=float), p.shape).astype(float)
    if np.isnan(p).any() or np.isnan(o).any() or np.isnan(w).any() or np.isnan(f).any():
        raise ValueError("prices, outcomes, sizes and fees must not contain missing values")
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("prices must be probabilities strictly between 0 and 1 (0.95, not 95)")
    if not np.isin(o, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0/1 or False/True")
    if (w <= 0).any():
        raise ValueError("sizes must be positive")
    if (f < 0).any():
        raise ValueError("fees must be non-negative")
    q = p + f
    if (q >= 1).any():
        raise ValueError("price + fee reaches 1 on some bet: it cannot profit even when it wins")
    if exit_values is None:
        ex = np.full(n0, np.nan)
    else:
        ex = np.array([np.nan if v is None else float(v) for v in exit_values], dtype=float)
        if ex.shape != p.shape:
            raise ValueError(f"exit_values has length {len(ex)}, prices has {n0}")
        closed = ~np.isnan(ex)
        if ((ex[closed] < 0) | (ex[closed] > 1)).any():
            raise ValueError("exit_values must be between 0 and 1 (net proceeds per contract)")
    v = np.where(np.isnan(ex), o, ex)  # what each contract finally returned

    merged = 0
    if groups is not None:
        g = list(groups)
        if len(g) != n0:
            raise ValueError(f"groups has length {len(g)}, prices has {n0}")
        order: dict[Any, list[int]] = {}
        for i, key in enumerate(g):
            order.setdefault(key, []).append(i)
        P, O, W, V, E = [], [], [], [], []
        for key, idx in order.items():
            oo = o[idx]
            if (oo != oo[0]).any():
                raise ValueError(f"group {key!r} has bets with different outcomes; groups must be "
                                 f"the same contract")
            if not (np.isnan(ex[idx]).all() or (ex[idx] == ex[idx][0]).all()):
                raise ValueError(f"group {key!r} mixes exits; merge only fills that were managed "
                                 f"as one position")
            ww = w[idx]
            P.append(float((ww * q[idx]).sum() / ww.sum()))  # the size-weighted breakeven
            O.append(oo[0])
            W.append(float(ww.sum()))
            V.append(v[idx][0])
            E.append(ex[idx][0])
        merged = n0 - len(P)
        q, o, w, v, ex = np.array(P), np.array(O), np.array(W), np.array(V), np.array(E)

    n = len(q)
    if n == 0:
        raise ValueError("no bets")
    pnl = float((w * (v - q)).sum())
    s_obs = float((w * v).sum())
    method, pv = _upper_tail(np, w, q, s_obs)
    # P(S <= s) is P(contracts lost >= the shortfall from winning them all):
    # the same upper tail, for the losing side of every bet.
    _, p_worse = _upper_tail(np, w, 1 - q, float(w.sum()) - s_obs)

    def flips(sign, alpha_):
        """Largest uniform shift d (cost moved by sign*d) that keeps the tail significant."""
        cap = float((1 - q).min()) if sign > 0 else float(q.min())
        hi = cap * (1 - 1e-9)
        lo = 0.0

        def still(d):
            qq = q + sign * d
            if sign > 0:
                return _upper_tail(np, w, qq, s_obs)[1] <= alpha_
            return _upper_tail(np, w, 1 - qq, float(w.sum()) - s_obs)[1] <= alpha_

        if hi <= 0:
            return None
        if still(hi):
            return hi
        while hi - lo > 1e-5:  # 0.001c
            mid = (lo + hi) / 2
            if still(mid):
                lo = mid
            else:
                hi = mid
        return lo

    margin_edge = flips(+1, alpha) if pv <= alpha else None
    margin_worse = flips(-1, alpha) if p_worse <= alpha else None

    lost = v < q - 1e-12
    losses = int(lost.sum())
    wsum = float(w.sum())
    be_loss = float((1 - q).mean())  # per bet, like the observed loss rate and its bound
    closed = ~np.isnan(ex)
    return ContractsResult(
        verdict="SIGNIFICANT" if pv <= alpha else "NOT SIGNIFICANT",
        p_value=pv,
        p_worse=p_worse,
        method=method,
        n_bets=n,
        n_losses=losses,
        pnl=pnl,
        expected_losses=float((1 - q).sum()),
        loss_rate=losses / n,
        breakeven_loss_rate=be_loss,
        loss_rate_upper95=_clopper_pearson_upper(np, losses, n),
        alpha=alpha,
        merged=merged,
        contract_loss_rate=float((w * lost).sum() / wsum),
        contract_breakeven_loss_rate=float((w * (1 - q)).sum() / wsum),
        equal_sizes=bool(np.all(w == w[0])),
        bets_for_perfect_record=math.ceil(math.log(alpha) / math.log(float(q.mean()))),
        cost_margin_edge=margin_edge,
        cost_margin_worse=margin_worse,
        n_exited=int(closed.sum()),
        mean_exit_value=float(ex[closed].mean()) if closed.any() else math.nan,
        exit_pnl_vs_settlement=float((w * (ex - o))[closed].sum()) if closed.any() else math.nan,
        provenance=stamp(),
    )
