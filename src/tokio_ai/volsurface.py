"""Implied-volatility surfaces from free, delayed option quotes.

`vol_surface("SPY")` downloads Cboe's delayed option chain (no key, ~15 min
delay), and computes every implied volatility itself rather than trusting the
vendor's number:

* The forward price F and discount factor D for each expiry come from put-call
  parity, C - P = D*(F - K), fitted on the strikes nearest the money. No
  interest-rate or dividend assumption is needed, and the implied carry is
  reported so you can sanity-check it.
* Only out-of-the-money options are used (puts below F, calls above), where
  quotes are tightest and early exercise matters least.
* Each option gets three vols: at the bid, the mid and the ask. The bid-ask
  vol width is how much of the surface is real and how much is spread. Wide
  wings are flagged, not hidden.

Caveats, all stated in the output: US equity and ETF options are American; the
Black (European) formula is a close approximation for out-of-the-money options
but not exact. Quotes are delayed. Expiries are treated as 16:00 New York time.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

_CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
# Cash-settled indices are listed under an underscore on Cboe's feed.
_INDICES = {"SPX", "NDX", "RUT", "VIX", "XSP", "DJX", "OEX", "XEO", "MRUT"}
_OPTION_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
_YEAR_SECONDS = 365.25 * 86400
_NCDF = np.frompyfunc(lambda z: 0.5 * math.erfc(-z / math.sqrt(2.0)), 1, 1)


def _ncdf(z):
    return np.asarray(_NCDF(np.asarray(z, dtype=float)), dtype=float)


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas ships with every notebook
        raise ImportError("vol_surface needs pandas: pip install pandas") from exc
    return pd


def _cboe_symbol(symbol: str) -> str:
    s = symbol.strip().upper().lstrip("^").lstrip("_")
    return f"_{s}" if s in _INDICES else s


def fetch_option_chain(symbol: str, timeout: float = 30.0):
    """Cboe's delayed option chain for `symbol` as a DataFrame, plus quote metadata.

    Returns ``(chain, meta)``: one row per contract with expiry, strike, type,
    bid, ask, open interest, volume and Cboe's own iv; `meta` holds the spot
    price and the quote timestamp (New York time).
    """
    url = _CBOE_URL.format(symbol=_cboe_symbol(symbol))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tokio-ai)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # follows Cboe's redirect
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_cboe_chain(payload, symbol)


def parse_cboe_chain(payload: dict, symbol: str = ""):
    """Turn a Cboe delayed-quotes JSON payload into ``(chain, meta)``."""
    pd = _pandas()
    data = payload.get("data") or {}
    rows = []
    for o in data.get("options") or []:
        m = _OPTION_RE.match(str(o.get("option", "")))
        if not m:
            continue
        rows.append({
            "expiry": datetime.strptime(m["date"], "%y%m%d"),
            "strike": int(m["strike"]) / 1000.0,
            "type": "call" if m["cp"] == "C" else "put",
            "bid": float(o.get("bid") or 0.0),
            "ask": float(o.get("ask") or 0.0),
            "open_interest": float(o.get("open_interest") or 0.0),
            "volume": float(o.get("volume") or 0.0),
            "vendor_iv": float(o.get("iv") or 0.0),
        })
    if not rows:
        raise RuntimeError(f"Cboe returned no options for {symbol or data.get('symbol')!r}")
    chain = pd.DataFrame(rows)
    ts = payload.get("timestamp")
    meta = {
        "symbol": data.get("symbol", symbol),
        "spot": float(data.get("current_price") or data.get("close") or float("nan")),
        "asof": datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") if ts else datetime.now(),
        "source": "Cboe delayed quotes",
    }
    return chain, meta


def black_price(F, K, T, sigma, is_call):
    """Undiscounted Black-76 price."""
    F, K, T, sigma = (np.asarray(a, dtype=float) for a in (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    call = F * _ncdf(d1) - K * _ncdf(d2)
    put = call - (F - K)
    return np.where(is_call, call, put)


def implied_vol(price, F, K, T, is_call, lo: float = 1e-4, hi: float = 5.0, iters: int = 60):
    """Black-76 implied vol of an undiscounted price, by vectorised bisection.

    NaN where the price is at or below intrinsic value or above the no-arbitrage
    bound (no vol reproduces it).
    """
    price, F, K, T = (np.asarray(a, dtype=float) for a in (price, F, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = np.where(is_call, F, K)
    ok = (price > intrinsic + 1e-12) & (price < upper) & (T > 0)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    ok &= black_price(F, K, T, b, is_call) > price
    for _ in range(iters):
        mid = 0.5 * (a + b)
        high = black_price(F, K, T, mid, is_call) > price
        b = np.where(high, mid, b)
        a = np.where(high, a, mid)
    return np.where(ok, 0.5 * (a + b), np.nan)


def _parity_pairs(calls, puts):
    both = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    both = both[(both.bid_c > 0) & (both.bid_p > 0) & (both.ask_c > both.bid_c) & (both.ask_p > both.bid_p)]
    if len(both) < 3:
        return None
    diff = (both.bid_c + both.ask_c) / 2 - (both.bid_p + both.ask_p) / 2
    # The money is where C - P crosses zero; use the strikes closest to it.
    centre = both.strike.iloc[int(np.argmin(np.abs(diff.to_numpy())))]
    near = both.assign(diff=diff, dist=np.abs(both.strike - centre)).nsmallest(10, "dist")
    spread = (near.ask_c - near.bid_c) + (near.ask_p - near.bid_p)
    return near.assign(w=1.0 / np.maximum(spread.to_numpy(), 1e-3) ** 2)


def _parity_fit(near):
    """Free fit of C - P = D*F - D*K. Returns (F, D)."""
    X = np.column_stack([np.ones(len(near)), near.strike.to_numpy()])
    W = np.sqrt(near.w.to_numpy())
    coef, *_ = np.linalg.lstsq(X * W[:, None], near["diff"].to_numpy() * W, rcond=None)
    D = -coef[1]
    if not (0.5 < D < 1.05):
        return float("nan"), float("nan")
    return float(coef[0] / D), float(D)


def _tbill_rate() -> float:
    """13-week T-bill yield from Yahoo (^IRX), continuously compounded; NaN on failure."""
    try:
        from .tools.prices import fetch_daily_bars
        y = fetch_daily_bars("^IRX", "5d")[-1].close / 100.0
    except Exception:  # noqa: BLE001 - network or parse failure -> caller falls back
        return float("nan")
    return math.log1p(y) if 0 <= y < 0.5 else float("nan")


def _forward_given_discount(near, D):
    """F = K + (C - P) / D, averaged with spread weights."""
    f = near.strike.to_numpy() + near["diff"].to_numpy() / D
    return float(np.average(f, weights=near.w.to_numpy()))


@dataclass
class VolSurface:
    """An implied-volatility surface measured from one option-chain snapshot.

    `points` holds one row per usable out-of-the-money option (iv, iv_bid,
    iv_ask, log-moneyness k = ln(K/F), years to expiry T). `forwards` holds the
    parity-implied forward and discount factor per expiry.
    """

    symbol: str
    spot: float
    asof: datetime
    points: Any
    forwards: Any
    source: str
    dropped: dict
    rate: float = float("nan")
    rate_source: str = ""
    european: bool = False

    # ---- views -----------------------------------------------------------------
    def atm_term_structure(self):
        """ATM implied vol per expiry (interpolated at k=0 in total variance)."""
        pd = _pandas()
        out = []
        for exp, g in self.points.groupby("expiry"):
            g = g.sort_values("k")
            if g.k.min() > 0 or g.k.max() < 0:
                continue
            T = g["T"].iloc[0]
            w = np.interp(0.0, g.k.to_numpy(), (g.iv.to_numpy() ** 2) * T)
            out.append({"expiry": exp, "dte": g.dte.iloc[0], "atm_iv": math.sqrt(w / T)})
        return pd.DataFrame(out)

    def smile(self, expiry=None):
        """The smile for one expiry (default: the one nearest 30 days)."""
        pts = self.points
        if expiry is None:
            dtes = pts.groupby("expiry").dte.first()
            expiry = (dtes - 30).abs().idxmin()
        expiry = _pandas().Timestamp(expiry)
        return pts[pts.expiry == expiry].sort_values("strike").reset_index(drop=True)

    def grid(self, k_points: int = 41, k_range: tuple[float, float] | None = None, max_dte: float | None = None):
        """Interpolate onto a regular (log-moneyness x maturity) grid.

        Interpolation is linear in total variance (iv^2 * T): along k within each
        expiry, then along T between expiries. No extrapolation: cells outside
        an expiry's quoted strikes are NaN, so the plot never shows a vol nobody
        quoted.
        Returns ``(k, T, iv)`` with iv shaped (len(T), len(k)).
        """
        pts = self.points if max_dte is None else self.points[self.points.dte <= max_dte]
        if k_range is None:
            k_range = (float(np.nanpercentile(pts.k, 2)), float(np.nanpercentile(pts.k, 98)))
        k = np.linspace(k_range[0], k_range[1], k_points)
        Ts, rows = [], []
        for _, g in pts.groupby("expiry"):
            g = g.sort_values("k")
            if len(g) < 3:
                continue
            T = float(g["T"].iloc[0])
            w = np.interp(k, g.k.to_numpy(), (g.iv.to_numpy() ** 2) * T, left=np.nan, right=np.nan)
            Ts.append(T)
            rows.append(w)
        order = np.argsort(Ts)
        T_arr = np.asarray(Ts)[order]
        W = np.asarray(rows)[order]
        with np.errstate(invalid="ignore"):
            iv = np.sqrt(W / T_arr[:, None])
        return k, T_arr, iv

    def plot(self, min_dte: float = 7, max_dte: float = 400, moneyness: tuple[float, float] = (0.8, 1.15),
             k_points: int = 61, show_points: bool = True):
        """Interactive 3D surface (plotly).

        x = strike as % of the forward, y = days to expiry, z = implied vol.
        The quotes behind the surface are a toggleable layer; hovering one shows
        its strike, expiry and bid/ask vol range. Very short expiries are left
        out by default because their wings dwarf everything else.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotting needs plotly: pip install plotly") from exc
        k_range = (math.log(moneyness[0]), math.log(moneyness[1]))
        sub = VolSurface(**{**self.__dict__, "points": self.points[self.points.dte >= min_dte]})
        k, T, iv = sub.grid(k_points=k_points, k_range=k_range, max_dte=max_dte)
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=np.exp(k) * 100, y=T * 365.25, z=iv * 100, colorscale="Viridis", opacity=0.95,
            colorbar=dict(title="IV %", len=0.6), name="surface", connectgaps=False,
            hovertemplate="strike %{x:.0f}% of forward<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>",
        ))
        if show_points:
            p = sub.points[(sub.points.dte <= max_dte) & sub.points.k.between(*k_range)]
            fig.add_trace(go.Scatter3d(
                x=np.exp(p.k) * 100, y=p.dte, z=p.iv * 100, mode="markers", visible="legendonly",
                marker=dict(size=1.5, color="black", opacity=0.6), name="quotes (click to show)",
                customdata=np.column_stack([p.strike, p.iv_bid * 100, p.iv_ask * 100, p.expiry.dt.strftime("%Y-%m-%d")]),
                hovertemplate=("K=%{customdata[0]} exp %{customdata[3]}<br>IV %{z:.1f}% "
                               "(bid %{customdata[1]:.1f} / ask %{customdata[2]:.1f})<extra></extra>"),
            ))
        fig.update_layout(
            title=f"{self.symbol} implied volatility · {self.asof:%Y-%m-%d %H:%M} New York · spot {self.spot:.2f}",
            scene=dict(xaxis_title="strike, % of forward", yaxis_title="days to expiry", zaxis_title="IV %",
                       camera=dict(eye=dict(x=1.25, y=-1.75, z=0.8))),
            legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=50, b=0), height=650,
        )
        return fig

    # ---- summary ---------------------------------------------------------------
    def __str__(self) -> str:
        p = self.points
        lines = [
            f"{self.symbol} implied-volatility surface  ({self.source}, {self.asof:%Y-%m-%d %H:%M} New York, spot {self.spot:.2f})",
            f"{len(p)} out-of-the-money options across {p.expiry.nunique()} expiries "
            f"({int(p.dte.min())}-{int(p.dte.max())} days). IVs computed here from bid/mid/ask, not taken from the vendor.",
        ]
        ts = self.atm_term_structure()
        if len(ts):
            picks = []
            for target in (7, 30, 90, 180, 365):
                i = (ts.dte - target).abs().idxmin()
                row = ts.loc[i]
                if abs(row.dte - target) <= max(3, 0.25 * target) and not any(r[0] == row.dte for r in picks):
                    picks.append((row.dte, row.atm_iv))
            if picks:
                lines.append("ATM vol: " + ", ".join(f"{int(d)}d {v:.1%}" for d, v in picks))
        sm = self.smile()
        if len(sm) >= 5:
            T = sm["T"].iloc[0]
            sd = float(np.nan_to_num(ts.set_index("expiry").atm_iv.get(sm.expiry.iloc[0], np.nan))) * math.sqrt(T)
            if sd > 0:
                kk, vv = sm.k.to_numpy(), sm.iv.to_numpy()
                if kk.min() <= -sd and kk.max() >= sd:
                    down, up = np.interp(-sd, kk, vv), np.interp(sd, kk, vv)
                    lines.append(f"Skew ({int(sm.dte.iloc[0])}d, one standard deviation each side): "
                                 f"downside {down:.1%} vs upside {up:.1%} ({(down - up) * 100:+.1f} vol points).")
        width = (p.iv_ask - p.iv_bid) * 100
        wide = int((width > 5).sum())
        msg = f"Bid-ask vol width: median {np.nanmedian(width):.1f} vol points"
        msg += (f"; {wide} options wider than 5, where the shape is mostly spread, not information." if wide
                else "; none wider than 5.")
        lines.append(msg)
        lines.append(f"Discount rate {self.rate:.2%} ({self.rate_source}); forwards from put-call parity, "
                     f"so no dividend assumption.")
        if self.dropped:
            lines.append("Dropped: " + ", ".join(f"{v} {k}" for k, v in self.dropped.items() if v) + ".")
        lines.append("Quotes delayed." if self.european else
                     "Black-76 on American options (close for out-of-the-money, not exact); quotes delayed.")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle):  # pragma: no cover - IPython display
        p.text(str(self))


def vol_surface(
    symbol_or_chain,
    meta: dict | None = None,
    *,
    min_dte: float = 1.0,
    max_dte: float = 800.0,
    k_range: tuple[float, float] = (-0.6, 0.4),
    max_rel_spread: float = 0.6,
    rate: float | None = None,
) -> VolSurface:
    """Measure an implied-volatility surface.

    Pass a ticker ("SPY", "AAPL", "SPX") to download Cboe's delayed chain, or a
    ``(chain, meta)`` pair from `fetch_option_chain` / `parse_cboe_chain`.

    Options are kept when: out of the money vs the parity forward, bid > 0,
    (ask - bid) / mid <= `max_rel_spread`, ln(K/F) inside `k_range`, and all
    three vols (bid, mid, ask) solve. Everything dropped is counted.

    `rate` (continuously compounded) discounts the quotes. Default: for
    European index options it is read off put-call parity; for American
    equity options parity is bent by early exercise, so the 13-week T-bill
    yield (Yahoo ^IRX) is used, falling back to parity if that download fails.
    """
    pd = _pandas()
    if isinstance(symbol_or_chain, str):
        chain, meta = fetch_option_chain(symbol_or_chain)
    elif isinstance(symbol_or_chain, tuple):
        chain, meta = symbol_or_chain
    else:
        chain = symbol_or_chain
        if meta is None:
            raise ValueError("pass meta={'spot':..., 'asof': datetime} with a chain DataFrame")
    asof = pd.Timestamp(meta["asof"]).to_pydatetime()
    spot = float(meta.get("spot", float("nan")))

    chain = chain.copy()
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    expiry_close = chain["expiry"] + pd.Timedelta(hours=16)
    chain["T"] = (expiry_close - pd.Timestamp(asof)).dt.total_seconds() / _YEAR_SECONDS
    chain["dte"] = chain["T"] * 365.25
    dropped = {"expired or outside the dte window": int((~chain.dte.between(min_dte, max_dte)).sum())}
    chain = chain[chain.dte.between(min_dte, max_dte)]

    # Pass 1: free parity fit per expiry. The slope (discount factor) is only
    # well identified at longer maturities, so one rate is taken from expiries
    # of 60+ days and every forward is then refitted with D = exp(-r T).
    pairs, free_rates = {}, []
    for exp, g in chain.groupby("expiry"):
        near = _parity_pairs(g[g.type == "call"][["strike", "bid", "ask"]], g[g.type == "put"][["strike", "bid", "ask"]])
        if near is None:
            continue
        T = float(g["T"].iloc[0])
        pairs[exp] = near
        F, D = _parity_fit(near)
        if D == D and T * 365.25 >= 60:
            free_rates.append(-math.log(D) / T)
    parity_rate = float(np.median(free_rates)) if free_rates else float("nan")
    european = str(meta.get("symbol", "")).lstrip("_^").upper() in _INDICES
    rate_source = "given"
    if rate is None:
        if european and parity_rate == parity_rate:
            rate, rate_source = parity_rate, "put-call parity"
        else:
            rate, rate_source = _tbill_rate(), "13-week T-bill"
            if rate != rate:
                rate, rate_source = (parity_rate, "put-call parity") if parity_rate == parity_rate else (0.0, "assumed 0")

    fwd_rows, parts = [], []
    no_parity = 0
    for exp, g in chain.groupby("expiry"):
        T = float(g["T"].iloc[0])
        if exp not in pairs:
            no_parity += len(g)
            continue
        D = math.exp(-rate * T)
        F = _forward_given_discount(pairs[exp], D)
        fwd_rows.append({"expiry": exp, "dte": T * 365.25, "forward": F, "discount": D,
                         "carry": math.log(F / spot) / T if spot > 0 else float("nan")})
        otm = ((g.type == "put") & (g.strike < F)) | ((g.type == "call") & (g.strike >= F))
        parts.append(g[otm].assign(forward=F, discount=D))
    dropped["in expiries without a usable put-call parity fit"] = no_parity
    forwards = pd.DataFrame(fwd_rows)
    if not parts:
        raise RuntimeError("no expiry had enough two-sided quotes to infer a forward")
    pts = pd.concat(parts, ignore_index=True)
    n0 = len(pts)
    mid = (pts.bid + pts.ask) / 2
    keep = (pts.bid > 0) & (pts.ask > pts.bid) & ((pts.ask - pts.bid) / mid <= max_rel_spread)
    pts = pts[keep].copy()
    dropped["no bid or spread too wide"] = int(n0 - len(pts))
    pts["k"] = np.log(pts.strike / pts.forward)
    n1 = len(pts)
    pts = pts[pts.k.between(*k_range)].copy()
    dropped["outside the moneyness range"] = int(n1 - len(pts))

    is_call = (pts.type == "call").to_numpy()
    args = (pts.forward.to_numpy(), pts.strike.to_numpy(), pts["T"].to_numpy())
    D = pts.discount.to_numpy()
    pts["mid"] = (pts.bid + pts.ask) / 2
    pts["iv"] = implied_vol(pts.mid.to_numpy() / D, *args, is_call)
    pts["iv_bid"] = implied_vol(pts.bid.to_numpy() / D, *args, is_call)
    pts["iv_ask"] = implied_vol(pts.ask.to_numpy() / D, *args, is_call)
    n2 = len(pts)
    pts = pts.dropna(subset=["iv", "iv_bid", "iv_ask"])
    dropped["no vol solves the quote"] = int(n2 - len(pts))
    cols = ["expiry", "dte", "T", "strike", "type", "k", "forward", "bid", "ask", "mid",
            "iv", "iv_bid", "iv_ask", "vendor_iv", "open_interest", "volume"]
    pts = pts[[c for c in cols if c in pts.columns]].sort_values(["expiry", "strike"]).reset_index(drop=True)
    return VolSurface(symbol=str(meta.get("symbol") or "").lstrip("_^"), spot=spot, asof=asof,
                      points=pts, forwards=forwards, source=meta.get("source", "user chain"), dropped=dropped, rate=rate, rate_source=rate_source, european=european)
