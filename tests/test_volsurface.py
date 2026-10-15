import math
from datetime import datetime

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from tokio_ai.volsurface import black_price, implied_vol, parse_cboe_chain, vol_surface

ASOF = datetime(2026, 1, 5, 10, 0)


def _smile(k, T):
    # A skewed smile whose level rises with maturity.
    return 0.18 + 0.02 * math.sqrt(T) - 0.25 * k + 0.6 * k * k


def _synthetic_payload(spot=100.0, r=0.04, q=0.01, expiries=("260116", "260220", "260619", "261218"),
                       half_spread=0.01, root="XYZ"):
    options = []
    for e in expiries:
        exp = datetime.strptime(e, "%y%m%d")
        T = ((exp - ASOF).total_seconds() + 6 * 3600) / (365.25 * 86400)
        F = spot * math.exp((r - q) * T)
        D = math.exp(-r * T)
        for K in np.arange(60, 141, 2.5):
            sig = _smile(math.log(K / F), T)
            for cp in "CP":
                px = D * float(black_price(F, K, T, sig, cp == "C"))
                bid = max(round(px - half_spread, 2), 0.0)
                options.append({"option": f"{root}{e}{cp}{int(round(K * 1000)):08d}", "bid": bid,
                                "ask": round(px + half_spread, 2), "iv": sig, "open_interest": 10, "volume": 1})
    return {"timestamp": ASOF.strftime("%Y-%m-%d %H:%M:%S"),
            "data": {"symbol": root, "current_price": spot, "options": options}}


def test_black_round_trip():
    F, K, T = 100.0, np.array([80.0, 100.0, 125.0]), 0.5
    sig = np.array([0.35, 0.2, 0.15])
    for is_call in (True, False):
        px = black_price(F, K, T, sig, is_call)
        assert np.allclose(implied_vol(px, F, K, T, is_call), sig, atol=1e-8)


def test_implied_vol_nan_below_intrinsic_and_above_bound():
    iv = implied_vol(np.array([5.0, 0.0, 101.0]), 100.0, np.array([90.0, 110.0, 50.0]), 0.5, True)
    assert np.isnan(iv).all()


def test_parse_cboe_chain_and_index_symbol():
    chain, meta = parse_cboe_chain(_synthetic_payload())
    assert set(chain.type) == {"call", "put"}
    assert chain.strike.min() == 60.0 and meta["spot"] == 100.0 and meta["asof"] == ASOF
    with pytest.raises(RuntimeError):
        parse_cboe_chain({"data": {"options": []}}, "NOPE")


def test_recovers_known_surface_european_index():
    # Index symbols take the rate from put-call parity, so no network is used.
    surf = vol_surface(parse_cboe_chain(_synthetic_payload(root="SPX", half_spread=0.005)))
    assert surf.rate_source == "put-call parity"
    assert abs(surf.rate - 0.04) < 0.003
    fw = surf.forwards
    true_F = 100 * np.exp(0.03 * fw.dte / 365.25)
    assert np.allclose(fw.forward, true_F, rtol=2e-4)
    p = surf.points
    truth = np.array([_smile(k, T) for k, T in zip(np.log(p.strike / (100 * np.exp(0.03 * p["T"]))), p["T"])])
    assert np.median(np.abs(p.iv - truth)) < 0.002
    # Out-of-the-money only, and every mid vol lies inside its bid/ask vols.
    assert ((p.type == "put") == (p.strike < p.forward)).all()
    assert ((p.iv_bid <= p.iv + 1e-12) & (p.iv <= p.iv_ask + 1e-12)).all()


def test_given_rate_and_summary_text():
    surf = vol_surface(parse_cboe_chain(_synthetic_payload()), rate=0.04)
    assert surf.rate_source == "given"
    text = str(surf)
    assert "ATM vol" in text and "Skew" in text and "Black-76 on American options" in text
    ts = surf.atm_term_structure()
    assert (ts.atm_iv.diff().dropna() > 0).all()  # level rises with maturity by construction
    sm = surf.smile()
    assert sm.expiry.nunique() == 1 and abs(sm.dte.iloc[0] - 46) < 2


def test_grid_does_not_extrapolate():
    surf = vol_surface(parse_cboe_chain(_synthetic_payload()), rate=0.04)
    k, T, iv = surf.grid(k_points=11, k_range=(-1.5, 1.5))
    assert iv.shape == (len(T), 11)
    assert np.isnan(iv[:, 0]).all() and np.isnan(iv[:, -1]).all()
    assert np.isfinite(iv[:, 5]).all()


def test_wide_quotes_are_dropped_and_counted():
    surf = vol_surface(parse_cboe_chain(_synthetic_payload(half_spread=0.4)), rate=0.04, max_rel_spread=0.2)
    assert surf.dropped["no bid or spread too wide"] > 0


def test_plot_builds():
    pytest.importorskip("plotly")
    fig = vol_surface(parse_cboe_chain(_synthetic_payload()), rate=0.04).plot(min_dte=1)
    assert fig.data[0].type == "surface" and fig.data[1].visible == "legendonly"
