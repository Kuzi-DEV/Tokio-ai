import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from tokio_ai import mcp_server as m


@pytest.fixture(autouse=True)
def fresh_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKIO_AI_CACHE", str(tmp_path))
    m.LEDGER.reset()
    yield
    m.LEDGER.reset()


def _write(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index_label="date")
    return str(p)


def _noise(cols, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), index=idx, columns=cols)


def test_ledger_accumulates_across_calls(tmp_path):
    a = _write(tmp_path, "a.csv", _noise(["v1", "v2", "v3"]))
    b = _write(tmp_path, "b.csv", _noise(["only"], seed=1))
    out1 = m.check_backtest(a)
    assert "3 variant(s) checked so far" in out1
    out2 = m.check_backtest(b, label="second idea")
    assert m.LEDGER.total == 4
    assert "corrected for 4 trials" in out2
    assert "4 trials" in out2.replace("\n", " ")  # check_backtest itself was told about all 4
    assert "second idea" in m.session_trials()


def test_planted_edge_is_found(tmp_path):
    df = _noise(["edge"], n=2500, seed=2) + 0.0015
    out = m.check_backtest(_write(tmp_path, "e.csv", df))
    assert out.startswith("SIGNIFICANT")


def test_columns_and_errors(tmp_path):
    p = _write(tmp_path, "c.csv", _noise(["a", "b"]))
    m.check_backtest(p, columns=["a"])
    assert m.LEDGER.total == 1
    with pytest.raises(ValueError):
        m.check_backtest(p, columns=["nope"])
    with pytest.raises(FileNotFoundError):
        m.check_backtest(str(tmp_path / "missing.csv"))


def test_reset_needs_confirmation(tmp_path):
    m.check_backtest(_write(tmp_path, "r.csv", _noise(["x"])))
    assert "Not reset" in m.reset_session()
    assert m.LEDGER.total == 1
    assert "1 variants forgotten" in m.reset_session(confirm=True)
    assert m.LEDGER.total == 0


def test_check_contracts_tool(tmp_path):
    rng = np.random.default_rng(3)
    price = rng.uniform(0.85, 0.95, 400)
    df = pd.DataFrame({"price": price, "won": (rng.random(400) < price).astype(int)})
    p = tmp_path / "bets.csv"
    df.to_csv(p, index=False)
    out = m.check_contracts(str(p), "price", "won")
    assert "NOT SIGNIFICANT" in out or "not significant" in out.lower()
    assert m.LEDGER.total == 1


def test_backtest_positions_offline(tmp_path, monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=1200)
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
    px = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1}, index=idx)
    import tokio_ai.lab as lab
    monkeypatch.setattr(lab, "load_prices", lambda s: px)
    pos = pd.DataFrame({"long": 1.0}, index=idx)
    out = m.backtest_positions("FAKE", _write(tmp_path, "p.csv", pos))
    assert "earn the NEXT day's return" in out and m.LEDGER.total == 1


def test_server_registers_every_tool():
    pytest.importorskip("mcp")
    import asyncio
    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert names == {f.__name__ for f in m.TOOLS}
