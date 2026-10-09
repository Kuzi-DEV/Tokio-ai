import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from tokio_ai.tools.strategy import (  # noqa: E402
    StrategyBook, UnsafeCode, backtest_strategy, compile_signal, final_test_strategy)

SMA = """
def signal(d, fast=10, slow=50):
    f = d.close.rolling(fast).mean()
    s = d.close.rolling(slow).mean()
    return (f > s).astype(float)
"""


def _bars(symbol="TEST", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.standard_normal(n) * 0.002)
    return pd.DataFrame({"open": open_, "close": close}, index=idx)


def _book():
    return StrategyBook(loader=lambda sym: _bars(sym))


@pytest.mark.parametrize("bad", [
    "import os\ndef signal(d):\n    return d.close * 0",
    "def signal(d):\n    open('x.txt', 'w')\n    return d.close * 0",
    "def signal(d):\n    return d.__class__",
    "def signal(d):\n    pd.read_csv('secret.csv')\n    return d.close * 0",
    "def signal(d):\n    d.to_csv('out.csv')\n    return d.close * 0",
    "def signal(d):\n    return eval('1')",
    "def signal(d):\n    return np.ctypeslib",
    "def signal(d):\n    return pd.io",
    "def other(d):\n    return d.close * 0",
    "x = __import__('os')\ndef signal(d):\n    return d.close * 0",
])
def test_unsafe_or_wrong_code_is_refused(bad):
    with pytest.raises(UnsafeCode):
        compile_signal(bad)


def test_plain_signal_compiles_and_runs():
    fn = compile_signal(SMA, "sma_cross")
    out = fn(_bars(), fast=5, slow=20)
    assert fn.__name__ == "sma_cross" and len(out) == 2000


def test_backtest_strategy_reports_and_counts_variants():
    book = _book()
    text = backtest_strategy(book, "test", "sma_cross", SMA, {"fast": [5, 10], "slow": [50, 100]})
    assert "Ran 4 variant(s)" in text and "4 variant(s) run on TEST" in text
    assert "held out, unseen" in text and "Romano-Wolf" in text
    # the same strategy again: not new trials
    text = backtest_strategy(book, "test", "sma_cross", SMA, {"fast": [5], "slow": [50]})
    assert "4 variant(s) run on TEST" in text
    # different code under the same name: a new strategy, counted
    other = SMA.replace("f > s", "f < s")
    text = backtest_strategy(book, "test", "sma_cross", other, {"fast": [5], "slow": [50]})
    assert "5 variant(s) run on TEST" in text


def test_lookahead_strategy_is_rejected_with_a_message():
    book = _book()
    cheat = "def signal(d):\n    return (d.close.shift(-1) > d.close).astype(float)"
    with pytest.raises(Exception, match="uses future data"):
        backtest_strategy(book, "test", "cheat", cheat)


def test_final_test_once():
    book = _book()
    backtest_strategy(book, "test", "sma_cross", SMA, {"fast": [10], "slow": [50]})
    out = final_test_strategy(book, "test", "sma_cross", SMA, {"fast": 10, "slow": 50})
    assert "holdout data" in out
    with pytest.raises(RuntimeError, match="already been used"):
        final_test_strategy(book, "test", "sma_cross", SMA, {"fast": 10, "slow": 50})


def test_agent_routes_the_tools(monkeypatch):
    pytest.importorskip("openai")
    from tokio_ai.agent.loop import Agent

    a = Agent(api_key="test")
    a.strategies = _book()
    out = a._execute_tool("backtest_strategy", {"symbol": "TEST", "name": "sma", "code": SMA,
                                                "params": {"fast": [10], "slow": [50]}})
    assert "Ran 1 variant(s)" in out
    err = a._execute_tool("backtest_strategy", {"symbol": "TEST", "name": "x",
                                                "code": "import os\ndef signal(d): return d.close"})
    assert err.startswith("ERROR") and "no imports" in err
