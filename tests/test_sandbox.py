"""Functional tests for the isolated signal runner: same results, readable errors, limits."""

import os

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from tokio_ai import Lab  # noqa: E402
from tokio_ai.tools import sandbox  # noqa: E402
from tokio_ai.tools.sandbox import IsolatedSignal, SandboxError, run_isolated  # noqa: E402
from tokio_ai.tools.strategy import compile_signal  # noqa: E402

SIGNALS = {
    "sma": "def signal(d, fast=10, slow=50):\n"
           "    return (d.close.rolling(fast).mean() > d.close.rolling(slow).mean()).astype(float)\n",
    "ewm": "def signal(d, a=12, b=26):\n"
           "    return np.sign(d.close.ewm(span=a).mean() - d.close.ewm(span=b).mean()).fillna(0)\n",
    "rsi": "def signal(d, n=14, lo=30):\n"
           "    delta = d.close.diff()\n"
           "    up = delta.clip(lower=0).rolling(n).mean()\n"
           "    dn = (-delta.clip(upper=0)).rolling(n).mean()\n"
           "    rsi = 100 - 100 / (1 + up / dn)\n"
           "    return (rsi < lo).astype(float)\n",
    "zscore": "def signal(d, n=20):\n"
              "    z = (d.close - d.close.rolling(n).mean()) / d.close.rolling(n).std()\n"
              "    return (-np.tanh(z)).fillna(0)\n",
    "breakout": "def signal(d, n=20):\n"
                "    pos = pd.Series(0.0, index=d.index)\n"
                "    pos[d.close > d.close.rolling(n).max().shift(1)] = 1.0\n"
                "    return pos\n",
}


def _bars(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.standard_normal(n) * 0.002)
    return pd.DataFrame({"open": open_, "close": close}, index=idx)


@pytest.mark.parametrize("name", sorted(SIGNALS))
def test_isolated_results_match_in_process(name):
    d = _bars()
    code = SIGNALS[name]
    want = np.asarray(compile_signal(code, name)(d.copy()), dtype=float)
    got = run_isolated(code, name, d, [({}, len(d)), ({}, 700)])
    np.testing.assert_array_equal(got[0], want)
    np.testing.assert_array_equal(got[1], np.asarray(compile_signal(code, name)(d.iloc[:700].copy()), dtype=float))


def test_lab_sweep_is_identical_in_the_sandbox():
    d = _bars(2000)
    code = SIGNALS["sma"]
    a = Lab(d, holdout=0.25)
    a.sweep(compile_signal(code, "sma"), fast=[5, 10], slow=[50, 100])
    b = Lab(d, holdout=0.25)
    b.sweep(IsolatedSignal(code, "sma"), fast=[5, 10], slow=[50, 100])
    for ra, rb in zip(a.runs, b.runs):
        np.testing.assert_allclose(ra.returns.to_numpy(), rb.returns.to_numpy(), rtol=0, atol=0)
    assert a.check().p_value == pytest.approx(b.check().p_value)


def test_one_process_serves_a_whole_sweep(monkeypatch):
    calls = []
    real = sandbox.run_isolated

    def counting(*args, **kw):
        calls.append(len(args[3]))
        return real(*args, **kw)

    monkeypatch.setattr(sandbox, "run_isolated", counting)
    lab = Lab(_bars(2000), holdout=0.25)
    lab.sweep(IsolatedSignal(SIGNALS["sma"], "sma"), fast=[5, 10, 20], slow=[50, 100])
    assert len(calls) == 1  # 6 variants x (full run + lookahead cuts), one child process
    assert calls[0] >= 6 * 2


def test_lookahead_is_still_caught_through_the_sandbox():
    from tokio_ai import LookaheadError

    cheat = "def signal(d):\n    return (d.close.shift(-1) > d.close).astype(float)\n"
    with pytest.raises(LookaheadError, match="uses future data"):
        Lab(_bars(), holdout=0).run(IsolatedSignal(cheat, "cheat"))


def test_the_signals_own_error_is_readable():
    bad = "def signal(d):\n    return d['nonexistent_column'] * 0\n"
    sig = IsolatedSignal(bad, "bad")
    with pytest.raises(SandboxError, match="KeyError"):
        sig(_bars())


def test_wrong_length_output_is_reported():
    short = "def signal(d):\n    return d.close.iloc[:10]\n"
    out = run_isolated(short, "short", _bars(), [({}, 1500)])
    assert isinstance(out[0], str) and "positions for 1500 bars" in out[0]


def test_runaway_signal_is_stopped_by_the_timeout():
    slow = "def signal(d):\n    while True:\n        pass\n"
    with pytest.raises(SandboxError, match="was stopped"):
        run_isolated(slow, "slow", _bars(200), [({}, 200)], timeout=5)


def test_the_child_gets_no_environment_variables(monkeypatch):
    monkeypatch.setenv("TOKIO_TEST_SECRET", "do-not-pass")
    seen = {}
    real_popen = sandbox.subprocess.Popen

    def capture(*args, **kw):
        seen["env"] = dict(kw.get("env") or {})
        seen["argv"] = list(args[0])
        return real_popen(*args, **kw)

    monkeypatch.setattr(sandbox.subprocess, "Popen", capture)
    run_isolated(SIGNALS["sma"], "sma", _bars(300), [({}, 300)])
    assert "TOKIO_TEST_SECRET" not in seen["env"]
    assert set(seen["env"]) <= {"SYSTEMROOT", "WINDIR"}
    assert "-I" in seen["argv"] and "-S" in seen["argv"]


@pytest.mark.skipif(os.name != "nt", reason="job objects are Windows-only")
def test_windows_job_limits_are_applied(monkeypatch):
    applied = []
    real = sandbox._windows_job

    def spy(proc):
        h = real(proc)
        applied.append(h)
        return h

    monkeypatch.setattr(sandbox, "_windows_job", spy)
    run_isolated(SIGNALS["sma"], "sma", _bars(300), [({}, 300)])
    assert applied and applied[0]
