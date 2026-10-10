"""Runs inside the isolated child process. Not for import by anything else.

Sequence: read the job from the parent (trusted pickle on stdin), import
numpy/pandas and warm up the operations signals use (so their lazy imports
happen now), compile the strategy, then install an audit hook that allows
only the events pure computation raises and kills the process on anything
else, then run the requests and write plain numbers back as JSON.
"""

from __future__ import annotations

import base64
import json
import os
import sys

# Measured: after warm-up, rolling/ewm/rank/where/clip/apply/... raise only these.
ALLOWED_EVENTS = frozenset({"builtins.id", "sys._getframe"})


def _warm_up(pd, np, d):
    c = d["close"]
    f, s = c.rolling(10).mean(), c.rolling(50).mean()
    (f > s).astype(float)
    np.sign(c.ewm(span=20).mean() - c.ewm(span=50, adjust=False).mean())
    delta = c.diff()
    up, dn = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    (100 - 100 / (1 + up / dn)).where(c > c.rolling(200).mean(), 0.0)
    np.tanh((c - c.rolling(20).mean()) / c.rolling(20).std()).fillna(0)
    hi = d["high"] if "high" in d.columns else c
    (c > hi.rolling(20).max().shift(1)).astype(int) * 1.0
    pos = pd.Series(0.0, index=d.index)
    pos[f > s] = 1.0
    c.pct_change(5).rank(pct=True).fillna(0.5)
    c.rolling(30).apply(lambda w: float(w[-1] > w.mean()), raw=True)
    c.rolling(10).quantile(0.8)
    c.expanding().max()
    c.rolling(20).min(), c.rolling(20).sum(), c.rolling(20).median(), c.rolling(20).var()
    c.cummax(), c.cumsum(), c.abs(), c.round(2), c.ffill(), c.bfill(), c.shift(-0)
    np.where(c > c.shift(1), 1.0, -1.0), np.log(c), np.sqrt(c), np.maximum(c, 1), np.minimum(c, 1)
    pd.concat([f, s], axis=1).max(axis=1)
    d.iloc[: len(d) // 2].copy()


def _hook(event, args):
    if event in ALLOWED_EVENTS:
        return
    try:
        sys.stderr.write(f"TOKIO_SANDBOX_BLOCKED {event}\n")
        sys.stderr.flush()
    finally:
        os._exit(13)


def main(job: dict) -> None:
    import numpy as np
    import pandas as pd

    from tokio_ai.tools.strategy import compile_signal

    idx = pd.DatetimeIndex(np.asarray(job["index"], dtype="int64").view("datetime64[ns]"))
    d = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in job["columns"].items()}, index=idx)
    fn = compile_signal(job["code"], job["name"])
    _warm_up(pd, np, d)
    out = sys.stdout.buffer
    sys.addaudithook(_hook)  # from here on: computation only
    results = []
    for params, length in job["requests"]:
        try:
            pos = fn(d.iloc[:length].copy(), **params)
            arr = np.asarray(pos, dtype=float)
            if arr.shape != (length,):
                results.append({"error": f"returned {arr.size} positions for {length} bars"})
                continue
            results.append({"b64": base64.b64encode(arr.astype("<f8").tobytes()).decode()})
        except Exception as e:  # the strategy's own error, reported back as text
            results.append({"error": f"{type(e).__name__}: {str(e)[:300]}"})
    out.write(json.dumps(results).encode())
    out.flush()
