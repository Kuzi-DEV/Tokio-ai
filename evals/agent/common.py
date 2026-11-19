"""Shared pieces for the agent eval: cached prices, and the browser's Python shim.

The eval runs the SAME tool code the web terminal runs: it pulls the Python shim
and the system prompt out of docs/terminal/index.html, so a change to the page
is a change to what gets scored.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAGE = ROOT / "docs" / "terminal" / "index.html"
CACHE = HERE / "cache"
PROXY = "https://tokio-data.tokio-data.workers.dev"
END_DATE = "2026-10-30"  # prices are frozen here so every run sees the same data
RANGE = "20y"


def page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def load_shim() -> dict:
    """Exec the page's Python shim and return its namespace (load_chart, run_code, lab, ...)."""
    code = re.search(r'<script id="py-shim" type="text/python">(.*?)</script>', page_text(), re.S).group(1)
    ns: dict = {"__name__": "tokio_shim"}
    exec(compile(code, "py-shim", "exec"), ns)
    return ns


def system_prompt() -> str:
    m = re.search(r"const SYSTEM = `(.*?)`;", page_text(), re.S)
    return m.group(1).replace("\\`", "`")


TOOLS = [
    {"type": "function", "function": {
        "name": "get_prices",
        "description": "Load daily OHLCV bars for a ticker into prices['SYMBOL'] in the Python namespace.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Yahoo ticker, e.g. SPY, AAPL, BTC-USD, ^GSPC"},
            "range": {"type": "string", "enum": ["1y", "2y", "5y", "10y", "20y", "max"], "description": "History length. Default 20y."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run Python in the persistent namespace (np, pd, t=tokio_ai, prices, lab, check_backtest). Returns stdout and the last expression.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]


def chart_payload(symbol: str) -> str:
    """Yahoo chart JSON for `symbol`, fetched once through the TokIO proxy and frozen at END_DATE."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{symbol.replace('^', '_')}_{RANGE}.json"
    if not path.exists():
        url = f"{PROXY}/yahoo/chart/{urllib.request.quote(symbol)}?range={RANGE}&interval=1d"
        req = urllib.request.Request(url, headers={"Origin": "http://localhost:8766", "User-Agent": "tokio-eval"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
        res = payload["chart"]["result"][0]
        import datetime as dt
        cutoff = dt.datetime.fromisoformat(END_DATE).replace(tzinfo=dt.timezone.utc).timestamp() + 86400
        keep = [i for i, ts in enumerate(res["timestamp"]) if ts < cutoff]
        res["timestamp"] = [res["timestamp"][i] for i in keep]
        for block in (res["indicators"]["quote"][0], *(res["indicators"].get("adjclose") or [])):
            for k, v in list(block.items()):
                block[k] = [v[i] for i in keep]
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path.read_text(encoding="utf-8")
