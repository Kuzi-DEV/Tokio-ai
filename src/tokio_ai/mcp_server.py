"""TokIO as an MCP server: honest statistics for any AI coding assistant.

    claude mcp add tokio -- tokio-ai-mcp          # Claude Code
    pip install "tokio-ai[mcp]"                   # once

The AI does its research in its own code; TokIO checks the results. The
difference from calling a statistics library directly: this server keeps a
ledger of every strategy variant checked in the session, and every verdict is
corrected for all of them. An assistant that tries fifteen ideas and reports
the one that "worked" gets a verdict that knows about the other fourteen.

Data goes in and out as CSV files (paths), so large series never pass through
the chat.
"""

from __future__ import annotations

import io
import contextlib
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

INSTRUCTIONS = """\
TokIO checks whether trading/backtest results are real or luck.
Rules for using it well:
- Before you tell the user a strategy, signal or backtest works, run check_backtest on it.
- Pass EVERY variant you tried (one CSV column each), not just the best one. Every check in this
  session is recorded, and verdicts are corrected for all of them, so testing ideas one at a time
  does not escape the correction.
- Use get_prices to get data as a CSV you can load in your own code.
- Use backtest_positions when you have a position series: it adds costs and a lookahead check.
- Quote TokIO's verdict line as it is. Don't round "not significant" up to "promising".
"""


def _out_dir() -> Path:
    base = Path(os.environ.get("TOKIO_AI_CACHE", Path.home() / ".tokio_ai" / "cache"))
    p = base / "mcp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pd():
    import pandas as pd
    return pd


def _read_table(path: str):
    """CSV/TSV/XLSX -> DataFrame indexed by its date column when it has one."""
    pd = _pd()
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p, sep=None, engine="python")
    first = df.columns[0]
    lowered = str(first).lower()
    if lowered in ("date", "datetime", "time", "timestamp", "index", "unnamed: 0") or not pd.api.types.is_numeric_dtype(df[first]):
        try:
            idx = pd.to_datetime(df[first], errors="raise", format="mixed")
            df = df.drop(columns=[first])
            df.index = idx
        except (ValueError, TypeError):
            pass
    return df


@dataclass
class Ledger:
    """Every variant checked in this server process (= one assistant session)."""

    entries: list[tuple[str, str, int]] = field(default_factory=list)  # (time, label, n_variants)

    def add(self, label: str, n: int) -> int:
        self.entries.append((datetime.now().strftime("%H:%M:%S"), label, n))
        return self.total

    @property
    def total(self) -> int:
        return sum(n for _, _, n in self.entries)

    def reset(self) -> None:
        self.entries.clear()


LEDGER = Ledger()


def _ledger_note(total: int, passed: int) -> str:
    if total <= passed:
        return f"Session ledger: {total} variant(s) checked so far, all in this call."
    return (f"Session ledger: {total} variants checked this session ({passed} in this call), "
            f"so this verdict is corrected for {total} trials.")


# ---- tool implementations (plain functions, tested directly) ----------------------

def get_prices(symbol: str, start: str | None = None, end: str | None = None) -> str:
    """Daily OHLCV for a ticker (Yahoo, split- and dividend-adjusted), saved as a CSV you can load.

    symbol: e.g. "SPY", "AAPL", "^VIX", "BTC-USD". start/end: optional "YYYY-MM-DD".
    Returns the file path and a short summary. Full history by default (SPY from 1993).
    """
    from .lab import load_prices
    df = load_prices(symbol)
    if start:
        df = df[df.index >= _pd().Timestamp(start)]
    if end:
        df = df[df.index <= _pd().Timestamp(end)]
    path = _out_dir() / f"prices_{symbol.upper().replace('^', '')}.csv"
    df.to_csv(path, index_label="date")
    return (f"Saved {len(df)} daily bars for {symbol.upper()} ({df.index[0].date()} -> {df.index[-1].date()}) "
            f"to {path}\nColumns: date, {', '.join(df.columns)}. Last close {df['close'].iloc[-1]:.2f}.")


def check_backtest(returns_csv: str, columns: list[str] | None = None, trials: int | None = None,
                   costs: float = 0.0, positions_csv: str | None = None, label: str | None = None) -> str:
    """Is a backtest's return real or luck? Use before claiming any strategy works.

    returns_csv: path to a CSV of per-bar strategy returns (0.01 = +1%), first column = date,
      ONE COLUMN PER VARIANT you tried (every parameter set, not only the best).
    columns: optional subset of columns to test. trials: total variants tried if more than the
      columns given. costs: cost per unit of position traded (0.0005 = 5 bp; needs positions_csv).
    positions_csv: optional positions, same layout. label: a short name for the ledger.
    The verdict covers the best variant, corrected for all of them AND for everything checked
    earlier in this session. Reports HAC p-value, Deflated Sharpe, PBO for grids, robustness.
    """
    from .backtest import check_backtest as _cb
    df = _read_table(returns_csv)
    num = df.select_dtypes("number")
    if columns:
        missing = [c for c in columns if c not in num.columns]
        if missing:
            raise ValueError(f"columns not found or not numeric: {missing}; numeric columns are {list(num.columns)}")
        num = num[columns]
    if num.shape[1] == 0:
        raise ValueError(f"no numeric columns in {returns_csv}")
    n = num.shape[1]
    total = LEDGER.add(label or Path(returns_csv).name, n)
    use_trials = max(total, trials or 0)
    kwargs = {"trials": use_trials if use_trials > n else None, "costs": costs}
    if positions_csv:
        pos = _read_table(positions_csv).select_dtypes("number")
        kwargs["positions"] = pos.iloc[:, 0] if n == 1 else {c: pos[c] for c in num.columns}
    data = num.iloc[:, 0] if n == 1 else {c: num[c].dropna() for c in num.columns}
    res = _cb(data, **kwargs)
    return f"{res}\n\n{_ledger_note(total, n)}"


def backtest_positions(symbol: str, positions_csv: str, costs_bps: float = 1.0, vs: str = "market",
                       label: str | None = None) -> str:
    """Backtest daily positions on a ticker: costs, next-day returns, market comparison, lookahead check.

    symbol: ticker. positions_csv: first column date, then one column per variant with the
      position decided at that day's close (1 long, -1 short, 0 flat, fractions allowed).
    costs_bps: cost per unit of position traded, in basis points (default 1).
    vs: "market" (default) asks whether the timing beat simply holding the asset at the same
      average exposure; "cash" asks only whether it made money (a long-only rule usually does,
      because markets rise).
    Each position earns the NEXT day's close-to-close return. The verdict is corrected for
    every variant checked this session; with one variant under vs="cash" it also checks whether
    delaying the signal by a day destroys the result (a sign of lookahead).
    """
    from .backtest import check_backtest as _cb
    from .lab import load_prices
    pd = _pd()
    if vs not in ("market", "cash"):
        raise ValueError('vs must be "market" or "cash"')
    pos = _read_table(positions_csv).select_dtypes("number")
    if pos.shape[1] == 0:
        raise ValueError("positions file needs a numeric column (1 = long, -1 = short, 0 = flat)")
    asset = load_prices(symbol)["close"].pct_change()
    span = asset.loc[pos.index.min():pos.index.max()]
    held = pos.reindex(span.index).ffill().shift(1)  # decided at close t, held over t+1
    keep = held.notna().all(axis=1) & span.notna()
    held, span = held[keep], span[keep]
    n = held.shape[1]
    tested, lines = {}, []
    for c in held.columns:
        p = held[c]
        gross = p * span
        cost = p.diff().abs().fillna(p.abs()) * costs_bps / 1e4
        net = gross - cost
        eq = (1 + net).cumprod()
        bh = (1 + span).cumprod()
        lines.append(f"{c}: {len(net)} days ({net.index[0].date()} -> {net.index[-1].date()}), net return "
                     f"{eq.iloc[-1] - 1:+.1%} vs holding {symbol.upper()} {bh.iloc[-1] - 1:+.1%}, max drawdown "
                     f"{(eq / eq.cummax() - 1).min():.1%}, average position {p.mean():+.2f}.")
        tested[c] = gross - p.mean() * span if vs == "market" else gross
    total = LEDGER.add(label or f"{symbol}:{Path(positions_csv).name}", n)
    kw = {"trials": total if total > n else None, "costs": costs_bps / 1e4}
    if n == 1:
        c = held.columns[0]
        kw["positions"] = held[c]
        if vs == "cash":
            kw["asset_returns"] = span
        res = _cb(tested[c], **kw)
    else:
        kw["positions"] = {c: held[c] for c in held.columns}
        res = _cb(tested, **kw)
    head = ("Tested against the market: each variant's return minus holding the asset at its average "
            "exposure, so drift alone can't pass." if vs == "market" else
            "Tested against zero (vs=\"cash\"): a long-biased rule can pass on the market's rise alone.")
    return ("\n".join(lines) + f"\nPositions decided at the close earn the NEXT day's return; costs {costs_bps:g} bp "
            f"per unit traded.\n{head}\n\n{res}\n\n{_ledger_note(total, n)}")


def check_contracts(csv_path: str, price_column: str, outcome_column: str, contracts_column: str | None = None,
                    fees_per_contract: float = 0.0, label: str | None = None) -> str:
    """Exact test for bets on binary contracts (Kalshi/Polymarket style): edge or luck?

    csv_path: one row per bet. price_column: price paid for the side bought, as a probability
      (0.93, not 93). outcome_column: 1 if that side won, 0 if not. contracts_column: optional
      contracts per bet. fees_per_contract: in probability units (0.003 = 0.3 cents).
    Exact Poisson-binomial test of win rate vs the prices paid; use REAL fill prices.
    """
    from .contracts import check_contracts as _cc
    df = _read_table(csv_path)
    kw = {}
    if contracts_column:
        kw["sizes"] = df[contracts_column].to_numpy()
    res = _cc(df[price_column].to_numpy(), df[outcome_column].to_numpy(), fees=fees_per_contract, **kw)
    total = LEDGER.add(label or Path(csv_path).name, 1)
    note = "" if total == 1 else (f"\n\nSession ledger: {total} variants checked this session. check_contracts tests "
                                  f"this book alone; if it was chosen from the others, apply a correction for {total}.")
    return f"{res}{note}"


def vol_surface(symbol: str) -> str:
    """Implied-volatility surface for a US ticker or index from Cboe's free delayed option quotes.

    symbol: "SPY", "SPX", "AAPL", ... Vols are computed from bid/mid/ask with parity forwards.
    Returns ATM term structure, skew, bid-ask vol width, a CSV of every option used, and an
    interactive 3D HTML plot path.
    """
    from .volsurface import vol_surface as _vs
    surf = _vs(symbol)
    pts_path = _out_dir() / f"volsurface_{surf.symbol}.csv"
    surf.points.to_csv(pts_path, index=False)
    msg = f"{surf}\n\nEvery option used: {pts_path}"
    try:
        html = _out_dir() / f"volsurface_{surf.symbol}.html"
        surf.plot().write_html(html, include_plotlyjs="cdn")
        msg += f"\nInteractive 3D surface (open in a browser): {html}"
    except ImportError:
        msg += "\n(Install plotly for the 3D plot.)"
    return msg


def put_call(kind: str = "equity", start: str | None = None) -> str:
    """Daily Cboe put/call volume saved as CSV.

    kind: "equity", "index", "total" (from 2006) or "etp", "vix", "spx" (from 2019).
    start: optional "YYYY-MM-DD" (the first full download takes a few minutes; then cached).
    """
    from .putcall import load_put_call
    df = load_put_call(kind, start=start)
    path = _out_dir() / f"putcall_{kind}.csv"
    df.to_csv(path)
    return (f"Saved {len(df)} days of {kind} put/call volume ({df.index[0].date()} -> {df.index[-1].date()}) to {path}\n"
            f"Latest ratio {df['ratio'].iloc[-1]:.2f}. Notes: " + " ".join(df.attrs.get("notes", [])))


def session_trials() -> str:
    """Everything checked in this session, and the total the next verdict will be corrected for."""
    if not LEDGER.entries:
        return "Nothing checked yet this session."
    lines = [f"{t}  {lbl}  ({n} variant{'s' if n != 1 else ''})" for t, lbl, n in LEDGER.entries]
    return "\n".join(lines) + f"\nTotal: {LEDGER.total} variants. The next verdict is corrected for at least that many."


def reset_session(confirm: bool = False) -> str:
    """Clear the ledger. Only when the user starts a genuinely new question on new data."""
    if not confirm:
        return "Not reset. Pass confirm=true, and only for a new question: resetting to escape the correction defeats it."
    n = LEDGER.total
    LEDGER.reset()
    return f"Ledger cleared ({n} variants forgotten)."


TOOLS = [get_prices, check_backtest, backtest_positions, check_contracts, vol_surface, put_call,
         session_trials, reset_session]


def _quiet(fn):
    """MCP stdio uses stdout for protocol messages: keep library prints off it."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*a, **k)
    return wrapper


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
        except ImportError as exc:
            raise ImportError('the MCP server needs: pip install "tokio-ai[mcp]"') from exc
    from . import __version__
    try:
        server = Server("tokio", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:  # mcp 1.x has no version argument
        server = Server("tokio", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.tool()(_quiet(fn))
    return server


def main() -> None:
    from ._stdio import force_utf8_stdio
    force_utf8_stdio()
    build_server().run()


if __name__ == "__main__":
    main()
