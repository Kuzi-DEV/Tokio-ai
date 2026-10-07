"""Backtests that arrive as a list of trades: TradingView's Strategy Tester export, or any trade log.

TradingView's "List of trades" download (CSV, or the XLSX report's "List
of trades" sheet) has two rows per trade -- an entry and an exit, in either
order -- with columns like

    Trade #, Type, Date/Time, Signal, Price USD, Position size (qty),
    Position size (value), Net P&L USD, Net P&L %, Run-up USD, ...,
    Cumulative P&L USD, Cumulative P&L %

(older exports say "Profit USD", "Contracts", "Cum. Profit USD"; the
currency follows the symbol). `read_tradingview` turns that into one return
per closed trade, which `check_backtest` tests directly:

    trades = tokio_ai.read_tradingview("export.csv")
    print(tokio_ai.check_backtest(trades, trials=30))

Each trade's return is, by default, its P&L as a fraction of the position's
value (TradingView's "Net P&L %"), so the test doesn't depend on how the
position was sized. ``basis="equity"`` uses P&L over the account equity
before the trade instead, which is what the account actually earned.
Trades still open at the end of the export are left out.

The Sharpe is per trade, annualized by trades per year. Several exports
passed together as {name: trades} are put on a common calendar first (each
trade's return booked on its exit day), so their variants line up.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_YEAR_SECONDS = 365.25 * 86400


@dataclass(frozen=True)
class TradeList:
    returns: list[float]  # per closed trade, in exit order
    entry_times: list[datetime | None]
    exit_times: list[datetime | None]
    sides: list[str]  # "long" / "short" / ""
    basis: str
    source: str = ""
    skipped_open: int = 0
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.returns)

    @property
    def periods_per_year(self) -> float | None:
        """Closed trades per year over the export's span, or None without usable dates."""
        times = [t for t in self.exit_times if t is not None]
        starts = [t for t in self.entry_times if t is not None] or times
        if len(times) < 2:
            return None
        span = (max(map(_naive, times)) - min(map(_naive, starts))).total_seconds()
        if span <= 0:
            return None
        return len(self.returns) / (span / _YEAR_SECONDS)


def _naive(t: datetime) -> datetime:
    return t.replace(tzinfo=None) if t.tzinfo is None else t.astimezone().replace(tzinfo=None)


def _num(cell: Any) -> float | None:
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        return None if isinstance(cell, float) and math.isnan(cell) else float(cell)
    c = str(cell).strip().replace(",", "").replace("−", "-").replace("%", "")
    if not c or c.lower() in {"nan", "-", "n/a", "none"}:
        return None
    try:
        return float(c)
    except ValueError:
        return None


_TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M",
                 "%b %d, %Y, %H:%M", "%b %d, %Y %H:%M", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z")


def _time(cell: Any) -> datetime | None:
    if cell is None:
        return None
    if isinstance(cell, datetime):
        return cell
    to_py = getattr(cell, "to_pydatetime", None)  # pandas Timestamp
    if to_py is not None:
        return to_py()
    s = str(cell).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for f in _TIME_FORMATS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def _find(header: list[str], *patterns: str) -> int | None:
    """Index of the first column whose name matches a pattern (case-insensitive regex, anchored)."""
    norm = [h.strip().lower() for h in header]
    for p in patterns:
        rx = re.compile(p)
        for j, h in enumerate(norm):
            if rx.fullmatch(h):
                return j
    return None


def _rows(source: Any) -> tuple[list[str], list[list[Any]], str]:
    """Header and rows from a CSV/XLSX path or a pandas DataFrame."""
    columns = getattr(source, "columns", None)
    if columns is not None and hasattr(source, "itertuples"):
        return [str(c) for c in columns], [list(r) for r in source.itertuples(index=False)], "DataFrame"
    path = str(source)
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading an .xlsx export needs pandas and openpyxl") from e
        sheets = pd.read_excel(path, sheet_name=None)
        name = next((s for s in sheets if s.strip().lower() == "list of trades"), None)
        if name is None:
            name = next((s for s, df in sheets.items()
                         if any(str(c).strip().lower() == "trade #" for c in df.columns)), None)
        if name is None:
            raise ValueError(f"{path}: no 'List of trades' sheet")
        df = sheets[name]
        return [str(c) for c in df.columns], [list(r) for r in df.itertuples(index=False)], path
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(f, dialect))
    if len(rows) < 2:
        raise ValueError(f"{path}: no trades")
    return rows[0], rows[1:], path


def is_tradingview_header(header: list[str]) -> bool:
    return _find(header, r"trade #") is not None and _find(header, r"type") is not None


def read_tradingview(source: Any, *, basis: str = "position", initial_capital: float | None = None) -> TradeList:
    """Closed-trade returns from a TradingView Strategy Tester "List of trades" export.

    `source`: path to the CSV or XLSX download, or a DataFrame of it.
    `basis`: "position" (default) -- each trade's P&L over the position's
    value, TradingView's "Net P&L %"; or "equity" -- P&L over the account
    equity before the trade (needs the cumulative P&L columns, or
    `initial_capital`).
    """
    if basis not in ("position", "equity"):
        raise ValueError(f"basis must be 'position' or 'equity', got {basis!r}")
    header, rows, name = _rows(source)
    if not is_tradingview_header(header):
        raise ValueError(f"{name}: not a TradingView 'List of trades' export (no 'Trade #' and 'Type' columns)")
    c_id = _find(header, r"trade #")
    c_type = _find(header, r"type")
    c_time = _find(header, r"date/time", r"date and time", r"date")
    c_signal = _find(header, r"signal")
    c_pct = _find(header, r"net p&l %", r"profit %")
    c_pnl = _find(header, r"net p&l(?! %).*", r"profit(?! %)[^%]*")
    c_cum = _find(header, r"cumulative p&l(?! %).*", r"cum\. profit(?! %)[^%]*")
    c_cum_pct = _find(header, r"cumulative p&l %", r"cum\. profit %")
    c_price = _find(header, r"price.*")
    c_qty = _find(header, r"position size \(qty\)", r"contracts", r"quantity", r"qty")
    c_value = _find(header, r"position size \(value\)")

    def cell(r, j):
        return r[j] if r is not None and j is not None and j < len(r) else None

    trades: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in rows:
        tid = str(cell(r, c_id) or "").strip()
        if not tid:
            continue
        if tid not in trades:
            trades[tid] = {}
            order.append(tid)
        kind = str(cell(r, c_type) or "").strip().lower()
        slot = "exit" if kind.startswith("exit") else "entry" if kind.startswith("entry") else None
        if slot is None:
            continue
        trades[tid][slot] = r
        trades[tid]["side"] = "long" if "long" in kind else "short" if "short" in kind else ""

    rets, t_in, t_out, sides, notes, tids = [], [], [], [], [], []
    skipped_open = 0
    missing = 0
    equity_rows = []
    for tid in order:
        t = trades[tid]
        ex, en = t.get("exit"), t.get("entry")
        if ex is None or str(cell(ex, c_signal) or "").strip().lower() == "open":
            skipped_open += 1
            continue
        pnl = _num(cell(ex, c_pnl)) if c_pnl is not None else None
        if basis == "position":
            pct = _num(cell(ex, c_pct)) if c_pct is not None else None
            if pct is not None:
                ret = pct / 100
            else:
                value = _num(cell(ex, c_value)) or _num(cell(en, c_value)) if c_value is not None else None
                if value is None and en is not None:
                    price, qty = _num(cell(en, c_price)), _num(cell(en, c_qty))
                    value = abs(price * qty) if price is not None and qty is not None else None
                ret = pnl / value if pnl is not None and value else None
        else:
            ret = None
            equity_rows.append((pnl, _num(cell(ex, c_cum)), _num(cell(ex, c_cum_pct))))
        if basis == "position" and ret is None:
            missing += 1
            continue
        rets.append(ret)
        t_in.append(_time(cell(en, c_time)) if en is not None else None)
        t_out.append(_time(cell(ex, c_time)))
        sides.append(t["side"])
        tids.append(tid)

    if missing:
        notes.append(f"{missing} closed trades had no P&L or position value and were left out.")
    if not rets:
        raise ValueError(f"{name}: no closed trades with a P&L")
    # Exit order: TradingView numbers trades by entry, and some exports list
    # newest first. Sorted before the equity curve is compounded, which must
    # run forward in time.
    if all(t is not None for t in t_out):
        idx = sorted(range(len(rets)), key=lambda i: (_naive(t_out[i]), _tid_key(tids[i])))
        rets, t_in, t_out, sides = ([v[i] for i in idx] for v in (rets, t_in, t_out, sides))
        if basis == "equity":
            equity_rows = [equity_rows[i] for i in idx]
    if basis == "equity":
        rets = _equity_returns(equity_rows, initial_capital, name)
    return TradeList(rets, t_in, t_out, sides, basis, name, skipped_open, notes)


def _tid_key(tid: str) -> tuple:
    """Trade numbers sort numerically; same-minute exits keep TradingView's order."""
    try:
        return (0, float(tid), "")
    except ValueError:
        return (1, 0.0, tid)


def _equity_returns(rows, initial_capital, name) -> list[float]:
    pnls = [p for p, _, _ in rows]
    if any(p is None for p in pnls):
        raise ValueError(f"{name}: basis='equity' needs a Net P&L column on every closed trade")
    start = initial_capital
    if start is None:
        for _, cum, cum_pct in rows:
            if cum and cum_pct:
                start = cum / (cum_pct / 100)
                break
    if not start or start <= 0:
        raise ValueError(f"{name}: basis='equity' needs the cumulative P&L columns or initial_capital=")
    out, eq = [], start
    for p in pnls:
        out.append(p / eq if eq > 0 else float("nan"))
        eq += p
    return out


def on_calendar(lists: dict[str, TradeList]) -> tuple[dict[str, list[float]], float]:
    """Several trade lists as aligned per-day series (each trade booked on its exit day)."""
    days: dict[str, dict[Any, float]] = {}
    for nm, tl in lists.items():
        if any(t is None for t in tl.exit_times):
            raise ValueError(f"{nm}: exit times are needed to line up several trade lists")
        d: dict[Any, float] = {}
        for r, t in zip(tl.returns, tl.exit_times):
            k = _naive(t).date()
            d[k] = d.get(k, 0.0) + r
        days[nm] = d
    allk = sorted(set().union(*days.values()))
    series = {nm: [d.get(k, 0.0) for k in allk] for nm, d in days.items()}
    span = (allk[-1] - allk[0]).days / 365.25 if len(allk) > 1 else 0
    ppy = len(allk) / span if span > 0 else 252.0
    return series, ppy
