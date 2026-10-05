"""Take a backtest straight from the library that ran it.

`check_backtest` wants per-bar returns and, ideally, the position held over
each bar. Most people's backtest already lives in vectorbt or backtesting.py,
so `check_backtest` accepts those objects directly and these functions do the
unpacking:

- a vectorbt ``Portfolio`` (one column, or a whole parameter grid -- every
  column becomes a variant, which is exactly what the selection correction
  needs);
- a backtesting.py stats object, the Series returned by ``Backtest.run()``;
- a dict ``{name: either of the above}`` for variants run one at a time.

Neither library is imported here; objects are recognised by shape, so
TokIO doesn't depend on them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _is_vectorbt(obj: Any) -> bool:
    mod = type(obj).__module__ or ""
    return mod.split(".")[0] in ("vectorbt", "vectorbtpro") and callable(getattr(obj, "returns", None))


def _is_backtesting_stats(obj: Any) -> bool:
    index = getattr(obj, "index", None)
    if index is None or not hasattr(obj, "__getitem__"):
        return False
    try:
        return "_equity_curve" in index and "_trades" in index
    except TypeError:
        return False


def is_backtest_object(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return bool(obj) and all(_is_vectorbt(v) or _is_backtesting_stats(v) for v in obj.values())
    return _is_vectorbt(obj) or _is_backtesting_stats(obj)


def from_vectorbt(portfolio: Any) -> tuple[Any, Any]:
    """(returns, positions) from a vectorbt Portfolio.

    Returns are vectorbt's own per-bar returns, so they are already net of
    the fees and slippage you gave it. The position over bar t is the
    exposure (asset value / portfolio value, signed) at the end of bar t-1:
    vectorbt fills at the bar's close, so that is what earns bar t's return.
    For a multi-column portfolio both come back as DataFrames, one column
    per variant.
    """
    returns = portfolio.returns()
    exposure = (portfolio.asset_value() / portfolio.value()).shift(1).fillna(0.0)
    if getattr(returns, "ndim", 1) == 2:
        names = [_name(c) for c in returns.columns]
        rets = {nm: returns.iloc[:, j] for j, nm in enumerate(names)}
        pos = {nm: exposure.iloc[:, j] for j, nm in enumerate(names)}
        return rets, pos
    return returns, exposure


def from_backtesting(stats: Any) -> tuple[Any, Any]:
    """(returns, positions) from the stats Series that backtesting.py's ``Backtest.run()`` returns.

    Returns come from the equity curve, so they are net of the commission
    you set. Positions are rebuilt from the trade list: each trade's signed
    size times entry price, as a fraction of equity at entry, over its bars
    EntryBar..ExitBar. Its first and last bar are only partly held (fills
    happen at the open), which matters for the holding-run length only at
    the margin. Trades still open at the end are held to the last bar.
    """
    import numpy as np

    equity = stats["_equity_curve"]["Equity"]
    returns = equity.pct_change()
    eq = np.asarray(equity, dtype=float)
    pos = np.zeros(len(eq))
    trades = stats["_trades"]
    if len(trades):
        for size, price, a, b in zip(trades["Size"], trades["EntryPrice"], trades["EntryBar"], trades["ExitBar"]):
            a, b = int(a), int(b)
            if eq[a] > 0:
                pos[a:b + 1] += float(size) * float(price) / eq[a]
    # Trades still open at the end aren't in _trades unless the run used
    # finalize_trades=True; the strategy object still holds them.
    for t in getattr(_get(stats, "_strategy"), "trades", ()) or ():
        a = int(t.entry_bar)
        if 0 <= a < len(eq) and eq[a] > 0:
            pos[a:] += float(t.size) * float(t.entry_price) / eq[a]
    try:
        import pandas as pd

        positions = pd.Series(pos, index=equity.index)
    except ImportError:  # pragma: no cover - backtesting.py itself needs pandas
        positions = pos
    return returns, positions


def _get(stats: Any, key: str) -> Any:
    try:
        return stats[key]
    except (KeyError, IndexError, TypeError):
        return None


def unpack(obj: Any) -> tuple[Any, Any]:
    """(returns, positions) from a vectorbt Portfolio, backtesting.py stats, or a dict of them."""
    if isinstance(obj, Mapping):
        rets, pos = {}, {}

        def add(key, r, p):
            if key in rets:  # a silently overwritten variant would undercount the trials
                raise ValueError(f"two variants are both named {key!r}; rename one")
            rets[key], pos[key] = r, p

        for name, v in obj.items():
            r, p = unpack(v)
            if isinstance(r, Mapping):
                for sub, col in r.items():
                    add(f"{name}/{sub}", col, p[sub])
            else:
                add(str(name), r, p)
        return rets, pos
    if _is_vectorbt(obj):
        return from_vectorbt(obj)
    if _is_backtesting_stats(obj):
        return from_backtesting(obj)
    raise TypeError(f"not a vectorbt Portfolio or backtesting.py stats: {type(obj).__name__}")


def _name(col: Any) -> str:
    if isinstance(col, tuple):
        return ",".join(map(str, col))
    return str(col)
