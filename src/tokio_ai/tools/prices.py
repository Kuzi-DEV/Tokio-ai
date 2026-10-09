"""Free daily-bar price history from Yahoo Finance's public chart endpoint.

No API key needed. See the granularity check in `fetch_daily_bars` -- Yahoo
silently downsamples `interval=1d` to coarser bars for very long ranges on
old tickers (confirmed empirically: IBM with range=max comes back quarterly,
no error, no flag). This refuses to return data that isn't actually daily
rather than handing an agent a wrong answer.
"""

from __future__ import annotations

import json
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MAX_MEDIAN_GAP_DAYS = 4


@dataclass(frozen=True)
class DailyBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int

    def to_dict(self) -> dict:
        return asdict(self)


def _epoch_to_date(t: int) -> date:
    # datetime.fromtimestamp() rejects pre-1970 (negative) timestamps on
    # Windows; adding a timedelta to a fixed epoch works on any platform.
    return (_EPOCH + timedelta(seconds=t)).date()


def _parse_chart_payload(payload: dict, symbol: str) -> list[DailyBar]:
    result = payload["chart"]["result"]
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {payload['chart'].get('error')}")
    res = result[0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")

    bars: list[DailyBar] = []
    for i, t in enumerate(ts):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c):  # Yahoo pads gaps with nulls
            continue
        bars.append(
            DailyBar(
                date=_epoch_to_date(t).isoformat(),
                open=o,
                high=h,
                low=l,
                close=c,
                adj_close=(adj[i] if adj else c) or c,
                volume=v or 0,
            )
        )
    return bars


def fetch_daily_bars(symbol: str, range_: str = "10y") -> list[DailyBar]:
    """Pull daily OHLCV bars for `symbol`.

    Avoid range='max' on tickers with decades of history -- see module
    docstring. 20y is safe for essentially every liquid large-cap.
    """
    url = _CHART_URL.format(symbol=urllib.parse.quote(symbol), range_=range_)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())

    bars = _parse_chart_payload(payload, symbol)
    _assert_daily_granularity(bars, symbol, range_)
    return bars


_CHART_URL_SPAN = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                   "?interval=1d&period1={start}&period2={end}")


def fetch_full_daily_bars(symbol: str) -> list[DailyBar]:
    """All daily bars Yahoo has for `symbol`.

    range=max can come back silently downsampled for old tickers; an explicit
    period1/period2 span returns true daily bars for the whole history
    (checked: SPY from 1993, median gap 1 day).
    """
    end = int(datetime.now(timezone.utc).timestamp()) + 86400
    url = _CHART_URL_SPAN.format(symbol=urllib.parse.quote(symbol), start=0, end=end)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())
    bars = _parse_chart_payload(payload, symbol)
    _assert_daily_granularity(bars, symbol, "full history")
    return bars


def _assert_daily_granularity(bars: list[DailyBar], symbol: str, range_: str) -> None:
    if len(bars) < 3:
        return
    dates = [date.fromisoformat(b.date) for b in bars]
    gaps = [(b2 - b1).days for b1, b2 in zip(dates, dates[1:])]
    median_gap = statistics.median(gaps)
    if median_gap > _MAX_MEDIAN_GAP_DAYS:
        raise RuntimeError(
            f"{symbol}: got {len(bars)} bars for range={range_} with a median gap "
            f"of {median_gap} days -- that's not daily granularity. Yahoo silently "
            f"downsampled this response; retry with a shorter range (e.g. 10y-20y)."
        )
