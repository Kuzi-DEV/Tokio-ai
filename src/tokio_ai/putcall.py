"""Daily put/call volume from Cboe's free public files, 2006 to today.

Two Cboe publications are stitched together:

* historical CSVs (equity, index, total), 2006-11-01 to 2019-10-04;
* one JSON file per trading day from October 2019 on (also ETPs, VIX, SPX).

Daily files are cached on disk (``~/.tokio_ai/cache/cboe_daily``), so only the
first call downloads; later calls fetch just the new days.

Known breaks, also in ``df.attrs["notes"]``: from 2012-06-11 Cboe's equity and
index volumes exclude ETPs, and the source changes on 2019-10-07. Levels can
shift at both, so compare a day with its own recent history (a z-score over a
trailing window) rather than raw levels across them.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

_CSV_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{name}pc.csv"
_DAILY_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
_CSV_END = date(2019, 10, 4)
_DAILY_START = date(2019, 10, 7)
_KINDS = {
    # kind: (CSV file stem or None, section in the daily JSON)
    "equity": ("equity", "EQUITY OPTIONS"),
    "index": ("index", "INDEX OPTIONS"),
    "total": ("total", "SUM OF ALL PRODUCTS"),
    "etp": (None, "EXCHANGE TRADED PRODUCTS"),
    "vix": (None, "CBOE VOLATILITY INDEX (VIX)"),
    "spx": (None, "SPX + SPXW"),
}
_NOTES = [
    "2012-06-11: Cboe equity and index volumes exclude ETPs from here on.",
    "2019-10-07: source switches from Cboe's historical CSV to its daily statistics files.",
    "Volume is Cboe's published daily volume; compare a day with its own recent history, not raw levels across breaks.",
]
_HEADERS = {"User-Agent": "Mozilla/5.0 (tokio-ai)"}


def _cache_dir(cache_dir) -> Path:
    base = Path(cache_dir) if cache_dir else Path(os.environ.get("TOKIO_AI_CACHE", Path.home() / ".tokio_ai" / "cache"))
    path = base / "cboe_daily"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_day(d: date, folder: Path, today: date):
    """Cached daily file -> parsed dict, or None for a non-trading day."""
    path, miss = folder / f"{d}.json", folder / f"{d}.none"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if miss.exists():
        return None
    for attempt in range(3):
        try:
            body = _get(_DAILY_URL.format(d=d))
            data = json.loads(body)
            path.write_bytes(body)
            return data
        except (urllib.error.HTTPError, json.JSONDecodeError) as exc:
            # Weekends and holidays come back as 403/404 or a non-JSON page.
            if isinstance(exc, urllib.error.HTTPError) and exc.code >= 500 and attempt < 2:
                continue
            if (today - d).days > 3:  # recent days may simply not be published yet
                miss.touch()
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
    return None


def _parse_day(data: dict, section: str):
    rows = data.get(section) or []
    vol = next((x for x in rows if x.get("name") == "VOLUME"), None)
    if not vol:
        return None
    return float(vol.get("call") or 0), float(vol.get("put") or 0)


def _read_csv(stem: str, pd):
    raw = _get(_CSV_URL.format(name=stem)).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), skiprows=2, skipinitialspace=True)
    df.columns = [str(c).strip().upper() for c in df.columns]
    calls = df["CALLS"] if "CALLS" in df else df["CALL"]
    puts = df["PUTS"] if "PUTS" in df else df["PUT"]
    out = pd.DataFrame({"call": pd.to_numeric(calls, errors="coerce"), "put": pd.to_numeric(puts, errors="coerce")})
    out.index = pd.to_datetime(df["DATE"].astype(str).str.strip(), format="mixed", errors="coerce")
    out = out[out.index.notna()].dropna()
    return out[out.index <= pd.Timestamp(_CSV_END)]


def load_put_call(kind: str = "equity", start=None, end=None, *, cache_dir=None, workers: int = 8):
    """Daily Cboe call volume, put volume and put/call ratio.

    `kind`: "equity", "index", "total" (from 2006-11), or "etp", "vix", "spx"
    (from 2019-10). Returns a DataFrame indexed by date with columns call,
    put, ratio (= put / call) and source ("csv" or "daily").

    The first call downloads about 1,800 small daily files (a few minutes);
    they are cached, so later calls are instant. Pass `start` to fetch less.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_put_call needs pandas: pip install pandas") from exc
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    stem, section = _KINDS[kind]
    start = pd.Timestamp(start).date() if start is not None else (date(2006, 11, 1) if stem else _DAILY_START)
    end = pd.Timestamp(end).date() if end is not None else date.today()
    if end < start:
        raise ValueError("end is before start")

    parts = []
    if stem and start <= _CSV_END:
        csv = _read_csv(stem, pd)
        parts.append(csv.assign(source="csv"))

    if end >= _DAILY_START:
        first = max(start, _DAILY_START)
        days = [first + timedelta(n) for n in range((end - first).days + 1)]
        days = [d for d in days if d.weekday() < 5]
        folder, today = _cache_dir(cache_dir), date.today()
        with ThreadPoolExecutor(max(1, workers)) as ex:
            fetched = list(ex.map(lambda d: _fetch_day(d, folder, today), days))
        rows = {}
        for d, data in zip(days, fetched):
            if data is None:
                continue
            cp = _parse_day(data, section)
            if cp is not None:
                rows[pd.Timestamp(d)] = cp
        if rows:
            daily = pd.DataFrame.from_dict(rows, orient="index", columns=["call", "put"])
            parts.append(daily.assign(source="daily"))

    if not parts:
        raise RuntimeError(f"no put/call data for {kind!r} between {start} and {end}")
    df = pd.concat(parts).sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df = df[(df.call > 0) & (df.put > 0)]
    df["ratio"] = df.put / df.call
    df = df[["call", "put", "ratio", "source"]]
    df.index.name = "date"
    df.attrs["notes"] = list(_NOTES)
    df.attrs["kind"] = kind
    return df
