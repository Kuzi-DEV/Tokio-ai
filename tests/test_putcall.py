import json
import urllib.error
from datetime import date

import pytest

pd = pytest.importorskip("pandas")

import tokio_ai.putcall as pc

CSV = """Disclaimer text,,,,
, PRODUCT: EQUITY,,EXCHANGE: Cboe,
DATE,CALL,PUT,TOTAL,P/C Ratio
10/03/2019, 1000, 500, 1500, 0.50
10/04/2019, 800, 800, 1600, 1.00
"""


def _day(call, put):
    return {"EQUITY OPTIONS": [{"name": "VOLUME", "call": call, "put": put, "total": call + put}]}


@pytest.fixture
def fake_net(monkeypatch):
    calls = []
    days = {date(2019, 10, 7): _day(1000, 700), date(2019, 10, 8): _day(1000, 900)}

    def get(url, timeout=30.0):
        calls.append(url)
        if url.endswith("pc.csv"):
            return CSV.encode()
        d = date.fromisoformat(url.rsplit("/", 1)[1][:10])
        if d in days:
            return json.dumps(days[d]).encode()
        raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)

    monkeypatch.setattr(pc, "_get", get)
    return calls


def test_stitches_csv_and_daily(fake_net, tmp_path):
    df = pc.load_put_call("equity", start="2019-10-01", end="2019-10-10", cache_dir=tmp_path)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2019-10-03", "2019-10-04", "2019-10-07", "2019-10-08"]
    assert list(df.source) == ["csv", "csv", "daily", "daily"]
    assert df.ratio.round(2).tolist() == [0.5, 1.0, 0.7, 0.9]
    assert any("2012-06-11" in n for n in df.attrs["notes"])


def test_cache_means_no_second_download(fake_net, tmp_path):
    pc.load_put_call("equity", start="2019-10-07", end="2019-10-10", cache_dir=tmp_path)
    n = len(fake_net)
    pc.load_put_call("equity", start="2019-10-07", end="2019-10-10", cache_dir=tmp_path)
    assert len(fake_net) == n  # trading days cached, holidays remembered as misses


def test_bad_kind_and_daily_only_kinds(fake_net, tmp_path):
    with pytest.raises(ValueError):
        pc.load_put_call("bogus")
    with pytest.raises(RuntimeError):  # spx exists only in the daily files; none in this fake
        pc.load_put_call("spx", start="2019-10-07", end="2019-10-08", cache_dir=tmp_path)
