from pathlib import Path

import pytest

from tokio_ai.trades import is_tradingview_header, on_calendar, read_tradingview

DATA = Path(__file__).parent / "data" / "tradingview_2025_format.csv"


def _write(tmp_path, text, name="t.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_reads_the_current_export_format():
    t = read_tradingview(DATA)
    assert len(t) == 5
    assert t.returns[0] == pytest.approx(0.8691)
    assert t.sides[0] == "long" and "short" in t.sides
    assert t.periods_per_year > 0


def test_equity_basis_uses_cumulative_columns():
    t = read_tradingview(DATA, basis="equity")
    # initial capital = 8692.71 / 0.8693 ~ 10,000
    assert t.returns[0] == pytest.approx(8692.71 / (8692.71 / 0.8693), rel=1e-3)


def test_old_format_profit_columns_and_entry_after_exit(tmp_path):
    p = _write(tmp_path, "Trade #,Type,Signal,Date/Time,Price,Contracts,Profit,Profit %\n"
                         "2,Exit Short,Close,2024-01-04 00:00,95,2,10,5.26\n"
                         "2,Entry Short,S,2024-01-03 00:00,100,2,,\n"
                         "1,Exit Long,Close,2024-01-02 00:00,110,1,10,10\n"
                         "1,Entry Long,L,2024-01-01 00:00,100,1,,\n")
    t = read_tradingview(p)
    assert t.returns == pytest.approx([0.10, 0.0526])  # sorted into exit order
    assert t.sides == ["long", "short"]


def test_falls_back_to_pnl_over_entry_notional(tmp_path):
    p = _write(tmp_path, "Trade #,Type,Date/Time,Signal,Price USD,Contracts,Profit USD\n"
                         "1,Entry long,2024-01-01 00:00,L,50,4,\n"
                         "1,Exit long,2024-01-05 00:00,X,55,4,20\n")
    assert read_tradingview(p).returns == pytest.approx([0.10])


def test_open_trades_are_skipped(tmp_path):
    p = _write(tmp_path, "Trade #,Type,Date/Time,Signal,Price USD,Net P&L USD,Net P&L %\n"
                         "1,Entry long,2024-01-01 00:00,L,50,,\n"
                         "1,Exit long,2024-01-05 00:00,X,55,5,10\n"
                         "2,Entry long,2024-02-01 00:00,L,50,,\n"
                         "2,Exit long,2024-02-05 00:00,Open,52,2,4\n")
    t = read_tradingview(p)
    assert len(t) == 1 and t.skipped_open == 1


def test_iso_times_with_offsets_and_semicolons(tmp_path):
    p = _write(tmp_path, "Trade #;Type;Date/Time;Signal;Price USD;Net P&L USD;Net P&L %\n"
                         "1;Entry long;2022-01-03T04:00:00-0500;L;1.1;;\n"
                         "1;Exit long;2022-01-03T14:00:00-0500;X;1.2;0.1;9.09\n")
    t = read_tradingview(p)
    assert t.exit_times[0] is not None and t.returns == pytest.approx([0.0909])


def test_rejects_non_tradingview(tmp_path):
    p = _write(tmp_path, "date,pnl\n2024-01-01,0.01\n")
    with pytest.raises(ValueError, match="not a TradingView"):
        read_tradingview(p)
    assert not is_tradingview_header(["date", "pnl"])


def test_calendar_alignment_and_check_backtest(tmp_path):
    pytest.importorskip("numpy")
    from tokio_ai import check_backtest

    a = read_tradingview(DATA)
    series, ppy = on_calendar({"a": a, "b": a})
    assert len(series["a"]) == len(series["b"]) and ppy > 0
    res = check_backtest(a)
    assert res.bar_unit == "trades" and "per closed trade" in str(res)


def _export(tmp_path, n, seed, name):
    import random

    rng = random.Random(seed)
    lines = ["Trade #,Type,Date/Time,Signal,Price USD,Net P&L USD,Net P&L %"]
    for i in range(1, n + 1):
        pct = rng.gauss(0.1, 1.0)
        lines.append(f"{i},Exit long,2024-{1 + i // 28:02d}-{1 + i % 28:02d} 16:00,X,100,{pct},{pct}")
        lines.append(f"{i},Entry long,2024-{1 + i // 28:02d}-{1 + i % 28:02d} 09:30,L,100,,")
    return _write(tmp_path, "\n".join(lines) + "\n", name)


def test_cli_reads_tradingview_exports(tmp_path, capsys):
    pytest.importorskip("numpy")
    from tokio_ai.backtest_cli import main

    assert main([str(_export(tmp_path, 200, 1, "a.csv")), "--trials", "5"]) == 0
    out = capsys.readouterr().out
    assert "5 trials" in out and "200 trades" in out
    assert main([str(DATA)]) == 0
    assert "NOT REPORTABLE: 5 usable trades" in capsys.readouterr().out
    # two exports become two variants
    assert main([str(_export(tmp_path, 200, 2, "x.csv")), str(_export(tmp_path, 150, 3, "y.csv"))]) == 0
    assert "2 trials" in capsys.readouterr().out
    plain = _write(tmp_path, "pnl\n0.01\n", "plain.csv")
    assert main([str(DATA), str(plain)]) == 2
