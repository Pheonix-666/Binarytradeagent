"""Tests for tools/convert_history.py historical data converter."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
import pandas as pd

from core.config import CalendarConfig, CalendarWindowConfig, DataConfig, load_config
from tools.convert_history import (
    parse_tz,
    detect_format,
    parse_binance,
    parse_histdata,
    validate_candles,
    generate_gap_report,
    convert_history,
    main,
)
from backtest.data_loader import load_history_data


# ── Timezone parsing tests ─────────────────────────────────────────────


def test_parse_tz_est_fixed_offset():
    """HistData EST timezone must be fixed at UTC-5 with no DST transitions."""
    tz = parse_tz("EST")
    # Winter date (January): UTC-5
    dt_winter = datetime(2021, 1, 15, 12, 0, tzinfo=tz)
    assert dt_winter.utcoffset() == timedelta(hours=-5)
    assert dt_winter.astimezone(timezone.utc) == datetime(2021, 1, 15, 17, 0, tzinfo=timezone.utc)

    # Summer date (July): still strictly UTC-5 (no DST shift to UTC-4)
    dt_summer = datetime(2021, 7, 15, 12, 0, tzinfo=tz)
    assert dt_summer.utcoffset() == timedelta(hours=-5)
    assert dt_summer.astimezone(timezone.utc) == datetime(2021, 7, 15, 17, 0, tzinfo=timezone.utc)


def test_parse_tz_custom_offsets():
    """parse_tz must parse custom offset formats like UTC+2, -05:00, and standard UTC."""
    assert parse_tz("UTC") == timezone.utc
    assert parse_tz("Z") == timezone.utc

    tz_plus2 = parse_tz("UTC+2")
    dt = datetime(2021, 1, 1, 12, 0, tzinfo=tz_plus2)
    assert dt.utcoffset() == timedelta(hours=2)

    tz_minus5 = parse_tz("-05:00")
    dt_m5 = datetime(2021, 1, 1, 12, 0, tzinfo=tz_minus5)
    assert dt_m5.utcoffset() == timedelta(hours=-5)


# ── Binance parser tests ───────────────────────────────────────────────


def test_binance_millisecond_timestamp(tmp_path: Path):
    """Binance klines with 13-digit millisecond timestamps are parsed to UTC."""
    content = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1609459200000,29000.5,29050.0,28990.0,29020.0,12.5,1609459259999,0,0,0,0,0\n"
        "1609459260000,29020.0,29080.0,29010.0,29070.0,15.2,1609459319999,0,0,0,0,0\n"
    )
    file_path = tmp_path / "binance_ms.csv"
    file_path.write_text(content, encoding="utf-8")

    df = parse_binance(file_path)
    assert len(df) == 2
    assert list(df.columns) == ["ts", "open", "high", "low", "close"]

    # 1609459200000 ms = 2021-01-01 00:00:00 UTC
    assert df["ts"].iloc[0] == datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert df["ts"].iloc[1] == datetime(2021, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    assert df["open"].iloc[0] == 29000.5
    assert df["high"].iloc[0] == 29050.0
    assert df["low"].iloc[0] == 28990.0
    assert df["close"].iloc[0] == 29020.0


def test_binance_microsecond_timestamp(tmp_path: Path):
    """Binance klines with 16-digit microsecond timestamps are auto-detected and parsed to UTC."""
    content = (
        "1609459200000000,30000.0,30100.0,29950.0,30050.0,20.0,1609459259999999,0,0,0,0,0\n"
        "1609459260000000,30050.0,30120.0,30020.0,30080.0,18.5,1609459319999999,0,0,0,0,0\n"
    )
    file_path = tmp_path / "binance_us.csv"
    file_path.write_text(content, encoding="utf-8")

    assert detect_format(file_path) == "binance"

    df = parse_binance(file_path)
    assert len(df) == 2
    # 1609459200000000 us = 2021-01-01 00:00:00 UTC
    assert df["ts"].iloc[0] == datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert df["ts"].iloc[1] == datetime(2021, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    assert df["open"].iloc[0] == 30000.0
    assert df["high"].iloc[0] == 30100.0
    assert df["low"].iloc[0] == 29950.0
    assert df["close"].iloc[0] == 30050.0


# ── HistData parser tests ──────────────────────────────────────────────


def test_histdata_semicolon_est_conversion(tmp_path: Path):
    """HistData semicolon-delimited M1 file in EST converts to UTC correctly without DST."""
    content = (
        "20210103 170000;1.223400;1.223800;1.223200;1.223500;0\n"
        "20210701 120000;1.185000;1.185500;1.184800;1.185200;0\n"
    )
    file_path = tmp_path / "histdata_sample.csv"
    file_path.write_text(content, encoding="utf-8")

    assert detect_format(file_path) == "histdata"

    df = parse_histdata(file_path, source_tz="EST")
    assert len(df) == 2
    assert df["ts"].iloc[0] == datetime(2021, 1, 3, 22, 0, 0, tzinfo=timezone.utc)
    assert df["ts"].iloc[1] == datetime(2021, 7, 1, 17, 0, 0, tzinfo=timezone.utc)
    assert df["open"].iloc[0] == 1.2234
    assert df["high"].iloc[0] == 1.2238
    assert df["low"].iloc[0] == 1.2232
    assert df["close"].iloc[0] == 1.2235


def test_histdata_split_date_time_and_custom_tz(tmp_path: Path):
    """HistData with separate date and time columns, converted with custom source_tz."""
    content = (
        "20210103;170000;1.2234;1.2238;1.2232;1.2235;0\n"
        "20210103;170100;1.2235;1.2239;1.2234;1.2238;0\n"
    )
    file_path = tmp_path / "histdata_split.csv"
    file_path.write_text(content, encoding="utf-8")

    # With source_tz = UTC+2, 17:00 local is 15:00 UTC
    df = parse_histdata(file_path, source_tz="UTC+2")
    assert len(df) == 2
    assert df["ts"].iloc[0] == datetime(2021, 1, 3, 15, 0, 0, tzinfo=timezone.utc)
    assert df["ts"].iloc[1] == datetime(2021, 1, 3, 15, 1, 0, tzinfo=timezone.utc)


def test_histdata_comma_delimited_with_header(tmp_path: Path):
    """HistData comma-delimited file with header."""
    content = (
        "DateTime,Open,High,Low,Close,Volume\n"
        "2021-01-03 17:00:00,1.2234,1.2238,1.2232,1.2235,100\n"
        "2021-01-03 17:01:00,1.2235,1.2240,1.2233,1.2239,150\n"
    )
    file_path = tmp_path / "histdata_header.csv"
    file_path.write_text(content, encoding="utf-8")

    df = parse_histdata(file_path, source_tz="EST")
    assert len(df) == 2
    assert df["ts"].iloc[0] == datetime(2021, 1, 3, 22, 0, 0, tzinfo=timezone.utc)
    assert df["close"].iloc[1] == 1.2239


# ── Validation tests (Default Fail vs --drop-invalid) ──────────────────


def test_validate_candles_positive_and_bounds():
    """Candle validation ensures high >= max(open, close), low <= min(open, close), and prices > 0."""
    t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    valid_df = pd.DataFrame([
        {"ts": t0, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0},
        {"ts": t0 + timedelta(minutes=1), "open": 11.0, "high": 11.0, "low": 8.0, "close": 8.5},
    ])
    out, count, _ = validate_candles(valid_df)
    assert len(out) == 2
    assert count == 0

    # High lower than open
    bad_high = pd.DataFrame([
        {"ts": t0, "open": 10.0, "high": 9.5, "low": 9.0, "close": 9.2},
    ])
    with pytest.raises(ValueError, match="Validation failed"):
        validate_candles(bad_high)

    # Low higher than close
    bad_low = pd.DataFrame([
        {"ts": t0, "open": 10.0, "high": 12.0, "low": 10.5, "close": 10.2},
    ])
    with pytest.raises(ValueError, match="Validation failed"):
        validate_candles(bad_low)

    # Negative price
    bad_negative = pd.DataFrame([
        {"ts": t0, "open": 10.0, "high": 12.0, "low": -1.0, "close": 10.0},
    ])
    with pytest.raises(ValueError, match="Validation failed"):
        validate_candles(bad_negative)


def test_invalid_rows_default_fail_prints_first_10_and_count(tmp_path: Path):
    """By default, validation fails (raises ValueError) with count and first 10 offending rows."""
    t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # 12 invalid rows
    rows = []
    for i in range(12):
        rows.append({"ts": t0 + timedelta(minutes=i), "open": 10.0, "high": 8.0, "low": 7.0, "close": 9.0})  # high < open
    invalid_df = pd.DataFrame(rows)

    with pytest.raises(ValueError) as excinfo:
        validate_candles(invalid_df, drop_invalid=False)

    err = str(excinfo.value)
    assert "found 12 invalid candle row(s)" in err
    # Verifies first 10 rows are formatted in error message
    assert "  1. ts=" in err
    assert " 10. ts=" in err
    assert " 11. ts=" not in err


def test_drop_invalid_never_drops_silently(tmp_path: Path, capsys: pytest.CaptureFixture):
    """--drop-invalid drops offending rows, prints count dropped (never silent), and writes clean data."""
    t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    valid_row = {"ts": t0, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5}
    bad_row1 = {"ts": t0 + timedelta(minutes=1), "open": 10.0, "high": 8.0, "low": 7.0, "close": 9.0}
    bad_row2 = {"ts": t0 + timedelta(minutes=2), "open": -5.0, "high": 10.0, "low": -6.0, "close": 5.0}

    df = pd.DataFrame([valid_row, bad_row1, bad_row2])
    cleaned, dropped_count, first_10 = validate_candles(df, drop_invalid=True)

    captured = capsys.readouterr()
    assert dropped_count == 2
    assert len(cleaned) == 1
    assert "Dropped 2 invalid candle row(s) failing price rules." in captured.out


# ── Duplicate detection (Identical vs Different Prices) ────────────────


def test_duplicate_timestamps_reporting_and_first_kept(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Duplicate timestamps are reported, different-price duplicates are counted, and first row is kept."""
    # Write file with:
    # 00:00 - initial row (open 10.0)
    # 00:00 - exact duplicate (open 10.0)
    # 00:00 - different-price duplicate (open 12.0)
    # 00:01 - unique row
    content = (
        "open_time,open,high,low,close,volume\n"
        "1609459200000,10.0,11.0,9.0,10.0,1.0\n"   # 00:00
        "1609459200000,10.0,11.0,9.0,10.0,1.0\n"   # 00:00 (exact)
        "1609459200000,12.0,13.0,11.0,12.5,1.0\n"  # 00:00 (different price)
        "1609459260000,10.5,11.5,9.5,10.2,1.0\n"   # 00:01
    )
    in_file = tmp_path / "dups.csv"
    in_file.write_text(content, encoding="utf-8")

    out_file, df, _ = convert_history(
        input_files=in_file,
        asset="BTCUSD",
        output_dir=tmp_path / "out",
        print_report=True,
    )

    captured = capsys.readouterr()
    # 2 duplicate rows total: 1 with different prices, 1 identical
    assert "Duplicate timestamps: 2 (1 with different prices, keeping first row; 1 identical)" in captured.out
    assert len(df) == 2
    # Verify the first row (open=10.0) was preserved
    assert df["open"].iloc[0] == 10.0


# ── Timestamp convention & --ts-is-close flag ──────────────────────────


def test_ts_is_close_flag_shifts_timestamps_back_one_minute(tmp_path: Path):
    """--ts-is-close shifts timestamps back 1 minute so ts reflects candle OPEN time."""
    # A source that provides candle CLOSE time:
    # 00:01:00 close time corresponds to 00:00:00 open time
    content = (
        "open_time,open,high,low,close,volume\n"
        "1609459260000,29000.0,29050.0,28950.0,29020.0,10.0\n"  # 2021-01-01 00:01:00 UTC
    )
    in_file = tmp_path / "close_ts.csv"
    in_file.write_text(content, encoding="utf-8")

    # Without flag: remains 00:01:00
    _, df_normal, _ = convert_history(in_file, asset="BTCUSD", output_dir=tmp_path / "normal", ts_is_close=False, print_report=False)
    assert df_normal["ts"].iloc[0] == "2021-01-01T00:01:00Z"

    # With ts_is_close=True: shifts back to 00:00:00
    _, df_shifted, _ = convert_history(in_file, asset="BTCUSD", output_dir=tmp_path / "shifted", ts_is_close=True, print_report=False)
    assert df_shifted["ts"].iloc[0] == "2021-01-01T00:00:00Z"


# ── Calendar-aware Gap Report Tests ────────────────────────────────────


def test_calendar_gap_report_forex_weekend_suppressed():
    """Gap report does NOT flag forex weekend market closures as gaps."""
    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="Sun",
                open_time="17:00",
                close_day="Fri",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )
    mock_cfg = SimpleNamespace(
        data=DataConfig(
            calendars={"forex": forex_cal},
            assets={"EURUSD": "forex"},
        )
    )

    fri_close = datetime(2021, 1, 8, 22, 0, 0, tzinfo=timezone.utc)
    sun_open = datetime(2021, 1, 10, 22, 0, 0, tzinfo=timezone.utc)

    df = pd.DataFrame([
        {"ts": fri_close - timedelta(minutes=1), "open": 1.2, "high": 1.21, "low": 1.19, "close": 1.2},
        {"ts": fri_close, "open": 1.2, "high": 1.21, "low": 1.19, "close": 1.2},
        {"ts": sun_open, "open": 1.2, "high": 1.21, "low": 1.19, "close": 1.2},
        {"ts": sun_open + timedelta(minutes=1), "open": 1.2, "high": 1.21, "low": 1.19, "close": 1.2},
    ])

    report, gaps = generate_gap_report(df, asset="EURUSD", config=mock_cfg)
    assert len(gaps) == 0
    assert "CLEAN (0 unexpected market gaps)" in report


def test_calendar_gap_report_flags_intraweek_gaps():
    """Gap report flags unexpected missing minutes during open market hours."""
    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="Sun",
                open_time="17:00",
                close_day="Fri",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )
    mock_cfg = SimpleNamespace(
        data=DataConfig(
            calendars={"forex": forex_cal},
            assets={"EURUSD": "forex"},
        )
    )

    tue_1 = datetime(2021, 1, 12, 10, 0, 0, tzinfo=timezone.utc)
    tue_2 = datetime(2021, 1, 12, 11, 0, 0, tzinfo=timezone.utc)  # 60-minute gap

    df = pd.DataFrame([
        {"ts": tue_1, "open": 1.2, "high": 1.21, "low": 1.19, "close": 1.2},
        {"ts": tue_2, "open": 1.2, "high": 1.21, "low": 1.19, "close": 1.2},
    ])

    report, gaps = generate_gap_report(df, asset="EURUSD", config=mock_cfg)
    assert len(gaps) == 1
    assert "1 UNEXPECTED GAP(S) DETECTED" in report
    assert "59 missing minutes" in report


# ── Backtest Compatibility, CLI, and Final Summary Tests ───────────────


def test_converted_file_compatible_with_backtest_loader(tmp_path: Path):
    """The output of convert_history must load seamlessly in backtest.data_loader."""
    content = (
        "20210103 170000;1.2234;1.2238;1.2232;1.2235;0\n"
        "20210103 170100;1.2235;1.2240;1.2234;1.2239;0\n"
        "20210103 170200;1.2239;1.2242;1.2237;1.2241;0\n"
    )
    in_file = tmp_path / "eurusd_hist.csv"
    in_file.write_text(content, encoding="utf-8")

    out_file, _, _ = convert_history(
        input_files=in_file,
        asset="EURUSD",
        output_dir=tmp_path / "data_history",
        print_report=False,
    )

    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="Sun",
                open_time="17:00",
                close_day="Fri",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )

    loaded_df = load_history_data(out_file, calendar=forex_cal)
    assert len(loaded_df) == 3
    assert "ts_close" in loaded_df.columns
    assert loaded_df["ts"].iloc[0] == datetime(2021, 1, 3, 22, 0, 0, tzinfo=timezone.utc)


def test_final_one_line_summary_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture):
    """CLI prints gap report, final one-line summary with all 6 fields, and exits 0."""
    content = (
        "open_time,open,high,low,close,volume\n"
        "1609459200000,29000.0,29050.0,28950.0,29020.0,10.0\n"
        "1609459260000,29020.0,29080.0,29010.0,29060.0,8.0\n"
    )
    in_file = tmp_path / "input.csv"
    in_file.write_text(content, encoding="utf-8")
    out_dir = tmp_path / "data_history"

    ret = main([
        str(in_file),
        "--asset", "BTCUSD",
        "--output-dir", str(out_dir),
    ])

    assert ret == 0
    captured = capsys.readouterr()
    assert "CALENDAR GAP REPORT: BTCUSD" in captured.out
    # Check final one-line summary
    assert "Summary: rows_read=2, duplicates=0 (different_prices=0), invalid=0, rows_written=2, date_range=2021-01-01T00:00:00Z -> 2021-01-01T00:01:00Z, gaps=0" in captured.out
    assert (out_dir / "BTCUSD.csv").exists()


def test_cli_fails_on_invalid_rows_without_drop_invalid(tmp_path: Path, capsys: pytest.CaptureFixture):
    """CLI fails (returns 1) and prints first 10 offending rows when invalid rows exist without --drop-invalid."""
    content = (
        "open_time,open,high,low,close,volume\n"
        "1609459200000,29000.0,28000.0,27000.0,27500.0,10.0\n"  # high (28000) < open (29000)
    )
    in_file = tmp_path / "bad.csv"
    in_file.write_text(content, encoding="utf-8")

    ret = main([
        str(in_file),
        "--asset", "BTCUSD",
        "--output-dir", str(tmp_path / "out"),
    ])

    assert ret != 0
    captured = capsys.readouterr()
    assert "Validation failed: found 1 invalid candle row(s)" in captured.err
    assert "1. ts=" in captured.err
