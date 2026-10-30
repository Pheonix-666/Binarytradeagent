"""Unit tests for candle builder, resampling, and partial-candle handling."""

from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest

from data.candle_builder import (
    CandleBuilder,
    resample_candles,
    ticks_to_1m,
)


def test_ticks_to_1m_empty():
    """Empty ticks dataframe returns an empty dataframe with standard columns."""
    df_empty = pd.DataFrame(columns=["ts", "price"])
    result = ticks_to_1m(df_empty)
    assert result.empty
    assert list(result.columns) == ["ts", "ts_close", "open", "high", "low", "close", "volume"]


def test_ticks_to_1m_partial_candle_excluded():
    """A 1m candle still in progress must NOT be returned as closed."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    # Ticks at 10:00:10, 10:00:30, 10:00:50. Current time is 10:00:55.
    ticks = pd.DataFrame([
        {"ts": base + timedelta(seconds=10), "price": 100.0, "volume": 1.0},
        {"ts": base + timedelta(seconds=30), "price": 105.0, "volume": 2.0},
        {"ts": base + timedelta(seconds=50), "price": 102.0, "volume": 1.0},
    ])
    now = base + timedelta(seconds=55)
    result = ticks_to_1m(ticks, current_time=now)
    # The 10:00 candle is not closed because now < 10:01:00
    assert result.empty


def test_ticks_to_1m_closed_candle():
    """A 1m candle is closed once the minute ends (current_time >= 10:01:00)."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    ticks = pd.DataFrame([
        {"ts": base + timedelta(seconds=5), "price": 100.0, "volume": 1.0},
        {"ts": base + timedelta(seconds=20), "price": 108.0, "volume": 2.0},
        {"ts": base + timedelta(seconds=35), "price": 99.0, "volume": 1.5},
        {"ts": base + timedelta(seconds=50), "price": 104.0, "volume": 3.0},
    ])
    now = base + timedelta(minutes=1, seconds=2)
    result = ticks_to_1m(ticks, current_time=now)
    assert len(result) == 1
    candle = result.iloc[0]
    assert candle["ts"] == base
    assert candle["open"] == 100.0
    assert candle["high"] == 108.0
    assert candle["low"] == 99.0
    assert candle["close"] == 104.0
    assert candle["volume"] == 7.5


def test_ticks_to_1m_multiple_minutes_with_partial():
    """Given ticks across 10:00, 10:01, 10:02, only 10:00 and 10:01 should be closed at 10:02:30."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    records = []
    # 10:00 minute
    records.append({"ts": base + timedelta(seconds=15), "price": 100.0})
    records.append({"ts": base + timedelta(seconds=45), "price": 102.0})
    # 10:01 minute
    records.append({"ts": base + timedelta(minutes=1, seconds=10), "price": 103.0})
    records.append({"ts": base + timedelta(minutes=1, seconds=50), "price": 101.0})
    # 10:02 minute (in progress)
    records.append({"ts": base + timedelta(minutes=2, seconds=15), "price": 105.0})

    df_ticks = pd.DataFrame(records)
    now = base + timedelta(minutes=2, seconds=30)
    result = ticks_to_1m(df_ticks, current_time=now)

    assert len(result) == 2
    assert result.iloc[0]["ts"] == base
    assert result.iloc[1]["ts"] == base + timedelta(minutes=1)


def test_resample_candles_to_5m_partial_excluded():
    """If fewer than 5 minutes elapsed in a 5m bar, it must be excluded."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    # Only 3 closed 1m bars: 10:00, 10:01, 10:02
    candles_1m = pd.DataFrame([
        {"ts": base + timedelta(minutes=i), "open": 100 + i, "high": 102 + i, "low": 99 + i, "close": 101 + i, "volume": 10}
        for i in range(3)
    ])
    # Cutoff at 10:03 (end of data is 10:03, which is < 10:05)
    res = resample_candles(candles_1m, target_timeframe="5m")
    assert res.empty


def test_resample_candles_to_5m_complete():
    """A full 5m bar (10:00-10:04) must produce exactly one closed 5m candle."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    # 5 closed 1m bars: 10:00, 10:01, 10:02, 10:03, 10:04
    candles_1m = pd.DataFrame([
        {"ts": base + timedelta(minutes=0), "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 10},
        {"ts": base + timedelta(minutes=1), "open": 102.0, "high": 107.0, "low": 101.0, "close": 106.0, "volume": 12},
        {"ts": base + timedelta(minutes=2), "open": 106.0, "high": 108.0, "low": 104.0, "close": 105.0, "volume": 8},
        {"ts": base + timedelta(minutes=3), "open": 105.0, "high": 106.0, "low": 97.0, "close": 99.0, "volume": 15},
        {"ts": base + timedelta(minutes=4), "open": 99.0, "high": 103.0, "low": 98.5, "close": 101.5, "volume": 20},
    ])
    res = resample_candles(candles_1m, target_timeframe="5m")
    assert len(res) == 1
    c = res.iloc[0]
    assert c["ts"] == base
    assert c["open"] == 100.0   # first open
    assert c["high"] == 108.0   # max high
    assert c["low"] == 97.0     # min low
    assert c["close"] == 101.5  # last close
    assert c["volume"] == 65.0  # sum of volume


def test_resample_candles_to_15m_boundary():
    """14 minutes of 1m candles produces 0 closed 15m candles; 15 minutes produces 1."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    candles_14 = pd.DataFrame([
        {"ts": base + timedelta(minutes=i), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1}
        for i in range(14)
    ])
    assert resample_candles(candles_14, target_timeframe="15m").empty

    candles_15 = pd.DataFrame([
        {"ts": base + timedelta(minutes=i), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1}
        for i in range(15)
    ])
    res15 = resample_candles(candles_15, target_timeframe="15m")
    assert len(res15) == 1
    assert res15.iloc[0]["ts"] == base


def test_resample_invalid_timeframe_raises():
    """Unsupported timeframe should raise ValueError."""
    df = pd.DataFrame([{"ts": datetime.now(timezone.utc), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}])
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        resample_candles(df, target_timeframe="3m")


def test_stateful_candle_builder():
    """Stateful CandleBuilder accumulates ticks and produces closed candles."""
    builder = CandleBuilder()
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)

    # Add ticks during 10:00
    closed1 = builder.add_tick("EURUSD", base + timedelta(seconds=10), price=1.1000)
    assert closed1 == []

    closed2 = builder.add_tick("EURUSD", base + timedelta(seconds=40), price=1.1020)
    assert closed2 == []

    # Tick at 10:01:05 closes the 10:00 candle
    closed3 = builder.add_tick("EURUSD", base + timedelta(minutes=1, seconds=5), price=1.1015)
    assert len(closed3) == 1
    assert closed3[0]["open"] == 1.1000
    assert closed3[0]["close"] == 1.1020

    # Query closed candles via get_candles
    df_1m = builder.get_candles("EURUSD", timeframe="1m")
    assert len(df_1m) == 1
    assert df_1m.iloc[0]["close"] == 1.1020


def test_resample_candles_to_5m_incomplete_component_bars_skipped():
    """If a 5m bucket has only 4 1m bars (one missing in the middle), it must be SKIPPED."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    # 1m bars at 10:00, 10:01, 10:02, and 10:04 (10:03 is missing!)
    # Plus a bar at 10:05 to close the 5m interval
    records = [
        {"ts": base + timedelta(minutes=0), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"ts": base + timedelta(minutes=1), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"ts": base + timedelta(minutes=2), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        # minute 3 missing!
        {"ts": base + timedelta(minutes=4), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"ts": base + timedelta(minutes=5), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
    ]
    df_1m = pd.DataFrame(records)
    res = resample_candles(df_1m, target_timeframe="5m")
    # Bucket [10:00, 10:05) only had 4 bars out of 5 -> must NOT be emitted
    assert res.empty


def test_resample_candles_daily_non_utc_tz():
    """Daily resampling under a non-UTC daily_boundary_tz aligns to that timezone's midnight."""
    # 24 hours of 1m bars in UTC
    base_utc = datetime(2026, 10, 29, 4, 0, 0, tzinfo=timezone.utc)  # midnight America/New_York (UTC-4)
    records = [
        {"ts": base_utc + timedelta(minutes=i), "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1}
        for i in range(1440)
    ]
    df_1m = pd.DataFrame(records)
    res_daily = resample_candles(df_1m, target_timeframe="1d", daily_boundary_tz="America/New_York")
    assert len(res_daily) == 1
    # Boundary aligned to midnight in America/New_York (which is 04:00 UTC)
    assert res_daily.iloc[0]["ts"] == base_utc


def test_detect_gaps():
    """Gap detector flags intervals where consecutive candles exceed expected interval."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    # Candle at 10:00, 10:01, skip 10:02-10:04, next at 10:05
    df = pd.DataFrame([
        {"ts": base, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"ts": base + timedelta(minutes=1), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"ts": base + timedelta(minutes=5), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
    ])
    from data.candle_builder import detect_gaps
    gaps = detect_gaps(df, timeframe="1m")
    assert len(gaps) == 1
    assert gaps[0][0] == base + timedelta(minutes=1)
    assert gaps[0][1] == base + timedelta(minutes=5)


# ── Calendar-Aware Resampling & Gap Tests ──────────────────────────────


def test_forex_weekend_is_not_a_gap():
    """Forex weekend closures (Friday 17:00 NY to Sunday 17:00 NY) are NOT gaps."""
    from core.config import CalendarConfig, CalendarWindowConfig
    from data.candle_builder import detect_gaps

    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="sunday",
                open_time="17:00",
                close_day="friday",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )

    # Friday 2026-10-30 at 16:59 NY (20:59 UTC, market open)
    # Market closes at 17:00 NY (21:00 UTC)
    # Market reopens Sunday 2026-11-01 at 17:00 NY (22:00 UTC, post-DST change)
    fri_close_utc = datetime(2026, 10, 30, 20, 59, 0, tzinfo=timezone.utc)
    sun_open_utc = datetime(2026, 11, 1, 22, 0, 0, tzinfo=timezone.utc)

    df = pd.DataFrame([
        {"ts": fri_close_utc, "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 10},
        {"ts": sun_open_utc, "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 10},
    ])

    gaps = detect_gaps(df, timeframe="1m", calendar=forex_cal)
    # Across the entire weekend, expected_bars == 0, so no gap is recorded
    assert gaps == []


def test_crypto_missing_bars_are_gaps():
    """For 24/7 crypto calendar, weekend missing bars ARE recorded as gaps."""
    from core.config import CalendarConfig
    from data.candle_builder import detect_gaps

    crypto_cal = CalendarConfig(type="always_open", daily_boundary_tz="UTC", daily_boundary_hour=0)

    t1 = datetime(2026, 10, 30, 21, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 30, 23, 0, 0, tzinfo=timezone.utc)  # 2-hour gap

    df = pd.DataFrame([
        {"ts": t1, "open": 60000, "high": 60100, "low": 59900, "close": 60050, "volume": 5},
        {"ts": t2, "open": 60050, "high": 60200, "low": 60000, "close": 60150, "volume": 5},
    ])

    gaps = detect_gaps(df, timeframe="1m", calendar=crypto_cal)
    assert len(gaps) == 1
    assert gaps[0] == (t1, t2)


def test_daily_bar_with_one_missing_minute_emitted():
    """A daily candle with 1 missing minute (1439/1440 = 99.9% >= 98%) IS emitted."""
    from core.config import CalendarConfig, CalendarWindowConfig

    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="sunday",
                open_time="17:00",
                close_day="friday",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )

    # Monday 17:00 NY to Tuesday 17:00 NY in EDT (21:00 UTC Monday to 21:00 UTC Tuesday)
    base_utc = datetime(2026, 10, 26, 21, 0, 0, tzinfo=timezone.utc)
    # 1439 1m bars (omit minute 500)
    records = []
    for m in range(1440):
        if m == 500:
            continue
        records.append({
            "ts": base_utc + timedelta(minutes=m),
            "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 1
        })
    # Add one bar at next daily boundary to close the interval
    records.append({
        "ts": base_utc + timedelta(minutes=1440),
        "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 1
    })

    df = pd.DataFrame(records)
    res = resample_candles(df, target_timeframe="1d", calendar=forex_cal, min_bar_coverage=0.98)
    assert len(res) == 1
    assert res.iloc[0]["ts"] == base_utc


def test_daily_bar_with_5_percent_missing_is_skipped_and_recorded_as_gap():
    """A daily candle with 5% missing minutes (95% < 98% coverage) is SKIPPED and recorded as a gap."""
    from core.config import CalendarConfig, CalendarWindowConfig
    from data.candle_builder import detect_gaps

    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="sunday",
                open_time="17:00",
                close_day="friday",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )

    base_utc = datetime(2026, 10, 26, 21, 0, 0, tzinfo=timezone.utc)
    # Only 1368 bars (95.0% coverage)
    records = []
    for m in range(1368):
        records.append({
            "ts": base_utc + timedelta(minutes=m),
            "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 1
        })
    records.append({
        "ts": base_utc + timedelta(minutes=1440),
        "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 1
    })

    df = pd.DataFrame(records)
    res = resample_candles(df, target_timeframe="1d", calendar=forex_cal, min_bar_coverage=0.98)
    # Skipped because 95% < 98%
    assert res.empty

    # And detect_gaps detects the missing chunk between minute 1367 and 1440
    gaps = detect_gaps(df, timeframe="1m", calendar=forex_cal)
    assert len(gaps) == 1


def test_daily_bars_start_at_1700_ny_across_dst_change():
    """Daily candle start aligns to 17:00 New York both before (EDT = 21:00 UTC) and after DST (EST = 22:00 UTC)."""
    from core.config import CalendarConfig
    from data.calendar import get_daily_bucket_start

    cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
    )

    # 1. October 2026 (Daylight Saving Time EDT, UTC-4)
    # 17:00 EDT is 21:00 UTC
    dt_edt = datetime(2026, 10, 20, 22, 30, 0, tzinfo=timezone.utc)
    bucket_edt = get_daily_bucket_start(dt_edt, cal)
    assert bucket_edt.hour == 21
    assert bucket_edt == datetime(2026, 10, 20, 21, 0, 0, tzinfo=timezone.utc)

    # 2. November 2026 (Standard Time EST, UTC-5)
    # 17:00 EST is 22:00 UTC
    dt_est = datetime(2026, 11, 10, 23, 30, 0, tzinfo=timezone.utc)
    bucket_est = get_daily_bucket_start(dt_est, cal)
    assert bucket_est.hour == 22
    assert bucket_est == datetime(2026, 11, 10, 22, 0, 0, tzinfo=timezone.utc)


def test_no_partial_sunday_candle():
    """The first daily candle of the forex week starts Sunday 17:00 NY and runs 24h to Monday 17:00 NY (no 2h candle)."""
    from core.config import CalendarConfig, CalendarWindowConfig
    from data.calendar import expected_bars, get_daily_bucket_start

    forex_cal = CalendarConfig(
        type="weekly_windows",
        daily_boundary_tz="America/New_York",
        daily_boundary_hour=17,
        windows=[
            CalendarWindowConfig(
                open_day="sunday",
                open_time="17:00",
                close_day="friday",
                close_time="17:00",
                tz="America/New_York",
            )
        ],
    )

    # Sunday 2026-10-25 at 17:00 NY (21:00 UTC)
    sun_1700_ny = datetime(2026, 10, 25, 21, 0, 0, tzinfo=timezone.utc)
    bucket_start = get_daily_bucket_start(sun_1700_ny, forex_cal)
    assert bucket_start == sun_1700_ny

    # Expected bars for this Sunday 17:00 NY candle is 1440 (a full 24h day!)
    exp = expected_bars(bucket_start, "1d", forex_cal)
    assert exp == 1440


def test_resampled_bars_ts_close_is_bucket_end():
    """Verify that resample_candles sets ts_close = bucket end exactly for every timeframe (5m, 15m, 1h, 1d)."""
    from core.config import CalendarConfig
    from data.calendar import get_daily_bucket_end

    cal = CalendarConfig(type="always_open", daily_boundary_tz="UTC", daily_boundary_hour=0)
    base = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)

    # 3 days of 1-minute bars
    rows = []
    for m in range(3 * 1440):
        t = base + timedelta(minutes=m)
        rows.append({
            "ts": t,
            "ts_close": t + timedelta(minutes=1),
            "open": 1.1000,
            "high": 1.1010,
            "low": 1.0990,
            "close": 1.1005,
            "volume": 10.0,
        })
    df_1m = pd.DataFrame(rows)

    expected_deltas = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }

    for tf, delta in expected_deltas.items():
        res = resample_candles(df_1m, target_timeframe=tf, calendar=cal)
        assert not res.empty, f"Resampling produced empty df for {tf}"
        assert "ts_close" in res.columns, f"Missing ts_close column for {tf}"

        for _, row in res.iterrows():
            if tf == "1d":
                expected_end = get_daily_bucket_end(row["ts"], cal)
            else:
                expected_end = row["ts"] + delta

            assert row["ts_close"] == expected_end, (
                f"Mismatch on {tf}: bar starting at {row['ts']} had ts_close={row['ts_close']}, "
                f"expected bucket end {expected_end}"
            )


def test_forex_calendar_dst_summer_and_winter_boundaries():
    """Forex weekly_windows defined in America/New_York (Sun 17:00 to Fri 17:00)

    properly convert to UTC per date, handling DST:
      - Summer (EDT = UTC-4): closes Friday 21:00 UTC, reopens Sunday 21:00 UTC.
      - Winter (EST = UTC-5): closes Friday 22:00 UTC, reopens Sunday 22:00 UTC.
    """
    from core.config import CalendarConfig, CalendarWindowConfig
    from data.calendar import is_market_open
    from data.candle_builder import detect_gaps

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

    # 1. Summer week (July 2021) - EDT is UTC-4:
    # Fri 2021-07-02 17:00 EDT = 21:00 UTC
    fri_summer_open = datetime(2021, 7, 2, 20, 59, 0, tzinfo=timezone.utc)
    fri_summer_close = datetime(2021, 7, 2, 21, 0, 0, tzinfo=timezone.utc)
    fri_summer_closed = datetime(2021, 7, 2, 21, 1, 0, tzinfo=timezone.utc)

    assert is_market_open(fri_summer_open, forex_cal) is True
    assert is_market_open(fri_summer_close, forex_cal) is False
    assert is_market_open(fri_summer_closed, forex_cal) is False

    # Sun 2021-07-04 17:00 EDT = 21:00 UTC
    sun_summer_closed = datetime(2021, 7, 4, 20, 59, 0, tzinfo=timezone.utc)
    sun_summer_open = datetime(2021, 7, 4, 21, 0, 0, tzinfo=timezone.utc)
    sun_summer_opened = datetime(2021, 7, 4, 21, 1, 0, tzinfo=timezone.utc)

    assert is_market_open(sun_summer_closed, forex_cal) is False
    assert is_market_open(sun_summer_open, forex_cal) is True
    assert is_market_open(sun_summer_opened, forex_cal) is True

    # 2. Winter week (January 2021) - EST is UTC-5:
    # Fri 2021-01-08 17:00 EST = 22:00 UTC
    fri_winter_open = datetime(2021, 1, 8, 21, 59, 0, tzinfo=timezone.utc)
    fri_winter_close = datetime(2021, 1, 8, 22, 0, 0, tzinfo=timezone.utc)
    fri_winter_closed = datetime(2021, 1, 8, 22, 1, 0, tzinfo=timezone.utc)

    assert is_market_open(fri_winter_open, forex_cal) is True
    assert is_market_open(fri_winter_close, forex_cal) is False
    assert is_market_open(fri_winter_closed, forex_cal) is False

    # Sun 2021-01-10 17:00 EST = 22:00 UTC
    sun_winter_closed = datetime(2021, 1, 10, 21, 59, 0, tzinfo=timezone.utc)
    sun_winter_open = datetime(2021, 1, 10, 22, 0, 0, tzinfo=timezone.utc)
    sun_winter_opened = datetime(2021, 1, 10, 22, 1, 0, tzinfo=timezone.utc)

    assert is_market_open(sun_winter_closed, forex_cal) is False
    assert is_market_open(sun_winter_open, forex_cal) is True
    assert is_market_open(sun_winter_opened, forex_cal) is True

    # 3. Gap detector over summer and winter weekends: 0 unexpected gaps
    df_summer = pd.DataFrame([
        {"ts": fri_summer_open, "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.1, "volume": 1},
        {"ts": sun_summer_open, "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.1, "volume": 1},
    ])
    assert detect_gaps(df_summer, timeframe="1m", calendar=forex_cal) == []

    df_winter = pd.DataFrame([
        {"ts": fri_winter_open, "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.1, "volume": 1},
        {"ts": sun_winter_open, "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.1, "volume": 1},
    ])
    assert detect_gaps(df_winter, timeframe="1m", calendar=forex_cal) == []




