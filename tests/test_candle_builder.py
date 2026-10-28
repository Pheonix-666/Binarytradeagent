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
    assert list(result.columns) == ["ts", "open", "high", "low", "close", "volume"]


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

