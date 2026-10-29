"""Tests for behavior module: volatility percentiles and confirmation candle directions."""

import numpy as np
import pandas as pd
import pytest

from analysis.behavior import (
    compute_volatility_percentile,
    get_confirmation_candle_direction,
)
from core.models import TradeDirection


def test_compute_volatility_percentile_known_ranks():
    """Verify percentile calculation on known array."""
    # 10 values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    hist = list(range(10, 110, 10))

    # ATR = 5 is below all -> 0%
    assert compute_volatility_percentile(5.0, hist) == 0.0

    # ATR = 55 is greater than 5 values (10..50) out of 10 -> 50%
    assert compute_volatility_percentile(55.0, hist) == 50.0

    # ATR = 105 is greater than all -> 100%
    assert compute_volatility_percentile(105.0, hist) == 100.0


def test_compute_volatility_percentile_empty_fallback():
    """Empty or insufficient historical data returns 50.0 default."""
    assert compute_volatility_percentile(10.0, []) == 50.0
    assert compute_volatility_percentile(10.0, [5.0, 6.0]) == 50.0


def test_compute_volatility_percentile_dataframe():
    """Test unified volatility percentile calculation directly from candle DataFrames."""
    # 1. Insufficient candles (< 20 bars) returns 50.0 default
    df_short = pd.DataFrame([
        {"ts": pd.Timestamp("2026-10-29") + pd.Timedelta(minutes=15 * i),
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(15)
    ])
    assert compute_volatility_percentile(df_short, atr_period=14) == 50.0

    # 2. DataFrame with normal candles followed by high volatility candle
    base_ts = pd.Timestamp("2026-10-29", tz="UTC")
    rows = []
    # 50 normal bars with ATR ~ 1.0 (range 99.5 to 100.5)
    for i in range(50):
        rows.append({
            "ts": base_ts + pd.Timedelta(minutes=15 * i),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
        })
    # 51st bar: huge surge with range 90.0 to 110.0 (high ATR)
    rows.append({
        "ts": base_ts + pd.Timedelta(minutes=15 * 50),
        "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
    })
    df_surge = pd.DataFrame(rows)
    # The surge ATR should be strictly above all historical ATRs -> 100%
    pct_surge = compute_volatility_percentile(
        df_surge,
        atr_period=14,
        lookback_days=1,
        timeframe_minutes=15,
    )
    assert pct_surge == 100.0


def test_get_confirmation_candle_direction():
    """Bullish candle -> CALL, Bearish -> PUT, Doji -> None."""
    bullish = {"open": 1.1000, "close": 1.1050}
    assert get_confirmation_candle_direction(bullish) == TradeDirection.CALL

    bearish = {"open": 1.1050, "close": 1.1000}
    assert get_confirmation_candle_direction(bearish) == TradeDirection.PUT

    doji = {"open": 1.1000, "close": 1.1000}
    assert get_confirmation_candle_direction(doji) is None

