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


def test_get_confirmation_candle_direction():
    """Bullish candle -> CALL, Bearish -> PUT, Doji -> None."""
    bullish = {"open": 1.1000, "close": 1.1050}
    assert get_confirmation_candle_direction(bullish) == TradeDirection.CALL

    bearish = {"open": 1.1050, "close": 1.1000}
    assert get_confirmation_candle_direction(bearish) == TradeDirection.PUT

    doji = {"open": 1.1000, "close": 1.1000}
    assert get_confirmation_candle_direction(doji) is None

