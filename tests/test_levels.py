"""Tests for swing support/resistance levels and no look-ahead verification."""

import pandas as pd
import pytest

from analysis.levels import (
    find_swing_levels,
    get_nearest_level,
    get_nearest_opposing_level,
)
from core.models import TradeDirection


def test_find_swing_levels_synthetic():
    """Verify swing high and swing low detection with left=3, right=2."""
    # Synthetic chart:
    # 0..2 flat at 10.0
    # 3 peaks at 15.0 (swing high)
    # 4..5 drops to 10.0
    # 6 bottoms at 5.0 (swing low)
    # 7..8 rises to 10.0
    highs = [10.0, 10.0, 10.0, 15.0, 10.0, 10.0, 7.0, 10.0, 10.0]
    lows = [9.0, 9.0, 9.0, 13.0, 9.0, 8.0, 5.0, 9.0, 9.0]
    closes = [9.5, 9.5, 9.5, 14.0, 9.5, 9.0, 6.0, 9.5, 9.5]

    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    levels = find_swing_levels(df, left_bars=2, right_bars=2)
    # Should find resistance at 15.0 and support at 5.0
    resistances = [l for l in levels if l.level_type == "resistance"]
    supports = [l for l in levels if l.level_type == "support"]

    assert len(resistances) >= 1
    assert any(r.price == 15.0 for r in resistances)

    assert len(supports) >= 1
    assert any(s.price == 5.0 for s in supports)


def test_levels_no_look_ahead():
    """Swings confirmed at time t must not depend on candles at t+1.."""
    # 20 bars
    highs = [10.0] * 5 + [15.0] + [10.0] * 5 + [5.0] + [10.0] * 8
    lows = [9.0] * 5 + [13.0] + [9.0] * 5 + [4.0] + [9.0] * 8
    closes = [9.5] * len(highs)

    df_past = pd.DataFrame({"high": highs[:15], "low": lows[:15], "close": closes[:15]})
    df_extended = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    levels_past = find_swing_levels(df_past, left_bars=3, right_bars=2)

    # In df_extended, consider only swings confirmed up to index 14
    levels_extended = find_swing_levels(df_extended, left_bars=3, right_bars=2)
    levels_extended_past_only = [l for l in levels_extended if l.candle_index <= 15 - 1 - 2]

    past_prices = [(l.level_type, l.price) for l in levels_past]
    ext_prices = [(l.level_type, l.price) for l in levels_extended_past_only]

    assert past_prices == ext_prices


def test_nearest_level_and_opposing():
    """Verify distance calculations and opposing levels."""
    levels = find_swing_levels(
        pd.DataFrame({
            "high": [10.0, 10.0, 10.0, 15.0, 10.0, 10.0, 7.0, 10.0, 10.0],
            "low": [9.0, 9.0, 9.0, 13.0, 9.0, 8.0, 5.0, 9.0, 9.0],
            "close": [9.5] * 9,
        }),
        left_bars=2,
        right_bars=2,
    )

    current_price = 14.8  # Near resistance at 15.0
    atr = 1.0

    nearest_res = get_nearest_level(current_price, levels, target_type="resistance", atr=atr)
    assert nearest_res["type"] == "resistance"
    assert nearest_res["price"] == 15.0
    assert nearest_res["distance_atr"] == 0.2
    assert nearest_res["is_at_level"] is True

    # For a CALL trade, opposing level is resistance above price
    opposing_call = get_nearest_opposing_level(current_price, TradeDirection.CALL, levels, atr=atr)
    assert opposing_call["type"] == "resistance"
    assert opposing_call["price"] == 15.0

    # For a PUT trade, opposing level is support below price (support at 5.0)
    opposing_put = get_nearest_opposing_level(current_price, TradeDirection.PUT, levels, atr=atr)
    assert opposing_put["type"] == "support"
    assert opposing_put["price"] == 5.0
