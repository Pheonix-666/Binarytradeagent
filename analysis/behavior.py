"""Behavior analysis — volatility regimes and confirmation candle checks."""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

from analysis.indicators import compute_atr
from core.models import TradeDirection


def compute_volatility_percentile(
    data: Union[float, pd.DataFrame],
    historical_atrs: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    atr_period: int = 14,
    lookback_bars: Optional[int] = None,
    lookback_days: Optional[int] = None,
    timeframe_minutes: int = 15,
) -> float:
    """Compute percentile rank (0-100) of current volatility against history.

    Supports two calling patterns:
    1. Direct ATR values:
       compute_volatility_percentile(current_atr, historical_atrs)
    2. DataFrame of candles:
       compute_volatility_percentile(df_candles, atr_period=14, lookback_days=30, timeframe_minutes=15)

    Percentile represents the percentage of historical ATR values strictly below current_atr.
    Returns 50.0 (median) if data is empty or has too few values (< 5).
    """
    if isinstance(data, (int, float)):
        current_atr = float(data)
        if current_atr <= 0:
            return 0.0

        if historical_atrs is None:
            return 50.0

        if isinstance(historical_atrs, pd.Series):
            vals = historical_atrs.dropna().values
        else:
            vals = np.array(historical_atrs, dtype=float)
            vals = vals[~np.isnan(vals)]

        if len(vals) < 5:
            return 50.0

        count_below = np.sum(vals < current_atr)
        pct = (count_below / len(vals)) * 100.0
        return float(np.clip(pct, 0.0, 100.0))

    elif isinstance(data, pd.DataFrame):
        df_candles = data
        if len(df_candles) < max(20, atr_period + 5):
            return 50.0

        atr_series = compute_atr(df_candles, period=atr_period).dropna()
        if len(atr_series) < 5:
            return 50.0

        if lookback_bars is None:
            if lookback_days is not None:
                bars_per_day = max(1, 1440 // max(1, timeframe_minutes))
                lookback_bars = int(lookback_days * bars_per_day)
            else:
                lookback_bars = 200

        recent_atr = atr_series.tail(lookback_bars).values
        cur_atr = float(recent_atr[-1])
        hist_vals = recent_atr[:-1]
        if len(hist_vals) < 5:
            return 50.0

        return compute_volatility_percentile(cur_atr, hist_vals)

    return 50.0


def get_confirmation_candle_direction(
    candle: Union[pd.Series, dict],
) -> Optional[TradeDirection]:
    """Check direction of a confirmation candle (e.g. 1m or 15m closed candle).

    - close > open: CALL (bullish confirmation)
    - close < open: PUT (bearish confirmation)
    - close == open: None (doji / neutral)
    """
    open_val = float(candle["open"])
    close_val = float(candle["close"])

    if close_val > open_val:
        return TradeDirection.CALL
    elif close_val < open_val:
        return TradeDirection.PUT
    return None

