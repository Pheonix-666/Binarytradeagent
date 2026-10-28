"""Behavior analysis — volatility regimes and confirmation candle checks."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd

from core.models import TradeDirection


def compute_volatility_percentile(
    current_atr: float,
    historical_atrs: Union[pd.Series, np.ndarray, list[float]],
) -> float:
    """Compute percentile rank (0-100) of current_atr against historical_atrs.

    Percentile represents the percentage of historical ATR values strictly below current_atr.
    Returns 50.0 if historical_atrs is empty or has too few values.
    """
    if current_atr <= 0:
        return 0.0

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
