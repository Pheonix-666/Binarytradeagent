"""Technical indicators — EMA, ATR, ADX, and true range calculations.

Strict Rule:
    No look-ahead. The value at index i depends strictly on data with index <= i.
"""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA).

    Uses alpha = 2 / (period + 1), equivalent to span=period, adjust=False.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")
    if series.empty:
        return pd.Series(dtype=float)
    return series.ewm(span=period, adjust=False).mean()


def compute_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Compute True Range (TR) for each candle.

    TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    For t=0, TR_0 = H_0 - L_0.
    """
    if len(high) == 0:
        return pd.Series(dtype=float)

    prev_close = close.shift(1)
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()

    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # First row has no previous close, TR is simply high - low
    tr.iloc[0] = hl.iloc[0]
    return tr


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR) using Wilder's smoothing.

    Expected DataFrame columns: 'high', 'low', 'close'
    """
    if period < 1:
        raise ValueError(f"ATR period must be >= 1, got {period}")
    if len(df) == 0:
        return pd.Series(dtype=float)

    tr = compute_true_range(df["high"], df["low"], df["close"])
    # Wilder's smoothing is equivalent to ewm with alpha=1/period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def compute_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Compute Average Directional Index (ADX) along with +DI and -DI.

    Expected DataFrame columns: 'high', 'low', 'close'
    Returns DataFrame with columns: ['adx', 'plus_di', 'minus_di']
    """
    if period < 1:
        raise ValueError(f"ADX period must be >= 1, got {period}")

    cols = ["adx", "plus_di", "minus_di"]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    # +DM: up_move if up_move > down_move and up_move > 0 else 0
    plus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    plus_dm[plus_mask] = up_move[plus_mask]

    # -DM: down_move if down_move > up_move and down_move > 0 else 0
    minus_dm = pd.Series(0.0, index=df.index)
    minus_mask = (down_move > up_move) & (down_move > 0)
    minus_dm[minus_mask] = down_move[minus_mask]

    # True range
    tr = compute_true_range(high, low, close)

    # Wilder's smoothing on TR, +DM, -DM
    alpha = 1.0 / period
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # Directional indicators (+DI, -DI)
    # Avoid zero division
    tr_safe = tr_smooth.replace(0.0, np.nan)
    plus_di = (100.0 * plus_dm_smooth / tr_safe).fillna(0.0)
    minus_di = (100.0 * minus_dm_smooth / tr_safe).fillna(0.0)

    # Directional Movement Index (DX)
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    di_sum_safe = di_sum.replace(0.0, np.nan)
    dx = (100.0 * di_diff / di_sum_safe).fillna(0.0)

    # ADX is Wilder's smoothed DX
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )
