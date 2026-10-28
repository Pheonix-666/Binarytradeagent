"""Tests for technical indicators and proof of no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from analysis.indicators import compute_adx, compute_atr, compute_ema, compute_true_range


def test_compute_ema_constant_series():
    """EMA of a constant series must equal the constant value."""
    series = pd.Series([100.0] * 50)
    ema = compute_ema(series, period=20)
    assert len(ema) == 50
    assert np.allclose(ema, 100.0)


def test_compute_ema_known_ramp():
    """EMA on a stepping series matches manual recursive formulation."""
    vals = [10.0, 12.0, 14.0, 16.0, 18.0]
    series = pd.Series(vals)
    period = 3
    alpha = 2.0 / (period + 1.0)  # 0.5
    expected = [10.0]
    for v in vals[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])

    ema = compute_ema(series, period=period)
    assert np.allclose(ema.values, expected)


def test_compute_true_range_synthetic():
    """True Range correctly handles intrabar range and gaps from prior close."""
    high = pd.Series([10.0, 15.0, 12.0])
    low = pd.Series([8.0, 11.0, 9.0])
    close = pd.Series([9.0, 14.0, 10.0])

    tr = compute_true_range(high, low, close)
    # Row 0: H - L = 10 - 8 = 2.0
    assert tr.iloc[0] == 2.0
    # Row 1: max(15-11=4, |15-9|=6, |11-9|=2) = 6.0 (gap up)
    assert tr.iloc[1] == 6.0
    # Row 2: max(12-9=3, |12-14|=2, |9-14|=5) = 5.0 (gap down)
    assert tr.iloc[2] == 5.0


def test_compute_atr_constant():
    """ATR of constant bar ranges equals that range."""
    df = pd.DataFrame({
        "high": [110.0] * 30,
        "low": [100.0] * 30,
        "close": [105.0] * 30,
    })
    atr = compute_atr(df, period=14)
    assert np.allclose(atr.values, 10.0)


def test_compute_adx_strong_trend():
    """ADX rises significantly in a strong sustained trend."""
    n = 60
    # Strong upward trend: prices steadily climbing by 1.5 per bar
    highs = [100.0 + i * 1.5 + 0.5 for i in range(n)]
    lows = [100.0 + i * 1.5 - 0.5 for i in range(n)]
    closes = [100.0 + i * 1.5 for i in range(n)]

    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    adx_df = compute_adx(df, period=14)

    assert "adx" in adx_df.columns
    assert "plus_di" in adx_df.columns
    assert "minus_di" in adx_df.columns

    # In a pure uptrend, +DI should far exceed -DI and ADX should be high (> 40)
    assert adx_df["plus_di"].iloc[-1] > adx_df["minus_di"].iloc[-1]
    assert adx_df["adx"].iloc[-1] > 40.0


def test_indicators_no_look_ahead():
    """CRITICAL TEST: Appending future bars must NEVER alter historical indicator values at time <= t."""
    np.random.seed(42)
    n_past = 50
    n_future = 25

    # Generate 50 past bars
    prices_past = 100.0 + np.cumsum(np.random.randn(n_past) * 0.5)
    df_past = pd.DataFrame({
        "high": prices_past + 0.5,
        "low": prices_past - 0.5,
        "close": prices_past,
    })

    # Generate 25 future bars
    prices_future = prices_past[-1] + np.cumsum(np.random.randn(n_future) * 0.5)
    df_future = pd.DataFrame({
        "high": prices_future + 0.5,
        "low": prices_future - 0.5,
        "close": prices_future,
    })
    df_extended = pd.concat([df_past, df_future], ignore_index=True)

    # Compute indicators on past only
    ema_past = compute_ema(df_past["close"], period=20)
    atr_past = compute_atr(df_past, period=14)
    adx_past = compute_adx(df_past, period=14)

    # Compute indicators on extended series
    ema_extended = compute_ema(df_extended["close"], period=20)
    atr_extended = compute_atr(df_extended, period=14)
    adx_extended = compute_adx(df_extended, period=14)

    # Values for indices 0..n_past-1 must be exactly identical
    np.testing.assert_allclose(
        ema_past.values,
        ema_extended.iloc[:n_past].values,
        err_msg="EMA at t changed when future data was added!",
    )
    np.testing.assert_allclose(
        atr_past.values,
        atr_extended.iloc[:n_past].values,
        err_msg="ATR at t changed when future data was added!",
    )
    np.testing.assert_allclose(
        adx_past["adx"].values,
        adx_extended["adx"].iloc[:n_past].values,
        err_msg="ADX at t changed when future data was added!",
    )

