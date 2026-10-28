"""Reference indicator check comparing our EMA, ATR, and ADX against the 'ta' library.

Dev-only test verifying calculation accuracy after warm-up against a pure-Python reference.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.indicators import compute_adx, compute_atr, compute_ema

try:
    from ta.trend import ADXIndicator, EMAIndicator
    from ta.volatility import AverageTrueRange
    HAS_TA = True
except ImportError:
    HAS_TA = False


@pytest.mark.skipif(not HAS_TA, reason="'ta' library not installed")
def test_reference_ema():
    """Verify our EMA matches the 'ta' library reference calculation."""
    np.random.seed(101)
    n = 200
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.7)
    series = pd.Series(prices)

    our_ema = compute_ema(series, period=20)
    ta_ema = EMAIndicator(close=series, window=20).ema_indicator()

    # After warm-up period (first 50 bars), values should be virtually identical
    np.testing.assert_allclose(our_ema.iloc[50:].values, ta_ema.iloc[50:].values, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_TA, reason="'ta' library not installed")
def test_reference_atr():
    """Verify our ATR matches the 'ta' library reference calculation after warm-up."""
    np.random.seed(202)
    n = 250
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "high": base + np.abs(np.random.randn(n) * 0.4) + 0.2,
        "low": base - np.abs(np.random.randn(n) * 0.4) - 0.2,
        "close": base,
    })

    our_atr = compute_atr(df, period=14)
    ta_atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()

    # After warm-up period (first 100 bars), ATR matches within 0.001
    np.testing.assert_allclose(our_atr.iloc[100:].values, ta_atr.iloc[100:].values, atol=1e-3)


@pytest.mark.skipif(not HAS_TA, reason="'ta' library not installed")
def test_reference_adx():
    """Verify our ADX matches the 'ta' library reference calculation after warm-up."""
    np.random.seed(303)
    n = 250
    # Create trending series
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.3 + 0.2)
    df = pd.DataFrame({
        "high": base + np.abs(np.random.randn(n) * 0.3) + 0.2,
        "low": base - np.abs(np.random.randn(n) * 0.3) - 0.2,
        "close": base,
    })

    our_adx = compute_adx(df, period=14)["adx"]
    ta_adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()

    # After warm-up (first 100 bars), ADX matches within 0.5 points
    np.testing.assert_allclose(our_adx.iloc[100:].values, ta_adx.iloc[100:].values, atol=0.5)
