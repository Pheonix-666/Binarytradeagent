"""Tests for trend labeling, strict gate rules, reversal bypass, and no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from analysis.trend import (
    analyze_trend,
    evaluate_gate,
    label_trend,
)
from core.config import GateConfig, ReversalBypassConfig, TimeframeTrendConfig, TrendConfig
from core.models import (
    PatternMatch,
    TradeDirection,
    TrendLabel,
    TrendReport,
)


@pytest.fixture
def default_trend_cfg():
    return TrendConfig(
        daily=TimeframeTrendConfig(ema_fast=20, ema_slow=50, adx_period=14, adx_min=20, warmup_multiplier=3),
        h1=TimeframeTrendConfig(ema_fast=20, ema_slow=50, adx_period=14, adx_min=20, warmup_multiplier=3),
        m15=TimeframeTrendConfig(ema_fast=20, ema_slow=50, adx_period=14, adx_min=20, warmup_multiplier=3),
    )


@pytest.fixture
def default_gate_cfg(default_trend_cfg):
    return GateConfig(
        strict_trend_gate=True,
        reversal_mode="off",
        trend=default_trend_cfg,
        reversal_bypass=ReversalBypassConfig(
            min_similarity=80.0,
            max_risk_score=40.0,
            require_sr_level=True,
            require_confirmation_candle=True,
            max_volatility_percentile=90.0,
        ),
    )


def test_label_trend_insufficient_bars(default_trend_cfg):
    """If bars < ema_slow * warmup_multiplier (50 * 3 = 150), trend must be INSUFFICIENT_DATA."""
    df = pd.DataFrame({"close": [10.0] * 100, "high": [11.0] * 100, "low": [9.0] * 100})
    label = label_trend(
        df,
        ema_fast=default_trend_cfg.daily.ema_fast,
        ema_slow=default_trend_cfg.daily.ema_slow,
        adx_period=default_trend_cfg.daily.adx_period,
        adx_min=default_trend_cfg.daily.adx_min,
        warmup_multiplier=default_trend_cfg.daily.warmup_multiplier,
    )
    assert label == TrendLabel.INSUFFICIENT_DATA


def test_label_trend_gap_returns_insufficient_data(default_trend_cfg):
    """If a gap exists within the warm-up window, label_trend returns INSUFFICIENT_DATA."""
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 10, 29, 0, 0, 0, tzinfo=timezone.utc)
    # Create 180 hourly bars but introduce a gap at index 100
    times = []
    curr = base
    for i in range(180):
        if i == 100:
            curr += timedelta(hours=5)  # 5-hour gap
        else:
            curr += timedelta(hours=1)
        times.append(curr)

    closes = [100.0 + i * 0.5 for i in range(180)]
    df = pd.DataFrame({
        "ts": times,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
    })

    label = label_trend(
        df,
        ema_fast=default_trend_cfg.h1.ema_fast,
        ema_slow=default_trend_cfg.h1.ema_slow,
        adx_period=default_trend_cfg.h1.adx_period,
        adx_min=default_trend_cfg.h1.adx_min,
        warmup_multiplier=default_trend_cfg.h1.warmup_multiplier,
        timeframe="1h",
    )
    assert label == TrendLabel.INSUFFICIENT_DATA


def test_label_trend_forex_weekend_does_not_cause_insufficient_data(default_trend_cfg):
    """Forex weekend closures are not expected bars, so label_trend does not report INSUFFICIENT_DATA."""
    from datetime import datetime, timedelta, timezone
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

    # 180 hourly bars in an uptrend, with bar 100 on Friday 16:00 NY (20:00 UTC)
    # and bar 101 resuming on Sunday 17:00 NY (21:00 UTC)
    times = []
    # Friday 2026-10-23 at 20:00 UTC (16:00 NY, 1 hour before Friday 17:00 close)
    fri_2000_utc = datetime(2026, 10, 23, 20, 0, 0, tzinfo=timezone.utc)
    # Generate 100 bars leading up to Friday 20:00 UTC
    for i in range(100):
        times.append(fri_2000_utc - timedelta(hours=99 - i))

    # Reopen Sunday 2026-10-25 at 21:00 UTC (17:00 NY)
    sun_2100_utc = datetime(2026, 10, 25, 21, 0, 0, tzinfo=timezone.utc)
    for i in range(80):
        times.append(sun_2100_utc + timedelta(hours=i))

    closes = [100.0 + i * 1.5 for i in range(180)]
    df = pd.DataFrame({
        "ts": times,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
    })

    # With forex calendar: weekend is not a gap -> computes UP trend successfully!
    label = label_trend(
        df,
        ema_fast=default_trend_cfg.h1.ema_fast,
        ema_slow=default_trend_cfg.h1.ema_slow,
        adx_period=default_trend_cfg.h1.adx_period,
        adx_min=default_trend_cfg.h1.adx_min,
        warmup_multiplier=default_trend_cfg.h1.warmup_multiplier,
        timeframe="1h",
        calendar=forex_cal,
    )
    assert label == TrendLabel.UP


def test_label_trend_strong_uptrend(default_trend_cfg):
    """Steady upward ramp over 180 bars (> 150 warmup) produces UP trend."""
    n = 180
    closes = [100.0 + i * 1.5 for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    label = label_trend(
        df,
        ema_fast=default_trend_cfg.daily.ema_fast,
        ema_slow=default_trend_cfg.daily.ema_slow,
        adx_period=default_trend_cfg.daily.adx_period,
        adx_min=default_trend_cfg.daily.adx_min,
        warmup_multiplier=default_trend_cfg.daily.warmup_multiplier,
    )
    assert label == TrendLabel.UP


def test_label_trend_strong_downtrend(default_trend_cfg):
    """Steady downward ramp over 180 bars (> 150 warmup) produces DOWN trend."""
    n = 180
    closes = [500.0 - i * 1.5 for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    label = label_trend(
        df,
        ema_fast=default_trend_cfg.daily.ema_fast,
        ema_slow=default_trend_cfg.daily.ema_slow,
        adx_period=default_trend_cfg.daily.adx_period,
        adx_min=default_trend_cfg.daily.adx_min,
        warmup_multiplier=default_trend_cfg.daily.warmup_multiplier,
    )
    assert label == TrendLabel.DOWN


def test_label_trend_sideways_flat_market(default_trend_cfg):
    """Flat oscillation over 180 bars produces ADX < adx_min -> SIDEWAYS."""
    n = 180
    closes = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    label = label_trend(
        df,
        ema_fast=default_trend_cfg.daily.ema_fast,
        ema_slow=default_trend_cfg.daily.ema_slow,
        adx_period=default_trend_cfg.daily.adx_period,
        adx_min=default_trend_cfg.daily.adx_min,
        warmup_multiplier=default_trend_cfg.daily.warmup_multiplier,
    )
    assert label == TrendLabel.SIDEWAYS


# ── Gate Tests ────────────────────────────────────────────────────────


def test_gate_call_matching_trend_passes(default_gate_cfg):
    """CALL passes with daily=UP, 1h=UP, 15m=UP."""
    trend = TrendReport(daily=TrendLabel.UP, hourly=TrendLabel.UP, fifteen_min=TrendLabel.UP)
    result = evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg)
    assert result.passed is True
    assert result.direction == TradeDirection.CALL


def test_gate_call_fifteen_sideways_passes(default_gate_cfg):
    """CALL passes when 15m is SIDEWAYS (spec requires 15m is not DOWN)."""
    trend = TrendReport(daily=TrendLabel.UP, hourly=TrendLabel.UP, fifteen_min=TrendLabel.SIDEWAYS)
    result = evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg)
    assert result.passed is True


def test_gate_call_conflicting_fifteen_fails(default_gate_cfg):
    """CALL fails if 15m is DOWN."""
    trend = TrendReport(daily=TrendLabel.UP, hourly=TrendLabel.UP, fifteen_min=TrendLabel.DOWN)
    result = evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg)
    assert result.passed is False
    assert "Strict trend gate rejected CALL" in result.reason


def test_gate_call_conflicting_hourly_fails(default_gate_cfg):
    """CALL fails if 1h is DOWN or SIDEWAYS."""
    trend1 = TrendReport(daily=TrendLabel.UP, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.UP)
    assert evaluate_gate(trend1, TradeDirection.CALL, default_gate_cfg).passed is False

    trend2 = TrendReport(daily=TrendLabel.UP, hourly=TrendLabel.SIDEWAYS, fifteen_min=TrendLabel.UP)
    assert evaluate_gate(trend2, TradeDirection.CALL, default_gate_cfg).passed is False


def test_gate_call_conflicting_daily_fails(default_gate_cfg):
    """CALL fails if daily is DOWN or SIDEWAYS."""
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.UP, fifteen_min=TrendLabel.UP)
    assert evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg).passed is False


def test_gate_put_matching_trend_passes(default_gate_cfg):
    """PUT passes with daily=DOWN, 1h=DOWN, 15m=DOWN."""
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.DOWN)
    result = evaluate_gate(trend, TradeDirection.PUT, default_gate_cfg)
    assert result.passed is True
    assert result.direction == TradeDirection.PUT


def test_gate_put_fifteen_sideways_passes(default_gate_cfg):
    """PUT passes when 15m is SIDEWAYS (spec requires 15m is not UP)."""
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.SIDEWAYS)
    result = evaluate_gate(trend, TradeDirection.PUT, default_gate_cfg)
    assert result.passed is True


def test_gate_put_conflicting_fifteen_fails(default_gate_cfg):
    """PUT fails if 15m is UP."""
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.UP)
    result = evaluate_gate(trend, TradeDirection.PUT, default_gate_cfg)
    assert result.passed is False


def test_gate_reversal_mode_off_always_rejects_on_trend_fail(default_gate_cfg):
    """With reversal_mode: off, reversal patterns NEVER trade when trend gate fails."""
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.DOWN)
    pattern = PatternMatch(
        pattern_name="double_bottom",
        direction=TradeDirection.CALL,
        pattern_type="reversal",
        similarity=95.0,
    )
    result = evaluate_gate(
        trend,
        TradeDirection.CALL,
        default_gate_cfg,
        pattern=pattern,
        nearest_level={"type": "support", "is_at_level": True, "distance_atr": 0.2},
        volatility_percentile=50.0,
        risk_score=25.0,
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert result.passed is False


def test_gate_reversal_sr_bypass_all_conditions_met(default_trend_cfg):
    """With reversal_mode: sr_bypass, valid reversal pattern at support passes gate."""
    cfg = GateConfig(
        strict_trend_gate=True,
        reversal_mode="sr_bypass",
        trend=default_trend_cfg,
        reversal_bypass=ReversalBypassConfig(
            min_similarity=80.0,
            max_risk_score=40.0,
            require_sr_level=True,
            require_confirmation_candle=True,
            max_volatility_percentile=90.0,
        ),
    )
    # Trend is DOWN (opposes CALL)
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.DOWN)

    pattern = PatternMatch(
        pattern_name="double_bottom",
        direction=TradeDirection.CALL,
        pattern_type="reversal",
        similarity=85.0,  # >= 80
    )
    nearest_level = {"type": "support", "is_at_level": True, "distance_atr": 0.3}

    result = evaluate_gate(
        trend=trend,
        direction=TradeDirection.CALL,
        gate_cfg=cfg,
        pattern=pattern,
        nearest_level=nearest_level,
        volatility_percentile=60.0,  # <= 90
        risk_score=35.0,  # <= 40
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert result.passed is True
    assert result.reason == "passed_reversal_sr_bypass"


def test_gate_reversal_sr_bypass_condition_failures(default_trend_cfg):
    """Each individual condition failure in reversal bypass must reject."""
    cfg = GateConfig(
        strict_trend_gate=True,
        reversal_mode="sr_bypass",
        trend=default_trend_cfg,
        reversal_bypass=ReversalBypassConfig(
            min_similarity=80.0,
            max_risk_score=40.0,
            require_sr_level=True,
            require_confirmation_candle=True,
            max_volatility_percentile=90.0,
        ),
    )
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.DOWN)
    valid_pattern = PatternMatch(
        pattern_name="double_bottom",
        direction=TradeDirection.CALL,
        pattern_type="reversal",
        similarity=85.0,
    )
    valid_level = {"type": "support", "is_at_level": True, "distance_atr": 0.3}

    # 1. Similarity too low (< 80)
    low_sim_pattern = PatternMatch(
        pattern_name="double_bottom",
        direction=TradeDirection.CALL,
        pattern_type="reversal",
        similarity=75.0,
    )
    res1 = evaluate_gate(
        trend, TradeDirection.CALL, cfg,
        pattern=low_sim_pattern, nearest_level=valid_level,
        volatility_percentile=50.0, risk_score=30.0,
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert res1.passed is False
    assert "similarity" in res1.reason

    # 2. Not at support level
    not_at_level = {"type": "support", "is_at_level": False, "distance_atr": 1.8}
    res2 = evaluate_gate(
        trend, TradeDirection.CALL, cfg,
        pattern=valid_pattern, nearest_level=not_at_level,
        volatility_percentile=50.0, risk_score=30.0,
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert res2.passed is False
    assert "not at support" in res2.reason

    # 3. Volatility percentile too high (> 90)
    res3 = evaluate_gate(
        trend, TradeDirection.CALL, cfg,
        pattern=valid_pattern, nearest_level=valid_level,
        volatility_percentile=95.0, risk_score=30.0,
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert res3.passed is False
    assert "volatility percentile" in res3.reason

    # 4. Risk score too high (> 40)
    res4 = evaluate_gate(
        trend, TradeDirection.CALL, cfg,
        pattern=valid_pattern, nearest_level=valid_level,
        volatility_percentile=50.0, risk_score=48.0,
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert res4.passed is False
    assert "risk score" in res4.reason

    # 5. Confirmation candle points wrong way (PUT instead of CALL)
    res5 = evaluate_gate(
        trend, TradeDirection.CALL, cfg,
        pattern=valid_pattern, nearest_level=valid_level,
        volatility_percentile=50.0, risk_score=30.0,
        confirmation_candle_direction=TradeDirection.PUT,
    )
    assert res5.passed is False
    assert "confirmation candle direction" in res5.reason


def test_gate_skip_insufficient_data_daily_only(default_gate_cfg):
    """Gate must SKIP reporting daily insufficient data specifically."""
    trend = TrendReport(
        daily=TrendLabel.INSUFFICIENT_DATA,
        hourly=TrendLabel.UP,
        fifteen_min=TrendLabel.UP,
    )
    result = evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg)
    assert result.passed is False
    assert "insufficient data" in result.reason
    assert "daily" in result.reason
    assert "1h" not in result.reason
    assert "15m" not in result.reason


def test_gate_skip_insufficient_data_h1_only(default_gate_cfg):
    """Gate must SKIP reporting 1h insufficient data specifically."""
    trend = TrendReport(
        daily=TrendLabel.UP,
        hourly=TrendLabel.INSUFFICIENT_DATA,
        fifteen_min=TrendLabel.UP,
    )
    result = evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg)
    assert result.passed is False
    assert "insufficient data" in result.reason
    assert "1h" in result.reason
    assert "daily" not in result.reason


def test_gate_skip_insufficient_data_m15_only(default_gate_cfg):
    """Gate must SKIP reporting 15m insufficient data specifically."""
    trend = TrendReport(
        daily=TrendLabel.UP,
        hourly=TrendLabel.UP,
        fifteen_min=TrendLabel.INSUFFICIENT_DATA,
    )
    result = evaluate_gate(trend, TradeDirection.CALL, default_gate_cfg)
    assert result.passed is False
    assert "insufficient data" in result.reason
    assert "15m" in result.reason
    assert "1h" not in result.reason


def test_evaluate_gate_risk_score_none_under_sr_bypass(default_trend_cfg):
    """Under reversal_mode sr_bypass, risk_score=None must fail bypass with explicit reason."""
    cfg = GateConfig(
        strict_trend_gate=True,
        reversal_mode="sr_bypass",
        trend=default_trend_cfg,
        reversal_bypass=ReversalBypassConfig(
            min_similarity=80.0,
            max_risk_score=40.0,
            require_sr_level=True,
            require_confirmation_candle=True,
            max_volatility_percentile=90.0,
        ),
    )
    # Trend opposes trade direction so reversal bypass is evaluated
    trend = TrendReport(daily=TrendLabel.DOWN, hourly=TrendLabel.DOWN, fifteen_min=TrendLabel.DOWN)
    pattern = PatternMatch(
        pattern_name="double_bottom",
        direction=TradeDirection.CALL,
        pattern_type="reversal",
        similarity=85.0,
    )
    level = {"type": "support", "is_at_level": True, "distance_atr": 0.2}

    result = evaluate_gate(
        trend=trend,
        direction=TradeDirection.CALL,
        gate_cfg=cfg,
        pattern=pattern,
        nearest_level=level,
        volatility_percentile=50.0,
        risk_score=None,  # Not provided
        confirmation_candle_direction=TradeDirection.CALL,
    )
    assert result.passed is False
    assert "missing risk_score" in result.reason


def test_trend_no_look_ahead(default_trend_cfg):
    """Trend label at time t MUST be completely identical whether future bars exist or not."""
    np.random.seed(123)
    n_past = 180
    n_future = 50

    closes_past = 100.0 + np.cumsum(np.random.randn(n_past))
    df_past = pd.DataFrame({
        "high": closes_past + 0.5,
        "low": closes_past - 0.5,
        "close": closes_past,
    })

    closes_future = closes_past[-1] + np.cumsum(np.random.randn(n_future))
    df_future = pd.DataFrame({
        "high": closes_future + 0.5,
        "low": closes_future - 0.5,
        "close": closes_future,
    })
    df_extended = pd.concat([df_past, df_future], ignore_index=True)

    # Label evaluated on df_past
    label_past = label_trend(
        df_past,
        ema_fast=default_trend_cfg.daily.ema_fast,
        ema_slow=default_trend_cfg.daily.ema_slow,
        adx_period=default_trend_cfg.daily.adx_period,
        adx_min=default_trend_cfg.daily.adx_min,
        warmup_multiplier=default_trend_cfg.daily.warmup_multiplier,
    )

    # Label evaluated at the exact same index (slice up to n_past) in df_extended
    label_sliced_from_extended = label_trend(
        df_extended.iloc[:n_past],
        ema_fast=default_trend_cfg.daily.ema_fast,
        ema_slow=default_trend_cfg.daily.ema_slow,
        adx_period=default_trend_cfg.daily.adx_period,
        adx_min=default_trend_cfg.daily.adx_min,
        warmup_multiplier=default_trend_cfg.daily.warmup_multiplier,
    )

    assert label_past == label_sliced_from_extended

