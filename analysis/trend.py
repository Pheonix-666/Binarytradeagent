"""Trend analyzer and strict gate with reversal exception.

Sections 6.2 & 6.3:
    - Multi-timeframe trend labeling (daily, 1h, 15m).
    - Strict gate (CALL requires daily=up, 1h=up, 15m!=down; PUT inverse).
    - Reversal bypass exception (sr_bypass mode).
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from analysis.indicators import compute_adx, compute_ema
from core.config import GateConfig, TrendConfig
from core.logging_setup import get_logger
from core.models import GateResult, PatternMatch, TradeDirection, TrendLabel, TrendReport

logger = get_logger()


def label_trend(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    adx_period: int = 14,
    adx_min: float = 20.0,
    warmup_multiplier: int = 3,
) -> TrendLabel:
    """Classify the trend of a candle series as UP, DOWN, SIDEWAYS, or INSUFFICIENT_DATA.

    Rules (Section 6.2):
      - INSUFFICIENT_DATA: len(df) < ema_slow * warmup_multiplier (default 3x ema_slow)
        ensuring EMA and ADX indicators are fully settled and stable.
      - UP: close > EMA(slow) AND EMA(fast) > EMA(slow) AND ADX >= adx_min
      - DOWN: close < EMA(slow) AND EMA(fast) < EMA(slow) AND ADX >= adx_min
      - otherwise SIDEWAYS

    Strict no look-ahead: Uses only closed candles up to the latest row.
    """
    min_required_bars = ema_slow * warmup_multiplier
    if len(df) < min_required_bars:
        return TrendLabel.INSUFFICIENT_DATA

    fast_series = compute_ema(df["close"], period=ema_fast)
    slow_series = compute_ema(df["close"], period=ema_slow)
    adx_df = compute_adx(df, period=adx_period)

    close_val = float(df["close"].iloc[-1])
    fast_val = float(fast_series.iloc[-1])
    slow_val = float(slow_series.iloc[-1])
    adx_val = float(adx_df["adx"].iloc[-1])

    if close_val > slow_val and fast_val > slow_val and adx_val >= adx_min:
        return TrendLabel.UP
    elif close_val < slow_val and fast_val < slow_val and adx_val >= adx_min:
        return TrendLabel.DOWN
    else:
        return TrendLabel.SIDEWAYS


def analyze_trend(
    candles_by_tf: dict[str, pd.DataFrame],
    trend_cfg: TrendConfig,
) -> TrendReport:
    """Label trends for daily, 1h, and 15m timeframes."""
    daily_df = candles_by_tf.get("1d", candles_by_tf.get("daily", pd.DataFrame()))
    hourly_df = candles_by_tf.get("1h", candles_by_tf.get("hourly", pd.DataFrame()))
    fifteen_df = candles_by_tf.get("15m", candles_by_tf.get("fifteen_min", pd.DataFrame()))

    daily_label = label_trend(
        daily_df,
        ema_fast=trend_cfg.ema_fast,
        ema_slow=trend_cfg.ema_slow,
        adx_period=trend_cfg.adx_period,
        adx_min=trend_cfg.adx_min,
        warmup_multiplier=trend_cfg.warmup_multiplier,
    )
    hourly_label = label_trend(
        hourly_df,
        ema_fast=trend_cfg.ema_fast,
        ema_slow=trend_cfg.ema_slow,
        adx_period=trend_cfg.adx_period,
        adx_min=trend_cfg.adx_min,
        warmup_multiplier=trend_cfg.warmup_multiplier,
    )
    fifteen_label = label_trend(
        fifteen_df,
        ema_fast=trend_cfg.ema_fast,
        ema_slow=trend_cfg.ema_slow,
        adx_period=trend_cfg.adx_period,
        adx_min=trend_cfg.adx_min,
        warmup_multiplier=trend_cfg.warmup_multiplier,
    )

    return TrendReport(
        daily=daily_label,
        hourly=hourly_label,
        fifteen_min=fifteen_label,
    )


def evaluate_gate(
    trend: TrendReport,
    direction: TradeDirection,
    gate_cfg: GateConfig,
    pattern: Optional[PatternMatch] = None,
    nearest_level: Optional[dict] = None,
    volatility_percentile: Optional[float] = None,
    risk_score: Optional[float] = None,
    confirmation_candle_direction: Optional[TradeDirection] = None,
) -> GateResult:
    """Evaluate candidate trade direction against strict trend gate and reversal bypass.

    Strict Trend Gate (Section 6.3):
      - CALL passes only if daily=UP and 1h=UP and 15m!=DOWN
      - PUT passes only if daily=DOWN and 1h=DOWN and 15m!=UP
      - Otherwise SKIP

    Reversal Exception (reversal_mode: sr_bypass):
      Bypasses strict trend gate only if ALL are true:
        - pattern.similarity >= min_similarity
        - price is at support (CALL) or resistance (PUT)
        - volatility_percentile <= max_volatility_percentile
        - risk_score <= max_risk_score
        - latest confirmation candle points same direction
    """
    # 0. Check for Insufficient Data
    if (
        trend.daily == TrendLabel.INSUFFICIENT_DATA
        or trend.hourly == TrendLabel.INSUFFICIENT_DATA
        or trend.fifteen_min == TrendLabel.INSUFFICIENT_DATA
    ):
        min_bars = gate_cfg.trend.ema_slow * gate_cfg.trend.warmup_multiplier
        reason = (
            f"Gate SKIP: insufficient data for {direction.value} "
            f"(requires >= {min_bars} bars for stable indicators; "
            f"daily={trend.daily.value}, 1h={trend.hourly.value}, 15m={trend.fifteen_min.value})"
        )
        logger.info(reason)
        return GateResult(passed=False, direction=direction, reason=reason)

    # 1. Standard Strict Trend Gate Check
    if direction == TradeDirection.CALL:
        standard_pass = (
            trend.daily == TrendLabel.UP
            and trend.hourly == TrendLabel.UP
            and trend.fifteen_min != TrendLabel.DOWN
        )
    else:  # PUT
        standard_pass = (
            trend.daily == TrendLabel.DOWN
            and trend.hourly == TrendLabel.DOWN
            and trend.fifteen_min != TrendLabel.UP
        )

    if standard_pass:
        return GateResult(
            passed=True,
            direction=direction,
            reason="passed_strict_trend_gate",
        )

    # 2. Reversal Exception Check
    if gate_cfg.reversal_mode == "off":
        reason = (
            f"Strict trend gate rejected {direction.value}: "
            f"daily={trend.daily.value}, 1h={trend.hourly.value}, 15m={trend.fifteen_min.value}"
        )
        logger.info(f"Gate SKIP: {reason}")
        return GateResult(passed=False, direction=direction, reason=reason)

    if gate_cfg.reversal_mode == "sr_bypass":
        bypass_cfg = gate_cfg.reversal_bypass
        fail_reasons: list[str] = []

        # Condition A: Valid reversal pattern matching trade direction
        if pattern is None or pattern.pattern_type != "reversal":
            fail_reasons.append("pattern is not reversal")
        elif pattern.direction != direction:
            fail_reasons.append(
                f"pattern direction {pattern.direction.value} != {direction.value}"
            )
        elif pattern.similarity < bypass_cfg.min_similarity:
            fail_reasons.append(
                f"pattern similarity {pattern.similarity} < {bypass_cfg.min_similarity}"
            )

        # Condition B: Price is at support (for CALL) or resistance (for PUT)
        required_level_type = "support" if direction == TradeDirection.CALL else "resistance"
        if bypass_cfg.require_sr_level:
            tolerance = bypass_cfg.sr_level_tolerance_atr
            if not nearest_level:
                fail_reasons.append(f"missing nearest {required_level_type} level")
            elif nearest_level.get("type") != required_level_type:
                fail_reasons.append(
                    f"nearest level type {nearest_level.get('type')} != {required_level_type}"
                )
            else:
                dist = nearest_level.get("distance_atr", 999.0)
                is_at = nearest_level.get("is_at_level", False) or (dist <= tolerance)
                if not is_at:
                    fail_reasons.append(
                        f"price is not at {required_level_type} level "
                        f"(distance={dist} ATR > tolerance={tolerance} ATR)"
                    )

        # Condition C: Volatility percentile <= max
        if volatility_percentile is None:
            fail_reasons.append("missing volatility_percentile")
        elif volatility_percentile > bypass_cfg.max_volatility_percentile:
            fail_reasons.append(
                f"volatility percentile {volatility_percentile} > {bypass_cfg.max_volatility_percentile}"
            )

        # Condition D: Risk score <= max (computed in Phase 4 / prior to bypass)
        if risk_score is None:
            fail_reasons.append(
                "missing risk_score (risk engine must evaluate candidate before reversal bypass)"
            )
        elif risk_score > bypass_cfg.max_risk_score:
            fail_reasons.append(
                f"risk score {risk_score} > {bypass_cfg.max_risk_score}"
            )

        # Condition E: Confirmation candle points the same way
        if bypass_cfg.require_confirmation_candle:
            if confirmation_candle_direction is None:
                fail_reasons.append("missing confirmation candle direction")
            elif confirmation_candle_direction != direction:
                fail_reasons.append(
                    f"confirmation candle direction {confirmation_candle_direction.value} != {direction.value}"
                )

        if not fail_reasons:
            logger.info(f"Gate PASS (reversal bypass) for {direction.value}")
            return GateResult(
                passed=True,
                direction=direction,
                reason="passed_reversal_sr_bypass",
            )
        else:
            reason = f"Reversal bypass rejected: {', '.join(fail_reasons)}"
            logger.info(f"Gate SKIP: {reason}")
            return GateResult(passed=False, direction=direction, reason=reason)

    return GateResult(
        passed=False,
        direction=direction,
        reason=f"Unknown reversal_mode {gate_cfg.reversal_mode}",
    )
