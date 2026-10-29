"""Single code path for candidate generation across backtest and live trading.

PROJECT_SPEC.md Sections 6.2, 6.3, 6.5, 6.6, & Section 11:
  - build_candidates(asset, t, candles_by_tf, stats_as_of, config, ...):
    For each direction (CALL/PUT):
      1. Find refined pattern matches in that direction with similarity >= patterns.min_similarity.
      2. Compute direction-aware S/R distance and levels.
      3. Compute volatility percentile (ATR).
      4. Assemble RiskInput from asset stats and hour risk as-of t.
      5. Compute composite risk score.
      6. Evaluate gate (strict trend or reversal bypass with risk_score).
      7. Compute max_stake_pct from risk score.
      8. Return qualifying candidates with complete evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.indicators import compute_atr
from analysis.levels import find_swing_levels, get_nearest_level, get_nearest_opposing_level
from analysis.trend import analyze_trend, evaluate_gate
from core.config import AppConfig
from core.models import GateResult, PatternMatch, TradeDirection, TrendLabel, TrendReport
from patterns.library import PatternLibrary, get_pattern_library
from risk.engine import RiskInput, RiskReport as RiskEngineReport, compute_risk_score
from risk.stake import max_stake_pct
from selection.stats import AssetStats


@dataclass
class CandidateTrade:
    """A fully qualified candidate trade passing all gates and pattern filters."""

    asset: str
    direction: TradeDirection
    t: datetime
    current_price: float
    atr: float
    pattern: dict
    pattern_match: PatternMatch
    risk_report: RiskEngineReport
    gate_result: GateResult
    max_stake_pct: float
    volatility_percentile: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEvaluationResult:
    """Result of evaluating candidates for an asset at time t, including skips."""

    asset: str
    t: datetime
    candidates: list[CandidateTrade]
    skip_reasons: list[str] = field(default_factory=list)


def compute_volatility_percentile(
    df_candles: pd.DataFrame,
    atr_period: int = 14,
    lookback_bars: int = 200,
) -> float:
    """Compute current ATR percentile rank (0-100) relative to recent history.

    Returns 50.0 (median) if insufficient history (< 20 bars).
    """
    if len(df_candles) < max(20, atr_period + 5):
        return 50.0

    atr_series = compute_atr(df_candles, period=atr_period).dropna()
    if len(atr_series) < 10:
        return 50.0

    recent_atr = atr_series.tail(lookback_bars).values
    cur_atr = recent_atr[-1]
    pct = float(np.sum(recent_atr <= cur_atr) / len(recent_atr) * 100.0)
    return float(np.clip(pct, 0.0, 100.0))


def build_candidates(
    asset: str,
    t: datetime,
    candles_by_tf: Mapping[str, pd.DataFrame],
    stats_as_of: Mapping[str, AssetStats],
    config: AppConfig,
    hour_risk_val: Optional[float] = None,
    payout: Optional[float] = None,
    pattern_library: Optional[PatternLibrary] = None,
    calendar: Optional[CalendarConfig] = None,
) -> CandidateEvaluationResult:
    """Generate trade candidates for an asset at time t.

    Single code path: used identically by backtest and live orchestrator.

    Parameters:
      asset: Asset ID (e.g. 'EURUSD').
      t: Evaluation timestamp (UTC).
      candles_by_tf: Closed candles up to t for timeframes ('1d', '1h', '15m', '5m', '1m').
      stats_as_of: Performance stats computed strictly from trades closed by t.
      config: System configuration.
      hour_risk_val: Pre-computed hour-of-day risk score as of t (optional).
      payout: Payout rate (e.g. 0.85); if None, uses default 0.85.
      pattern_library: PatternLibrary instance; if None, uses default singleton.
      calendar: Market calendar configuration (optional).
    """
    if pattern_library is None:
        templates_dir = getattr(config.patterns, "templates_dir", "patterns/templates")
        pattern_library = get_pattern_library(templates_dir)

    effective_payout = payout if (payout is not None and payout > 0.0) else 0.85

    # 1. Multi-timeframe trend report
    trend_report = analyze_trend(candles_by_tf, config.gate.trend, calendar=calendar)

    # Check for Insufficient Data in any trend timeframe
    if (
        trend_report.daily == TrendLabel.INSUFFICIENT_DATA
        or trend_report.hourly == TrendLabel.INSUFFICIENT_DATA
        or trend_report.fifteen_min == TrendLabel.INSUFFICIENT_DATA
    ):
        return CandidateEvaluationResult(
            asset=asset,
            t=t,
            candidates=[],
            skip_reasons=["insufficient data for trend indicators"],
        )

    # Direction eligibility pre-check: skip expensive DTW pattern matching if gate will reject
    can_call_trend = (
        trend_report.daily == TrendLabel.UP
        and trend_report.hourly == TrendLabel.UP
        and trend_report.fifteen_min != TrendLabel.DOWN
    )
    can_put_trend = (
        trend_report.daily == TrendLabel.DOWN
        and trend_report.hourly == TrendLabel.DOWN
        and trend_report.fifteen_min != TrendLabel.UP
    )
    is_reversal = (config.gate.reversal_mode == "sr_bypass")

    eligible_directions: list[TradeDirection] = []
    if can_call_trend or is_reversal:
        eligible_directions.append(TradeDirection.CALL)
    if can_put_trend or is_reversal:
        eligible_directions.append(TradeDirection.PUT)

    if not eligible_directions:
        return CandidateEvaluationResult(
            asset=asset,
            t=t,
            candidates=[],
            skip_reasons=[
                f"gate: trend not aligned (daily={trend_report.daily.value}, 1h={trend_report.hourly.value}, 15m={trend_report.fifteen_min.value})"
            ],
        )

    # Determine reference price and ATR
    df_5m = candles_by_tf.get("5m", pd.DataFrame())
    df_15m = candles_by_tf.get("15m", pd.DataFrame())
    df_1h = candles_by_tf.get("1h", pd.DataFrame())
    df_1m = candles_by_tf.get("1m", pd.DataFrame())

    ref_df = df_5m if not df_5m.empty else (df_15m if not df_15m.empty else df_1m)
    if ref_df.empty:
        return CandidateEvaluationResult(
            asset=asset,
            t=t,
            candidates=[],
            skip_reasons=["insufficient data: no candle data available"],
        )

    current_price = float(ref_df["close"].iloc[-1])

    # Compute ATR on 15m (fallback to 5m or ref)
    atr_source_df = df_15m if len(df_15m) >= config.risk.atr_period else ref_df
    atr_series = compute_atr(atr_source_df, period=config.risk.atr_period).dropna()
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else (current_price * 0.001)
    if atr_val <= 0.0 or np.isnan(atr_val):
        atr_val = current_price * 0.001

    # Volatility percentile
    vol_pct = compute_volatility_percentile(
        df_15m if not df_15m.empty else ref_df,
        atr_period=config.risk.atr_period,
    )

    # S/R Levels
    levels_source_df = df_1h if not df_1h.empty else df_15m
    sr_levels_known = len(levels_source_df) >= (
        config.levels.swing_left_bars + config.levels.swing_right_bars + 1
    )
    levels = find_swing_levels(
        levels_source_df,
        left_bars=config.levels.swing_left_bars,
        right_bars=config.levels.swing_right_bars,
        cluster_tolerance_atr=config.levels.cluster_tolerance_atr,
        atr=atr_val,
    )

    # Pattern search across configured timeframes (refined only)
    all_patterns = pattern_library.find_patterns_for_asset(
        candles_by_tf,
        config.patterns,
        refined_only=True,
    )

    # Confirmation candle direction from latest closed candle
    confirmation_candle_dir = None
    if len(ref_df) >= 1:
        last_c = ref_df.iloc[-1]
        if last_c["close"] > last_c["open"]:
            confirmation_candle_dir = TradeDirection.CALL
        elif last_c["close"] < last_c["open"]:
            confirmation_candle_dir = TradeDirection.PUT

    # Asset stats
    asset_stat = stats_as_of.get(asset)
    n_trades = asset_stat.n if asset_stat else 0
    asset_wr = asset_stat.win_rate if (asset_stat and n_trades >= config.trading.min_trades_for_stats) else None

    candidates: list[CandidateTrade] = []
    skip_reasons: list[str] = []

    # Check if overall insufficient data
    if (
        trend_report.daily == TrendLabel.INSUFFICIENT_DATA
        or trend_report.hourly == TrendLabel.INSUFFICIENT_DATA
        or trend_report.fifteen_min == TrendLabel.INSUFFICIENT_DATA
    ):
        skip_reasons.append("insufficient data for trend indicators")

    # Evaluate each eligible direction
    for direction in eligible_directions:
        # Filter patterns matching direction and min_similarity
        dir_patterns = [
            p for p in all_patterns
            if p["direction"].upper() == direction.value.upper()
            and p["similarity"] >= config.patterns.min_similarity
        ]

        if not dir_patterns:
            skip_reasons.append(f"no pattern matching {direction.value} >= {config.patterns.min_similarity}")
            continue

        best_pattern = dir_patterns[0]
        pattern_match_obj = PatternMatch(
            pattern_name=best_pattern["pattern"],
            direction=direction,
            pattern_type=best_pattern["type"],
            similarity=float(best_pattern["similarity"]),
        )

        # Nearest opposing level (distance to resistance above for CALL, support below for PUT)
        opposing_level = get_nearest_opposing_level(
            current_price=current_price,
            trade_direction=direction,
            levels=levels,
            atr=atr_val,
            at_level_atr_threshold=config.levels.at_level_tolerance_atr,
        )
        sr_dist_atr = opposing_level["distance_atr"] if opposing_level["touches"] > 0 else None

        # Nearest level for reversal bypass check (support for CALL, resistance for PUT)
        reversal_level_type = "support" if direction == TradeDirection.CALL else "resistance"
        nearest_reversal_lvl = get_nearest_level(
            current_price=current_price,
            levels=levels,
            target_type=reversal_level_type,
            atr=atr_val,
            at_level_atr_threshold=config.gate.reversal_bypass.sr_level_tolerance_atr or config.levels.at_level_tolerance_atr,
        )

        # Assemble RiskInput
        risk_input = RiskInput(
            volatility_percentile=vol_pct,
            best_pattern_similarity=float(best_pattern["similarity"]),
            asset_win_rate=asset_wr,
            asset_trades=n_trades,
            payout=effective_payout,
            hour_of_day_risk=hour_risk_val,
            sr_distance_atr=sr_dist_atr,
            sr_levels_known=sr_levels_known,
        )

        # Compute risk score
        risk_report = compute_risk_score(risk_input, config.risk)

        # Evaluate gate with computed risk score
        gate_res = evaluate_gate(
            trend=trend_report,
            direction=direction,
            gate_cfg=config.gate,
            pattern=pattern_match_obj,
            nearest_level=nearest_reversal_lvl,
            volatility_percentile=vol_pct,
            risk_score=float(risk_report.score),
            confirmation_candle_direction=confirmation_candle_dir,
            at_level_tolerance_atr=config.gate.reversal_bypass.sr_level_tolerance_atr or config.levels.at_level_tolerance_atr,
        )

        if not gate_res.passed:
            skip_reasons.append(f"gate: {gate_res.reason}")
            continue

        # Stake sizing
        stake_pct = max_stake_pct(
            float(risk_report.score),
            min_pct=config.stake.min_pct,
            max_pct=config.stake.max_pct,
        )

        evidence = {
            "trend": {
                "daily": trend_report.daily.value,
                "hourly": trend_report.hourly.value,
                "fifteen_min": trend_report.fifteen_min.value,
            },
            "gate_reason": gate_res.reason,
            "pattern": best_pattern,
            "risk_score": risk_report.score,
            "risk_breakdown": {
                "volatility": risk_report.volatility_score,
                "pattern": risk_report.pattern_score,
                "win_rate": risk_report.win_rate_score,
                "session": risk_report.session_score,
                "sr_distance": risk_report.sr_distance_score,
            },
            "sr_distance_atr": sr_dist_atr,
            "volatility_percentile": vol_pct,
            "payout": effective_payout,
        }

        candidates.append(
            CandidateTrade(
                asset=asset,
                direction=direction,
                t=t,
                current_price=current_price,
                atr=atr_val,
                pattern=best_pattern,
                pattern_match=pattern_match_obj,
                risk_report=risk_report,
                gate_result=gate_res,
                max_stake_pct=stake_pct,
                volatility_percentile=vol_pct,
                evidence=evidence,
            )
        )

    return CandidateEvaluationResult(
        asset=asset,
        t=t,
        candidates=candidates,
        skip_reasons=skip_reasons,
    )
