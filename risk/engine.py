"""Risk engine — composite risk scoring from multiple factors.

Section 6.5 & Section 12 (Phase 4):
  - Pure calculation (zero I/O).
  - Evaluates 5 sub-scores (0-100, higher = riskier):
      1. Volatility percentile (0-100)
      2. Pattern similarity: 100 - similarity (or default 100 if missing)
      3. Win rate: 100 * clip((break_even + 0.15 - win_rate) / 0.20, 0.01, 1)
         (or default 100 if trades < min_trades_for_stats)
      4. Session: historical risk by hour of day (or default 50 if missing / < session_min_samples)
      5. S/R distance: under 0.5 ATR = 90, 3+ ATR = 10, linear between (or default 10 if none)
  - Final score = weighted average rounded and clamped to 1-100.
  - Returns RiskReport with integer score AND individual sub-scores.
  - Never returns NaN or raises on missing inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from core.config import RiskConfig


@dataclass(frozen=True)
class RiskInput:
    """Input parameters for risk assessment."""

    volatility_percentile: Optional[float]
    best_pattern_similarity: Optional[float]
    asset_win_rate: Optional[float]
    asset_trades: int
    payout: float
    hour_of_day_risk: Optional[float] = None
    sr_distance_atr: Optional[float] = None


@dataclass(frozen=True)
class RiskReport:
    """Detailed risk assessment output."""

    score: int  # 1 to 100
    volatility_score: float
    pattern_score: float
    win_rate_score: float
    session_score: float
    sr_distance_score: float
    weights: dict[str, float]
    raw_weighted_score: float


def compute_volatility_subscore(
    volatility_percentile: Optional[float],
    risk_cfg: RiskConfig,
) -> float:
    """Compute volatility risk sub-score (0-100)."""
    if volatility_percentile is None or math.isnan(volatility_percentile):
        return float(risk_cfg.defaults.volatility_missing)
    return float(np.clip(volatility_percentile, 0.0, 100.0))


def compute_pattern_subscore(
    best_pattern_similarity: Optional[float],
    risk_cfg: RiskConfig,
) -> float:
    """Compute pattern risk sub-score (0-100, 100 - best_similarity)."""
    if best_pattern_similarity is None or math.isnan(best_pattern_similarity):
        return float(risk_cfg.defaults.pattern_missing)
    score = 100.0 - best_pattern_similarity
    return float(np.clip(score, 0.0, 100.0))


def compute_win_rate_subscore(
    asset_win_rate: Optional[float],
    asset_trades: int,
    payout: float,
    min_trades_for_stats: int,
    risk_cfg: RiskConfig,
) -> float:
    """Compute win rate risk sub-score.

    Formula: 100 * clip((break_even + 0.15 - win_rate) / 0.20, 0.01, 1).
    If trades < min_trades_for_stats or asset_win_rate is None -> default 100.
    """
    if (
        asset_win_rate is None
        or math.isnan(asset_win_rate)
        or asset_trades < min_trades_for_stats
    ):
        return float(risk_cfg.defaults.untested_asset)

    # Break-even = 1 / (1 + payout)
    if payout <= 0.0 or math.isnan(payout):
        be = 0.5
    else:
        be = 1.0 / (1.0 + payout)

    val = (be + 0.15 - asset_win_rate) / 0.20
    clamped = float(np.clip(val, 0.01, 1.0))
    return float(np.clip(100.0 * clamped, 1.0, 100.0))


def compute_session_subscore(
    hour_of_day_risk: Optional[float],
    risk_cfg: RiskConfig,
) -> float:
    """Compute session / hour of day risk sub-score (0-100)."""
    if hour_of_day_risk is None or math.isnan(hour_of_day_risk):
        return float(risk_cfg.defaults.session_missing)
    return float(np.clip(hour_of_day_risk, 0.0, 100.0))


def compute_sr_distance_subscore(
    sr_distance_atr: Optional[float],
    risk_cfg: RiskConfig,
) -> float:
    """Compute S/R distance sub-score.

    Distance to nearest opposing level in trade direction:
      - under 0.5 ATR = 90
      - 3 ATR or more = 10
      - linear between
      - if no opposing level found -> default (10.0)
    """
    if sr_distance_atr is None or math.isnan(sr_distance_atr):
        return float(risk_cfg.defaults.no_sr_level)

    min_d = risk_cfg.sr_distance_min_atr
    max_d = risk_cfg.sr_distance_max_atr
    near_score = risk_cfg.sr_distance_near_score
    far_score = risk_cfg.sr_distance_far_score

    if sr_distance_atr <= min_d:
        return float(near_score)
    if sr_distance_atr >= max_d:
        return float(far_score)

    # Linear interpolation between near_score and far_score
    fraction = (sr_distance_atr - min_d) / (max_d - min_d)
    score = near_score - fraction * (near_score - far_score)
    return float(np.clip(score, far_score, near_score))


def compute_risk_score(
    risk_input: RiskInput,
    risk_cfg: RiskConfig,
    min_trades_for_stats: int = 30,
) -> RiskReport:
    """Compute composite risk report from RiskInput and configuration."""
    s_vol = compute_volatility_subscore(risk_input.volatility_percentile, risk_cfg)
    s_pat = compute_pattern_subscore(risk_input.best_pattern_similarity, risk_cfg)
    s_win = compute_win_rate_subscore(
        risk_input.asset_win_rate,
        risk_input.asset_trades,
        risk_input.payout,
        min_trades_for_stats,
        risk_cfg,
    )
    s_sess = compute_session_subscore(risk_input.hour_of_day_risk, risk_cfg)
    s_sr = compute_sr_distance_subscore(risk_input.sr_distance_atr, risk_cfg)

    weights = {
        "volatility": risk_cfg.weights.volatility,
        "pattern": risk_cfg.weights.pattern,
        "win_rate": risk_cfg.weights.win_rate,
        "session": risk_cfg.weights.session,
        "sr_distance": risk_cfg.weights.sr_distance,
    }

    raw_weighted = (
        weights["volatility"] * s_vol
        + weights["pattern"] * s_pat
        + weights["win_rate"] * s_win
        + weights["session"] * s_sess
        + weights["sr_distance"] * s_sr
    )

    if math.isnan(raw_weighted):
        raw_weighted = 50.0

    final_score = int(np.clip(round(raw_weighted), 1, 100))

    return RiskReport(
        score=final_score,
        volatility_score=round(s_vol, 2),
        pattern_score=round(s_pat, 2),
        win_rate_score=round(s_win, 2),
        session_score=round(s_sess, 2),
        sr_distance_score=round(s_sr, 2),
        weights=weights,
        raw_weighted_score=round(float(raw_weighted), 4),
    )
