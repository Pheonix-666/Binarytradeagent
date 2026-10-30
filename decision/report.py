"""Pure report builder converting candidate trades into structured LLM JSON reports.

PROJECT_SPEC.md Section 7.2 & Phase 7:
  - Pure function build_report().
  - Deterministic candidate ordering (asset, direction).
  - Shadow-only tagging for paper-mode assets.
  - Strict asset name sanitization [A-Za-z0-9_/.-] (max 32 chars) against prompt injection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping, Optional, Sequence

from analysis.candidates import CandidateTrade
from core.config import AppConfig
from decision.schema import (
    AdaptiveReportItem,
    AssetStatsItem,
    CandidateReportItem,
    DecisionReport,
    NearestLevelItem,
    PatternReportItem,
    TrendReportItem,
)

ASSET_REGEX = re.compile(r"^[A-Za-z0-9_/.-]{1,32}$")


def sanitize_asset_name(asset: str) -> Optional[str]:
    """Validate and sanitize external asset identifiers.

    Allows ONLY characters in [A-Za-z0-9_/.-] with a maximum length of 32 characters.
    Rejects any string containing spaces, control codes, or prompt-injection text.
    """
    if not isinstance(asset, str):
        return None
    cleaned = asset.strip()
    if not cleaned or len(cleaned) > 32:
        return None
    if not ASSET_REGEX.match(cleaned):
        return None
    return cleaned


def build_report(
    candidates: Sequence[CandidateTrade],
    balance: float,
    open_trades: Sequence[Any],
    config: AppConfig,
    now: datetime,
    asset_modes: Optional[Mapping[str, str]] = None,
    asset_stats_map: Optional[Mapping[str, Any]] = None,
    adaptive_state: Optional[Any] = None,
) -> DecisionReport:
    """Turn CandidateTrade objects into the structured DecisionReport from Section 7.2.

    Pure function: no network or disk I/O.
    """
    if now.tzinfo is None:
        now_utc = now.replace(tzinfo=timezone.utc)
    else:
        now_utc = now.astimezone(timezone.utc)

    # Calculate portfolio exposure
    open_stakes = sum(float(getattr(ot, "stake", 0.0)) for ot in open_trades)
    equity = float(balance + open_stakes)
    open_exposure_pct = round((open_stakes / equity * 100.0), 2) if equity > 0.0 else 0.0
    total_cap_pct = float(config.stake.total_cap_pct)
    remaining_exposure_pct = round(max(0.0, total_cap_pct - open_exposure_pct), 2)
    min_stake_pct = float(config.stake.min_pct)

    # Process and sanitize candidates
    report_candidates: list[CandidateReportItem] = []
    modes_map = asset_modes or {}
    stats_map = asset_stats_map or {}

    for cand in candidates:
        sanitized_asset = sanitize_asset_name(cand.asset)
        if sanitized_asset is None:
            # Drop candidates with malformed or injection-suspect asset identifiers
            continue

        direction_str = cand.direction.value.upper()
        if direction_str not in ("CALL", "PUT"):
            continue

        # Asset mode and shadow flag
        asset_mode = modes_map.get(cand.asset, "paper").lower()
        if asset_mode not in ("paper", "live_eligible"):
            asset_mode = "paper"
        shadow_only = (asset_mode == "paper")

        # Trend breakdown
        ev = cand.evidence or {}
        trend_dict = ev.get("trend", {})
        trend_item = TrendReportItem(
            daily=str(trend_dict.get("daily", "unknown")).lower(),
            **{
                "1h": str(trend_dict.get("hourly", "unknown")).lower(),
                "15m": str(trend_dict.get("fifteen_min", "unknown")).lower(),
            },
        )

        # Nearest level
        level_type = "support" if cand.direction.value == "CALL" else "resistance"
        sr_dist = cand.evidence.get("sr_distance_atr")
        nearest_level = NearestLevelItem(
            type=level_type,
            distance_atr=round(float(sr_dist), 2) if sr_dist is not None else None,
        )

        # Pattern match (refined only)
        pattern_data = cand.pattern or {}
        pattern_name = str(pattern_data.get("pattern", cand.pattern_match.pattern_name))
        pattern_type = str(pattern_data.get("type", cand.pattern_match.pattern_type))
        similarity = round(float(pattern_data.get("similarity", cand.pattern_match.similarity)), 1)
        pattern_item = PatternReportItem(
            name=pattern_name,
            type=pattern_type,
            similarity=similarity,
            refined=True,
        )

        # Asset stats
        stat_obj = stats_map.get(cand.asset)
        trades_count = getattr(stat_obj, "n", 0) if stat_obj else 0
        wr = getattr(stat_obj, "win_rate", None) if stat_obj else None
        if wr is not None:
            wr = round(float(wr), 2)
        asset_stats_item = AssetStatsItem(
            trades=int(trades_count),
            win_rate=wr,
            mode=asset_mode,
        )

        payout = round(float(ev.get("payout", getattr(config.backtest, "default_payout", 0.85))), 2)
        vol_pct = round(float(cand.volatility_percentile), 1)
        max_stake = round(float(cand.max_stake_pct), 1)

        report_candidates.append(
            CandidateReportItem(
                asset=sanitized_asset,
                direction=direction_str,  # type: ignore
                expiry_minutes=int(config.trading.expiry_minutes),
                payout=payout,
                trend=trend_item,
                volatility_percentile=vol_pct,
                nearest_level=nearest_level,
                pattern=pattern_item,
                risk_score=int(cand.risk_report.score),
                max_stake_pct=max_stake,
                asset_stats=asset_stats_item,
                shadow_only=shadow_only,
                current_price=cand.current_price,
                atr=cand.atr,
            )
        )

    # Deterministic candidate ordering: sort by asset id, then direction
    report_candidates.sort(key=lambda c: (c.asset, c.direction))

    ad_item = None
    if adaptive_state is not None:
        ad_item = AdaptiveReportItem(
            tier=str(getattr(adaptive_state, "tier", "NORMAL")),
            stake_scale=float(getattr(adaptive_state, "stake_scale", 1.0)),
            min_similarity_add=float(getattr(adaptive_state, "min_similarity_add", 0.0)),
            max_risk_cap=float(getattr(adaptive_state, "max_risk_cap")) if getattr(adaptive_state, "max_risk_cap", None) is not None else None,
            real_trading_allowed=bool(getattr(adaptive_state, "real_trading_allowed", True)),
        )

    return DecisionReport(
        timestamp=now_utc.isoformat(),
        balance=round(balance, 2),
        equity=round(equity, 2),
        open_exposure_pct=open_exposure_pct,
        remaining_exposure_pct=remaining_exposure_pct,
        min_stake_pct=min_stake_pct,
        candidates=report_candidates,
        adaptive_state=ad_item,
    )

