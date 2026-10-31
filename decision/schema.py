"""Pydantic schemas for LLM decision report, output, and validation.

PROJECT_SPEC.md Section 7.2, 7.3, 7.4 & Phase 7:
  - Strict input report schema matching Section 7.2.
  - Strict output schema matching Section 7.3 with extra="forbid".
  - Validation models and system runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Section 7.2: Input Report Models ──────────────────────────────────


class TrendReportItem(BaseModel):
    """Multi-timeframe trend breakdown."""

    model_config = ConfigDict(extra="forbid")

    daily: str
    h1: str = Field(alias="1h")
    m15: str = Field(alias="15m")


class NearestLevelItem(BaseModel):
    """Nearest support/resistance level."""

    model_config = ConfigDict(extra="forbid")

    type: str  # "support" or "resistance"
    distance_atr: Optional[float] = None


class PatternReportItem(BaseModel):
    """Best refined pattern match."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str  # "continuation" or "reversal"
    similarity: float
    refined: bool = True


class AssetStatsItem(BaseModel):
    """Historical performance stats for the asset."""

    model_config = ConfigDict(extra="forbid")

    trades: int
    win_rate: Optional[float] = None
    mode: str  # "paper" or "live_eligible"


class CandidateReportItem(BaseModel):
    """A qualified candidate trade presented to the LLM."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset: str
    direction: Literal["CALL", "PUT"]
    expiry_minutes: int
    payout: float
    trend: TrendReportItem
    volatility_percentile: float
    nearest_level: NearestLevelItem
    pattern: PatternReportItem
    risk_score: int
    max_stake_pct: float
    asset_stats: AssetStatsItem
    shadow_only: bool = False
    current_price: Optional[float] = None
    atr: Optional[float] = None


class AdaptiveReportItem(BaseModel):
    """Read-only adaptive regulator telemetry included in LLM report."""

    model_config = ConfigDict(extra="forbid")

    tier: str = "NORMAL"
    stake_scale: float = 1.0
    min_similarity_add: float = 0.0
    max_risk_cap: Optional[float] = None
    real_trading_allowed: bool = True


class DecisionReport(BaseModel):
    """Complete structured JSON report supplied to the LLM."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: str
    balance: float
    equity: float
    open_exposure_pct: float
    remaining_exposure_pct: float
    min_stake_pct: float
    candidates: list[CandidateReportItem]
    adaptive_state: Optional[AdaptiveReportItem] = None


# ── Section 7.3: LLM Decision Output Models ───────────────────────────


class DecisionEvidence(BaseModel):
    """Evidence metrics quoted by the LLM in support of its decision."""

    model_config = ConfigDict(extra="forbid")

    risk_score: int
    similarity: float
    trend_daily: str
    trend_1h: str
    payout: float


class DecisionOutput(BaseModel):
    """Structured decision returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["CALL", "PUT", "SKIP"]
    asset: Optional[str] = None
    expiry_minutes: Optional[int] = None
    stake_pct: Optional[float] = None
    evidence: Optional[DecisionEvidence] = None
    reasoning: str
    decision_id: Optional[str] = None

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning_length(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 60:
            raise ValueError(f"reasoning must be at most 60 words (got {len(words)} words)")
        return v


# ── Runtime State & Validated Decision ────────────────────────────────


@dataclass
class SystemState:
    """Runtime system state passed to the validator."""

    balance: float = 0.0
    kill_switch_active: bool = False
    breaker_tripped: bool = False
    reconcile_done: bool = True
    current_prices: dict[str, float] = field(default_factory=dict)
    open_trades: list[Any] = field(default_factory=list)
    asset_modes: dict[str, str] = field(default_factory=dict)
    adaptive_state: Optional[Any] = None


@dataclass
class ValidatedDecision:
    """Outcome of validating an LLM decision."""

    decision: Literal["CALL", "PUT", "SKIP"]
    is_valid: bool
    decision_id: Optional[str] = None
    stake_amount: Optional[float] = None
    asset: Optional[str] = None
    direction: Optional[str] = None
    expiry_minutes: Optional[int] = None
    stake_pct: Optional[float] = None
    evidence: Optional[DecisionEvidence] = None
    reasoning: str = ""
    checks_passed: list[str] = field(default_factory=list)
    skip_reason: Optional[str] = None
    raw_decision: Optional[DecisionOutput] = None
    shadow_only: bool = False
    pattern_name: Optional[str] = None
    pattern_type: Optional[str] = None
    risk_score: Optional[int] = None
    payout: Optional[float] = None
    is_otc: bool = False
