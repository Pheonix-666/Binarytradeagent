"""Shared data models for the trading system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    """Direction of a fixed-time trade."""

    CALL = "CALL"
    PUT = "PUT"


class TradeResult(str, Enum):
    """Outcome of a trade."""

    WIN = "WIN"
    LOSS = "LOSS"
    PENDING = "PENDING"


class TrendLabel(str, Enum):
    """Trend classification."""

    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"


class AssetQuote(BaseModel):
    """Real-time quote snapshot for a single asset."""

    asset_id: str
    name: str
    is_otc: bool
    payout: float = Field(ge=0, le=1)
    is_open: bool


class TradeReceipt(BaseModel):
    """Immutable record created at trade entry."""

    trade_id: str
    asset_id: str
    direction: TradeDirection
    expiry_minutes: int
    stake_amount: float
    entry_price: float
    entry_time: datetime


class Trade(TradeReceipt):
    """Full trade record including outcome."""

    result: TradeResult
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    payout_received: Optional[float] = None


class TrendReport(BaseModel):
    """Multi-timeframe trend assessment."""

    daily: TrendLabel
    hourly: TrendLabel
    fifteen_min: TrendLabel


class PatternMatch(BaseModel):
    """Result of a pattern similarity search."""

    pattern_name: str
    direction: TradeDirection
    pattern_type: Literal["continuation", "reversal"]
    similarity: float = Field(ge=0, le=100)


class RiskReport(BaseModel):
    """Composite risk assessment for a candidate trade."""

    volatility_score: float
    pattern_score: float
    win_rate_score: float
    session_score: float
    sr_distance_score: float
    total_risk: float = Field(ge=1, le=100)
    max_stake_pct: float


class GateResult(BaseModel):
    """Outcome of the trend/reversal gate check."""

    passed: bool
    direction: Optional[TradeDirection] = None
    reason: str


class Candidate(BaseModel):
    """A fully evaluated trade candidate ready for LLM decision."""

    asset: str
    direction: TradeDirection
    expiry_minutes: int
    payout: float
    trend: TrendReport
    volatility_percentile: float
    nearest_level: dict
    pattern: Optional[PatternMatch] = None
    risk_score: float
    max_stake_pct: float
    asset_stats: dict
