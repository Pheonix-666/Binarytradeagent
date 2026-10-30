"""Shared data models for the trading system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    """Direction of a fixed-time trade."""

    CALL = "CALL"
    PUT = "PUT"


class TradeResult(str, Enum):
    """Outcome of a trade."""

    WIN = "WIN"
    LOSS = "LOSS"
    TIE = "TIE"
    PENDING = "PENDING"


class TrendLabel(str, Enum):
    """Trend classification."""

    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


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
    payout: float = 0.85
    platform_trade_id: Optional[str] = None
    status: str = "OPEN"


class Trade(TradeReceipt):
    """Full trade record including outcome."""

    result: TradeResult
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    payout_received: Optional[float] = None

    @property
    def asset(self) -> str:
        return self.asset_id

    @property
    def stake(self) -> float:
        return self.stake_amount


@dataclass
class CanonicalTrade:
    """Canonical trade representation across state, validator, and safety."""

    id: str
    asset: str
    direction: str  # "CALL" or "PUT"
    stake: float
    expiry_minutes: int
    payout: float = 0.85
    opened_at: Optional[str] = None
    status: str = "OPEN"
    decision_id: Optional[str] = None

    @classmethod
    def from_any(cls, item: Any) -> "CanonicalTrade":
        if isinstance(item, CanonicalTrade):
            return item
        if isinstance(item, dict):
            t_id = str(item.get("id") or item.get("trade_id") or "")
            asset = str(item.get("asset") or item.get("asset_id") or "")
            raw_dir = item.get("direction", "")
            raw_stk = item.get("stake") if item.get("stake") is not None else item.get("stake_amount", 0.0)
            try:
                stake = float(raw_stk)
            except (ValueError, TypeError):
                stake = float("nan")
            raw_exp = item.get("expiry_minutes") if item.get("expiry_minutes") is not None else item.get("expiry_min", 0)
            try:
                exp = int(raw_exp)
            except (ValueError, TypeError):
                exp = 0
            payout = float(item.get("payout", 0.85))
            return cls(
                id=t_id,
                asset=asset,
                direction=str(getattr(raw_dir, "value", raw_dir)).upper(),
                stake=stake,
                expiry_minutes=exp,
                payout=payout,
                opened_at=item.get("opened_at"),
                status=str(item.get("status", "OPEN")),
                decision_id=item.get("decision_id"),
            )
        # Object
        t_id = str(getattr(item, "id", getattr(item, "trade_id", "")))
        asset = str(getattr(item, "asset", getattr(item, "asset_id", "")))
        raw_dir = getattr(item, "direction", "")
        raw_stk = getattr(item, "stake", getattr(item, "stake_amount", 0.0))
        try:
            stake = float(raw_stk)
        except (ValueError, TypeError):
            stake = float("nan")
        raw_exp = getattr(item, "expiry_minutes", getattr(item, "expiry_min", 0))
        try:
            exp = int(raw_exp)
        except (ValueError, TypeError):
            exp = 0
        payout = float(getattr(item, "payout", 0.85))
        return cls(
            id=t_id,
            asset=asset,
            direction=str(getattr(raw_dir, "value", raw_dir)).upper(),
            stake=stake,
            expiry_minutes=exp,
            payout=payout,
            opened_at=getattr(item, "opened_at", None),
            status=str(getattr(item, "status", "OPEN")),
            decision_id=getattr(item, "decision_id", None),
        )


def to_canonical_trade(item: Any) -> CanonicalTrade:
    """Normalize any trade-like structure (dict or object) into a CanonicalTrade."""
    return CanonicalTrade.from_any(item)


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
