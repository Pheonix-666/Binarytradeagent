"""Tests for shared core models."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from core.models import (
    AssetQuote,
    TradeDirection,
    TradeResult,
    TrendLabel,
    TradeReceipt,
    Trade,
    TrendReport,
    PatternMatch,
    RiskReport,
    GateResult,
    Candidate,
)


def test_asset_quote_validation():
    quote = AssetQuote(
        asset_id="EURUSD",
        name="EUR/USD",
        is_otc=False,
        payout=0.85,
        is_open=True,
    )
    assert quote.payout == 0.85

    with pytest.raises(ValidationError):
        AssetQuote(
            asset_id="EURUSD",
            name="EUR/USD",
            is_otc=False,
            payout=1.5,  # payout must be <= 1.0
            is_open=True,
        )


def test_trade_and_receipt():
    now = datetime.now(timezone.utc)
    receipt = TradeReceipt(
        trade_id="T001",
        asset_id="EURUSD",
        direction=TradeDirection.CALL,
        expiry_minutes=30,
        stake_amount=50.0,
        entry_price=1.0850,
        entry_time=now,
    )
    assert receipt.direction == TradeDirection.CALL

    trade = Trade(
        **receipt.model_dump(),
        result=TradeResult.PENDING,
    )
    assert trade.result == TradeResult.PENDING
    assert trade.exit_price is None


def test_candidate_creation():
    candidate = Candidate(
        asset="EURUSD",
        direction=TradeDirection.CALL,
        expiry_minutes=30,
        payout=0.85,
        trend=TrendReport(
            daily=TrendLabel.UP,
            hourly=TrendLabel.UP,
            fifteen_min=TrendLabel.UP,
        ),
        volatility_percentile=55.0,
        nearest_level={"type": "support", "distance_atr": 1.8},
        pattern=PatternMatch(
            pattern_name="double_bottom",
            direction=TradeDirection.CALL,
            pattern_type="reversal",
            similarity=82.0,
        ),
        risk_score=38.0,
        max_stake_pct=13.4,
        asset_stats={"trades": 240, "win_rate": 0.57, "mode": "live_eligible"},
    )
    assert candidate.direction == TradeDirection.CALL
    assert candidate.risk_score == 38.0

