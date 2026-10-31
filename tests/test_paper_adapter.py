"""Unit tests for PaperAdapter execution, trade simulation, and settlement."""

from datetime import datetime, timedelta, timezone
import pytest

from adapters.paper import PaperAdapter
from core.models import AssetQuote, TradeDirection, TradeResult


@pytest.fixture
def paper_adapter():
    adapter = PaperAdapter(initial_balance=1000.0, default_payout=0.85)
    quote = AssetQuote(
        asset_id="EURUSD",
        name="EUR/USD",
        is_otc=False,
        payout=0.85,
        is_open=True,
    )
    adapter.set_assets([quote])
    return adapter


def test_paper_adapter_initial_state(paper_adapter):
    assert paper_adapter.get_balance() == 1000.0
    assert paper_adapter.is_healthy() is True
    assert len(paper_adapter.get_open_trades()) == 0
    assert len(paper_adapter.get_trade_history()) == 0


def test_paper_adapter_place_trade_balance_deducted(paper_adapter):
    receipt = paper_adapter.place_trade(
        asset_id="EURUSD",
        direction=TradeDirection.CALL,
        expiry_min=15,
        stake_amount=100.0,
        payout=0.85,
    )
    assert receipt.stake_amount == 100.0
    assert receipt.direction == TradeDirection.CALL
    assert receipt.payout == 0.85
    assert paper_adapter.get_balance() == 900.0

    open_trades = paper_adapter.get_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].result == TradeResult.PENDING


def test_paper_adapter_payout_required(paper_adapter):
    """PaperAdapter.place_trade must raise if payout is None or <= 0."""
    with pytest.raises(ValueError, match="payout must be positive"):
        paper_adapter.place_trade(
            asset_id="EURUSD",
            direction=TradeDirection.CALL,
            expiry_min=15,
            stake_amount=100.0,
            payout=None,  # type: ignore
        )

    with pytest.raises(ValueError, match="payout must be positive"):
        paper_adapter.place_trade(
            asset_id="EURUSD",
            direction=TradeDirection.CALL,
            expiry_min=15,
            stake_amount=100.0,
            payout=-0.1,
        )


def test_paper_adapter_insufficient_balance(paper_adapter):
    with pytest.raises(ValueError, match="Insufficient balance"):
        paper_adapter.place_trade(
            asset_id="EURUSD",
            direction=TradeDirection.CALL,
            expiry_min=15,
            stake_amount=1500.0,
            payout=0.85,
        )


def test_paper_adapter_negative_stake(paper_adapter):
    with pytest.raises(ValueError, match="stake_amount must be positive"):
        paper_adapter.place_trade(
            asset_id="EURUSD",
            direction=TradeDirection.CALL,
            expiry_min=15,
            stake_amount=-10.0,
            payout=0.85,
        )


def test_paper_adapter_settle_call_win(paper_adapter):
    """CALL trade wins when exit_price > entry_price; balance credited with profit."""
    start_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    paper_adapter.set_simulated_time(start_time)

    receipt = paper_adapter.place_trade(
        asset_id="EURUSD",
        direction=TradeDirection.CALL,
        expiry_min=15,
        stake_amount=100.0,
        payout=0.85,
    )
    # Entry price is 1.0 (from fallback dummy price)
    assert receipt.entry_price == 1.0
    assert paper_adapter.get_balance() == 900.0

    # Advance time to expiry and provide higher price (win)
    expiry_time = start_time + timedelta(minutes=15)
    settled = paper_adapter.settle_open_trades(
        current_time=expiry_time,
        prices_at_expiry={"EURUSD": 1.05},
    )

    assert len(settled) == 1
    trade = settled[0]
    assert trade.result == TradeResult.WIN
    assert trade.payout_received == 100.0 * (1.0 + 0.85)  # 185.0
    assert paper_adapter.get_balance() == 900.0 + 185.0  # 1085.0
    assert len(paper_adapter.get_open_trades()) == 0
    assert len(paper_adapter.get_trade_history()) == 1


def test_paper_adapter_settle_call_loss(paper_adapter):
    """CALL trade loses when exit_price < entry_price."""
    start_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    paper_adapter.set_simulated_time(start_time)

    paper_adapter.place_trade(
        asset_id="EURUSD",
        direction=TradeDirection.CALL,
        expiry_min=30,
        stake_amount=100.0,
        payout=0.85,
    )
    assert paper_adapter.get_balance() == 900.0

    expiry_time = start_time + timedelta(minutes=30)
    settled = paper_adapter.settle_open_trades(
        current_time=expiry_time,
        prices_at_expiry={"EURUSD": 0.95},
    )

    assert len(settled) == 1
    assert settled[0].result == TradeResult.LOSS
    assert settled[0].payout_received == 0.0
    assert paper_adapter.get_balance() == 900.0


def test_paper_adapter_settle_put_win(paper_adapter):
    """PUT trade wins when exit_price < entry_price."""
    start_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    paper_adapter.set_simulated_time(start_time)

    paper_adapter.place_trade(
        asset_id="EURUSD",
        direction=TradeDirection.PUT,
        expiry_min=15,
        stake_amount=200.0,
        payout=0.85,
    )
    assert paper_adapter.get_balance() == 800.0

    expiry_time = start_time + timedelta(minutes=15)
    settled = paper_adapter.settle_open_trades(
        current_time=expiry_time,
        prices_at_expiry={"EURUSD": 0.98},
    )

    assert len(settled) == 1
    assert settled[0].result == TradeResult.WIN
    assert settled[0].payout_received == 200.0 * 1.85  # 370.0
    assert paper_adapter.get_balance() == 800.0 + 370.0  # 1170.0


def test_paper_adapter_settle_put_loss(paper_adapter):
    """PUT trade loses when exit_price > entry_price."""
    start_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    paper_adapter.set_simulated_time(start_time)

    paper_adapter.place_trade(
        asset_id="EURUSD",
        direction=TradeDirection.PUT,
        expiry_min=15,
        stake_amount=100.0,
        payout=0.85,
    )

    expiry_time = start_time + timedelta(minutes=15)
    settled = paper_adapter.settle_open_trades(
        current_time=expiry_time,
        prices_at_expiry={"EURUSD": 1.05},
    )

    assert len(settled) == 1
    assert settled[0].result == TradeResult.LOSS
    assert paper_adapter.get_balance() == 900.0


def test_paper_adapter_settle_tie_refunds_stake(paper_adapter):
    """Tie (exit_price == entry_price) refunds stake amount when tie_policy is refund."""
    paper_adapter.tie_policy = "refund"
    start_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    paper_adapter.set_simulated_time(start_time)

    paper_adapter.place_trade(
        asset_id="EURUSD",
        direction=TradeDirection.CALL,
        expiry_min=15,
        stake_amount=100.0,
        payout=0.85,
    )

    expiry_time = start_time + timedelta(minutes=15)
    settled = paper_adapter.settle_open_trades(
        current_time=expiry_time,
        prices_at_expiry={"EURUSD": 1.0},  # Same as entry
    )

    assert len(settled) == 1
    assert settled[0].payout_received == 100.0
    assert paper_adapter.get_balance() == 1000.0  # Full refund

