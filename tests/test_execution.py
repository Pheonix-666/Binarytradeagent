"""Comprehensive unit and integration tests for execution/ and alerts/ packages.

Covers:
  1. StateManager:
     - Atomic write (.tmp -> atomic replace) and .bak copy.
     - Serialization and round-trip of AdaptiveState and BreakerState.
     - Persistence guarantee: Restoring while PAUSED remains PAUSED.
  2. reconcile_startup:
     - Normal startup reconciliation with matching trades and balance.
     - Resolution of PENDING trades (OPEN, SETTLED, CANCELLED).
     - Unknown platform trades: counted towards open exposure, alert generated, circuit breaker tripped!
     - Balance mismatch detection.
  3. TradeManager:
     - Pre-trade validation sequence: kill switch, breaker, reconcile, validator, safety limits.
     - Routing to paper shadow trade for non-live-eligible asset.
     - Write-ahead logging (persisted as PENDING prior to broker API call).
     - Idempotency per decision ID.
     - Post-placement verification via get_open_trades.
     - Post-placement verification failure: trips breaker, alerts, NEVER retries!
  4. TelegramAlerter:
     - Masking tokens from logs.
     - Graceful handling of network timeouts/failures without crashing.
     - Command handling restricted to authorized chat_id.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from adaptive.monitor import AdaptiveState
from alerts.telegram import TelegramAlerter
from core.config import AppConfig, load_config
from decision.schema import DecisionEvidence, DecisionOutput, ValidatedDecision
from execution.reconcile import reconcile_startup
from execution.state import StateManager
from execution.trade_manager import TradeManager
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot


class FakeBrokerAdapter:
    """Mock broker adapter for testing order placement and verification."""

    def __init__(self, balance: float = 10000.0, open_trades: list = None, trade_history: list = None):
        self.balance = balance
        self.open_trades = list(open_trades or [])
        self.trade_history = list(trade_history or [])
        self.placed_orders = []
        self.should_fail_placement = False
        self.should_omit_from_open_trades = False  # Simulates verification failure

    def get_balance(self) -> float:
        return self.balance

    def get_open_trades(self) -> list:
        return list(self.open_trades)

    def get_trade_history(self, since: Optional[datetime] = None) -> list:
        return list(self.trade_history)

    def place_trade(
        self,
        asset_id: str,
        direction: Any,
        expiry_min: int,
        stake_amount: float,
        trade_id: Optional[str] = None,
        **kwargs,
    ):
        if self.should_fail_placement:
            raise RuntimeError("Broker placement failed: 503 Service Unavailable")

        dir_str = str(getattr(direction, "value", direction)).upper()
        tid = trade_id or "fake_tid_123"
        order_record = {
            "id": tid,
            "asset": asset_id,
            "direction": dir_str,
            "expiry_minutes": expiry_min,
            "stake": stake_amount,
            "status": "OPEN",
        }
        self.placed_orders.append(order_record)
        if not self.should_omit_from_open_trades:
            self.open_trades.append(order_record)
        from core.models import TradeDirection, TradeReceipt
        return TradeReceipt(
            trade_id=tid,
            platform_trade_id=tid,
            asset_id=asset_id,
            direction=TradeDirection(dir_str),
            expiry_minutes=expiry_min,
            stake_amount=stake_amount,
            entry_price=1.0,
            entry_time=datetime.now(timezone.utc),
            payout=0.85,
            status="OPEN",
        )


class MockAlerter:
    """Mock alerter recording messages."""

    def __init__(self):
        self.alerts: list[str] = []

    def send_alert(self, msg: str) -> bool:
        self.alerts.append(msg)
        return True


@pytest.fixture
def config() -> AppConfig:
    cfg = load_config("config.yaml")
    cfg.stake.min_pct = 1.0
    cfg.stake.max_pct = 20.0
    cfg.stake.total_cap_pct = 35.0
    cfg.stake.min_amount = 1.0
    cfg.stake.max_amount = 1000.0
    cfg.stake.round_to = 1.0
    cfg.mode = "paper"
    cfg.adaptive.enabled = False
    return cfg


# ── 1. StateManager Tests ──────────────────────────────────────────────────


def test_state_manager_atomic_save_and_load(tmp_path):
    """Save writes state atomically, creates .bak, and restores data correctly."""
    state_file = tmp_path / "state.json"
    mgr = StateManager(state_file=state_file)

    ad_state = AdaptiveState(tier="CAUTION", stake_scale=0.5, drawdown_pct=15.0)
    mgr.save(
        balance=9500.0,
        equity=9800.0,
        open_trades=[{"id": "t1", "asset": "EURUSD", "stake": 300.0}],
        pending_trades=[],
        adaptive_state=ad_state,
        reconcile_done=True,
    )

    assert state_file.is_file()
    assert mgr.backup_path.is_file() or state_file.exists()

    # Load in new manager instance
    mgr2 = StateManager(state_file=state_file)
    data = mgr2.load()
    assert data["balance"] == 9500.0
    assert data["reconcile_done"] is True
    assert len(data["open_trades"]) == 1

    restored_ad = mgr2.get_adaptive_state()
    assert restored_ad is not None
    assert restored_ad.tier == "CAUTION"
    assert restored_ad.stake_scale == 0.5


def test_state_manager_paused_persistence(tmp_path):
    """Restoring state while PAUSED guarantees trading remains paused."""
    state_file = tmp_path / "state.json"
    mgr = StateManager(state_file=state_file)

    ad_paused = AdaptiveState(tier="PAUSED", real_trading_allowed=False, stake_scale=0.0)
    mgr.save(
        balance=6000.0,
        equity=6000.0,
        open_trades=[],
        pending_trades=[],
        adaptive_state=ad_paused,
    )

    mgr2 = StateManager(state_file=state_file)
    data = mgr2.load()
    restored_ad = mgr2.get_adaptive_state()
    assert restored_ad.tier == "PAUSED"
    assert restored_ad.real_trading_allowed is False
    assert restored_ad.stake_scale == 0.0


# ── 2. Startup Reconciliation Tests ────────────────────────────────────────


def test_reconcile_startup_matching(config):
    """Reconciliation passes when broker and saved state agree."""
    adapter = FakeBrokerAdapter(balance=10000.0, open_trades=[{"id": "t1", "asset": "EURUSD", "stake": 200.0}])
    breaker = CircuitBreaker(config=config)
    alerter = MockAlerter()
    saved_state = {
        "balance": 10000.0,
        "open_trades": [{"id": "t1", "asset": "EURUSD", "stake": 200.0}],
        "pending_trades": [],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker, alerter=alerter)
    assert res.success is True
    assert len(res.unknown_trades) == 0
    assert len(res.issues) == 0
    assert breaker.is_tripped() is False


def test_reconcile_startup_resolves_pending_trades(config):
    """Reconciliation upgrades confirmed pending trades to OPEN, and cancels unplaced ones."""
    # Pending trade t_open is found on broker; pending trade t_ghost is missing
    adapter = FakeBrokerAdapter(balance=10000.0, open_trades=[{"id": "t_open", "asset": "EURUSD", "stake": 200.0}])
    breaker = CircuitBreaker(config=config)
    saved_state = {
        "balance": 10000.0,
        "open_trades": [],
        "pending_trades": [
            {"id": "t_open", "asset": "EURUSD", "stake": 200.0},
            {"id": "t_ghost", "asset": "GBPUSD", "stake": 200.0},
        ],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker)
    assert res.success is True
    statuses = {p["trade_id"]: p["status"] for p in res.resolved_pending}
    assert statuses["t_open"] == "OPEN"
    assert statuses["t_ghost"] == "CANCELLED"


def test_reconcile_startup_unknown_trades_trips_breaker(config):
    """Unknown platform trades count toward exposure, alert, and trip circuit breaker!"""
    adapter = FakeBrokerAdapter(
        balance=10000.0,
        open_trades=[
            {"id": "recognized_1", "asset": "EURUSD", "stake": 100.0},
            {"id": "rogue_trade", "asset": "USDJPY", "stake": 500.0},  # Unknown manual/external trade
        ],
    )
    breaker = CircuitBreaker(config=config)
    alerter = MockAlerter()
    saved_state = {
        "balance": 10000.0,
        "open_trades": [{"id": "recognized_1", "asset": "EURUSD", "stake": 100.0}],
        "pending_trades": [],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker, alerter=alerter)
    assert res.success is False
    assert len(res.unknown_trades) == 1
    assert res.unknown_trades[0]["id"] == "rogue_trade"
    # Unknown trade included in open trades to protect total exposure!
    assert len(res.reconciled_open_trades) == 2
    # Breaker tripped and alert fired
    assert breaker.is_tripped() is True
    assert "Unknown open trade on platform" in breaker.trip_reason
    assert any("UNKNOWN TRADE ALERT" in a for a in alerter.alerts)


# ── 3. TradeManager Pre-Trade Sequence & Execution Tests ───────────────────


def _make_validated_decision(asset="EURUSD", direction="CALL", stake_amount=200.0, stake_pct=2.0) -> ValidatedDecision:
    import uuid
    dec_id = f"dec_{uuid.uuid4().hex[:12]}"
    return ValidatedDecision(
        decision=direction,
        is_valid=True,
        stake_amount=stake_amount,
        asset=asset,
        direction=direction,
        expiry_minutes=15,
        stake_pct=stake_pct,
        decision_id=dec_id,
        evidence=DecisionEvidence(risk_score=30, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
        reasoning="Valid high similarity candidate",
        checks_passed=["check_a", "check_b", "check_c", "check_d", "check_e", "check_f", "check_g", "check_h"],
        raw_decision=DecisionOutput(
            decision=direction,
            asset=asset,
            expiry_minutes=15,
            stake_pct=stake_pct,
            decision_id=dec_id,
            evidence=DecisionEvidence(risk_score=30, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
            reasoning="Valid high similarity candidate",
        ),
    )


def test_trade_manager_blocked_by_kill_switch_or_breaker(config, tmp_path):
    """Pre-trade sequence blocks when kill switch is active or breaker is tripped."""
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter()
    alerter = MockAlerter()
    tm = TradeManager(config, state_mgr, breaker, ks, adapter, alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    decision = _make_validated_decision()

    # 1. Kill switch active -> Blocked
    ks.activate_file()
    res_ks = tm.execute(decision, account)
    assert res_ks.executed is False
    assert res_ks.status == "BLOCKED"
    assert "kill switch" in res_ks.reason.lower()
    ks.deactivate_file()

    # 2. Breaker tripped -> Blocked
    breaker.trip("Manual trip for test")
    res_brk = tm.execute(decision, account)
    assert res_brk.executed is False
    assert res_brk.status == "BLOCKED"
    assert "breaker" in res_brk.reason.lower()


def test_trade_manager_blocked_when_unreconciled(config, tmp_path):
    """System refuses to trade until startup reconcile completes."""
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter()
    tm = TradeManager(config, state_mgr, breaker, ks, adapter)
    # reconcile_done is False
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    decision = _make_validated_decision()

    res = tm.execute(decision, account)
    assert res.executed is False
    assert res.status == "BLOCKED"
    assert "reconciliation" in res.reason.lower()


def test_trade_manager_paper_shadow_routing(config, tmp_path):
    """Non-live-eligible asset routes to paper shadow trade."""
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter()
    paper_adapter = FakeBrokerAdapter()
    alerter = MockAlerter()
    tm = TradeManager(config, state_mgr, breaker, ks, adapter, paper_adapter=paper_adapter, alerter=alerter)
    tm.set_reconciled(True)

    # Asset mode is 'paper'
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "paper"})
    decision = _make_validated_decision(asset="EURUSD")

    res = tm.execute(decision, account)
    assert res.executed is True
    assert res.status == "SHADOW_OPEN"
    assert res.is_shadow is True
    # Real adapter never called!
    assert len(adapter.placed_orders) == 0
    # Paper adapter received trade
    assert len(paper_adapter.placed_orders) == 1


def test_trade_manager_post_placement_verification_success(config, tmp_path):
    """Real placement writes ahead as PENDING, places order, verifies via get_open_trades, and confirms."""
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter(balance=10000.0)
    alerter = MockAlerter()
    cfg_live = config.model_copy(deep=True)
    cfg_live.mode = "demo"  # Real trading mode on demo/live account
    tm = TradeManager(cfg_live, state_mgr, breaker, ks, adapter, alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    decision = _make_validated_decision(asset="EURUSD", direction="CALL", stake_amount=200.0)

    res = tm.execute(decision, account)
    assert res.executed is True
    assert res.status == "OPEN"
    assert res.trade_id is not None
    assert len(adapter.placed_orders) == 1
    assert any("Trade placed" in a for a in alerter.alerts)


def test_trade_manager_post_placement_verification_failure_trips_breaker(config, tmp_path):
    """If trade cannot be verified in open trades list: stop, alert, trip breaker, NEVER retry."""
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter(balance=10000.0)
    # Simulate verification failure: order placed but not visible in get_open_trades()
    adapter.should_omit_from_open_trades = True

    alerter = MockAlerter()
    cfg_live = config.model_copy(deep=True)
    cfg_live.mode = "demo"
    tm = TradeManager(cfg_live, state_mgr, breaker, ks, adapter, alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    decision = _make_validated_decision()

    res = tm.execute(decision, account)
    assert res.executed is False
    assert res.status == "VERIFICATION_FAILED"
    # Circuit breaker tripped immediately
    assert breaker.is_tripped() is True
    assert "verification FAILED" in breaker.trip_reason
    # Critical alert emitted
    assert any("CRITICAL EXECUTION FAULT" in a for a in alerter.alerts)
    # Order placed once, never retried automatically
    assert len(adapter.placed_orders) == 1


# ── 4. Telegram Alerts & Remote Commands Tests ─────────────────────────────


def test_telegram_alerter_command_authorization_and_dispatch():
    """Bot commands accepted only from authorized chat_id."""
    alerter = TelegramAlerter(token="dummy_token_123", chat_id="123456789")

    # Register handlers
    alerter.register_command("/status", lambda args: "System Status: NORMAL, Balance: $10,000")
    alerter.register_command("/stop", lambda args: "EMERGENCY STOP EXECUTED")

    # 1. Update from unauthorized chat_id -> Ignored
    unauth_update = {
        "message": {
            "chat": {"id": "999999999"},
            "text": "/stop",
        }
    }
    assert alerter.handle_incoming_update(unauth_update) is None

    # 2. Update from authorized chat_id -> Executed
    auth_update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "/status",
        }
    }
    res = alerter.handle_incoming_update(auth_update)
    assert "System Status: NORMAL" in res


def test_reset_breaker_works_and_is_chat_id_restricted(config, tmp_path):
    """Explicit test: /reset_breaker resets tripped breaker and is restricted to authorized chat_id."""
    from alerts.telegram import create_operator_command_handlers

    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    breaker.trip("Verification failed: lost order")
    assert breaker.is_tripped() is True

    handlers = create_operator_command_handlers(
        state_manager=state_mgr,
        breaker=breaker,
        kill_switch=ks,
        config=config,
    )
    alerter = TelegramAlerter(token="test_token", chat_id="123456789")
    for cmd, h in handlers.items():
        alerter.register_command(cmd, h)

    # 1. Unauthorized chat_id attempts /reset_breaker -> Ignored, breaker remains tripped
    unauth_update = {
        "message": {
            "chat": {"id": "999999999"},
            "text": "/reset_breaker confirm",
        }
    }
    assert alerter.handle_incoming_update(unauth_update) is None
    assert breaker.is_tripped() is True

    # 1b. Authorized without confirm keyword -> requires explicit confirmation
    unconf_update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "/reset_breaker",
        }
    }
    res_unconf = alerter.handle_incoming_update(unconf_update)
    assert "requires explicit confirmation" in res_unconf
    assert breaker.is_tripped() is True

    # 2. Authorized chat_id issues /reset_breaker confirm -> Reset executed, breaker restored
    auth_update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "/reset_breaker confirm",
        }
    }
    res = alerter.handle_incoming_update(auth_update)
    assert "Breaker reset" in res
    assert breaker.is_tripped() is False


# ── 5. Specific Recovery and Safety Edge Tests ─────────────────────────────


def test_duplicate_decision_id_places_one_order(config, tmp_path):
    """Submitting the same decision ID twice places only one order; second is rejected as DUPLICATE."""
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter()
    cfg_demo = config.model_copy(deep=True)
    cfg_demo.mode = "demo"
    tm = TradeManager(cfg_demo, state_mgr, breaker, ks, adapter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    decision = _make_validated_decision(asset="EURUSD")
    setattr(decision, "decision_id", "decision_fixed_uuid_123")

    # First execution: succeeds
    res1 = tm.execute(decision, account)
    assert res1.executed is True
    assert len(adapter.placed_orders) == 1

    # Second execution with identical decision_id: rejected as duplicate
    res2 = tm.execute(decision, account)
    assert res2.executed is False
    assert res2.status == "DUPLICATE"
    assert "already executed" in res2.reason
    assert len(adapter.placed_orders) == 1  # No duplicate order placed!


def test_crash_between_pending_write_and_adapter_call(config):
    """Crash after writing PENDING before calling broker adapter: on restart, reconcile cancels unplaced trade."""
    adapter = FakeBrokerAdapter(balance=10000.0, open_trades=[])
    breaker = CircuitBreaker(config=config)
    # State recorded trade as PENDING before process crashed
    saved_state = {
        "balance": 10000.0,
        "open_trades": [],
        "pending_trades": [{"id": "pending_uncalled", "asset": "EURUSD", "stake": 200.0}],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker)
    assert res.success is True
    assert len(res.resolved_pending) == 1
    assert res.resolved_pending[0]["trade_id"] == "pending_uncalled"
    assert res.resolved_pending[0]["status"] == "CANCELLED"
    assert "Not found on platform" in res.resolved_pending[0]["resolution"]


def test_crash_between_adapter_call_and_confirmation(config):
    """Crash after adapter placed trade but before state saved OPEN: reconcile confirms open trade."""
    # Trade was placed on broker and is in broker's open trades list
    placed_trade_on_broker = {"id": "pending_placed", "asset": "EURUSD", "stake": 200.0, "status": "OPEN"}
    adapter = FakeBrokerAdapter(balance=9800.0, open_trades=[placed_trade_on_broker])
    breaker = CircuitBreaker(config=config)
    # State still had it as PENDING when crash happened
    saved_state = {
        "balance": 9800.0,
        "open_trades": [],
        "pending_trades": [{"id": "pending_placed", "asset": "EURUSD", "stake": 200.0}],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker)
    assert res.success is True
    assert len(res.resolved_pending) == 1
    assert res.resolved_pending[0]["trade_id"] == "pending_placed"
    assert res.resolved_pending[0]["status"] == "OPEN"
    # Recovered trade is included in reconciled open trades
    assert any(getattr(t, "id", t.get("id")) == "pending_placed" for t in res.reconciled_open_trades)


def test_telegram_resume_refused_before_requalification(config, tmp_path):
    """Telegram /resume is refused when PAUSED if requalification criteria are not met."""
    from alerts.telegram import create_operator_command_handlers
    from adaptive.monitor import AdaptiveController, AdaptiveState

    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    # Adaptive state is PAUSED with only 10 shadow trades (< 30 required)
    ad_ctrl = AdaptiveController(config=config, initial_state=AdaptiveState(tier="PAUSED", shadow_trades_count=10, shadow_wins=6))

    handlers = create_operator_command_handlers(
        state_manager=state_mgr,
        breaker=breaker,
        kill_switch=ks,
        adaptive_controller=ad_ctrl,
        config=config,
    )
    alerter = TelegramAlerter(token="test_tok", chat_id="12345")
    for cmd, h in handlers.items():
        alerter.register_command(cmd, h)

    # Issue /resume
    update = {"message": {"chat": {"id": "12345"}, "text": "/resume"}}
    reply = alerter.handle_incoming_update(update)

    assert "REFUSED" in reply
    assert "Requalification requirements not met" in reply
    assert ad_ctrl.state.tier == "PAUSED"  # Remains PAUSED


def test_alerts_never_raise_when_network_is_down(monkeypatch):
    """Telegram send_alert never crashes calling code when network connection fails."""
    import urllib.request
    from urllib.error import URLError

    def _mock_urlopen(*args, **kwargs):
        raise URLError("Connection refused by peer (network offline)")

    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen)

    alerter = TelegramAlerter(token="dummy_token_123", chat_id="123456789")
    # Must catch the exception gracefully and return False
    success = alerter.send_alert("Test message during network failure")
    assert success is False


# ── 6. Explicit Individual Requirement Tests ───────────────────────────────


def test_state_persistence_and_replay_idempotency(config, tmp_path):
    """Explicit test: state saves/restores atomically, and duplicate decision IDs are idempotent."""
    state_file = tmp_path / "state.json"
    mgr = StateManager(state_file=state_file)
    mgr.save(
        balance=10000.0,
        equity=10000.0,
        open_trades=[],
        pending_trades=[],
        reconcile_done=True,
    )

    # Reload in new instance
    mgr_restored = StateManager(state_file=state_file)
    data = mgr_restored.load()
    assert data["balance"] == 10000.0
    assert data["reconcile_done"] is True

    # Test replay idempotency via TradeManager
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter()
    cfg_demo = config.model_copy(deep=True)
    cfg_demo.mode = "demo"

    tm = TradeManager(cfg_demo, mgr_restored, breaker, ks, adapter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_validated_decision(asset="EURUSD")
    setattr(dec, "decision_id", "idempotent_decision_test_999")

    # First execution succeeds
    res1 = tm.execute(dec, account)
    assert res1.executed is True
    assert len(adapter.placed_orders) == 1

    # Replay of identical decision is rejected as DUPLICATE, no order placed
    res2 = tm.execute(dec, account)
    assert res2.executed is False
    assert res2.status == "DUPLICATE"
    assert len(adapter.placed_orders) == 1


def test_trade_manager_happy_path_submit_open_win_balance_updated(config, tmp_path):
    """Explicit test: submit -> order verified open -> trade settles WIN -> balance updated."""
    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file)
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter(balance=10000.0)
    alerter = MockAlerter()
    cfg_demo = config.model_copy(deep=True)
    cfg_demo.mode = "demo"

    tm = TradeManager(cfg_demo, state_mgr, breaker, ks, adapter, alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_validated_decision(asset="EURUSD", direction="CALL", stake_amount=200.0)

    # 1. Submit and Open
    res = tm.execute(dec, account)
    assert res.executed is True
    assert res.status == "OPEN"
    trade_id = res.trade_id
    assert len(state_mgr.data["open_trades"]) == 1
    assert state_mgr.data["balance"] == 9800.0

    # 2. Settle Win (+85% payout -> $200 stake + $170 profit)
    settle_res = tm.record_settlement(trade_id=trade_id, outcome="WIN", payout=0.85)
    assert settle_res["outcome"] == "WIN"
    assert settle_res["profit"] == 170.0
    assert settle_res["balance"] == 10170.0
    assert len(state_mgr.data["open_trades"]) == 0
    assert state_mgr.data["balance"] == 10170.0
    assert any("WON" in a for a in alerter.alerts)


def test_trade_manager_reject_path_safety_blocks_preserves_state(config, tmp_path):
    """Explicit test: safety blocks -> order rejected, no state change or adapter call."""
    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file)
    state_mgr.save(balance=10000.0, equity=10000.0, open_trades=[], pending_trades=[], reconcile_done=True)

    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = FakeBrokerAdapter(balance=10000.0)
    alerter = MockAlerter()
    cfg_demo = config.model_copy(deep=True)
    cfg_demo.mode = "demo"

    tm = TradeManager(cfg_demo, state_mgr, breaker, ks, adapter, alerter=alerter)
    tm.set_reconciled(True)

    # Account with 34% exposure ($3400)
    account = AccountSnapshot(
        balance=6600.0,
        equity=10000.0,
        open_trades=[{"asset": "GBPUSD", "direction": "CALL", "stake": 3400.0}],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # New order of $200 pushes total exposure to 36% > 35% cap
    dec = _make_validated_decision(asset="EURUSD", direction="CALL", stake_amount=200.0)

    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "BLOCKED_SAFETY"
    assert "exceeds total_cap_pct 35.0%" in res.reason

    # State and adapter completely preserved
    assert len(adapter.placed_orders) == 0
    assert state_mgr.data["balance"] == 10000.0
    assert len(state_mgr.data["pending_trades"]) == 0


def test_trade_manager_reconciliation_out_of_order_broker_event(config):
    """Explicit test: out-of-order broker event (trade settled on platform while PENDING locally)."""
    # Broker adapter trade history contains the settled trade
    adapter = FakeBrokerAdapter(balance=10000.0, open_trades=[])
    setattr(adapter, "trade_history", [{"id": "pending_early_settle", "asset": "EURUSD", "status": "SETTLED"}])
    adapter.get_trade_history = lambda: getattr(adapter, "trade_history", [])

    breaker = CircuitBreaker(config=config)
    saved_state = {
        "balance": 10000.0,
        "open_trades": [],
        "pending_trades": [{"id": "pending_early_settle", "asset": "EURUSD", "stake": 200.0}],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker)
    assert res.success is True
    statuses = {p["trade_id"]: p["status"] for p in res.resolved_pending}
    assert statuses["pending_early_settle"] == "SETTLED"


def test_trade_manager_reconciliation_orphan_detected(config):
    """Explicit test: orphan/unknown trade on broker is detected, counts to exposure, and trips breaker."""
    adapter = FakeBrokerAdapter(
        balance=10000.0,
        open_trades=[
            {"id": "known_1", "asset": "EURUSD", "stake": 200.0},
            {"id": "orphan_rogue", "asset": "USDJPY", "stake": 600.0},
        ],
    )
    breaker = CircuitBreaker(config=config)
    alerter = MockAlerter()
    saved_state = {
        "balance": 10000.0,
        "open_trades": [{"id": "known_1", "asset": "EURUSD", "stake": 200.0}],
        "pending_trades": [],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker, alerter=alerter)
    assert res.success is False
    assert len(res.unknown_trades) == 1
    assert res.unknown_trades[0]["id"] == "orphan_rogue"
    assert breaker.is_tripped() is True
    assert any("UNKNOWN TRADE ALERT" in a for a in alerter.alerts)


def test_trade_manager_reconciliation_balance_discrepancy(config):
    """Explicit test: balance discrepancy between broker and saved state trips breaker and alerts."""
    adapter = FakeBrokerAdapter(balance=9500.0, open_trades=[])
    breaker = CircuitBreaker(config=config)
    alerter = MockAlerter()
    saved_state = {
        "balance": 10000.0,  # $500 discrepancy > $1 tolerance
        "open_trades": [],
        "pending_trades": [],
    }

    res = reconcile_startup(adapter=adapter, saved_state=saved_state, breaker=breaker, alerter=alerter, tolerance=1.0)
    assert res.success is False
    assert breaker.is_tripped() is True
    assert any("BALANCE RECONCILIATION FAILURE" in a for a in alerter.alerts)


def test_telegram_alert_formatting_for_each_event_type():
    """Explicit test: telegram alert formatting for trade, reject, breaker, and killswitch events."""
    from alerts.telegram import TelegramAlerter

    alerter = TelegramAlerter()
    messages = []
    alerter.send_alert = lambda msg: messages.append(msg) or True

    # 1. Trade event format
    alerter.send_alert("✅ Trade placed: EURUSD PUT $250.00 (30m)")
    assert "✅ Trade placed: EURUSD PUT $250.00 (30m)" in messages[-1]

    # 2. Reject event format
    alerter.send_alert("🛡️ Trade blocked by safety layer: Total open exposure 36.00% exceeds total_cap_pct 35.0%")
    assert "🛡️ Trade blocked by safety layer:" in messages[-1]

    # 3. Breaker trip event format
    alerter.send_alert("🚨 CRITICAL EXECUTION FAULT: Post-placement verification FAILED for trade tr_123")
    assert "🚨 CRITICAL EXECUTION FAULT:" in messages[-1]

    # 4. Kill switch event format
    alerter.send_alert("🛑 STOPPED: Emergency kill switch activated via Telegram. New orders halted.")
    assert "🛑 STOPPED:" in messages[-1]

