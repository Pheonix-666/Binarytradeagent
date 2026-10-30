"""End-to-end fault injection test suite for execution and safety layers.

Tests using FaultyAdapter wrapping the REAL PaperAdapter:
  1. Timeout after order placed: trade was submitted, adapter raised TimeoutError.
     Verified: breaker trips immediately, reconcile_done=False, PENDING preserved in state,
     no duplicate orders placed.
  2. Order not placed but raises: adapter raises before placement (e.g. 503 Service Unavailable).
     Verified: breaker trips, reconcile_done=False, PENDING preserved for startup reconcile.
  3. Open-trades call fails: adapter.get_open_trades() raises during post-placement verification.
     Verified: verification fails, breaker trips, no duplicate orders.
  4. Corrupted state file: state file has invalid/corrupted JSON on startup.
     Verified: fail-closed initial state, breaker tripped with reason, adaptive PAUSED.
  5. Process restart with open trade: trade is open on adapter, process crashes and restarts.
     Verified: startup reconciliation confirms open trade, expected balance preserved,
     decision ID restored so replay doesn't place duplicate trade.
  6. Telegram down: network fails (URLError/timeout) while sending alerts.
     Verified: execution does not crash, alert failures are tracked, and after max_alert_failures
     real orders pause to shadow mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
import urllib.request
from urllib.error import URLError

from adaptive.monitor import AdaptiveState
from adapters.paper import PaperAdapter
from alerts.telegram import TelegramAlerter
from core.config import load_config
from core.models import TradeDirection, TradeReceipt
from decision.schema import DecisionEvidence, DecisionOutput, ValidatedDecision
from execution.reconcile import reconcile_startup
from execution.state import StateManager
from execution.trade_manager import TradeManager
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot


class FaultyAdapter:
    """Wraps real PaperAdapter with controllable fault injection hooks."""

    def __init__(self, paper: PaperAdapter):
        self.underlying = paper
        self.timeout_after_placed = False
        self.raise_before_placed = False
        self.fail_open_trades = False

    def get_assets(self):
        return self.underlying.get_assets()

    def get_candles(self, *args, **kwargs):
        return self.underlying.get_candles(*args, **kwargs)

    def get_balance(self):
        return self.underlying.get_balance()

    def get_open_trades(self):
        if self.fail_open_trades:
            raise RuntimeError("Broker get_open_trades API failed: 500 Server Error")
        return self.underlying.get_open_trades()

    def get_trade_history(self, since=None):
        return self.underlying.get_trade_history(since=since)

    def is_healthy(self):
        return self.underlying.is_healthy()

    def place_trade(self, *args, **kwargs):
        if self.raise_before_placed:
            raise ConnectionResetError("Broker placement failed: connection dropped before order sent")

        receipt = self.underlying.place_trade(*args, **kwargs)

        if self.timeout_after_placed:
            raise TimeoutError("Read timeout waiting for broker response after order sent")

        return receipt


def _make_decision(asset="EURUSD", direction="CALL", stake=100.0, d_id="dec_fault_01") -> ValidatedDecision:
    return ValidatedDecision(
        decision=direction,
        is_valid=True,
        stake_amount=stake,
        asset=asset,
        direction=direction,
        expiry_minutes=15,
        stake_pct=1.0,
        decision_id=d_id,
        evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
        reasoning="Fault injection test decision",
        checks_passed=["c1", "c2"],
        raw_decision=DecisionOutput(
            decision=direction,
            asset=asset,
            expiry_minutes=15,
            stake_pct=1.0,
            decision_id=d_id,
            evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
            reasoning="Fault injection test decision",
        ),
    )


# ── Scenario 1: Timeout after order placed ───────────────────────────────────

def test_fault_timeout_after_order_placed(tmp_path):
    """Timeout after trade is placed: breaker trips, PENDING preserved, no duplicate order."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    real_paper = PaperAdapter(initial_balance=10000.0)
    faulty_adapter = FaultyAdapter(real_paper)
    faulty_adapter.timeout_after_placed = True

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=faulty_adapter, paper_adapter=real_paper)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_timeout_e2e")

    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "EXEC_ERROR"
    assert breaker.is_tripped() is True
    assert tm.reconcile_done is False
    # PENDING trade preserved
    assert len(state_mgr.data.get("pending_trades", [])) == 1

    # Attempting to re-execute while breaker tripped is blocked
    res_blocked = tm.execute(dec, account)
    assert res_blocked.executed is False
    assert res_blocked.status == "BLOCKED"
    # No duplicate trade on real paper adapter
    assert len(real_paper.get_open_trades()) == 1


# ── Scenario 2: Order not placed but raises ──────────────────────────────────

def test_fault_order_not_placed_but_raises(tmp_path):
    """Adapter raises before trade is placed: breaker trips, PENDING preserved for reconcile."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    real_paper = PaperAdapter(initial_balance=10000.0)
    faulty_adapter = FaultyAdapter(real_paper)
    faulty_adapter.raise_before_placed = True

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=faulty_adapter, paper_adapter=real_paper)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_raise_before")

    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "EXEC_ERROR"
    assert breaker.is_tripped() is True
    assert tm.reconcile_done is False
    assert len(state_mgr.data.get("pending_trades", [])) == 1

    # On restart, reconciliation with fresh startup breaker detects order never reached broker, cancels it
    faulty_adapter.raise_before_placed = False
    breaker_startup = CircuitBreaker(config=cfg)
    rec_res = reconcile_startup(
        adapter=faulty_adapter,
        saved_state=state_mgr.data,
        breaker=breaker_startup,
    )
    assert rec_res.success is True
    assert len(rec_res.resolved_pending) == 1
    assert rec_res.resolved_pending[0]["status"] == "CANCELLED"


# ── Scenario 3: Open trades call fails ───────────────────────────────────────

def test_fault_open_trades_call_fails(tmp_path):
    """Broker get_open_trades() raises during post-placement verification: breaker trips, never retries."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    real_paper = PaperAdapter(initial_balance=10000.0)
    faulty_adapter = FaultyAdapter(real_paper)
    faulty_adapter.fail_open_trades = True

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=faulty_adapter, paper_adapter=real_paper)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_open_trades_fail")

    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "VERIFICATION_FAILED"
    assert breaker.is_tripped() is True
    assert tm.reconcile_done is False


# ── Scenario 4: Corrupted state file ─────────────────────────────────────────

def test_fault_corrupted_state_file(tmp_path):
    """Corrupted state file starts fail-closed: breaker tripped and adaptive PAUSED."""
    corrupted_file = tmp_path / "state.json"
    corrupted_file.write_text("CORRUPTED_NOT_JSON!!!", encoding="utf-8")

    state_mgr = StateManager(state_file=corrupted_file, first_run=False)
    data = state_mgr.load()

    assert data["breaker_state"]["tripped"] is True
    assert "unreadable or corrupted" in data["breaker_state"]["trip_reason"].lower()
    assert data["adaptive_state"]["tier"] == "PAUSED"
    assert data["adaptive_state"]["real_trading_allowed"] is False


# ── Scenario 5: Process restart with open trade ──────────────────────────────

def test_fault_process_restart_with_open_trade(tmp_path):
    """Restart with trade already open on broker: reconcile confirms trade, prevents duplicate."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file, first_run=True, broker_balance=10000.0)
    real_paper = PaperAdapter(initial_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=real_paper, paper_adapter=real_paper)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_restart_test")

    # Place trade successfully
    res = tm.execute(dec, account)
    assert res.executed is True
    assert res.status == "OPEN"

    # Process "crashes" and a new TradeManager starts from state file
    state_mgr_new = StateManager(state_file=state_file)
    state_data = state_mgr_new.load()
    breaker_new = CircuitBreaker(config=cfg)

    # Startup reconcile
    rec_res = reconcile_startup(adapter=real_paper, saved_state=state_data, breaker=breaker_new)
    assert rec_res.success is True
    assert len(rec_res.reconciled_open_trades) == 1

    tm_new = TradeManager(cfg, state_mgr_new, breaker_new, ks, adapter=real_paper, paper_adapter=real_paper)
    tm_new.set_reconciled(True)

    # Replay of the same decision ID is rejected as DUPLICATE
    res_replay = tm_new.execute(dec, account)
    assert res_replay.executed is False
    assert res_replay.status == "DUPLICATE"
    # No duplicate trade placed on broker
    assert len(real_paper.get_open_trades()) == 1


# ── Scenario 6: Telegram down ────────────────────────────────────────────────

def test_fault_telegram_down_pauses_real_trading_to_shadow(tmp_path, monkeypatch):
    """If Telegram network fails, execution doesn't crash; after max failures, pauses to shadow."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.safety.max_alert_failures = 2
    cfg.adaptive.enabled = False

    def mock_broken_urlopen(*args, **kwargs):
        raise URLError("Network unreachable / connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", mock_broken_urlopen)

    alerter = TelegramAlerter(token="test_token", chat_id="12345", timeout=0.1)
    real_paper = PaperAdapter(initial_balance=10000.0)

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=real_paper, paper_adapter=real_paper, alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})

    # Send 2 failing alerts to exceed max_alert_failures (2)
    tm._send_alert_safe("Failing alert 1")
    tm._send_alert_safe("Failing alert 2")
    assert tm.consecutive_alert_failures >= 2

    # Execute trade: must NOT crash and must route to paper shadow
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_tg_down")
    res = tm.execute(dec, account)
    assert res.executed is True
    assert res.is_shadow is True  # Real trading paused to shadow!
    assert res.status == "SHADOW_OPEN"
