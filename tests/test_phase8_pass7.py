"""Verification tests for Phase 8 fix pass 7.

Covers:
  1. TRADE LOGGING:
     - Real & shadow trades written to trades table via TradeRepository.
     - Mode (paper/demo/live), asset, is_otc, direction, expiry, stake, payout_at_entry, pattern_name, pattern_type, risk_score, hour_utc.
     - Integration test with REAL in-memory TradeRepository and stats computed from rows.
     - Failed real write alerts and trips breaker; failed shadow write alerts and logs without tripping.
     - settle_shadow_trades settles paper trades via PaperAdapter and updates repository.
  2. BEFORE COUNT:
     - Failed open-trades snapshot before placement blocks order, alerts, and records 1 execution error.
  3. PENDING CLEANUP:
     - PENDING record removed on block before adapter call (breaker rate limit, failed snapshot).
  4. SETTLEMENT UNKNOWN ID:
     - record_settlement raises KeyError and alerts when trade id is not in open trades.
  5. PAYOUT:
     - Real receipt missing payout alerts and trips breaker.
     - No 0.85 defaults in receipt, open record, or settlement.
  6. STATE:
     - state.update with adaptive_state=None and breaker_state=None treats as no change (keeps PAUSED).
     - clear_adaptive_state explicitly clears adaptive_state to None.
  7. DEMO MODE BYPASS:
     - execution.demo_bypasses_paper_gate (default false).
     - When true AND mode == "demo", non-live_eligible assets allowed for real order.
     - Never bypasses in live mode.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from adaptive.monitor import AdaptiveState
from adapters.paper import PaperAdapter
from core.config import load_config
from core.models import AssetQuote, CanonicalTrade, TradeDirection, TradeReceipt, TradeResult
from data.store import DataStore, TradeRecord, TradeRepository
from decision.schema import DecisionEvidence, ValidatedDecision
from execution.state import StateManager
from execution.trade_manager import TradeManager
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot, OrderRequest, check_order
from selection.stats import compute_asset_stats


class MockAlerter:
    def __init__(self):
        self.alerts: list[str] = []

    def send_alert(self, msg: str) -> bool:
        self.alerts.append(msg)
        return True


class DummyBrokerAdapter:
    """Mock broker adapter for testing real order execution."""

    def __init__(self, payout: float = 0.85, fail_snapshot: bool = False):
        self.payout = payout
        self.fail_snapshot = fail_snapshot
        self.open_trades: list[dict] = []
        self.placed_calls: list[dict] = []

    def get_open_trades(self) -> list[dict]:
        if self.fail_snapshot:
            raise RuntimeError("Broker snapshot network failure")
        return list(self.open_trades)

    def place_trade(
        self,
        asset_id: str,
        direction: TradeDirection,
        expiry_min: int,
        stake_amount: float,
        trade_id: str,
    ) -> TradeReceipt:
        now = datetime.now(timezone.utc)
        self.placed_calls.append({
            "asset_id": asset_id,
            "direction": direction,
            "expiry_min": expiry_min,
            "stake_amount": stake_amount,
            "trade_id": trade_id,
        })
        receipt = TradeReceipt(
            trade_id=trade_id,
            platform_trade_id=f"plat_{trade_id}",
            asset_id=asset_id,
            direction=direction,
            expiry_minutes=expiry_min,
            stake_amount=stake_amount,
            entry_price=1.1000,
            entry_time=now,
            payout=self.payout,
            status="OPEN",
        )
        self.open_trades.append({
            "id": trade_id,
            "platform_trade_id": f"plat_{trade_id}",
            "asset": asset_id,
            "direction": direction.value,
            "expiry_minutes": expiry_min,
            "stake": stake_amount,
            "payout": self.payout,
            "status": "OPEN",
        })
        return receipt


def _make_decision(
    asset="EURUSD",
    direction="CALL",
    stake=100.0,
    d_id="dec_pass7_01",
    pattern_name="double_bottom",
    pattern_type="reversal",
    risk_score=25,
    is_otc=False,
    payout=0.85,
):
    return ValidatedDecision(
        decision=direction,
        is_valid=True,
        stake_amount=stake,
        asset=asset,
        direction=direction,
        expiry_minutes=15,
        stake_pct=1.0,
        payout=payout,
        decision_id=d_id,
        evidence=DecisionEvidence(
            risk_score=risk_score,
            similarity=85.0,
            trend_daily="up",
            trend_1h="up",
            payout=payout,
        ),
        reasoning="Phase 8 fix pass 7 test decision",
        checks_passed=["check_a", "check_b"],
        pattern_name=pattern_name,
        pattern_type=pattern_type,
        risk_score=risk_score,
        is_otc=is_otc,
    )


# ── ITEM 1: Real and Shadow Trade Logging Integration ─────────────────────────


def test_trade_manager_real_and_shadow_trade_logging_integration(tmp_path):
    """Real and shadow trades are logged to TradeRepository; settlements update result; stats reflect rows."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"

    # Real in-memory DataStore & TradeRepository
    data_store = DataStore(":memory:")
    repo = TradeRepository(data_store)

    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    broker = DummyBrokerAdapter(payout=0.85)
    paper = PaperAdapter(initial_balance=10000.0)
    paper.set_assets([AssetQuote(asset_id="GBPUSD", name="GBP/USD", is_otc=True, payout=0.85, is_open=True)])
    alerter = MockAlerter()

    tm = TradeManager(
        cfg,
        state_mgr,
        breaker,
        ks,
        adapter=broker,
        paper_adapter=paper,
        repository=repo,
        alerter=alerter,
    )
    tm.set_reconciled(True)

    now = datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc)
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        asset_modes={"EURUSD": "live_eligible", "GBPUSD": "paper"},
    )
    ad_state = AdaptiveState()

    # 1. Real Trade on EURUSD (live_eligible)
    dec_real = _make_decision(
        asset="EURUSD",
        direction="CALL",
        stake=100.0,
        d_id="dec_real_01",
        pattern_name="double_bottom",
        pattern_type="reversal",
        risk_score=22,
        is_otc=False,
    )
    res_real = tm.execute(dec_real, account, adaptive_state=ad_state, now=now)
    assert res_real.executed is True
    assert res_real.is_shadow is False
    assert res_real.trade_id is not None

    # Check repository has real trade
    trades = repo.get_trades()
    assert len(trades) == 1
    t1 = trades[0]
    assert t1.id == res_real.trade_id
    assert t1.asset == "EURUSD"
    assert t1.mode == "demo"
    assert t1.is_otc is False
    assert t1.direction == "CALL"
    assert t1.expiry_minutes == 15
    assert t1.stake == 100.0
    assert t1.payout_at_entry == 0.85
    assert t1.result == "PENDING"
    assert t1.pattern_name == "double_bottom"
    assert t1.pattern_type == "reversal"
    assert t1.risk_score == 22
    assert t1.hour_utc == 14

    # 2. Shadow Trade on GBPUSD (paper asset)
    dec_shadow = _make_decision(
        asset="GBPUSD",
        direction="PUT",
        stake=100.0,
        d_id="dec_shadow_01",
        pattern_name="resistance_bounce",
        pattern_type="continuation",
        risk_score=30,
        is_otc=True,
    )
    paper.set_current_time(now)
    res_shadow = tm.execute(dec_shadow, account, adaptive_state=ad_state, now=now)
    assert res_shadow.executed is True
    assert res_shadow.is_shadow is True

    trades = repo.get_trades()
    assert len(trades) == 2
    t2 = [t for t in trades if t.id == res_shadow.trade_id][0]
    assert t2.mode == "paper"
    assert t2.asset == "GBPUSD"
    assert t2.direction == "PUT"
    assert t2.stake == 100.0
    assert t2.is_otc is True
    assert t2.pattern_name == "resistance_bounce"
    assert t2.pattern_type == "continuation"
    assert t2.risk_score == 30
    assert t2.result == "PENDING"

    # 3. Settle Real Trade
    now_settle = now + timedelta(minutes=15)
    settle_res = tm.record_settlement(res_real.trade_id, outcome="WIN", now=now_settle)
    assert settle_res["outcome"] == "WIN"

    t1_updated = repo.get_trade_by_id(res_real.trade_id)
    assert t1_updated.result == "WIN"
    assert t1_updated.pnl == 85.0
    assert t1_updated.closed_at is not None

    # 4. Settle Shadow Trades via settle_shadow_trades
    settled_shadow = tm.settle_shadow_trades(now=now_settle, prices_at_expiry={"GBPUSD": 0.9500})
    assert len(settled_shadow) == 1

    t2_updated = repo.get_trade_by_id(res_shadow.trade_id)
    assert t2_updated.result == "WIN"
    assert t2_updated.pnl == 100.0 * 0.85

    # 5. Compute Stats from Repository Rows
    stats_eur = compute_asset_stats(repo.get_trades(), asset="EURUSD", mode="demo")
    assert stats_eur.n == 1
    assert stats_eur.wins == 1
    assert stats_eur.win_rate == 1.0

    stats_gbp = compute_asset_stats(repo.get_trades(), asset="GBPUSD", mode="paper")
    assert stats_gbp.n == 1
    assert stats_gbp.wins == 1
    assert stats_gbp.win_rate == 1.0


def test_repository_write_failure_trips_breaker_for_real_trades_only(tmp_path):
    """Failed write raises alert and trips breaker for real trades; shadow write alerts and logs without tripping."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"

    class FailingRepository:
        def __init__(self):
            self.fail_insert = False

        def insert(self, record):
            if self.fail_insert:
                raise RuntimeError("DB Disk I/O error on insert")

        def update_result(self, *args, **kwargs):
            return True

    repo = FailingRepository()
    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    broker = DummyBrokerAdapter(payout=0.85)
    paper = PaperAdapter(initial_balance=10000.0)
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=broker, paper_adapter=paper, repository=repo, alerter=alerter)
    tm.set_reconciled(True)
    now = datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc)

    # A: Shadow trade write failure -> logs and alerts, does NOT trip breaker
    repo.fail_insert = True
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"GBPUSD": "paper"})
    dec_shadow = _make_decision(asset="GBPUSD", stake=100.0, d_id="dec_shadow_fail")
    res_sh = tm.execute(dec_shadow, account, adaptive_state=AdaptiveState(), now=now)
    assert res_sh.executed is True
    assert breaker.is_tripped() is False
    assert any("Database write failure for shadow trade" in a for a in alerter.alerts)

    # B: Real trade write failure -> trips breaker and alerts
    alerter.alerts.clear()
    account_real = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec_real = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_real_fail")
    res_real = tm.execute(dec_real, account_real, adaptive_state=AdaptiveState(), now=now)
    assert breaker.is_tripped() is True
    assert "Database write failure for real trade" in breaker.trip_reason
    assert any("CRITICAL: Database write failure for real trade" in a for a in alerter.alerts)


# ── ITEM 2 & 3: Before Count Snapshot Failure & Pending Cleanup ───────────────


def test_failed_before_count_blocks_order_cleans_pending_and_counts_exec_error(tmp_path):
    """If open-trades snapshot before placement cannot be read: return BLOCKED, clean PENDING, count 1 exec error."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"

    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    broker = DummyBrokerAdapter(payout=0.85, fail_snapshot=True)
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=broker, paper_adapter=PaperAdapter(), alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_fail_snap")

    res = tm.execute(dec, account, adaptive_state=AdaptiveState())
    assert res.executed is False
    assert res.status == "BLOCKED"
    assert "Failed to read open-trades snapshot before placement" in res.reason

    # Verify execution error was counted
    assert breaker.state.consecutive_exec_errors == 1

    # Verify PENDING trade was cleaned up from state
    assert len(state_mgr.data.get("pending_trades", [])) == 0
    assert any("Failed to read open-trades snapshot before placement" in a for a in alerter.alerts)


def test_pending_cleanup_on_breaker_rate_limit(tmp_path):
    """Pending record is removed when rate limit trips breaker before adapter call."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.safety.max_orders_per_window = 1

    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    broker = DummyBrokerAdapter(payout=0.85)
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=broker, paper_adapter=PaperAdapter(), alerter=alerter)
    tm.set_reconciled(True)

    now = datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc)
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})

    # First order succeeds
    res1 = tm.execute(_make_decision(stake=100.0, d_id="ord1"), account, adaptive_state=AdaptiveState(), now=now)
    assert res1.executed is True

    # Second order in same window exceeds rate limit
    res2 = tm.execute(_make_decision(stake=100.0, d_id="ord2"), account, adaptive_state=AdaptiveState(), now=now)
    assert res2.executed is False
    assert res2.status == "BREAKER_TRIPPED"

    # Pending trades list must NOT retain ord2
    assert len(state_mgr.data.get("pending_trades", [])) == 0


# ── ITEM 4: Settlement Unknown ID ─────────────────────────────────────────────


def test_record_settlement_unknown_trade_id_raises_and_alerts(tmp_path):
    """record_settlement raises KeyError and alerts when trade id is not in open trades."""
    cfg = load_config("config.yaml")
    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=DummyBrokerAdapter(), alerter=alerter)

    with pytest.raises(KeyError, match="not found in open trades"):
        tm.record_settlement("tr_ghost_999", outcome="WIN")

    assert any("tr_ghost_999" in a and "not found in open trades" in a for a in alerter.alerts)


# ── ITEM 5: Payout Handling ───────────────────────────────────────────────────


def test_real_order_receipt_missing_payout_trips_breaker(tmp_path):
    """If real receipt has no payout or payout <= 0, alert and trip the breaker."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"

    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    broker = DummyBrokerAdapter(payout=0.0)  # Invalid payout 0.0
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=broker, paper_adapter=PaperAdapter(), alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    res = tm.execute(_make_decision(asset="EURUSD", stake=100.0, d_id="dec_nopayout"), account, adaptive_state=AdaptiveState())

    assert res.executed is False
    assert res.status == "EXEC_ERROR"
    assert breaker.is_tripped() is True
    assert "missing or invalid payout" in breaker.trip_reason
    assert any("CRITICAL: Real order" in a and "missing or invalid payout" in a for a in alerter.alerts)


def test_settlement_missing_payout_in_record_and_call_raises_and_alerts(tmp_path):
    """record_settlement raises ValueError and alerts if neither target trade nor argument has payout."""
    cfg = load_config("config.yaml")
    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=1000.0)
    state_mgr.load()
    # Add open trade explicitly with payout=None
    state_mgr.update(balance=900.0, open_trades=[{"id": "tr_nopay", "stake": 100.0, "payout": None}])
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=DummyBrokerAdapter(), alerter=alerter)

    with pytest.raises(ValueError, match="no payout available"):
        tm.record_settlement("tr_nopay", outcome="WIN", payout=None)

    assert any("no payout available in trade record or settlement call" in a for a in alerter.alerts)


# ── ITEM 6: State Update None Preserves PAUSED & Clear Method ─────────────────


def test_state_update_none_preserves_adaptive_paused_and_clear_method(tmp_path):
    """state.update with adaptive_state=None keeps saved PAUSED; clear_adaptive_state clears it."""
    state_file = tmp_path / "state.json"
    mgr = StateManager(state_file, first_run=True, broker_balance=1000.0)
    mgr.load()

    # Set PAUSED adaptive state
    paused_dict = {
        "tier": "PAUSED",
        "real_trading_allowed": False,
        "stake_scale": 0.0,
        "transition_reason": "Drawdown limit reached",
    }
    mgr.update(adaptive_state=paused_dict, balance=950.0)
    assert mgr.data["adaptive_state"]["tier"] == "PAUSED"

    # Update with adaptive_state=None and breaker_state=None -> MUST NOT wipe PAUSED state
    mgr.update(adaptive_state=None, breaker_state=None, balance=940.0)
    assert mgr.data["adaptive_state"]["tier"] == "PAUSED"
    assert mgr.data["balance"] == 940.0

    # Reload from disk to verify persistence
    reloaded = StateManager(state_file).load()
    assert reloaded["adaptive_state"]["tier"] == "PAUSED"

    # Explicit clear_adaptive_state clears it
    mgr.clear_adaptive_state()
    assert mgr.data["adaptive_state"] is None

    reloaded_after_clear = StateManager(state_file).load()
    assert reloaded_after_clear["adaptive_state"] is None


# ── ITEM 7: Demo Mode Bypasses Paper Gate ─────────────────────────────────────


def test_demo_bypasses_paper_gate_when_configured(tmp_path, monkeypatch):
    """When demo_bypasses_paper_gate is True and mode is demo, non-live_eligible assets are allowed for real orders; never in live."""
    cfg = load_config("config.yaml")

    # 1. Default: demo_bypasses_paper_gate is False
    assert cfg.execution.demo_bypasses_paper_gate is False
    cfg.mode = "demo"

    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0, shadow_only=False)
    account_paper_asset = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "paper"})
    ad_state = AdaptiveState()

    # Blocked by check_order under default False
    res_default = check_order(order, account_paper_asset, cfg, adaptive_state=ad_state)
    assert res_default.allowed is False
    assert "requires 'live_eligible'" in res_default.reason

    # 2. Enable bypass in demo mode: allowed!
    cfg.execution.demo_bypasses_paper_gate = True
    res_demo_bypass = check_order(order, account_paper_asset, cfg, adaptive_state=ad_state)
    assert res_demo_bypass.allowed is True

    # Real order executes on broker adapter in TradeManager
    state_mgr = StateManager(tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    state_mgr.load()
    breaker = CircuitBreaker(cfg)
    ks = KillSwitch(cfg.safety.kill_file)
    broker = DummyBrokerAdapter(payout=0.85)

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=broker, paper_adapter=PaperAdapter())
    tm.set_reconciled(True)

    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_bypass_demo")
    exec_res = tm.execute(dec, account_paper_asset, adaptive_state=ad_state)
    assert exec_res.executed is True
    assert exec_res.is_shadow is False
    assert len(broker.placed_calls) == 1

    # 3. Live mode: NEVER bypasses paper gate, even if demo_bypasses_paper_gate is True!
    monkeypatch.setenv("LIVE_CONFIRM", "YES")
    cfg.mode = "live"
    res_live_blocked = check_order(order, account_paper_asset, cfg, adaptive_state=ad_state)
    assert res_live_blocked.allowed is False
    assert "requires 'live_eligible'" in res_live_blocked.reason
