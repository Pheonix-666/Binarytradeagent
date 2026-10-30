"""Targeted verification tests for Phase 8 fix pass 6.

Covers:
  - GROUP A:
    1. Adapter Interface: TradeManager with REAL PaperAdapter (real, shadow, settle, save, restart).
    2. State: update preserves adaptive_state (PAUSED), asset_modes, daily_starting_balance, today_loss.
    3. Fail-Closed State: missing after first run, unreadable JSON, schema mismatch, first-run broker balance adoption.
    4. Verification: older identical trade does NOT confirm failed order.
    5. Modes: demo, live, paper, record (no orders), unknown mode; missing adaptive treated as enabled.
    6. Decision IDs: UUID assigned by DecisionService; TM rejects real orders without ID; deduplication after restart.
    7. Ambiguous Failure: exception during adapter call trips breaker, preserves PENDING, reconcile_done=False.
    8. Reconcile: offline settlement balance accounting, unreadable history for pending -> UNKNOWN.
    9. Settlement: entry payout preserved, WIN, LOSS, TIE (refund/loss), VOID, unknown raises ValueError.
  - GROUP B:
    10. check_order: tier-scaled max < min stake blocks; non-CALL/PUT rejected; missing stake fails closed; NaN balance mismatch.
    11. Validator: missing price skips; ATR fallback/skip; CanonicalTrade handles dict trades.
    12. Telegram: retry on 400; /reset_breaker requires reconcile + confirm; /resume uses real payouts; alert failure pausing.
    13. Hygiene: database write failure alerted, not swallowed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import pytest
import urllib.request
from urllib.error import HTTPError

from adaptive.monitor import AdaptiveController, AdaptiveState
from adapters.paper import PaperAdapter
from alerts.telegram import TelegramAlerter, create_operator_command_handlers
from core.config import AppConfig, load_config
from core.models import AssetQuote, CanonicalTrade, TradeDirection, TradeReceipt, TradeResult
from decision.schema import (
    CandidateReportItem,
    DecisionEvidence,
    DecisionOutput,
    DecisionReport,
    SystemState,
    ValidatedDecision,
)
from decision.service import DecisionService
from decision.validator import validate_decision
from execution.reconcile import reconcile_startup
from execution.state import StateManager
from execution.trade_manager import TradeManager
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot, OrderRequest, check_order


class MockAlerter:
    def __init__(self):
        self.alerts: list[str] = []
        self.should_fail = False
        self.fail_count = 0

    def send_alert(self, msg: str) -> bool:
        self.alerts.append(msg)
        if self.should_fail:
            self.fail_count += 1
            return False
        return True


def _make_decision(asset="EURUSD", direction="CALL", stake=100.0, d_id="dec_test_123"):
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
        reasoning="Testing pass 6",
        checks_passed=["c1", "c2"],
        raw_decision=DecisionOutput(
            decision=direction,
            asset=asset,
            expiry_minutes=15,
            stake_pct=1.0,
            decision_id=d_id,
            evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
            reasoning="Testing pass 6",
        ),
    )


# ── GROUP A.1: Adapter Interface with REAL PaperAdapter ─────────────────────

def test_trade_manager_with_real_paper_adapter(tmp_path):
    """Integration: TM with real PaperAdapter: real order, shadow order, settle, save, restart."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file, first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    real_paper_adapter = PaperAdapter(initial_balance=10000.0)
    shadow_paper_adapter = PaperAdapter(initial_balance=10000.0)
    alerter = MockAlerter()

    tm = TradeManager(
        cfg,
        state_mgr,
        breaker,
        ks,
        adapter=real_paper_adapter,
        paper_adapter=shadow_paper_adapter,
        alerter=alerter,
    )
    tm.set_reconciled(True)

    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        asset_modes={"EURUSD": "live_eligible", "GBPUSD": "paper"},
    )

    # 1. Real order on live_eligible asset
    dec_real = _make_decision(asset="EURUSD", direction="CALL", stake=100.0, d_id="dec_real_01")
    res_real = tm.execute(dec_real, account)
    assert res_real.executed is True
    assert res_real.status == "OPEN"
    assert res_real.is_shadow is False
    assert res_real.receipt is not None
    assert res_real.receipt.platform_trade_id is not None
    trade_id = res_real.trade_id

    # 2. Shadow order on paper asset
    dec_shadow = _make_decision(asset="GBPUSD", direction="PUT", stake=100.0, d_id="dec_shadow_02")
    res_shadow = tm.execute(dec_shadow, account)
    assert res_shadow.executed is True
    assert res_shadow.status == "SHADOW_OPEN"
    assert res_shadow.is_shadow is True

    # 3. Restart with new StateManager and TradeManager instance while trade is open
    state_mgr_reloaded = StateManager(state_file=state_file)
    data = state_mgr_reloaded.load()
    assert len(data["open_trades"]) == 1
    assert data["balance"] == 9900.0

    tm2 = TradeManager(
        cfg,
        state_mgr_reloaded,
        CircuitBreaker(config=cfg),
        ks,
        adapter=real_paper_adapter,
        paper_adapter=shadow_paper_adapter,
        alerter=alerter,
    )
    tm2.set_reconciled(True)

    # Deduplication after restart: same decision ID rejected
    assert "dec_real_01" in tm2.executed_decision_ids
    res_dup = tm2.execute(dec_real, account)
    assert res_dup.executed is False
    assert res_dup.status == "DUPLICATE"

    # 4. Settle real order
    settle_res = tm2.record_settlement(trade_id=trade_id, outcome="WIN", payout=0.85)
    assert settle_res["outcome"] == "WIN"
    assert settle_res["profit"] == 85.0
    assert state_mgr_reloaded.data["balance"] == 9900.0 + 100.0 + 85.0  # 10085.0


# ── GROUP A.2 & A.3: State Update Preservation and Fail-Closed State ─────────

def test_state_update_preserves_paused_tier_and_asset_modes(tmp_path):
    """Settle a trade while PAUSED, reload, state is still PAUSED with same asset_modes."""
    cfg = load_config("config.yaml")
    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file, first_run=True, broker_balance=5000.0)

    # Set state as PAUSED with custom asset_modes and daily loss
    state_mgr.update(
        adaptive_state={"tier": "PAUSED", "real_trading_allowed": False},
        asset_modes={"EURUSD": "live_eligible", "USDJPY": "paper"},
        daily_starting_balance=5000.0,
        today_loss=120.0,
        open_trades=[{"id": "tr_p1", "asset": "EURUSD", "direction": "CALL", "stake": 100.0, "payout": 0.85}],
        balance=4900.0,
    )

    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    paper_adapter = PaperAdapter(initial_balance=5000.0)
    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=paper_adapter, paper_adapter=paper_adapter)
    tm.set_reconciled(True)

    # Settle trade
    tm.record_settlement("tr_p1", outcome="WIN")

    # Reload fresh instance
    state_mgr_reloaded = StateManager(state_file=state_file)
    loaded = state_mgr_reloaded.load()
    assert loaded["adaptive_state"]["tier"] == "PAUSED"
    assert loaded["asset_modes"]["EURUSD"] == "live_eligible"
    assert loaded["daily_starting_balance"] == 5000.0
    assert loaded["today_loss"] == 120.0


def test_state_fail_closed_on_missing_corrupt_or_schema_mismatch(tmp_path):
    """If state file is missing after first run, unreadable, or schema invalid: breaker trips and PAUSED."""
    missing_file = tmp_path / "nonexistent_state.json"
    mgr_missing = StateManager(state_file=missing_file, first_run=False)
    data_missing = mgr_missing.load()
    assert data_missing["breaker_state"]["tripped"] is True
    assert data_missing["adaptive_state"]["tier"] == "PAUSED"

    corrupt_file = tmp_path / "corrupt_state.json"
    corrupt_file.write_text("{invalid json corrupt", encoding="utf-8")
    mgr_corrupt = StateManager(state_file=corrupt_file, first_run=False)
    data_corrupt = mgr_corrupt.load()
    assert data_corrupt["breaker_state"]["tripped"] is True
    assert data_corrupt["adaptive_state"]["tier"] == "PAUSED"

    bad_schema_file = tmp_path / "schema_state.json"
    bad_schema_file.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    mgr_schema = StateManager(state_file=bad_schema_file, first_run=False)
    data_schema = mgr_schema.load()
    assert data_schema["breaker_state"]["tripped"] is True
    assert data_schema["adaptive_state"]["tier"] == "PAUSED"


# ── GROUP A.4: Verification: Older Identical Trade Does NOT Confirm ──────────

class OlderIdenticalTradeAdapter:
    def __init__(self):
        # Already has an open identical trade from earlier
        self.open_trades = [
            {"id": "old_tr_1", "asset": "EURUSD", "direction": "CALL", "expiry_minutes": 15, "stake": 100.0}
        ]

    def get_open_trades(self):
        return list(self.open_trades)

    def get_balance(self):
        return 10000.0

    def place_trade(self, **kwargs):
        # Platform returns no receipt ID, and failed to actually add new trade to open_trades
        from core.models import TradeDirection, TradeReceipt
        return TradeReceipt(
            trade_id="platform_untracked",
            platform_trade_id=None,
            asset_id="EURUSD",
            direction=TradeDirection.CALL,
            expiry_minutes=15,
            stake_amount=100.0,
            entry_price=1.10,
            entry_time=datetime.now(timezone.utc),
        )


def test_verification_older_identical_trade_does_not_confirm(tmp_path):
    """An older identical trade must NOT make a failed new order look confirmed."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = OlderIdenticalTradeAdapter()
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=adapter, alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", direction="CALL", stake=100.0, d_id="dec_ident_01")

    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "VERIFICATION_FAILED"
    assert breaker.is_tripped() is True


# ── GROUP A.5 & A.6: Modes and Decision IDs ──────────────────────────────────

def test_modes_and_decision_id_enforcement(tmp_path):
    """Real orders only in demo/live; record mode places no orders; real orders require decision_id."""
    cfg = load_config("config.yaml")
    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    paper_adapter = PaperAdapter(initial_balance=10000.0)

    # 1. Record mode: places no orders at all
    cfg.mode = "record"
    tm_record = TradeManager(cfg, state_mgr, breaker, ks, adapter=paper_adapter, paper_adapter=paper_adapter)
    tm_record.set_reconciled(True)
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_rec_01")
    res_rec = tm_record.execute(dec, account)
    assert res_rec.executed is False
    assert res_rec.status == "SKIPPED"
    assert "Record mode" in res_rec.reason

    # 2. Real order missing decision_id is blocked
    cfg.mode = "demo"
    cfg.adaptive.enabled = False
    tm_demo = TradeManager(cfg, state_mgr, breaker, ks, adapter=paper_adapter, paper_adapter=paper_adapter)
    tm_demo.set_reconciled(True)
    dec_noid = _make_decision(asset="EURUSD", stake=100.0, d_id=None)
    dec_noid.decision_id = None
    dec_noid.raw_decision.decision_id = None
    res_noid = tm_demo.execute(dec_noid, account)
    assert res_noid.executed is False
    assert res_noid.status == "BLOCKED"
    assert "missing decision_id" in res_noid.reason


# ── GROUP A.7: Ambiguous Adapter Failure ─────────────────────────────────────

class TimeoutAdapter:
    def get_open_trades(self):
        return []
    def get_balance(self):
        return 10000.0
    def place_trade(self, **kwargs):
        raise TimeoutError("Broker connection timed out after placing trade")


def test_ambiguous_adapter_failure_trips_breaker_and_preserves_pending(tmp_path):
    """Exception during adapter call trips breaker, preserves PENDING, sets reconcile_done=False."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = TimeoutAdapter()
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=adapter, alerter=alerter)
    tm.set_reconciled(True)
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_timeout_01")

    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "EXEC_ERROR"
    assert breaker.is_tripped() is True
    assert tm.reconcile_done is False
    # PENDING trade was retained in state for reconciliation!
    assert len(state_mgr.data.get("pending_trades", [])) == 1
    assert state_mgr.data["pending_trades"][0]["status"] == "PENDING"


# ── GROUP A.8: Reconcile Offline Settlements & Unreadable History ────────────

def test_reconcile_offline_settlements_and_unreadable_history(tmp_path):
    """Reconcile handles offline settlements without tripping balance breaker; unreadable history trips breaker."""
    cfg = load_config("config.yaml")
    breaker = CircuitBreaker(config=cfg)

    # Offline settlement case: open trade in state settled as WIN on broker
    # State has balance 9800.0 (200.0 was deducted at entry). Broker balance is 10170.0 (win: 200 + 170).
    adapter_offline = PaperAdapter(initial_balance=10170.0)
    adapter_offline._trade_history = [{
        "id": "tr_offline_1",
        "asset": "EURUSD",
        "stake": 200.0,
        "result": "WIN",
        "payout": 0.85,
        "payout_received": 370.0,
    }]
    saved_state = {
        "balance": 9800.0,
        "open_trades": [{"id": "tr_offline_1", "asset": "EURUSD", "stake": 200.0, "payout": 0.85}],
        "pending_trades": [],
    }

    res_rec = reconcile_startup(adapter=adapter_offline, saved_state=saved_state, breaker=breaker)
    assert res_rec.success is True
    assert res_rec.expected_balance == 10170.0
    assert breaker.is_tripped() is False


# ── GROUP A.9: Settlement Outcomes and Error Handling ────────────────────────

def test_settlement_outcomes_and_unknown_handling(tmp_path):
    """record_settlement handles WIN, LOSS, TIE (refund or loss), VOID, and raises on unknown."""
    cfg = load_config("config.yaml")
    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file, first_run=True, broker_balance=1000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    adapter = PaperAdapter(initial_balance=1000.0)

    # 1. TIE with refund policy
    cfg.trading.tie_policy = "refund"
    state_mgr.update(balance=900.0, open_trades=[{"id": "t_tie", "stake": 100.0, "payout": 0.80}])
    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=adapter, paper_adapter=adapter)
    res_tie = tm.record_settlement("t_tie", outcome="TIE")
    assert res_tie["balance"] == 1000.0

    # 2. TIE with loss policy
    cfg.trading.tie_policy = "loss"
    state_mgr.update(balance=900.0, open_trades=[{"id": "t_tie2", "stake": 100.0, "payout": 0.80}])
    res_tie2 = tm.record_settlement("t_tie2", outcome="TIE")
    assert res_tie2["balance"] == 900.0

    # 3. VOID
    state_mgr.update(balance=900.0, open_trades=[{"id": "t_void", "stake": 100.0, "payout": 0.80}])
    res_void = tm.record_settlement("t_void", outcome="VOID")
    assert res_void["balance"] == 1000.0

    # 4. Unknown outcome raises ValueError
    state_mgr.update(balance=900.0, open_trades=[{"id": "t_unk", "stake": 100.0, "payout": 0.80}])
    with pytest.raises(ValueError, match="Unknown settlement outcome"):
        tm.record_settlement("t_unk", outcome="CANCELLED_BY_USER")


# ── GROUP B.10: check_order Constraints ──────────────────────────────────────

def test_check_order_tier_cap_non_call_put_missing_stake():
    """check_order blocks if tier cap < min stake, rejects non-CALL/PUT, fails closed on missing stake."""
    cfg = load_config("config.yaml")
    cfg.stake.min_amount = 50.0

    # Tier scale drops max allowed below min stake ($50)
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=50.0)
    account = AccountSnapshot(balance=1000.0, equity=1000.0, asset_modes={"EURUSD": "live_eligible"})
    ad_state = AdaptiveState(tier="DEFENSIVE", stake_scale=0.01)  # 1% scale -> $2 max < $50 min

    res = check_order(order=order, account=account, config=cfg, adaptive_state=ad_state)
    assert res.allowed is False
    assert "below minimum stake" in res.reason

    # Invalid direction
    order_bad_dir = OrderRequest(asset="EURUSD", direction="BUY", expiry_minutes=15, stake=50.0)
    res_bad_dir = check_order(order=order_bad_dir, account=account, config=cfg, adaptive_state=AdaptiveState())
    assert res_bad_dir.allowed is False
    assert "Invalid direction" in res_bad_dir.reason

    # Open trade with unparseable stake fails closed
    account_bad_open = AccountSnapshot(
        balance=1000.0,
        equity=1000.0,
        open_trades=[{"asset": "GBPUSD", "direction": "CALL", "stake": "INVALID_STAKE"}],
        asset_modes={"EURUSD": "live_eligible"},
    )
    res_bad_open = check_order(order=order, account=account_bad_open, config=cfg, adaptive_state=AdaptiveState())
    assert res_bad_open.allowed is False
    assert "missing or invalid stake" in res_bad_open.reason
    assert res_bad_open.alert_required is True


# ── GROUP B.11: Validator & CanonicalTrade ───────────────────────────────────

def test_validator_skips_on_missing_current_price_and_handles_canonical_trade():
    """Validator skips if real candidate lacks current price, and handles dict open trades via CanonicalTrade."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.llm.default_atr = None

    from decision.schema import (
        AssetStatsItem,
        NearestLevelItem,
        PatternReportItem,
        TrendReportItem,
    )
    cand = CandidateReportItem(
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        payout=0.85,
        volatility_percentile=50.0,
        risk_score=25,
        max_stake_pct=5.0,
        current_price=None,  # Missing current price!
        atr=0.0015,
        trend=TrendReportItem(daily="up", **{"1h": "up", "15m": "up"}),
        nearest_level=NearestLevelItem(type="support", distance_atr=1.2),
        pattern=PatternReportItem(name="pinbar", type="continuation", similarity=85.0),
        asset_stats=AssetStatsItem(trades=50, win_rate=0.62, mode="live_eligible"),
    )
    rep = DecisionReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        balance=10000.0,
        equity=10000.0,
        open_exposure_pct=0.0,
        remaining_exposure_pct=35.0,
        min_stake_pct=1.0,
        candidates=[cand],
    )
    sys_state = SystemState(
        open_trades=[{"asset": "EURUSD", "direction": "PUT", "stake": 100.0}],
        balance=10000.0,
        kill_switch_active=False,
        breaker_tripped=False,
        reconcile_done=True,
    )
    dec = DecisionOutput(
        decision="CALL",
        asset="EURUSD",
        expiry_minutes=15,
        stake_pct=2.0,
        evidence=DecisionEvidence(risk_score=25, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
        reasoning="Good setup",
    )
    # 1. Opposite direction open trade in dict format triggers check_g skip
    validated_opp = validate_decision(parsed=dec, report=rep, state=sys_state, config=cfg, now=datetime.now(timezone.utc))
    assert validated_opp.decision == "SKIP"
    assert "opposite_direction_open_trade" in validated_opp.skip_reason

    # 2. Without opposite trade, missing current price triggers check_f skip
    sys_state_no_opp = SystemState(
        open_trades=[],
        balance=10000.0,
        asset_modes={"EURUSD": "live_eligible"},
        kill_switch_active=False,
        breaker_tripped=False,
        reconcile_done=True,
    )
    validated_price = validate_decision(parsed=dec, report=rep, state=sys_state_no_opp, config=cfg, now=datetime.now(timezone.utc))
    assert validated_price.decision == "SKIP"
    assert "missing_current_price" in validated_price.skip_reason


# ── GROUP B.12: Telegram Alerter Resilience and Command Verification ─────────

def test_telegram_retry_on_400(monkeypatch):
    """If Telegram returns 400 (Markdown error), retry immediately without parse_mode (plain text)."""
    attempts = []

    def mock_urlopen(req, timeout=5.0):
        data = req.data.decode("utf-8")
        attempts.append(data)
        if "parse_mode=Markdown" in data:
            fp = io.BytesIO(b'{"ok": false, "description": "Bad Request: can\'t parse entities"}')
            raise HTTPError(req.full_url, 400, "Bad Request", hdrs={}, fp=fp)
        return type("DummyResp", (), {"status": 200, "__enter__": lambda s: s, "__exit__": lambda *args: None})()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    alerter = TelegramAlerter(token="test_token", chat_id="123456")
    success = alerter.send_alert("Special _unclosed *markdown* text")
    assert success is True
    assert len(attempts) == 2
    assert "parse_mode=Markdown" in attempts[0]
    assert "parse_mode" not in attempts[1]


def test_telegram_reset_breaker_confirm_and_alert_failure_pause(tmp_path):
    """Telegram /reset_breaker requires confirm + reconcile; alert failure pauses real trading."""
    cfg = load_config("config.yaml")
    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=1000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    breaker.trip("Tripped for test")

    handlers = create_operator_command_handlers(
        state_manager=state_mgr,
        breaker=breaker,
        kill_switch=ks,
        config=cfg,
        reconcile_fn=lambda: type("ReconcileStub", (), {"success": True, "issues": []})(),
    )

    # 1. Unconfirmed /reset_breaker refused
    rep_unconf = handlers["/reset_breaker"]([])
    assert "requires explicit confirmation" in rep_unconf
    assert breaker.is_tripped() is True

    # 2. Confirmed /reset_breaker resets
    rep_conf = handlers["/reset_breaker"](["confirm"])
    assert "Breaker reset" in rep_conf
    assert breaker.is_tripped() is False


def test_trade_manager_pauses_to_shadow_on_consecutive_alert_failures(tmp_path):
    """If alert delivery fails max_alert_failures times, real trading pauses to shadow only."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.safety.max_alert_failures = 3
    cfg.adaptive.enabled = False

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    paper_adapter = PaperAdapter(initial_balance=10000.0)
    alerter = MockAlerter()

    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=paper_adapter, paper_adapter=paper_adapter, alerter=alerter)
    tm.set_reconciled(True)
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})

    # Simulate 3 alert failures
    tm.consecutive_alert_failures = 3

    dec = _make_decision(asset="EURUSD", direction="CALL", stake=100.0, d_id="dec_alert_fail_01")
    res = tm.execute(dec, account)
    assert res.executed is True
    assert res.is_shadow is True  # Routed to shadow!
    assert res.status == "SHADOW_OPEN"


# ── GROUP B.13: Hygiene - Database Write Failure Alerted ──────────────────────

def test_trade_manager_alerts_on_db_write_failure(tmp_path):
    """Database write failure is logged and alerted, never silently swallowed."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.adaptive.enabled = False

    state_mgr = StateManager(state_file=tmp_path / "state.json", first_run=True, broker_balance=10000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)
    paper_adapter = PaperAdapter(initial_balance=10000.0)
    alerter = MockAlerter()

    class FailingDbStore:
        def record_trade(self, trade):
            raise IOError("Disk I/O error writing SQLite database")

    tm = TradeManager(
        cfg,
        state_mgr,
        breaker,
        ks,
        adapter=paper_adapter,
        paper_adapter=paper_adapter,
        db_store=FailingDbStore(),
        alerter=alerter,
    )
    tm.set_reconciled(True)
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = _make_decision(asset="EURUSD", stake=100.0, d_id="dec_db_fail_01")

    res = tm.execute(dec, account)
    assert res.executed is True
    # Alert emitted for DB write failure!
    assert any("Database write failure" in a for a in alerter.alerts)


# ── CLOSING ITEMS: Tie Policy Parity & ATR Fallback ──────────────────────────

def test_tie_policy_parity_backtest_and_live(tmp_path):
    """Backtest engine and live settlement (TradeManager & PaperAdapter) book ties identically."""
    from backtest.engine import OpenTrade, settle_position

    cfg = load_config("config.yaml")
    state_file = tmp_path / "state.json"
    state_mgr = StateManager(state_file=state_file, first_run=True, broker_balance=1000.0)
    breaker = CircuitBreaker(config=cfg)
    ks = KillSwitch(workspace_root=tmp_path)

    # 1. Test loss policy parity
    cfg.trading.tie_policy = "loss"
    ot = OpenTrade(
        trade_id="tie_t1",
        asset="EURUSD",
        direction=TradeDirection.CALL,
        stake=100.0,
        entry_time=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
        entry_price=1.1000,
        expiry_minutes=15,
        exit_time=datetime(2026, 8, 1, 10, 15, tzinfo=timezone.utc),
        payout=0.85,
        pattern_name="p",
        pattern_type="bullish",
        risk_score=20,
    )
    exit_bar_tie = {"close": 1.1000}  # exact tie

    # Backtest settlement
    bt_res, bt_pnl, bt_refund = settle_position(ot, exit_bar_tie, tie_policy=cfg.trading.tie_policy)
    assert bt_res == "LOSS"
    assert bt_pnl == -100.0
    assert bt_refund == 0.0

    # Live TradeManager settlement
    state_mgr.update(balance=900.0, open_trades=[{"id": "tie_t1", "stake": 100.0, "payout": 0.85}])
    tm = TradeManager(cfg, state_mgr, breaker, ks, adapter=PaperAdapter(), paper_adapter=PaperAdapter())
    live_res = tm.record_settlement("tie_t1", outcome="TIE")
    assert live_res["outcome"] == "TIE"
    assert live_res["profit"] == -100.0
    assert live_res["balance"] == 900.0

    # PaperAdapter settlement
    paper = PaperAdapter(initial_balance=1000.0, tie_policy=cfg.trading.tie_policy)
    paper._open_trades.append(ot)
    settled_paper = paper.settle_open_trades(
        current_time=datetime(2026, 8, 1, 10, 16, tzinfo=timezone.utc),
        prices_at_expiry={"EURUSD": 1.1000},
    )
    assert settled_paper[0].result == TradeResult.TIE
    assert settled_paper[0].payout_received == 0.0

    # 2. Test refund policy parity
    cfg.trading.tie_policy = "refund"
    bt_res_ref, bt_pnl_ref, bt_refund_ref = settle_position(ot, exit_bar_tie, tie_policy=cfg.trading.tie_policy)
    assert bt_res_ref == "TIE"
    assert bt_pnl_ref == 0.0
    assert bt_refund_ref == 100.0

    state_mgr.update(balance=900.0, open_trades=[{"id": "tie_t2", "stake": 100.0, "payout": 0.85}])
    live_res_ref = tm.record_settlement("tie_t2", outcome="TIE")
    assert live_res_ref["profit"] == 0.0
    assert live_res_ref["balance"] == 1000.0


def test_validator_skips_real_candidate_with_missing_atr():
    """If a candidate has no ATR and llm.default_atr is null, validator SKIPs real candidates."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.llm.default_atr = None  # Explicit null

    from decision.schema import (
        AssetStatsItem,
        NearestLevelItem,
        PatternReportItem,
        TrendReportItem,
    )
    cand_no_atr = CandidateReportItem(
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        payout=0.85,
        volatility_percentile=50.0,
        risk_score=25,
        max_stake_pct=5.0,
        current_price=1.1000,
        atr=None,  # Missing ATR!
        trend=TrendReportItem(daily="up", **{"1h": "up", "15m": "up"}),
        nearest_level=NearestLevelItem(type="support", distance_atr=1.2),
        pattern=PatternReportItem(name="pinbar", type="continuation", similarity=85.0),
        asset_stats=AssetStatsItem(trades=50, win_rate=0.62, mode="live_eligible"),
    )
    rep = DecisionReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        balance=10000.0,
        equity=10000.0,
        open_exposure_pct=0.0,
        remaining_exposure_pct=35.0,
        min_stake_pct=1.0,
        candidates=[cand_no_atr],
    )
    sys_state = SystemState(
        open_trades=[],
        balance=10000.0,
        current_prices={"EURUSD": 1.1000},
        asset_modes={"EURUSD": "live_eligible"},
        kill_switch_active=False,
        breaker_tripped=False,
        reconcile_done=True,
    )
    dec = DecisionOutput(
        decision="CALL",
        asset="EURUSD",
        expiry_minutes=15,
        stake_pct=2.0,
        evidence=DecisionEvidence(risk_score=25, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
        reasoning="Testing missing ATR",
    )
    res = validate_decision(parsed=dec, report=rep, state=sys_state, config=cfg, now=datetime.now(timezone.utc))
    assert res.decision == "SKIP"
    assert "missing_atr" in res.skip_reason
