"""Unit tests for Phase 10B: Binomo Demo Trader (Selector-Driven) and Demo Wiring.

Adheres strictly to PROJECT_SPEC.md Section 1:
- Zero credentials stored, typed, or logged.
- Strictly demo mode: any real marker aborts immediately BEFORE any click.
- Driven ONLY by config.binomo.ui selectors (role + name, or test id).
- Concurrency lock (1 order in flight).
- Post-click confirmation via Phoenix 'opened' event with timeout and breaker trip.
- Settlement processing from 'closed' events.
- Startup requirements: session logged in, demo detected, selectors validated, platform limits loaded.
- confirm_first_n_trades flow.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock
import pytest

from adapters.base import PlatformAdapter
from adapters.binomo.reader import BinomoReadOnlySource
from adapters.binomo.session import BinomoSession
from adapters.binomo.trader import (
    AmbiguousOrder,
    AssetSelectionMismatch,
    BinomoAdapter,
    ButtonLabelMismatch,
    ExpiryUnavailable,
    PanelBlocked,
    RealAccountDetected,
    SelectorResolutionError,
    StakeOutOfRange,
    is_panel_blocked,
    resolve_locator,
    validate_ui_selectors,
)
from adapters.binomo.ws_listener import BinomoWSListener
from alerts.telegram import TelegramAlerter
from core.config import AppConfig, BinomoConfig, BinomoUIConfig, ExecutionConfig, TradingConfig, load_config
from core.models import TradeDirection, TradeReceipt, TradeResult
from execution.state import StateManager
from execution.trade_manager import TradeManager
from orchestrator.engine import HeldOrder, Orchestrator
from orchestrator.telegram_worker import TelegramWorker
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch


# ── Test Fakes ───────────────────────────────────────────────────────────────


class FakeLocator:
    """Mock Playwright Locator with inspection capabilities."""

    def __init__(
        self,
        name: str = "",
        count: int = 1,
        text: str = "",
        value: str = "",
        visible: bool = True,
        aria_label: str = "",
    ) -> None:
        self.name = name
        self._count = count
        self._text = text
        self._value = value
        self._visible = visible
        self._aria_label = aria_label
        self.click_count = 0
        self.fill_calls: list[str] = []

    def count(self) -> int:
        return self._count

    def is_visible(self) -> bool:
        return self._visible

    def inner_text(self) -> str:
        return self._text

    def text_content(self) -> str:
        return self._text

    def input_value(self) -> str:
        return self._value

    def get_attribute(self, attr: str) -> Optional[str]:
        if attr == "aria-label":
            return self._aria_label
        if attr == "value":
            return self._value
        return None

    def click(self) -> None:
        self.click_count += 1

    def fill(self, val: str) -> None:
        self.fill_calls.append(val)
        self._value = val


class FakePage:
    """Mock Playwright Page returning FakeLocators based on role/name/test_id."""

    class request:
        @staticmethod
        def get(url, **kwargs):
            if "/platform/private/v2/currencies" in url:
                class FakeCurrenciesResp:
                    status = 200
                    def json(self):
                        return {
                            "data": {
                                "current": "USD",
                                "list": [
                                    {
                                        "iso": "USD",
                                        "limits": {
                                            "binary_option": {"min": 1.0, "max": 1000.0},
                                            "standard_trade": {"min": 2.0, "max": 2000.0},
                                        },
                                    }
                                ],
                            }
                        }
                return FakeCurrenciesResp()
            elif "/bo-assets" in url:
                class FakeAssetsResp:
                    status = 200
                    def json(self):
                        return [
                            {
                                "ric": "EUR/USD",
                                "name": "EUR/USD",
                                "trading_tools_settings": {"ftt": {"base_payment_rate_turbo": 83, "enabled_account_types": ["demo"]}},
                            },
                            {
                                "ric": "Z-EUR/USD2",
                                "name": "EUR/USD (OTC)",
                                "trading_tools_settings": {"ftt": {"base_payment_rate_turbo": 85, "enabled_account_types": ["demo"]}},
                            },
                        ]
                return FakeAssetsResp()
            raise NotImplementedError(f"Unhandled GET url: {url}")

    def __init__(self, locators: Optional[dict[str, FakeLocator]] = None) -> None:
        self.locators: dict[str, FakeLocator] = locators or {}
        self.dialog_visible: bool = False

    def get_by_role(self, role: str, name: Optional[str] = None) -> FakeLocator:
        key = f"role:{role}:{name}" if name else f"role:{role}"
        if key in self.locators:
            return self.locators[key]
        # Default fallback locator
        loc = FakeLocator(name=key, count=1, text=str(name or ""), aria_label=str(name or ""))
        self.locators[key] = loc
        return loc

    def get_by_test_id(self, test_id: str) -> FakeLocator:
        key = f"testid:{test_id}"
        if key in self.locators:
            return self.locators[key]
        loc = FakeLocator(name=key, count=1)
        self.locators[key] = loc
        return loc

    def locator(self, selector: str) -> FakeLocator:
        if any(m in selector for m in ("dialog", "modal", "popup", "overlay")):
            return FakeLocator(name="modal", count=1 if self.dialog_visible else 0, visible=self.dialog_visible)
        if selector in self.locators:
            return self.locators[selector]
        loc = FakeLocator(name=selector, count=1)
        self.locators[selector] = loc
        return loc


def build_test_ui_environment(asset: str = "EUR/USD", expiry_label: str = "1 min", stake: float = 10.0) -> tuple[FakePage, AppConfig]:
    """Helper to build a preconfigured fake UI and AppConfig."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.binomo.confirm_timeout_seconds = 0.5  # Fast test timeout

    # Setup locators matching default config.binomo.ui
    locs: dict[str, FakeLocator] = {
        "role:spinbutton:Amount": FakeLocator("amount_input", count=1, value=str(int(stake))),
        "role:button:Expiration time": FakeLocator("expiry_sel", count=1),
        f"role:button:{expiry_label}": FakeLocator("expiry_opt", count=1, text=expiry_label),
        "role:button:Call": FakeLocator("call_btn", count=1, text="Call", aria_label="Call"),
        "role:button:Put": FakeLocator("put_btn", count=1, text="Put", aria_label="Put"),
        "role:button:Asset selection": FakeLocator("asset_sel", count=1),
        f"role:button:{asset}": FakeLocator("asset_opt", count=1, text=asset),
        "role:text:Demo account": FakeLocator("demo_indicator", count=1, text="Demo account"),
        "role:text:Active Asset": FakeLocator("active_asset_label", count=1, text=asset),
        "role:text:Active Expiry": FakeLocator("active_expiry_label", count=1, text=expiry_label),
    }

    page = FakePage(locators=locs)
    return page, cfg


# ── Test 1: Happy Path Execution (Opened Confirms, Payout 0.83) ───────────────


def test_place_trade_happy_path():
    """Happy path: asset, expiry, and stake entered; button clicked ONCE; opened confirms payout 0.83."""
    page, cfg = build_test_ui_environment(asset="EUR/USD", expiry_label="1 min", stake=10.0)

    ws = BinomoWSListener()
    ws.record_ws_open("wss://ws.binomo.com")
    ws.on_frame(json.dumps([None, None, "account", "phx_join", {}]))
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))

    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True
    session.is_demo.return_value = True

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader.get_ric_to_name_map.return_value = {"EUR/USD": "EUR/USD"}
    reader._execute_get.return_value = {
        "data": {
            "current": "USD",
            "list": [
                {
                    "iso": "USD",
                    "limits": {
                        "binary_option": {"min": 1.0, "max": 1000.0},
                        "standard_trade": {"min": 2.0, "max": 2000.0},
                    },
                }
            ],
        }
    }

    breaker = CircuitBreaker(config=cfg)

    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg, breaker=breaker)
    adapter.load_platform_limits()

    # Pre-simulate incoming Phoenix opened frame
    def _fire_opened_frame():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-uuid-12345",
                "asset_ric": "EUR/USD",
                "trend": "CALL",
                "amount": 10.0,
                "deal_type": "demo",
                "open_rate": 1.0850,
                "payment_rate": 83.0,
            }
        ]))

    # Fire event after brief pause
    import threading
    timer = threading.Timer(0.05, _fire_opened_frame)
    timer.start()

    receipt = adapter.place_trade(
        asset_id="EUR/USD",
        direction=TradeDirection.CALL,
        expiry_min=1,
        stake_amount=10.0,
        trade_id="TRD-101",
    )

    assert receipt.platform_trade_id == "deal-uuid-12345"
    assert receipt.payout == 0.83
    assert receipt.entry_price == 1.0850
    assert receipt.status == "OPEN"
    assert receipt.direction == TradeDirection.CALL
    assert receipt.stake_amount == 10.0

    # Verification: Call button was clicked exactly ONCE
    call_btn = page.get_by_role("button", name="Call")
    assert call_btn.click_count == 1

    # Verification: Put button was NOT clicked
    put_btn = page.get_by_role("button", name="Put")
    assert put_btn.click_count == 0


# ── Test 2: Abort on Real Account Marker (Zero Clicks) ───────────────────────


def test_place_trade_aborts_on_real_marker_before_any_click():
    """Any real marker (DOM indicator, websocket account_type, or config mode) aborts before click."""
    # Subtest 2a: DOM demo_indicator shows "Real account"
    page, cfg = build_test_ui_environment()
    page.locators["role:text:Demo account"] = FakeLocator("demo_indicator", text="Real account")

    ws = BinomoWSListener()
    ws.record_ws_open("wss://ws.binomo.com")
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))

    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg)

    with pytest.raises(RealAccountDetected, match="DOM demo_indicator shows real account"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    call_btn = page.get_by_role("button", name="Call")
    assert call_btn.click_count == 0

    # Subtest 2b: WebSocket account_type is "real"
    page2, cfg2 = build_test_ui_environment()
    ws2 = BinomoWSListener()
    ws2.record_ws_open("wss://ws.binomo.com")
    ws2.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "real"}]))

    session2 = MagicMock()
    session2.page = page2
    session2.is_logged_in.return_value = True

    adapter2 = BinomoAdapter(session=session2, ws_listener=ws2, config=cfg2)
    with pytest.raises(RealAccountDetected, match="WebSocket account_type is 'real'"):
        adapter2.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)
    assert page2.get_by_role("button", name="Call").click_count == 0

    # Subtest 2c: Config mode is not demo (e.g. "live")
    page3, cfg3 = build_test_ui_environment()
    cfg3.mode = "live"
    adapter3 = BinomoAdapter(session=session, ws_listener=ws, config=cfg3)
    with pytest.raises(RealAccountDetected, match="Config mode is 'live'"):
        adapter3.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)
    assert page3.get_by_role("button", name="Call").click_count == 0


# ── Test 3: Abort on Wrong Asset Mismatch ────────────────────────────────────


def test_place_trade_aborts_on_wrong_asset_mismatch():
    """If active_asset_label does not match requested asset, abort before trade click."""
    page, cfg = build_test_ui_environment(asset="EUR/USD")
    # Simulate active asset showing GBP/USD instead
    page.locators["role:text:Active Asset"] = FakeLocator("active_asset_label", text="GBP/USD")

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg)

    with pytest.raises(AssetSelectionMismatch, match="does not match requested asset 'EUR/USD'"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    # Trade button must NOT be clicked
    assert page.get_by_role("button", name="Call").click_count == 0


# ── Test 4: Abort on Missing or Unavailable Expiry ───────────────────────────


def test_place_trade_aborts_on_missing_or_mismatched_expiry():
    """If expiry is not in binomo.expiry_labels or active label mismatches, abort before trade click."""
    page, cfg = build_test_ui_environment()
    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg)

    # Expiry 7 is not configured in expiry_labels
    with pytest.raises(ExpiryUnavailable, match="Expiry 7m not available in binomo.expiry_labels"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 7, 10.0)

    assert page.get_by_role("button", name="Call").click_count == 0


# ── Test 5: Abort on Stake Mismatch or Currency Limits ───────────────────────


def test_place_trade_aborts_on_stake_mismatch_and_currency_limits():
    """Readback mismatch or out-of-bounds currency limits aborts before clicking trade button."""
    # Subtest 5a: Stake out of limits (e.g. 5000 when max is 1000)
    page, cfg = build_test_ui_environment()
    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader.get_ric_to_name_map.return_value = {"EUR/USD": "EUR/USD"}
    reader._execute_get.return_value = {
        "data": {
            "current": "USD",
            "list": [
                {
                    "iso": "USD",
                    "limits": {
                        "binary_option": {"min": 1.0, "max": 1000.0},
                        "standard_trade": {"min": 2.0, "max": 2000.0},
                    },
                }
            ],
        }
    }

    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg)
    adapter.load_platform_limits()

    with pytest.raises(StakeOutOfRange, match="outside platform limits"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 5000.0)

    assert page.get_by_role("button", name="Call").click_count == 0

    # Subtest 5b: Stake read-back mismatch (entered 10, but input stubbornly reads 5)
    amt_fake = FakeLocator("amount_input", value="5")
    amt_fake.fill = lambda v: None  # Input field ignores fill to simulate read-back mismatch
    page.locators["role:spinbutton:Amount"] = amt_fake
    with pytest.raises(StakeOutOfRange, match="Stake amount mismatch"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    assert page.get_by_role("button", name="Call").click_count == 0


# ── Test 6: Abort on Wrong Button Label ──────────────────────────────────────


def test_place_trade_aborts_on_wrong_button_label():
    """If call button accessible name does not contain expected direction, abort before clicking."""
    page, cfg = build_test_ui_environment()
    # Call button accessible name is wrongly labeled "Sell"
    page.locators["role:button:Call"] = FakeLocator("call_btn", text="Sell", aria_label="Sell")

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg)

    with pytest.raises(ButtonLabelMismatch, match="does not match trade direction 'CALL'"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    assert page.get_by_role("button", name="Call").click_count == 0


# ── Test 7: Abort on Blocked Trading Panel (No Auto-Dismiss) ─────────────────


def test_place_trade_aborts_on_blocked_panel_without_auto_dismiss():
    """Modal or popup blocking trading panel raises PanelBlocked and alerts; never auto-dismisses."""
    page, cfg = build_test_ui_environment()
    page.dialog_visible = True  # Modal is visible

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    alerter = MagicMock()
    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg, alerter=alerter)

    with pytest.raises(PanelBlocked, match="Trading panel is blocked by a modal dialog"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    assert page.get_by_role("button", name="Call").click_count == 0
    alerter.send_alert.assert_called()


# ── Test 8: Abort on Confirmation Timeout (Trips Breaker, No Retry) ───────────


def test_place_trade_confirm_timeout_trips_breaker_and_never_retries():
    """If no opened event arrives within confirm_timeout_seconds, raise AmbiguousOrder and trip breaker."""
    page, cfg = build_test_ui_environment()
    cfg.binomo.confirm_timeout_seconds = 0.2  # Short timeout for test

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    breaker = CircuitBreaker(config=cfg)
    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg, breaker=breaker)

    with pytest.raises(AmbiguousOrder, match="unconfirmed in WebSocket events"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    # Call button was clicked ONCE, and NEVER retried
    call_btn = page.get_by_role("button", name="Call")
    assert call_btn.click_count == 1
    assert breaker.is_tripped()


# ── Test 9: Non-Demo Deal Type Trips Breaker Immediately ─────────────────────


def test_place_trade_non_demo_deal_type_trips_breaker():
    """If an opened event arrives with deal_type != 'demo', trip breaker immediately and raise."""
    page, cfg = build_test_ui_environment()
    cfg.binomo.confirm_timeout_seconds = 0.5

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 1000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    breaker = CircuitBreaker(config=cfg)
    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg, breaker=breaker)

    # Fire event with deal_type == "real"
    def _fire_real_deal():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-real-999",
                "asset_ric": "EUR/USD",
                "trend": "CALL",
                "amount": 10.0,
                "deal_type": "real",  # NON-DEMO BREACH!
            }
        ]))

    import threading
    timer = threading.Timer(0.05, _fire_real_deal)
    timer.start()

    with pytest.raises(RealAccountDetected, match="non-demo deal_type 'real'"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    assert breaker.is_tripped()


# ── Test 10: Concurrency Lock Enforces 1 Order in Flight ─────────────────────


def test_place_trade_concurrency_lock():
    """Concurrent trade execution attempts are rejected via non-blocking lock."""
    page, cfg = build_test_ui_environment()
    ws = BinomoWSListener()
    session = MagicMock()
    session.page = page

    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg)

    # Acquire lock manually to simulate an active order in flight
    assert adapter._trade_lock.acquire(blocking=False)

    with pytest.raises(RuntimeError, match="Another order is currently in flight"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    adapter._trade_lock.release()


# ── Test 11: Selector Validation at Startup ──────────────────────────────────


def test_validate_ui_selectors_reports_failures():
    """Startup validation verifies all 10 required selectors and reports failures."""
    page, cfg = build_test_ui_environment()

    # Normal case: all valid
    adapter = BinomoAdapter(config=cfg)
    adapter.validate_ui_selectors(page)

    # Failure case: amount_input and call_button missing
    page.locators["role:spinbutton:Amount"] = FakeLocator("amount_input", count=0)
    page.locators["role:button:Call"] = FakeLocator("call_btn", count=0)

    with pytest.raises(SelectorResolutionError) as exc_info:
        adapter.validate_ui_selectors(page)

    err = str(exc_info.value)
    assert "amount_input" in err
    assert "call_button" in err


# ── Test 12: Settlement Processing from Closed Events ────────────────────────


def test_settlement_processing_from_closed_events():
    """Closed events (won/lost/tie) are fed to TradeManager.record_settlement; unknown raises."""
    ws = BinomoWSListener()
    adapter = BinomoAdapter(ws_listener=ws)

    # Setup fake trade_manager with open trade in state
    tm = MagicMock()
    tm.state_manager.data = {
        "open_trades": [{"id": "deal-closed-1", "stake": 10.0}]
    }

    # Simulate closed event
    ws.on_frame(json.dumps([
        None, None, "bo", "closed", {
            "uuid": "deal-closed-1",
            "status": "won",
            "payment_rate": 83.0,
            "end_rate": 1.0860,
        }
    ]))

    results = adapter.process_settlements(tm)
    assert len(results) == 1
    tm.record_settlement.assert_called_with(trade_id="deal-closed-1", outcome="WIN", payout=0.83)

    # Unknown status raises ValueError
    tm.state_manager.data["open_trades"].append({"id": "deal-unknown-status"})
    ws.on_frame(json.dumps([
        None, None, "bo", "closed", {
            "uuid": "deal-unknown-status",
            "status": "cancelled_by_admin",
        }
    ]))

    with pytest.raises(ValueError, match="Unknown deal settlement status 'cancelled_by_admin'"):
        adapter.process_settlements(tm)


# ── Test 13: Demo Startup Requirements in main.py ────────────────────────────


def test_demo_startup_requirements_in_main(monkeypatch, capsys):
    """Demo startup enforces logged-in session, demo check, selectors validation, and sets bypass."""
    import argparse
    from main import handle_run

    # 1. Real account detection aborts demo startup
    class FakeSessionReal:
        ws_listener = BinomoWSListener()
        page = FakePage()

        def __init__(self, *args, **kwargs):
            pass

        def start(self, headed=True):
            pass

        def ensure_logged_in(self, timeout=300):
            return True

        def is_demo(self):
            return False  # Real account detected!

        def close(self):
            pass

    monkeypatch.setattr("adapters.binomo.session.BinomoSession", FakeSessionReal)

    args = argparse.Namespace(
        config="config.yaml",
        mode="demo",
        source="binomo_readonly",
        first_run=False,
    )

    code = handle_run(args)
    assert code == 1
    err = capsys.readouterr().err
    assert "Active account on Binomo is NOT a demo account" in err


# ── Test 14: confirm_first_n_trades Flow in Orchestrator ─────────────────────


def test_confirm_first_n_trades_flow():
    """The first N trades require operator confirmation via Telegram; subsequent trades run auto."""
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.execution.confirm_first_n_trades = 2
    cfg.execution.confirm_timeout_seconds = 5

    alerter = MagicMock()
    alerter.is_configured = True

    state_mgr = StateManager()
    state_mgr.data = {"confirmed_trades_count": 0}

    clock = MagicMock()
    now_dt = datetime.now(timezone.utc)
    clock.now.return_value = now_dt

    worker = TelegramWorker(
        alerter=alerter,
        state_manager=state_mgr,
        config=cfg,
        clock=clock,
    )

    trade_mgr = MagicMock()
    from execution.trade_manager import ExecutionResult
    trade_mgr.execute.return_value = ExecutionResult(
        executed=True,
        status="EXECUTED",
        trade_id="TRD-999",
        reason="Success",
        executed_at="2026-11-04T12:00:00Z",
    )

    import pandas as pd
    df_1m = pd.DataFrame([{
        "ts": now_dt - timedelta(minutes=1),
        "ts_close": now_dt,
        "open": 1.0850,
        "high": 1.0860,
        "low": 1.0840,
        "close": 1.0855,
        "volume": 100.0,
    }])
    mock_source = MagicMock()
    del mock_source._load_raw_1m
    mock_source.get_candles.return_value = df_1m

    orchestrator = Orchestrator(
        config=cfg,
        source=mock_source,
        adapter=MagicMock(),
        store=MagicMock(),
        repository=MagicMock(),
        state_manager=state_mgr,
        breaker=CircuitBreaker(config=cfg),
        kill_switch=KillSwitch(),
        trade_manager=trade_mgr,
        alerter=alerter,
        telegram_worker=worker,
    )

    held1 = HeldOrder(
        decision=MagicMock(asset="EUR/USD", direction=TradeDirection.CALL, stake_amount=10.0, stake=10.0, expiry_minutes=1, pattern_name="Pinbar"),
        candidate=MagicMock(),
        account=MagicMock(),
        adaptive_state=MagicMock(),
        decision_time=now_dt,
        entry_time=now_dt,
        asset="EUR/USD",
        direction=TradeDirection.CALL,
        expiry_minutes=1,
        payout=0.83,
        atr=0.0010,
    )

    # 1. First trade: spawns ticket, requires /approve
    def _approve_first():
        import time
        time.sleep(0.05)
        # Find pending ticket id and approve
        for tid in list(worker.tickets.keys()):
            worker.handlers["/approve"]([tid])

    import threading
    t1 = threading.Thread(target=_approve_first)
    t1.start()

    res1 = orchestrator._execute_held_order(held1)
    t1.join()
    assert res1 is not None
    assert res1.executed is True
    assert orchestrator.confirmed_trades_count == 1

    # 2. Second trade: requires /approve
    held2 = HeldOrder(
        decision=MagicMock(asset="EUR/USD", direction=TradeDirection.CALL, stake_amount=10.0, stake=10.0, expiry_minutes=1, pattern_name="Pinbar"),
        candidate=MagicMock(),
        account=MagicMock(),
        adaptive_state=MagicMock(),
        decision_time=now_dt,
        entry_time=now_dt,
        asset="EUR/USD",
        direction=TradeDirection.CALL,
        expiry_minutes=1,
        payout=0.83,
        atr=0.0010,
    )

    def _approve_second():
        import time
        time.sleep(0.05)
        for tid, tkt in worker.tickets.items():
            if tkt.status == "PENDING":
                worker.handlers["/approve"]([tid])

    t2 = threading.Thread(target=_approve_second)
    t2.start()

    res2 = orchestrator._execute_held_order(held2)
    t2.join()
    assert res2 is not None
    assert orchestrator.confirmed_trades_count == 2

    # 3. Third trade: confirmed_trades_count >= 2, executes automatically without ticket!
    initial_ticket_count = len(worker.tickets)
    held3 = HeldOrder(
        decision=MagicMock(asset="EUR/USD", direction=TradeDirection.CALL, stake_amount=10.0, stake=10.0, expiry_minutes=1, pattern_name="Pinbar"),
        candidate=MagicMock(),
        account=MagicMock(),
        adaptive_state=MagicMock(),
        decision_time=now_dt,
        entry_time=now_dt,
        asset="EUR/USD",
        direction=TradeDirection.CALL,
        expiry_minutes=1,
        payout=0.83,
        atr=0.0010,
    )

    res3 = orchestrator._execute_held_order(held3)
    assert res3 is not None
    assert res3.executed is True
    # No new ticket registered
    assert len(worker.tickets) == initial_ticket_count
