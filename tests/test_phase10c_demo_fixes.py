"""Unit tests for Phase 10C: Demo Trader Fixes.

Covers all 8 Phase 10C requirements:
1. SETTLEMENT IDS: Real TradeManager and real state file; platform_trade_id -> internal id mapping;
   payout from deal (never default); unknown uuids logged once and skipped; no double settlement.
2. ASSET NAMES: ric-to-display-name map from reader's asset table; UI selection and active label use
   display name; confirmation uses deal ric; missing name aborts before any click (Z-EUR/USD2 vs EUR/USD (OTC)).
3. STAKE LIMITS: Parse /platform/private/v2/currencies as data.list[] with iso == data.current;
   limits.<product>.min/max; product from binomo.limits_product (binary_option or standard_trade);
   log both products; raise StakeLimitsUnavailable on parse failure (no default limits).
4. AMOUNT SCALE: binomo.amount_scale applied to balance_changed.balance and deal amounts everywhere;
   100x or 1/100x amount mismatch triggers amount scale suspected alert and raises AmbiguousOrder.
5. BUTTONS: binomo.ui.call_button_labels / put_button_labels (case-insensitive contains);
   empty list refuses to trade.
6. REMOVE HARDCODED VALUES: no 83.0 payout fallback; empty expiry_labels refuses; no default platform limits;
   binomo.ui.blocking_overlays controls overlay check; validation samples from config.
7. EXPIRY & ENTRY TIME: derive expiry_minutes and entry_time from deal created_at and close_quote_created_at/finished_at.
8. CONFIRM FIRST N TRADES: first N wait for Telegram approval, N+1th runs automatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import threading
import time
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
    StakeLimitsUnavailable,
    StakeOutOfRange,
    is_panel_blocked,
    parse_deal_timestamp,
    resolve_locator,
    validate_ui_selectors,
)
from adapters.binomo.ws_listener import BinomoWSListener
from alerts.telegram import TelegramAlerter
from core.config import AppConfig, BinomoConfig, BinomoUIConfig, ExecutionConfig, TradingConfig, load_config
from core.models import AssetQuote, TradeDirection, TradeReceipt, TradeResult
from execution.state import StateManager
from execution.trade_manager import ExecutionResult, TradeManager
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
        if attr == "name":
            return self.name
        return None

    def fill(self, val: str) -> None:
        self.fill_calls.append(val)
        self._value = val

    def click(self) -> None:
        self.click_count += 1


class FakePage:
    """Mock Playwright Page with request context returning discovery fixtures."""

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
                                            "standard_trade": {"min": 5.0, "max": 5000.0},
                                        },
                                    },
                                    {
                                        "iso": "EUR",
                                        "limits": {
                                            "binary_option": {"min": 1.0, "max": 900.0},
                                            "standard_trade": {"min": 5.0, "max": 4500.0},
                                        },
                                    },
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


def build_test_ui(asset_display_name: str = "EUR/USD", expiry_label: str = "1 min", stake: float = 10.0) -> tuple[FakePage, AppConfig]:
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.binomo.confirm_timeout_seconds = 0.5

    locs: dict[str, FakeLocator] = {
        "role:spinbutton:Amount": FakeLocator("amount_input", count=1, value=str(int(stake))),
        "role:button:Expiration time": FakeLocator("expiry_sel", count=1),
        f"role:button:{expiry_label}": FakeLocator("expiry_opt", count=1, text=expiry_label),
        "role:button:Call": FakeLocator("call_btn", count=1, text="Call", aria_label="Call"),
        "role:button:Put": FakeLocator("put_btn", count=1, text="Put", aria_label="Put"),
        "role:button:Asset selection": FakeLocator("asset_sel", count=1),
        f"role:button:{asset_display_name}": FakeLocator("asset_opt", count=1, text=asset_display_name),
        "role:text:Demo account": FakeLocator("demo_indicator", count=1, text="Demo account"),
        "role:text:Active Asset": FakeLocator("active_asset_label", count=1, text=asset_display_name),
        "role:text:Active Expiry": FakeLocator("active_expiry_label", count=1, text=expiry_label),
    }

    page = FakePage(locators=locs)
    return page, cfg


# ── 1. SETTLEMENT IDS TESTS ──────────────────────────────────────────────────


def test_process_settlements_maps_platform_uuid_to_internal_id_with_real_trade_manager(tmp_path: Path):
    """process_settlements maps platform uuid to internal id via platform_trade_id;

    calls record_settlement with internal id and deal payout (never default).
    Unknown uuids logged once and skipped. No double settlement.
    Tested with real TradeManager and real state file.
    """
    state_file = tmp_path / "trading_state.json"
    state_mgr = StateManager(state_file=state_file)

    # Initial state with one open trade
    internal_id = "INTERNAL-TRD-42"
    platform_uuid = "binomo-deal-uuid-999"

    state_mgr.update(
        balance=4900.0,
        equity=5000.0,
        open_trades=[
            {
                "id": internal_id,
                "platform_trade_id": platform_uuid,
                "asset": "EURUSD",
                "direction": "CALL",
                "expiry_minutes": 1,
                "stake": 100.0,
                "payout": 0.85,
                "status": "OPEN",
                "opened_at": "2026-11-05T12:00:00Z",
            }
        ],
    )

    cfg = load_config("config.yaml")
    breaker = CircuitBreaker(config=cfg)
    kill = KillSwitch()
    adapter = BinomoAdapter(config=cfg)

    tm = TradeManager(
        config=cfg,
        state_manager=state_mgr,
        breaker=breaker,
        kill_switch=kill,
        adapter=adapter,
    )

    ws = adapter.ws_listener

    # 1. Unknown uuid is logged once and skipped
    ws.on_frame(json.dumps([
        None, None, "bo", "closed", {
            "uuid": "unknown-uuid-777",
            "status": "won",
            "payment_rate": 80.0,
        }
    ]))

    # 2. Known uuid with deal payment_rate 85.0
    ws.on_frame(json.dumps([
        None, None, "bo", "closed", {
            "uuid": platform_uuid,
            "status": "won",
            "payment_rate": 85.0,
            "end_rate": 1.0865,
        }
    ]))

    results = adapter.process_settlements(tm)
    assert len(results) == 1
    assert results[0]["trade_id"] == internal_id
    assert results[0]["outcome"] == "WIN"
    assert results[0]["profit"] == 85.0

    # Check state updates: open_trades cleared, balance credited with stake + profit (100 + 85 = 185)
    assert len(state_mgr.data["open_trades"]) == 0
    assert state_mgr.data["balance"] == 4900.0 + 100.0 + 85.0

    # Verify state file persisted to disk
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(persisted["open_trades"]) == 0
    assert persisted["balance"] == 5085.0

    # 3. Double settlement prevention: calling again returns empty list and balance unchanged
    results2 = adapter.process_settlements(tm)
    assert len(results2) == 0
    assert state_mgr.data["balance"] == 5085.0

    # 4. Unknown uuid was tracked so not logged repeatedly
    assert "unknown-uuid-777" in adapter._logged_unknown_deal_uuids


# ── 2. ASSET NAMES TESTS (Z-EUR/USD2 vs EUR/USD (OTC)) ───────────────────────


def test_asset_names_mapping_and_ui_selection():
    """Builds ric-to-display-name map from reader's asset table.

    UI selection and active label use display name 'EUR/USD (OTC)';
    confirmation uses RIC 'Z-EUR/USD2'.
    Missing name aborts before any click.
    """
    page, cfg = build_test_ui(asset_display_name="EUR/USD (OTC)", expiry_label="1 min", stake=10.0)

    ws = BinomoWSListener()
    ws.record_ws_open("wss://ws.binomo.com")
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))

    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True
    session.is_demo.return_value = True

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader.get_ric_to_name_map.return_value = {
        "Z-EUR/USD2": "EUR/USD (OTC)",
        "EUR/USD": "EUR/USD",
    }
    reader._execute_get.return_value = {
        "data": {
            "current": "USD",
            "list": [
                {
                    "iso": "USD",
                    "limits": {
                        "binary_option": {"min": 1.0, "max": 1000.0},
                        "standard_trade": {"min": 5.0, "max": 5000.0},
                    },
                }
            ],
        }
    }

    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg)
    adapter.load_platform_limits()

    # Pre-simulate incoming deal with RIC "Z-EUR/USD2"
    def _fire_opened():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-otc-123",
                "asset_ric": "Z-EUR/USD2",
                "trend": "CALL",
                "amount": 10.0,
                "deal_type": "demo",
                "open_rate": 1.0920,
                "payment_rate": 85.0,
            }
        ]))

    timer = threading.Timer(0.05, _fire_opened)
    timer.start()

    receipt = adapter.place_trade(
        asset_id="Z-EUR/USD2",
        direction=TradeDirection.CALL,
        expiry_min=1,
        stake_amount=10.0,
    )

    # UI selection clicked the display name "EUR/USD (OTC)"
    opt_btn = page.get_by_role("button", name="EUR/USD (OTC)")
    assert opt_btn.click_count == 1

    # Confirmation matched the RIC
    assert receipt.platform_trade_id == "deal-otc-123"
    assert receipt.asset_id == "Z-EUR/USD2"
    assert receipt.payout == 0.85


def test_asset_names_missing_name_aborts_before_any_click():
    """If RIC is not present in reader's asset table, abort before clicking anything."""
    page, cfg = build_test_ui()
    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))

    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    reader = MagicMock()
    reader.get_ric_to_name_map.return_value = {"EUR/USD": "EUR/USD"}
    reader.base_api_url = "https://api.binomo.com"
    reader._execute_get.return_value = {
        "data": {
            "current": "USD",
            "list": [{"iso": "USD", "limits": {"binary_option": {"min": 1.0, "max": 1000.0}}}],
        }
    }

    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg)
    adapter.load_platform_limits()

    with pytest.raises(AssetSelectionMismatch, match="No display name found in asset table for RIC 'UNKNOWN-RIC'"):
        adapter.place_trade("UNKNOWN-RIC", TradeDirection.CALL, 1, 10.0)

    # Assert ZERO clicks occurred
    assert page.get_by_role("button", name="Asset selection").click_count == 0
    assert page.get_by_role("button", name="Call").click_count == 0


# ── 3. STAKE LIMITS TESTS ────────────────────────────────────────────────────


def test_stake_limits_parsing_discovery_focus_structure():
    """Parse /platform/private/v2/currencies as data.list[] with iso == data.current;

    logs both products' limits once and selects configured product.
    """
    currencies_payload = {
        "data": {
            "current": "USD",
            "list": [
                {
                    "iso": "USD",
                    "limits": {
                        "binary_option": {"min": 1.0, "max": 1000.0},
                        "standard_trade": {"min": 5.0, "max": 5000.0},
                    },
                },
                {
                    "iso": "EUR",
                    "limits": {
                        "binary_option": {"min": 1.0, "max": 900.0},
                        "standard_trade": {"min": 5.0, "max": 4500.0},
                    },
                },
            ],
        }
    }

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader._execute_get.return_value = currencies_payload

    # Subtest 3a: Default product binary_option
    cfg = load_config("config.yaml")
    cfg.binomo.limits_product = "binary_option"
    adapter = BinomoAdapter(reader=reader, config=cfg)
    min_s, max_s = adapter.load_platform_limits()
    assert min_s == 1.0
    assert max_s == 1000.0

    # Subtest 3b: Configured product standard_trade
    cfg.binomo.limits_product = "standard_trade"
    adapter_st = BinomoAdapter(reader=reader, config=cfg)
    min_st, max_st = adapter_st.load_platform_limits()
    assert min_st == 5.0
    assert max_st == 5000.0


def test_stake_limits_unparseable_raises_stake_limits_unavailable():
    """If limits cannot be parsed, raise StakeLimitsUnavailable and refuse to trade (no defaults)."""
    broken_payload = {
        "data": {
            "current": "USD",
            "list": [{"iso": "GBP", "limits": {}}],  # Current currency not found!
        }
    }

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader._execute_get.return_value = broken_payload

    cfg = load_config("config.yaml")
    adapter = BinomoAdapter(reader=reader, config=cfg)

    with pytest.raises(StakeLimitsUnavailable, match="Currency 'USD' not found"):
        adapter.load_platform_limits()

    # Confirms no default limits exist
    assert adapter._platform_min_stake is None
    assert adapter._platform_max_stake is None


# ── 4. AMOUNT SCALE TESTS ────────────────────────────────────────────────────


def test_amount_scale_applied_to_balance_and_deals():
    """binomo.amount_scale config applies to balance_changed and deal amounts."""
    cfg = load_config("config.yaml")
    cfg.binomo.amount_scale = 0.01  # e.g., cents to dollars

    ws = BinomoWSListener(amount_scale=0.01)
    ws.record_ws_open("wss://ws.binomo.com")

    # 1. Balance scaling
    ws.on_frame(json.dumps([
        None, None, "account", "balance_changed", {
            "balance": 500000.0,  # 500,000 cents
            "account_type": "demo",
        }
    ]))
    assert ws.get_balance() == 5000.0

    # 2. Deal amount scaling
    ws.on_frame(json.dumps([
        None, None, "bo", "opened", {
            "uuid": "deal-scale-1",
            "amount": 2500.0,  # 2,500 cents
            "asset_ric": "EUR/USD",
            "trend": "CALL",
            "deal_type": "demo",
        }
    ]))
    deal = ws.get_open_deals()[0]
    assert deal["raw_amount"] == 2500.0
    assert deal["amount"] == 25.0


def test_amount_scale_suspected_100x_alerts_and_raises_ambiguous_order():
    """If opened event matches ric, trend, and demo type but amount is 100x stake, alert and raise AmbiguousOrder."""
    page, cfg = build_test_ui(stake=10.0)
    cfg.binomo.amount_scale = 1.0  # expected scale 1

    ws = BinomoWSListener()
    ws.record_ws_open("wss://ws.binomo.com")
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
            "list": [{"iso": "USD", "limits": {"binary_option": {"min": 1.0, "max": 2000.0}}}],
        }
    }

    alerter = MagicMock()
    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg, alerter=alerter)
    adapter.load_platform_limits()

    # Incoming opened event has raw amount 1000.0 (100 * 10.0 stake)
    def _fire_scaled_deal():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-suspect-100x",
                "asset_ric": "EUR/USD",
                "trend": "CALL",
                "amount": 1000.0,  # 100x stake!
                "deal_type": "demo",
                "payment_rate": 83.0,
            }
        ]))

    timer = threading.Timer(0.05, _fire_scaled_deal)
    timer.start()

    with pytest.raises(AmbiguousOrder, match="amount scale suspected: raw 1000.0, expected 10.0"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    alerter.send_alert.assert_called_with("amount scale suspected: raw 1000.0, expected 10.0")


def test_amount_scale_suspected_div100_alerts_and_raises_ambiguous_order():
    """If opened event matches ric, trend, and demo type but amount is 1/100 stake, alert and raise AmbiguousOrder."""
    page, cfg = build_test_ui(stake=100.0)
    cfg.binomo.amount_scale = 1.0

    ws = BinomoWSListener()
    ws.record_ws_open("wss://ws.binomo.com")
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
            "list": [{"iso": "USD", "limits": {"binary_option": {"min": 1.0, "max": 2000.0}}}],
        }
    }

    alerter = MagicMock()
    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg, alerter=alerter)
    adapter.load_platform_limits()

    # Incoming opened event has raw amount 1.0 (1/100 of 100.0 stake)
    def _fire_scaled_deal():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-suspect-div100",
                "asset_ric": "EUR/USD",
                "trend": "CALL",
                "amount": 1.0,  # 1/100x stake!
                "deal_type": "demo",
                "payment_rate": 83.0,
            }
        ]))

    timer = threading.Timer(0.05, _fire_scaled_deal)
    timer.start()

    with pytest.raises(AmbiguousOrder, match="amount scale suspected: raw 1.0, expected 100.0"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 100.0)

    alerter.send_alert.assert_called_with("amount scale suspected: raw 1.0, expected 100.0")


# ── 5. BUTTONS TESTS ─────────────────────────────────────────────────────────


def test_button_label_case_insensitive_contains():
    """Button label matches configured call_button_labels and put_button_labels case-insensitively."""
    page, cfg = build_test_ui()
    cfg.binomo.ui.call_button_labels = ["higher", "up"]
    # Accessible name is "Click to go Up now"
    page.locators["role:button:Call"] = FakeLocator("call_btn", text="Click to go Up now", aria_label="Click to go Up now")

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))

    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader.get_ric_to_name_map.return_value = {"EUR/USD": "EUR/USD"}
    reader._execute_get.return_value = {
        "data": {
            "current": "USD",
            "list": [{"iso": "USD", "limits": {"binary_option": {"min": 1.0, "max": 1000.0}}}],
        }
    }

    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg)
    adapter.load_platform_limits()

    # Pre-simulate deal
    def _fire_deal():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-up-btn",
                "asset_ric": "EUR/USD",
                "trend": "CALL",
                "amount": 10.0,
                "deal_type": "demo",
                "payment_rate": 80.0,
            }
        ]))

    timer = threading.Timer(0.05, _fire_deal)
    timer.start()

    receipt = adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)
    assert receipt.direction == TradeDirection.CALL
    assert page.get_by_role("button", name="Call").click_count == 1


def test_button_label_empty_list_refuses_to_trade():
    """Empty call_button_labels or put_button_labels refuses to trade before clicking."""
    page, cfg = build_test_ui()
    cfg.binomo.ui.call_button_labels = []  # Empty!

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    reader = MagicMock()
    reader.base_api_url = "https://api.binomo.com"
    reader.get_ric_to_name_map.return_value = {"EUR/USD": "EUR/USD"}
    reader._execute_get.return_value = {
        "data": {
            "current": "USD",
            "list": [{"iso": "USD", "limits": {"binary_option": {"min": 1.0, "max": 1000.0}}}],
        }
    }

    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg)
    adapter.load_platform_limits()

    with pytest.raises(ButtonLabelMismatch, match="Refusing to trade: binomo.ui.call_button_labels is empty"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    assert page.get_by_role("button", name="Call").click_count == 0


# ── 6. REMOVE HARDCODED VALUES TESTS ─────────────────────────────────────────


def test_missing_payout_raises_and_alerts_never_defaults_83():
    """Missing payment_rate / payout on confirmed deal alerts and raises ValueError; never defaults to 83.0."""
    page, cfg = build_test_ui()
    ws = BinomoWSListener()
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
            "list": [{"iso": "USD", "limits": {"binary_option": {"min": 1.0, "max": 1000.0}}}],
        }
    }

    alerter = MagicMock()
    adapter = BinomoAdapter(session=session, ws_listener=ws, reader=reader, config=cfg, alerter=alerter)
    adapter.load_platform_limits()

    # Confirmed deal without payment_rate
    def _fire_no_payout_deal():
        ws.on_frame(json.dumps([
            None, None, "bo", "opened", {
                "uuid": "deal-no-payout",
                "asset_ric": "EUR/USD",
                "trend": "CALL",
                "amount": 10.0,
                "deal_type": "demo",
                # "payment_rate" missing!
            }
        ]))

    timer = threading.Timer(0.05, _fire_no_payout_deal)
    timer.start()

    with pytest.raises(ValueError, match="payment_rate/payout is missing"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    alerter.send_alert.assert_called()


def test_empty_expiry_labels_refuses_to_trade():
    """Empty expiry_labels in config refuses to trade before clicking."""
    page, cfg = build_test_ui()
    cfg.binomo.expiry_labels = {}  # Empty

    ws = BinomoWSListener()
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))
    session = MagicMock()
    session.page = page
    session.is_logged_in.return_value = True

    adapter = BinomoAdapter(session=session, ws_listener=ws, config=cfg)

    with pytest.raises(ExpiryUnavailable, match="Refusing to trade: binomo.expiry_labels is empty"):
        adapter.place_trade("EUR/USD", TradeDirection.CALL, 1, 10.0)

    assert page.get_by_role("button", name="Call").click_count == 0


def test_blocking_overlays_configuration_and_startup_warning(caplog):
    """Empty blocking_overlays logs warning at startup and skips check; non-empty checks locator."""
    # 1. Empty overlays logs warning at startup
    page, cfg = build_test_ui()
    cfg.binomo.ui.blocking_overlays = []

    with caplog.at_level(logging.WARNING):
        validate_ui_selectors(page, cfg.binomo.ui)
    assert any("blocking_overlays is empty" in rec.message for rec in caplog.records)

    # 2. Empty overlays returns False from is_panel_blocked even if dialog exists
    page.dialog_visible = True
    assert is_panel_blocked(page, []) is False

    # 3. Configured overlays detects visible modal
    assert is_panel_blocked(page, ["[role='dialog']"]) is True


# ── 7. EXPIRY MINUTES & ENTRY TIME TESTS ─────────────────────────────────────


def test_derive_expiry_minutes_and_entry_time_from_deal():
    """get_open_trades and get_trade_history derive expiry_minutes from created_at to

    close_quote_created_at/finished_at rounded to minutes, and entry_time from created_at.
    """
    ws = BinomoWSListener()
    ws.record_ws_open("wss://ws.binomo.com")
    ws.on_frame(json.dumps([None, None, "account", "balance_changed", {"balance": 5000.0, "account_type": "demo"}]))

    # 1. Open deal: created_at 07:30, close_quote_created_at 07:35 -> 5 min expiry
    ws.on_frame(json.dumps([
        None, None, "bo", "opened", {
            "uuid": "deal-open-5m",
            "asset_ric": "EUR/USD",
            "trend": "CALL",
            "amount": 10.0,
            "open_rate": 1.0850,
            "deal_type": "demo",
            "created_at": "2026-11-01T07:30:00Z",
            "close_quote_created_at": "2026-11-01T07:35:00Z",
            "payment_rate": 83.0,
        }
    ]))

    cfg = load_config("config.yaml")
    adapter = BinomoAdapter(ws_listener=ws, config=cfg)
    open_trades = adapter.get_open_trades()
    assert len(open_trades) == 1
    t = open_trades[0]
    assert t.expiry_minutes == 5
    assert t.entry_time == datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc)

    # 2. Closed deal: created_at 08:00, finished_at 08:15 -> 15 min expiry
    ws.on_frame(json.dumps([
        None, None, "bo", "closed", {
            "uuid": "deal-closed-15m",
            "asset_ric": "EUR/USD",
            "trend": "PUT",
            "amount": 20.0,
            "open_rate": 1.0850,
            "end_rate": 1.0830,
            "status": "won",
            "deal_type": "demo",
            "created_at": "2026-11-01T08:00:00Z",
            "finished_at": "2026-11-01T08:15:00Z",
            "payment_rate": 85.0,
        }
    ]))

    history = adapter.get_trade_history()
    assert len(history) == 1
    h = history[0]
    assert h.expiry_minutes == 15
    assert h.entry_time == datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)


# ── 8. CONFIRM FIRST N TRADES ENFORCEMENT TEST ────────────────────────────────


def test_confirm_first_n_trades_first_n_wait_n_plus_one_auto():
    """First N orders wait for Telegram ticket approval, N+1th order executes automatically."""
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
    trade_mgr.execute.return_value = ExecutionResult(
        executed=True,
        status="EXECUTED",
        trade_id="TRD-1",
        reason="ok",
        executed_at="2026-11-05T12:00:00Z",
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

    def _make_held_order():
        return HeldOrder(
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

    # 1. Order 1: waits for approval
    def _approve_ticket(tid: str):
        worker.handlers["/approve"]([tid])

    def _auto_approver():
        time.sleep(0.05)
        for tid in list(worker.tickets.keys()):
            if worker.tickets[tid].status == "PENDING":
                _approve_ticket(tid)

    th1 = threading.Thread(target=_auto_approver)
    th1.start()
    res1 = orchestrator._execute_held_order(_make_held_order())
    th1.join()
    assert res1.executed is True
    assert orchestrator.confirmed_trades_count == 1

    # 2. Order 2: waits for approval
    th2 = threading.Thread(target=_auto_approver)
    th2.start()
    res2 = orchestrator._execute_held_order(_make_held_order())
    th2.join()
    assert res2.executed is True
    assert orchestrator.confirmed_trades_count == 2

    # 3. Order 3 (N+1th): runs automatically without ticket creation!
    tickets_before = len(worker.tickets)
    res3 = orchestrator._execute_held_order(_make_held_order())
    assert res3.executed is True
    assert len(worker.tickets) == tickets_before  # No new ticket registered!
