"""Tests for Phase 10A: Binomo Session, Passive Reader, and Calibration.

Adheres strictly to PROJECT_SPEC.md:
- Strictly read-only: grep test asserts zero clicks, fills, types, presses, checks, dblclicks.
- Passive Phoenix frame parsing with real recorded discovery fixtures.
- Demo detection from balance_changed payload.
- Asset parsing with OTC flag and schedule/payout filtering.
- Candle parsing with closed-only zero look-ahead guarantee.
- Rate limiting and fatal stop without retry on HTTP 401, 403, 429.
- Calibration file generation with fake page and error recording.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import MagicMock
import pytest

from adapters.binomo.calibrate import (
    extract_trading_panel_snapshot,
    generate_calibration_markdown,
    mask_calibration_text,
    run_calibration,
)
from adapters.binomo.reader import (
    BinomoReadOnlySource,
    is_otc_instrument,
    is_schedule_open,
)
from adapters.binomo.session import BinomoSession
from adapters.binomo.ws_listener import BinomoWSListener, parse_phoenix_frame
from core.config import AppConfig, BinomoConfig, SourceConfig


# ── Test 1: The Grep Test (Strict Read-Only Verification) ────────────────────

def test_grep_binomo_adapter_contains_no_trade_control_automation():
    """Grep adapters/binomo/ for forbidden interaction methods.
    
    Adheres strictly to Phase 10B allow-list:
    - .click( appears ONLY in trader.py's place_trade path and nowhere else.
    - All other files in adapters/binomo/ have zero interaction calls.
    - No .type(, .press(, .check(, .dblclick( appear anywhere in adapters/binomo/.
    """
    adapter_dir = Path("adapters") / "binomo"
    assert adapter_dir.exists(), "adapters/binomo directory must exist"

    py_files = list(adapter_dir.glob("*.py"))
    assert len(py_files) >= 4, "adapters/binomo must contain session, ws_listener, reader, trader, etc."

    forbidden_pattern = re.compile(r"\.(click|fill|type|press|check|dblclick)\s*\(")

    violations: list[str] = []
    for file_path in py_files:
        content = file_path.read_text(encoding="utf-8")
        if file_path.name != "trader.py":
            matches = forbidden_pattern.findall(content)
            if matches:
                violations.append(f"{file_path.name}: {matches}")
        else:
            # trader.py verification:
            # 1. No interaction calls outside place_trade
            parts = content.split("def place_trade(")
            assert len(parts) == 2, "trader.py must contain exactly one place_trade definition"
            before_place_trade = parts[0]
            after_parts = parts[1].split("\n    def ")
            place_trade_body = after_parts[0]
            outside_place_trade = before_place_trade + ("\n    def ".join(after_parts[1:]) if len(after_parts) > 1 else "")

            outside_matches = forbidden_pattern.findall(outside_place_trade)
            if outside_matches:
                violations.append(f"trader.py outside place_trade: {outside_matches}")

            # 2. Inside place_trade, .click( must appear (confirming allow-list is applied)
            click_matches = re.findall(r"\.click\s*\(", place_trade_body)
            assert len(click_matches) >= 1, "trader.py place_trade must contain allow-listed .click() calls"

            # 3. No other forbidden interaction methods (.type, .press, .check, .dblclick) anywhere in trader.py
            other_forbidden = re.compile(r"\.(type|press|check|dblclick)\s*\(")
            other_matches = other_forbidden.findall(content)
            if other_matches:
                violations.append(f"trader.py: {other_matches}")

    assert not violations, (
        f"Interaction allow-list violation in adapters/binomo! "
        f"Forbidden automated control interactions detected: {violations}"
    )


# ── Test 2: WS Listener Balance Parsing & Demo Detection ─────────────────────

def test_ws_listener_balance_parsing_and_demo_detection():
    """Parse balance_changed frames and verify demo account detection."""
    listener = BinomoWSListener()

    # Initial state
    assert listener.get_balance() is None
    assert listener.get_account_type() is None
    assert listener.is_demo() is False

    # 1. Real recorded demo balance frame
    demo_frame = json.dumps([
        "1",
        "2",
        "account",
        "balance_changed",
        {
            "account_type": "demo",
            "amount": 1000.0,
            "balance": 15420.50,
            "balance_version": 42,
            "currency": "INR",
        },
    ])
    listener.on_frame(demo_frame)

    assert listener.get_balance() == 15420.50
    assert listener.get_account_type() == "demo"
    assert listener.get_currency() == "INR"
    assert listener.is_demo() is True

    # 2. Subsequent real balance frame
    real_frame = json.dumps([
        "1",
        "3",
        "account",
        "balance_changed",
        {
            "account_type": "real",
            "amount": 500.0,
            "balance": 250.0,
            "currency": "USD",
        },
    ])
    listener.on_frame(real_frame)

    assert listener.get_balance() == 250.0
    assert listener.get_account_type() == "real"
    assert listener.get_currency() == "USD"
    assert listener.is_demo() is False


# ── Test 3: WS Listener Deal Events Parsing ─────────────────────────────────

def test_ws_listener_deal_events_parsing():
    """Parse bo opened, closed, close_deal_batch, and phx_reply frames."""
    listener = BinomoWSListener()

    # 1. bo opened frame
    open_frame = json.dumps([
        None,
        None,
        "bo",
        "opened",
        {
            "uuid": "deal-uuid-1111",
            "asset_ric": "Z-CRY/IDX",
            "trend": "call",
            "amount": 100.0,
            "open_rate": 1234.56,
            "payment_rate": 83,
            "status": "open",
            "deal_type": "demo",
            "created_at": "2026-11-01T07:30:57.967068Z",
        },
    ])
    listener.on_frame(open_frame)

    open_deals = listener.get_open_deals()
    assert len(open_deals) == 1
    deal = open_deals[0]
    assert deal["uuid"] == "deal-uuid-1111"
    assert deal["asset_ric"] == "Z-CRY/IDX"
    assert deal["trend"] == "call"
    assert deal["amount"] == 100.0
    assert deal["open_rate"] == 1234.56
    assert deal["payment_rate"] == 83.0
    assert deal["status"] == "open"

    # 2. bo closed frame
    close_frame = json.dumps([
        None,
        None,
        "bo",
        "closed",
        {
            "uuid": "deal-uuid-1111",
            "asset_ric": "Z-CRY/IDX",
            "trend": "call",
            "amount": 100.0,
            "end_rate": 1235.10,
            "status": "win",
            "deal_type": "demo",
            "finished_at": "2026-11-01T07:32:00.000000Z",
        },
    ])
    listener.on_frame(close_frame)

    # Deal must be moved from open to closed
    assert len(listener.get_open_deals()) == 0
    closed_deals = listener.get_closed_deals()
    assert len(closed_deals) == 1
    closed_deal = closed_deals[0]
    assert closed_deal["uuid"] == "deal-uuid-1111"
    assert closed_deal["status"] == "win"
    assert closed_deal["end_rate"] == 1235.10
    assert closed_deal["finished_at"] == "2026-11-01T07:32:00.000000Z"

    # 3. close_deal_batch
    batch_frame = json.dumps([
        None,
        None,
        "bo",
        "close_deal_batch",
        {
            "ric": "Z-CRY/IDX",
            "finished_at": "2026-11-01T07:32:00Z",
            "end_rate": 1235.10,
        },
    ])
    listener.on_frame(batch_frame)
    assert len(listener._batch_close_events) == 1

    # 4. phx_reply
    reply_frame = json.dumps([
        "1",
        "1",
        "bo",
        "phx_reply",
        {"status": "ok", "response": {}},
    ])
    listener.on_frame(reply_frame)
    assert listener.get_last_phx_reply_status() == "ok"


# ── Test 4: Session Login Detection and Timeout ──────────────────────────────

def test_session_login_detection_and_timeout():
    """Verify ensure_logged_in succeeds on valid indicators and times out when missing."""
    mock_listener = BinomoWSListener()
    session = BinomoSession(ws_listener=mock_listener)

    # Initially not logged in
    assert session.is_logged_in() is False

    # Simulate WebSocket to ws.binomo.com open
    mock_listener.record_ws_open("wss://ws.binomo.com/socket/websocket")
    assert session.is_logged_in() is False

    # Simulate account phx_join
    join_frame = json.dumps(["1", "1", "account:12345", "phx_join", {}])
    mock_listener.on_frame(join_frame)

    # Now both conditions met
    assert session.is_logged_in() is True
    assert session.ensure_logged_in(timeout=2.0) is True

    # Test timeout with un-logged-in session
    empty_listener = BinomoWSListener()
    unauthed_session = BinomoSession(ws_listener=empty_listener)
    with pytest.raises(TimeoutError) as exc_info:
        unauthed_session.ensure_logged_in(timeout=0.2, check_interval=0.05)
    assert "Binomo login verification timed out" in str(exc_info.value)


# ── Test 5: Reader Asset Parsing & OTC Flag ──────────────────────────────────

def test_reader_asset_parsing_and_otc_flag():
    """Verify GET /bo-assets/v7/assets parsing, OTC detection, payout extraction, and open schedule."""
    fake_assets_data = [
        {
            "name": "Crypto IDX",
            "ric": "Z-CRY/IDX",
            "trading_tools_settings": {
                "ftt": {
                    "base_payment_rate_turbo": 85,
                    "base_payment_rate_standard": 82,
                    "enabled_account_types": ["demo", "standard", "vip"],
                    "schedule": {"status": "open"},
                }
            },
        },
        {
            "name": "EUR/USD (OTC)",
            "ric": "EURUSD_otc",
            "trading_tools_settings": {
                "ftt": {
                    "base_payment_rate_turbo": 83,
                    "enabled_account_types": ["demo", "vip"],
                }
            },
        },
        {
            "name": "Apple",
            "ric": "AAPL",
            "payment_rate": 80,
            "enabled_account_types": ["demo"],
        },
        {
            # Closed asset -> should be skipped
            "name": "Closed Commodity",
            "ric": "XAUUSD",
            "trading_tools_settings": {
                "ftt": {
                    "base_payment_rate_turbo": 75,
                    "enabled_account_types": ["demo"],
                    "schedule": {"status": "closed"},
                }
            },
        },
        {
            # Real-only asset (no demo) -> should be skipped when demo_required=True
            "name": "VIP Real Only",
            "ric": "VIPCOIN",
            "trading_tools_settings": {
                "ftt": {
                    "base_payment_rate_turbo": 90,
                    "enabled_account_types": ["real", "vip"],
                }
            },
        },
    ]

    class FakeResponse:
        status = 200
        def json(self):
            return fake_assets_data

    class FakePage:
        class request:
            @staticmethod
            def get(url, timeout=None):
                assert "/bo-assets/v7/assets" in url
                return FakeResponse()

    class FakeSession:
        page = FakePage()

    reader = BinomoReadOnlySource(session=FakeSession())
    assets = reader.get_assets()

    assert len(assets) == 3

    # Asset 1: Crypto IDX (OTC)
    a1 = next(a for a in assets if a.asset_id == "Z-CRY/IDX")
    assert a1.is_otc is True
    assert a1.payout == 0.85
    assert a1.is_open is True

    # Asset 2: EUR/USD (OTC)
    a2 = next(a for a in assets if a.asset_id == "EURUSD_otc")
    assert a2.is_otc is True
    assert a2.payout == 0.83

    # Asset 3: Apple (Not OTC)
    a3 = next(a for a in assets if a.asset_id == "AAPL")
    assert a3.is_otc is False
    assert a3.payout == 0.80


# ── Test 6: Reader Candle Parsing & Zero Look-Ahead Guarantee ────────────────

def test_reader_candle_parsing_and_zero_lookahead():
    """Verify /candles/ parsing enforces closed-only bars with zero look-ahead."""
    now_utc = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)

    # Candle 1: Closed (11:50 to 11:55)
    # Candle 2: Closed (11:55 to 12:00)
    # Candle 3: Incomplete / Forming (12:00 to 12:05) -> must be excluded!
    fake_candles = [
        {"created_at": "2026-11-02T11:50:00.000000Z", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 10},
        {"created_at": "2026-11-02T11:55:00.000000Z", "open": 104.0, "high": 106.0, "low": 103.0, "close": 105.5, "volume": 15},
        {"created_at": "2026-11-02T12:00:00.000000Z", "open": 105.5, "high": 108.0, "low": 105.0, "close": 107.0, "volume": 5},
    ]

    class FakeResponse:
        status = 200
        def json(self):
            return fake_candles

    class FakePage:
        class request:
            @staticmethod
            def get(url, timeout=None):
                assert "/candles/v1/" in url
                return FakeResponse()

    class FakeSession:
        page = FakePage()

    reader = BinomoReadOnlySource(session=FakeSession())
    reader.now = lambda: now_utc  # Mock clock to 12:00:00

    df = reader.get_candles(asset="Z-CRY/IDX", timeframe="5m", n=10)

    # Candle 3 closes at 12:05 > 12:00 -> must be filtered out!
    assert len(df) == 2
    assert list(df.columns) == ["ts", "ts_close", "open", "high", "low", "close", "volume"]

    # Verify closed-only guarantee on every row
    for _, row in df.iterrows():
        assert row["ts_close"] <= now_utc
        assert row["ts_close"] == row["ts"] + timedelta(minutes=5)

    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[1]["close"] == 105.5


# ── Test 7: Rate Limit and Fatal Stop on 401/403/429 ──────────────────────────

def test_reader_rate_limit_and_fatal_stop_on_429_401_403():
    """HTTP 429 or 401/403 triggers immediate reader stop and alert without retry loop."""
    mock_alerter = MagicMock()

    class Fake429Response:
        status = 429
        def text(self):
            return "Too Many Requests"

    call_count = 0
    class FakePage:
        class request:
            @staticmethod
            def get(url, timeout=None):
                nonlocal call_count
                call_count += 1
                return Fake429Response()

    class FakeSession:
        page = FakePage()

    reader = BinomoReadOnlySource(session=FakeSession(), alerter=mock_alerter)

    with pytest.raises(RuntimeError) as exc_info:
        reader.get_assets()

    assert "Binomo API HTTP 429" in str(exc_info.value)
    # Must NOT retry
    assert call_count == 1
    assert reader._stopped is True
    assert mock_alerter.send_alert.called

    # Subsequent calls must immediately fail closed without network request
    with pytest.raises(RuntimeError) as exc_info2:
        reader.get_assets()
    assert "is stopped due to prior" in str(exc_info2.value)
    assert call_count == 1


# ── Test 8: Calibration Snapshot Generation with Fake Page ───────────────────

def test_calibration_generation_with_fake_page(tmp_path: Path):
    """Test passive calibration outline generation with fake page and sensitive data masking."""
    fake_aria_yaml = """
- button "Call"
- button "Put"
- textbox "Amount $10,500.00"
- combobox "Expiry 5 min"
- button "Asset Crypto IDX (85%)"
- button "Account Switch Demo"
- menu "Profile user@domain.com":
  - menuitem "Settings 12345678"
"""

    class MockLocator:
        def aria_snapshot(self):
            return fake_aria_yaml

    class MockPage:
        url = "https://binomo.com/en/trading?query=secret_token"
        def locator(self, selector):
            assert selector == "body"
            return MockLocator()

    class FakeSession:
        page = MockPage()
        ws_listener = BinomoWSListener()
        def is_demo(self):
            return True

    # Pre-populate session with login indicators
    FakeSession.ws_listener.record_ws_open("wss://ws.binomo.com/socket/websocket")
    FakeSession.ws_listener.on_frame(json.dumps(["1", "1", "account:1", "phx_join", {}]))

    reports = run_calibration(
        session=FakeSession(),
        page=MockPage(),
        reports_dir=tmp_path,
        duration_seconds=5,
        interval_seconds=1,
        max_iterations=1,
    )

    assert len(reports) == 1
    report_file = reports[0]
    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")

    # URL path without query
    assert "- **Current URL Path**: `/en/trading`" in content
    assert "secret_token" not in content

    # Logged-in indicator
    assert "- **Logged-in indicators**: yes" in content
    assert "- **Account Type**: DEMO" in content

    # Controls outline
    assert "Call" in content
    assert "Put" in content
    assert "Amount <num>" in content
    assert "Expiry 5 min" in content

    # Excluded areas and masked secrets
    assert "user@domain.com" not in content
    assert "Profile" not in content
    assert "12345678" not in content


# ── Test 9: Calibration Error Written to Report ──────────────────────────────

def test_calibration_error_written_to_report(tmp_path: Path):
    """When DOM inspection raises an error, it is recorded in the report instead of failing silently."""
    class BrokenPage:
        url = "https://binomo.com/trading"
        def locator(self, selector):
            raise RuntimeError("Playwright connection closed")

    items, error_msg = extract_trading_panel_snapshot(BrokenPage())
    assert len(items) == 0
    assert error_msg is not None
    assert "Playwright connection closed" in error_msg

    content = generate_calibration_markdown(
        url_path="/trading",
        login_indicators={"logged_in": False},
        is_demo=False,
        assets=[],
        controls=[],
        error_msg=error_msg,
    )

    assert "- **Snapshot Error**: `RuntimeError: Playwright connection closed`" in content


# ── Test 10: Orchestrator Integration & Startup Safeguards ───────────────────

def test_binomo_readonly_source_record_mode_syncs_datastore(tmp_path: Path):
    """source.name binomo_readonly records candles and quotes into DataStore."""
    from data.store import DataStore

    db_path = tmp_path / "test_market.db"
    store = DataStore(db_path=db_path)

    now_utc = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
    fake_candles = [
        {"created_at": "2026-11-02T11:58:00.000000Z", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.8, "volume": 12},
        {"created_at": "2026-11-02T11:59:00.000000Z", "open": 100.8, "high": 102.0, "low": 100.5, "close": 101.5, "volume": 18},
    ]

    class FakeResponse:
        status = 200
        def json(self):
            return fake_candles

    class FakePage:
        class request:
            @staticmethod
            def get(url, timeout=None):
                return FakeResponse()

    class FakeSession:
        page = FakePage()

    reader = BinomoReadOnlySource(session=FakeSession())
    reader.now = lambda: now_utc

    # Sync candles from source into store
    df = store.sync_candles_from_source(source=reader, asset_id="Z-CRY/IDX", timeframe="1m", n_needed=2)
    assert len(df) == 2

    # Query stored candles from SQLite database
    stored_df = store.get_candles(asset_id="Z-CRY/IDX", timeframe="1m", n=5)
    assert len(stored_df) == 2
    assert stored_df.iloc[-1]["close"] == 101.5


def test_main_startup_rejects_live_mode_in_phase_10b(capsys):
    """Startup refuses live mode with fail-closed message."""
    import argparse
    from main import handle_run

    args_live = argparse.Namespace(
        config="config.yaml",
        mode="live",
        source="binomo_readonly",
        first_run=False,
    )
    code_live = handle_run(args_live)
    assert code_live == 1
    err_live = capsys.readouterr().err
    assert "Startup refused: mode 'live' is rejected in Phase 10B" in err_live

