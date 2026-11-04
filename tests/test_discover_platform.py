"""Tests for tools/discover_platform.py.

PROJECT_SPEC Part A & Hardening:
  - Tool must be strictly read-only: no click, fill, type, press, check, evaluate calls.
  - Redaction: fake payloads containing tokens, JWT, email, account id, balance never appear in output.
  - Sensitive subtree masking: if a key matches the sensitive list (user, account, token,
    session, auth, email, phone, uid, ssid, key, secret, password, cookie, jwt), mask the
    entire subtree under it. Allowlisted keys only shown when no ancestor key is sensitive.
  - Key 'name' rules: allowlist key 'name' only shown when an ancestor key looks like an asset
    (asset, instrument, active, pair, symbol, ticker, market); otherwise masked.
  - Second page / popup handling: context.on('page') attaches listeners to new tabs/popups.
  - Binary frames: handle count, length, first 16 bytes as hex.
  - Classification: categories based on key names.
  - URL query stripping and auth URL skipping.
  - CLI: --url is required.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from tools.discover_platform import (
    CATEGORY_KEYS,
    PayloadRedactor,
    PlatformDiscovery,
    SAFE_NUMERIC_KEYS,
    SAFE_STRING_KEYS,
    classify_category_from_keys,
    classify_message,
    extract_message_type,
    is_identifier_string,
    is_i18n_payload,
    mask_dom_text,
)


def test_discover_platform_is_strictly_read_only():
    """Ensure tools/discover_platform.py never invokes click, fill, type, press, check, or evaluate."""
    src_file = Path("tools") / "discover_platform.py"
    content = src_file.read_text(encoding="utf-8")

    forbidden_pattern = re.compile(r"\.(click|fill|type|press|check|evaluate)\s*\(")
    matches = forbidden_pattern.findall(content)
    assert not matches, f"Strictly read-only violation! Found forbidden Playwright method calls: {matches}"


def test_redaction_removes_tokens_jwt_email_account_and_masks_balance():
    """Redaction test with fake payloads: tokens, JWT, email, phone, IP, account ID, and balance must not appear."""
    redactor = PayloadRedactor()

    fake_payload = {
        "action": "subscribe",
        "symbol": "EURUSD",
        "price": 1.0855,
        "token": "secret_session_token_xyz987",
        "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFsaWNlIn0.g_7g9_z8x-K3L",
        "user_email": "trader_alice@securebank.com",
        "phone_number": "+1-555-019-2834",
        "ip_address": "192.168.1.100",
        "account_id": 99887766,
        "balance": 15420.50,
        "demo_balance": 10000.0,
        "quote": {
            "payout": 0.88,
        },
        "nested_user": {
            "uid": "usr_999_secret",
            "password_hash": "a1b2c3d4e5f67890abcdef1234567890",
            "payout": 0.88,
        },
        "random_arbitrary_key": "some_arbitrary_string_value",
    }

    redacted = redactor.redact(fake_payload)
    output_str = json.dumps(redacted)

    # 1. Assert NONE of the sensitive values appear in the output
    assert "secret_session_token_xyz987" not in output_str
    assert "eyJhbGciOiJIUzI1Ni" not in output_str
    assert "trader_alice@securebank.com" not in output_str
    assert "+1-555-019-2834" not in output_str
    assert "192.168.1.100" not in output_str
    assert "99887766" not in output_str
    assert "15420.5" not in output_str
    assert "usr_999_secret" not in output_str
    assert "a1b2c3d4e5f67890abcdef1234567890" not in output_str
    assert "some_arbitrary_string_value" not in output_str

    # 2. Assert safe fields are preserved when not under sensitive ancestor
    assert redacted["action"] == "subscribe"
    assert redacted["symbol"] == "EURUSD"
    assert redacted["price"] == 1.0855
    assert redacted["quote"]["payout"] == 0.88

    # 3. Assert balance numeric values are masked
    assert redacted["balance"] == "<num>"
    assert redacted["demo_balance"] == "<num>"

    # 4. Assert token/auth/user strings are masked as <str:len>
    assert redacted["token"] == f"<str:{len('secret_session_token_xyz987')}>"
    assert redacted["user_email"] == f"<str:{len('trader_alice@securebank.com')}>"

    # 5. Assert nested_user subtree has payout masked (<num>) because ancestor is sensitive
    assert redacted["nested_user"]["payout"] == "<num>"

    # Redactions were counted
    assert redactor.redaction_count >= 8


def test_nested_user_subtree_masking_and_name_key_rules():
    """Mask entire subtree under sensitive keys, and only show 'name' when ancestor is an asset."""
    redactor = PayloadRedactor()

    # 1. Nested user subtree: {"user": {"name": ...}} must mask ALL keys under it
    nested_user_payload = {
        "user": {
            "name": "Alice Secret",
            "status": "active",
            "type": "vip",
            "payout": 0.90,
            "profile": {
                "symbol": "BTCUSD",
                "phone": "+1234567890",
            },
        },
        "quote": {
            "symbol": "EURUSD",
            "price": 1.0850,
            "status": "open",
            "payout": 0.85,
        },
    }
    res = redactor.redact(nested_user_payload)

    # In user subtree, even allowlisted keys (name, status, type, payout, symbol) are masked
    assert res["user"]["name"] == f"<str:{len('Alice Secret')}>"
    assert res["user"]["status"] == f"<str:{len('active')}>"
    assert res["user"]["type"] == f"<str:{len('vip')}>"
    assert res["user"]["payout"] == "<num>"
    assert res["user"]["profile"]["symbol"] == f"<str:{len('BTCUSD')}>"
    assert res["user"]["profile"]["phone"] == f"<str:{len('+1234567890')}>"

    # Outside user subtree, allowlisted keys are preserved
    assert res["quote"]["symbol"] == "EURUSD"
    assert res["quote"]["price"] == 1.0850
    assert res["quote"]["status"] == "open"
    assert res["quote"]["payout"] == 0.85

    # 2. Key 'name' is only shown when an ancestor key looks like an asset
    # Case A: Top-level name (no ancestor) -> masked
    top_name = redactor.redact({"name": "Bob Trader"})
    assert top_name["name"] == f"<str:{len('Bob Trader')}>"

    # Case B: Non-asset ancestor (e.g. 'company', 'trader', 'profile') -> masked
    trader_name = redactor.redact({"trader": {"name": "Bob Trader"}})
    assert trader_name["trader"]["name"] == f"<str:{len('Bob Trader')}>"

    # Case C: Asset ancestors (asset, instrument, active, pair, symbol, ticker, market) -> shown
    assert redactor.redact({"asset": {"name": "EUR/USD"}})["asset"]["name"] == "EUR/USD"
    assert redactor.redact({"instrument": {"name": "BTC/USD"}})["instrument"]["name"] == "BTC/USD"
    assert redactor.redact({"active": {"name": "Crypto IDX"}})["active"]["name"] == "Crypto IDX"
    assert redactor.redact({"pair": {"name": "GBP/USD"}})["pair"]["name"] == "GBP/USD"
    assert redactor.redact({"symbol": {"name": "USD/JPY"}})["symbol"]["name"] == "USD/JPY"
    assert redactor.redact({"ticker": {"name": "AAPL"}})["ticker"]["name"] == "AAPL"
    assert redactor.redact({"market": {"name": "Forex"}})["market"]["name"] == "Forex"
    assert redactor.redact({"instruments": [{"name": "ETH/USD"}]})["instruments"][0]["name"] == "ETH/USD"

    # Case D: Sensitive ancestor overrides asset ancestor
    user_asset = redactor.redact({"user": {"asset": {"name": "EUR/USD"}}})
    assert user_asset["user"]["asset"]["name"] == f"<str:{len('EUR/USD')}>"


def test_context_listener_handles_second_page_popup():
    """Ensure context.on('page') attaches listeners to new popups/tabs and captures traffic."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    class MockPage:
        def __init__(self, name="popup"):
            self.name = name
            self.callbacks = {}

        def on(self, event, cb):
            self.callbacks[event] = cb

    class MockContext:
        def __init__(self):
            self.pages = []
            self.callbacks = {}

        def on(self, event, cb):
            self.callbacks[event] = cb

    mock_context = MockContext()

    # Wire context listener as in discover_platform.py run()
    mock_context.on("page", lambda new_page: discovery._attach_page_listeners(new_page))
    for pg in mock_context.pages:
        discovery._attach_page_listeners(pg)

    # Open a second page (popup)
    popup_page = MockPage(name="oauth_popup")
    mock_context.callbacks["page"](popup_page)

    assert popup_page in discovery._attached_pages
    assert "response" in popup_page.callbacks
    assert "websocket" in popup_page.callbacks

    # Verify that popup WebSocket traffic is recorded
    class MockPopupWebSocket:
        def __init__(self):
            self.url = "wss://popup.example.com/ws/stream?session_token=xyz"
            self.callbacks = {}

        def on(self, event, cb):
            self.callbacks[event] = cb

    mock_ws = MockPopupWebSocket()
    popup_page.callbacks["websocket"](mock_ws)
    assert "wss://popup.example.com/ws/stream" in discovery.ws_endpoints


def test_binary_frame_handling():
    """Binary frames must only record count, length, and first 16 bytes as hex."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    class MockWebSocket:
        def __init__(self):
            self.url = "wss://stream.example.com/ws?token=secret123"
            self.callbacks = {}

        def on(self, event, cb):
            self.callbacks[event] = cb

    mock_ws = MockWebSocket()
    discovery._on_websocket(mock_ws)

    # Simulate sending and receiving binary payloads
    bin_in = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                   0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x99, 0x88])
    bin_out = bytes([0xaa, 0xbb, 0xcc, 0xdd])

    mock_ws.callbacks["framereceived"](bin_in)
    mock_ws.callbacks["framesent"](bin_out)

    assert discovery.binary_frame_count == 2
    assert len(discovery.binary_samples) == 2
    sample1 = discovery.binary_samples[0]
    assert sample1["direction"] == "in"
    assert sample1["len"] == 18
    assert sample1["hex_prefix"] == "0102030405060708090a0b0c0d0e0f10"

    sample2 = discovery.binary_samples[1]
    assert sample2["direction"] == "out"
    assert sample2["len"] == 4
    assert sample2["hex_prefix"] == "aabbccdd"


def test_url_query_stripping_and_skip_patterns():
    """Query strings must be stripped and auth/login endpoints must be skipped."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com/trade?session=abc&token=xyz", duration_seconds=10)
    clean = discovery.strip_url("https://platform.example.com/api/v1/quotes?apiKey=secret123&user=99#tab")
    assert clean == "https://platform.example.com/api/v1/quotes"
    assert "apiKey" not in clean
    assert "secret123" not in clean

    # Semicolon matrix parameters stripped
    clean_matrix = discovery.strip_url("https://ad.doubleclick.net/activityi;src=123;u=998877;dc_pre=xyz")
    assert clean_matrix == "https://ad.doubleclick.net/activityi"
    assert "u=998877" not in clean_matrix

    class MockResponse:
        def __init__(self, url, content_type="application/json"):
            self.url = url
            self.headers = {"content-type": content_type}
            self._text = json.dumps({"token": "secret_token_123", "status": "ok"})

        def text(self):
            return self._text

    # Response with login/auth URL should have body skipped
    auth_resp = MockResponse("https://platform.example.com/api/auth/token?code=123")
    discovery._on_response(auth_resp)
    # The URL itself is stored query-stripped, but message types was not parsed
    assert "https://platform.example.com/api/auth/token" in discovery.http_endpoints
    assert "token" not in discovery.classified_types


def test_message_classification_by_key_names():
    """Messages must be categorized into price, assets, payout, balance, trade/order, other based on keys."""
    assert classify_category_from_keys({"price", "time"}) == "price"
    assert classify_category_from_keys({"bid", "ask"}) == "price"
    assert classify_category_from_keys({"assets", "name"}) == "assets"
    assert classify_category_from_keys({"payout", "asset"}) == "payout"
    assert classify_category_from_keys({"balance", "currency"}) == "balance"
    assert classify_category_from_keys({"order", "status", "trade_id"}) == "trade/order"
    assert classify_category_from_keys({"unknown_key", "meta"}) == "other"

    # Message type extraction
    assert extract_message_type({"type": "quote_tick", "price": 1.10}) == "quote_tick"
    assert extract_message_type({"event": "balance_update", "val": 100}) == "balance_update"
    assert extract_message_type({"action": "deal_created"}) == "deal_created"
    assert extract_message_type({"custom_key": 123}) == "custom_key"


def test_cli_requires_url():
    """CLI must fail if --url is omitted."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "tools/discover_platform.py"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "--url" in res.stderr or "required" in res.stderr


def test_tracker_hosts_dropped():
    """Ad, analytics, and tracker hosts must be dropped before storing endpoints or parsing bodies."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    class MockResponse:
        def __init__(self, url):
            self.url = url
            self.headers = {"content-type": "application/json"}

        def text(self):
            return json.dumps({"tracker_data": "secret_data"})

    # Tracker HTTP responses should be dropped
    trackers = [
        "https://11559740.fls.doubleclick.net/activityi;src=123",
        "https://google-analytics.com/collect?v=1",
        "https://www.googletagmanager.com/gtm.js?id=GTM-123",
        "https://adservice.google.com/ddm/fls/z/dc_pre=abc",
        "https://sentry.io/api/123/envelope/",
        "https://connect.facebook.net/en_US/fbevents.js",
    ]
    for trk in trackers:
        resp = MockResponse(trk)
        discovery._on_response(resp)

    assert discovery.dropped_tracker_count == len(trackers)
    assert len(discovery.http_endpoints) == 0

    # Tracker WebSocket should also be dropped
    class MockWS:
        def __init__(self, url):
            self.url = url

    discovery._on_websocket(MockWS("wss://mixpanel.com/ws/events"))
    assert discovery.dropped_tracker_count == len(trackers) + 1
    assert len(discovery.ws_endpoints) == 0


def test_http_endpoints_capped_at_2000_unique_collapsed_paths():
    """HTTP endpoints must collapse IDs/UUIDs and cap storage at 2000 unique paths."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)
    discovery.max_endpoint_cap = 5  # Test with smaller cap to verify logic

    class MockResponse:
        def __init__(self, url):
            self.url = url
            self.headers = {"content-type": "application/json"}

        def text(self):
            return "{}"

    # 1. Test path collapsing: numbers and UUIDs collapse
    discovery._on_response(MockResponse("https://api.example.com/v1/users/101/balance?token=abc;param=1"))
    discovery._on_response(MockResponse("https://api.example.com/v1/users/202/balance#section"))
    discovery._on_response(MockResponse("https://api.example.com/v1/deals/12345678-abcd-1234-abcd-1234567890ab/status"))

    assert "https://api.example.com/v1/users/{id}/balance" in discovery.http_endpoints
    assert "https://api.example.com/v1/deals/{uuid}/status" in discovery.http_endpoints
    assert len(discovery.http_endpoints) == 2  # 101 and 202 collapsed to same path

    # 2. Add more unique endpoints up to and exceeding the cap of 5
    discovery._on_response(MockResponse("https://api.example.com/v1/assets"))
    discovery._on_response(MockResponse("https://api.example.com/v1/quotes"))
    discovery._on_response(MockResponse("https://api.example.com/v1/payouts"))
    assert len(discovery.http_endpoints) == 5

    # 6th and 7th endpoints exceed cap
    discovery._on_response(MockResponse("https://api.example.com/v1/extra1"))
    discovery._on_response(MockResponse("https://api.example.com/v1/extra2"))
    assert len(discovery.http_endpoints) == 5
    assert discovery.dropped_endpoint_cap_count == 2


def test_stored_examples_per_message_type_capped_at_3():
    """Stored examples per classified message type must be capped at 3."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    # Send 5 frames for the same message type
    for i in range(5):
        frame = json.dumps({"action": "deal_tick", "price": 1.1000 + i * 0.0005})
        discovery._process_text_frame(frame, "WS_IN", "wss://example.com/stream")

    assert "deal_tick" in discovery.classified_types
    entry = discovery.classified_types["deal_tick"]
    assert entry["count"] == 5
    assert len(entry["examples"]) == 3


def test_report_includes_dropped_counts(tmp_path: Path):
    """Generated report must contain lines with dropped tracker and capacity counts."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)
    discovery.dropped_tracker_count = 15
    discovery.dropped_endpoint_cap_count = 7

    rep_file = tmp_path / "test_report.md"
    from unittest.mock import patch
    with patch("tools.discover_platform.REPORT_PATH", rep_file):
        discovery.write_report()

    content = rep_file.read_text(encoding="utf-8")
    assert "- **Dropped Tracker Requests/Frames**: 15" in content
    assert "- **Dropped Endpoints (Over 2000 Cap)**: 7" in content


# ── DISCOVERY PASS 2 TESTS ───────────────────────────────────────────────────

def test_pass2_safelist_keys_and_identifiers_survive_while_secrets_masked():
    """Verify Pass 2 safe-listing rules:
    - Safe string leaves survive: topic, event, ric, symbol, status, direction, etc.
    - Safe numeric leaves survive: rate, price, payout, payment_rate, ts, open_time, etc.
    - Balance numbers are masked.
    - Identifiers with separators survive: asset_trading_settings_changed_v2, Z-CRY/IDX.
    - Secrets masked: JWT, long random token, email, UUID, nested {"user": {"name": ...}}.
    """
    redactor = PayloadRedactor()

    # Identifiers check
    assert is_identifier_string("asset_trading_settings_changed_v2") is True
    assert is_identifier_string("Z-CRY/IDX") is True
    assert is_identifier_string("quotes:Z-CRY/IDX") is True
    assert is_identifier_string("12345678-abcd-1234-abcd-1234567890ab") is False  # UUID masked
    assert is_identifier_string("d8f9a2c1b4e6f8a0c2e4b6d8f0") is False  # Long random hex token
    assert is_identifier_string("plainword") is False  # No separator

    payload = {
        "topic": "quotes:Z-CRY/IDX",
        "event": "quote",
        "event_type": "price_tick",
        "action": "subscribe",
        "ric": "Z-CRY/IDX",
        "asset_ric": "Z-CRY/IDX",
        "asset": "Crypto IDX",
        "symbol": "BTCUSD",
        "currency": "USD",
        "status": "active",
        "type": "binary",
        "kind": "turbo",
        "mode": "standard",
        "deal_type": "call",
        "trend": "up",
        "direction": "call",
        "rate": 123.456,
        "price": 108.50,
        "quote": 108.50,
        "bid": 108.45,
        "ask": 108.55,
        "open": 108.00,
        "high": 109.00,
        "low": 107.50,
        "close": 108.50,
        "value": 50.0,
        "payout": 85,
        "payment_rate": 85,
        "profit": 18.5,
        "percent": 85,
        "ts": 1700000000,
        "time": 1700000001,
        "created_at": 1700000002,
        "open_time": 1700000003,
        "close_time": 1700000063,
        "expire_at": 1700000063,
        "expiration": 1700000063,
        "duration": 60,
        # Balance numbers masked
        "balance": 15000.0,
        "demo_balance": 10000.0,
        # IDs masked
        "deal_id": 998877,
        "user_id": 12345,
        # Secrets masked
        "jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFsaWNlIn0.g_7g9_z8x-K3L",
        "session_secret": "abcdef1234567890abcdef1234567890",
        "email_address": "trader@securebroker.com",
        "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
        # Nested sensitive subtree
        "user": {
            "name": "Alice Secret",
            "status": "vip",
        },
        # Safe identifier in custom key
        "custom_identifier": "asset_trading_settings_changed_v2",
    }

    redacted = redactor.redact(payload)

    # 1. Safe leaf values survive
    assert redacted["topic"] == "quotes:Z-CRY/IDX"
    assert redacted["event"] == "quote"
    assert redacted["event_type"] == "price_tick"
    assert redacted["action"] == "subscribe"
    assert redacted["ric"] == "Z-CRY/IDX"
    assert redacted["asset_ric"] == "Z-CRY/IDX"
    assert redacted["asset"] == "Crypto IDX"
    assert redacted["symbol"] == "BTCUSD"
    assert redacted["currency"] == "USD"
    assert redacted["status"] == "active"
    assert redacted["type"] == "binary"
    assert redacted["kind"] == "turbo"
    assert redacted["mode"] == "standard"
    assert redacted["deal_type"] == "call"
    assert redacted["trend"] == "up"
    assert redacted["direction"] == "call"

    # 2. Safe numeric values survive
    assert redacted["rate"] == 123.456
    assert redacted["price"] == 108.50
    assert redacted["quote"] == 108.50
    assert redacted["bid"] == 108.45
    assert redacted["ask"] == 108.55
    assert redacted["open"] == 108.00
    assert redacted["high"] == 109.00
    assert redacted["low"] == 107.50
    assert redacted["close"] == 108.50
    assert redacted["value"] == 50.0
    assert redacted["payout"] == 85
    assert redacted["payment_rate"] == 85
    assert redacted["profit"] == 18.5
    assert redacted["percent"] == 85
    assert redacted["ts"] == 1700000000
    assert redacted["time"] == 1700000001
    assert redacted["created_at"] == 1700000002
    assert redacted["open_time"] == 1700000003
    assert redacted["close_time"] == 1700000063
    assert redacted["expire_at"] == 1700000063
    assert redacted["expiration"] == 1700000063
    assert redacted["duration"] == 60

    # 3. Identifier strings survive
    assert redacted["custom_identifier"] == "asset_trading_settings_changed_v2"

    # 4. Balance numbers and IDs are masked
    assert redacted["balance"] == "<num>"
    assert redacted["demo_balance"] == "<num>"
    assert redacted["deal_id"] == "<num>"
    assert redacted["user_id"] == "<num>"

    # 5. Secrets are masked
    out_str = json.dumps(redacted)
    assert "eyJhbGciOiJIUzI1Ni" not in out_str
    assert "abcdef1234567890abcdef1234567890" not in out_str
    assert "trader@securebroker.com" not in out_str
    assert "550e8400-e29b-41d4-a716-446655440000" not in out_str
    assert "Alice Secret" not in out_str
    assert redacted["user"]["name"] == f"<str:{len('Alice Secret')}>"
    assert redacted["user"]["status"] == f"<str:{len('vip')}>"


def test_pass2_demo_real_fields_extracted_and_preserved():
    """Keys containing demo, real, or account_type preserve their values."""
    redactor = PayloadRedactor()

    payload = {
        "is_demo": True,
        "is_real": False,
        "account_type": "demo",
        "demo": True,
        "real": False,
        "demo_balance": 10000.0,
    }

    redacted = redactor.redact(payload)
    assert redacted["is_demo"] is True
    assert redacted["is_real"] is False
    assert redacted["account_type"] == "demo"
    assert redacted["demo"] is True
    assert redacted["real"] is False
    # Numeric balance still masked
    assert redacted["demo_balance"] == "<num>"


def test_pass2_structure_depth_6_and_list_slicing():
    """Full structure preserved up to depth 6; lists sliced to first 2 elements."""
    redactor = PayloadRedactor()

    deep_data = {
        "l1": {
            "l2": {
                "l3": {
                    "l4": {
                        "l5": {
                            "l6": {
                                "l7_key": "l7_value",
                            },
                            "status": "active",
                        },
                    },
                },
            },
        },
        "quotes": [
            {"rate": 1.10, "time": 1000},
            {"rate": 1.11, "time": 1001},
            {"rate": 1.12, "time": 1002},
            {"rate": 1.13, "time": 1003},
        ],
    }

    res = redactor.redact(deep_data)
    # Check all levels preserved
    assert "l1" in res
    assert "l2" in res["l1"]
    assert "l3" in res["l1"]["l2"]
    assert "l4" in res["l1"]["l2"]["l3"]
    assert "l5" in res["l1"]["l2"]["l3"]["l4"]
    assert res["l1"]["l2"]["l3"]["l4"]["l5"]["status"] == "active"

    # Lists shortened to 2 elements
    assert len(res["quotes"]) == 2
    assert res["quotes"][0]["rate"] == 1.10
    assert res["quotes"][1]["rate"] == 1.11


def test_pass2_i18n_detection_and_classification():
    """Translation / i18n JSON is detected; message pairs classified into 8 canonical categories."""
    # 1. i18n URL detection
    assert is_i18n_payload({}, url="https://platform.example.com/api/locale/en.json") is True
    assert is_i18n_payload({}, url="https://platform.example.com/i18n/translations") is True

    # 2. i18n content detection
    i18n_dict = {f"msg_{i}": f"This is translation sentence number {i}" for i in range(25)}
    assert is_i18n_payload(i18n_dict) is True
    assert is_i18n_payload({"quote": {"rate": 1.10, "time": 1000}}) is False

    # 3. Canonical 8 categories classification
    assert classify_message("quotes:Z-CRY/IDX", "quote") == "prices/quotes"
    assert classify_message("candles:EURUSD", "candles") == "prices/quotes"
    assert classify_message("assets", "asset_trading_settings_changed_v2") == "assets/payouts"
    assert classify_message("payouts", "update_payout") == "assets/payouts"
    assert classify_message("user:12345", "balance_changed") == "balance"
    assert classify_message("deals", "create_deal") == "deals"
    assert classify_message("deals", "deal_opened") == "deals"
    assert classify_message("deals", "deal_closed") == "deals"
    assert classify_message("social:chat", "message") == "social"
    assert classify_message("phoenix", "phx_join") == "system"
    assert classify_message("phoenix", "heartbeat") == "system"
    assert classify_message("unknown", "unknown", is_i18n=True) == "i18n"
    assert classify_message("misc", "random") == "other"


def test_pass2_phoenix_sequence_end_to_end_in_discovery(tmp_path: Path):
    """End-to-end test of Phoenix message sequence (join, quotes, balance, deal create/open/close)."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    # 1. Join
    frame_join = json.dumps(["1", "1", "quotes:Z-CRY/IDX", "phx_join", {}])
    discovery._process_text_frame(frame_join, "WS_OUT", "wss://ws.example.com/socket/websocket")

    # 2. Quotes streaming
    frame_quote = json.dumps([None, None, "quotes:Z-CRY/IDX", "quote", {
        "ric": "Z-CRY/IDX",
        "rate": 123.45,
        "time": 1700000000,
        "trend": "up",
    }])
    discovery._process_text_frame(frame_quote, "WS_IN", "wss://ws.example.com/socket/websocket")

    # 3. Balance change
    frame_balance = json.dumps([None, None, "user:12345", "balance_changed", {
        "balance": 10500.0,
        "currency": "USD",
        "is_demo": True,
        "account_type": "demo",
    }])
    discovery._process_text_frame(frame_balance, "WS_IN", "wss://ws.example.com/socket/websocket")

    # 4. Deal create / open / close
    frame_create = json.dumps(["2", "2", "deals", "create_deal", {
        "ric": "Z-CRY/IDX",
        "direction": "call",
        "duration": 60,
    }])
    discovery._process_text_frame(frame_create, "WS_OUT", "wss://ws.example.com/socket/websocket")

    frame_open = json.dumps(["2", "3", "deals", "deal_opened", {
        "deal_id": 9999,
        "ric": "Z-CRY/IDX",
        "rate": 123.45,
        "open_time": 1700000001,
        "direction": "call",
        "payment_rate": 85,
    }])
    discovery._process_text_frame(frame_open, "WS_IN", "wss://ws.example.com/socket/websocket")

    frame_close = json.dumps(["2", "4", "deals", "deal_closed", {
        "deal_id": 9999,
        "ric": "Z-CRY/IDX",
        "close_rate": 123.75,
        "close_time": 1700000061,
        "profit": 18.5,
        "status": "win",
    }])
    discovery._process_text_frame(frame_close, "WS_IN", "wss://ws.example.com/socket/websocket")

    # Assertions
    assert discovery.phx_join_topics["quotes:Z-CRY/IDX"] == 1
    assert len(discovery.subscribe_messages) >= 1
    assert discovery.subscribe_messages[0]["topic"] == "quotes:Z-CRY/IDX"
    assert discovery.subscribe_messages[0]["host"] == "wss://ws.example.com/socket/websocket"

    assert ("quotes:Z-CRY/IDX", "quote") in discovery.pair_stats
    assert discovery.pair_stats[("quotes:Z-CRY/IDX", "quote")]["category"] == "prices/quotes"

    assert ("user:12345", "balance_changed") in discovery.pair_stats
    assert discovery.pair_stats[("user:12345", "balance_changed")]["category"] == "balance"

    assert ("deals", "deal_closed") in discovery.pair_stats
    assert discovery.pair_stats[("deals", "deal_closed")]["category"] == "deals"

    assert discovery.demo_real_fields.get("is_demo") is True
    assert discovery.demo_real_fields.get("account_type") == "demo"

    # Write report and verify content
    rep_file = tmp_path / "test_phoenix_report.md"
    from unittest.mock import patch
    with patch("tools.discover_platform.REPORT_PATH", rep_file):
        discovery.write_report()

    rep_text = rep_file.read_text(encoding="utf-8")
    assert "quotes:Z-CRY/IDX" in rep_text
    assert "deal_closed" in rep_text
    assert "balance_changed" in rep_text
    assert "is_demo" in rep_text


def test_json_parsing_before_size_cut():
    """Parse JSON BEFORE any size cut; invalid JSON becomes <str:len> rather than raw string."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    # Valid large JSON payload
    large_payload = {
        "topic": "quotes:BTCUSD",
        "event": "quote",
        "data": [{"rate": 50000 + i, "time": 1700000000 + i} for i in range(100)],
    }
    raw_valid = json.dumps(large_payload)
    discovery._process_text_frame(raw_valid, "WS_IN", "wss://ws.example.com/ws")

    pair = ("quotes:BTCUSD", "quote")
    assert pair in discovery.pair_stats
    ex = discovery.pair_stats[pair]["examples"][0]
    # Pruned to 2 elements, not truncated raw string
    assert len(ex["data"]) == 2
    assert ex["data"][0]["rate"] == 50000
    assert ex["data"][1]["rate"] == 50001

    # Invalid JSON payload
    invalid_raw = "INVALID_NON_JSON_SECRET_TOKEN_XYZ12345"
    discovery._process_text_frame(invalid_raw, "WS_IN", "wss://ws.example.com/ws")
    # Redacted to <str:len>, secret never stored raw
    assert "INVALID_NON_JSON_SECRET_TOKEN" not in str(discovery.classified_types)
    found_masked = any("<str:" in str(v) for v in discovery.classified_types.values())
    assert found_masked


def test_candles_url_collapsing_preserves_timeframe_and_query_param_names():
    """Candle URLs retain numeric timeframe and query parameter names without values."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    candle_url = "https://api.binomo.com/candles/v1/assets/999/60?locale=en&from=1700000000&to=1700003600"
    collapsed = discovery.collapse_url_path(candle_url)
    assert collapsed == "https://api.binomo.com/candles/v1/assets/{id}/60?locale&from&to"

    # Non-candle endpoint strips query params and collapses trailing id
    normal_url = "https://api.binomo.com/bo-assets/v7/assets/123?filter=all"
    collapsed_normal = discovery.collapse_url_path(normal_url)
    assert collapsed_normal == "https://api.binomo.com/bo-assets/v7/assets/{id}"
    assert "?" not in collapsed_normal


def test_outgoing_websocket_frames_captured():
    """Outgoing frames for bo, asset, range_stream, or create events are captured up to 3 examples."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    for i in range(5):
        frame = json.dumps(["1", f"{i}", "bo", "create", {
            "ric": "Z-CRY/IDX",
            "amount": 100,
            "duration": 60,
            "token": "secret_out_token",
        }])
        discovery._process_outgoing_frame(frame, "WS_OUT", "wss://ws.example.com/socket/websocket")

    pair = ("bo", "create")
    assert pair in discovery.outgoing_frames
    assert len(discovery.outgoing_frames[pair]) == 3
    # Check redaction applied: token is masked
    assert "secret_out_token" not in json.dumps(discovery.outgoing_frames[pair])


def test_dom_accessibility_snapshot_filtering_and_masking():
    """Read-only DOM accessibility snapshot filters keywords, masks digit runs > 3, keeps durations, and omits profile."""
    # 1. Test mask_dom_text directly
    assert mask_dom_text("5 min") == "5 min"
    assert mask_dom_text("60 s") == "60 s"
    assert mask_dom_text("Payout 85%") == "Payout 85%"
    assert mask_dom_text("Account ID: 99887766") in ("Account ID: <num>", "Account ID: <phone>")
    assert mask_dom_text("Balance: $10,500.00") == "Balance: <num>"
    assert mask_dom_text("Email user@example.com") == "Email <email>"

    # 2. Test snapshot walk_node via PlatformDiscovery
    discovery = PlatformDiscovery(
        target_url="https://platform.example.com",
        duration_seconds=10,
        dom_snapshots=True,
    )

    fake_aria_yaml = """
- button "Call"
- button "Put"
- textbox "Amount $10,500.00"
- combobox "Duration 5 min"
- menu "User Profile Menu user@domain.com":
  - menuitem "Account Settings 12345"
"""

    class MockLocator:
        def aria_snapshot(self):
            return fake_aria_yaml

    class MockPage:
        def locator(self, selector):
            assert selector == "body"
            return MockLocator()

    discovery._take_dom_snapshot(MockPage(), "start")
    assert len(discovery.dom_snapshots) == 1
    snap = discovery.dom_snapshots[0]
    elements = snap["elements"]

    names = [el["name"] for el in elements]
    roles = [el["role"] for el in elements]

    assert "Call" in names
    assert "Put" in names
    assert "Amount <num>" in names
    assert "Duration 5 min" in names

    # Profile menu must be completely excluded
    for n in names:
        assert "Profile" not in n
        assert "user@domain.com" not in n
        assert "Settings" not in n


def test_full_structure_examples_up_to_6000_chars_in_report(tmp_path: Path):
    """Focus endpoints and WS pairs retain full structure examples up to 6000 chars in discovery_report.md."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    # Focus WS pair: range_stream quotes_range
    large_quotes = [{"r": 1.234 + i * 0.001, "t": 1700000000 + i} for i in range(50)]
    frame = json.dumps([None, None, "range_stream:Z-CRY/IDX", "quotes_range", {"quotes": large_quotes}])
    discovery._process_text_frame(frame, "WS_IN", "wss://ws.example.com")

    # Focus HTTP endpoint: /bo-assets/v7/assets
    rep_file = tmp_path / "test_report_6000.md"
    from unittest.mock import patch
    with patch("tools.discover_platform.REPORT_PATH", rep_file):
        discovery.write_report()

    report_text = rep_file.read_text(encoding="utf-8")
    assert "range_stream:Z-CRY/IDX" in report_text
    assert "quotes_range" in report_text


def test_safelist_additions_survive_redaction():
    """All safe-list additions survive redaction: option_type, asset_name, payment_rate, min_amount,

    max_amount, min_bet, max_bet, step, expiration, expirations, durations, expire_in,
    interval, timeframe, trading_status, is_active, is_open, schedule.
    """
    redactor = PayloadRedactor()

    payload = {
        "option_type": "turbo",
        "asset_name": "Crypto IDX",
        "payment_rate": 85,
        "min_amount": 1,
        "max_amount": 1000,
        "min_bet": 1,
        "max_bet": 1000,
        "step": 1,
        "expiration": 60,
        "expirations": [60, 120, 300],
        "durations": [60, 300],
        "expire_in": 30,
        "interval": 60,
        "timeframe": "1m",
        "trading_status": "open",
        "is_active": True,
        "is_open": True,
        "schedule": {"status": "available"},
        "token": "secret_token_12345",
    }

    redacted = redactor.redact(payload)

    # 1. Verify safe fields survive with real values
    assert redacted["option_type"] == "turbo"
    assert redacted["asset_name"] == "Crypto IDX"
    assert redacted["payment_rate"] == 85
    assert redacted["min_amount"] == 1
    assert redacted["max_amount"] == 1000
    assert redacted["min_bet"] == 1
    assert redacted["max_bet"] == 1000
    assert redacted["step"] == 1
    assert redacted["expiration"] == 60
    assert redacted["expirations"] == [60, 120]  # lists truncated to 2 elements
    assert redacted["durations"] == [60, 300]
    assert redacted["expire_in"] == 30
    assert redacted["interval"] == 60
    assert redacted["timeframe"] == "1m"
    assert redacted["trading_status"] == "open"
    assert redacted["is_active"] is True
    assert redacted["is_open"] is True
    assert redacted["schedule"]["status"] == "available"

    # 2. Verify sensitive key is still masked
    assert "secret_token_12345" not in json.dumps(redacted)


def test_extract_asset_table_populates_compact_rows_and_first_two():
    """_extract_asset_table extracts all assets into asset_table and preserves first two."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    asset_data = {
        "data": {
            "assets": [
                {
                    "name": "Crypto IDX",
                    "ric": "Z-CRY/IDX",
                    "type": "crypto",
                    "trading_tools_settings": {
                        "ftt": {
                            "base_payment_rate_turbo": 85,
                            "trading_status": "open",
                            "limits": {"min": 1, "max": 1000},
                            "schedule": {"durations": [60, 120]},
                        }
                    },
                },
                {
                    "name": "EUR/USD (OTC)",
                    "ric": "EURUSD_otc",
                    "type": "forex",
                    "payment_rate": 82,
                    "is_active": True,
                },
                {
                    "name": "Apple",
                    "ric": "AAPL",
                    "type": "stock",
                    "payment_rate": 80,
                },
            ]
        }
    }

    discovery._extract_asset_table(asset_data, "https://api.binomo.com/bo-assets/v7/assets")

    assert len(discovery.asset_table) == 3
    assert discovery.asset_table[0]["name"] == "Crypto IDX"
    assert discovery.asset_table[0]["ric"] == "Z-CRY/IDX"
    assert discovery.asset_table[0]["payment_rate"] == 85
    assert discovery.asset_table[1]["name"] == "EUR/USD (OTC)"
    assert discovery.asset_table[2]["name"] == "Apple"

    assert len(discovery.asset_first_two) == 2
    assert discovery.asset_first_two[0]["ric"] == "Z-CRY/IDX"
    assert discovery.asset_first_two[1]["ric"] == "EURUSD_otc"


def test_alt_ws_process_frame_tracks_binomo_non_phoenix_frames():
    """_process_alt_ws_frame tracks frames on non-Phoenix sockets (as.binomo.com)."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    sub_frame = json.dumps({"action": "subscribe", "asset": "Z-CRY/IDX", "event_type": "quotes"})
    data_frame = json.dumps({
        "event_type": "quotes",
        "asset": "Z-CRY/IDX",
        "data": {"price": 1234.56, "time": 1700000000},
    })

    host = "as.binomo.com"
    discovery._process_alt_ws_frame(sub_frame, f"wss://{host}/ws")
    discovery._process_alt_ws_frame(data_frame, f"wss://{host}/ws")

    url_key = f"wss://{host}/ws"
    assert url_key in discovery.alt_ws_data
    host_info = discovery.alt_ws_data[url_key]
    assert host_info["msg_count"] == 2
    assert "quotes" in host_info["event_types"]
    assert host_info["asset_rates"].get("Z-CRY/IDX") == 2
    assert host_info["price_field"] == "price"
    assert host_info["time_field"] == "time"
    assert len(host_info["subscribe_frames"]) >= 1
    assert len(host_info["data_frames"]) >= 1


def test_pass4_safelist_numerics_and_ancestors():
    """Pass 4: base_payment_rate_*, payment_rate, step, durations, and safe ancestors survive; balance and ids masked."""
    redactor = PayloadRedactor()
    payload = {
        "base_payment_rate_turbo": 85,
        "base_payment_rate_standard": 82,
        "base_payment_rate_classic": 80,
        "payment_rate": 84,
        "step": 5,
        "duration": 60,
        "durations": [60, 120, 300],
        "expire_in": 300,
        "limits": {
            "min": 1,
            "max": 1000,
            "min_bet": 10,
            "max_bet": 5000,
        },
        "summs": {
            "min_amount": 100,
            "max_amount": 50000,
        },
        "schedule": {
            "duration": 180,
            "status": "active",
        },
        "enabled_account_types": ["demo", "standard", "vip"],
        # Sensitive fields that must be masked
        "balance": 12500.50,
        "demo_balance": 10000.0,
        "user_balance": 500.0,
        "account_id": 98765,
        "user_id": 12345,
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
    }
    redacted = redactor.redact(payload)
    # Check preserved numbers and strings
    assert redacted["base_payment_rate_turbo"] == 85
    assert redacted["base_payment_rate_standard"] == 82
    assert redacted["base_payment_rate_classic"] == 80
    assert redacted["payment_rate"] == 84
    assert redacted["step"] == 5
    assert redacted["duration"] == 60
    assert redacted["durations"] == [60, 120]  # lists capped at 2
    assert redacted["expire_in"] == 300
    assert redacted["limits"]["min"] == 1
    assert redacted["limits"]["max"] == 1000
    assert redacted["limits"]["min_bet"] == 10
    assert redacted["limits"]["max_bet"] == 5000
    assert redacted["summs"]["min_amount"] == 100
    assert redacted["summs"]["max_amount"] == 50000
    assert redacted["schedule"]["duration"] == 180
    assert redacted["schedule"]["status"] == "active"
    assert redacted["enabled_account_types"] == ["demo", "standard"]

    # Check masked fields
    assert redacted["balance"] == "<num>"
    assert redacted["demo_balance"] == "<num>"
    assert redacted["user_balance"] == "<num>"
    assert redacted["account_id"] == "<num>"
    assert redacted["user_id"] == "<num>"


def test_pass4_login_detection_and_report_header(tmp_path: Path):
    """Pass 2 login detection observation and report header indicator."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    # Initially not logged in
    st = discovery.check_login_status()
    assert st["logged_in"] is False
    assert st["ws_binomo_open"] is False
    assert st["seen_passport_or_assets"] is False
    assert st["seen_account_join"] is False

    # 1. ws.binomo.com opens
    discovery.open_websockets.add("wss://ws.binomo.com/socket/websocket")
    st = discovery.check_login_status()
    assert st["ws_binomo_open"] is True
    assert st["logged_in"] is False

    # 2. request to /passport/
    class FakeReq:
        url = "https://api.binomo.com/passport/v1/user"
        method = "GET"
    discovery._on_request(FakeReq())
    st = discovery.check_login_status()
    assert st["seen_passport_or_assets"] is True
    assert st["logged_in"] is False

    # 3. join on "account" topic
    join_frame = json.dumps(["1", "1", "account:12345", "phx_join", {}])
    discovery._process_outgoing_frame(join_frame, "WS_OUT", "wss://ws.binomo.com/socket/websocket")
    st = discovery.check_login_status()
    assert st["seen_account_join"] is True
    assert st["logged_in"] is True

    # Report header test
    rep_file = tmp_path / "test_login_report.md"
    from unittest.mock import patch
    with patch("tools.discover_platform.REPORT_PATH", rep_file):
        discovery.write_report()
    rep_text = rep_file.read_text(encoding="utf-8")
    assert "- **Logged-in indicators**: yes" in rep_text


def test_pass4_dom_snapshot_aria_cdp_and_error():
    """Pass 4 DOM snapshots: CDP fallback and safe error recording without exception."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)
    discovery.dom_snapshots_enabled = True

    # Case 1: CDP fallback when aria_snapshot is missing
    cdp_ax_tree = {
        "nodes": [
            {"role": {"value": "button"}, "name": {"value": "Call"}, "data-testid": "call-btn"},
            {"role": {"value": "button"}, "name": {"value": "Put"}, "data-testid": "put-btn"},
            {"role": {"value": "textbox"}, "name": {"value": "Amount $500"}, "data-testid": "amt-in"},
        ]
    }
    class FakeCDPSession:
        def send(self, method: str):
            assert method == "Accessibility.getFullAXTree"
            return cdp_ax_tree

    class FakeContext:
        def new_cdp_session(self, page):
            return FakeCDPSession()

    class FakeCDPPage:
        context = FakeContext()

    discovery._take_dom_snapshot(FakeCDPPage(), "cdp_test")
    assert len(discovery.dom_snapshots) == 1
    snap = discovery.dom_snapshots[0]
    assert snap["label"] == "cdp_test"
    names = [el["name"] for el in snap["elements"]]
    assert "Call" in names
    assert "Put" in names
    assert "Amount <num>" in names

    # Case 2: Error handling when both fail
    class BrokenPage:
        def locator(self, sel):
            raise RuntimeError("CDP disconnected")
        class context:
            @staticmethod
            def new_cdp_session(page):
                raise RuntimeError("CDP session failed")

    discovery._take_dom_snapshot(BrokenPage(), "error_test")
    assert len(discovery.dom_snapshots) == 2
    err_snap = discovery.dom_snapshots[1]
    assert err_snap["label"] == "error_test"
    assert "error" in err_snap
    assert err_snap["error"] is not None


def test_pass4_asset_table_all_assets_and_otc(tmp_path: Path):
    """Pass 4: /bo-assets/v7/assets stores ALL assets with OTC flag, 3 base rates, accounts, and schedule."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    asset_data = {
        "data": {
            "assets": [
                {
                    "name": "Crypto IDX",
                    "ric": "Z-CRY/IDX",
                    "type": "crypto",
                    "trading_tools_settings": {
                        "ftt": {
                            "base_payment_rate_turbo": 85,
                            "base_payment_rate_standard": 82,
                            "base_payment_rate_classic": 80,
                            "enabled_account_types": ["demo", "standard", "gold", "vip"],
                            "schedule": {"durations": [60, 120, 300]},
                        }
                    },
                },
                {
                    "name": "EUR/USD (OTC)",
                    "ric": "EURUSD_otc",
                    "type": "forex",
                    "trading_tools_settings": {
                        "ftt": {
                            "base_payment_rate_turbo": 84,
                            "base_payment_rate_standard": 81,
                            "base_payment_rate_classic": 78,
                            "enabled_account_types": ["demo", "vip"],
                            "schedule": {"status": "open_24_7"},
                        }
                    },
                },
                {
                    "name": "Apple",
                    "ric": "AAPL",
                    "type": "stock",
                    "payment_rate": 80,
                    "enabled_account_types": ["vip"],
                },
                {
                    "name": "Gold",
                    "ric": "XAUUSD",
                    "type": "commodity",
                    "payment_rate": 75,
                },
            ]
        }
    }

    discovery._extract_asset_table(asset_data, "https://api.binomo.com/bo-assets/v7/assets")

    # ALL assets must be stored (4 total)
    assert len(discovery.asset_table) == 4
    assert discovery.asset_table[0]["ric"] == "Z-CRY/IDX"
    assert discovery.asset_table[0]["otc"] is True
    assert discovery.asset_table[0]["base_rate_turbo"] == 85
    assert discovery.asset_table[0]["base_rate_standard"] == 82
    assert discovery.asset_table[0]["base_rate_classic"] == 80
    assert "demo, standard" in discovery.asset_table[0]["enabled_account_types"]

    assert discovery.asset_table[1]["ric"] == "EURUSD_otc"
    assert discovery.asset_table[1]["otc"] is True

    assert discovery.asset_table[2]["ric"] == "AAPL"
    assert discovery.asset_table[2]["otc"] is False

    assert discovery.asset_table[3]["ric"] == "XAUUSD"
    assert discovery.asset_table[3]["otc"] is False

    # Check report writing has Section 14 table with 8 columns
    rep_file = tmp_path / "test_asset_table_report.md"
    from unittest.mock import patch
    with patch("tools.discover_platform.REPORT_PATH", rep_file):
        discovery.write_report()

    rep_text = rep_file.read_text(encoding="utf-8")
    assert "## 14. Asset List Table (/bo-assets/v7/assets)" in rep_text
    assert "| Name | RIC | Type | OTC | Base Rates (Turbo/Std/Classic) | Enabled Accounts | Trading Types | Schedule |" in rep_text
    assert "Crypto IDX" in rep_text
    assert "EUR/USD (OTC)" in rep_text
    assert "Apple" in rep_text
    assert "Gold" in rep_text


def test_pass4_never_overwrite_and_archive(tmp_path: Path):
    """Pass 4: Report writes to timestamped path and keeps an archive copy in reports_archive/."""
    custom_report = tmp_path / "discovery_report_20261102_1500.md"
    archive_dir = tmp_path / "reports_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    discovery = PlatformDiscovery(
        target_url="https://platform.example.com",
        duration_seconds=10,
        report_path=custom_report,
    )
    discovery.archive_path = archive_dir / custom_report.name

    discovery.write_report()

    assert custom_report.exists()
    assert (archive_dir / custom_report.name).exists()
    assert custom_report.read_text(encoding="utf-8") == (archive_dir / custom_report.name).read_text(encoding="utf-8")


def test_pass4_socket_frame_counts():
    """Pass 4: In/out/total frame counts are tracked per WebSocket endpoint."""
    discovery = PlatformDiscovery(target_url="https://platform.example.com", duration_seconds=10)

    class FakeWS:
        def __init__(self, url):
            self.url = url
            self.handlers = {}
        def on(self, event, handler):
            self.handlers[event] = handler

    sock1 = FakeWS("wss://ws.binomo.com/socket/websocket?v=1")
    discovery._on_websocket(sock1)
    sock1.handlers["framereceived"]("frame1_in")
    sock1.handlers["framereceived"]("frame2_in")
    sock1.handlers["framesent"]("frame1_out")

    sock2 = FakeWS("wss://as.binomo.com/ws?v=2")
    discovery._on_websocket(sock2)
    sock2.handlers["framereceived"]('{"type":"ping"}')
    sock2.handlers["framesent"]('{"type":"pong"}')

    assert "wss://ws.binomo.com/socket/websocket" in discovery.socket_frame_counts
    c1 = discovery.socket_frame_counts["wss://ws.binomo.com/socket/websocket"]
    assert c1["in"] == 2
    assert c1["out"] == 1
    assert c1["total"] == 3

    assert "wss://as.binomo.com/ws" in discovery.socket_frame_counts
    c2 = discovery.socket_frame_counts["wss://as.binomo.com/ws"]
    assert c2["in"] == 1
    assert c2["out"] == 1
    assert c2["total"] == 2




