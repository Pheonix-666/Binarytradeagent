"""Tests for tools/discover_platform.py.

PROJECT_SPEC Part A:
  - Tool must be strictly read-only: no click, fill, type, press, check, evaluate calls.
  - Redaction: fake payloads containing tokens, JWT, email, account id, balance never appear in output.
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
    classify_category_from_keys,
    extract_message_type,
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

    # 2. Assert safe fields are preserved
    assert redacted["action"] == "subscribe"
    assert redacted["symbol"] == "EURUSD"
    assert redacted["price"] == 1.0855
    assert redacted["nested_user"]["payout"] == 0.88

    # 3. Assert balance numeric values are masked
    assert redacted["balance"] == "<num>"
    assert redacted["demo_balance"] == "<num>"

    # 4. Assert token/auth/user strings are masked as <str:len>
    assert redacted["token"] == f"<str:{len('secret_session_token_xyz987')}>"
    assert redacted["user_email"] == f"<str:{len('trader_alice@securebank.com')}>"

    # Redactions were counted
    assert redactor.redaction_count >= 8


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

