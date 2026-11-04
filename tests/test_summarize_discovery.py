"""Tests for tools/summarize_discovery.py.

PROJECT_SPEC Discovery Report Slimming:
  - Summarizes discovery_report.md into discovery_summary.md of at most 600 lines.
  - Excludes ad/analytics/tracker hosts.
  - Collapses numbers, UUIDs, and ';' parameters in HTTP paths.
  - Groups message types by category with one compact example each.
  - Verifies counts for text frames, binary frames, distinct WS hosts/paths, HTTP endpoints, redactions.
  - Confirms secret-pattern matches line is present.
"""

import json
from pathlib import Path
import pytest

from tools.summarize_discovery import (
    collapse_url,
    compact_json_snippet,
    count_secret_matches,
    is_tracker,
    parse_discovery_report,
    summarize,
    summarize_focus,
)


@pytest.fixture
def fake_report(tmp_path: Path) -> Path:
    """Create a representative fake discovery report with tracker URLs, duplicates, binary frames, and a token."""
    content = """# Platform Discovery Report

> [!WARNING]
> ## REVIEW BEFORE SHARING
> Please review this report before sharing externally to verify that no sensitive
> credentials, account IDs, or unintended tokens are present.

- **Generated At**: 2026-11-01T10:00:00.000000+00:00
- **Target URL**: https://platform.example.com/trading
- **Monitoring Duration**: 900s
- **Total Redactions Applied**: 42
- **Binary Frames Count**: 5
- **Dropped Tracker Requests/Frames**: 12
- **Dropped Endpoints (Over 2000 Cap)**: 0

## 1. Safety Compliance
- **Read-Only Mode**: Confirmed.
- **Credential Safety**: No request bodies stored.
- **URL Sanitation**: All query strings stripped.

## 2. WebSocket Endpoints
- `wss://ws.example.com/v1/stream?token=fake_secret_token_123`
- `wss://ws.example.com/v1/stream?token=another_token_456`
- `wss://feed.example.com/quotes`

## 3. Binary Frames Summary
- Total binary frames: 5
```json
[
  {
    "direction": "in",
    "len": 18,
    "hex_prefix": "0102030405060708090a0b0c0d0e0f10"
  }
]
```

## 4. Message Classification & Redacted Samples

### Price / Candle / Tick Messages (1 types captured)
- **Type**: `quote_tick` (Count: 25)
```json
{
  "symbol": "EURUSD",
  "price": 1.0855,
  "time": 1700000000
}
```

### Balance / Account Messages (1 types captured)
- **Type**: `balance_update` (Count: 4)
```json
{
  "balance": "<num>",
  "currency": "<str:3>",
  "account_id": "<num>"
}
```

### Unknown / Other Messages (0 types captured)
_No messages captured in this category during monitoring window._

## 5. HTTP Endpoints Sample (Top 30, Query Strings Stripped)
- `https://11559740.fls.doubleclick.net/activityi;src=11559740;type=all;u=998877`
- `https://google-analytics.com/collect?v=1&tid=UA-12345`
- `https://adservice.google.com/ddm/fls/z/dc_pre=CJ7y2b`
- `https://api.example.com/v1/users/101/balance?token=fake_token_xyz`
- `https://api.example.com/v1/users/202/balance`
- `https://api.example.com/v1/deals/12345678-abcd-1234-cdef-1234567890ab/status`
- `https://api.example.com/bo-assets/v7/assets;matrix=param`

## 6. DOM Elements Identified (Read-Only)
- Page Title: Example Platform
- Current URL: https://platform.example.com/trading
"""
    rep_file = tmp_path / "fake_discovery_report.md"
    rep_file.write_text(content, encoding="utf-8")
    return rep_file


def test_summarize_discovery_generates_valid_summary(fake_report: Path, tmp_path: Path):
    """Summarizer generates discovery_summary.md <= 600 lines, excludes trackers, and verifies counts."""
    out_file = tmp_path / "discovery_summary.md"
    summary_text = summarize(input_path=fake_report, output_path=out_file)

    assert out_file.exists()
    lines = summary_text.splitlines()

    # 1. Line count constraint: strictly at most 600 lines
    assert len(lines) <= 600
    assert len(lines) > 20

    # 2. Key counts verified
    assert "- **Text WebSocket Frames**: 29" in summary_text  # 25 quote_tick + 4 balance_update
    assert "- **Binary Frames**: 5" in summary_text
    assert "- **Distinct WebSocket Hosts/Paths**: 2" in summary_text  # ws.example.com/v1/stream & feed.example.com/quotes
    assert "- **Redactions Applied**: 42" in summary_text

    # 3. Trackers excluded from platform HTTP endpoints
    assert "doubleclick" not in summary_text.lower()
    assert "google-analytics" not in summary_text.lower()
    assert "adservice" not in summary_text.lower()

    # 4. Collapsed paths verified
    assert "/v1/users/{id}/balance" in summary_text
    assert "/v1/deals/{uuid}/status" in summary_text
    assert "/bo-assets/v7/assets" in summary_text
    assert "matrix=param" not in summary_text

    # 5. Categorized messages present
    assert "### Category: PRICE" in summary_text
    assert "quote_tick" in summary_text
    assert "### Category: Balance" in summary_text
    assert "balance_update" in summary_text

    # 6. Final secret-pattern line is present and zero
    last_line = lines[-1].strip()
    assert last_line == "secret-pattern matches in this summary: 0"


def test_collapse_url_logic():
    """collapse_url strips queries, ';' params, and collapses numbers and UUIDs."""
    host, path = collapse_url("https://api.binomo.com/v1/users/98765/balance?apiKey=secret123;session=abc#tab")
    assert host == "api.binomo.com"
    assert path == "/v1/users/{id}/balance"

    host2, path2 = collapse_url("https://api.binomo.com/deals/abcdef01-2345-6789-abcd-ef0123456789/close")
    assert host2 == "api.binomo.com"
    assert path2 == "/deals/{uuid}/close"


def test_is_tracker_detection():
    """is_tracker identifies ad, analytics, and tracking domains."""
    assert is_tracker("11559740.fls.doubleclick.net")
    assert is_tracker("google-analytics.com")
    assert is_tracker("adservice.google.com")
    assert is_tracker("sentry.io")
    assert is_tracker("connect.facebook.net")
    assert not is_tracker("api.binomo.com")
    assert not is_tracker("platform.example.com")


def test_count_secret_matches_detects_real_leaks():
    """count_secret_matches finds raw JWT, email, phone, IP, or long token if unredacted."""
    clean_lines = [
        "- **Type**: `quote_tick` (Count: 10)",
        "```json",
        '{"price": 1.0855, "symbol": "EURUSD", "token": "<str:20>"}',
        "```",
    ]
    assert count_secret_matches(clean_lines) == 0

    dirty_lines = [
        "Leaked email: trader_bob@example.com",
        "Leaked IP: 198.51.100.25",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc1234567890",
    ]
    assert count_secret_matches(dirty_lines) == 3


def test_pass2_phoenix_sequence_summary_fixture(tmp_path: Path):
    """Pass 2 fixture test: Phoenix sequence summary generation with all required sections."""
    phoenix_report_content = """# Platform Discovery Report

> [!WARNING]
> ## REVIEW BEFORE SHARING
> Please review this report before sharing externally.

- **Generated At**: 2026-11-01T12:00:00.000000+00:00
- **Target URL**: https://platform.example.com/trading
- **Monitoring Duration**: 60s
- **Total Redactions Applied**: 88
- **Binary Frames Count**: 14
- **Dropped Tracker Requests/Frames**: 22
- **Dropped Endpoints (Over 2000 Cap)**: 0
- **i18n Messages Dropped**: 18
- **Social Messages Ignored**: 5

## 1. Safety Compliance
- **Read-Only Mode**: Confirmed.
- **Credential Safety**: No request bodies stored.
- **URL Sanitation**: All query strings stripped.

## 2. WebSocket Endpoints
- `wss://ws.example.com/socket/websocket`

## 3. Phoenix Join Topics
- Topic: `quotes:Z-CRY/IDX` (Count: 12)
- Topic: `quotes:EUR/USD` (Count: 4)
- Topic: `user:12345` (Count: 1)

## 4. Subscribe and Unsubscribe Messages
- Host: `wss://ws.example.com/socket/websocket` | Topic: `quotes:Z-CRY/IDX` | Event: `phx_join`
```json
[
  "1",
  "1",
  "quotes:Z-CRY/IDX",
  "phx_join",
  {}
]
```

## 5. Binary Frames Summary
- Total binary frames: 14
```json
[
  {
    "direction": "in",
    "len": 18,
    "hex_prefix": "0102030405060708090a0b0c0d0e0f10"
  }
]
```

## 6. Message Classification & Redacted Samples

### Category: Prices / Quotes (1 pairs captured)
- **Pair**: `(quotes:Z-CRY/IDX, quote)` (Count: 120, Rate: 2.00 msg/s)
```json
{
  "ric": "Z-CRY/IDX",
  "rate": 123.45,
  "time": 1700000001,
  "trend": "up"
}
```
```json
{
  "ric": "Z-CRY/IDX",
  "rate": 123.50,
  "time": 1700000002,
  "trend": "up"
}
```

### Category: Assets / Payouts (1 pairs captured)
- **Pair**: `(assets, asset_trading_settings_changed_v2)` (Count: 8, Rate: 0.13 msg/s)
```json
{
  "ric": "Z-CRY/IDX",
  "payment_rate": 85,
  "status": "active"
}
```

### Category: Balance (1 pairs captured)
- **Pair**: `(user:12345, balance_changed)` (Count: 6, Rate: 0.10 msg/s)
```json
{
  "balance": "<num>",
  "currency": "USD",
  "is_demo": true,
  "account_type": "demo"
}
```

### Category: Deals (2 pairs captured)
- **Pair**: `(deals, deal_opened)` (Count: 3, Rate: 0.05 msg/s)
```json
{
  "deal_id": "<num>",
  "ric": "Z-CRY/IDX",
  "direction": "call",
  "rate": 123.45,
  "open_time": 1700000010,
  "payment_rate": 85
}
```
- **Pair**: `(deals, deal_closed)` (Count: 3, Rate: 0.05 msg/s)
```json
{
  "deal_id": "<num>",
  "ric": "Z-CRY/IDX",
  "close_rate": 123.75,
  "close_time": 1700000070,
  "profit": 18.5,
  "status": "win"
}
```

### Category: System (1 pairs captured)
- **Pair**: `(quotes:Z-CRY/IDX, phx_join)` (Count: 12, Rate: 0.20 msg/s)
```json
[
  "1",
  "1",
  "quotes:Z-CRY/IDX",
  "phx_join",
  {}
]
```

## 7. Large Data Frames Breakdown
- **Topic**: `quotes:Z-CRY/IDX`
  - Structure: Array of objects with keys: ['rate', 'ric', 'time', 'trend']
  - Price Field: `rate`
  - Time Field: `time`
```json
[
  {
    "ric": "Z-CRY/IDX",
    "rate": 123.45,
    "time": 1700000000,
    "trend": "up"
  },
  {
    "ric": "Z-CRY/IDX",
    "rate": 123.50,
    "time": 1700000001,
    "trend": "up"
  }
]
```

## 8. Demo vs Real Account Fields
- `account_type`: `"demo"`
- `demo`: `true`
- `is_demo`: `true`
- `real`: `false`

## 9. Platform HTTP Endpoints
- [GET] `https://api.example.com/v1/assets` (status: 200, count: 5)
  - Sample JSON payload:
```json
{
  "status": "ok",
  "actives": [
    {
      "ric": "Z-CRY/IDX",
      "name": "Crypto IDX"
    }
  ]
}
```
- [GET] `https://api.example.com/v1/users/{id}/balance` (status: 200, count: 2)
  - Sample JSON payload:
```json
{
  "balance": "<num>",
  "is_demo": true,
  "account_type": "demo"
}
```

## 10. DOM Elements Identified (Read-Only)
- Page Title: Trading Platform
- Current URL: https://platform.example.com/trading
"""
    rep_file = tmp_path / "phoenix_discovery_report.md"
    rep_file.write_text(phoenix_report_content, encoding="utf-8")

    out_summary = tmp_path / "phoenix_discovery_summary.md"
    summary_text = summarize(input_path=rep_file, output_path=out_summary)

    lines = summary_text.splitlines()

    # 1. Total line count strictly <= 900
    assert len(lines) <= 900
    assert len(lines) >= 30

    # 2. Key counts
    assert "- **Text WebSocket Frames**: 152" in summary_text  # 120 + 8 + 6 + 3 + 3 + 12 = 152
    assert "- **Binary Frames**: 14" in summary_text
    assert "- **Distinct WebSocket Hosts/Paths**: 1" in summary_text
    assert "- **Redactions Applied**: 88" in summary_text
    assert "- **i18n Messages Filtered**: 18" in summary_text

    # 3. WebSocket host and phx_join topics with counts
    assert "wss://ws.example.com/socket/websocket" in summary_text
    assert "- Topic: `quotes:Z-CRY/IDX` (Count: 12)" in summary_text
    assert "- Topic: `quotes:EUR/USD` (Count: 4)" in summary_text

    # 4. Full redacted subscribe / unsubscribe messages with rics and host
    assert "## 3. Subscribe / Unsubscribe Messages (Full Redacted)" in summary_text
    assert "Host: `wss://ws.example.com/socket/websocket`" in summary_text
    assert "quotes:Z-CRY/IDX" in summary_text

    # 5. Core categories: prices/quotes, assets/payouts, balance, deals with count, rate, and examples
    assert "### Category: PRICE / Quotes" in summary_text or "Prices / Quotes" in summary_text
    assert "(quotes:Z-CRY/IDX, quote)" in summary_text
    assert "2.00 msg/s" in summary_text
    assert "### Category: Assets / Payouts" in summary_text
    assert "asset_trading_settings_changed_v2" in summary_text
    assert "### Category: Balance" in summary_text
    assert "balance_changed" in summary_text
    assert "### Category: Deals" in summary_text
    assert "deal_opened" in summary_text
    assert "deal_closed" in summary_text

    # 6. Large Data Frames breakdown with price and time fields
    assert "## 5. Large Data Frames Breakdown" in summary_text
    assert "Price Field: `rate`" in summary_text
    assert "Time Field: `time`" in summary_text

    # 7. Platform HTTP endpoints with method, status, count, and matching sample
    assert "## 6. Platform HTTP Endpoints" in summary_text
    assert "[GET] `/v1/assets` (status: 200, count: 5)" in summary_text
    assert "Sample payload (verified matching endpoint):" in summary_text

    # 8. Demo vs real fields shown with values
    assert "## 7. Demo vs Real Account Indicators" in summary_text
    assert "- `is_demo`: `true`" in summary_text
    assert "- `account_type`: `\"demo\"`" in summary_text

    # 9. Final line: secret-pattern matches in this summary: 0
    last_line = lines[-1].strip()
    assert last_line == "secret-pattern matches in this summary: 0"


def test_per_section_budgets_thousands_of_messages_and_endpoints(tmp_path: Path):
    """Test summarizer with thousands of price messages, many deals, and many endpoints.
    Verifies:
      1. Every section still appears (including HTTP endpoints, samples, demo-vs-real).
      2. Overflowing sections end with '[section X truncated: N more entries omitted]'.
      3. Overview table at top reports line counts and truncation status.
      4. Total lines stay strictly at or under 900.
      5. Final secret-pattern line is preserved with 0 matches.
    """
    report_lines = [
        "# Platform Discovery Report",
        "",
        "- **Generated At**: 2026-11-01T12:00:00.000000+00:00",
        "- **Target URL**: https://platform.example.com/trading",
        "- **Monitoring Duration**: 900s",
        "- **Total Redactions Applied**: 50000",
        "- **Binary Frames Count**: 25",
        "- **Dropped Tracker Requests/Frames**: 100",
        "- **Dropped Endpoints (Over 2000 Cap)**: 0",
        "- **i18n Messages Dropped**: 50",
        "",
        "## 2. WebSocket Endpoints",
        "- `wss://ws.binomo.com/socket`",
        "- `wss://as.binomo.com/socket`",
        "",
        "## 3. Phoenix Join Topics",
    ]
    # Add 25 join topics
    for i in range(25):
        report_lines.append(f"- Topic: `quotes:ASSET_{i}` (Count: {100 + i})")

    report_lines.extend([
        "",
        "## 4. Subscribe and Unsubscribe Messages",
    ])
    for i in range(8):
        report_lines.extend([
            f"- Host: `wss://ws.binomo.com/socket` | Topic: `quotes:ASSET_{i}` | Event: `phx_join`",
            "```json",
            json.dumps(["1", str(i), f"quotes:ASSET_{i}", "phx_join", {"asset": f"ASSET_{i}"}], indent=2),
            "```",
        ])

    report_lines.extend([
        "",
        "## 6. Message Classification & Redacted Samples",
        "",
        "### Category: Prices / Quotes (30 pairs captured)",
    ])
    # 30 price pairs with 2 examples each (thousands of frames simulated)
    for i in range(30):
        report_lines.extend([
            f"- **Pair**: `(quotes:PAIR_{i}, quote)` (Count: 500, Rate: 5.50 msg/s)",
            "```json",
            json.dumps({"ric": f"PAIR_{i}", "rate": 100.0 + i, "time": 1700000000 + i}, indent=2),
            "```",
            "```json",
            json.dumps({"ric": f"PAIR_{i}", "rate": 100.5 + i, "time": 1700000001 + i}, indent=2),
            "```",
        ])

    report_lines.extend([
        "",
        "### Category: Balance (5 pairs captured)",
    ])
    for i in range(5):
        report_lines.extend([
            f"- **Pair**: `(user:{i}, balance_changed)` (Count: 10, Rate: 0.10 msg/s)",
            "```json",
            json.dumps({"balance": "<num>", "currency": "USD", "is_demo": True}, indent=2),
            "```",
        ])

    report_lines.extend([
        "",
        "### Category: Deals (25 pairs captured)",
    ])
    # 25 deal pairs with multiple examples
    for i in range(25):
        report_lines.extend([
            f"- **Pair**: `(deals, deal_{i})` (Count: 20, Rate: 0.20 msg/s)",
            "```json",
            json.dumps({"deal_id": "<num>", "direction": "call", "profit": 10.0 + i}, indent=2),
            "```",
            "```json",
            json.dumps({"deal_id": "<num>", "direction": "put", "status": "win"}, indent=2),
            "```",
        ])

    report_lines.extend([
        "",
        "### Category: Assets / Payouts (6 pairs captured)",
    ])
    for i in range(6):
        report_lines.extend([
            f"- **Pair**: `(assets, asset_{i})` (Count: 5, Rate: 0.05 msg/s)",
            "```json",
            json.dumps({"ric": f"ASSET_{i}", "payout": 85}, indent=2),
            "```",
        ])

    report_lines.extend([
        "",
        "### Category: System (10 pairs captured)",
    ])
    for i in range(10):
        report_lines.extend([
            f"- **Pair**: `(phoenix, sys_{i})` (Count: 15, Rate: 0.15 msg/s)",
            "```json",
            json.dumps({"status": "ok"}, indent=2),
            "```",
        ])

    report_lines.extend([
        "",
        "## 8. Demo vs Real Account Fields",
        "- `account_type`: `\"demo\"`",
        "- `demo`: `true`",
        "- `is_demo`: `true`",
        "- `real`: `false`",
        "",
        "## 9. Platform HTTP Endpoints",
    ])
    # 150 distinct HTTP endpoints to trigger HTTP list truncation
    for i in range(150):
        method = "POST" if i % 3 == 0 else "GET"
        if i < 15:
            # Endpoints matching trading primitives with sample payloads
            report_lines.extend([
                f"- [{method}] `https://api.binomo.com/v1/assets/item_{i}` (status: 200, count: {i + 1})",
                "  - Sample JSON payload:",
                "```json",
                json.dumps({"id": "<num>", "name": f"Asset {i}", "payout": 80 + (i % 10)}, indent=2),
                "```",
            ])
        else:
            report_lines.append(f"- [{method}] `https://api.binomo.com/v1/endpoint/path_{i}` (status: 200, count: 1)")

    huge_report = tmp_path / "huge_discovery_report.md"
    huge_report.write_text("\n".join(report_lines), encoding="utf-8")

    out_file = tmp_path / "huge_discovery_summary.md"
    summary_text = summarize(input_path=huge_report, output_path=out_file)

    lines = summary_text.splitlines()

    # 1. Total line count stays strictly at or under 900 lines
    assert len(lines) <= 900
    assert len(lines) >= 100

    # 2. Overview table is present at top with line counts and truncation status
    assert "## Summary Sections Overview" in summary_text
    assert "| Section | Budget | Line Count | Truncated |" in summary_text
    assert "| Prices / Quotes | 140 |" in summary_text
    assert "| Deals | 160 |" in summary_text
    assert "| HTTP Endpoints List | 70 |" in summary_text

    # 3. Truncated sections end with expected notice
    assert "[section Prices / Quotes truncated:" in summary_text
    assert "[section Deals truncated:" in summary_text
    assert "[section HTTP Endpoints List truncated:" in summary_text

    # 4. Later sections STILL APPEAR even with heavy truncation of earlier messages:
    assert "Platform HTTP Endpoints (Trackers Excluded)" in summary_text
    assert "## HTTP Key Trading Samples" in summary_text
    assert "## 7. Demo vs Real Account Indicators" in summary_text or "Demo vs Real Account Indicators" in summary_text
    assert "- `is_demo`: `true`" in summary_text
    assert "- `account_type`: `\"demo\"`" in summary_text

    # 5. Final line is strictly preserved and zero secret matches
    last_line = lines[-1].strip()
    assert last_line == "secret-pattern matches in this summary: 0"


def test_focus_report_generation(tmp_path: Path):
    """Part 1 Focus Extraction: fixture report with all target WS pairs, HTTP endpoints,
    non-i18n demo/real fields, candle URL pattern, and dropped static assets.
    Verifies focus output contains all items, stays <= 500 lines, and has 0 secret matches.
    """
    report_lines = [
        "# Platform Discovery Report",
        "",
        "- **Generated At**: 2026-11-01T10:00:00.000000+00:00",
        "- **Target URL**: https://platform.example.com/trading",
        "- **Monitoring Duration**: 900s",
        "- **Total Redactions Applied**: 50",
        "- **Binary Frames Count**: 2",
        "- **Dropped Tracker Requests/Frames**: 10",
        "- **Dropped Endpoints (Over 2000 Cap)**: 0",
        "",
        "## 2. WebSocket Endpoints",
        "- `wss://ws.binomo.com/socket/websocket`",
        "",
        "## 3. Phoenix Join Topics",
        "- Topic: `quotes:EURUSD` (Count: 1)",
        "- Topic: `range_stream:Z-CRY/IDX` (Count: 1)",
        "- Topic: `asset:123` (Count: 1)",
        "- Topic: `user:1001` (Count: 1)",
        "- Topic: `bo` (Count: 1)",
        "- Topic: `social:chat` (Count: 1)",
        "",
        "## 4. Subscribe and Unsubscribe Messages",
        "- Host: `wss://ws.binomo.com/socket/websocket` | Topic: `quotes:EURUSD` | Event: `phx_join`",
        "```json",
        json.dumps(["1", "1", "quotes:EURUSD", "phx_join", {}], indent=2),
        "```",
        "",
        "## 6. Message Classification & Redacted Samples",
        "",
        "### Category: Prices / Quotes (2 pairs captured)",
        "- **Pair**: `(range_stream:Z-CRY/IDX, quotes_range)` (Count: 10, Rate: 0.01 msg/s)",
        "```json",
        json.dumps({
            "quotes": [
                {"rate": 1.0855, "time": 1700000000},
                {"rate": 1.0856, "time": 1700000001}
            ]
        }, indent=2),
        "```",
        "",
        "### Category: Assets / Payouts (2 pairs captured)",
        "- **Pair**: `(asset:123, updated)` (Count: 5, Rate: 0.01 msg/s)",
        "```json",
        json.dumps({"id": 123, "name": "EUR/USD", "status": "active"}, indent=2),
        "```",
        "- **Pair**: `(asset:123, asset_trading_settings_changed_v2)` (Count: 2, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"ric": "EURUSD", "payment_rate": 85}, indent=2),
        "```",
        "",
        "### Category: Balance (1 pairs captured)",
        "- **Pair**: `(account, balance_changed)` (Count: 3, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"balance": "<num>", "currency": "USD", "is_demo": True}, indent=2),
        "```",
        "",
        "### Category: Deals (3 pairs captured)",
        "- **Pair**: `(bo, opened)` (Count: 2, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"deal_id": "<num>", "direction": "call", "amount": 100}, indent=2),
        "```",
        "- **Pair**: `(bo, closed)` (Count: 2, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"deal_id": "<num>", "profit": 85.0, "status": "win"}, indent=2),
        "```",
        "- **Pair**: `(bo, close_deal_batch)` (Count: 1, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"deals": [{"deal_id": "<num>", "status": "win"}]}, indent=2),
        "```",
        "",
        "### Category: System (2 pairs captured)",
        "- **Pair**: `(bo, create)` (Count: 1, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"ric": "EURUSD", "direction": "call", "amount": 10}, indent=2),
        "```",
        "- **Pair**: `(bo, phx_reply)` (Count: 1, Rate: 0.00 msg/s)",
        "```json",
        json.dumps({"status": "ok", "response": {"deal_id": "<num>"}}, indent=2),
        "```",
        "",
        "## 8. Demo vs Real Account Fields",
        "- `account_type`: `\"demo\"`",
        "- `is_demo`: `true`",
        "- `real`: `false`",
        "- `profit_on_demo`: `\"you are in profit on the demo account today\"`",
        "- `welcome_text`: `\"Welcome to Binomo trading platform please choose demo or real\"`",
        "",
        "## 9. Platform HTTP Endpoints",
        "- [GET] `https://api.binomo.com/bo-assets/v7/assets` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps([{"id": 1, "name": "BTC"}, {"id": 2, "name": "ETH"}], indent=2),
        "```",
        "- [GET] `https://api.binomo.com/bo-deals-history/v3/deals/trade` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps([{"id": 10, "ric": "BTCUSD"}, {"id": 11, "ric": "EURUSD"}], indent=2),
        "```",
        "- [GET] `https://api.binomo.com/bo/v1/config` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps({"min_duration": 60, "max_duration": 3600}, indent=2),
        "```",
        "- [GET] `https://api.binomo.com/platform/private/v2/currencies` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps([{"iso": "USD"}, {"iso": "EUR"}], indent=2),
        "```",
        "- [GET] `https://api.binomo.com/platform/time` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps({"server_time": 1700000000}, indent=2),
        "```",
        "- [GET] `https://api.binomo.com/candles/v1/assets/999/60?locale=en&from=1000&to=2000` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps({
            "data": [
                {"open": 1.085, "high": 1.086, "low": 1.084, "close": 1.0855, "created_at": 1700000000},
                {"open": 1.0855, "high": 1.087, "low": 1.085, "close": 1.0865, "created_at": 1700000060}
            ],
            "errors": [],
            "success": True
        }, indent=2),
        "```",
        "- [GET] `https://static.binomo.com/assets/app.js` (status: 200, count: 1)",
        "- [GET] `https://static.binomo.com/assets/style.css` (status: 200, count: 1)",
        "- [GET] `https://static.binomo.com/static/logo.png` (status: 200, count: 1)",
    ]

    fixture_file = tmp_path / "focus_fixture_report.md"
    fixture_file.write_text("\n".join(report_lines), encoding="utf-8")

    out_focus = tmp_path / "discovery_focus_test.md"
    focus_text = summarize_focus(input_path=fixture_file, output_path=out_focus)

    lines = focus_text.splitlines()

    # 1. Stays <= 500 lines
    assert len(lines) <= 500

    # 2. Contains all focus WS items
    assert "quotes_range" in focus_text
    assert "asset:123" in focus_text
    assert "asset_trading_settings_changed_v2" in focus_text
    assert "opened" in focus_text
    assert "closed" in focus_text
    assert "close_deal_batch" in focus_text
    assert "balance_changed" in focus_text
    assert "create" in focus_text
    assert "phx_reply" in focus_text

    # 3. Contains one phx_join per family
    assert "quotes:EURUSD" in focus_text
    assert "range_stream:Z-CRY/IDX" in focus_text

    # 4. HTTP Focus endpoints present
    assert "/bo-assets/v7/assets" in focus_text
    assert "/bo-deals-history/v3/deals/trade" in focus_text
    assert "/bo/v1/config" in focus_text
    assert "/platform/private/v2/currencies" in focus_text
    assert "/platform/time" in focus_text
    assert "/candles/" in focus_text

    # 5. Candle URL pattern prints timeframe 60 and query param names only (?locale&from&to)
    assert "/60?locale&from&to" in focus_text

    # 6. Status tags present (complete or truncated)
    assert "Status: complete" in focus_text or "Status: truncated" in focus_text

    # 7. Static assets are dropped
    assert "app.js" not in focus_text
    assert "style.css" not in focus_text
    assert "logo.png" not in focus_text

    # 8. Demo vs Real: UI translation sentences dropped, discriminator flags preserved
    assert "- `account_type`: `\"demo\"`" in focus_text
    assert "- `is_demo`: `true`" in focus_text
    assert "- `real`: `false`" in focus_text
    assert "welcome_text" not in focus_text
    assert "profit_on_demo" not in focus_text

    # 9. Secret scan: final line must say secret-pattern matches in this focus: 0
    assert lines[-1].strip() == "secret-pattern matches in this focus: 0"


def test_focus_report_pass_3_additions(tmp_path: Path):
    """Pass 3 focus report additions:

    - as.binomo.com non-Phoenix socket data (events, price/time fields, rate per asset, frames)
    - asset:<ric> non-phx events
    - Deals status counts (open, win, lost) and distinct option_types (turbo, binary)
    - Trading panel DOM accessibility controls table (amount, expiry, call/put, asset, account)
    - All assets compact table and first 2 asset objects
    - Output <= 500 lines and secret-pattern matches == 0
    """
    report_lines = [
        "# Platform Discovery Report",
        "- **Target URL**: https://binomo.com",
        "- **Monitoring Duration**: 60s",
        "- **Total Redactions Applied**: 10",
        "- **Binary Frames Count**: 0",
        "- **Dropped Tracker Requests/Frames**: 0",
        "- **Dropped Endpoints (Over 2000 Cap)**: 0",
        "",
        "## 2. WebSocket Endpoints",
        "- `wss://ws.binomo.com/socket/websocket`",
        "- `wss://as.binomo.com/ws`",
        "",
        "## 4. Subscribe and Unsubscribe Messages",
        "- Host: `wss://ws.binomo.com` | Topic: `quotes:EURUSD` | Event: `phx_join`",
        "```json",
        json.dumps({"topic": "quotes:EURUSD", "event": "phx_join", "payload": {}}),
        "```",
        "",
        "## 6. Message Classification & Redacted Samples",
        "",
        "### Deals Messages (2 types captured)",
        "- **Pair**: `(bo, opened)` (Count: 5, Rate: 0.1 msg/s)",
        "```json",
        json.dumps({"status": "open", "option_type": "turbo", "amount": "<num>"}),
        "```",
        "- **Pair**: `(bo, closed)` (Count: 3, Rate: 0.05 msg/s)",
        "```json",
        json.dumps({"status": "win", "option_type": "turbo", "amount": "<num>"}),
        "```",
        "- **Pair**: `(bo, closed)` (Count: 2, Rate: 0.03 msg/s)",
        "```json",
        json.dumps({"status": "lost", "option_type": "binary", "amount": "<num>"}),
        "```",
        "",
        "### Assets / Payouts Messages (1 types captured)",
        "- **Pair**: `(asset:EURUSD, settings_changed)` (Count: 2, Rate: 0.02 msg/s)",
        "```json",
        json.dumps({"asset_name": "EUR/USD", "payment_rate": 85, "trading_status": "open"}),
        "```",
        "",
        "## 8. Demo vs Real Account Fields",
        "- `is_demo`: `true`",
        "",
        "## 9. Platform HTTP Endpoints",
        "- [GET] `https://api.binomo.com/bo-assets/v7/assets` (status: 200, count: 1)",
        "  - Sample JSON payload:",
        "```json",
        json.dumps({
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
                        "trading_status": "open",
                    },
                ]
            }
        }, indent=2),
        "```",
        "",
        "## 12. DOM Accessibility Snapshots",
        "### Snapshot: `start` (Elements: 7, Time: `12:00:00`)",
        "```json",
        json.dumps([
            {"role": "textbox", "name": "Amount", "data-testid": "qa_amount_input"},
            {"role": "combobox", "name": "Expiration", "data-testid": "qa_expiry_select"},
            {"role": "option", "name": "1 min", "data-testid": "qa_expiry_1m"},
            {"role": "button", "name": "Call", "data-testid": "qa_call_btn"},
            {"role": "button", "name": "Put", "data-testid": "qa_put_btn"},
            {"role": "button", "name": "Asset EUR/USD", "data-testid": "qa_asset_select"},
            {"role": "button", "name": "Account Switch Demo", "data-testid": "qa_account_switch"},
        ], indent=2),
        "```",
        "",
        "## 13. Non-Phoenix WebSocket Data",
        "### Host: `wss://as.binomo.com/ws` (Total messages: 42)",
        "- Distinct Event Types: {'quotes': 40, 'subscribe': 2}",
        "- Price Field: `price` | Time Field: `time`",
        "- Message Rate Per Asset:",
        "  - `Z-CRY/IDX`: 25 msgs (0.42 msg/s)",
        "  - `EURUSD`: 17 msgs (0.28 msg/s)",
        "- Full Subscribe / Unsubscribe Messages:",
        "```json",
        json.dumps([{"action": "subscribe", "asset": "Z-CRY/IDX"}], indent=2),
        "```",
        "- First 3 Data Frames Structure:",
        "```json",
        json.dumps([{"event_type": "quotes", "price": 100.5, "time": 1700000000}], indent=2),
        "```",
        "",
        "## 14. Asset List Table",
        "| Name | RIC | Type | Rate/Payout | Status | Min/Max Amount | Expiry/Duration |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        "| Crypto IDX | Z-CRY/IDX | crypto | 85 | open | 1 - 1000 | [60, 120] |",
        "| EUR/USD (OTC) | EURUSD_otc | forex | 82 | open | - | - |",
        "",
        "#### First 2 Asset Objects (Full Structure)",
        "```json",
        json.dumps([
            {"name": "Crypto IDX", "ric": "Z-CRY/IDX", "type": "crypto"},
            {"name": "EUR/USD (OTC)", "ric": "EURUSD_otc", "type": "forex"}
        ], indent=2),
        "```",
    ]

    fixture_file = tmp_path / "pass3_fixture_report.md"
    fixture_file.write_text("\n".join(report_lines), encoding="utf-8")

    out_focus = tmp_path / "discovery_focus_pass3.md"
    focus_text = summarize_focus(input_path=fixture_file, output_path=out_focus)

    lines = focus_text.splitlines()

    # 1. Stays <= 500 lines
    assert len(lines) <= 500

    # 2. Non-Phoenix socket data is present
    assert "wss://as.binomo.com" in focus_text
    assert "**Price Field**: `price`" in focus_text
    assert "**Time Field**: `time`" in focus_text
    assert "Z-CRY/IDX" in focus_text

    # 3. asset:<ric> non-phx event present
    assert "settings_changed" in focus_text

    # 4. Deals status & option types breakdown
    assert "`open`: 1" in focus_text
    assert "`win`: 1" in focus_text
    assert "`lost`: 1" in focus_text
    assert "`turbo`" in focus_text
    assert "`binary`" in focus_text

    # 5. DOM accessibility controls table
    assert "Amount Input" in focus_text
    assert "qa_amount_input" in focus_text
    assert "qa_expiry_select" in focus_text
    assert "qa_call_btn" in focus_text
    assert "qa_put_btn" in focus_text
    assert "qa_asset_select" in focus_text
    assert "qa_account_switch" in focus_text

    # 6. Compact asset table present
    assert "Crypto IDX" in focus_text
    assert "EURUSD_otc" in focus_text

    # 7. Final line secret matches count is 0
    assert lines[-1].strip() == "secret-pattern matches in this focus: 0"


def test_pass4_summarize_focus_with_all_assets_and_archive(tmp_path: Path):
    """Pass 4: Focus report formats all assets with 8 columns, archives copy, stays <= 500 lines, 0 secrets."""
    report_lines = [
        "# Platform Discovery Report",
        "- **Generated At**: 2026-11-02T12:00:00Z",
        "- **Target URL**: https://binomo.com/trading",
        "- **Logged-in indicators**: yes",
        "",
        "## 2. WebSocket Endpoints",
        "- `wss://ws.binomo.com/socket/websocket`: 100 frames (60 in, 40 out)",
        "",
        "## 3. Phoenix Join Topics",
        "- Topic: `quotes:Z-CRY/IDX` (Count: 10)",
        "- Topic: `account` (Count: 2)",
        "",
        "## 4. Subscribe and Unsubscribe Messages",
        "- Host: `wss://ws.binomo.com` | Topic: `quotes:Z-CRY/IDX` | Event: `phx_join`",
        "```json",
        json.dumps(["1", "1", "quotes:Z-CRY/IDX", "phx_join", {}], indent=2),
        "```",
        "",
        "## 8. WebSocket Message Pairs by Category",
        "### Category: deals (2 pairs captured)",
        "- **Pair**: `(bo, opened)` (Count: 5, Rate: 0.1 msg/s)",
        "```json",
        json.dumps({
            "status": "opened",
            "option_type": "turbo",
            "ric": "Z-CRY/IDX",
            "amount": "<num>",
            "payment_rate": 85,
        }, indent=2),
        "```",
        "- **Pair**: `(bo, closed)` (Count: 5, Rate: 0.1 msg/s)",
        "```json",
        json.dumps({
            "status": "win",
            "option_type": "turbo",
            "profit": 18.5,
            "close_rate": 125.5,
        }, indent=2),
        "```",
        "",
        "## 14. Asset List Table (/bo-assets/v7/assets)",
        "| Name | RIC | Type | OTC | Base Rates (Turbo/Std/Classic) | Enabled Accounts | Trading Types | Schedule |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| Crypto IDX | Z-CRY/IDX | crypto | Yes | turbo: 85 | std: 82 | classic: 80 | demo, standard, gold, vip | ftt | durations: [60, 120] |",
        "| EUR/USD (OTC) | EURUSD_otc | forex | Yes | turbo: 84 | std: 81 | classic: 78 | demo, vip | ftt | open_24_7 |",
        "| Apple | AAPL | stock | No | turbo: 80 | std: - | classic: - | vip | ftt | durations: [300] |",
        "| Gold | XAUUSD | commodity | No | turbo: 75 | std: 72 | classic: - | standard, vip | ftt | weekdays |",
        "",
        "### First 2 Asset Objects (Full Structure)",
        "```json",
        json.dumps([
            {"name": "Crypto IDX", "ric": "Z-CRY/IDX", "type": "crypto", "base_payment_rate_turbo": 85},
            {"name": "EUR/USD (OTC)", "ric": "EURUSD_otc", "type": "forex", "base_payment_rate_turbo": 84},
        ], indent=2),
        "```",
    ]

    fixture_file = tmp_path / "discovery_report_20261102_1200.md"
    fixture_file.write_text("\n".join(report_lines), encoding="utf-8")

    out_focus = tmp_path / "discovery_focus_20261102_1200.md"
    focus_text = summarize_focus(input_path=fixture_file, output_path=out_focus)

    lines = focus_text.splitlines()

    # 1. Output file exists
    assert out_focus.exists()

    # 2. reports_archive copy created
    archive_copy = Path("reports_archive") / out_focus.name
    assert archive_copy.exists()
    assert archive_copy.read_text(encoding="utf-8") == focus_text

    # 3. Stays <= 500 lines
    assert len(lines) <= 500

    # 4. All assets present in the table
    assert "Crypto IDX" in focus_text
    assert "Z-CRY/IDX" in focus_text
    assert "EUR/USD (OTC)" in focus_text
    assert "EURUSD_otc" in focus_text
    assert "Apple" in focus_text
    assert "AAPL" in focus_text
    assert "Gold" in focus_text
    assert "XAUUSD" in focus_text
    assert "turbo: 85" in focus_text

    # 5. Secret matches is 0
    assert lines[-1].strip() == "secret-pattern matches in this focus: 0"


def test_pass4_main_timestamp_suffix_and_archive(tmp_path: Path):
    """Pass 4: CLI main() correctly derives suffix and archives."""
    from unittest.mock import patch
    import sys
    from tools.summarize_discovery import main

    report_lines = [
        "# Platform Discovery Report",
        "- **Generated At**: 2026-11-02T13:00:00Z",
        "- **Target URL**: https://binomo.com/trading",
        "- **Logged-in indicators**: no",
        "## 2. WebSocket Endpoints",
        "- `wss://ws.binomo.com/socket/websocket`: 5 frames",
        "## 14. Asset List Table (/bo-assets/v7/assets)",
        "| Name | RIC | Type | OTC | Base Rates (Turbo/Std/Classic) | Enabled Accounts | Trading Types | Schedule |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| Crypto IDX | Z-CRY/IDX | crypto | Yes | turbo: 85 | std: 82 | classic: 80 | demo | ftt | 24/7 |",
    ]

    report_name = "discovery_report_20261102_1300.md"
    rep_file = tmp_path / report_name
    rep_file.write_text("\n".join(report_lines), encoding="utf-8")

    out_focus_name = tmp_path / "discovery_focus_20261102_1300.md"

    test_args = ["tools/summarize_discovery.py", "--input", str(rep_file), "--output", str(out_focus_name), "--focus"]
    with patch.object(sys, "argv", test_args):
        main()

    assert out_focus_name.exists()
    archive_file = Path("reports_archive") / out_focus_name.name
    assert archive_file.exists()






