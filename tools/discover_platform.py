"""Platform discovery tool — inspects platform traffic and DOM read-only.

Usage:
    python tools/discover_platform.py --url https://platform.example.com [--duration 900]

Safety Rule:
    This tool NEVER clicks, fills, types, presses, checks, or evaluates anything.
    It is strictly READ-ONLY. It launches a persistent Playwright browser profile,
    attaches listeners to all pages, monitors network requests and WebSocket frames,
    redacts all credentials and identifiers, and compiles discovery_report.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
import urllib.parse


PROFILE_DIR = Path("state") / "playwright_profile"
REPORT_PATH = Path("discovery_report.md")

# ── Redaction Configuration ───────────────────────────────────────────────────

SAFE_KEYS = {
    "price", "bid", "ask", "open", "high", "low", "close", "rate",
    "payout", "profit", "percent", "ts", "time", "timestamp",
    "symbol", "name", "pair", "ticker", "asset", "type", "event",
    "action", "status", "expiry", "duration",
}

ALWAYS_MASK_KEY_REGEX = re.compile(
    r"token|auth|session|cookie|password|secret|key|email|phone|user|uid|account|ssid|jwt",
    re.IGNORECASE,
)

BALANCE_KEY_REGEX = re.compile(r"balance", re.IGNORECASE)

SKIP_URL_REGEX = re.compile(
    r"login|auth|token|session|signin|password|oauth",
    re.IGNORECASE,
)

JWT_REGEX = re.compile(r"^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}$")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LONG_HEX_B64_REGEX = re.compile(r"^[A-Za-z0-9+/=_-]{25,}$")

CATEGORY_KEYS = {
    "balance": {"balance", "balances", "demo_balance", "live_balance", "account_balance"},
    "trade/order": {"order", "orders", "trade", "trades", "deal", "deals", "position", "positions"},
    "payout": {"payout", "payouts", "profit_percent", "yield", "rate_payout"},
    "price": {"price", "bid", "ask", "quote", "candles", "candle", "ohlc", "rates", "rate", "tick", "ticks"},
    "assets": {"assets", "instruments", "actives", "pairs", "currencies"},
}


class PayloadRedactor:
    """Recursively redacts JSON objects and string values according to strict rules."""

    def __init__(self) -> None:
        self.redaction_count = 0

    def is_sensitive_value(self, val: str) -> bool:
        """Check if a string matches sensitive patterns like JWT, email, phone, IP, or long token."""
        if JWT_REGEX.match(val):
            return True
        if EMAIL_REGEX.search(val):
            return True
        if IP_REGEX.search(val):
            return True
        if len(val) > 24 and LONG_HEX_B64_REGEX.match(val):
            return True
        # Check for typical phone number format
        if len(val) >= 9 and PHONE_REGEX.search(val) and sum(c.isdigit() for c in val) >= 7:
            return True
        return False

    def redact(self, data: Any, parent_key: str = "") -> Any:
        """Redact arbitrary python data structure in-place / copy."""
        if isinstance(data, dict):
            redacted_dict: dict[str, Any] = {}
            for k, v in data.items():
                k_lower = str(k).lower()
                if ALWAYS_MASK_KEY_REGEX.search(k_lower):
                    self.redaction_count += 1
                    if isinstance(v, str):
                        redacted_dict[k] = f"<str:{len(v)}>"
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        redacted_dict[k] = "<num>"
                    elif isinstance(v, dict):
                        redacted_dict[k] = self.redact(v, parent_key=k)
                    elif isinstance(v, list):
                        redacted_dict[k] = self.redact(v, parent_key=k)
                    else:
                        redacted_dict[k] = "<masked>"
                else:
                    redacted_dict[k] = self.redact(v, parent_key=k)
            return redacted_dict

        if isinstance(data, list):
            return [self.redact(item, parent_key=parent_key) for item in data]

        if isinstance(data, str):
            k_lower = parent_key.lower()
            if self.is_sensitive_value(data):
                self.redaction_count += 1
                return f"<str:{len(data)}>"
            if k_lower in SAFE_KEYS and not ALWAYS_MASK_KEY_REGEX.search(k_lower):
                return data
            self.redaction_count += 1
            return f"<str:{len(data)}>"

        if isinstance(data, (int, float)) and not isinstance(data, bool):
            k_lower = parent_key.lower()
            if BALANCE_KEY_REGEX.search(k_lower):
                self.redaction_count += 1
                return "<num>"
            if k_lower in SAFE_KEYS and not ALWAYS_MASK_KEY_REGEX.search(k_lower):
                return data
            self.redaction_count += 1
            return "<num>"

        return data


def classify_category_from_keys(keys: set[str]) -> str:
    """Determine message category based on explicit key names."""
    lower_keys = {k.lower() for k in keys}
    for cat in ["balance", "trade/order", "payout", "price", "assets"]:
        if any(ck in lower_keys for ck in CATEGORY_KEYS[cat]):
            return cat
    return "other"


def extract_all_keys(data: Any) -> set[str]:
    """Recursively collect all dictionary keys in a JSON structure."""
    keys: set[str] = set()
    if isinstance(data, dict):
        for k, v in data.items():
            keys.add(str(k))
            keys.update(extract_all_keys(v))
    elif isinstance(data, list):
        for item in data:
            keys.update(extract_all_keys(item))
    return keys


def extract_message_type(data: Any) -> str:
    """Identify message type from common discriminator keys or first key."""
    if isinstance(data, dict):
        for candidate in ["type", "event", "action", "cmd", "topic", "e"]:
            val = data.get(candidate)
            if val is not None and isinstance(val, (str, int)):
                return str(val)
        if data:
            return str(list(data.keys())[0])
    elif isinstance(data, list) and data:
        return extract_message_type(data[0])
    return "unknown"


# ── Main Discovery Class ──────────────────────────────────────────────────────

class PlatformDiscovery:
    """Strictly read-only platform inspector using Playwright."""

    def __init__(self, target_url: str, duration_seconds: int = 900) -> None:
        self.target_url = target_url
        self.duration_seconds = duration_seconds
        self.redactor = PayloadRedactor()

        # Listeners & Traffic Tracking
        self._attached_pages: set[Any] = set()
        self.ws_endpoints: set[str] = set()
        self.http_endpoints: set[str] = set()

        # Binary frames tracking: count, length, first 16 bytes as hex
        self.binary_frame_count: int = 0
        self.binary_samples: list[dict[str, Any]] = []

        # Message type classification: type -> {count, category, example}
        self.classified_types: dict[str, dict[str, Any]] = {}

    def strip_url(self, raw_url: str) -> str:
        """Strip query parameters and fragments from URL."""
        return raw_url.split("?")[0].split("#")[0]

    def _process_text_frame(self, raw_text: str, source_label: str, clean_url: str) -> None:
        """Parse, redact, and classify a WebSocket or HTTP text payload."""
        # Attempt to parse as JSON
        try:
            data = json.loads(raw_text)
        except Exception:
            # Non-JSON payload: mask length
            self.redactor.redaction_count += 1
            data = {"raw_text": f"<str:{len(raw_text)}>"}

        # Collect keys before redaction for classification
        keys = extract_all_keys(data)
        category = classify_category_from_keys(keys)
        msg_type = extract_message_type(data)

        # Apply redaction
        redacted_data = self.redactor.redact(data)

        if msg_type not in self.classified_types:
            self.classified_types[msg_type] = {
                "count": 1,
                "category": category,
                "example": redacted_data,
                "source": source_label,
            }
        else:
            self.classified_types[msg_type]["count"] += 1

    def _on_response(self, response: Any) -> None:
        """Analyze HTTP responses read-only; never records request headers/bodies/cookies."""
        clean_url = self.strip_url(response.url)
        self.http_endpoints.add(clean_url)

        # Skip all bodies for auth/login/token/password URLs
        if SKIP_URL_REGEX.search(clean_url):
            self.redactor.redaction_count += 1
            return

        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                body_text = response.text()
                self._process_text_frame(body_text, "HTTP_RESP", clean_url)
            except Exception:
                pass

    def _on_websocket(self, ws: Any) -> None:
        """Handle text and binary WebSocket frames."""
        clean_url = self.strip_url(ws.url)
        parsed = urllib.parse.urlparse(clean_url)
        clean_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.ws_endpoints.add(clean_endpoint)

        def on_frame_received(payload: Any) -> None:
            if isinstance(payload, bytes):
                self.binary_frame_count += 1
                if len(self.binary_samples) < 5:
                    self.binary_samples.append({
                        "direction": "in",
                        "len": len(payload),
                        "hex_prefix": payload[:16].hex(),
                    })
            elif isinstance(payload, str):
                self._process_text_frame(payload, "WS_IN", clean_url)

        def on_frame_sent(payload: Any) -> None:
            if isinstance(payload, bytes):
                self.binary_frame_count += 1
                if len(self.binary_samples) < 5:
                    self.binary_samples.append({
                        "direction": "out",
                        "len": len(payload),
                        "hex_prefix": payload[:16].hex(),
                    })
            elif isinstance(payload, str):
                self._process_text_frame(payload, "WS_OUT", clean_url)

        ws.on("framereceived", on_frame_received)
        ws.on("framesent", on_frame_sent)

    def _attach_page_listeners(self, page: Any) -> None:
        """Attach listeners to a page, popup, or newly opened tab."""
        if page in self._attached_pages:
            return
        self._attached_pages.add(page)
        page.on("response", self._on_response)
        page.on("websocket", self._on_websocket)

    def _inspect_page_elements(self, page: Any) -> dict[str, Any]:
        """Read-only inspection of potential DOM elements without any interaction."""
        info: dict[str, Any] = {"title": page.title(), "url": self.strip_url(page.url), "elements_found": []}
        test_selectors = [
            "[class*='balance']",
            "[data-test*='balance']",
            "[class*='payout']",
            "[class*='asset']",
            "[class*='instrument']",
            "[class*='chart']",
        ]
        for sel in test_selectors:
            try:
                matches = page.query_selector_all(sel)
                if matches:
                    info["elements_found"].append({
                        "selector": sel,
                        "count": len(matches),
                        "text_sample": matches[0].inner_text()[:100] if matches else "",
                    })
            except Exception:
                pass
        return info

    def write_report(self, dom_data: Optional[dict[str, Any]] = None) -> None:
        """Compile and write discovery_report.md."""
        now = datetime.now(timezone.utc).isoformat()
        dom = dom_data or {"title": "N/A", "url": "N/A", "elements_found": []}

        # Group classified types by category
        categories = ["price", "assets", "payout", "balance", "trade/order", "other"]
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {c: [] for c in categories}
        for msg_type, info in sorted(self.classified_types.items()):
            cat = info.get("category", "other")
            if cat not in grouped:
                cat = "other"
            grouped[cat].append((msg_type, info))

        def format_section(cat_name: str, items: list[tuple[str, dict[str, Any]]]) -> str:
            lines = [f"### {cat_name.title()} Messages ({len(items)} types captured)"]
            if not items:
                lines.append("_No messages captured in this category during monitoring window._\n")
                return "\n".join(lines)
            for m_type, details in items:
                lines.append(f"- **Type**: `{m_type}` (Count: {details['count']})")
                lines.append("```json")
                lines.append(json.dumps(details["example"], indent=2))
                lines.append("```")
            lines.append("")
            return "\n".join(lines)

        content = f"""# Platform Discovery Report

> [!WARNING]
> ## REVIEW BEFORE SHARING
> Please review this report before sharing externally to verify that no sensitive
> credentials, account IDs, or unintended tokens are present.

- **Generated At**: {now}
- **Target URL**: {self.strip_url(self.target_url)}
- **Monitoring Duration**: {self.duration_seconds}s
- **Total Redactions Applied**: {self.redactor.redaction_count}
- **Binary Frames Count**: {self.binary_frame_count}

## 1. Safety Compliance
- **Read-Only Mode**: Confirmed. Zero clicks, fills, types, presses, checks, or evaluates executed.
- **Credential Safety**: User credentials entered manually in persistent Chromium profile; no request bodies, headers, or cookies stored.
- **URL Sanitation**: All query strings stripped; auth/login endpoints skipped.

## 2. WebSocket Endpoints
{chr(10).join(f"- `{ep}`" for ep in sorted(self.ws_endpoints)) if self.ws_endpoints else "- _No WebSocket endpoints recorded._"}

## 3. Binary Frames Summary
- Total binary frames: {self.binary_frame_count}
```json
{json.dumps(self.binary_samples, indent=2)}
```

## 4. Message Classification & Redacted Samples

{format_section("Price / Candle / Tick", grouped["price"])}
{format_section("Asset / Instrument List", grouped["assets"])}
{format_section("Payout / Yield", grouped["payout"])}
{format_section("Balance / Account", grouped["balance"])}
{format_section("Order / Trade / Position", grouped["trade/order"])}
{format_section("Unknown / Other", grouped["other"])}

## 5. HTTP Endpoints Sample (Top 30, Query Strings Stripped)
{chr(10).join(f"- `{ep}`" for ep in sorted(list(self.http_endpoints))[:30]) if self.http_endpoints else "- _No HTTP endpoints recorded._"}

## 6. DOM Elements Identified (Read-Only)
- Page Title: {dom.get('title', 'N/A')}
- Current URL: {dom.get('url', 'N/A')}
- Identified Selectors:
```json
{json.dumps(dom.get('elements_found', []), indent=2)}
```
"""
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write(content)

    def run(self) -> None:
        """Launch browser, attach context-level listeners, monitor traffic, and write report."""
        from playwright.sync_api import sync_playwright

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print("PLATFORM DISCOVERY TOOL (STRICTLY READ-ONLY)")
        print("Safety Rule: NEVER clicks, fills, types, presses, checks, or evaluates.")
        print(f"Profile directory: {PROFILE_DIR.resolve()}")
        print(f"Target URL: {self.strip_url(self.target_url)}")
        print("=" * 80)

        print("\nCHECKLIST FOR PLATFORM DISCOVERY:")
        print("  1. Log in manually in the browser window.")
        print("  2. Switch to the DEMO account.")
        print("  3. Open the trading screen.")
        print("  4. Open EUR/USD and two other assets.")
        print("  5. Open the asset list / instrument selector.")
        print("  6. Change the chart timeframe (e.g., 1m, 5m).")
        print("  7. Keep a chart open so real-time price ticks stream in.")
        print("  8. [OPTIONAL] Place ONE minimum-size trade by hand on DEMO account")
        print("     so order/trade messages are captured.")
        print("=" * 80 + "\n")

        dom_data: dict[str, Any] = {}
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR.resolve()),
                headless=False,
                viewport={"width": 1280, "height": 800},
            )

            # Listeners at context level: handle all new pages, popups, and existing tabs
            context.on("page", lambda new_page: self._attach_page_listeners(new_page))
            for pg in context.pages:
                self._attach_page_listeners(pg)

            page = context.pages[0] if context.pages else context.new_page()
            self._attach_page_listeners(page)

            print(f"Navigating to {self.target_url}...")
            try:
                page.goto(self.target_url, timeout=30000)
            except Exception as e:
                print(f"Navigation notice: {e}")

            print(f"Monitoring network and WebSocket traffic for {self.duration_seconds}s (Press Ctrl+C to finish early)...")

            start = time.time()
            last_reminder = start
            try:
                while time.time() - start < self.duration_seconds:
                    time.sleep(1)
                    now_t = time.time()
                    if now_t - last_reminder >= 120:
                        elapsed_min = int((now_t - start) // 60)
                        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Reminder ({elapsed_min}m elapsed):")
                        print("  - Ensure you are on DEMO account")
                        print("  - View asset list & switch assets")
                        print("  - Change chart timeframe")
                        print("  - Keep a chart open for ticks")
                        last_reminder = now_t
            except KeyboardInterrupt:
                print("\nMonitoring stopped early by user (Ctrl+C). Saving captured data...")

            try:
                dom_data = self._inspect_page_elements(page)
            except Exception as e:
                print(f"DOM inspection note: {e}")

            try:
                context.close()
            except Exception:
                pass

        print(f"Generating report: {REPORT_PATH}...")
        self.write_report(dom_data)
        print(f"Report generated successfully at {REPORT_PATH.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly read-only platform discovery tool")
    parser.add_argument("--url", required=True, help="Target platform URL (required)")
    parser.add_argument("--duration", type=int, default=900, help="Monitoring duration in seconds (default: 900)")
    args = parser.parse_args()

    discovery = PlatformDiscovery(target_url=args.url, duration_seconds=args.duration)
    discovery.run()


if __name__ == "__main__":
    main()
