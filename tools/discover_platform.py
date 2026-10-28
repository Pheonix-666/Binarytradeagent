"""Platform discovery tool — inspects platform traffic and DOM read-only.

Usage:
    python tools/discover_platform.py [--url https://...] [--duration 60]

Safety Rule:
    This tool NEVER clicks any trade button. It is strictly READ-ONLY.
    It launches a persistent Playwright browser profile, waits for the user to log in,
    monitors network requests and WebSocket frames, and compiles discovery_report.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import Request, Response, WebSocket, sync_playwright


PROFILE_DIR = Path("state") / "playwright_profile"
REPORT_PATH = Path("discovery_report.md")


class PlatformDiscovery:
    """Read-only platform inspector using Playwright persistent browser context."""

    def __init__(self, target_url: str, duration_seconds: int = 60):
        self.target_url = target_url
        self.duration_seconds = duration_seconds
        self.http_traffic: list[dict[str, Any]] = []
        self.ws_traffic: list[dict[str, Any]] = []
        self.price_candidates: list[dict[str, Any]] = []
        self.asset_candidates: list[dict[str, Any]] = []
        self.balance_candidates: list[dict[str, Any]] = []
        self.payout_candidates: list[dict[str, Any]] = []

    def _on_response(self, response: Response) -> None:
        """Analyze HTTP responses for relevant trading market data."""
        url = response.url
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                data = response.json()
                data_str = json.dumps(data)
                self._classify_payload("HTTP", url, data_str)
                self.http_traffic.append({"url": url, "status": response.status})
            except Exception:
                pass

    def _on_websocket(self, ws: WebSocket) -> None:
        """Listen to incoming and outgoing WebSocket frames."""
        ws_url = ws.url

        def on_frame_received(payload: str) -> None:
            self._classify_payload("WS_IN", ws_url, payload)
            if len(self.ws_traffic) < 500:
                self.ws_traffic.append({"url": ws_url, "direction": "in", "sample": payload[:200]})

        def on_frame_sent(payload: str) -> None:
            if len(self.ws_traffic) < 500:
                self.ws_traffic.append({"url": ws_url, "direction": "out", "sample": payload[:200]})

        ws.on("framereceived", on_frame_received)
        ws.on("framesent", on_frame_sent)

    def _classify_payload(self, source_type: str, source_url: str, text: str) -> None:
        """Classify if payload carries prices, assets, payouts, or balance."""
        lower = text.lower()

        # Balance indicators
        if any(kw in lower for kw in ["balance", "account_balance", "real_balance", "demo_balance"]):
            self.balance_candidates.append(
                {"source": source_type, "url": source_url, "sample": text[:300]}
            )

        # Asset / Universe indicators
        if any(kw in lower for kw in ["assets", "instruments", "actives", "pair", "currency"]):
            if any(pair in text for pair in ["EUR", "USD", "GBP", "JPY", "BTC"]):
                self.asset_candidates.append(
                    {"source": source_type, "url": source_url, "sample": text[:300]}
                )

        # Payout indicators
        if any(kw in lower for kw in ["payout", "profit", "percent", "rate", "yield"]):
            self.payout_candidates.append(
                {"source": source_type, "url": source_url, "sample": text[:300]}
            )

        # Price / candle / tick indicators
        if any(kw in lower for kw in ["quote", "tick", "candle", "ohlc", "bid", "ask", "price"]):
            self.price_candidates.append(
                {"source": source_type, "url": source_url, "sample": text[:300]}
            )

    def run(self) -> None:
        """Launch browser, wait for user login, collect traffic, and write report."""
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("PLATFORM DISCOVERY TOOL (READ-ONLY)")
        print("Safety Rule: This tool NEVER clicks any trade button.")
        print(f"Profile directory: {PROFILE_DIR.resolve()}")
        print(f"Target URL: {self.target_url}")
        print("=" * 60)

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR.resolve()),
                headless=False,
                viewport={"width": 1280, "height": 800},
            )

            page = context.new_page() if not context.pages else context.pages[0]
            page.on("response", self._on_response)
            page.on("websocket", self._on_websocket)

            print(f"Navigating to {self.target_url}...")
            try:
                page.goto(self.target_url, timeout=30000)
            except Exception as e:
                print(f"Navigation note: {e}")

            print("\nPlease log in to the platform manually in the browser window.")
            print("Listening for traffic and page elements...")
            print(f"Will collect data for {self.duration_seconds} seconds.")

            start = time.time()
            try:
                while time.time() - start < self.duration_seconds:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user.")

            print("\nCollecting DOM element snapshots...")
            dom_data = self._inspect_page_elements(page)

            print("Closing browser context...")
            context.close()

        print(f"Generating report: {REPORT_PATH}...")
        self._write_report(dom_data)
        print(f"Discovery report generated successfully at {REPORT_PATH.resolve()}")

    def _inspect_page_elements(self, page) -> dict[str, Any]:
        """Read-only inspection of potential DOM elements."""
        info: dict[str, Any] = {"title": page.title(), "url": page.url, "elements_found": []}
        # Look for typical trading element classes/testids without interacting
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
                    info["elements_found"].append(
                        {
                            "selector": sel,
                            "count": len(matches),
                            "text_sample": matches[0].inner_text()[:100] if matches else "",
                        }
                    )
            except Exception:
                pass
        return info

    def _write_report(self, dom_data: dict[str, Any]) -> None:
        """Write findings to discovery_report.md."""
        now = datetime.now(timezone.utc).isoformat()
        content = f"""# Platform Discovery Report

Generated at: {now}
Target URL: {self.target_url}
Monitoring Duration: {self.duration_seconds}s

## 1. Safety Compliance
- **Read-Only Mode**: Confirmed. Zero clicks executed on any trading or order buttons.
- **Credential Safety**: User credentials entered manually in persistent Chromium profile; no credentials stored in code or config.

## 2. WebSocket Traffic Summary
- Total recorded frames: {len(self.ws_traffic)}
- Potential Price/Tick Frames: {len(self.price_candidates)}
- Potential Asset List Frames: {len(self.asset_candidates)}
- Potential Payout Frames: {len(self.payout_candidates)}
- Potential Balance Frames: {len(self.balance_candidates)}

### Sample Price Message
```json
{json.dumps(self.price_candidates[:2], indent=2) if self.price_candidates else "No direct match captured during window"}
```

### Sample Balance Message
```json
{json.dumps(self.balance_candidates[:2], indent=2) if self.balance_candidates else "No direct match captured during window"}
```

## 3. HTTP Traffic Summary
- Total recorded HTTP endpoints: {len(self.http_traffic)}

## 4. DOM Elements Identified (Read-Only)
- Page Title: {dom_data.get('title', 'N/A')}
- Current URL: {dom_data.get('url', 'N/A')}
- Identified Selectors:
```json
{json.dumps(dom_data.get('elements_found', []), indent=2)}
```

## 5. Adapter Implementation Notes (For Phase 10)
- Price feed: WebSocket stream inspection
- Balance feed: WebSocket stream / DOM element
- Assets & Payouts: HTTP initial state or WebSocket message
- Trade action elements: To be wired in Phase 10 demo testing
"""
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write(content)


def main():
    parser = argparse.ArgumentParser(description="Read-only platform discovery tool")
    parser.add_argument("--url", default="https://example.com", help="Platform URL")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in seconds")
    args = parser.parse_args()

    discovery = PlatformDiscovery(target_url=args.url, duration_seconds=args.duration)
    discovery.run()


if __name__ == "__main__":
    main()
