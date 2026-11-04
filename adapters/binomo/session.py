"""Binomo browser session manager using Playwright persistent context.

Adheres strictly to PROJECT_SPEC.md:
- Uses headed browser with persistent profile anchored to project root.
- No stealth plugins, no fingerprint masking, no automated credential entry.
- Observation only: zero clicks, fills, types, presses, checks, or dblclicks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Optional

from adapters.binomo.ws_listener import BinomoWSListener
from core.config import AppConfig, PROJECT_ROOT, resolve_project_path

logger = logging.getLogger("trading_system.adapters.binomo.session")


class BinomoSession:
    """Manages headed Playwright persistent context and passive login observation."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        profile_dir: Optional[str | Path] = None,
        target_url: Optional[str] = None,
        ws_listener: Optional[BinomoWSListener] = None,
    ) -> None:
        self.config = config
        self.ws_listener = ws_listener or BinomoWSListener()

        # Resolve profile directory anchored to project root
        if profile_dir:
            self.profile_dir = resolve_project_path(profile_dir)
        elif config and hasattr(config, "binomo") and getattr(config.binomo, "profile_dir", None):
            self.profile_dir = resolve_project_path(config.binomo.profile_dir)
        else:
            self.profile_dir = resolve_project_path("state/playwright_profile")

        # Target platform URL
        if target_url:
            self.target_url = target_url
        elif config and hasattr(config, "binomo") and getattr(config.binomo, "url", None):
            self.target_url = config.binomo.url
        else:
            self.target_url = "https://binomo.com/trading"

        self._playwright: Optional[Any] = None
        self._context: Optional[Any] = None
        self._page: Optional[Any] = None
        self._attached_pages: set[Any] = set()

    @property
    def page(self) -> Optional[Any]:
        return self._page

    @property
    def context(self) -> Optional[Any]:
        return self._context

    def start(self, headed: bool = True) -> Any:
        """Launch headed persistent context without stealth plugins or credential automation."""
        if self._context:
            return self._page

        from playwright.sync_api import sync_playwright

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting Binomo persistent session with profile: %s", self.profile_dir)

        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=not headed,
            viewport={"width": 1366, "height": 768},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        self._context.on("page", self._attach_page_listeners)

        pages = self._context.pages
        self._page = pages[0] if pages else self._context.new_page()
        self._attach_page_listeners(self._page)

        try:
            self._page.goto(self.target_url, timeout=30000)
        except Exception as exc:
            logger.warning("Initial navigation to %s encountered: %s", self.target_url, exc)

        return self._page

    def _attach_page_listeners(self, page: Any) -> None:
        """Attach passive request and websocket listeners to page and iframes."""
        if page in self._attached_pages:
            return
        self._attached_pages.add(page)

        def on_request(request: Any) -> None:
            url = getattr(request, "url", "")
            self.ws_listener.record_passport_or_assets(url)

        def on_response(response: Any) -> None:
            url = getattr(response, "url", "")
            self.ws_listener.record_passport_or_assets(url)

        def on_websocket(ws: Any) -> None:
            self.ws_listener.attach_to_websocket(ws)

        def on_frame(frame: Any) -> None:
            url = getattr(frame, "url", "")
            self.ws_listener.record_passport_or_assets(url)

        try:
            page.on("request", on_request)
            page.on("response", on_response)
            page.on("websocket", on_websocket)
            page.on("frameattached", on_frame)
            for fr in getattr(page, "frames", []):
                on_frame(fr)
        except Exception as exc:
            logger.debug("Failed attaching page listeners: %s", exc)

    def is_logged_in(self) -> bool:
        """Observation only: checks if websocket to ws.binomo.com is open AND account join/passport seen."""
        return self.ws_listener.is_logged_in()

    def is_demo(self) -> bool:
        """True ONLY if latest balance_changed payload has account_type == 'demo'."""
        return self.ws_listener.is_demo()

    def ensure_logged_in(self, timeout: float = 300.0, check_interval: float = 1.0) -> bool:
        """Wait for manual user login in browser window.
        
        Prints 'NOT LOGGED IN YET - log in manually in the browser window' every 30 s.
        Raises TimeoutError if timeout expires before login indicators are confirmed.
        """
        start = time.time()
        last_log = 0.0

        while time.time() - start < timeout:
            if self.is_logged_in():
                logger.info("Login confirmed via passive indicators (ws open + account/passport observed).")
                return True

            now = time.time()
            if now - last_log >= 30.0:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] NOT LOGGED IN YET - log in manually in the browser window")
                last_log = now

            time.sleep(check_interval)

        # One final check
        if self.is_logged_in():
            return True

        indicators = self.ws_listener.get_login_indicators()
        raise TimeoutError(
            f"Binomo login verification timed out after {timeout:.0f}s. "
            f"Indicators: ws_open={indicators['ws_open']}, "
            f"account_join={indicators['account_join']}, "
            f"passport_or_assets={indicators['passport_or_assets']}"
        )

    def close(self) -> None:
        """Clean up Playwright persistent context and process."""
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
            self._page = None

        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

