"""Tests for Phase 10D: Launch Args Hardening, Selector Safety, and UI Validation Command.

Adheres strictly to Phase 10D instructions:
1. Launch Args: No "AutomationControlled", "no-sandbox", or init script changing navigator.webdriver.
2. Selector Safety: Placeholders in binomo.ui must be literal string TODO_CALIBRATE; startup in demo
   mode refuses while any TODO_CALIBRATE or invalid role (rejecting 'text') remains, naming each one.
   Valid roles are Playwright ARIA roles. Supports specs by test_id, role+name, or css selector.
3. Command `python main.py binomo-validate-ui`: Opens session, verifies demo, prints "resolved (count N)"
   or "NOT FOUND", plus visible text of active labels. Zero clicks, types, or fills. Exits non-zero on failure.
4. config.yaml has TODO_CALIBRATE placeholders; tests stay green via calibrated fixtures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import pytest

from adapters.binomo.session import BinomoSession
from adapters.binomo.trader import (
    BinomoAdapter,
    PLAYWRIGHT_ARIA_ROLES,
    SelectorResolutionError,
    check_selector_spec,
    resolve_locator,
    validate_ui_selectors,
)
from core.config import AppConfig, BinomoConfig, BinomoUIConfig, load_config
import main


# ── Test Fakes ───────────────────────────────────────────────────────────────


class FakeLocator:
    """Mock Playwright Locator with count and text tracking, strictly tracking interactions."""

    def __init__(
        self,
        name: str = "locator",
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

    @property
    def first(self) -> "FakeLocator":
        return self


class FakePage:
    """Mock Playwright Page supporting role, test_id, and css selector lookup."""

    def __init__(self, locators: Optional[dict[str, FakeLocator]] = None) -> None:
        self.locators: dict[str, FakeLocator] = locators or {}
        self.click_count = 0
        self.fill_calls: list[str] = []

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
        if selector in self.locators:
            return self.locators[selector]
        loc = FakeLocator(name=selector, count=1)
        self.locators[selector] = loc
        return loc


def build_calibrated_ui_config() -> BinomoUIConfig:
    """Build a fully calibrated UI configuration for tests."""
    return BinomoUIConfig(
        amount_input={"role": "spinbutton", "name": "Amount"},
        expiry_selector={"role": "button", "name": "Expiration time"},
        expiry_option={"role": "button", "name": "{label}"},
        call_button={"role": "button", "name": "Call"},
        put_button={"role": "button", "name": "Put"},
        asset_selector={"role": "button", "name": "Asset selection"},
        asset_option={"role": "button", "name": "{name}"},
        demo_indicator={"selector": ".demo-indicator"},
        active_asset_label={"selector": ".active-asset"},
        active_expiry_label={"selector": ".active-expiry"},
        blocking_overlays=["[role='dialog']", ".modal-dialog"],
        call_button_labels=["call", "up", "higher"],
        put_button_labels=["put", "down", "lower"],
        validation_samples={"expiry_option": "1 min", "asset_option": "EUR/USD"},
    )


# ── 1. LAUNCH ARGS HARDENING TESTS ──────────────────────────────────────────


def test_session_launch_args_contain_no_stealth_or_insecure_flags():
    """adapters/binomo/session.py must not contain AutomationControlled, no-sandbox, or navigator.webdriver scripts."""
    session_file = Path("adapters") / "binomo" / "session.py"
    assert session_file.exists(), "session.py must exist"

    content = session_file.read_text(encoding="utf-8")

    # Static assertion on source code
    assert "AutomationControlled" not in content, (
        "Found 'AutomationControlled' in adapters/binomo/session.py; stealth automation flags are forbidden"
    )
    assert "no-sandbox" not in content, (
        "Found 'no-sandbox' in adapters/binomo/session.py; weakening sandbox security is forbidden"
    )
    assert "navigator.webdriver" not in content, (
        "Found 'navigator.webdriver' modification in adapters/binomo/session.py; stealth init scripts forbidden"
    )


def test_session_start_runtime_launch_does_not_pass_forbidden_args_or_scripts(tmp_path: Path):
    """Runtime execution of BinomoSession.start() does not pass stealth args or init scripts."""
    cfg = load_config("config.yaml")
    cfg.binomo.profile_dir = str(tmp_path / "test_profile")

    session = BinomoSession(config=cfg)

    mock_playwright = MagicMock()
    mock_chromium = MagicMock()
    mock_playwright.chromium = mock_chromium

    mock_context = MagicMock()
    mock_context.pages = []
    mock_page = MagicMock()
    mock_context.new_page.return_value = mock_page
    mock_chromium.launch_persistent_context.return_value = mock_context

    with patch("playwright.sync_api.sync_playwright") as mock_sync:
        mock_sync.return_value.start.return_value = mock_playwright

        session.start(headed=False)

        assert mock_chromium.launch_persistent_context.called
        _, kwargs = mock_chromium.launch_persistent_context.call_args

        launch_args = kwargs.get("args") or []
        for arg in launch_args:
            assert "AutomationControlled" not in arg, f"Forbidden arg in launch_persistent_context: {arg}"
            assert "no-sandbox" not in arg, f"Forbidden arg in launch_persistent_context: {arg}"

        # Assert no init scripts were attached modifying navigator.webdriver
        if mock_context.add_init_script.called:
            for call in mock_context.add_init_script.call_args_list:
                script_body = str(call)
                assert "webdriver" not in script_body


# ── 2. SELECTOR SAFETY & ROLE VALIDATION TESTS ──────────────────────────────


def test_demo_mode_startup_refuses_when_todo_calibrate_remains():
    """Startup in demo mode refuses to run while any TODO_CALIBRATE remains, naming each one."""
    page = FakePage()
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    # Ensure config has default uncalibrated TODO_CALIBRATE
    cfg.binomo.ui.amount_input = "TODO_CALIBRATE"
    cfg.binomo.ui.call_button = "TODO_CALIBRATE"
    cfg.binomo.ui.demo_indicator = "TODO_CALIBRATE"

    adapter = BinomoAdapter(config=cfg)

    with pytest.raises(SelectorResolutionError) as exc_info:
        adapter.validate_ui_selectors(page)

    err = str(exc_info.value)
    assert "amount_input" in err
    assert "call_button" in err
    assert "demo_indicator" in err
    assert "TODO_CALIBRATE" in err


def test_demo_mode_startup_rejects_invalid_aria_role_text():
    """Role 'text' is not a valid Playwright ARIA role and must be rejected explicitly, naming selector."""
    page = FakePage()
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.binomo.ui = build_calibrated_ui_config()
    cfg.binomo.option_type = "turbo"
    cfg.binomo.expiry_labels = {1: "1 min"}

    # Inject invalid role "text" into demo_indicator and active_asset_label
    cfg.binomo.ui.demo_indicator = {"role": "text", "name": "Demo account"}
    cfg.binomo.ui.active_asset_label = {"role": "text", "name": "Active Asset"}

    adapter = BinomoAdapter(config=cfg)

    with pytest.raises(SelectorResolutionError) as exc_info:
        adapter.validate_ui_selectors(page)

    err = str(exc_info.value)
    assert "demo_indicator" in err
    assert "active_asset_label" in err
    assert "invalid role 'text'" in err


def test_demo_mode_startup_rejects_unknown_aria_role():
    """Any role not in Playwright's standard ARIA roles is rejected, naming the selector."""
    page = FakePage()
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.binomo.ui = build_calibrated_ui_config()
    cfg.binomo.option_type = "turbo"
    cfg.binomo.expiry_labels = {1: "1 min"}

    cfg.binomo.ui.call_button = {"role": "clickable_element", "name": "Call"}

    adapter = BinomoAdapter(config=cfg)

    with pytest.raises(SelectorResolutionError) as exc_info:
        adapter.validate_ui_selectors(page)

    err = str(exc_info.value)
    assert "call_button" in err
    assert "clickable_element" in err


def test_supports_specs_by_test_id_role_and_css():
    """resolve_locator supports specs defined by test_id, role+name, or css selector."""
    locs = {
        "testid:trade-amt": FakeLocator(name="testid:trade-amt", count=1),
        "role:button:Call": FakeLocator(name="role:button:Call", count=1),
        "role:button": FakeLocator(name="role:button", count=1),
        ".custom-asset-class": FakeLocator(name=".custom-asset-class", count=1),
        "#expiry-dropdown": FakeLocator(name="#expiry-dropdown", count=1),
    }
    page = FakePage(locators=locs)

    # 1. Spec by test_id
    loc_tid = resolve_locator(page, {"test_id": "trade-amt"})
    assert loc_tid.count() == 1
    assert loc_tid.name == "testid:trade-amt"

    # 2. Spec by role+name
    loc_rn = resolve_locator(page, {"role": "button", "name": "Call"})
    assert loc_rn.count() == 1
    assert loc_rn.name == "role:button:Call"

    # 3. Spec by role only
    loc_r = resolve_locator(page, {"role": "button"})
    assert loc_r.count() == 1
    assert loc_r.name == "role:button"

    # 4. Spec by css dict
    loc_css = resolve_locator(page, {"selector": ".custom-asset-class"})
    assert loc_css.count() == 1
    assert loc_css.name == ".custom-asset-class"

    # 5. Spec by css string
    loc_str = resolve_locator(page, "#expiry-dropdown")
    assert loc_str.count() == 1
    assert loc_str.name == "#expiry-dropdown"


def test_playwright_aria_roles_completeness():
    """Verify that common Playwright ARIA roles are recognized and 'text' is excluded."""
    assert "button" in PLAYWRIGHT_ARIA_ROLES
    assert "spinbutton" in PLAYWRIGHT_ARIA_ROLES
    assert "textbox" in PLAYWRIGHT_ARIA_ROLES
    assert "combobox" in PLAYWRIGHT_ARIA_ROLES
    assert "option" in PLAYWRIGHT_ARIA_ROLES
    assert "tab" in PLAYWRIGHT_ARIA_ROLES
    assert "status" in PLAYWRIGHT_ARIA_ROLES
    assert "heading" in PLAYWRIGHT_ARIA_ROLES
    assert "text" not in PLAYWRIGHT_ARIA_ROLES


def test_demo_mode_startup_refuses_when_option_type_uncalibrated_or_expiry_labels_empty():
    """Startup in demo mode refuses when option_type is TODO_CALIBRATE or expiry_labels is empty."""
    page = FakePage()
    cfg = load_config("config.yaml")
    cfg.mode = "demo"
    cfg.binomo.ui = build_calibrated_ui_config()
    cfg.binomo.option_type = "TODO_CALIBRATE"
    cfg.binomo.expiry_labels = {}

    adapter = BinomoAdapter(config=cfg)

    with pytest.raises(SelectorResolutionError) as exc_info:
        adapter.validate_ui_selectors(page)

    err = str(exc_info.value)
    assert "binomo.option_type" in err
    assert "binomo.expiry_labels" in err


# ── 3. CLI COMMAND binomo-validate-ui TESTS ─────────────────────────────────


def test_binomo_validate_ui_with_uncalibrated_config_exits_nonzero(capsys):
    """Command python main.py binomo-validate-ui with uncalibrated config.yaml prints NOT FOUND and exits 1."""
    # Mock session so browser doesn't try to open real network
    with patch("adapters.binomo.session.BinomoSession") as MockSession:
        instance = MockSession.return_value
        instance.start.return_value = FakePage()
        instance.ensure_logged_in.return_value = True
        instance.is_demo.return_value = True

        code = main.main(["binomo-validate-ui", "--config", "config.yaml"])

        out, err = capsys.readouterr()
        assert code == 1, f"Expected non-zero exit code, got {code}"
        assert "NOT FOUND" in out
        assert "TODO_CALIBRATE" in out


def test_binomo_validate_ui_success_prints_resolved_and_active_label_text(capsys):
    """Calibrated UI prints 'resolved (count N)', active labels show visible text, zero clicks/types/fills."""
    locs = {
        "role:spinbutton:Amount": FakeLocator(name="amount_input", count=1, value="10"),
        "role:button:Expiration time": FakeLocator(name="expiry_sel", count=1),
        "role:button:1 min": FakeLocator(name="expiry_opt", count=1, text="1 min"),
        "role:button:Call": FakeLocator(name="call_btn", count=1, text="Call"),
        "role:button:Put": FakeLocator(name="put_btn", count=1, text="Put"),
        "role:button:Asset selection": FakeLocator(name="asset_sel", count=1),
        "role:button:EUR/USD": FakeLocator(name="asset_opt", count=1, text="EUR/USD"),
        ".demo-indicator": FakeLocator(name="demo_indicator", count=1, text="Demo account"),
        ".active-asset": FakeLocator(name="active_asset_label", count=1, text="EUR/USD (OTC)"),
        ".active-expiry": FakeLocator(name="active_expiry_label", count=1, text="1 min"),
    }
    fake_page = FakePage(locators=locs)

    cfg = load_config("config.yaml")
    cfg.binomo.ui = build_calibrated_ui_config()
    cfg.binomo.option_type = "turbo"
    cfg.binomo.expiry_labels = {1: "1 min"}

    with patch("main.load_config", return_value=cfg), \
         patch("adapters.binomo.session.BinomoSession") as MockSession:

        instance = MockSession.return_value
        instance.start.return_value = fake_page
        instance.ensure_logged_in.return_value = True
        instance.is_demo.return_value = True

        code = main.main(["binomo-validate-ui", "--config", "config.yaml"])

        out, err = capsys.readouterr()
        assert code == 0, f"Expected exit code 0, got {code}"
        assert "amount_input: resolved (count 1)" in out
        assert "call_button: resolved (count 1)" in out
        assert "put_button: resolved (count 1)" in out
        assert "demo_indicator: resolved (count 1)" in out
        assert "visible text: 'Demo account'" in out
        assert "active_asset_label: resolved (count 1)" in out
        assert "visible text: 'EUR/USD (OTC)'" in out
        assert "active_expiry_label: resolved (count 1)" in out
        assert "visible text: '1 min'" in out

        # ZERO clicking, typing or filling guaranteed:
        for loc in locs.values():
            assert loc.click_count == 0, f"Click called on {loc.name}"
            assert len(loc.fill_calls) == 0, f"Fill called on {loc.name}"


def test_binomo_validate_ui_refuses_live_account(capsys):
    """Command python main.py binomo-validate-ui refuses and exits 1 if live account detected."""
    with patch("adapters.binomo.session.BinomoSession") as MockSession:
        instance = MockSession.return_value
        instance.start.return_value = FakePage()
        instance.ensure_logged_in.return_value = True
        instance.is_demo.return_value = False  # Real account!

        code = main.main(["binomo-validate-ui", "--config", "config.yaml"])

        out, err = capsys.readouterr()
        assert code == 1
        assert "Demo account required" in err
