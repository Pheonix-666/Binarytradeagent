"""Binomo platform adapter for demo trading execution and observation.

Adheres strictly to PROJECT_SPEC.md:
- Built on BinomoSession, BinomoWSListener, and BinomoReadOnlySource.
- Driven ONLY by selectors from config.binomo.ui (role + name, or test id).
- Zero hardcoded selectors or UI text in code.
- Demo mode only: any real account marker immediately aborts before any click.
- One order in flight via thread-safe non-blocking lock.
- Post-click confirmation via Phoenix 'opened' event (no retry loops).
- Settlement processing from Phoenix 'closed' events fed to TradeManager.
- Click interaction allow-listed ONLY within place_trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Mapping, Optional, Union

import pandas as pd

from adapters.base import PlatformAdapter
from adapters.binomo.reader import BinomoReadOnlySource
from adapters.binomo.session import BinomoSession
from adapters.binomo.ws_listener import BinomoWSListener
from alerts.telegram import TelegramAlerter
from core.config import AppConfig
from core.models import AssetQuote, Trade, TradeDirection, TradeReceipt, TradeResult

logger = logging.getLogger("trading_system.adapters.binomo.trader")


# ── Custom Exceptions ────────────────────────────────────────────────────────


class RealAccountDetected(Exception):
    """Raised when any marker (DOM, websocket, or config) indicates a real account."""


class ExpiryUnavailable(Exception):
    """Raised when an expiry is not available in config or the UI."""


class StakeOutOfRange(Exception):
    """Raised when a stake amount is outside platform currency min/max limits or mismatches readback."""


class StakeLimitsUnavailable(Exception):
    """Raised when platform stake limits cannot be parsed from /platform/private/v2/currencies."""


class PanelBlocked(Exception):
    """Raised when a modal or popup blocks the trading panel."""


class AmbiguousOrder(Exception):
    """Raised when an order was placed but confirmation was not received within the timeout."""


class SelectorResolutionError(Exception):
    """Raised when one or more required UI selectors cannot be resolved at startup."""


class AssetSelectionMismatch(Exception):
    """Raised when active asset label does not match requested asset after selection."""


class ButtonLabelMismatch(Exception):
    """Raised when the action button's accessible name does not match the trade direction."""


# ── Helper Functions ─────────────────────────────────────────────────────────


def parse_deal_timestamp(val: Any) -> Optional[datetime]:
    """Parse deal timestamp (ISO string, unix epoch in sec or ms) to UTC datetime."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e11:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(val, str):
        val_clean = val.strip()
        try:
            num = float(val_clean)
            if num > 1e11:
                num = num / 1000.0
            return datetime.fromtimestamp(num, tz=timezone.utc)
        except ValueError:
            pass
        if val_clean.endswith("Z"):
            val_clean = val_clean[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(val_clean)
        except Exception:
            return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    return None


# ── Selector Resolution & Validation Helpers ─────────────────────────────────


# ── Valid Playwright ARIA Roles ──────────────────────────────────────────────

PLAYWRIGHT_ARIA_ROLES: set[str] = {
    "alert", "alertdialog", "application", "article", "banner", "blockquote", "button",
    "caption", "cell", "checkbox", "code", "columnheader", "combobox", "complementary",
    "contentinfo", "definition", "deletion", "dialog", "directory", "document", "emphasis",
    "feed", "figure", "form", "generic", "grid", "gridcell", "group", "heading", "img",
    "insertion", "link", "list", "listbox", "listitem", "log", "main", "marquee", "math",
    "meter", "menu", "menubar", "menuitem", "menuitemcheckbox", "menuitemradio", "navigation",
    "none", "note", "option", "paragraph", "presentation", "progressbar", "radio",
    "radiogroup", "region", "row", "rowgroup", "rowheader", "scrollbar", "search",
    "searchbox", "separator", "slider", "spinbutton", "status", "strong", "subscript",
    "superscript", "switch", "tab", "table", "tablist", "tabpanel", "term", "textbox",
    "time", "timer", "toolbar", "tooltip", "tree", "treegrid", "treeitem",
}


def check_selector_spec(key: str, spec: Any) -> Optional[str]:
    """Check if a selector spec is uncalibrated (TODO_CALIBRATE) or has an invalid role.

    Returns an error message if invalid, or None if valid.
    """
    if spec is None:
        return f"{key}: missing selector spec"
    if spec == "TODO_CALIBRATE":
        return f"{key}: placeholder 'TODO_CALIBRATE'"
    if isinstance(spec, str):
        if "TODO_CALIBRATE" in spec:
            return f"{key}: placeholder 'TODO_CALIBRATE'"
        if not spec.strip():
            return f"{key}: empty selector spec"
        return None

    spec_dict = spec.model_dump() if hasattr(spec, "model_dump") else spec
    if isinstance(spec_dict, dict):
        for k, v in spec_dict.items():
            if v == "TODO_CALIBRATE" or (isinstance(v, str) and "TODO_CALIBRATE" in v):
                return f"{key}: placeholder 'TODO_CALIBRATE'"

        if "role" in spec_dict and spec_dict["role"]:
            role = str(spec_dict["role"]).strip().lower()
            if role == "text":
                return f"{key}: invalid role 'text' (Playwright does not support role 'text'; use a css selector, test_id, or valid ARIA role)"
            if role not in PLAYWRIGHT_ARIA_ROLES:
                return f"{key}: invalid role '{role}' (not a valid Playwright ARIA role)"
            return None

        if "test_id" in spec_dict and spec_dict["test_id"]:
            return None

        if ("selector" in spec_dict and spec_dict["selector"]) or ("css" in spec_dict and spec_dict["css"]):
            return None

        return f"{key}: unrecognized spec format (must specify 'test_id', 'role', or 'selector'/'css')"

    return f"{key}: unrecognized spec type {type(spec).__name__}"


def resolve_locator(page: Any, spec: Any, **kwargs: Any) -> Any:
    """Resolve a selector specification into a Playwright locator.

    Supported spec formats:
      - dict or BaseModel with 'test_id': page.get_by_test_id(test_id)
      - dict or BaseModel with 'role' (and optional 'name'): page.get_by_role(role, name=name)
      - dict or BaseModel with 'selector' or 'css': page.locator(selector)
      - str: page.locator(spec)
    Templates ({label}, {name}, etc.) are formatted using kwargs.
    """
    if spec is None:
        raise ValueError("Cannot resolve locator for None selector spec")

    err = check_selector_spec("selector", spec)
    if err:
        raise SelectorResolutionError(f"Cannot resolve locator: {err}")

    spec_dict: dict[str, Any] = {}
    if hasattr(spec, "model_dump"):
        spec_dict = spec.model_dump()
    elif isinstance(spec, dict):
        spec_dict = spec
    elif isinstance(spec, str):
        formatted_str = spec.format(**kwargs) if kwargs else spec
        return page.locator(formatted_str)

    if "test_id" in spec_dict and spec_dict["test_id"]:
        tid = str(spec_dict["test_id"])
        if kwargs:
            tid = tid.format(**kwargs)
        if hasattr(page, "get_by_test_id"):
            return page.get_by_test_id(tid)
        return page.locator(f"[data-testid='{tid}']")

    if "role" in spec_dict and spec_dict["role"]:
        role = str(spec_dict["role"]).strip().lower()
        name = spec_dict.get("name")
        if name:
            name_str = str(name)
            if kwargs:
                name_str = name_str.format(**kwargs)
            if hasattr(page, "get_by_role"):
                return page.get_by_role(role, name=name_str)
            return page.locator(f"[role='{role}'][name='{name_str}']")
        if hasattr(page, "get_by_role"):
            return page.get_by_role(role)
        return page.locator(f"[role='{role}']")

    sel = spec_dict.get("selector") or spec_dict.get("css")
    if sel:
        sel_str = str(sel)
        if kwargs:
            sel_str = sel_str.format(**kwargs)
        return page.locator(sel_str)

    raise ValueError(f"Unrecognized selector spec format: {spec}")


def validate_ui_selectors(page: Any, ui_config: Any) -> list[str]:
    """Validate that every required selector in binomo.ui is calibrated and resolves on page.

    Refuses to run if any TODO_CALIBRATE placeholder or invalid role remains, naming each one.
    Returns a list of failed selector descriptions (empty if all resolved).
    """
    failed: list[str] = []

    required_static = [
        "amount_input",
        "expiry_selector",
        "call_button",
        "put_button",
        "asset_selector",
        "demo_indicator",
        "active_asset_label",
        "active_expiry_label",
    ]

    for key in required_static:
        spec = getattr(ui_config, key, None)
        if spec is None and isinstance(ui_config, dict):
            spec = ui_config.get(key)

        err = check_selector_spec(key, spec)
        if err:
            failed.append(err)
            continue

        try:
            loc = resolve_locator(page, spec)
            count = loc.count() if hasattr(loc, "count") else 1
            if count == 0:
                failed.append(f"{key}: not found on page (count 0)")
        except Exception as exc:
            logger.debug("Selector '%s' failed resolution check: %s", key, exc)
            failed.append(f"{key}: resolution error ({exc})")

    # Template selectors: verify formatting and resolution syntax using config samples
    val_samples = getattr(ui_config, "validation_samples", {}) or {}
    if isinstance(ui_config, dict) and not val_samples:
        val_samples = ui_config.get("validation_samples", {}) or {}

    sample_exp = val_samples.get("expiry_option") or val_samples.get("expiry_label")
    sample_asset = val_samples.get("asset_option") or val_samples.get("asset_name")

    if not sample_exp or sample_exp == "TODO_CALIBRATE":
        failed.append("validation_samples.expiry_option: missing or TODO_CALIBRATE")
    if not sample_asset or sample_asset == "TODO_CALIBRATE":
        failed.append("validation_samples.asset_option: missing or TODO_CALIBRATE")

    templates: list[tuple[str, dict[str, str]]] = []
    if sample_exp and sample_exp != "TODO_CALIBRATE":
        templates.append(("expiry_option", {"label": sample_exp}))
    if sample_asset and sample_asset != "TODO_CALIBRATE":
        templates.append(("asset_option", {"name": sample_asset}))

    for key, sample_kwargs in templates:
        spec = getattr(ui_config, key, None)
        if spec is None and isinstance(ui_config, dict):
            spec = ui_config.get(key)

        err = check_selector_spec(key, spec)
        if err:
            failed.append(err)
            continue
        try:
            resolve_locator(page, spec, **sample_kwargs)
        except Exception as exc:
            logger.debug("Template selector '%s' failed validation: %s", key, exc)
            failed.append(f"{key}: template resolution error ({exc})")

    # Validate button label lists
    call_labels = getattr(ui_config, "call_button_labels", None)
    if call_labels is None and isinstance(ui_config, dict):
        call_labels = ui_config.get("call_button_labels")
    if not call_labels:
        failed.append("call_button_labels: empty list (refuse to trade)")
    elif any(lbl == "TODO_CALIBRATE" for lbl in call_labels):
        failed.append("call_button_labels: contains placeholder 'TODO_CALIBRATE'")

    put_labels = getattr(ui_config, "put_button_labels", None)
    if put_labels is None and isinstance(ui_config, dict):
        put_labels = ui_config.get("put_button_labels")
    if not put_labels:
        failed.append("put_button_labels: empty list (refuse to trade)")
    elif any(lbl == "TODO_CALIBRATE" for lbl in put_labels):
        failed.append("put_button_labels: contains placeholder 'TODO_CALIBRATE'")

    # Validate blocking overlays list
    overlays = getattr(ui_config, "blocking_overlays", None)
    if overlays is None and isinstance(ui_config, dict):
        overlays = ui_config.get("blocking_overlays")
    if not overlays:
        logger.warning("Startup warning: binomo.ui.blocking_overlays is empty; overlay blocking checks will be skipped.")

    return failed


def is_panel_blocked(page: Any, blocking_overlays: Optional[list[str]] = None) -> bool:
    """Check if an overlay, modal, or dialog blocks the trading panel."""
    if page is None or not blocking_overlays:
        return False

    for sel in blocking_overlays:
        try:
            loc = page.locator(sel)
            if hasattr(loc, "count") and loc.count() > 0:
                first = loc.first if hasattr(loc, "first") else loc
                if hasattr(first, "is_visible") and first.is_visible():
                    return True
        except Exception:
            pass

    return False


# ── Binomo Platform Adapter ──────────────────────────────────────────────────


class BinomoAdapter(PlatformAdapter):
    """Full PlatformAdapter implementation for Binomo Demo trading."""

    def __init__(
        self,
        session: Optional[BinomoSession] = None,
        ws_listener: Optional[BinomoWSListener] = None,
        reader: Optional[BinomoReadOnlySource] = None,
        config: Optional[AppConfig] = None,
        breaker: Optional[Any] = None,
        alerter: Optional[TelegramAlerter] = None,
    ) -> None:
        self.session = session
        self.config = config
        self.breaker = breaker
        self.alerter = alerter

        # Concurrency control: strictly 1 order in flight
        self._trade_lock = threading.Lock()

        # Settled deals tracking to avoid duplicate settlement calls
        self._settled_deal_uuids: set[str] = set()
        self._logged_unknown_deal_uuids: set[str] = set()

        # Cached platform currency stake limits (no default limits)
        self._platform_min_stake: Optional[float] = None
        self._platform_max_stake: Optional[float] = None
        self._platform_limits_loaded: bool = False

        # Load configurations
        self.confirm_timeout_seconds: float = 15.0
        self.max_frame_age_seconds: float = 60.0
        self.limits_product: str = "binary_option"
        self.amount_scale: float = 1.0
        self.expiry_labels: dict[int, str] = {}
        self._ric_to_name_map: dict[str, str] = {}

        if config and hasattr(config, "binomo"):
            b_cfg = config.binomo
            self.confirm_timeout_seconds = float(getattr(b_cfg, "confirm_timeout_seconds", 15.0))
            self.max_frame_age_seconds = float(getattr(b_cfg, "max_frame_age_seconds", 60.0))
            self.limits_product = str(getattr(b_cfg, "limits_product", "binary_option"))
            self.amount_scale = float(getattr(b_cfg, "amount_scale", 1.0))
            if getattr(b_cfg, "expiry_labels", None):
                self.expiry_labels = {
                    int(k): str(v) for k, v in b_cfg.expiry_labels.items()
                }

        self.ws_listener = ws_listener or (session.ws_listener if session else BinomoWSListener(amount_scale=self.amount_scale))
        if hasattr(self.ws_listener, "amount_scale"):
            self.ws_listener.amount_scale = self.amount_scale

        self.reader = reader or BinomoReadOnlySource(session=session, config=config, alerter=alerter)

    @property
    def ui_config(self) -> Any:
        if self.config and hasattr(self.config, "binomo"):
            return getattr(self.config.binomo, "ui", None)
        return None

    # ── Asset Name Resolution ────────────────────────────────────────────────

    def get_ric_to_name_map(self) -> dict[str, str]:
        """Build or return ric-to-display-name map from the reader's asset table."""
        if not self._ric_to_name_map:
            if self.reader and hasattr(self.reader, "get_ric_to_name_map"):
                try:
                    res = self.reader.get_ric_to_name_map()
                    if isinstance(res, dict):
                        self._ric_to_name_map = res
                except Exception:
                    pass
            if not self._ric_to_name_map and self.reader and hasattr(self.reader, "get_assets"):
                try:
                    assets = self.reader.get_assets()
                    if isinstance(assets, list):
                        self._ric_to_name_map = {a.asset_id: a.name for a in assets if hasattr(a, "asset_id") and hasattr(a, "name")}
                except Exception:
                    pass
        return dict(self._ric_to_name_map)

    def get_display_name(self, ric: str) -> Optional[str]:
        """Return display name for given RIC; None if unmapped."""
        mapping = self.get_ric_to_name_map()
        return mapping.get(ric)

    # ── Platform Currency Limits ─────────────────────────────────────────────

    def load_platform_limits(self) -> tuple[float, float]:
        """Fetch and cache min/max stake limits from /platform/private/v2/currencies.

        Parses data.list[] with iso equal to data.current, taking limits.<product>.min/max;
        product from config binomo.limits_product (binary_option or standard_trade,
        default binary_option; logs both products' limits once).
        If limits cannot be parsed, raises StakeLimitsUnavailable and refuses to trade
        (no default limits).
        """
        try:
            url = f"{self.reader.base_api_url}/platform/private/v2/currencies"
            data = self.reader._execute_get(url)
        except Exception as exc:
            raise StakeLimitsUnavailable(f"Failed to fetch /platform/private/v2/currencies: {exc}") from exc

        data_obj: dict[str, Any] = {}
        if isinstance(data, dict):
            if isinstance(data.get("data"), dict):
                data_obj = data["data"]
            else:
                data_obj = data
        else:
            raise StakeLimitsUnavailable(f"Unexpected currencies response type: {type(data)}")

        current_iso = data_obj.get("current")
        currencies_list = data_obj.get("list")

        if not current_iso or not isinstance(currencies_list, list):
            raise StakeLimitsUnavailable(
                f"Cannot parse currencies: current={current_iso}, list_type={type(currencies_list)}"
            )

        matched_currency: Optional[dict[str, Any]] = None
        for item in currencies_list:
            if isinstance(item, dict) and item.get("iso") == current_iso:
                matched_currency = item
                break

        if not matched_currency:
            raise StakeLimitsUnavailable(
                f"Currency '{current_iso}' not found in currencies list"
            )

        limits_dict = matched_currency.get("limits")
        if not isinstance(limits_dict, dict):
            raise StakeLimitsUnavailable(
                f"No 'limits' found for currency '{current_iso}'"
            )

        # Log both products' limits once so operator can choose
        bo_limits = limits_dict.get("binary_option") or {}
        st_limits = limits_dict.get("standard_trade") or {}
        logger.info(
            "Platform stake limits for %s: binary_option=[%s, %s], standard_trade=[%s, %s]",
            current_iso,
            bo_limits.get("min"),
            bo_limits.get("max"),
            st_limits.get("min"),
            st_limits.get("max"),
        )

        chosen_product = self.limits_product or "binary_option"
        product_limits = limits_dict.get(chosen_product)
        if not isinstance(product_limits, dict) or "min" not in product_limits or "max" not in product_limits:
            raise StakeLimitsUnavailable(
                f"Stake limits unavailable for product '{chosen_product}' in currency '{current_iso}'"
            )

        try:
            self._platform_min_stake = float(product_limits["min"])
            self._platform_max_stake = float(product_limits["max"])
            self._platform_limits_loaded = True
            logger.info("Loaded platform stake limits for '%s': [%.2f, %.2f]",
                        chosen_product, self._platform_min_stake, self._platform_max_stake)
        except (ValueError, TypeError) as exc:
            raise StakeLimitsUnavailable(
                f"Failed parsing float limits for product '{chosen_product}': {product_limits}"
            ) from exc

        return self._platform_min_stake, self._platform_max_stake

    def validate_ui_selectors(self, page: Optional[Any] = None) -> None:
        """Validate all required UI selectors at startup; refuse to trade on failure."""
        target_page = page or (self.session.page if self.session else None)
        if target_page is None:
            raise SelectorResolutionError("Cannot validate UI selectors: No active Playwright page available")

        ui_cfg = self.ui_config
        if not ui_cfg:
            raise SelectorResolutionError("Missing binomo.ui configuration block")

        failed = validate_ui_selectors(target_page, ui_cfg)

        if self.config and hasattr(self.config, "binomo"):
            b_cfg = self.config.binomo
            if getattr(b_cfg, "option_type", None) == "TODO_CALIBRATE":
                failed.append("binomo.option_type: placeholder 'TODO_CALIBRATE'")
            if not getattr(b_cfg, "expiry_labels", None):
                failed.append("binomo.expiry_labels: empty (refuse to trade)")

        if failed:
            msg = f"Startup refused: binomo.ui selectors uncalibrated or failed to resolve on page: {failed}"
            logger.critical(msg)
            if self.alerter:
                try:
                    self.alerter.send_alert(f"🚨 {msg}")
                except Exception:
                    pass
            raise SelectorResolutionError(msg)

        logger.info("All binomo.ui selectors successfully resolved and validated on page.")

    # ── PlatformAdapter Core Interface ───────────────────────────────────────

    def get_assets(self) -> list[AssetQuote]:
        """Return available assets filtered for demo account, open status, and option_type payout."""
        return self.reader.get_assets()

    def get_candles(self, asset_id: str, timeframe: str, n: int) -> pd.DataFrame:
        """Return latest n closed candles for asset and timeframe (zero look-ahead)."""
        return self.reader.get_candles(asset_id, timeframe, n)

    def get_balance(self) -> float:
        """Return the current account balance from WebSocket state with staleness enforcement."""
        self._enforce_frame_freshness()
        bal = self.ws_listener.get_balance()
        if bal is None:
            raise RuntimeError("Account balance has not been received from WebSocket yet")
        return float(bal)

    def get_open_trades(self) -> list[Trade]:
        """Return all open/active trades from WebSocket state with staleness enforcement."""
        self._enforce_frame_freshness()
        open_deals = self.ws_listener.get_open_deals()
        now_dt = datetime.now(timezone.utc)

        trades: list[Trade] = []
        for d in open_deals:
            uuid_str = str(d.get("uuid") or "")
            if not uuid_str:
                continue

            trend_str = str(d.get("trend") or "").upper()
            direction = TradeDirection.CALL if trend_str in ("CALL", "UP") else TradeDirection.PUT

            p_rate = d.get("payment_rate") or 0.0
            payout_dec = (p_rate / 100.0) if p_rate > 1.0 else p_rate

            start_dt = parse_deal_timestamp(d.get("created_at") or d.get("open_quote_created_at") or d.get("requested_at"))
            end_dt = parse_deal_timestamp(d.get("close_quote_created_at") or d.get("finished_at"))
            entry_time = start_dt or now_dt

            if start_dt and end_dt:
                diff_sec = (end_dt - start_dt).total_seconds()
                derived_expiry = max(1, int(round(diff_sec / 60.0)))
            else:
                derived_expiry = int(self.config.trading.expiry_minutes if (self.config and hasattr(self.config, "trading")) else 1)

            trades.append(
                Trade(
                    trade_id=uuid_str,
                    asset_id=str(d.get("asset_ric") or ""),
                    direction=direction,
                    expiry_minutes=derived_expiry,
                    stake_amount=float(d.get("amount") or 0.0),
                    entry_price=float(d.get("open_rate") or 0.0),
                    entry_time=entry_time,
                    payout=float(payout_dec),
                    platform_trade_id=uuid_str,
                    status="OPEN",
                    result=TradeResult.PENDING,
                )
            )

        return trades

    def get_trade_history(self, since: Optional[datetime] = None) -> list[Trade]:
        """Return completed/settled trades via GET /bo-deals-history/v3/deals/trade (fallback: websocket)."""
        deals: list[dict[str, Any]] = []

        try:
            url = f"{self.reader.base_api_url}/bo-deals-history/v3/deals/trade"
            res = self.reader._execute_get(url)
            if isinstance(res, list):
                deals = res
            elif isinstance(res, dict):
                if isinstance(res.get("data"), list):
                    deals = res["data"]
                elif isinstance(res.get("deals"), list):
                    deals = res["deals"]
        except Exception as exc:
            logger.warning("GET /bo-deals-history/v3/deals/trade failed: %s. Using websocket fallback.", exc)
            deals = self.ws_listener.get_closed_deals()

        trades: list[Trade] = []
        now_dt = datetime.now(timezone.utc)

        for d in deals:
            if not isinstance(d, dict):
                continue

            uuid_str = str(d.get("uuid") or d.get("id") or "")
            if not uuid_str:
                continue

            # Map status
            status_raw = str(d.get("status") or "").strip().lower()
            if status_raw in ("won", "win"):
                res_enum = TradeResult.WIN
            elif status_raw in ("lost", "loss"):
                res_enum = TradeResult.LOSS
            elif status_raw in ("tie", "draw"):
                res_enum = TradeResult.TIE
            else:
                continue

            trend_str = str(d.get("trend") or d.get("direction") or "").upper()
            direction = TradeDirection.CALL if trend_str in ("CALL", "UP") else TradeDirection.PUT

            p_rate = d.get("payment_rate") or 0.0
            payout_dec = (p_rate / 100.0) if p_rate > 1.0 else p_rate

            start_dt = parse_deal_timestamp(d.get("created_at") or d.get("open_quote_created_at") or d.get("requested_at"))
            end_dt = parse_deal_timestamp(d.get("close_quote_created_at") or d.get("finished_at"))
            entry_time = start_dt or now_dt

            if start_dt and end_dt:
                diff_sec = (end_dt - start_dt).total_seconds()
                derived_expiry = max(1, int(round(diff_sec / 60.0)))
            else:
                derived_expiry = int(self.config.trading.expiry_minutes if (self.config and hasattr(self.config, "trading")) else 1)

            raw_amt = float(d.get("amount") or d.get("bet") or 0.0)
            scaled_amt = raw_amt * self.amount_scale if "raw_amount" not in d else raw_amt

            trades.append(
                Trade(
                    trade_id=uuid_str,
                    asset_id=str(d.get("asset_ric") or d.get("ric") or ""),
                    direction=direction,
                    expiry_minutes=derived_expiry,
                    stake_amount=scaled_amt,
                    entry_price=float(d.get("open_rate") or d.get("rate") or 0.0),
                    entry_time=entry_time,
                    payout=float(payout_dec),
                    platform_trade_id=uuid_str,
                    status="CLOSED",
                    result=res_enum,
                    exit_price=float(d["end_rate"]) if d.get("end_rate") is not None else None,
                )
            )

        return trades

    def is_healthy(self) -> bool:
        """Check connection health: session logged in, demo mode, fresh frames, reader active."""
        if not self.session or not self.session.is_logged_in():
            return False
        if not self.session.is_demo():
            return False

        last_frame = self.ws_listener.get_last_frame_time()
        if not last_frame:
            return False
        if (datetime.now(timezone.utc) - last_frame).total_seconds() > self.max_frame_age_seconds:
            return False

        if self.reader and self.reader._stopped:
            return False

        return True

    def _enforce_frame_freshness(self) -> None:
        """Verify that websocket frame stream is not stale; raise RuntimeError if stale."""
        last_frame = self.ws_listener.get_last_frame_time()
        if not last_frame:
            raise RuntimeError("No WebSocket frames received from Binomo yet")

        age = (datetime.now(timezone.utc) - last_frame).total_seconds()
        if age > self.max_frame_age_seconds:
            raise RuntimeError(f"Binomo WebSocket connection is stale: last frame received {age:.1f}s ago (limit: {self.max_frame_age_seconds}s)")

    # ── Settlement Processing ────────────────────────────────────────────────

    def process_settlements(self, trade_manager: Any) -> list[dict[str, Any]]:
        """Feed closed deal outcomes to TradeManager.record_settlement.

        Maps platform deal uuid to internal trade id via open trade's platform_trade_id
        in trade_manager.state_manager.data["open_trades"].
        Calls record_settlement with the INTERNAL id and the payout from the deal (never default).
        Unknown uuids are logged once and skipped.
        Prevents double settlement by tracking settled uuids.
        """
        closed_deals = self.ws_listener.get_closed_deals()
        settled_results: list[dict[str, Any]] = []

        platform_to_internal: dict[str, str] = {}
        if hasattr(trade_manager, "state_manager") and hasattr(trade_manager.state_manager, "data"):
            open_list = trade_manager.state_manager.data.get("open_trades", [])
            for ot in open_list:
                int_id = ot.get("id") if isinstance(ot, dict) else getattr(ot, "id", None)
                plat_id = ot.get("platform_trade_id") if isinstance(ot, dict) else getattr(ot, "platform_trade_id", None)
                if int_id:
                    if plat_id:
                        platform_to_internal[str(plat_id)] = str(int_id)
                    platform_to_internal[str(int_id)] = str(int_id)

        for deal in closed_deals:
            uuid_str = str(deal.get("uuid") or "")
            if not uuid_str:
                continue

            if uuid_str in self._settled_deal_uuids:
                continue

            if uuid_str not in platform_to_internal:
                if uuid_str not in self._logged_unknown_deal_uuids:
                    logger.warning("Closed deal uuid '%s' not found in open trades; skipping settlement.", uuid_str)
                    self._logged_unknown_deal_uuids.add(uuid_str)
                continue

            internal_trade_id = platform_to_internal[uuid_str]

            st_raw = str(deal.get("status") or "").strip().lower()
            if st_raw in ("won", "win"):
                outcome = "WIN"
            elif st_raw in ("lost", "loss"):
                outcome = "LOSS"
            elif st_raw in ("tie", "draw"):
                outcome = "TIE"
            else:
                msg = f"Unknown deal settlement status '{st_raw}' received for deal {uuid_str}"
                logger.error(msg)
                raise ValueError(msg)

            p_rate = deal.get("payment_rate")
            if p_rate is None:
                p_rate = deal.get("payout")
            if p_rate is None:
                msg = f"Deal {uuid_str} has no payment_rate or payout (never default)"
                logger.error(msg)
                raise ValueError(msg)

            payout_val = (float(p_rate) / 100.0) if float(p_rate) > 1.0 else float(p_rate)

            try:
                res = trade_manager.record_settlement(
                    trade_id=internal_trade_id,
                    outcome=outcome,
                    payout=payout_val,
                )
                self._settled_deal_uuids.add(uuid_str)
                settled_results.append(res)
                logger.info("Settlement recorded for deal %s (internal %s): outcome=%s, payout=%s",
                            uuid_str, internal_trade_id, outcome, payout_val)
            except Exception as exc:
                logger.warning("Failed recording settlement for deal %s: %s", uuid_str, exc)

        return settled_results

    # ── place_trade Execution Path (.click allow-listed only here) ────────────

    def place_trade(
        self,
        asset_id: str,
        direction: Union[TradeDirection, str],
        expiry_min: int,
        stake_amount: float,
        payout: Optional[float] = None,
        trade_id: Optional[str] = None,
    ) -> TradeReceipt:
        """Place a demo trade adhering strictly to Phase 10B/10C rules.

        Aborts BEFORE any click on the first failure.
        Allows click interactions exclusively within this method.
        """
        # Concurrency Lock: Strictly one order in flight
        if not self._trade_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent trade placement rejected: Another order is currently in flight")

        try:
            page = self.session.page if self.session else None
            if page is None:
                raise RuntimeError("No active Playwright page available for trade placement")

            dir_enum = TradeDirection(direction) if isinstance(direction, str) else direction
            ui_cfg = self.ui_config
            if not ui_cfg:
                raise SelectorResolutionError("Missing binomo.ui configuration for trade execution")

            # ─────────────────────────────────────────────────────────────────
            # STEP 3a: Strict Demo & Login Agreement (Abort Before Any Click)
            # ─────────────────────────────────────────────────────────────────
            config_mode = str(getattr(self.config, "mode", "")).strip().lower()
            if config_mode != "demo":
                raise RealAccountDetected(f"Safety violation: Config mode is '{config_mode}', not 'demo'")

            if not self.session.is_logged_in():
                raise RuntimeError("Cannot place trade: Binomo session is not logged in")

            ws_account_type = str(self.ws_listener.get_account_type() or "").strip().lower()
            if ws_account_type != "demo":
                raise RealAccountDetected(f"Safety violation: WebSocket account_type is '{ws_account_type}', not 'demo'")

            demo_loc = resolve_locator(page, getattr(ui_cfg, "demo_indicator"))
            demo_text = ""
            if hasattr(demo_loc, "inner_text"):
                demo_text = str(demo_loc.inner_text() or "")
            elif hasattr(demo_loc, "text_content"):
                demo_text = str(demo_loc.text_content() or "")
            demo_lower = demo_text.lower()

            if "real" in demo_lower and "demo" not in demo_lower:
                raise RealAccountDetected(f"Safety violation: DOM demo_indicator shows real account: '{demo_text}'")
            if "demo" not in demo_lower:
                raise RealAccountDetected(f"Safety violation: DOM demo_indicator does not confirm demo: '{demo_text}'")

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Modal / Popup Check (Never Auto-Dismiss)
            # ─────────────────────────────────────────────────────────────────
            blocking_overlays = getattr(ui_cfg, "blocking_overlays", None)
            if blocking_overlays is None and isinstance(ui_cfg, dict):
                blocking_overlays = ui_cfg.get("blocking_overlays")
            if is_panel_blocked(page, blocking_overlays):
                msg = "Trading panel is blocked by a modal dialog or popup! Aborting before click."
                logger.critical(msg)
                if self.alerter:
                    try:
                        self.alerter.send_alert(f"🚨 {msg}")
                    except Exception:
                        pass
                raise PanelBlocked(msg)

            # ─────────────────────────────────────────────────────────────────
            # STEP 3c (Validation): Expiry must exist in binomo.expiry_labels
            # ─────────────────────────────────────────────────────────────────
            if not self.expiry_labels:
                raise ExpiryUnavailable("Refusing to trade: binomo.expiry_labels is empty")
            if expiry_min not in self.expiry_labels:
                raise ExpiryUnavailable(f"Expiry {expiry_min}m not available in binomo.expiry_labels {sorted(self.expiry_labels.keys())}")
            expected_expiry_label = self.expiry_labels[expiry_min]

            # ─────────────────────────────────────────────────────────────────
            # STEP 3d (Validation): Platform Currency Min/Max Limits
            # ─────────────────────────────────────────────────────────────────
            if not self._platform_limits_loaded or self._platform_min_stake is None or self._platform_max_stake is None:
                self.load_platform_limits()

            if stake_amount < self._platform_min_stake or stake_amount > self._platform_max_stake:
                raise StakeOutOfRange(
                    f"Stake amount {stake_amount} is outside platform limits [{self._platform_min_stake}, {self._platform_max_stake}]"
                )

            # ─────────────────────────────────────────────────────────────────
            # STEP 3b (Validation): Asset Display Name Lookup (Abort before click)
            # ─────────────────────────────────────────────────────────────────
            display_name = self.get_display_name(asset_id)
            if not display_name:
                raise AssetSelectionMismatch(f"No display name found in asset table for RIC '{asset_id}'. Aborting before click.")

            # ─────────────────────────────────────────────────────────────────
            # STEP 3e (Validation): Button Labels Configured (Abort before click)
            # ─────────────────────────────────────────────────────────────────
            if dir_enum == TradeDirection.CALL:
                btn_loc = resolve_locator(page, getattr(ui_cfg, "call_button"))
                button_labels = getattr(ui_cfg, "call_button_labels", None)
                if button_labels is None and isinstance(ui_cfg, dict):
                    button_labels = ui_cfg.get("call_button_labels")
                if not button_labels:
                    raise ButtonLabelMismatch("Refusing to trade: binomo.ui.call_button_labels is empty")
            else:
                btn_loc = resolve_locator(page, getattr(ui_cfg, "put_button"))
                button_labels = getattr(ui_cfg, "put_button_labels", None)
                if button_labels is None and isinstance(ui_cfg, dict):
                    button_labels = ui_cfg.get("put_button_labels")
                if not button_labels:
                    raise ButtonLabelMismatch("Refusing to trade: binomo.ui.put_button_labels is empty")

            # ─────────────────────────────────────────────────────────────────
            # UI SELECTION & EXECUTION (Allow-listed clicks exclusively here)
            # ─────────────────────────────────────────────────────────────────
            # 1. Select Asset through UI and verify active_asset_label
            asset_sel_loc = resolve_locator(page, getattr(ui_cfg, "asset_selector"))
            asset_sel_loc.click()

            asset_opt_loc = resolve_locator(page, getattr(ui_cfg, "asset_option"), name=display_name)
            asset_opt_loc.click()

            active_asset_loc = resolve_locator(page, getattr(ui_cfg, "active_asset_label"))
            active_asset_text = (active_asset_loc.inner_text() or active_asset_loc.text_content() or "").strip()
            if display_name.lower() not in active_asset_text.lower():
                raise AssetSelectionMismatch(
                    f"Active asset label '{active_asset_text}' does not match requested asset '{asset_id}' (display name '{display_name}')"
                )

            # 2. Select Expiry through UI and verify active_expiry_label
            expiry_sel_loc = resolve_locator(page, getattr(ui_cfg, "expiry_selector"))
            expiry_sel_loc.click()

            expiry_opt_loc = resolve_locator(page, getattr(ui_cfg, "expiry_option"), label=expected_expiry_label)
            expiry_opt_loc.click()

            active_exp_loc = resolve_locator(page, getattr(ui_cfg, "active_expiry_label"))
            active_exp_text = (active_exp_loc.inner_text() or active_exp_loc.text_content() or "").strip()
            if expected_expiry_label.lower() not in active_exp_text.lower():
                raise ExpiryUnavailable(
                    f"Active expiry label '{active_exp_text}' does not match expected '{expected_expiry_label}'"
                )

            # 3. Enter Stake, Read Back & Compare
            amt_loc = resolve_locator(page, getattr(ui_cfg, "amount_input"))
            formatted_stake = str(int(stake_amount) if float(stake_amount).is_integer() else stake_amount)
            if hasattr(amt_loc, "fill"):
                amt_loc.fill(formatted_stake)

            readback_val = ""
            if hasattr(amt_loc, "input_value"):
                readback_val = amt_loc.input_value()
            elif hasattr(amt_loc, "get_attribute"):
                readback_val = amt_loc.get_attribute("value") or ""
            else:
                readback_val = amt_loc.inner_text() or ""

            clean_num_str = readback_val.replace("$", "").replace("€", "").replace("₹", "").replace(",", "").strip()
            try:
                readback_num = float(clean_num_str)
            except (ValueError, TypeError):
                raise StakeOutOfRange(f"Failed to parse read-back stake value: '{readback_val}'")

            if abs(readback_num - stake_amount) > 0.01:
                raise StakeOutOfRange(
                    f"Stake amount mismatch: entered {stake_amount}, but read back {readback_num}"
                )

            # 4. Check Button Accessible Name / aria-label / text against allowed labels
            btn_texts: list[str] = []
            if hasattr(btn_loc, "get_attribute"):
                aria = btn_loc.get_attribute("aria-label")
                if aria:
                    btn_texts.append(aria)
                name_attr = btn_loc.get_attribute("name")
                if name_attr:
                    btn_texts.append(name_attr)
            if hasattr(btn_loc, "inner_text"):
                it = btn_loc.inner_text()
                if it:
                    btn_texts.append(it)
            if hasattr(btn_loc, "text_content"):
                tc = btn_loc.text_content()
                if tc:
                    btn_texts.append(tc)

            combined_label = " ".join(btn_texts).lower()
            if not any(str(lbl).lower() in combined_label for lbl in button_labels):
                raise ButtonLabelMismatch(
                    f"Button label '{combined_label}' does not match trade direction '{dir_enum.value}' (configured labels: {button_labels})"
                )

            preexisting_uuids = {
                str(d["uuid"])
                for d in (self.ws_listener.get_open_deals() + self.ws_listener.get_closed_deals())
                if "uuid" in d
            }

            # CLICK ONCE!
            btn_loc.click()
            click_time = time.time()

            # ─────────────────────────────────────────────────────────────────
            # STEP 3f & 3g: Confirmation & Safety Breaker Enforcement
            # ─────────────────────────────────────────────────────────────────
            confirmed_deal: Optional[dict[str, Any]] = None
            timeout = self.confirm_timeout_seconds

            while time.time() - click_time < timeout:
                deals_to_check = self.ws_listener.get_open_deals() + self.ws_listener.get_closed_deals()
                for deal in deals_to_check:
                    d_uuid = str(deal.get("uuid") or "")
                    if not d_uuid or d_uuid in preexisting_uuids:
                        continue

                    d_type = str(deal.get("deal_type") or "").strip().lower()
                    if d_type and d_type != "demo":
                        err_msg = f"CRITICAL SAFETY BREACH: Deal {d_uuid} opened with non-demo deal_type '{d_type}'!"
                        logger.critical(err_msg)
                        if self.breaker and hasattr(self.breaker, "trip"):
                            self.breaker.trip(err_msg)
                        if self.alerter:
                            try:
                                self.alerter.send_alert(f"🚨 {err_msg}")
                            except Exception:
                                pass
                        raise RealAccountDetected(err_msg)

                    d_ric = str(deal.get("asset_ric") or deal.get("ric") or "")
                    d_trend = str(deal.get("trend") or "").strip().upper()
                    expected_trends = {"CALL", "UP"} if dir_enum == TradeDirection.CALL else {"PUT", "DOWN"}

                    ric_matches = (d_ric.lower() == asset_id.lower() or asset_id.lower() in d_ric.lower())
                    trend_matches = (d_trend in expected_trends)
                    type_matches = (d_type == "demo")

                    raw_amt = deal.get("raw_amount") if deal.get("raw_amount") is not None else deal.get("amount")
                    deal_amt = deal.get("amount")

                    if ric_matches and trend_matches and type_matches:
                        # Check for amount scale suspicion on raw or deal amount
                        for check_val in (raw_amt, deal_amt):
                            if check_val is not None:
                                c_float = float(check_val)
                                is_100x = abs(c_float - (stake_amount * 100.0)) < 0.01
                                is_div100 = abs(c_float - (stake_amount / 100.0)) < 0.0001
                                if is_100x or is_div100:
                                    raw_pl = deal.get("raw_payload") if isinstance(deal.get("raw_payload"), dict) else deal
                                    logger.debug(
                                        "Raw opened payload keys: %s, amounts: raw=%s, deal=%s, expected=%s",
                                        list(raw_pl.keys()), raw_amt, deal_amt, stake_amount
                                    )
                                    alert_msg = f"amount scale suspected: raw {check_val}, expected {stake_amount}"
                                    logger.error(alert_msg)
                                    if self.alerter:
                                        try:
                                            self.alerter.send_alert(alert_msg)
                                        except Exception:
                                            pass
                                    raise AmbiguousOrder(alert_msg)

                    amount_matches = (deal_amt is not None and abs(float(deal_amt) - stake_amount) < 0.01)

                    if ric_matches and trend_matches and amount_matches and type_matches:
                        confirmed_deal = deal
                        break

                if confirmed_deal:
                    break
                time.sleep(0.05)

            if not confirmed_deal:
                trip_msg = f"Order on {asset_id} submitted via UI but unconfirmed in WebSocket events within {timeout:.0f}s"
                logger.error(trip_msg)
                if self.breaker and hasattr(self.breaker, "trip"):
                    self.breaker.trip(trip_msg)
                raise AmbiguousOrder(trip_msg)

            platform_id = str(confirmed_deal.get("uuid"))
            p_rate = confirmed_deal.get("payment_rate")
            if p_rate is None and payout is not None:
                p_rate = payout * 100.0 if float(payout) <= 1.0 else payout

            if p_rate is None or float(p_rate) <= 0:
                err_msg = f"Deal {platform_id} confirmed but payment_rate/payout is missing"
                logger.critical(err_msg)
                if self.alerter:
                    try:
                        self.alerter.send_alert(f"🚨 {err_msg}")
                    except Exception:
                        pass
                raise ValueError(err_msg)

            payout_dec = (float(p_rate) / 100.0) if float(p_rate) > 1.0 else float(p_rate)
            open_rate = float(confirmed_deal.get("open_rate") or 0.0)

            start_ts = parse_deal_timestamp(confirmed_deal.get("created_at") or confirmed_deal.get("open_quote_created_at"))
            entry_time = start_ts or datetime.now(timezone.utc)

            receipt = TradeReceipt(
                trade_id=str(trade_id or platform_id),
                asset_id=asset_id,
                direction=dir_enum,
                expiry_minutes=expiry_min,
                stake_amount=stake_amount,
                entry_price=open_rate,
                entry_time=entry_time,
                payout=payout_dec,
                platform_trade_id=platform_id,
                status="OPEN",
            )
            logger.info("Trade successfully executed on Binomo Demo: %s", receipt)
            return receipt

        finally:
            self._trade_lock.release()
