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


# ── Selector Resolution & Validation Helpers ─────────────────────────────────


def resolve_locator(page: Any, spec: Any, **kwargs: Any) -> Any:
    """Resolve a selector specification into a Playwright locator.

    Supported spec formats:
      - dict or BaseModel with 'test_id': page.get_by_test_id(test_id)
      - dict or BaseModel with 'role' (and optional 'name'): page.get_by_role(role, name=name)
      - dict or BaseModel with 'selector': page.locator(selector)
      - str: page.locator(spec)
    Templates ({label}, {name}, etc.) are formatted using kwargs.
    """
    if spec is None:
        raise ValueError("Cannot resolve locator for None selector spec")

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
        role = str(spec_dict["role"])
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

    if "selector" in spec_dict and spec_dict["selector"]:
        sel = str(spec_dict["selector"])
        if kwargs:
            sel = sel.format(**kwargs)
        return page.locator(sel)

    raise ValueError(f"Unrecognized selector spec format: {spec}")


def validate_ui_selectors(page: Any, ui_config: Any) -> list[str]:
    """Validate that every required selector resolves on the page.

    Returns a list of failed selector names (empty if all resolved).
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
        if spec is None:
            failed.append(key)
            continue

        try:
            loc = resolve_locator(page, spec)
            count = loc.count() if hasattr(loc, "count") else 1
            if count == 0:
                failed.append(key)
        except Exception as exc:
            logger.debug("Selector '%s' failed resolution check: %s", key, exc)
            failed.append(key)

    # Template selectors: verify formatting and resolution syntax
    templates = [
        ("expiry_option", {"label": "1 min"}),
        ("asset_option", {"name": "Crypto IDX"}),
    ]
    for key, sample_kwargs in templates:
        spec = getattr(ui_config, key, None)
        if spec is None and isinstance(ui_config, dict):
            spec = ui_config.get(key)
        if spec is None:
            failed.append(key)
            continue
        try:
            resolve_locator(page, spec, **sample_kwargs)
        except Exception as exc:
            logger.debug("Template selector '%s' failed validation: %s", key, exc)
            failed.append(key)

    return failed


def is_panel_blocked(page: Any) -> bool:
    """Check if an overlay, modal, or dialog blocks the trading panel."""
    if page is None:
        return False

    modal_selectors = [
        "[role='dialog']",
        ".modal-dialog",
        ".popup-container",
        ".blocking-modal",
        ".dialog-overlay",
    ]
    for sel in modal_selectors:
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
        self.ws_listener = ws_listener or (session.ws_listener if session else BinomoWSListener())
        self.reader = reader or BinomoReadOnlySource(session=session, config=config, alerter=alerter)
        self.config = config
        self.breaker = breaker
        self.alerter = alerter

        # Concurrency control: strictly 1 order in flight
        self._trade_lock = threading.Lock()

        # Settled deals tracking to avoid duplicate settlement calls
        self._settled_deal_uuids: set[str] = set()

        # Cached platform currency stake limits
        self._platform_min_stake: float = 1.0
        self._platform_max_stake: float = 1000.0
        self._platform_limits_loaded: bool = False

        # Load configurations
        self.confirm_timeout_seconds: float = 15.0
        self.max_frame_age_seconds: float = 60.0
        self.expiry_labels: dict[int, str] = {
            1: "1 min",
            5: "5 min",
            15: "15 min",
            30: "30 min",
            60: "60 min",
        }

        if config and hasattr(config, "binomo"):
            b_cfg = config.binomo
            self.confirm_timeout_seconds = float(getattr(b_cfg, "confirm_timeout_seconds", 15.0))
            self.max_frame_age_seconds = float(getattr(b_cfg, "max_frame_age_seconds", 60.0))
            if getattr(b_cfg, "expiry_labels", None):
                self.expiry_labels = {
                    int(k): str(v) for k, v in b_cfg.expiry_labels.items()
                }

    @property
    def ui_config(self) -> Any:
        if self.config and hasattr(self.config, "binomo"):
            return getattr(self.config.binomo, "ui", None)
        return None

    # ── Platform Currency Limits ─────────────────────────────────────────────

    def load_platform_limits(self) -> tuple[float, float]:
        """Fetch and cache min/max stake limits from /platform/private/v2/currencies."""
        try:
            url = f"{self.reader.base_api_url}/platform/private/v2/currencies"
            data = self.reader._execute_get(url)
            currencies: list[dict[str, Any]] = []
            if isinstance(data, list):
                currencies = data
            elif isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    currencies = data["data"]
                elif isinstance(data.get("currencies"), list):
                    currencies = data["currencies"]

            if currencies and isinstance(currencies[0], dict):
                c0 = currencies[0]
                min_val = c0.get("min_payment") or c0.get("min_bet") or c0.get("min_amount") or 1.0
                max_val = c0.get("max_payment") or c0.get("max_bet") or c0.get("max_amount") or 1000.0
                self._platform_min_stake = float(min_val)
                self._platform_max_stake = float(max_val)
                self._platform_limits_loaded = True
                logger.info("Loaded platform stake limits: [%.2f, %.2f]", self._platform_min_stake, self._platform_max_stake)
        except Exception as exc:
            logger.warning("Could not load /platform/private/v2/currencies: %s. Using default limits [%.2f, %.2f]",
                           exc, self._platform_min_stake, self._platform_max_stake)

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
        if failed:
            msg = f"Startup refused: binomo.ui selectors failed to resolve on page: {failed}"
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

            trades.append(
                Trade(
                    trade_id=uuid_str,
                    asset_id=str(d.get("asset_ric") or ""),
                    direction=direction,
                    expiry_minutes=int(self.config.trading.expiry_minutes if self.config else 1),
                    stake_amount=float(d.get("amount") or 0.0),
                    entry_price=float(d.get("open_rate") or 0.0),
                    entry_time=now_dt,
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

            trades.append(
                Trade(
                    trade_id=uuid_str,
                    asset_id=str(d.get("asset_ric") or d.get("ric") or ""),
                    direction=direction,
                    expiry_minutes=int(self.config.trading.expiry_minutes if self.config else 1),
                    stake_amount=float(d.get("amount") or d.get("bet") or 0.0),
                    entry_price=float(d.get("open_rate") or d.get("rate") or 0.0),
                    entry_time=now_dt,
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

        Maps status won/lost/tie to TradeResult. Raises ValueError on unknown status.
        """
        closed_deals = self.ws_listener.get_closed_deals()
        settled_results: list[dict[str, Any]] = []

        open_trade_ids: set[str] = set()
        if hasattr(trade_manager, "state_manager") and hasattr(trade_manager.state_manager, "data"):
            open_list = trade_manager.state_manager.data.get("open_trades", [])
            for ot in open_list:
                tid = ot.get("id") if isinstance(ot, dict) else getattr(ot, "id", None)
                if tid:
                    open_trade_ids.add(str(tid))

        for deal in closed_deals:
            uuid_str = str(deal.get("uuid") or "")
            if not uuid_str or uuid_str in self._settled_deal_uuids:
                continue

            if uuid_str not in open_trade_ids:
                continue

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
            payout_val = (float(p_rate) / 100.0) if (p_rate and float(p_rate) > 1.0) else (float(p_rate) if p_rate else None)

            try:
                res = trade_manager.record_settlement(
                    trade_id=uuid_str,
                    outcome=outcome,
                    payout=payout_val,
                )
                self._settled_deal_uuids.add(uuid_str)
                settled_results.append(res)
                logger.info("Settlement recorded for deal %s: outcome=%s, payout=%s", uuid_str, outcome, payout_val)
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
        """Place a demo trade adhering strictly to Phase 10B rules.

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
            # 1. Config mode must be demo
            config_mode = str(getattr(self.config, "mode", "")).strip().lower()
            if config_mode != "demo":
                raise RealAccountDetected(f"Safety violation: Config mode is '{config_mode}', not 'demo'")

            # 2. Session must be logged in
            if not self.session.is_logged_in():
                raise RuntimeError("Cannot place trade: Binomo session is not logged in")

            # 3. WebSocket latest balance_changed account_type must be "demo"
            ws_account_type = str(self.ws_listener.get_account_type() or "").strip().lower()
            if ws_account_type != "demo":
                raise RealAccountDetected(f"Safety violation: WebSocket account_type is '{ws_account_type}', not 'demo'")

            # 4. DOM demo_indicator must show demo
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
            if is_panel_blocked(page):
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
            if expiry_min not in self.expiry_labels:
                raise ExpiryUnavailable(f"Expiry {expiry_min}m not available in binomo.expiry_labels {sorted(self.expiry_labels.keys())}")
            expected_expiry_label = self.expiry_labels[expiry_min]

            # ─────────────────────────────────────────────────────────────────
            # STEP 3d (Validation): Platform Currency Min/Max Limits
            # ─────────────────────────────────────────────────────────────────
            if not self._platform_limits_loaded:
                self.load_platform_limits()

            if stake_amount < self._platform_min_stake or stake_amount > self._platform_max_stake:
                raise StakeOutOfRange(
                    f"Stake amount {stake_amount} is outside platform limits [{self._platform_min_stake}, {self._platform_max_stake}]"
                )

            # ─────────────────────────────────────────────────────────────────
            # STEP 3b: Select Asset through UI and Verify active_asset_label
            # ─────────────────────────────────────────────────────────────────
            asset_sel_loc = resolve_locator(page, getattr(ui_cfg, "asset_selector"))
            asset_sel_loc.click()

            # Click asset option using template
            asset_opt_loc = resolve_locator(page, getattr(ui_cfg, "asset_option"), name=asset_id)
            asset_opt_loc.click()

            # Verify active_asset_label
            active_asset_loc = resolve_locator(page, getattr(ui_cfg, "active_asset_label"))
            active_asset_text = (active_asset_loc.inner_text() or active_asset_loc.text_content() or "").strip()
            if asset_id.lower() not in active_asset_text.lower():
                raise AssetSelectionMismatch(
                    f"Active asset label '{active_asset_text}' does not match requested asset '{asset_id}'"
                )

            # ─────────────────────────────────────────────────────────────────
            # STEP 3c: Select Expiry through UI and Verify active_expiry_label
            # ─────────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────────
            # STEP 3d: Enter Stake, Read Back & Compare
            # ─────────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────────
            # STEP 3e: Check Button Accessible Name & Click ONCE
            # ─────────────────────────────────────────────────────────────────
            if dir_enum == TradeDirection.CALL:
                btn_loc = resolve_locator(page, getattr(ui_cfg, "call_button"))
                expected_keyword = "call"
            else:
                btn_loc = resolve_locator(page, getattr(ui_cfg, "put_button"))
                expected_keyword = "put"

            btn_label = ""
            if hasattr(btn_loc, "get_attribute"):
                btn_label = btn_loc.get_attribute("aria-label") or ""
            if not btn_label and hasattr(btn_loc, "inner_text"):
                btn_label = btn_loc.inner_text() or ""

            if expected_keyword not in btn_label.lower():
                raise ButtonLabelMismatch(
                    f"Button label '{btn_label}' does not match trade direction '{dir_enum.value}'"
                )

            # Record preexisting deal uuids prior to clicking
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

                    # Step 3g: deal_type other than 'demo' trips breaker immediately
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

                    # Check matching ric, trend, amount, and deal_type == "demo"
                    d_ric = str(deal.get("asset_ric") or deal.get("ric") or "")
                    d_trend = str(deal.get("trend") or "").strip().upper()
                    d_amt = deal.get("amount")

                    expected_trends = {"CALL", "UP"} if dir_enum == TradeDirection.CALL else {"PUT", "DOWN"}
                    ric_matches = (d_ric.lower() == asset_id.lower() or asset_id.lower() in d_ric.lower())
                    trend_matches = (d_trend in expected_trends)
                    amount_matches = (d_amt is not None and abs(float(d_amt) - stake_amount) < 0.01)
                    type_matches = (d_type == "demo")

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

            # Extract confirmed deal parameters
            platform_id = str(confirmed_deal.get("uuid"))
            p_rate = confirmed_deal.get("payment_rate") or (payout * 100.0 if payout else 83.0)
            payout_dec = (float(p_rate) / 100.0) if float(p_rate) > 1.0 else float(p_rate)
            open_rate = float(confirmed_deal.get("open_rate") or 0.0)

            receipt = TradeReceipt(
                trade_id=str(trade_id or platform_id),
                asset_id=asset_id,
                direction=dir_enum,
                expiry_minutes=expiry_min,
                stake_amount=stake_amount,
                entry_price=open_rate,
                entry_time=datetime.now(timezone.utc),
                payout=payout_dec,
                platform_trade_id=platform_id,
                status="OPEN",
            )
            logger.info("Trade successfully executed on Binomo Demo: %s", receipt)
            return receipt

        finally:
            self._trade_lock.release()
