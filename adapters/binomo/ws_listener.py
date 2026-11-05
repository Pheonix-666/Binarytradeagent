"""Passive Phoenix WebSocket frame parser for Binomo platform.

Adheres strictly to PROJECT_SPEC.md:
- Passive listener only, never sends any frames or calls evaluate/click/type/fill.
- Thread-safe state for balance, account type, deals, and frame timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("trading_system.adapters.binomo.ws")


def parse_phoenix_frame(raw_text: str) -> Optional[tuple[Optional[str], Optional[str], str, str, Any]]:
    """Parse raw WebSocket text into Phoenix components: (join_ref, ref, topic, event, payload)."""
    try:
        data = json.loads(raw_text)
    except Exception:
        return None

    if isinstance(data, list) and len(data) >= 5:
        join_ref = str(data[0]) if data[0] is not None else None
        ref = str(data[1]) if data[1] is not None else None
        topic = str(data[2] or "")
        event = str(data[3] or "")
        payload = data[4]
        return join_ref, ref, topic, event, payload

    if isinstance(data, dict):
        join_ref = str(data.get("join_ref")) if data.get("join_ref") is not None else None
        ref = str(data.get("ref")) if data.get("ref") is not None else None
        topic = str(data.get("topic") or "")
        event = str(data.get("event") or "")
        payload = data.get("payload")
        return join_ref, ref, topic, event, payload

    return None


class BinomoWSListener:
    """Thread-safe passive Phoenix WebSocket frame listener."""

    def __init__(self, amount_scale: float = 1.0) -> None:
        self._lock = threading.Lock()
        self.amount_scale: float = float(amount_scale)

        # Account & balance state
        self._latest_balance: Optional[float] = None
        self._latest_account_type: Optional[str] = None
        self._latest_currency: Optional[str] = None

        # Deals state keyed by uuid
        self._open_deals: dict[str, dict[str, Any]] = {}
        self._closed_deals: dict[str, dict[str, Any]] = {}
        self._batch_close_events: list[dict[str, Any]] = []

        # Heartbeat & frame tracking
        self._last_frame_time: Optional[datetime] = None
        self._total_frames_received: int = 0
        self._last_phx_reply_status: Optional[str] = None

        # Login observation indicators
        self._seen_ws_open: bool = False
        self._seen_account_join: bool = False
        self._seen_passport_or_assets: bool = False

    # ── Thread-Safe State Accessors ──────────────────────────────────────────

    def get_balance(self) -> Optional[float]:
        with self._lock:
            return self._latest_balance

    def get_account_type(self) -> Optional[str]:
        with self._lock:
            return self._latest_account_type

    def get_currency(self) -> Optional[str]:
        with self._lock:
            return self._latest_currency

    def is_demo(self) -> bool:
        """True ONLY if the latest balance_changed payload has account_type == 'demo'."""
        with self._lock:
            return (self._latest_account_type or "").strip().lower() == "demo"

    def get_open_deals(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._open_deals.values())

    def get_closed_deals(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._closed_deals.values())

    def get_deal_by_uuid(self, uuid: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._open_deals.get(uuid) or self._closed_deals.get(uuid)

    def get_last_frame_time(self) -> Optional[datetime]:
        with self._lock:
            return self._last_frame_time

    def get_last_phx_reply_status(self) -> Optional[str]:
        with self._lock:
            return self._last_phx_reply_status

    def is_logged_in(self) -> bool:
        """True if ws.binomo.com is open AND (account join seen OR passport seen)."""
        with self._lock:
            return bool(self._seen_ws_open and (self._seen_account_join or self._seen_passport_or_assets))

    def get_login_indicators(self) -> dict[str, bool]:
        with self._lock:
            return {
                "ws_open": self._seen_ws_open,
                "account_join": self._seen_account_join,
                "passport_or_assets": self._seen_passport_or_assets,
                "logged_in": bool(self._seen_ws_open and (self._seen_account_join or self._seen_passport_or_assets)),
            }

    def record_ws_open(self, url: str) -> None:
        if "ws.binomo.com" in url or "as.binomo.com" in url:
            with self._lock:
                self._seen_ws_open = True

    def record_passport_or_assets(self, url: str) -> None:
        if "/passport/" in url or "/bo-assets/" in url:
            with self._lock:
                self._seen_passport_or_assets = True

    # ── Passive Frame Ingestion ─────────────────────────────────────────────

    def on_frame(self, raw_text: str, direction: str = "in") -> None:
        """Parse frame and update internal state in a passive, thread-safe manner."""
        parsed = parse_phoenix_frame(raw_text)
        if not parsed:
            return

        join_ref, ref, topic, event, payload = parsed
        now_utc = datetime.now(timezone.utc)

        with self._lock:
            self._last_frame_time = now_utc
            self._total_frames_received += 1

            # 1. Topic join tracking
            if event == "phx_join":
                t_lower = topic.lower()
                if t_lower == "account" or t_lower.startswith("account:") or t_lower.startswith("user:"):
                    self._seen_account_join = True
                return

            # 2. Phoenix Reply tracking
            if event == "phx_reply":
                if isinstance(payload, dict):
                    status = str(payload.get("status") or "")
                    self._last_phx_reply_status = status
                return

            # 3. Heartbeat
            if event == "heartbeat" or topic == "phoenix":
                return

            # 4. Balance Changed
            if event == "balance_changed" or (topic == "account" and "balance" in event):
                if isinstance(payload, dict):
                    if "balance" in payload and payload["balance"] not in (None, "<num>", ""):
                        try:
                            raw_bal = float(payload["balance"])
                            self._latest_balance = raw_bal * self.amount_scale
                        except (ValueError, TypeError):
                            pass
                    if "account_type" in payload:
                        self._latest_account_type = str(payload["account_type"]).strip().lower()
                    if "currency" in payload or "currency_iso" in payload:
                        curr = payload.get("currency") or payload.get("currency_iso")
                        if curr:
                            self._latest_currency = str(curr).strip().upper()
                return

            # 5. Deal Events: opened, closed, close_deal_batch
            if topic == "bo" or event in ("opened", "closed", "close_deal_batch"):
                self._parse_deal_event(topic, event, payload, now_utc)

    def _parse_deal_event(self, topic: str, event: str, payload: Any, now_utc: datetime) -> None:
        """Internal helper called with lock held."""
        if not isinstance(payload, dict):
            return

        if event in ("opened", "deal_opened"):
            deal_uuid = str(payload.get("uuid") or payload.get("id") or payload.get("deal_id") or "")
            if not deal_uuid:
                return

            raw_amt = self._safe_float(payload.get("amount") or payload.get("bet"))
            scaled_amt = (raw_amt * self.amount_scale) if raw_amt is not None else None
            close_created = payload.get("close_quote_created_at") or payload.get("finished_at")
            fin_at = payload.get("finished_at") or payload.get("close_quote_created_at")

            deal_info: dict[str, Any] = {
                "uuid": deal_uuid,
                "asset_ric": payload.get("asset_ric") or payload.get("ric") or payload.get("asset") or "",
                "trend": payload.get("trend") or payload.get("direction") or "",
                "amount": scaled_amt,
                "raw_amount": raw_amt,
                "open_rate": self._safe_float(payload.get("open_rate") or payload.get("rate")),
                "end_rate": None,
                "payment_rate": self._safe_float(payload.get("payment_rate")),
                "status": str(payload.get("status") or "open"),
                "deal_type": str(payload.get("deal_type") or payload.get("option_type") or ""),
                "created_at": payload.get("created_at") or payload.get("open_quote_created_at") or payload.get("requested_at"),
                "close_quote_created_at": close_created,
                "finished_at": fin_at,
                "updated_at": now_utc.isoformat(),
                "raw_payload": payload,
            }
            self._open_deals[deal_uuid] = deal_info

        elif event in ("closed", "deal_closed"):
            deal_uuid = str(payload.get("uuid") or payload.get("id") or payload.get("deal_id") or "")
            if not deal_uuid:
                return

            deal_info = self._open_deals.pop(deal_uuid, {})
            deal_info["uuid"] = deal_uuid
            deal_info["asset_ric"] = payload.get("asset_ric") or payload.get("ric") or deal_info.get("asset_ric", "")
            deal_info["trend"] = payload.get("trend") or payload.get("direction") or deal_info.get("trend", "")
            deal_info["created_at"] = payload.get("created_at") or payload.get("open_quote_created_at") or payload.get("requested_at") or deal_info.get("created_at")
            if "amount" in payload or "bet" in payload:
                raw_amt = self._safe_float(payload.get("amount") or payload.get("bet"))
                deal_info["raw_amount"] = raw_amt
                deal_info["amount"] = (raw_amt * self.amount_scale) if raw_amt is not None else None
            if "end_rate" in payload or "close_rate" in payload:
                deal_info["end_rate"] = self._safe_float(payload.get("end_rate") or payload.get("close_rate"))
            if "payment_rate" in payload:
                deal_info["payment_rate"] = self._safe_float(payload.get("payment_rate"))
            deal_info["status"] = str(payload.get("status") or "closed")
            if "deal_type" in payload or "option_type" in payload:
                deal_info["deal_type"] = str(payload.get("deal_type") or payload.get("option_type") or "")
            close_created = payload.get("close_quote_created_at") or payload.get("finished_at") or deal_info.get("close_quote_created_at")
            fin_at = payload.get("finished_at") or payload.get("close_quote_created_at") or deal_info.get("finished_at")
            deal_info["close_quote_created_at"] = close_created
            deal_info["finished_at"] = fin_at
            deal_info["updated_at"] = now_utc.isoformat()
            deal_info["raw_payload"] = payload

            self._closed_deals[deal_uuid] = deal_info

        elif event == "close_deal_batch":
            batch_record = {
                "ric": payload.get("ric"),
                "finished_at": payload.get("finished_at"),
                "end_rate": self._safe_float(payload.get("end_rate")),
                "received_at": now_utc.isoformat(),
            }
            self._batch_close_events.append(batch_record)

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None or val == "<num>":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ── Playwright Integration ──────────────────────────────────────────────

    def attach_to_websocket(self, ws: Any) -> None:
        """Attach passive listener to a Playwright WebSocket."""
        clean_url = getattr(ws, "url", "")
        self.record_ws_open(clean_url)

        def on_frame_received(payload: Any) -> None:
            if isinstance(payload, str):
                self.on_frame(payload, direction="in")

        def on_frame_sent(payload: Any) -> None:
            if isinstance(payload, str):
                self.on_frame(payload, direction="out")

        try:
            ws.on("framereceived", on_frame_received)
            ws.on("framesent", on_frame_sent)
        except Exception as exc:
            logger.debug("Failed attaching listeners to WebSocket: %s", exc)

