"""Telegram background polling worker with operator commands and trade ticket confirmation.

PROJECT_SPEC.md Phase 9:
  - Background thread for Telegram getUpdates long-polling with exception isolation.
  - Persists update offset in state; drops pending updates on startup.
  - Dispatches commands to existing handlers (/status, /stop, /resume, etc.) from configured chat ID only.
  - Handles ticket confirmation (/approve <id>, /reject <id>) with expiration timeout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import threading
import time
from typing import Any, Callable, Optional
import urllib.error
import urllib.parse
import urllib.request

from alerts.telegram import TelegramAlerter
from core.clock import Clock, RealClock
from core.config import AppConfig

logger = logging.getLogger("trading_system.telegram_worker")


@dataclass
class ConfirmationTicket:
    """A trade awaiting operator confirmation."""

    ticket_id: str
    decision: Any
    account: Any
    created_at: datetime
    expires_at: datetime
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED, EXPIRED
    on_approve: Optional[Callable[[], Any]] = None
    on_reject: Optional[Callable[[], Any]] = None


class TelegramWorker(threading.Thread):
    """Background polling worker for Telegram bot commands and order approvals."""

    def __init__(
        self,
        alerter: TelegramAlerter,
        state_manager: Any,
        config: AppConfig,
        clock: Optional[Clock] = None,
        command_handlers: Optional[dict[str, Callable[[list[str]], str]]] = None,
        poll_interval: float = 1.0,
    ) -> None:
        super().__init__(daemon=True, name="TelegramWorker")
        self.alerter = alerter
        self.state_manager = state_manager
        self.config = config
        self.clock = clock or RealClock()
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self.tickets: dict[str, ConfirmationTicket] = {}
        self.lock = threading.Lock()

        # Command handlers
        self.handlers: dict[str, Callable[[list[str]], str]] = dict(command_handlers or {})
        self.handlers["/approve"] = self._cmd_approve
        self.handlers["/reject"] = self._cmd_reject

        # Offset tracking
        saved_offset = getattr(state_manager, "data", {}).get("telegram_offset", 0)
        self.offset: int = int(saved_offset or 0)

    def register_ticket(
        self,
        ticket_id: str,
        decision: Any,
        account: Any,
        on_approve: Optional[Callable[[], Any]] = None,
        on_reject: Optional[Callable[[], Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ConfirmationTicket:
        """Register a new confirmation ticket and send alert prompt."""
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.config.execution, "confirm_timeout_seconds", 60)
        )
        now_dt = self.clock.now()
        expires_at = now_dt + timedelta(seconds=timeout)

        ticket = ConfirmationTicket(
            ticket_id=ticket_id,
            decision=decision,
            account=account,
            created_at=now_dt,
            expires_at=expires_at,
            on_approve=on_approve,
            on_reject=on_reject,
        )

        with self.lock:
            self.tickets[ticket_id] = ticket

        msg = (
            f"🔔 *Order Confirmation Required*\n"
            f"• Ticket: `{ticket_id}`\n"
            f"• Asset: {getattr(decision, 'asset', 'N/A')}\n"
            f"• Direction: {getattr(decision, 'direction', 'N/A')}\n"
            f"• Stake: ${getattr(decision, 'stake', 0.0):,.2f}\n"
            f"• Expiry: {getattr(decision, 'expiry_minutes', 0)}m\n"
            f"• Pattern: {getattr(decision, 'pattern_name', 'N/A')}\n"
            f"• Timeout: {timeout}s\n\n"
            f"Reply with `/approve {ticket_id}` or `/reject {ticket_id}`"
        )
        self.alerter.send_alert(msg)
        return ticket

    def _cmd_approve(self, args: list[str]) -> str:
        if not args:
            return "Usage: /approve <ticket_id>"
        tid = args[0].strip()
        now_dt = self.clock.now()

        with self.lock:
            ticket = self.tickets.get(tid)
            if not ticket:
                return f"❌ Ticket `{tid}` not found."
            if ticket.status != "PENDING":
                return f"ℹ️ Ticket `{tid}` is already {ticket.status}."
            if now_dt > ticket.expires_at:
                ticket.status = "EXPIRED"
                return f"⏰ Ticket `{tid}` has expired."

            ticket.status = "APPROVED"
            fn = ticket.on_approve

        if fn:
            try:
                fn()
                return f"✅ Ticket `{tid}` approved and dispatched."
            except Exception as exc:
                logger.error("Error executing approved ticket %s: %s", tid, exc)
                return f"❌ Error executing approved ticket `{tid}`: {exc}"
        return f"✅ Ticket `{tid}` approved."

    def _cmd_reject(self, args: list[str]) -> str:
        if not args:
            return "Usage: /reject <ticket_id>"
        tid = args[0].strip()
        now_dt = self.clock.now()

        with self.lock:
            ticket = self.tickets.get(tid)
            if not ticket:
                return f"❌ Ticket `{tid}` not found."
            if ticket.status != "PENDING":
                return f"ℹ️ Ticket `{tid}` is already {ticket.status}."

            ticket.status = "REJECTED"
            fn = ticket.on_reject

        if fn:
            try:
                fn()
            except Exception as exc:
                logger.warning("Error running on_reject for %s: %s", tid, exc)
        return f"🚫 Ticket `{tid}` rejected."

    def check_expired_tickets(self) -> None:
        """Sweep and mark expired tickets."""
        now_dt = self.clock.now()
        with self.lock:
            for tid, t in list(self.tickets.items()):
                if t.status == "PENDING" and now_dt > t.expires_at:
                    t.status = "EXPIRED"
                    if t.on_reject:
                        try:
                            t.on_reject()
                        except Exception:
                            pass
                    self.alerter.send_alert(f"⏰ Trade ticket `{tid}` timed out without confirmation.")

    def drop_pending_updates(self) -> None:
        """Drop pending Telegram updates so stale messages are ignored on startup."""
        if not self.alerter.is_configured:
            return
        try:
            updates = self._fetch_updates(offset=-1, timeout=0)
            if updates:
                max_update_id = max(u.get("update_id", 0) for u in updates)
                self.offset = max_update_id + 1
                self._save_offset()
                logger.info("Dropped pending updates, starting offset at %d", self.offset)
        except Exception as exc:
            logger.warning("Could not drop pending updates on startup: %s", exc)

    def _fetch_updates(self, offset: int, timeout: int = 5) -> list[dict[str, Any]]:
        token = getattr(self.alerter, "_token", None)
        if not token:
            return []
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset={offset}&timeout={timeout}"
        req = urllib.request.Request(url, headers={"User-Agent": "TradeAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return data.get("result", [])
        return []

    def _save_offset(self) -> None:
        if hasattr(self.state_manager, "update"):
            self.state_manager.update(telegram_offset=self.offset)

    def process_message(self, message_obj: dict[str, Any]) -> Optional[str]:
        """Validate chat ID, extract text, and route command."""
        chat_id = str(message_obj.get("chat", {}).get("id", ""))
        expected_chat_id = str(getattr(self.alerter, "_chat_id", ""))
        if chat_id != expected_chat_id:
            logger.warning("Unauthorized command from chat %s (expected %s)", chat_id, expected_chat_id)
            return None

        text = message_obj.get("text", "").strip()
        if not text.startswith("/"):
            return None

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        handler = self.handlers.get(cmd)
        if handler:
            try:
                res = handler(args)
                self.alerter.send_alert(res)
                return res
            except Exception as exc:
                err = f"Error processing {cmd}: {exc}"
                logger.error(err)
                self.alerter.send_alert(f"❌ {err}")
                return err

        unknown = f"Unknown command: {cmd}"
        self.alerter.send_alert(unknown)
        return unknown

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Main long-polling loop with exception isolation."""
        logger.info("TelegramWorker started.")
        self.drop_pending_updates()

        while not self._stop_event.is_set():
            try:
                self.check_expired_tickets()

                if not self.alerter.is_configured:
                    time.sleep(self.poll_interval)
                    continue

                updates = self._fetch_updates(offset=self.offset, timeout=5)
                for update in updates:
                    update_id = update.get("update_id", 0)
                    self.offset = max(self.offset, update_id + 1)
                    self._save_offset()

                    if "message" in update:
                        self.process_message(update["message"])

            except Exception as exc:
                logger.warning("TelegramWorker polling error (resilient, retrying): %s", exc)
                time.sleep(self.poll_interval)

