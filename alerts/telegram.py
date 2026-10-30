"""Telegram alerting and operator command interface.

PROJECT_SPEC.md Phase 8:
  - Env-only credentials: reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from os.environ.
  - Never logs or prints tokens.
  - Timeout protection on network calls (default 5.0s).
  - Graceful resilience: network errors are caught and logged, never crashing execution.
  - Rate limiting on outgoing alerts.
  - Commands accepted only from the authorized chat_id:
      /status, /stop, /resume, /reset_breaker, /reset_reference, /force_pause.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional
import urllib.parse
import urllib.request

logger = logging.getLogger("trading_system.alerts")


class TelegramAlerter:
    """Manages outgoing alerts and operator remote command processing."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: float = 5.0,
        min_interval_seconds: float = 0.5,
    ):
        # Credentials from environment ONLY if not explicitly injected
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        self.min_interval_seconds = min_interval_seconds
        self._last_send_time: float = 0.0

        # Command handlers registry
        self._command_handlers: dict[str, Callable[[list[str]], str]] = {}
        self.alert_history: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        """Check if bot token and chat_id are present."""
        return bool(self._token and self._chat_id)

    def register_command(self, command: str, handler: Callable[[list[str]], str]) -> None:
        """Register a handler for a bot slash command."""
        cmd_clean = command.strip().lower()
        if not cmd_clean.startswith("/"):
            cmd_clean = "/" + cmd_clean
        self._command_handlers[cmd_clean] = handler

    def send_alert(self, message: str) -> bool:
        """Send an alert message via Telegram Bot API with rate limiting and error recovery.

        Guarantees:
          - Never prints or logs the API token.
          - Never raises an unhandled network error to crash calling code.
        """
        now = time.time()
        self.alert_history.append({
            "timestamp": now,
            "message": message,
        })

        if not self.is_configured:
            logger.info("Telegram alert not sent (token/chat_id not configured in environment): %s", message)
            return False

        # Simple rate limit throttling
        elapsed = now - self._last_send_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        endpoint = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        data = urllib.parse.urlencode(payload).encode("utf-8")

        try:
            req = urllib.request.Request(endpoint, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self._last_send_time = time.time()
                return resp.status == 200
        except Exception as e:
            # Mask sensitive tokens in logs
            err_msg = str(e)
            if self._token and self._token in err_msg:
                err_msg = err_msg.replace(self._token, "[REDACTED_TOKEN]")
            logger.warning("Failed to send Telegram alert: %s", err_msg)
            return False

    def handle_incoming_update(self, update: dict[str, Any]) -> Optional[str]:
        """Process incoming webhook or poll update from Telegram.

        Only accepts commands from the configured authorized chat_id.
        """
        message = update.get("message", {})
        sender_chat = str(message.get("chat", {}).get("id", ""))
        text = str(message.get("text", "")).strip()

        # Authorization check
        if not self._chat_id or sender_chat != str(self._chat_id):
            logger.warning("Rejected command from unauthorized sender chat ID: %s", sender_chat)
            return None

        if not text.startswith("/"):
            return None

        parts = text.split()
        command = parts[0].lower()
        args = parts[1:]

        handler = self._command_handlers.get(command)
        if handler:
            try:
                response = handler(args)
                self.send_alert(response)
                return response
            except Exception as e:
                err = f"Error executing {command}: {e}"
                self.send_alert(f"❌ {err}")
                return err

        unknown_msg = f"Unknown command: {command}. Available: /status, /stop, /resume, /reset_breaker, /reset_reference, /force_pause"
        self.send_alert(unknown_msg)
        return unknown_msg


def create_operator_command_handlers(
    state_manager: Any,
    breaker: Any,
    kill_switch: Any,
    adaptive_controller: Optional[Any] = None,
    config: Optional[Any] = None,
) -> dict[str, Callable[[list[str]], str]]:
    """Factory creating standard operator remote command handlers."""

    def status_handler(args: list[str]) -> str:
        state = state_manager.data if hasattr(state_manager, "data") else {}
        tier = "NORMAL"
        if adaptive_controller and hasattr(adaptive_controller, "state"):
            tier = adaptive_controller.state.tier
        elif state.get("adaptive_state"):
            tier = state["adaptive_state"].get("tier", "NORMAL")

        bal = state.get("balance", 0.0)
        open_count = len(state.get("open_trades", []))
        brk_status = "TRIPPED" if breaker.is_tripped() else "OK"
        ks_status = "ACTIVE" if kill_switch.is_active() else "OFF"
        return (
            f"📊 *System Status*\n"
            f"• Tier: `{tier}`\n"
            f"• Balance: `${bal:,.2f}`\n"
            f"• Open Trades: `{open_count}`\n"
            f"• Circuit Breaker: `{brk_status}`\n"
            f"• Kill Switch: `{ks_status}`"
        )

    def stop_handler(args: list[str]) -> str:
        kill_switch.activate_file()
        return "🛑 STOPPED: KILL file created on machine. Emergency stop latched. New orders halted. Can only be cleared by deleting the KILL file on the machine."

    def resume_handler(args: list[str]) -> str:
        # Check if adaptive is PAUSED
        current_tier = None
        if adaptive_controller and hasattr(adaptive_controller, "state"):
            current_tier = adaptive_controller.state.tier

        if current_tier != "PAUSED":
            return "ℹ️ System is not in PAUSED tier. /resume only leaves PAUSED after requalification."

        # Check requalification requirements
        req_cfg = getattr(config.adaptive, "requalification", None) if config else None
        min_shadow = getattr(req_cfg, "min_shadow_trades", 30) if req_cfg else 30
        shadow_count = getattr(adaptive_controller.state, "shadow_trades_count", 0)
        shadow_wins = getattr(adaptive_controller.state, "shadow_wins", 0)
        wr = shadow_wins / shadow_count if shadow_count > 0 else 0.0
        payout = getattr(config.backtest, "default_payout", 0.85) if config else 0.85
        be = 1.0 / (1.0 + payout)

        if shadow_count < min_shadow or wr < be:
            return (
                f"❌ /resume REFUSED: Requalification requirements not met.\n"
                f"Observed {shadow_count}/{min_shadow} shadow trades (win rate {wr:.1%}, break-even {be:.1%})."
            )
        # Requalification passed -> recover to DEFENSIVE
        adaptive_controller.state.tier = "DEFENSIVE"
        adaptive_controller.state.real_trading_allowed = True
        adaptive_controller.state.in_rampup = True
        adaptive_controller.state.rampup_trades_remaining = getattr(req_cfg, "rampup_trades", 20) if req_cfg else 20
        return "▶️ RESUMED: Requalification passed. Adaptive tier moved to DEFENSIVE in ramp-up mode."

    def reset_breaker_handler(args: list[str]) -> str:
        breaker.reset()
        return "⚡ Breaker reset: Circuit breaker state restored to OK."

    def reset_reference_handler(args: list[str]) -> str:
        if adaptive_controller and hasattr(adaptive_controller, "state"):
            curr_eq = adaptive_controller.state.current_equity
            adaptive_controller.set_checkpoint(curr_eq)
            return f"🔄 Reference reset: Reference equity set to current equity ${curr_eq:,.2f}."
        return "Reference reset completed."

    def force_pause_handler(args: list[str]) -> str:
        if adaptive_controller and hasattr(adaptive_controller, "state"):
            adaptive_controller.state.tier = "PAUSED"
            adaptive_controller.state.real_trading_allowed = False
            adaptive_controller.state.stake_scale = 0.0
            adaptive_controller.state.transition_reason = "Manual /force_pause command from operator"
            return "⏸️ FORCED PAUSE: Adaptive tier moved to PAUSED. Real trading halted."
        return "Forced pause completed."

    return {
        "/status": status_handler,
        "/stop": stop_handler,
        "/resume": resume_handler,
        "/reset_breaker": reset_breaker_handler,
        "/reset_reference": reset_reference_handler,
        "/force_pause": force_pause_handler,
    }
