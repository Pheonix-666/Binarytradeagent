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
import urllib.error
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
        self.consecutive_failures: int = 0

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
          - On HTTP 400 (Markdown formatting errors), retries immediately with plain text.
          - Tracks consecutive failures.
        """
        now = time.time()
        self.alert_history.append({
            "timestamp": now,
            "message": message,
        })

        if not self.is_configured:
            logger.info("Telegram alert not sent (token/chat_id not configured in environment): %s", message)
            self.consecutive_failures = 0
            return False

        # Simple rate limit throttling
        elapsed = now - self._last_send_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        endpoint = f"https://api.telegram.org/bot{self._token}/sendMessage"

        def _send(payload: dict[str, str]) -> bool:
            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self._last_send_time = time.time()
                return resp.status == 200

        # Try with Markdown first
        try:
            success = _send({
                "chat_id": str(self._chat_id),
                "text": message,
                "parse_mode": "Markdown",
            })
            if success:
                self.consecutive_failures = 0
                return True
        except urllib.error.HTTPError as e:
            if e.code == 400:
                # Markdown parse error — retry without parse_mode (plain text)
                try:
                    logger.debug("Markdown parse failed; retrying with plain text")
                    success = _send({
                        "chat_id": str(self._chat_id),
                        "text": message,
                    })
                    if success:
                        self.consecutive_failures = 0
                        return True
                except Exception as retry_e:
                    logger.warning("Failed retry with plain text: %s", retry_e)
            else:
                err_msg = str(e)
                if self._token and self._token in err_msg:
                    err_msg = err_msg.replace(self._token, "[REDACTED_TOKEN]")
                logger.warning("Failed to send Telegram alert: %s", err_msg)
        except Exception as e:
            err_msg = str(e)
            if self._token and self._token in err_msg:
                err_msg = err_msg.replace(self._token, "[REDACTED_TOKEN]")
            logger.warning("Failed to send Telegram alert: %s", err_msg)

        self.consecutive_failures += 1
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
    trade_manager: Optional[Any] = None,
    reconcile_fn: Optional[Callable[[], Any]] = None,
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
            f"📊 System Status\n"
            f"• Tier: {tier}\n"
            f"• Balance: ${bal:,.2f}\n"
            f"• Open Trades: {open_count}\n"
            f"• Circuit Breaker: {brk_status}\n"
            f"• Kill Switch: {ks_status}"
        )

    def stop_handler(args: list[str]) -> str:
        kill_switch.activate_file()
        return "🛑 STOPPED: KILL file created on machine. Emergency stop latched. New orders halted. Can only be cleared by deleting the KILL file on the machine."

    def resume_handler(args: list[str]) -> str:
        current_tier = None
        if adaptive_controller and hasattr(adaptive_controller, "state"):
            current_tier = adaptive_controller.state.tier

        if current_tier != "PAUSED":
            return "ℹ️ System is not in PAUSED tier. /resume only leaves PAUSED after requalification."

        req_cfg = getattr(config.adaptive, "requalification", None) if config else None
        min_shadow = getattr(req_cfg, "min_shadow_trades", 30) if req_cfg else 30

        # Use real payouts from shadow trade records in state
        shadow_trades = getattr(adaptive_controller.state, "shadow_trades", [])
        shadow_count = len(shadow_trades)
        if shadow_count == 0:
            shadow_count = getattr(adaptive_controller.state, "shadow_trades_count", 0)

        if shadow_trades:
            wins = sum(1 for t in shadow_trades if str(t.get("result") if isinstance(t, dict) else getattr(t, "result", "")).upper() == "WIN")
            payouts = [float(t.get("payout", 0.85) if isinstance(t, dict) else getattr(t, "payout", 0.85)) for t in shadow_trades]
            avg_payout = sum(payouts) / len(payouts) if payouts else 0.85
            wr = wins / len(shadow_trades)
        else:
            shadow_wins = getattr(adaptive_controller.state, "shadow_wins", 0)
            wr = shadow_wins / shadow_count if shadow_count > 0 else 0.0
            avg_payout = getattr(config.backtest, "default_payout", 0.85) if config else 0.85

        be = 1.0 / (1.0 + avg_payout)
        if shadow_count < min_shadow or wr < be:
            return (
                f"❌ /resume REFUSED: Requalification requirements not met.\n"
                f"Observed {shadow_count}/{min_shadow} shadow trades (win rate {wr:.1%}, break-even {be:.1%})."
            )

        # Transition tier via controller method
        if hasattr(adaptive_controller, "recover_from_pause"):
            adaptive_controller.recover_from_pause()
        elif hasattr(adaptive_controller, "transition_tier"):
            adaptive_controller.transition_tier("DEFENSIVE", "Operator /resume requalification passed")
        else:
            adaptive_controller.state.tier = "DEFENSIVE"
            adaptive_controller.state.real_trading_allowed = True
            adaptive_controller.state.in_rampup = True
            adaptive_controller.state.rampup_trades_remaining = getattr(req_cfg, "rampup_trades", 20) if req_cfg else 20

        return "▶️ RESUMED: Requalification passed. Adaptive tier moved to DEFENSIVE in ramp-up mode."

    def reset_breaker_handler(args: list[str]) -> str:
        # Require explicit confirmation reply
        args_lower = [a.lower() for a in args]
        if "confirm" not in args_lower:
            return "⚠️ /reset_breaker requires explicit confirmation. Reply '/reset_breaker confirm' to proceed."

        # Require fresh successful reconcile first
        if reconcile_fn is not None:
            try:
                rec_res = reconcile_fn()
                if not rec_res.success:
                    return f"❌ Breaker reset refused: startup reconciliation failed ({'; '.join(rec_res.issues)})."
            except Exception as e:
                return f"❌ Breaker reset refused: reconcile check threw error: {e}"
        elif trade_manager is not None and not getattr(trade_manager, "reconcile_done", False):
            return "❌ Breaker reset refused: startup reconciliation has not completed successfully."

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
            if hasattr(adaptive_controller, "transition_tier"):
                adaptive_controller.transition_tier("PAUSED", "Manual /force_pause command from operator")
            else:
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
