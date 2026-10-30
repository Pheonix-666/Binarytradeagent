"""Kill switch — emergency stop for new trade executions.

PROJECT_SPEC.md Section 8 (Safety Layer):
  - The system halts new trades when a file named KILL exists in the project folder
    or a Telegram /stop command arrives.
  - Open trades are left to settle (never cancels open trades).
  - Pure file existence / flag check immediately prior to order placement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


from core.config import resolve_project_path


class KillSwitch:
    """Manages manual emergency stop via KILL file or Telegram flag."""

    def __init__(
        self,
        kill_file: str = "KILL",
        workspace_root: Optional[Path] = None,
    ):
        self.kill_file_name = kill_file
        self.workspace_root = Path(workspace_root) if workspace_root else resolve_project_path(".")
        self._telegram_stop_active: bool = False

    @property
    def kill_file_path(self) -> Path:
        return self.workspace_root / self.kill_file_name

    def is_file_present(self) -> bool:
        """Check if the KILL file exists on disk."""
        return self.kill_file_path.is_file()

    def set_telegram_stop(self, active: bool = True) -> None:
        """Set or clear the Telegram stop flag."""
        self._telegram_stop_active = active

    def is_telegram_stop_active(self) -> bool:
        """Check if the Telegram stop flag is set."""
        return self._telegram_stop_active

    def is_active(self) -> bool:
        """Check if kill switch is activated via either file or Telegram stop.

        Does NOT cancel any open trades; merely blocks initiation of new trades.
        """
        return self.is_file_present() or self._telegram_stop_active

    def activate_file(self) -> Path:
        """Create the KILL file to trigger emergency stop."""
        path = self.kill_file_path
        path.write_text("KILL SWITCH ACTIVATED", encoding="utf-8")
        return path

    def deactivate_file(self) -> bool:
        """Remove the KILL file."""
        path = self.kill_file_path
        if path.is_file():
            path.unlink()
            return True
        return False


def check_kill_switch(
    kill_file: str = "KILL",
    telegram_stop: bool = False,
    workspace_root: Optional[Path] = None,
) -> bool:
    """Convenience helper to check if kill switch is active."""
    root = workspace_root or Path.cwd()
    target_path = root / kill_file
    if target_path.is_file():
        return True
    return bool(telegram_stop)
