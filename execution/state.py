"""Atomic state management with backup copy and schema versioning.

PROJECT_SPEC.md Section 9 & Phase 8:
  - Atomic writes: write to temp file, flush/fsync, and atomically rename.
  - Automatic .bak backup copy maintained.
  - Schema version embedded.
  - Adaptive state saved via to_dict() / restored via from_dict().
  - Restarting while PAUSED remains PAUSED.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

from adaptive.monitor import AdaptiveState
from safety.breaker import BreakerState

SCHEMA_VERSION: str = "1.0"


class StateManager:
    """Manages persistent execution runtime state."""

    def __init__(self, state_file: str | Path = "state.json"):
        self.state_path = Path(state_file)
        self.backup_path = self.state_path.with_suffix(".json.bak")
        self.data: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load state from file, falling back to backup if corrupted.

        Guarantees:
          If adaptive tier was PAUSED when persisted, it remains PAUSED.
        """
        target = None
        if self.state_path.is_file():
            target = self.state_path
        elif self.backup_path.is_file():
            target = self.backup_path

        if target is None:
            self.data = self._create_default_state()
            return self.data

        try:
            with open(target, "r", encoding="utf-8") as f:
                content = json.load(f)
            self.data = content
        except Exception:
            # Try backup if primary failed
            if target != self.backup_path and self.backup_path.is_file():
                try:
                    with open(self.backup_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    self.data = content
                    return self.data
                except Exception:
                    pass
            self.data = self._create_default_state()

        # Enforce PAUSED persistence guarantee
        ad_dict = self.data.get("adaptive_state")
        if ad_dict and isinstance(ad_dict, dict):
            if ad_dict.get("tier") == "PAUSED":
                ad_dict["real_trading_allowed"] = False
                ad_dict["stake_scale"] = 0.0

        return self.data

    def save(
        self,
        balance: float,
        equity: float,
        open_trades: list[Any],
        pending_trades: list[Any],
        breaker_state: Optional[BreakerState] = None,
        adaptive_state: Optional[AdaptiveState] = None,
        reconcile_done: bool = False,
        asset_modes: Optional[dict[str, str]] = None,
        daily_starting_balance: Optional[float] = None,
        today_loss: float = 0.0,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Atomically persist state to disk with backup copy."""
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()

        def _to_json_item(item: Any) -> Any:
            if hasattr(item, "to_dict"):
                return item.to_dict()
            if hasattr(item, "__dict__"):
                return {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
            if isinstance(item, dict):
                return item
            return str(item)

        serialized: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "last_updated": now_iso,
            "balance": round(float(balance), 2),
            "equity": round(float(equity), 2),
            "open_trades": [_to_json_item(t) for t in open_trades],
            "pending_trades": [_to_json_item(t) for t in pending_trades],
            "reconcile_done": bool(reconcile_done),
            "asset_modes": dict(asset_modes or {}),
            "daily_starting_balance": float(daily_starting_balance) if daily_starting_balance else float(balance),
            "today_loss": float(today_loss),
            "breaker_state": as_dict_or_raw(breaker_state),
            "adaptive_state": adaptive_state.to_dict() if adaptive_state else None,
        }
        if extra:
            serialized.update(extra)

        self.data = serialized

        # 1. Write backup if current valid file exists
        if self.state_path.is_file():
            try:
                shutil.copy2(self.state_path, self.backup_path)
            except Exception:
                pass

        # 2. Atomic write via temp file + atomic rename
        parent_dir = self.state_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            prefix="state_",
            suffix=".tmp",
            dir=str(parent_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            os.replace(temp_path, self.state_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise

    def get_adaptive_state(self) -> Optional[AdaptiveState]:
        """Restore AdaptiveState from loaded state dictionary."""
        ad_data = self.data.get("adaptive_state")
        if ad_data and isinstance(ad_data, dict):
            return AdaptiveState.from_dict(ad_data)
        return None

    def get_breaker_state(self) -> Optional[BreakerState]:
        """Restore BreakerState from loaded state dictionary."""
        b_data = self.data.get("breaker_state")
        if b_data and isinstance(b_data, dict):
            return BreakerState(**b_data)
        return None

    def _create_default_state(self) -> dict[str, Any]:
        from datetime import datetime, timezone

        return {
            "schema_version": SCHEMA_VERSION,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "balance": 10000.0,
            "equity": 10000.0,
            "open_trades": [],
            "pending_trades": [],
            "reconcile_done": False,
            "asset_modes": {},
            "daily_starting_balance": 10000.0,
            "today_loss": 0.0,
            "breaker_state": None,
            "adaptive_state": None,
        }


def as_dict_or_raw(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj
