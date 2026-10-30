"""Atomic state management with backup copy, schema versioning, and fail-closed security.

PROJECT_SPEC.md Section 9 & Phase 8:
  - Atomic writes: write to temp file, flush/fsync, and atomically rename.
  - Automatic .bak backup copy maintained: JSON is validated BEFORE copying over .bak.
  - Schema version embedded. Unknown schema version fails closed.
  - Fail-closed: missing state after first run, unreadable JSON, or unknown schema
    starts with breaker tripped and adaptive PAUSED; require explicit manual command.
  - First run (first_run=True) adopts broker balance. No fictional 10000.0 defaults.
  - Paths anchored to absolute project root (not working directory).
  - update(**kwargs) merges into existing state, never defaulting missing fields to empty.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

from adaptive.monitor import AdaptiveState
from core.config import resolve_project_path
from safety.breaker import BreakerState

logger = logging.getLogger("trading_system.state")

SCHEMA_VERSION: str = "1.0"


def _to_json_item(item: Any) -> Any:
    """Recursively convert model, dict, or object into JSON-serializable structure."""
    if item is None:
        return None
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if hasattr(item, "__dict__"):
        return {k: _to_json_item(v) for k, v in item.__dict__.items() if not k.startswith("_")}
    if isinstance(item, dict):
        return {k: _to_json_item(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [_to_json_item(x) for x in item]
    return item


def as_dict_or_raw(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


class StateManager:
    """Manages persistent execution runtime state."""

    def __init__(
        self,
        state_file: str | Path = "state.json",
        first_run: bool = False,
        broker_balance: float = 0.0,
    ):
        self.state_path = resolve_project_path(state_file)
        self.backup_path = self.state_path.with_suffix(".json.bak")
        self.first_run = first_run
        self.broker_balance = broker_balance
        self.data: dict[str, Any] = {}

    def _create_fail_closed_state(self, reason: str, balance: float = 0.0) -> dict[str, Any]:
        """Create a fail-closed initial state with breaker tripped and adaptive PAUSED."""
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": SCHEMA_VERSION,
            "last_updated": now_iso,
            "balance": round(float(balance), 2),
            "equity": round(float(balance), 2),
            "open_trades": [],
            "pending_trades": [],
            "reconcile_done": False,
            "asset_modes": {},
            "daily_starting_balance": round(float(balance), 2),
            "today_loss": 0.0,
            "breaker_state": {
                "tripped": True,
                "trip_reason": reason,
                "tripped_at": now_iso,
                "consecutive_exec_errors": 0,
                "recent_order_timestamps": [],
            },
            "adaptive_state": {
                "tier": "PAUSED",
                "real_trading_allowed": False,
                "stake_scale": 0.0,
                "transition_reason": f"Fail-closed: {reason}",
            },
        }

    def _create_clean_state(self, balance: float = 0.0) -> dict[str, Any]:
        """Create a clean state for first run adopting broker balance."""
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": SCHEMA_VERSION,
            "last_updated": now_iso,
            "balance": round(float(balance), 2),
            "equity": round(float(balance), 2),
            "open_trades": [],
            "pending_trades": [],
            "reconcile_done": False,
            "asset_modes": {},
            "daily_starting_balance": round(float(balance), 2),
            "today_loss": 0.0,
            "breaker_state": None,
            "adaptive_state": None,
        }

    def load(
        self,
        first_run: Optional[bool] = None,
        broker_balance: Optional[float] = None,
    ) -> dict[str, Any]:
        """Load state from file, enforcing fail-closed guarantees."""
        is_first_run = self.first_run if first_run is None else first_run
        balance = float(broker_balance if broker_balance is not None else self.broker_balance)

        # 1. State file missing
        if not self.state_path.is_file():
            if not self.backup_path.is_file():
                if is_first_run:
                    # First run adopts broker balance
                    self.data = self._create_clean_state(balance=balance)
                    self._persist_to_disk(self.data)
                    return self.data
                # Missing after first run -> fail closed!
                msg = f"State file missing after first run at {self.state_path}"
                logger.error(msg)
                self.data = self._create_fail_closed_state(reason=msg, balance=balance)
                return self.data
            target = self.backup_path
        else:
            target = self.state_path

        # 2. Read and validate JSON
        content: Optional[dict[str, Any]] = None
        read_error: Optional[str] = None
        try:
            with open(target, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as exc:
            read_error = str(exc)

        # Fallback to backup if primary failed
        if content is None and target != self.backup_path and self.backup_path.is_file():
            try:
                with open(self.backup_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                read_error = None
            except Exception as exc:
                read_error = f"Primary failed ({read_error}), backup also failed ({exc})"

        if content is None:
            msg = f"State file unreadable or corrupted at {target}: {read_error}"
            logger.error(msg)
            self.data = self._create_fail_closed_state(reason=msg, balance=balance)
            return self.data

        # 3. Schema version verification
        file_version = content.get("schema_version")
        if file_version != SCHEMA_VERSION:
            msg = f"Unknown state schema version '{file_version}' (expected '{SCHEMA_VERSION}')"
            logger.error(msg)
            self.data = self._create_fail_closed_state(reason=msg, balance=balance)
            return self.data

        self.data = content

        # Enforce PAUSED persistence guarantee
        ad_dict = self.data.get("adaptive_state")
        if ad_dict and isinstance(ad_dict, dict):
            if ad_dict.get("tier") == "PAUSED":
                ad_dict["real_trading_allowed"] = False
                ad_dict["stake_scale"] = 0.0

        return self.data

    def update(self, **kwargs: Any) -> dict[str, Any]:
        """Merge updates into existing state without defaulting missing fields, and atomically persist."""
        if not self.data:
            self.data = self._create_clean_state(balance=kwargs.get("balance", 0.0))

        # Merge updates into self.data
        for k, v in kwargs.items():
            if k == "adaptive_state":
                self.data["adaptive_state"] = v.to_dict() if hasattr(v, "to_dict") else as_dict_or_raw(v)
            elif k == "breaker_state":
                self.data["breaker_state"] = as_dict_or_raw(v)
            elif k in ("open_trades", "pending_trades"):
                self.data[k] = [_to_json_item(t) for t in (v or [])]
            elif k in ("balance", "equity", "daily_starting_balance", "today_loss"):
                if v is not None:
                    self.data[k] = round(float(v), 2)
            elif k == "asset_modes":
                if v is not None:
                    # Merge or replace asset modes
                    current = dict(self.data.get("asset_modes", {}))
                    current.update(dict(v))
                    self.data["asset_modes"] = current
            elif k == "extra" and isinstance(v, dict):
                self.data.update(v)
            else:
                self.data[k] = v

        now_iso = datetime.now(timezone.utc).isoformat()
        self.data["last_updated"] = now_iso
        self.data["schema_version"] = SCHEMA_VERSION

        self._persist_to_disk(self.data)
        return self.data

    def save(self, **kwargs: Any) -> None:
        """Compatibility wrapper delegating to update()."""
        self.update(**kwargs)

    def _persist_to_disk(self, data_to_write: dict[str, Any]) -> None:
        """Atomically persist state to disk, validating JSON before copying to backup."""
        # 1. Validate existing state file before copying over .bak
        if self.state_path.is_file():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    json.load(f)  # Must be valid JSON
                shutil.copy2(self.state_path, self.backup_path)
            except Exception as e:
                logger.warning("Existing state file invalid or copy failed, skipping backup: %s", e)

        # 2. Atomic write via temp file in same directory + replace
        parent_dir = self.state_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            prefix="state_",
            suffix=".tmp",
            dir=str(parent_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data_to_write, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

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
