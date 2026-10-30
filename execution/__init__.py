"""Execution package (PROTECTED).

Orchestrates pre-trade verification, write-ahead logging, placement, and reconciliation.
The LLM cannot read, change, or bypass this layer.
"""

from __future__ import annotations

from execution.reconcile import ReconcileResult, reconcile_startup
from execution.state import SCHEMA_VERSION, StateManager
from execution.trade_manager import ExecutionResult, TradeManager

__all__ = [
    "TradeManager",
    "ExecutionResult",
    "StateManager",
    "SCHEMA_VERSION",
    "reconcile_startup",
    "ReconcileResult",
]
