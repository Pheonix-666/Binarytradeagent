"""Safety package (PROTECTED).

Provides hard validation limits, malfunction circuit breaker, and emergency kill switch.
The LLM cannot read, change, or bypass this layer.
"""

from __future__ import annotations

from safety.breaker import BreakerState, CircuitBreaker
from safety.killswitch import KillSwitch, check_kill_switch
from safety.limits import (
    AccountSnapshot,
    Allow,
    Block,
    OrderCheckResult,
    OrderRequest,
    check_order,
)

__all__ = [
    "check_order",
    "Allow",
    "Block",
    "OrderRequest",
    "AccountSnapshot",
    "OrderCheckResult",
    "CircuitBreaker",
    "BreakerState",
    "KillSwitch",
    "check_kill_switch",
]
