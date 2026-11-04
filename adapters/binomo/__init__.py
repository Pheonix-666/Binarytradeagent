"""Binomo platform adapter package.

Strictly adheres to PROJECT_SPEC.md rules:
- Passive reading and session management.
- Zero clicks, zero fills, zero key presses, zero button presses.
- Credentials never stored, typed, or logged.
"""

from __future__ import annotations

from adapters.binomo.reader import BinomoReadOnlySource
from adapters.binomo.session import BinomoSession
from adapters.binomo.trader import (
    AmbiguousOrder,
    AssetSelectionMismatch,
    BinomoAdapter,
    ButtonLabelMismatch,
    ExpiryUnavailable,
    PanelBlocked,
    RealAccountDetected,
    SelectorResolutionError,
    StakeOutOfRange,
)
from adapters.binomo.ws_listener import BinomoWSListener

__all__ = [
    "BinomoSession",
    "BinomoWSListener",
    "BinomoReadOnlySource",
    "BinomoAdapter",
    "RealAccountDetected",
    "ExpiryUnavailable",
    "StakeOutOfRange",
    "PanelBlocked",
    "AmbiguousOrder",
    "SelectorResolutionError",
    "AssetSelectionMismatch",
    "ButtonLabelMismatch",
]

