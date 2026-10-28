"""Platform adapter interface.

All platform access goes through this interface. Nothing else in the system
knows which adapter is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from core.models import AssetQuote, Trade, TradeDirection, TradeReceipt


class PlatformAdapter(ABC):
    """Abstract interface for all platform adapters (Paper, Playwright, etc.)."""

    @abstractmethod
    def get_assets(self) -> list[AssetQuote]:
        """Return all assets currently available on the platform."""
        ...

    @abstractmethod
    def get_candles(self, asset_id: str, timeframe: str, n: int) -> pd.DataFrame:
        """Return the latest n closed candles for asset_id and timeframe.

        DataFrame columns must be: ['ts', 'open', 'high', 'low', 'close']
        ts must be datetime or int timestamp, sorted ascending.
        """
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Return the current account balance."""
        ...

    @abstractmethod
    def place_trade(
        self,
        asset_id: str,
        direction: TradeDirection,
        expiry_min: int,
        stake_amount: float,
    ) -> TradeReceipt:
        """Place a fixed-time trade on the platform."""
        ...

    @abstractmethod
    def get_open_trades(self) -> list[Trade]:
        """Return all currently open/active trades."""
        ...

    @abstractmethod
    def get_trade_history(self, since: Optional[datetime] = None) -> list[Trade]:
        """Return completed/settled trades, optionally since a given timestamp."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the connection/adapter is healthy and ready to trade."""
        ...


# Alias for backwards compatibility
BaseAdapter = PlatformAdapter
