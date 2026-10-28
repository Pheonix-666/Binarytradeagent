"""Base adapter — abstract interface for all platform adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.models import AssetQuote, TradeDirection, TradeReceipt


class BaseAdapter(ABC):
    """Abstract base class that every platform adapter must implement."""

    @abstractmethod
    async def get_assets(self) -> list[AssetQuote]:
        """Return available assets with current quotes."""
        ...

    @abstractmethod
    async def place_trade(
        self,
        asset_id: str,
        direction: TradeDirection,
        amount: float,
        expiry_minutes: int,
    ) -> TradeReceipt:
        """Place a fixed-time trade and return a receipt."""
        ...

    @abstractmethod
    async def get_balance(self) -> float:
        """Return the current account balance."""
        ...

    @abstractmethod
    async def get_price_history(
        self,
        asset_id: str,
        interval_seconds: int,
        count: int,
    ) -> list[dict]:
        """Return recent OHLCV candles as list of dicts."""
        ...
