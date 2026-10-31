"""Paper trading entry execution model.

Matches backtest engine logic:
  - entry_time = decision_time + timedelta(minutes=entry_delay_bars)
  - slippage = slippage_atr * atr
  - entry_price = base_price + slippage (CALL) or base_price - slippage (PUT)
  - exit_time = entry_time + timedelta(minutes=expiry_minutes)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Union

from core.models import TradeDirection


@dataclass(frozen=True)
class PaperExecutionDetails:
    """Calculated execution details for a simulated paper trade."""

    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    base_price: float
    slippage: float
    entry_price: float
    direction: TradeDirection
    expiry_minutes: int


class PaperEntryModel:
    """Simulates realistic execution conditions for paper trading."""

    def __init__(
        self,
        entry_delay_bars: int = 1,
        slippage_atr: float = 0.05,
    ) -> None:
        self.entry_delay_bars = max(0, int(entry_delay_bars))
        self.slippage_atr = max(0.0, float(slippage_atr))

    def calculate_entry(
        self,
        decision_time: datetime,
        base_price: float,
        direction: Union[TradeDirection, str],
        expiry_minutes: int,
        atr: Optional[float] = None,
    ) -> PaperExecutionDetails:
        """Calculate entry price and execution timings with slippage and delay."""
        if isinstance(direction, str):
            direction = TradeDirection(direction.upper())

        entry_time = decision_time + timedelta(minutes=self.entry_delay_bars)
        exit_time = entry_time + timedelta(minutes=expiry_minutes)

        effective_atr = atr if atr is not None and atr > 0 else 0.0
        slippage = self.slippage_atr * effective_atr

        # Slippage works against the trade:
        # CALL entries execute higher, PUT entries execute lower
        if direction == TradeDirection.CALL:
            entry_price = float(base_price + slippage)
        else:
            entry_price = float(base_price - slippage)

        return PaperExecutionDetails(
            decision_time=decision_time,
            entry_time=entry_time,
            exit_time=exit_time,
            base_price=float(base_price),
            slippage=float(slippage),
            entry_price=entry_price,
            direction=direction,
            expiry_minutes=int(expiry_minutes),
        )


def calculate_entry_price(
    open_price: float,
    direction: Union[TradeDirection, str],
    atr: Optional[float] = None,
    slippage_atr: float = 0.02,
) -> float:
    """Calculate entry price from bar open price and slippage (matches backtest logic)."""
    if isinstance(direction, str):
        direction = TradeDirection(direction.upper())
    effective_atr = atr if atr is not None and atr > 0 else 0.0
    slippage = float(slippage_atr) * effective_atr
    if direction == TradeDirection.CALL:
        return float(open_price + slippage)
    else:
        return float(open_price - slippage)

