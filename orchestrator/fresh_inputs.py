"""Fresh inputs builder for TradeManager and Safety Limits.

Section 5:
  - AccountSnapshot built fresh from adapter.get_balance() and adapter.get_open_trades().
  - asset_modes using selection.stats.asset_mode.
  - today_loss and daily_starting_balance calculated from TradeRepository.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Optional, Sequence

from core.config import AppConfig
from data.store import TradeRecord, TradeRepository
from safety.limits import AccountSnapshot
from selection.stats import asset_mode, compute_asset_stats

logger = logging.getLogger("trading_system.fresh_inputs")


def build_account_snapshot(
    adapter: Any,
    repository: TradeRepository,
    config: AppConfig,
    now: datetime,
    assets: Sequence[str],
    daily_starting_balance: Optional[float] = None,
) -> AccountSnapshot:
    """Construct an AccountSnapshot with fresh financial, mode, and loss metrics."""
    balance = float(adapter.get_balance())
    open_trades = list(adapter.get_open_trades())
    equity = balance

    all_trades = repository.get_trades()
    asset_modes_dict: dict[str, str] = {}
    for asset in assets:
        stats = compute_asset_stats(
            trades=all_trades,
            asset=asset,
            mode="paper",
            as_of=now,
        )
        if stats is None:
            asset_modes_dict[asset] = "paper"
        else:
            mode_str, _ = asset_mode(stats, config)
            asset_modes_dict[asset] = mode_str

    # Compute today_loss and daily_starting_balance strictly from TradeRepository
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    today_trades = repository.get_trades(since=today_start, until=now)
    today_pnl = sum(
        float(t.pnl) for t in today_trades if t.result in {"WIN", "LOSS", "TIE"}
    )
    today_loss = max(0.0, -today_pnl)

    if daily_starting_balance is None or daily_starting_balance <= 0.0:
        daily_starting_balance = max(1.0, balance - today_pnl)

    return AccountSnapshot(
        balance=balance,
        equity=equity,
        open_trades=open_trades,
        asset_modes=asset_modes_dict,
        daily_starting_balance=float(daily_starting_balance),
        today_loss=float(today_loss),
    )

