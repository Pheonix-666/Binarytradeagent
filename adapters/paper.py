"""Paper trading adapter — simulated trade execution against real or recorded market data."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from adapters.base import PlatformAdapter
from core.clock import now_utc
from core.models import AssetQuote, Trade, TradeDirection, TradeReceipt, TradeResult
from data.store import DataStore


class PaperAdapter(PlatformAdapter):
    """Simulates fixed-time trading on paper without any live platform clicks."""

    def __init__(
        self,
        initial_balance: float = 1000.0,
        store: Optional[DataStore] = None,
        default_payout: float = 0.85,
        tie_policy: str = "loss",
    ):
        self.balance = float(initial_balance)
        self.initial_balance = float(initial_balance)
        self.store = store
        self.default_payout = default_payout
        self.tie_policy = tie_policy
        self._assets: dict[str, AssetQuote] = {}
        self._candles: dict[str, dict[str, pd.DataFrame]] = {}  # {asset: {tf: df}}
        self._open_trades: list[Trade] = []
        self._trade_history: list[Trade] = []
        self._healthy: bool = True
        self._current_time: Optional[datetime] = None

    def set_assets(self, assets: list[AssetQuote]) -> None:
        """Register or update active assets in memory."""
        for a in assets:
            self._assets[a.asset_id] = a

    def set_candles(self, asset_id: str, timeframe: str, df: pd.DataFrame) -> None:
        """Set in-memory closed candles for an asset and timeframe."""
        if asset_id not in self._candles:
            self._candles[asset_id] = {}
        self._candles[asset_id][timeframe] = df.sort_values("ts").reset_index(drop=True)

    def set_price(self, asset_id: str, price: float) -> None:
        """Set an explicit current price for an asset."""
        now = self._current_time or now_utc()
        df = pd.DataFrame([{
            "ts": now,
            "ts_close": now + timedelta(minutes=1),
            "open": float(price),
            "high": float(price),
            "low": float(price),
            "close": float(price),
            "volume": 1.0,
        }])
        self.set_candles(asset_id, "1m", df)

    def set_current_time(self, dt: datetime) -> None:
        """Set simulated clock time without settling trades."""
        if dt.tzinfo is None:
            self._current_time = dt.replace(tzinfo=timezone.utc)
        else:
            self._current_time = dt.astimezone(timezone.utc)

    def set_simulated_time(self, dt: datetime) -> None:
        """Advance or set simulated time for backtesting/paper checks."""
        self.set_current_time(dt)
        self.settle_open_trades()

    def get_assets(self) -> list[AssetQuote]:
        """Return available assets from memory or fallback to store."""
        if self._assets:
            return list(self._assets.values())
        return []

    def get_candles(self, asset_id: str, timeframe: str, n: int) -> pd.DataFrame:
        """Return latest n closed candles for asset_id and timeframe."""
        # Check in-memory candles first
        if asset_id in self._candles and timeframe in self._candles[asset_id]:
            df = self._candles[asset_id][timeframe]
            if self._current_time is not None:
                df = df[df["ts"] <= self._current_time]
            return df.tail(n).reset_index(drop=True)

        # Fallback to store if configured
        if self.store is not None:
            return self.store.get_candles(asset_id, timeframe, n=n)

        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    def get_balance(self) -> float:
        """Return current simulated account balance."""
        return self.balance

    def _get_current_price(self, asset_id: str) -> float:
        """Look up the latest known price for asset_id."""
        for tf in ["1m", "5m", "15m", "1h", "1d"]:
            df = self.get_candles(asset_id, tf, n=1)
            if not df.empty:
                return float(df.iloc[-1]["close"])
        return 1.0  # Fallback dummy price if no candles present

    def _get_payout(self, asset_id: str) -> float:
        """Look up current payout for asset_id."""
        if asset_id in self._assets:
            return self._assets[asset_id].payout
        if self.store is not None:
            p = self.store.get_latest_payout(asset_id)
            if p is not None:
                return p
        return self.default_payout

    def place_trade(
        self,
        asset_id: str,
        direction: TradeDirection | str,
        expiry_min: int,
        stake_amount: float,
        payout: float,
        trade_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TradeReceipt:
        """Execute a simulated fixed-time trade."""
        if payout is None or payout <= 0:
            raise ValueError(f"payout must be positive, got {payout}")
        if stake_amount <= 0:
            raise ValueError(f"stake_amount must be positive, got {stake_amount}")
        if stake_amount > self.balance:
            raise ValueError(
                f"Insufficient balance: stake={stake_amount} > balance={self.balance}"
            )

        if isinstance(direction, str):
            direction = TradeDirection(direction.upper())

        entry_time = self._current_time or now_utc()
        entry_price = self._get_current_price(asset_id)
        effective_payout = float(payout)
        tid = trade_id or f"paper_{uuid.uuid4().hex[:10]}"

        # Deduct stake from available balance
        self.balance -= stake_amount

        receipt = TradeReceipt(
            trade_id=tid,
            asset_id=asset_id,
            direction=direction,
            expiry_minutes=expiry_min,
            stake_amount=stake_amount,
            entry_price=entry_price,
            entry_time=entry_time,
            payout=effective_payout,
            platform_trade_id=tid,
        )

        open_trade = Trade(
            **receipt.model_dump(),
            result=TradeResult.PENDING,
        )
        self._open_trades.append(open_trade)

        return receipt

    def get_open_trades(self) -> list[Trade]:
        """Return currently active trades after checking for settlements."""
        self.settle_open_trades()
        return list(self._open_trades)

    def get_trade_history(self, since: Optional[datetime] = None) -> list[Trade]:
        """Return settled trades."""
        if since is None:
            return list(self._trade_history)

        effective_since = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        return [
            t
            for t in self._trade_history
            if t.exit_time is not None and t.exit_time >= effective_since
        ]

    def settle_open_trades(
        self,
        current_time: Optional[datetime] = None,
        prices_at_expiry: Optional[dict[str, float]] = None,
    ) -> list[Trade]:
        """Settle any open trades that have reached or passed their expiry time.

        Settlement rules:
          - CALL wins if exit_price > entry_price
          - PUT wins if exit_price < entry_price
          - On WIN: balance += stake * (1 + payout)
          - On TIE (exit_price == entry_price): balance += stake (refund)
          - On LOSS: no balance addition (stake was deducted at entry)
        """
        now = current_time or self._current_time or now_utc()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        still_open: list[Trade] = []
        settled: list[Trade] = []

        for trade in self._open_trades:
            expiry_time = trade.entry_time + timedelta(minutes=trade.expiry_minutes)
            if now >= expiry_time:
                # Find exit price
                asset = getattr(trade, "asset_id", getattr(trade, "asset", ""))
                stake = float(getattr(trade, "stake_amount", getattr(trade, "stake", 0.0)))
                exit_price = None
                if prices_at_expiry and asset in prices_at_expiry:
                    exit_price = prices_at_expiry[asset]
                else:
                    exit_price = self._get_current_price(asset)

                # Determine outcome
                if trade.direction == TradeDirection.CALL:
                    if exit_price > trade.entry_price:
                        outcome = TradeResult.WIN
                    elif exit_price < trade.entry_price:
                        outcome = TradeResult.LOSS
                    else:
                        outcome = TradeResult.TIE
                else:  # PUT
                    if exit_price < trade.entry_price:
                        outcome = TradeResult.WIN
                    elif exit_price > trade.entry_price:
                        outcome = TradeResult.LOSS
                    else:
                        outcome = TradeResult.TIE

                # Payout computation
                payout_rate = trade.payout
                if outcome == TradeResult.WIN:
                    payout_received = stake * (1.0 + payout_rate)
                    self.balance += payout_received
                elif outcome == TradeResult.TIE:
                    if self.tie_policy == "refund":
                        payout_received = stake
                        self.balance += payout_received
                    else:
                        payout_received = 0.0
                else:
                    payout_received = 0.0

                settled_trade = Trade(
                    trade_id=trade.trade_id,
                    asset_id=asset,
                    direction=trade.direction,
                    expiry_minutes=trade.expiry_minutes,
                    stake_amount=stake,
                    entry_price=trade.entry_price,
                    entry_time=trade.entry_time,
                    payout=trade.payout,
                    result=outcome,
                    exit_price=exit_price,
                    exit_time=expiry_time,
                    payout_received=payout_received,
                )
                self._trade_history.append(settled_trade)
                settled.append(settled_trade)
            else:
                still_open.append(trade)

        self._open_trades = still_open
        return settled

    def is_healthy(self) -> bool:
        """Return operational health status."""
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        """Control health status for testing error handlers."""
        self._healthy = healthy
