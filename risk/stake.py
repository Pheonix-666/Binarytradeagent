"""Stake calculator — position sizing based on risk score and account equity.

Section 6.5 & Section 12 (Phase 4):
  - max_stake_pct(risk, min_pct, max_pct) = max_pct - (risk - 1) / 99 * (max_pct - min_pct)
    clamped to [min_pct, max_pct].
  - stake_amount(equity, pct, min_amount, round_to):
    computes dollar stake from equity and percentage.
    If the computed amount is below min_amount, returns None (meaning SKIP) rather
    than rounding up.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def max_stake_pct(
    risk: float,
    min_pct: float = 1.0,
    max_pct: float = 20.0,
) -> float:
    """Calculate maximum allowed stake percentage for a given risk score (1-100).

    Formula:
      max_pct - (risk - 1) / 99 * (max_pct - min_pct)
    Clamped to [min_pct, max_pct].
    """
    if min_pct >= max_pct:
        return float(min_pct)

    # Clamped risk to 1-100
    r = float(np.clip(risk, 1.0, 100.0))
    pct = max_pct - ((r - 1.0) / 99.0) * (max_pct - min_pct)
    return float(np.clip(pct, min_pct, max_pct))


def compute_equity(
    cash_balance: float,
    open_stakes: float = 0.0,
) -> float:
    """Calculate total account equity: cash balance + open stakes."""
    return max(0.0, float(cash_balance) + max(0.0, float(open_stakes)))


def stake_amount(
    equity: float,
    pct: float,
    min_amount: float = 1.0,
    round_to: float = 1.0,
    max_amount: Optional[float] = None,
) -> Optional[float]:
    """Calculate position stake amount from equity and percentage.

    Rules:
      - Raw cap amount = equity * (pct / 100), capped at max_amount if configured.
      - Round DOWN to nearest round_to (floor), so the rounded stake never exceeds
        the percentage cap or platform maximum.
      - If the floored amount is below min_amount, return None (SKIP) rather
        than rounding up.
    """
    if equity <= 0.0 or pct <= 0.0:
        return None

    raw_cap = equity * (pct / 100.0)
    if max_amount is not None and max_amount > 0:
        raw_cap = min(raw_cap, float(max_amount))

    if round_to > 0:
        # Determine decimal precision to avoid binary floating point representation artifacts
        s = f"{round_to:.8f}".rstrip("0")
        decimals = len(s.split(".")[1]) if "." in s else 2
        decimals = max(decimals, 2)

        # Floor to round_to increment
        steps = math.floor((raw_cap / round_to) + 1e-12)
        floored = round(steps * round_to, decimals)
        if floored > (raw_cap + 1e-12):
            floored = round((steps - 1) * round_to, decimals)
    else:
        floored = round(raw_cap, 2)

    if max_amount is not None and max_amount > 0:
        floored = min(floored, float(max_amount))

    # If the floored amount is below min_amount, return None (SKIP)
    if floored < (min_amount - 1e-9):
        return None

    return float(floored)
