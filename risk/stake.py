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
    min_pct: float = 2.0,
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
) -> Optional[float]:
    """Calculate position stake amount from equity and percentage.

    Rules:
      - Raw amount = equity * (pct / 100).
      - If raw amount < min_amount, return None (SKIP) rather than rounding up.
      - Round to nearest round_to increment.
      - If rounded amount < min_amount, return None (SKIP).
    """
    if equity <= 0.0 or pct <= 0.0:
        return None

    raw_amount = equity * (pct / 100.0)

    # Strict guard: do not round up if below minimum
    if raw_amount < min_amount:
        return None

    if round_to > 0:
        multiplier = 1.0 / round_to
        rounded = round(raw_amount * multiplier) / multiplier
        # Avoid float precision artifacts (e.g. 1.000000000004)
        if round_to >= 1.0:
            rounded = round(rounded, 2)
        else:
            decimals = int(math.ceil(-math.log10(round_to)))
            rounded = round(rounded, decimals)
    else:
        rounded = round(raw_amount, 2)

    if rounded < min_amount:
        return None

    return float(rounded)
