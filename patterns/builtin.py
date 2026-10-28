"""Built-in pattern templates generated analytically in pure code.

Generates:
  1. double_bottom (reversal, up)
  2. double_top (reversal, down)
  3. support_bounce (continuation, up)
  4. resistance_bounce (continuation, down)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from patterns.similarity import z_normalize


@dataclass
class PatternTemplate:
    """Metadata and normalized array for a chart pattern template."""

    name: str
    direction: str  # "up" or "down"
    pattern_type: str  # "continuation" or "reversal"
    data: np.ndarray  # z-normalized 1D array
    timeframe_hint: Optional[str] = None


def generate_double_bottom(num_points: int = 64) -> PatternTemplate:
    """Generate canonical Double Bottom pattern (reversal, up).

    Features two symmetric troughs at x ≈ 0.25 and x ≈ 0.75, a central neckline
    peak at x ≈ 0.5, and an upward breakout at the right side.
    """
    x = np.linspace(0.0, 1.0, num_points)
    # Two inverted Gaussians with an upward trend bias towards the right
    trough1 = -1.5 * np.exp(-(((x - 0.25) / 0.12) ** 2))
    trough2 = -1.5 * np.exp(-(((x - 0.75) / 0.12) ** 2))
    trend = 1.0 * (x - 0.5)
    raw = trough1 + trough2 + trend
    normed, _ = z_normalize(raw)

    return PatternTemplate(
        name="double_bottom",
        direction="up",
        pattern_type="reversal",
        data=normed,
        timeframe_hint="5m",
    )


def generate_double_top(num_points: int = 64) -> PatternTemplate:
    """Generate canonical Double Top pattern (reversal, down).

    Exact inverse of double bottom: two peaks at x ≈ 0.25 and x ≈ 0.75,
    central trough, and a downward breakdown.
    """
    db = generate_double_bottom(num_points)
    normed, _ = z_normalize(-db.data)

    return PatternTemplate(
        name="double_top",
        direction="down",
        pattern_type="reversal",
        data=normed,
        timeframe_hint="5m",
    )


def generate_support_bounce(num_points: int = 64) -> PatternTemplate:
    """Generate Support Bounce pattern (continuation, up).

    Declines into a support floor around x ≈ 0.35, consolidates, then breaks
    out into a steady upward trend.
    """
    x = np.linspace(0.0, 1.0, num_points)
    # Descent into base, then strong parabolic upward continuation
    raw = 1.2 * np.exp(-x / 0.2) + 2.2 * (x**2)
    normed, _ = z_normalize(raw)

    return PatternTemplate(
        name="support_bounce",
        direction="up",
        pattern_type="continuation",
        data=normed,
        timeframe_hint="15m",
    )


def generate_resistance_bounce(num_points: int = 64) -> PatternTemplate:
    """Generate Resistance Bounce pattern (continuation, down).

    Exact inverse of support bounce: rallies into resistance ceiling,
    then rolls over into a sharp downward continuation.
    """
    sb = generate_support_bounce(num_points)
    normed, _ = z_normalize(-sb.data)

    return PatternTemplate(
        name="resistance_bounce",
        direction="down",
        pattern_type="continuation",
        data=normed,
        timeframe_hint="15m",
    )


def get_builtin_templates(num_points: int = 64) -> list[PatternTemplate]:
    """Return all built-in pattern templates."""
    return [
        generate_double_bottom(num_points),
        generate_double_top(num_points),
        generate_support_bounce(num_points),
        generate_resistance_bounce(num_points),
    ]
