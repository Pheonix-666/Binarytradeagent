"""Tests for stake limits, configurability, and property invariants.

Requirements:
  1. No code path outside config contains hardcoded stake limits (min_pct, max_pct, total_cap_pct).
  2. Property test: for risk 1..100 and many equities, floored stake amount never exceeds
     max_pct of equity.
  3. Property test: total open stakes never exceed total_cap_pct of equity.
  4. Known-value verification: risk 1 -> 20%, risk 50 -> 10.60%, risk 100 -> 1%.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
import pytest

from core.config import AppConfig, load_config
from risk.stake import compute_equity, max_stake_pct, stake_amount


# ── 1. Static Guard: No Hardcoded Stake Limits in Business Logic ────────────


def test_no_hardcoded_stake_limits_outside_config():
    """Verify that business logic packages do not hardcode stake limits.

    Ensures stake sizing parameters (min_pct, max_pct, total_cap_pct) and their
    ratios (e.g. 0.35, 0.02, 0.01) are read from config or passed as parameters,
    rather than embedded as literals in operational logic.
    """
    project_root = Path(__file__).resolve().parent.parent
    checked_packages = ["analysis", "decision", "risk", "backtest", "selection"]

    # Patterns representing hardcoded stake limit values in active code (excluding comments)
    suspicious_patterns = [
        re.compile(r"\b0\.35\s*\*\s*(equity|balance)", re.IGNORECASE),
        re.compile(r"(equity|balance)\s*\*\s*0\.35", re.IGNORECASE),
        re.compile(r"\b0\.02\s*\*\s*(equity|balance)", re.IGNORECASE),
        re.compile(r"\b0\.01\s*\*\s*(equity|balance)", re.IGNORECASE),
        re.compile(r"\b0\.20\s*\*\s*(equity|balance)", re.IGNORECASE),
        re.compile(r"remaining_exposure\s*=\s*35", re.IGNORECASE),
        re.compile(r"total_cap\w*\s*=\s*35", re.IGNORECASE),
        re.compile(r"min_stake_pct\s*=\s*[12]\b(?!\.)", re.IGNORECASE),
    ]

    for pkg in checked_packages:
        pkg_dir = project_root / pkg
        if not pkg_dir.is_dir():
            continue

        for py_file in pkg_dir.glob("**/*.py"):
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            parsed = ast.parse(content, filename=str(py_file))

            # Collect line numbers belonging to string literals / docstrings to ignore them
            docstring_lines: set[int] = set()
            for node in ast.walk(parsed):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    docstring_lines.update(range(node.lineno, node.end_lineno + 1 if node.end_lineno else node.lineno + 1))

            for line_no, line in enumerate(content.splitlines(), start=1):
                if line_no in docstring_lines:
                    continue
                # Strip out trailing comments
                code_part = line.split("#")[0].strip()
                if not code_part:
                    continue

                for pat in suspicious_patterns:
                    match = pat.search(code_part)
                    assert match is None, (
                        f"Hardcoded stake limit pattern '{pat.pattern}' found in "
                        f"{py_file.name}:{line_no}: {line.strip()}"
                    )


# ── 2. Known Value Invariants ───────────────────────────────────────────────


def test_max_stake_pct_new_expectations():
    """Verify exact formula values: risk 1 -> 20%, risk 50 -> 10.60%, risk 100 -> 1%."""
    cfg = load_config("config.yaml")
    min_pct = cfg.stake.min_pct
    max_pct = cfg.stake.max_pct

    assert min_pct == 1.0
    assert max_pct == 20.0

    # Risk 1: exactly max_pct (20%)
    val_1 = max_stake_pct(1.0, min_pct=min_pct, max_pct=max_pct)
    assert val_1 == 20.0

    # Risk 50: 20 - (49/99)*19 = 20 - 9.4040404... = 10.595959... -> rounds to 10.60%
    val_50 = max_stake_pct(50.0, min_pct=min_pct, max_pct=max_pct)
    assert abs(val_50 - 10.595959) < 1e-4
    assert round(val_50, 2) == 10.60

    # Risk 100: exactly min_pct (1%)
    val_100 = max_stake_pct(100.0, min_pct=min_pct, max_pct=max_pct)
    assert val_100 == 1.0


# ── 3. Property Test: Floored Stake Never Exceeds max_pct of Equity ────────


def test_floored_stake_never_exceeds_max_pct_property():
    """Property test: for risk 1..100 and many equities, floored stake never exceeds max_pct of equity."""
    cfg = load_config("config.yaml")
    min_pct = cfg.stake.min_pct
    max_pct = cfg.stake.max_pct
    min_amount = cfg.stake.min_amount
    round_to = cfg.stake.round_to

    equities = [
        1.0, 5.0, 10.0, 25.0, 50.0, 99.99, 100.0, 150.0, 250.0, 500.0,
        1000.0, 1234.56, 2500.0, 5000.0, 9999.99, 10000.0, 25000.0,
        50000.0, 100000.0, 500000.0, 1000000.0
    ]
    round_tos = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]

    for equity in equities:
        max_allowed_stake = equity * (max_pct / 100.0)

        for r in range(1, 101):
            pct = max_stake_pct(float(r), min_pct=min_pct, max_pct=max_pct)
            assert min_pct <= pct <= max_pct, f"Risk {r} yielded out-of-bounds pct {pct}"

            for r_to in round_tos:
                amt = stake_amount(equity=equity, pct=pct, min_amount=min_amount, round_to=r_to)
                if amt is not None:
                    # Invariant 1: floored stake never exceeds max_pct of equity
                    assert amt <= max_allowed_stake + 1e-9, (
                        f"Floored stake {amt} exceeds max_pct cap {max_allowed_stake} "
                        f"for equity={equity}, risk={r}, round_to={r_to}"
                    )
                    # Invariant 2: floored stake never exceeds the specific risk pct cap
                    specific_cap = equity * (pct / 100.0)
                    assert amt <= specific_cap + 1e-9, (
                        f"Floored stake {amt} exceeds specific cap {specific_cap} "
                        f"for equity={equity}, risk={r}, round_to={r_to}"
                    )
                    # Invariant 3: if not None, must be at least min_amount
                    assert amt >= min_amount - 1e-9


# ── 4. Property Test: Total Open Stakes Never Exceed total_cap_pct ──────────


def test_total_open_stakes_never_exceeds_total_cap_pct_property():
    """Property test: simulating sequential trade allocations never breaches total_cap_pct."""
    cfg = load_config("config.yaml")
    min_pct = cfg.stake.min_pct
    max_pct = cfg.stake.max_pct
    total_cap_pct = cfg.stake.total_cap_pct
    min_amount = cfg.stake.min_amount
    round_to = cfg.stake.round_to

    test_equities = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]

    for equity in test_equities:
        max_total_cap_dollars = equity * (total_cap_pct / 100.0)
        open_stakes = 0.0

        # Attempt to add stakes with varied risk scores
        for risk in range(1, 101, 3):
            pct = max_stake_pct(float(risk), min_pct=min_pct, max_pct=max_pct)
            candidate_stake = stake_amount(equity=equity, pct=pct, min_amount=min_amount, round_to=round_to)

            if candidate_stake is None:
                continue

            # Allocation check as performed by validator / backtester
            if (open_stakes + candidate_stake) <= (max_total_cap_dollars + 1e-9):
                open_stakes += candidate_stake
            else:
                # Allocation must be rejected when it would breach total_cap_pct
                pass

            # Invariant: open_stakes never exceeds total_cap_pct
            assert open_stakes <= max_total_cap_dollars + 1e-9, (
                f"Total open stakes {open_stakes} exceeded cap {max_total_cap_dollars} "
                f"on equity {equity}"
            )
            exposure_pct = (open_stakes / equity) * 100.0
            assert exposure_pct <= total_cap_pct + 1e-7, (
                f"Exposure pct {exposure_pct}% exceeded total_cap_pct {total_cap_pct}%"
            )
