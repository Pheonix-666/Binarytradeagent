"""Pure validation engine for LLM trade decisions.

PROJECT_SPEC.md Section 7.4 & Phase 7:
  - validate_decision() pure function (no network/disk I/O).
  - All 8 checks (a through h) must pass for CALL/PUT or result is SKIP:
      a) Schema valid.
      b) Decision in {CALL, PUT, SKIP}; SKIP passes through immediately.
      c) Asset, direction, and expiry match exactly one candidate in the report.
      d) Stake pct within [min_stake_pct, candidate.max_stake_pct] and keeps
         floored stake amount within total_cap_pct exposure cap.
      e) Every number in evidence matches report (float tolerance from config, exact for int/enums).
      f) Decision age <= max_decision_age_seconds, and current price has not moved
         against signal by > max_price_move_atr.
      g) No open trade in the opposite direction on the same asset.
      h) Trading allowed: kill switch inactive, breaker not tripped, reconcile completed.
  - Returns ValidatedDecision carrying final stake_amount and list of checks passed.
  - Any exception inside validator results in SKIP.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union
from pydantic import ValidationError

from core.config import AppConfig
from decision.schema import DecisionOutput, DecisionReport, SystemState, ValidatedDecision
from risk.stake import stake_amount


def validate_decision(
    parsed: Union[dict[str, Any], DecisionOutput],
    report: DecisionReport,
    state: SystemState,
    config: AppConfig,
    now: datetime,
) -> ValidatedDecision:
    """Validate an LLM decision against market report, risk limits, and safety state.

    Pure function: no network or disk I/O.
    """
    passed_checks: list[str] = []

    try:
        # Check a: Schema validity
        if isinstance(parsed, dict):
            try:
                decision_out = DecisionOutput.model_validate(parsed)
            except ValidationError as exc:
                return ValidatedDecision(
                    decision="SKIP",
                    is_valid=False,
                    skip_reason=f"schema_invalid: {exc.errors()[0]['msg']}",
                    checks_passed=[],
                )
        elif isinstance(parsed, DecisionOutput):
            decision_out = parsed
        else:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"schema_invalid: unexpected parsed type {type(parsed).__name__}",
                checks_passed=[],
            )

        passed_checks.append("check_a_schema_valid")

        # Check b: Decision in {CALL, PUT, SKIP}
        if decision_out.decision not in ("CALL", "PUT", "SKIP"):
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"invalid_decision_type: {decision_out.decision}",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # SKIP always passes through immediately
        if decision_out.decision == "SKIP":
            passed_checks.append("check_b_skip_passthrough")
            return ValidatedDecision(
                decision="SKIP",
                is_valid=True,
                reasoning=decision_out.reasoning,
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        passed_checks.append("check_b_decision_in_set")

        # Check c: Asset, direction, and expiry match exactly one candidate in the report
        asset = decision_out.asset
        expiry = decision_out.expiry_minutes
        direction = decision_out.decision

        if not asset or expiry is None:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason="missing_asset_or_expiry_for_trade",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        matching = [
            c for c in report.candidates
            if c.asset == asset and c.direction == direction and c.expiry_minutes == expiry
        ]

        if len(matching) == 0:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"candidate_not_found: no match for {asset} {direction} {expiry}m",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )
        elif len(matching) > 1:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"ambiguous_candidates: multiple matches for {asset} {direction} {expiry}m",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        matched_cand = matching[0]
        passed_checks.append("check_c_candidate_matched")

        # Check d: Stake sizing and exposure cap
        stake_pct = decision_out.stake_pct
        if stake_pct is None:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason="missing_stake_pct",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Tolerance for minor floating-point rounding
        if stake_pct < (report.min_stake_pct - 1e-4):
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"stake_below_min_pct: {stake_pct} < min {report.min_stake_pct}",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Tier-scaled maximum stake percentage check
        stake_scale = 1.0
        if report.adaptive_state is not None:
            stake_scale = float(report.adaptive_state.stake_scale)
        elif getattr(state, "adaptive_state", None) is not None:
            stake_scale = float(getattr(state.adaptive_state, "stake_scale", 1.0))

        tier_scaled_max = round(matched_cand.max_stake_pct * stake_scale, 2)
        if stake_pct > (tier_scaled_max + 1e-4):
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"stake_above_max_pct: {stake_pct} > tier-scaled max {tier_scaled_max} (scale={stake_scale})",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Floor stake amount to config.stake.round_to
        equity = float(report.equity)
        computed_stake = stake_amount(
            equity=equity,
            pct=stake_pct,
            min_amount=config.stake.min_amount,
            round_to=config.stake.round_to,
            max_amount=config.stake.max_amount,
        )

        if computed_stake is None:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"stake_below_min_amount: floored amount below {config.stake.min_amount}",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Enforce portfolio exposure cap with floored stake
        current_open_stakes = sum(float(getattr(ot, "stake", 0.0)) for ot in state.open_trades)
        max_allowed_exposure = (config.stake.total_cap_pct / 100.0) * equity
        if (current_open_stakes + computed_stake) > (max_allowed_exposure + 1e-4):
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=(
                    f"exposure_cap_breached: {current_open_stakes + computed_stake:.2f} > "
                    f"max exposure {max_allowed_exposure:.2f}"
                ),
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        passed_checks.append("check_d_stake_and_exposure")

        # Check e: Evidence consistency
        ev = decision_out.evidence
        if ev is None:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason="missing_evidence",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Exact integer match for risk_score
        if ev.risk_score != matched_cand.risk_score:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"misquoted_risk_score: {ev.risk_score} != report {matched_cand.risk_score}",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Float tolerance from config for similarity and payout
        float_tol = getattr(config.llm, "evidence_float_tolerance", 0.01)
        if abs(ev.similarity - matched_cand.pattern.similarity) > float_tol:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=(
                    f"misquoted_similarity: {ev.similarity} != report {matched_cand.pattern.similarity} "
                    f"(tol {float_tol})"
                ),
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        if abs(ev.payout - matched_cand.payout) > float_tol:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=(
                    f"misquoted_payout: {ev.payout} != report {matched_cand.payout} "
                    f"(tol {float_tol})"
                ),
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Exact string match for trends
        if ev.trend_daily.lower() != matched_cand.trend.daily.lower():
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"misquoted_trend_daily: {ev.trend_daily} != report {matched_cand.trend.daily}",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        if ev.trend_1h.lower() != matched_cand.trend.h1.lower():
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"misquoted_trend_1h: {ev.trend_1h} != report {matched_cand.trend.h1}",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        passed_checks.append("check_e_evidence_matched")

        # Check f: Decision age and adverse price movement
        if now.tzinfo is None:
            now_utc = now.replace(tzinfo=timezone.utc)
        else:
            now_utc = now.astimezone(timezone.utc)

        try:
            report_ts = datetime.fromisoformat(report.timestamp)
            if report_ts.tzinfo is None:
                report_ts = report_ts.replace(tzinfo=timezone.utc)
            else:
                report_ts = report_ts.astimezone(timezone.utc)
        except Exception:
            report_ts = now_utc

        age_seconds = (now_utc - report_ts).total_seconds()
        if age_seconds < 0 or age_seconds > config.llm.max_decision_age_seconds:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"stale_decision: age {age_seconds:.1f}s > max {config.llm.max_decision_age_seconds}s",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        # Check adverse price movement
        curr_price = state.current_prices.get(asset)
        if curr_price is not None and matched_cand.current_price is not None:
            atr = matched_cand.atr if (matched_cand.atr and matched_cand.atr > 0) else 0.001
            max_move_atr = getattr(config.llm, "max_price_move_atr", 0.25)
            max_allowed_adverse_delta = max_move_atr * atr

            if direction == "CALL":
                # For CALL: price dropping is adverse
                adverse_delta = matched_cand.current_price - curr_price
            else:  # PUT
                # For PUT: price rising is adverse
                adverse_delta = curr_price - matched_cand.current_price

            if adverse_delta > max_allowed_adverse_delta + 1e-6:
                return ValidatedDecision(
                    decision="SKIP",
                    is_valid=False,
                    skip_reason=(
                        f"price_moved_against_direction: adverse delta {adverse_delta:.5f} > "
                        f"allowed {max_allowed_adverse_delta:.5f} ({max_move_atr} ATR)"
                    ),
                    checks_passed=passed_checks,
                    raw_decision=decision_out,
                )

        passed_checks.append("check_f_timing_and_price")

        # Check g: No open trade in opposite direction on same asset
        for ot in state.open_trades:
            ot_asset = getattr(ot, "asset", "")
            ot_dir = getattr(ot, "direction", "")
            if hasattr(ot_dir, "value"):
                ot_dir = ot_dir.value
            ot_dir_str = str(ot_dir).upper()

            if ot_asset == asset and ot_dir_str != direction:
                return ValidatedDecision(
                    decision="SKIP",
                    is_valid=False,
                    skip_reason=f"opposite_direction_open_trade: active {ot_dir_str} on {asset}",
                    checks_passed=passed_checks,
                    raw_decision=decision_out,
                )

        passed_checks.append("check_g_no_opposite_trade")

        # Check h: Trading permissions (kill switch, breaker, reconcile)
        if state.kill_switch_active:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason="kill_switch_active",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        if state.breaker_tripped:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason="breaker_tripped",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        if not state.reconcile_done:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason="reconcile_not_done",
                checks_passed=passed_checks,
                raw_decision=decision_out,
            )

        passed_checks.append("check_h_trading_allowed")

        # ALL 8 CHECKS PASSED
        return ValidatedDecision(
            decision=direction,
            is_valid=True,
            stake_amount=computed_stake,
            asset=asset,
            direction=direction,
            expiry_minutes=expiry,
            stake_pct=stake_pct,
            evidence=ev,
            reasoning=decision_out.reasoning,
            checks_passed=passed_checks,
            skip_reason=None,
            raw_decision=decision_out,
            shadow_only=matched_cand.shadow_only,
        )

    except Exception as exc:
        # Any unexpected exception inside validator returns safe SKIP
        return ValidatedDecision(
            decision="SKIP",
            is_valid=False,
            skip_reason=f"validator_exception: {exc}",
            checks_passed=passed_checks,
        )
