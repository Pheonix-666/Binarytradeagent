"""Decision service coordinating report generation, LLM invocation, parsing, validation, and JSONL audit logging.

PROJECT_SPEC.md Section 7 & Phase 7:
  - decide(candidates, state, config, client, now) -> ValidatedDecision.
  - If no candidates, returns SKIP immediately WITHOUT calling LLM (assert call count 0).
  - Rate limiting cost guard: max_calls_per_hour (default 60).
  - Handles client errors, parsing failures, and validation rejections cleanly as SKIP.
  - Logs every decision to logs/decisions.jsonl.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from analysis.candidates import CandidateTrade
from core.config import AppConfig
from core.logging_setup import get_logger
from decision.llm_client import LLMClient, LLMResponse, LLMUnavailable
from decision.parser import ParseError, parse_strict_json
from decision.prompt import PROMPT_VERSION, get_system_prompt
from decision.report import build_report
from decision.schema import DecisionOutput, DecisionReport, SystemState, ValidatedDecision
from decision.validator import validate_decision

logger = get_logger()


class DecisionService:
    """Manages the full LLM trading decision lifecycle."""

    def __init__(self, log_path: Union[str, Path] = "logs/decisions.jsonl"):
        self.log_path = Path(log_path)
        self.call_timestamps: list[datetime] = []

    def _check_rate_limit(self, max_calls_per_hour: int, now: datetime) -> bool:
        """Sliding-window hourly rate limiter."""
        one_hour_ago = now - timedelta(hours=1)
        self.call_timestamps = [t for t in self.call_timestamps if t > one_hour_ago]
        return len(self.call_timestamps) < max_calls_per_hour

    def _log_decision(self, record: dict[str, Any]) -> None:
        """Append one JSON line to decisions log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write decision log: {exc}")

    def decide(
        self,
        candidates: Sequence[CandidateTrade],
        state: SystemState,
        config: AppConfig,
        client: LLMClient,
        now: datetime,
    ) -> ValidatedDecision:
        """Run candidate evaluation through LLM client and validator.

        Returns:
            ValidatedDecision (CALL, PUT, or safe SKIP with reason).
        """
        if now.tzinfo is None:
            now_utc = now.replace(tzinfo=timezone.utc)
        else:
            now_utc = now.astimezone(timezone.utc)

        # 1. Zero candidate guard: return SKIP immediately without calling the LLM
        if not candidates:
            return ValidatedDecision(
                decision="SKIP",
                is_valid=True,
                reasoning="No trade candidates available",
                skip_reason="no_candidates",
                checks_passed=["zero_candidate_guard"],
            )

        # 2. Hourly cost / rate-limit guard
        max_calls = getattr(config.llm, "max_calls_per_hour", 60)
        if not self._check_rate_limit(max_calls, now_utc):
            logger.warning(f"LLM decision rate limit reached ({max_calls}/hour). Skipping.")
            return ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                reasoning="LLM rate limit reached for current hour",
                skip_reason="llm_rate_limit",
                checks_passed=[],
            )

        # 3. Build input report
        report = build_report(
            candidates=candidates,
            balance=getattr(state, "balance", 10000.0),
            open_trades=state.open_trades,
            config=config,
            now=now_utc,
            asset_modes=state.asset_modes,
        )
        report_json = report.model_dump_json(by_alias=True)
        system_prompt = get_system_prompt()

        # 4. Invoke LLM Client
        raw_output = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        latency_sec = 0.0

        try:
            self.call_timestamps.append(now_utc)
            resp = client.decide(system_prompt=system_prompt, report_json=report_json)
            raw_output = resp.raw_text
            token_usage["prompt_tokens"] = resp.prompt_tokens
            token_usage["completion_tokens"] = resp.completion_tokens
            latency_sec = resp.latency_seconds
        except (LLMUnavailable, Exception) as exc:
            skip_res = ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"llm_unavailable: {exc}",
                checks_passed=[],
            )
            self._log_decision({
                "timestamp": now_utc.isoformat(),
                "prompt_version": PROMPT_VERSION,
                "report": report.model_dump(),
                "raw_output": str(exc),
                "parsed_output": None,
                "validator_result": "SKIP",
                "is_valid": False,
                "stake_amount": None,
                "skip_reason": skip_res.skip_reason,
                "checks_passed": [],
                "token_usage": token_usage,
                "latency_seconds": latency_sec,
            })
            return skip_res

        # 5. Strict Parse
        try:
            parsed_dict = parse_strict_json(raw_output)
        except ParseError as exc:
            skip_res = ValidatedDecision(
                decision="SKIP",
                is_valid=False,
                skip_reason=f"parse_error: {exc}",
                checks_passed=[],
            )
            self._log_decision({
                "timestamp": now_utc.isoformat(),
                "prompt_version": PROMPT_VERSION,
                "report": report.model_dump(),
                "raw_output": raw_output,
                "parsed_output": None,
                "validator_result": "SKIP",
                "is_valid": False,
                "stake_amount": None,
                "skip_reason": skip_res.skip_reason,
                "checks_passed": [],
                "token_usage": token_usage,
                "latency_seconds": latency_sec,
            })
            return skip_res

        # 6. Validate Decision
        validated = validate_decision(
            parsed=parsed_dict,
            report=report,
            state=state,
            config=config,
            now=now_utc,
        )

        # 7. Audit Logging (one JSONL line per decision)
        self._log_decision({
            "timestamp": now_utc.isoformat(),
            "prompt_version": PROMPT_VERSION,
            "report": report.model_dump(),
            "raw_output": raw_output,
            "parsed_output": parsed_dict,
            "validator_result": validated.decision,
            "is_valid": validated.is_valid,
            "stake_amount": validated.stake_amount,
            "skip_reason": validated.skip_reason,
            "checks_passed": validated.checks_passed,
            "token_usage": token_usage,
            "latency_seconds": latency_sec,
        })

        return validated

