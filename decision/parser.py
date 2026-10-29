"""Strict JSON parser for LLM decision outputs.

PROJECT_SPEC.md Section 7.3, 7.4 & Phase 7:
  - Accepts ONLY a single valid JSON object.
  - Rejects markdown fences (e.g. ```json).
  - Rejects text before or after the JSON object.
  - Rejects multiple objects.
  - Rejects duplicate keys.
  - Zero output repair: failure raises ParseError.
"""

from __future__ import annotations

import json
from typing import Any


class ParseError(ValueError):
    """Raised when LLM output violates strict single-JSON constraints."""

    pass


def _duplicate_key_checker(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Object pairs hook for json.loads that strictly forbids duplicate keys."""
    res: dict[str, Any] = {}
    for key, value in pairs:
        if key in res:
            raise ParseError(f"Duplicate key '{key}' in LLM JSON output")
        res[key] = value
    return res


def parse_strict_json(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM response under strict single-object, no-filler rules.

    Raises:
        ParseError: if markdown fences, leading/trailing text, multiple objects,
                    duplicate keys, or invalid JSON syntax are present.
    """
    if not isinstance(raw_text, str):
        raise ParseError(f"Expected string output from LLM, got {type(raw_text).__name__}")

    stripped = raw_text.strip()
    if not stripped:
        raise ParseError("LLM returned empty or whitespace-only output")

    # Reject markdown code blocks or backticks
    if "```" in stripped or "`" in stripped:
        raise ParseError("Markdown formatting or backticks detected in LLM output; raw JSON required")

    # Reject non-object root structures
    if not stripped.startswith("{"):
        raise ParseError("Output must be a single JSON object starting with '{' and ending with '}'")

    decoder = json.JSONDecoder(object_pairs_hook=_duplicate_key_checker)
    try:
        data, end_pos = decoder.raw_decode(stripped)
    except Exception as exc:
        raise ParseError(f"JSON decoding failed: {exc}") from exc

    # Ensure no extraneous text or subsequent objects exist after the root object
    if end_pos < len(stripped):
        remaining = stripped[end_pos:].strip()
        if remaining:
            raise ParseError(f"Extraneous text or multiple objects detected after JSON root: '{remaining[:40]}...'")

    if not isinstance(data, dict):
        raise ParseError(f"Root JSON entity must be an object/dict, got {type(data).__name__}")

    return data
