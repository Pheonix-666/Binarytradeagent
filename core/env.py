"""Environment loading and verification utilities.

Adheres strictly to PROJECT_SPEC.md:
- Loads .env from project root with override=False (real environment variables win).
- Never prints or logs environment variable values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from core.config import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

EXPECTED_ENV_VARS: list[str] = [
    "LIVE_CONFIRM",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
]

PROVIDER_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def load_project_env(env_path: Optional[Path | str] = None, override: bool = False) -> Path:
    """Load .env file using python-dotenv with override=False by default."""
    target_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if target_path.exists():
        load_dotenv(dotenv_path=target_path, override=override)
    return target_path


def is_var_set(name: str) -> bool:
    """Return True if environment variable is set and non-empty."""
    val = os.environ.get(name)
    return bool(val and val.strip())


def check_environment(config: AppConfig) -> tuple[int, list[str]]:
    """Check expected environment variables and active system configuration.
    
    Returns:
        (exit_code, output_lines): exit_code is 1 if paper.decision_mode == 'llm'
        and the chosen provider key is missing; 0 otherwise.
    """
    lines: list[str] = [
        "==================================================",
        "              ENVIRONMENT CONFIG CHECK            ",
        "==================================================",
        "Expected Environment Variables:",
    ]
    for var in EXPECTED_ENV_VARS:
        st = "set" if is_var_set(var) else "unset"
        lines.append(f"  - {var:<24}: {st}")

    lines.append("--------------------------------------------------")
    lines.append("Active System Configuration:")
    lines.append(f"  - mode:                    {config.mode}")
    lines.append(f"  - paper.decision_mode:     {config.paper.decision_mode}")
    lines.append(f"  - llm.provider:            {config.llm.provider}")
    lines.append(f"  - llm.model:               {config.llm.model}")

    provider = config.llm.provider.lower()
    req_key = PROVIDER_KEY_MAP.get(provider)
    if req_key:
        req_st = "set" if is_var_set(req_key) else "unset"
        lines.append(f"  - Provider Key ({req_key} for {provider}): {req_st}")
    else:
        lines.append(f"  - Provider Key: none required for '{provider}'")
    lines.append("==================================================")

    exit_code = 0

    # 1. Exit non-zero if decision_mode is llm and the provider's key is missing
    if str(config.paper.decision_mode).lower() == "llm":
        if req_key and not is_var_set(req_key):
            lines.append(f"\nERROR: paper.decision_mode is 'llm' but required key {req_key} is unset.")
            exit_code = 1

    # 2. Warn (not fail) if the Telegram variables are unset
    tg_token_set = is_var_set("TELEGRAM_BOT_TOKEN")
    tg_chat_set = is_var_set("TELEGRAM_CHAT_ID")
    if not tg_token_set or not tg_chat_set:
        unset_tg: list[str] = []
        if not tg_token_set:
            unset_tg.append("TELEGRAM_BOT_TOKEN")
        if not tg_chat_set:
            unset_tg.append("TELEGRAM_CHAT_ID")
        lines.append(f"\nWARNING: Telegram variable(s) {', '.join(unset_tg)} unset; Telegram alerts will be disabled.")

    # 3. Warn loudly if LIVE_CONFIRM is YES
    if os.environ.get("LIVE_CONFIRM", "").strip() == "YES":
        lines.append("\n" + "*" * 70)
        lines.append("*** WARNING: LIVE_CONFIRM IS SET TO 'YES'!                         ***")
        lines.append("*** LIVE ORDER EXECUTION GUARD IS DISABLED. REAL CAPITAL AT RISK!   ***")
        lines.append("*" * 70)

    return exit_code, lines
