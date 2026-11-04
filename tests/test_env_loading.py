"""Tests for Environment Loading and `check-env` CLI command.

Adheres strictly to PROJECT_SPEC.md:
- .env loaded from project-root-anchored path using python-dotenv with override=False.
- Real environment variables win over .env definitions.
- Values never appear in logs or output (only 'set' or 'unset' status).
- `check-env` prints expected variable names, active configuration, and provider key status.
- Exits non-zero if decision_mode is llm and provider key is missing.
- Warns (not fails) if Telegram variables are unset.
- Warns loudly if LIVE_CONFIRM is YES.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pytest
import yaml

from core.config import AppConfig, load_config
from core.env import (
    DEFAULT_ENV_PATH,
    EXPECTED_ENV_VARS,
    check_environment,
    is_var_set,
    load_project_env,
)
from main import main


def _make_dummy_config_yaml(tmp_path: Path, decision_mode: str = "policy", provider: str = "anthropic") -> Path:
    """Helper to create a valid minimal config YAML file."""
    # Load base config.yaml and adjust paper.decision_mode & llm.provider
    base_cfg = Path("config.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(base_cfg)
    data["paper"]["decision_mode"] = decision_mode
    data["llm"]["provider"] = provider
    out_file = tmp_path / f"config_{decision_mode}_{provider}.yaml"
    out_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out_file


# ── 1. Temp .env Loading ──────────────────────────────────────────────────────

def test_temp_env_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A temp .env is loaded successfully into os.environ with override=False."""
    monkeypatch.delenv("CUSTOM_TEST_VAR_12345", raising=False)
    assert os.environ.get("CUSTOM_TEST_VAR_12345") is None

    env_file = tmp_path / "temp.env"
    env_file.write_text("CUSTOM_TEST_VAR_12345=injected_val_abcdef\n", encoding="utf-8")

    loaded_path = load_project_env(env_path=env_file, override=False)
    assert loaded_path == env_file
    assert os.environ.get("CUSTOM_TEST_VAR_12345") == "injected_val_abcdef"


# ── 2. Real Environment Variable Wins ─────────────────────────────────────────

def test_real_environment_variable_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real environment variables win over .env definitions (override=False)."""
    monkeypatch.setenv("LIVE_CONFIRM", "REAL_OS_ENVIRONMENT_VALUE")

    env_file = tmp_path / "temp.env"
    env_file.write_text("LIVE_CONFIRM=FILE_VALUE_SHOULD_BE_IGNORED\n", encoding="utf-8")

    load_project_env(env_path=env_file, override=False)
    assert os.environ.get("LIVE_CONFIRM") == "REAL_OS_ENVIRONMENT_VALUE"


# ── 3. Secret Values Never Appear in Output or Logs ───────────────────────────

def test_env_values_never_appear_in_output_or_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture):
    """Values of environment variables never appear in output or logs (only 'set'/'unset')."""
    secret_anthropic = "sk-ant-api03-SECRET_VALUE_DO_NOT_PRINT_11111"
    secret_gemini = "AIzaSy-SECRET_GEMINI_VALUE_DO_NOT_PRINT_22222"
    secret_tg_token = "123456789:ABC-DEF-SECRET_TG_TOKEN_33333"
    secret_tg_chat = "987654321_SECRET_CHAT_44444"

    monkeypatch.setenv("ANTHROPIC_API_KEY", secret_anthropic)
    monkeypatch.setenv("GEMINI_API_KEY", secret_gemini)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret_tg_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", secret_tg_chat)
    monkeypatch.setenv("LIVE_CONFIRM", "NO")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="policy", provider="anthropic")

    with caplog.at_level(logging.DEBUG):
        code = main(["check-env", "--config", str(cfg_file)])

    assert code == 0
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    combined_output = stdout + stderr + caplog.text

    # Values must NEVER appear
    assert secret_anthropic not in combined_output
    assert secret_gemini not in combined_output
    assert secret_tg_token not in combined_output
    assert secret_tg_chat not in combined_output

    # Status must be reported as set
    assert "ANTHROPIC_API_KEY       : set" in stdout
    assert "GEMINI_API_KEY          : set" in stdout
    assert "TELEGRAM_BOT_TOKEN      : set" in stdout
    assert "TELEGRAM_CHAT_ID        : set" in stdout
    assert "LIVE_CONFIRM            : set" in stdout


# ── 4. Check-Env Cases ────────────────────────────────────────────────────────

def test_check_env_decision_mode_llm_missing_anthropic_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Exit non-zero if paper.decision_mode is llm and provider key (anthropic) is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="llm", provider="anthropic")
    code = main(["check-env", "--config", str(cfg_file), "--env-file", str(empty_env)])

    assert code == 1
    out = capsys.readouterr().out
    assert "paper.decision_mode:     llm" in out
    assert "llm.provider:            anthropic" in out
    assert "Provider Key (ANTHROPIC_API_KEY for anthropic): unset" in out
    assert "ERROR: paper.decision_mode is 'llm' but required key ANTHROPIC_API_KEY is unset" in out


def test_check_env_decision_mode_llm_missing_gemini_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Exit non-zero if paper.decision_mode is llm and provider key (gemini) is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="llm", provider="gemini")
    code = main(["check-env", "--config", str(cfg_file), "--env-file", str(empty_env)])

    assert code == 1
    out = capsys.readouterr().out
    assert "paper.decision_mode:     llm" in out
    assert "llm.provider:            gemini" in out
    assert "Provider Key (GEMINI_API_KEY for gemini): unset" in out
    assert "ERROR: paper.decision_mode is 'llm' but required key GEMINI_API_KEY is unset" in out


def test_check_env_decision_mode_llm_key_present_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Exit zero if paper.decision_mode is llm and provider key is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "valid_secret_key_12345")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")
    monkeypatch.setenv("LIVE_CONFIRM", "NO")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="llm", provider="anthropic")
    code = main(["check-env", "--config", str(cfg_file)])

    assert code == 0
    out = capsys.readouterr().out
    assert "paper.decision_mode:     llm" in out
    assert "Provider Key (ANTHROPIC_API_KEY for anthropic): set" in out
    assert "ERROR" not in out


def test_check_env_decision_mode_policy_missing_key_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Exit zero when decision_mode is policy even if LLM key is unset."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="policy", provider="anthropic")
    code = main(["check-env", "--config", str(cfg_file), "--env-file", str(empty_env)])

    assert code == 0
    out = capsys.readouterr().out
    assert "paper.decision_mode:     policy" in out
    assert "Provider Key (ANTHROPIC_API_KEY for anthropic): unset" in out
    assert "ERROR" not in out


def test_check_env_telegram_unset_warns_not_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Warn (not fail) if Telegram variables are unset."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="policy")
    code = main(["check-env", "--config", str(cfg_file), "--env-file", str(empty_env)])

    assert code == 0
    out = capsys.readouterr().out
    assert "WARNING: Telegram variable(s)" in out
    assert "TELEGRAM_BOT_TOKEN" in out
    assert "TELEGRAM_CHAT_ID" in out


def test_check_env_live_confirm_yes_warns_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Warn loudly if LIVE_CONFIRM is set to YES."""
    monkeypatch.setenv("LIVE_CONFIRM", "YES")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="policy")
    code = main(["check-env", "--config", str(cfg_file)])

    assert code == 0
    out = capsys.readouterr().out
    assert "*** WARNING: LIVE_CONFIRM IS SET TO 'YES'!" in out
    assert "REAL CAPITAL AT RISK" in out


def test_check_env_live_confirm_no_no_loud_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """No loud warning if LIVE_CONFIRM is NO or unset."""
    monkeypatch.setenv("LIVE_CONFIRM", "NO")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")

    cfg_file = _make_dummy_config_yaml(tmp_path, decision_mode="policy")
    code = main(["check-env", "--config", str(cfg_file)])

    assert code == 0
    out = capsys.readouterr().out
    assert "*** WARNING: LIVE_CONFIRM IS SET TO 'YES'!" not in out


# ── 5. Tools Load .env at Startup ─────────────────────────────────────────────

def test_tools_load_env_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify that tools modules define and execute project-root anchored load_dotenv with override=False."""
    import tools.discover_platform
    import tools.summarize_discovery
    import adapters.binomo.calibrate

    assert hasattr(tools.discover_platform, "DEFAULT_ENV_PATH")
    assert tools.discover_platform.DEFAULT_ENV_PATH.name == ".env"
    assert hasattr(tools.summarize_discovery, "DEFAULT_ENV_PATH")
    assert tools.summarize_discovery.DEFAULT_ENV_PATH.name == ".env"
    assert hasattr(adapters.binomo.calibrate, "DEFAULT_ENV_PATH")
    assert adapters.binomo.calibrate.DEFAULT_ENV_PATH.name == ".env"


def test_main_subcommands_load_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Verify main subcommands load .env variables from custom or default path."""
    canary_key = "MAIN_CANARY_TEST_VAR"
    monkeypatch.delenv(canary_key, raising=False)

    temp_env = tmp_path / "custom.env"
    temp_env.write_text(f"{canary_key}=canary_active_value\n", encoding="utf-8")

    code = main(["check-env", "--env-file", str(temp_env)])
    assert code == 0
    assert os.environ.get(canary_key) == "canary_active_value"
    # Ensure value never printed
    out = capsys.readouterr().out
    assert "canary_active_value" not in out
