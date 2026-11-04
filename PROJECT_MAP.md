# Trade-Agent Project Map

This document provides a single-line overview of every top-level directory and root file in the project, detailing its purpose and operational role (`system`, `tool`, `test`, or `documentation`).

## Top-Level Directories

- **`.venv/`**: Python virtual environment containing third-party runtime dependencies and package libraries (`system`).
- **`adapters/`**: Platform integration layer providing Binomo browser automation, paper trading simulation, and base interfaces (`system` / `tool`).
- **`adaptive/`**: Dynamic parameter tuning, market regime detection, and edge decay tracking (`system`).
- **`alerts/`**: Telegram notification bot and real-time operational alert dispatchers (`system`).
- **`analysis/`**: Technical indicators, multi-timeframe trend classification, and support/resistance candidate generation (`system`).
- **`backtest/`**: Simulation engine, walk-forward testing framework, and historical performance metrics (`tool` / `system`).
- **`core/`**: Shared domain models, centralized configuration loading, environment validation, and logging (`system`).
- **`data/`**: SQLite persistence, real-time candle aggregation, and multi-source market price providers (`system`).
- **`decision/`**: LLM trade reasoning prompts, structured output parsers, schema validators, and audit logging (`system`).
- **`docs/`**: Engineering guides, edge decay research, and pattern analysis documentation (`documentation`).
- **`execution/`**: Position lifecycle management, trade state machines, and execution reconciliation engines (`system`).
- **`orchestrator/`**: Main trading event loop, background task workers, and operational schedulers (`system`).
- **`patterns/`**: Candlestick pattern detection, DTW shape similarity matching, and template libraries (`system`).
- **`reports_archive/`**: Permanent historical archive of platform discovery dumps, focus reports, and network traces (`documentation`).
- **`risk/`**: Expected value calculation, Kelly stake sizing, and portfolio exposure limiters (`system`).
- **`safety/`**: Hard-coded safety rules, latching kill switches, and circuit breaker protection (`system`).
- **`selection/`**: Asset payout filtering, ranking, and paper-to-live graduation gates (`system`).
- **`state/`**: Persistent runtime storage containing browser automation profile and lockfiles (`system`).
- **`tests/`**: Pytest test suite containing unit, integration, and property tests across all modules (`test`).
- **`tools/`**: Standalone command-line utilities for platform discovery, history conversion, and integrity checking (`tool`).

## Project Root Files

- **`.env`**: Local environment file specifying runtime API keys, secrets, and credentials (`system`).
- **`.env.example`**: Reference template specifying expected environment variables without sensitive values (`documentation`).
- **`.gitignore`**: Git repository exclusions for caches, temporary outputs, and local state files (`system`).
- **`cleanup_deleted_untracked.txt`**: Manifest of untracked scratch and old output files deleted during Step 2 cleanup (`documentation`).
- **`cleanup_proposal.md`**: Step 1 repository inventory and categorisation audit report (`documentation`).
- **`config.yaml`**: Primary system configuration file specifying risk limits, UI selectors, and strategy parameters (`system`).
- **`main.py`**: Central CLI entry point offering `run`, `status`, `summary`, `binomo-calibrate`, and `check-env` commands (`system` / `tool`).
- **`PROJECT_MAP.md`**: Architectural directory and root file map detailing project components and execution roles (`documentation`).
- **`PROJECT_SPEC.md`**: Master system specification, architectural rules, and engineering constraints (`documentation`).
- **`PROTECTED_HASHES.json`**: Cryptographic SHA-256 hashes safeguarding safety and execution packages against modification (`system` / `test`).
- **`pytest.ini`**: Pytest configuration file with test discovery rules, marker definitions, and test options (`test`).
- **`requirements.txt`**: Production Python dependencies and version requirements (`system`).
- **`requirements-dev.txt`**: Development, linting, and testing dependencies (`test`).
- **`run_backtest.py`**: Convenience wrapper script executing full backtesting suites (`tool`).
- **`RUNNING_DEMO.md`**: Operator guide detailing configuration, startup, and validation for Binomo demo trading (`documentation`).

