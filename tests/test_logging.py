"""Tests for JSONL and console logging."""

import json
from pathlib import Path
from core.logging_setup import get_logger, log_decision, _get_jsonl_path


def test_get_logger():
    """Ensure standard logger is returned and configured."""
    logger = get_logger()
    assert logger.name == "trading_system"


def test_log_decision(tmp_path, monkeypatch):
    """Ensure log_decision writes a valid JSON line with timestamp."""
    monkeypatch.setattr("core.logging_setup._LOGS_DIR", tmp_path)
    
    test_data = {
        "asset": "EURUSD",
        "action": "SKIP",
        "reason": "gate_failed",
        "risk_score": 45.5,
    }
    
    log_decision(test_data)
    
    log_file = list(tmp_path.glob("decisions_*.jsonl"))
    assert len(log_file) == 1
    
    with open(log_file[0], "r", encoding="utf-8") as fh:
        lines = fh.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "timestamp" in record
        assert record["asset"] == "EURUSD"
        assert record["action"] == "SKIP"
        assert record["risk_score"] == 45.5

