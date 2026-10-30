"""Tests for protected package integrity guard."""

from __future__ import annotations

from pathlib import Path
from tools.check_protected import verify_protected


def test_protected_files_hashes_match():
    """All files in safety/ and execution/ must match PROTECTED_HASHES.json."""
    root = Path.cwd()
    ok, mismatches = verify_protected(root)
    assert ok is True, f"Protected file tampering detected: {mismatches}"


def test_protected_files_guard_detects_modification(tmp_path):
    """Integrity check fails if a file is tampered with."""
    from tools.check_protected import scan_protected_files, update_hashes

    # Create dummy protected package in tmp_path
    safe_dir = tmp_path / "safety"
    safe_dir.mkdir(parents=True)
    f1 = safe_dir / "limits.py"
    f1.write_text("print('hello')", encoding="utf-8")

    update_hashes(tmp_path)
    ok_before, _ = verify_protected(tmp_path)
    assert ok_before is True

    # Tamper with file
    f1.write_text("print('tampered')", encoding="utf-8")
    ok_after, mismatches = verify_protected(tmp_path)
    assert ok_after is False
    assert any("UNAUTHORIZED MODIFICATION" in m for m in mismatches)

