"""Protected packages integrity guard.

PROJECT_SPEC.md Sections 8 & 9:
  - safety/ and execution/ are protected packages.
  - The LLM cannot read, change, or bypass this code.
  - Generates and verifies SHA-256 hashes stored in PROTECTED_HASHES.json.
  - Exits with 0 if intact; non-zero if modified, missing, or unauthorized file added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROTECTED_DIRS = ["safety", "execution"]
HASH_FILE = "PROTECTED_HASHES.json"


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 digest of file with normalized newline encoding."""
    with open(path, "rb") as f:
        content = f.read()
    # Normalize CRLF to LF so hashes are invariant across operating systems
    normalized = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def scan_protected_files(root: Path) -> dict[str, str]:
    """Scan all .py files in protected directories and return {rel_path: sha256}."""
    hashes: dict[str, str] = {}
    for dir_name in PROTECTED_DIRS:
        target_dir = root / dir_name
        if not target_dir.is_dir():
            continue
        for file_path in sorted(target_dir.rglob("*.py")):
            if "__pycache__" in file_path.parts:
                continue
            rel_path = file_path.relative_to(root).as_posix()
            hashes[rel_path] = compute_file_hash(file_path)
    return hashes


def verify_protected(root: Path) -> tuple[bool, list[str]]:
    """Compare current files against recorded hashes in PROTECTED_HASHES.json."""
    hash_path = root / HASH_FILE
    if not hash_path.is_file():
        return False, [f"Missing hash registry file: {hash_path}"]

    try:
        with open(hash_path, "r", encoding="utf-8") as f:
            recorded = json.load(f)
    except Exception as e:
        return False, [f"Error reading {HASH_FILE}: {e}"]

    current = scan_protected_files(root)
    mismatches: list[str] = []

    # Check for modified or missing files
    for path, expected_hash in recorded.items():
        if path not in current:
            mismatches.append(f"MISSING PROTECTED FILE: {path}")
        elif current[path] != expected_hash:
            mismatches.append(f"UNAUTHORIZED MODIFICATION: {path} (expected {expected_hash}, got {current[path]})")

    # Check for unauthorized new files
    for path in current:
        if path not in recorded:
            mismatches.append(f"UNAUTHORIZED NEW FILE IN PROTECTED PACKAGE: {path}")

    return len(mismatches) == 0, mismatches


def update_hashes(root: Path) -> Path:
    """Scan protected files and write current SHA-256 hashes to PROTECTED_HASHES.json."""
    hashes = scan_protected_files(root)
    hash_path = root / HASH_FILE
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    return hash_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check integrity of protected packages")
    parser.add_argument("--update", action="store_true", help="Update PROTECTED_HASHES.json with current file hashes")
    args = parser.parse_args()

    root = Path.cwd()
    if args.update:
        hp = update_hashes(root)
        print(f"Updated {hp} with {len(scan_protected_files(root))} protected file hashes.")
        return 0

    ok, mismatches = verify_protected(root)
    if ok:
        print("PASS: All protected safety and execution package files match authorized hashes.")
        return 0
    else:
        print("FAIL: Protected packages integrity verification failed:")
        for m in mismatches:
            print(f"  - {m}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

