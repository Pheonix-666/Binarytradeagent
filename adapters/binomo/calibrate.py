"""Passive trading panel calibration tool for Binomo.

Adheres strictly to PROJECT_SPEC.md:
- Observation only: zero clicks, zero fills, zero key presses, zero button presses.
- Captures aria snapshot outline, URL path, asset table, and login indicators.
- Masks sensitive data (emails, phones, balance figures) while keeping durations and percentages.
- Never fails silently: writes errors directly into the calibration report.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Optional
import urllib.parse
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)

from adapters.binomo.reader import BinomoReadOnlySource
from adapters.binomo.session import BinomoSession
from core.config import AppConfig

logger = logging.getLogger("trading_system.adapters.binomo.calibrate")

# Keywords matching trading panel controls
TRADING_KEYWORDS_REGEX = re.compile(
    r"\b(call|put|up|down|higher|lower|amount|investment|expiry|time|duration|asset|demo|real|payout|profit|balance|trade)\b",
    re.IGNORECASE,
)

PROFILE_EXCLUDE_REGEX = re.compile(
    r"\b(profile|avatar|user_menu|main_menu|nav_menu|settings|security)\b",
    re.IGNORECASE,
)


def mask_calibration_text(text: str) -> str:
    """Mask emails, phones, balances, and long digit runs while keeping durations and percentages."""
    if not text:
        return ""
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<email>", text)
    text = re.sub(r"\+?\d[\d -]{7,}\d", "<phone>", text)
    # Currency amounts like $500, $10,500.00, €100
    text = re.sub(r"[\$€£₹]\s*\d{1,3}(,\d{3})*(\.\d+)?", "<num>", text)
    # Formatted comma numbers like 10,500.00
    text = re.sub(r"\b\d{1,3}(,\d{3})+(\.\d+)?", "<num>", text)
    # Long digit runs (4+ digits) not part of percentage or duration
    text = re.sub(r"\b\d{4,}\b", "<num>", text)
    return text


def extract_trading_panel_snapshot(page: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Capture and filter trading panel outline using aria_snapshot or CDP fallback."""
    items: list[dict[str, Any]] = []
    error_msg: Optional[str] = None

    try:
        raw_aria: Optional[str] = None
        loc_err: Optional[str] = None
        if hasattr(page, "locator"):
            try:
                body_loc = page.locator("body")
                if hasattr(body_loc, "aria_snapshot") and callable(body_loc.aria_snapshot):
                    raw_aria = body_loc.aria_snapshot()
            except Exception as e_loc:
                raw_aria = None
                loc_err = f"{type(e_loc).__name__}: {str(e_loc)}"

        if raw_aria and isinstance(raw_aria, str):
            for line in raw_aria.splitlines():
                # Format: - role "name": value
                m = re.match(r"^\s*-\s+([a-zA-Z0-9_-]+)(?:\s+\"([^\"]*)\")?(?::\s*(.*))?", line)
                if m:
                    role = (m.group(1) or "").strip()
                    name = (m.group(2) or "").strip()
                    val = (m.group(3) or "").strip()

                    combined = f"{role} {name} {val}"
                    if PROFILE_EXCLUDE_REGEX.search(combined):
                        continue

                    if TRADING_KEYWORDS_REGEX.search(combined):
                        entry: dict[str, Any] = {
                            "role": role,
                            "name": mask_calibration_text(name),
                            "test_id": "-",
                            "tag": "-",
                            "class": "-",
                        }
                        if val:
                            entry["value"] = mask_calibration_text(val)
                        items.append(entry)
        else:
            # Fallback to CDP Accessibility.getFullAXTree
            cdp_nodes: list[dict[str, Any]] = []
            if hasattr(page, "context") and hasattr(page.context, "new_cdp_session"):
                try:
                    cdp_client = page.context.new_cdp_session(page)
                    cdp_res = cdp_client.send("Accessibility.getFullAXTree")
                    if isinstance(cdp_res, dict):
                        cdp_nodes = cdp_res.get("nodes", [])
                except Exception as e_cdp:
                    error_msg = f"CDP error: {e_cdp}"

            if cdp_nodes:
                for node in cdp_nodes:
                    role_obj = node.get("role", {})
                    role = role_obj.get("value", "") if isinstance(role_obj, dict) else str(role_obj or "")
                    name_obj = node.get("name", {})
                    name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj or "")
                    val_obj = node.get("value", {})
                    val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj or "")

                    combined = f"{role} {name} {val}"
                    if PROFILE_EXCLUDE_REGEX.search(combined):
                        continue

                    if TRADING_KEYWORDS_REGEX.search(combined):
                        tid = str(node.get("data-testid") or node.get("data-test") or node.get("test_id") or "-")
                        tag = str(node.get("nodeName") or node.get("tag") or "-").lower()
                        cls = str(node.get("className") or node.get("class") or "-")

                        entry = {
                            "role": role,
                            "name": mask_calibration_text(name.strip()),
                            "test_id": tid,
                            "tag": tag,
                            "class": cls,
                        }
                        if val:
                            entry["value"] = mask_calibration_text(str(val).strip())
                        items.append(entry)
            elif not raw_aria:
                if loc_err and error_msg:
                    error_msg = f"{loc_err} | {error_msg}"
                elif loc_err:
                    error_msg = loc_err
                elif not error_msg:
                    error_msg = "Both locator('body').aria_snapshot() and CDP Accessibility.getFullAXTree returned no elements."

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"

    return items, error_msg


def generate_calibration_markdown(
    url_path: str,
    login_indicators: dict[str, bool],
    is_demo: bool,
    assets: list[Any],
    controls: list[dict[str, Any]],
    error_msg: Optional[str] = None,
    timestamp_str: Optional[str] = None,
) -> str:
    """Format single calibration markdown file content."""
    ts = timestamp_str or datetime.now(timezone.utc).isoformat()
    lines = [
        f"# Binomo Calibration Snapshot — {ts}",
        "",
        "## 1. Observation Context",
        f"- **Current URL Path**: `{url_path}`",
        f"- **Logged-in indicators**: {'yes' if login_indicators.get('logged_in') else 'no'} (ws_open={login_indicators.get('ws_open')}, account_join={login_indicators.get('account_join')}, passport_or_assets={login_indicators.get('passport_or_assets')})",
        f"- **Account Type**: {'DEMO' if is_demo else 'NON-DEMO / UNVERIFIED'}",
        f"- **Mode**: Passive Observation Only (zero clicks, zero key presses)",
        "",
        "## 2. Platform Assets and Payouts",
    ]

    if assets:
        lines.append("| Name | RIC | Type | OTC | Payout Rate | Status |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for a in assets:
            name = getattr(a, "name", None) or getattr(a, "asset_id", "-")
            ric = getattr(a, "asset_id", "-")
            otc = "Yes" if getattr(a, "is_otc", False) else "No"
            payout = f"{getattr(a, 'payout', 0.0):.0%}"
            st = "Open" if getattr(a, "is_open", True) else "Closed"
            a_type = getattr(a, "type", "crypto" if ric.startswith("Z-") else "forex")
            lines.append(f"| {name} | {ric} | {a_type} | {otc} | {payout} | {st} |")
        lines.append("")
    else:
        lines.append("- _No assets returned by reader._\n")

    lines.append("## 3. Trading Panel Controls Outline (Redacted)")
    if error_msg:
        lines.append(f"- **Snapshot Error**: `{error_msg}`")

    if controls:
        lines.append("| Role | Accessible Name | Test ID | Tag | Class Tokens |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for c in controls:
            role = c.get("role") or "-"
            name = c.get("name") or "-"
            tid = c.get("test_id") or "-"
            tag = c.get("tag") or "-"
            cls = c.get("class") or "-"
            lines.append(f"| `{role}` | `{name}` | `{tid}` | `{tag}` | `{cls}` |")
        lines.append("")
    elif not error_msg:
        lines.append("- _No trading panel controls matching keywords identified._\n")

    return "\n".join(lines) + "\n"


def run_calibration(
    session: Optional[BinomoSession] = None,
    reader: Optional[BinomoReadOnlySource] = None,
    page: Optional[Any] = None,
    reports_dir: Optional[Path | str] = None,
    duration_seconds: int = 300,
    interval_seconds: int = 20,
    max_iterations: Optional[int] = None,
) -> list[Path]:
    """Execute calibration loop every interval_seconds writing reports/binomo_calibration_<timestamp>.md."""
    rep_dir = Path(reports_dir or "reports")
    rep_dir.mkdir(parents=True, exist_ok=True)

    target_page = page or (session.page if session else None)
    if not target_page:
        raise RuntimeError("No active Playwright page available for calibration.")

    saved_reports: list[Path] = []
    start_time = time.time()
    iteration = 0

    print("=" * 80)
    print("BINOMO TRADING PANEL CALIBRATION (OBSERVATION ONLY)")
    print(f"Duration: {duration_seconds}s | Interval: {interval_seconds}s")
    print("Please switch assets and expiry options by hand in the browser window.")
    print("=" * 80)

    while time.time() - start_time < duration_seconds:
        iteration += 1
        now_utc = datetime.now(timezone.utc)
        ts_suffix = now_utc.strftime("%Y%m%d_%H%M%S")
        report_file = rep_dir / f"binomo_calibration_{ts_suffix}.md"

        # 1. Current URL path (no query string)
        current_url = getattr(target_page, "url", "https://binomo.com/trading")
        parsed_url = urllib.parse.urlparse(current_url)
        url_path = parsed_url.path or "/"

        # 2. Login indicators
        login_indicators = session.ws_listener.get_login_indicators() if session else {
            "ws_open": True,
            "account_join": True,
            "passport_or_assets": True,
            "logged_in": True,
        }
        is_demo = session.is_demo() if session else True

        # 3. Assets from reader
        assets: list[Any] = []
        if reader:
            try:
                assets = reader.get_assets()
            except Exception as e_assets:
                logger.warning("Could not fetch assets during calibration: %s", e_assets)

        # 4. DOM Snapshot
        controls, error_msg = extract_trading_panel_snapshot(target_page)

        # 5. Write Report
        content = generate_calibration_markdown(
            url_path=url_path,
            login_indicators=login_indicators,
            is_demo=is_demo,
            assets=assets,
            controls=controls,
            error_msg=error_msg,
            timestamp_str=now_utc.isoformat(),
        )

        report_file.write_text(content, encoding="utf-8")
        saved_reports.append(report_file)

        print(f"[{now_utc.strftime('%H:%M:%S')}] [Calibration #{iteration}] Saved snapshot to: {report_file.name} ({len(controls)} controls)")

        if max_iterations and iteration >= max_iterations:
            break

        time.sleep(interval_seconds)

    print(f"Calibration completed! Generated {len(saved_reports)} snapshots in {rep_dir.resolve()}.")
    return saved_reports
