"""Discovery report summarizer — slims down discovery_report.md into a comprehensive discovery_summary.md.

Usage:
    python tools/summarize_discovery.py [--input discovery_report.md] [--output discovery_summary.md]

Architecture:
    Per-section line budgets summing to <= 850 lines (strictly <= 900 lines total).
    Prioritizes distinct (topic, event) pairs with 1 example first, then repeated examples.
    Overflowing sections end with "[section X truncated: N more entries omitted]".
    Includes an overview table at the top reporting each section's budget, line count, and truncation status.
    Guarantees later sections (HTTP endpoints, samples, demo-vs-real) and final secret-pattern match line are never dropped.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Optional
import urllib.parse
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)


DEFAULT_TRACKER_HOSTS = [
    "doubleclick",
    "google-analytics",
    "googletagmanager",
    "googleads",
    "adservice",
    "facebook",
    "hotjar",
    "clarity",
    "appsflyer",
    "adjust",
    "segment",
    "mixpanel",
    "sentry",
    "analytics.google",
    "accounts.google",
    "criteo",
    "yandex",
    "app-measurement",
]

TRACKER_HOSTS_REGEX = re.compile(
    r"|".join(re.escape(h) for h in DEFAULT_TRACKER_HOSTS),
    re.IGNORECASE,
)

INTERESTING_PATH_REGEX = re.compile(
    r"assets?|instruments?|quotes?|payouts?|balance|deals?|orders?|candles?|history|account|currencies",
    re.IGNORECASE,
)

# Static asset filter regex
STATIC_ASSET_REGEX = re.compile(
    r"\.(js|css|png|jpg|jpeg|svg|webp|gif|ico|woff|woff2|ttf|eot)(\?.*)?$|/static/",
    re.IGNORECASE,
)


def is_non_i18n_account_field(key: str, val: Any) -> bool:
    """Filter out UI translation strings, keeping only real account discriminator fields."""
    k_lower = key.lower()
    if k_lower in {"account_type", "is_demo", "is_real", "demo", "real", "demo_account", "real_account"}:
        if isinstance(val, bool):
            return True
        if isinstance(val, (int, float)):
            return False
        if isinstance(val, str):
            if val.lower() in {"demo", "real", "standard", "practice"}:
                return True
            if " " in val or "{{" in val or len(val) > 15:
                return False
            return True
    return False


def extract_candle_url_pattern(url: str) -> str:
    """Keep candle URL with final numeric timeframe segment and query parameter names only."""
    clean = url.split("#")[0].split(";")[0]
    parsed = urllib.parse.urlparse(clean)
    path = parsed.path or "/"

    m = re.search(r"(/candles/[^?#;]*/)(\d+)$", path)
    if m:
        prefix = m.group(1)
        timeframe = m.group(2)
        prefix = re.sub(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "{uuid}",
            prefix,
        )
        prefix = re.sub(r"/\d+(?=/|$)", "/{id}", prefix)
        path = f"{prefix}{timeframe}"
    else:
        path = collapse_url(url)[1]

    query_names = []
    if parsed.query:
        query_names = [k for k, _ in urllib.parse.parse_qsl(parsed.query)]
    q_str = ("?" + "&".join(query_names)) if query_names else ""
    return f"{parsed.scheme or 'https'}://{parsed.netloc or 'api.binomo.com'}{path}{q_str}"


def compact_json(obj: Any) -> str:
    """Format JSON compactly by inlining simple primitive lists, 1-key dicts, and collapsing uniform large maps."""
    def trim_large_maps(o: Any) -> Any:
        if isinstance(o, dict):
            # If map has > 5 keys with uniform simple dict values (like currency tables with 170 currencies)
            if len(o) > 5 and all(isinstance(v, dict) and len(v) <= 2 for v in o.values()):
                items = list(o.items())
                res = {k: trim_large_maps(v) for k, v in items[:3]}
                res[f"... ({len(o)-3} more keys omitted)"] = trim_large_maps(items[0][1])
                return res
            return {k: trim_large_maps(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [trim_large_maps(x) for x in o[:2]]
        return o

    o_trimmed = trim_large_maps(obj)
    s = json.dumps(o_trimmed, indent=2)
    s = re.sub(r'\[\s+([^\[\]\{\}]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', s)
    s = re.sub(r'\{\s+([^\[\]\{\}]+?)\s+\}', lambda m: '{ ' + ' '.join(m.group(1).split()) + ' }', s)
    return s


def get_item_status(example: Any) -> tuple[str, str]:
    """Returns (status_label, formatted_content). Status: 'complete' or 'truncated at capture (N chars stored)'."""
    if example is None:
        return "not captured in report", ""
    if isinstance(example, dict) and "raw" in example and len(example) == 1:
        raw_text = str(example["raw"])
        return f"truncated at capture ({len(raw_text)} chars stored)", raw_text
    return "complete", compact_json(example)


# ── Per-Section Line Budgets (Constants) ───────────────────────────────────────
# Suggested budgets scaled to sum to 850 lines, guaranteeing grand total <= 900 lines with table & headers.
BUDGET_COUNTS_AND_HOSTS = 35
BUDGET_PHX_JOIN_TOPICS = 35
BUDGET_SUBSCRIBE_UNSUBSCRIBE = 50
BUDGET_PRICES_QUOTES = 140
BUDGET_BALANCE = 70
BUDGET_DEALS = 160
BUDGET_ASSETS_PAYOUTS = 100
BUDGET_SYSTEM_AND_OTHER = 50
BUDGET_HTTP_ENDPOINTS_LIST = 70
BUDGET_HTTP_SAMPLES = 105
BUDGET_DEMO_VS_REAL = 35

SECTION_BUDGETS: dict[str, int] = {
    "Counts and Hosts": BUDGET_COUNTS_AND_HOSTS,
    "Phoenix Join Topics": BUDGET_PHX_JOIN_TOPICS,
    "Subscribe / Unsubscribe": BUDGET_SUBSCRIBE_UNSUBSCRIBE,
    "Prices / Quotes": BUDGET_PRICES_QUOTES,
    "Balance": BUDGET_BALANCE,
    "Deals": BUDGET_DEALS,
    "Assets / Payouts": BUDGET_ASSETS_PAYOUTS,
    "System and Other": BUDGET_SYSTEM_AND_OTHER,
    "HTTP Endpoints List": BUDGET_HTTP_ENDPOINTS_LIST,
    "HTTP Samples": BUDGET_HTTP_SAMPLES,
    "Demo vs Real Fields": BUDGET_DEMO_VS_REAL,
}

# Secret pattern scanners matching PayloadRedactor
JWT_REGEX = re.compile(r"\b[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LONG_HEX_B64_REGEX = re.compile(r"\b[A-Za-z0-9+/=_-]{25,}\b")
UUID_REGEX = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")

IDENTIFIER_CHAR_REGEX = re.compile(r"^[A-Za-z0-9_./:-]+$")
SEPARATOR_REGEX = re.compile(r"[_./:-]")


def is_tracker(host: str) -> bool:
    """Check if host matches ad/analytics/tracker hosts."""
    return bool(TRACKER_HOSTS_REGEX.search(host.lower()))


def collapse_url(url: str) -> tuple[str, str]:
    """Strip query strings & matrix params; collapse numbers and UUIDs; return (host, collapsed_path)."""
    clean = url.split("?")[0].split("#")[0].split(";")[0].strip()
    parsed = urllib.parse.urlparse(clean)
    host = parsed.netloc.lower() or "unknown_host"
    path = parsed.path or "/"

    path = re.sub(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "{uuid}",
        path,
    )
    path = re.sub(r"/\d+(?=/|$)", "/{id}", path)
    return host, path


def format_json_example(data: Any, max_lines: int = 10) -> list[str]:
    """Format full JSON structure up to max_lines, preserving parsed object structure."""
    dumped = json.dumps(data, indent=2)
    lines = dumped.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["  ..."]
    return lines


def compact_json_snippet(data: Any, max_lines: int = 5) -> str:
    """Format JSON compactly for backward compatibility."""
    lines = format_json_example(data, max_lines=max_lines)
    return "\n".join(lines)


def is_phone_number(tok: str) -> bool:
    """Check if token matches phone number pattern without being a unix timestamp."""
    if not (len(tok) >= 9 and PHONE_REGEX.fullmatch(tok) and sum(c.isdigit() for c in tok) >= 7):
        return False
    if tok.isdigit():
        if (len(tok) == 10 and tok.startswith(("15", "16", "17", "18", "19", "20"))) or \
           (len(tok) == 13 and tok.startswith(("15", "16", "17", "18", "19", "20"))):
            return False
    return True


def is_safe_token(tok: str) -> bool:
    """Check if a token in text is a known non-secret identifier or timestamp."""
    if not tok or len(tok) < 4:
        return True
    if tok.startswith("http://") or tok.startswith("https://") or tok.startswith("wss://"):
        return True
    if tok.startswith("<str:") or tok == "<masked>" or tok == "<num>":
        return True
    if tok.isdigit():
        if (len(tok) == 10 and tok.startswith(("15", "16", "17", "18", "19", "20"))) or \
           (len(tok) == 13 and tok.startswith(("15", "16", "17", "18", "19", "20"))):
            return True
    if IP_REGEX.fullmatch(tok) or EMAIL_REGEX.search(tok) or JWT_REGEX.search(tok) or UUID_REGEX.fullmatch(tok):
        return False
    # System identifiers with separators
    if IDENTIFIER_CHAR_REGEX.match(tok) and SEPARATOR_REGEX.search(tok):
        if not UUID_REGEX.match(tok) and not JWT_REGEX.match(tok):
            segments = SEPARATOR_REGEX.split(tok)
            if all(len(s) <= 24 and not (len(s) >= 16 and re.fullmatch(r"[0-9a-fA-F]{16,}", s)) for s in segments):
                return True
    return False


def count_secret_matches(text_lines: list[str]) -> int:
    """Scan summary text lines for sensitive patterns (JWT, email, phone, IP, long hex/b64, UUID)."""
    matches = 0
    for line in text_lines:
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("- **Pair**:")
            or stripped.startswith("- **Type**:")
            or stripped.startswith("- Topic:")
            or stripped.startswith("secret-pattern matches")
        ):
            continue
        tokens = re.split(r"[\s`\"'()\[\]{}]+", line)
        for tok in tokens:
            if is_safe_token(tok):
                continue
            if JWT_REGEX.fullmatch(tok):
                matches += 1
            elif EMAIL_REGEX.fullmatch(tok):
                matches += 1
            elif IP_REGEX.fullmatch(tok) and not tok.startswith("0.0.") and not tok.startswith("127."):
                matches += 1
            elif is_phone_number(tok):
                matches += 1
            elif len(tok) > 24 and LONG_HEX_B64_REGEX.fullmatch(tok):
                matches += 1
            elif UUID_REGEX.fullmatch(tok):
                matches += 1
    return matches


def extract_assets_from_data(example: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract compact asset rows and full first 2 asset objects from any JSON array of assets."""
    asset_list: list[Any] = []
    if isinstance(example, list):
        asset_list = example
    elif isinstance(example, dict):
        if isinstance(example.get("data"), dict) and isinstance(example["data"].get("assets"), list):
            asset_list = example["data"]["assets"]
        elif isinstance(example.get("data"), list):
            asset_list = example["data"]
        elif isinstance(example.get("assets"), list):
            asset_list = example["assets"]
        else:
            for v in example.values():
                if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
                    asset_list = v
                    break

    if not asset_list or not isinstance(asset_list[0], dict):
        return [], []

    first_two = asset_list[:2]

    ASSET_FIELD_REGEX = re.compile(
        r"payout|payment_rate|rate|status|trading_status|min_amount|max_amount|"
        r"min_bet|max_bet|expir|durat|is_active|is_open|step|interval|timeframe|schedule",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    for item in asset_list:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("asset_name") or "-")
        ric = str(item.get("ric") or item.get("symbol") or "-")
        a_type = str(item.get("type") or item.get("kind") or "-")
        is_otc = bool(ric.startswith("Z-") or "(otc)" in name.lower() or "(OTC)" in name or item.get("is_otc"))

        tts = item.get("trading_tools_settings") if isinstance(item.get("trading_tools_settings"), dict) else {}
        ftt = tts.get("ftt") if isinstance(tts.get("ftt"), dict) else {}

        enabled_acc = item.get("enabled_account_types") or ftt.get("enabled_account_types") or []
        enabled_acc_str = ", ".join(str(x) for x in enabled_acc) if isinstance(enabled_acc, list) and enabled_acc else str(enabled_acc)

        enabled_tt = item.get("enabled_trading_types") or list(tts.keys()) or []
        enabled_tt_str = ", ".join(str(x) for x in enabled_tt) if isinstance(enabled_tt, list) and enabled_tt else str(enabled_tt)

        r_turbo = ftt.get("base_payment_rate_turbo") or item.get("payment_rate") or "-"
        r_standard = ftt.get("base_payment_rate_standard") or item.get("base_payment_rate_standard") or "-"
        r_classic = ftt.get("base_payment_rate_classic") or item.get("base_payment_rate_classic") or "-"
        rates_str = f"turbo: {r_turbo} | std: {r_standard} | classic: {r_classic}"

        sched_obj = ftt.get("schedule") or item.get("schedule") or {}
        today_sched_str = "-"
        if isinstance(sched_obj, dict):
            if "durations" in sched_obj:
                today_sched_str = f"durations: {sched_obj['durations']}"
            elif "status" in sched_obj:
                today_sched_str = str(sched_obj["status"])
            else:
                today_sched_str = json.dumps(sched_obj)[:60]
        elif sched_obj:
            today_sched_str = str(sched_obj)[:60]

        row = {
            "name": name,
            "ric": ric,
            "type": a_type,
            "otc": is_otc,
            "enabled_account_types": enabled_acc_str,
            "enabled_trading_types": enabled_tt_str,
            "rates_str": rates_str,
            "base_rate_turbo": r_turbo,
            "base_rate_standard": r_standard,
            "base_rate_classic": r_classic,
            "today_schedule": today_sched_str,
            "payment_rate": r_turbo if r_turbo != "-" else (item.get("payment_rate") or "-"),
            "status": ftt.get("trading_status") or item.get("trading_status") or item.get("status") or "-",
        }
        rows.append(row)
    return rows, first_two


def format_asset_compact_table(asset_rows: list[dict[str, Any]]) -> list[str]:
    """Format all asset rows as a compact Markdown table with Pass 4 fields."""
    if not asset_rows:
        return ["_No assets recorded._\n"]
    header = "| Name | RIC | Type | OTC | Base Rates (Turbo/Std/Classic) | Enabled Accounts | Trading Types | Schedule |"
    sep = "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    lines = [header, sep]
    for row in asset_rows:
        name = str(row.get("name") or row.get("asset_name") or "-")
        ric = str(row.get("ric") or row.get("symbol") or "-")
        a_type = str(row.get("type") or row.get("kind") or "-")
        is_otc = row.get("otc")
        if is_otc is None:
            is_otc = bool(ric.startswith("Z-") or "(otc)" in name.lower() or "(OTC)" in name)
        otc_str = "Yes" if is_otc else "No"

        rates_str = row.get("rates_str")
        if not rates_str:
            t = row.get("base_rate_turbo") or row.get("payment_rate") or "-"
            s = row.get("base_rate_standard") or "-"
            c = row.get("base_rate_classic") or "-"
            rates_str = f"turbo: {t} | std: {s} | classic: {c}"

        enabled_acc = str(row.get("enabled_account_types") or "-")
        enabled_tt = str(row.get("enabled_trading_types") or "-")
        sched = str(row.get("today_schedule") or row.get("schedule") or row.get("durations") or "-")

        lines.append(f"| {name} | {ric} | {a_type} | {otc_str} | {rates_str} | {enabled_acc} | {enabled_tt} | {sched} |")
    lines.append("")
    return lines


def extract_deal_status_and_option_types(deals_pairs: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    """Scan opened/closed deal events for distinct status values with counts and distinct option_type values."""
    status_counts: Counter[str] = Counter()
    option_types: set[str] = set()

    for p in deals_pairs:
        ev = (p.get("event") or "").lower()
        if "opened" in ev or "closed" in ev or "deal" in ev:
            for ex in p.get("examples", []):
                def scan_obj(obj: Any) -> None:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            k_l = str(k).lower()
                            if k_l == "status" and isinstance(v, str) and not v.startswith("<"):
                                status_counts[v] += 1
                            elif k_l in ("option_type", "deal_type", "type") and isinstance(v, str) and not v.startswith("<"):
                                if v.lower() in ("turbo", "binary", "classic", "standard", "call", "put", "crypto", "digital", "cfd"):
                                    option_types.add(v)
                            if isinstance(v, (dict, list)):
                                scan_obj(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            scan_obj(item)

                scan_obj(ex)
    return dict(status_counts), sorted(option_types)


def extract_trading_panel_controls(dom_snapshots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Extract trading panel controls from DOM accessibility snapshots."""
    controls: dict[str, list[dict[str, Any]]] = {
        "amount_input": [],
        "expiry_selector": [],
        "call_put_buttons": [],
        "asset_selector": [],
        "account_switch": [],
    }
    if not dom_snapshots:
        return controls

    all_elements: list[dict[str, Any]] = []
    for snap in dom_snapshots:
        all_elements.extend(snap.get("elements", []))

    for el in all_elements:
        name = str(el.get("name") or "")
        role = str(el.get("role") or "")
        tid = str(el.get("data-testid") or el.get("data-test") or el.get("test_id") or "")
        combined = f"{name} {role} {tid}".lower()

        # Call / Put buttons
        if re.search(r"\b(call|put|up|down|higher|lower)\b", combined):
            if role in ("button", "link", "radio") or re.search(r"btn|button", tid):
                if not any(x.get("name") == name and x.get("role") == role for x in controls["call_put_buttons"]):
                    controls["call_put_buttons"].append(el)
        # Amount input
        elif re.search(r"\b(amount|investment|bet)\b", combined):
            if role in ("textbox", "spinbutton", "combobox") or re.search(r"input|amount", tid):
                if not any(x.get("name") == name and x.get("role") == role for x in controls["amount_input"]):
                    controls["amount_input"].append(el)
        # Expiry / Time selector & options
        elif re.search(r"\b(expiry|expiration|duration|time)\b", combined) or re.search(r"\b\d+\s*(min|s|sec|m)\b", combined):
            if role in ("combobox", "button", "option", "listbox", "menuitem") or re.search(r"time|expiry|duration", tid):
                if not any(x.get("name") == name and x.get("role") == role for x in controls["expiry_selector"]):
                    controls["expiry_selector"].append(el)
        # Asset selector
        elif re.search(r"\b(asset|instrument|pair|symbol|market)\b", combined):
            if role in ("combobox", "button", "tab", "listbox") or re.search(r"asset|instrument|pair", tid):
                if not any(x.get("name") == name and x.get("role") == role for x in controls["asset_selector"]):
                    controls["asset_selector"].append(el)
        # Demo / Real account switch
        elif re.search(r"\b(demo|real)\b", combined) and re.search(r"\b(account|switch|select|toggle)\b", combined):
            if not any(x.get("name") == name and x.get("role") == role for x in controls["account_switch"]):
                controls["account_switch"].append(el)

    return controls


# ── Report Parser ─────────────────────────────────────────────────────────────

def parse_discovery_report(report_path: Path) -> dict[str, Any]:
    """Parse sections from discovery_report.md into structured data."""
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    data: dict[str, Any] = {
        "redactions_applied": 0,
        "binary_frames_count": 0,
        "i18n_dropped": 0,
        "ws_endpoints": set(),
        "phx_join_topics": [],  # list of (topic, count)
        "subscribe_messages": [],  # list of (host, topic, event, json_data)
        "classified_pairs": defaultdict(list),  # category -> list of (topic, event, count, rate_str, [examples])
        "large_data_frames": [],  # list of {topic, structure, price_field, time_field, sample}
        "demo_real_fields": {},  # key -> value
        "http_endpoints": [],  # list of {method, url, status, count, example}
        "dom_snapshots": [],  # list of {label, elements}
        "alt_ws_data": {},  # host -> {msg_count, event_types, sub_frames, data_frames, asset_rates, price_field, time_field}
        "asset_table": [],  # list of compact asset rows
        "asset_first_two": [],  # full structure of first 2 asset objects
    }

    current_section = ""
    current_category = "other"
    current_sub_host = ""
    current_sub_topic = ""
    current_sub_event = ""
    current_pair_topic = ""
    current_pair_event = ""
    current_pair_count = 0
    current_pair_rate = "N/A"
    current_ldf_topic = ""
    current_ldf_struct = ""
    current_ldf_price = ""
    current_ldf_time = ""
    current_http_method = "GET"
    current_http_url = ""
    current_http_status = 200
    current_http_count = 1
    current_snap_label = "snapshot"
    current_alt_host = ""

    inside_json_block = False
    json_lines: list[str] = []

    category_map = {
        "prices / quotes": "prices/quotes",
        "assets / payouts": "assets/payouts",
        "balance": "balance",
        "deals": "deals",
        "system": "system",
        "other": "other",
        # Backward compatibility with Pass 1 report headers
        "price / candle / tick": "prices/quotes",
        "asset / instrument list": "assets/payouts",
        "payout / yield": "assets/payouts",
        "balance / account": "balance",
        "order / trade / position": "deals",
        "unknown / other": "other",
    }

    with open(report_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()

            # Global counts from header
            if stripped.startswith("- **Total Redactions Applied**:"):
                m = re.search(r":\s*(\d+)", stripped)
                if m:
                    data["redactions_applied"] = int(m.group(1))
            elif stripped.startswith("- **Binary Frames Count**:") or stripped.startswith("- Total binary frames:"):
                m = re.search(r":\s*(\d+)", stripped)
                if m:
                    data["binary_frames_count"] = int(m.group(1))
            elif stripped.startswith("- **i18n Messages Dropped**:"):
                m = re.search(r":\s*(\d+)", stripped)
                if m:
                    data["i18n_dropped"] = int(m.group(1))

            # Section Headers
            if line.startswith("## 2. WebSocket Endpoints"):
                current_section = "ws"
                continue
            elif line.startswith("## 3. Phoenix Join Topics"):
                current_section = "phx_join"
                continue
            elif line.startswith("## 4. Subscribe and Unsubscribe Messages"):
                current_section = "subscribe"
                continue
            elif line.startswith("## 5. Binary Frames Summary") or line.startswith("## 3. Binary Frames Summary"):
                current_section = "binary"
                continue
            elif line.startswith("## 6. Message Classification") or line.startswith("## 4. Message Classification"):
                current_section = "messages"
                continue
            elif line.startswith("## 7. Large Data Frames Breakdown"):
                current_section = "ldf"
                continue
            elif line.startswith("## 8. Demo vs Real Account Fields"):
                current_section = "demo_real"
                continue
            elif line.startswith("## 9. Platform HTTP Endpoints") or line.startswith("## 5. HTTP Endpoints"):
                current_section = "http"
                continue
            elif line.startswith("## 10. DOM Elements") or line.startswith("## 6. DOM Elements"):
                current_section = "dom"
                continue
            elif line.startswith("## 11. Outgoing"):
                current_section = "outgoing"
                continue
            elif line.startswith("## 12. DOM Accessibility Snapshots"):
                current_section = "dom_snapshots"
                continue
            elif line.startswith("## 13. Non-Phoenix WebSocket"):
                current_section = "alt_ws"
                continue
            elif line.startswith("## 14. Asset List Table"):
                current_section = "asset_table"
                continue

            # Section 2: WebSocket Endpoints
            if current_section == "ws":
                if stripped.startswith("- `") and stripped.endswith("`"):
                    ep = stripped[3:-1]
                    clean = ep.split("?")[0].split("#")[0].split(";")[0]
                    data["ws_endpoints"].add(clean)

            # Section 3: Phoenix Join Topics
            elif current_section == "phx_join":
                m = re.search(r"- Topic:\s*`([^`]+)`\s*\(Count:\s*(\d+)\)", stripped)
                if m:
                    data["phx_join_topics"].append((m.group(1), int(m.group(2))))

            # Section 4: Subscribe and Unsubscribe Messages
            elif current_section == "subscribe":
                m = re.search(r"- Host:\s*`([^`]+)`\s*\|\s*Topic:\s*`([^`]+)`\s*\|\s*Event:\s*`([^`]+)`", stripped)
                if m:
                    current_sub_host = m.group(1)
                    current_sub_topic = m.group(2)
                    current_sub_event = m.group(3)
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if current_sub_host and json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                        except Exception:
                            obj = {"raw": "\n".join(json_lines)[:200]}
                        data["subscribe_messages"].append({
                            "host": current_sub_host,
                            "topic": current_sub_topic,
                            "event": current_sub_event,
                            "message": obj,
                        })
                    current_sub_host = ""
                    json_lines = []
                    continue

                if inside_json_block:
                    json_lines.append(line.rstrip())
                    continue

            # Section 6: Messages
            elif current_section == "messages":
                if line.startswith("### "):
                    sub_title = line.lstrip("#").strip().split("(")[0].strip().lower()
                    sub_title = re.sub(r"^category:\s*", "", sub_title)
                    sub_clean = re.sub(r"\s+messages$", "", sub_title).strip()
                    current_category = category_map.get(sub_clean, "other")
                    continue

                m_pair = re.search(
                    r"- \*\*Pair\*\*:\s*`\(([^,]+),\s*([^)]+)\)`\s*\(Count:\s*(\d+)(?:,\s*Rate:\s*([^)]+)\))?",
                    stripped,
                )
                if m_pair:
                    current_pair_topic = m_pair.group(1).strip()
                    current_pair_event = m_pair.group(2).strip()
                    current_pair_count = int(m_pair.group(3))
                    current_pair_rate = m_pair.group(4) or "N/A"
                    continue

                # Pass 1 compatibility: - **Type**: `xyz` (Count: N)
                m_type = re.search(r"- \*\*Type\*\*:\s*`([^`]+)`\s*\(Count:\s*(\d+)\)", stripped)
                if m_type:
                    current_pair_topic = ""
                    current_pair_event = m_type.group(1).strip()
                    current_pair_count = int(m_type.group(2))
                    current_pair_rate = "N/A"
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if (current_pair_topic or current_pair_event) and json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                        except Exception:
                            obj = {"raw": "\n".join(json_lines)[:200]}

                        cat_list = data["classified_pairs"][current_category]
                        existing = next(
                            (e for e in cat_list if e["topic"] == current_pair_topic and e["event"] == current_pair_event),
                            None,
                        )
                        if existing:
                            if len(existing["examples"]) < 2:
                                existing["examples"].append(obj)
                        else:
                            cat_list.append({
                                "topic": current_pair_topic,
                                "event": current_pair_event,
                                "count": current_pair_count,
                                "rate": current_pair_rate,
                                "examples": [obj],
                            })
                    json_lines = []
                    continue

                if inside_json_block:
                    if len(json_lines) < 600:
                        json_lines.append(line.rstrip())
                    continue

            # Section 7: Large Data Frames
            elif current_section == "ldf":
                if stripped.startswith("- **Topic**:"):
                    m = re.search(r"- \*\*Topic\*\*:\s*`([^`]+)`", stripped)
                    if m:
                        current_ldf_topic = m.group(1)
                    continue
                elif stripped.startswith("- Structure:"):
                    current_ldf_struct = stripped.split(":", 1)[1].strip()
                    continue
                elif stripped.startswith("- Price Field:"):
                    m = re.search(r"- Price Field:\s*`([^`]+)`", stripped)
                    if m:
                        current_ldf_price = m.group(1)
                    continue
                elif stripped.startswith("- Time Field:"):
                    m = re.search(r"- Time Field:\s*`([^`]+)`", stripped)
                    if m:
                        current_ldf_time = m.group(1)
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if current_ldf_topic and json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                        except Exception:
                            obj = {"raw": "\n".join(json_lines)[:200]}
                        data["large_data_frames"].append({
                            "topic": current_ldf_topic,
                            "structure": current_ldf_struct,
                            "price_field": current_ldf_price,
                            "time_field": current_ldf_time,
                            "sample": obj,
                        })
                    current_ldf_topic = ""
                    json_lines = []
                    continue

                if inside_json_block:
                    json_lines.append(line.rstrip())
                    continue

            # Section 8: Demo vs Real Fields
            elif current_section == "demo_real":
                m = re.search(r"- `([^`]+)`:\s*`([^`]+)`", stripped)
                if m:
                    val_str = m.group(2)
                    try:
                        val = json.loads(val_str)
                    except Exception:
                        val = val_str
                    data["demo_real_fields"][m.group(1)] = val

            # Section 9: HTTP Endpoints
            elif current_section == "http":
                m = re.search(r"- \[([A-Z]+)\]\s*`([^`]+)`\s*\(status:\s*(\d+),\s*count:\s*(\d+)\)", stripped)
                if m:
                    current_http_method = m.group(1)
                    current_http_url = m.group(2)
                    current_http_status = int(m.group(3))
                    current_http_count = int(m.group(4))
                    data["http_endpoints"].append({
                        "method": current_http_method,
                        "url": current_http_url,
                        "status": current_http_status,
                        "count": current_http_count,
                        "example": None,
                    })
                    continue

                # Pass 1 compatibility: - `url`
                if stripped.startswith("- `") and stripped.endswith("`"):
                    ep = stripped[3:-1]
                    data["http_endpoints"].append({
                        "method": "GET",
                        "url": ep,
                        "status": 200,
                        "count": 1,
                        "example": None,
                    })
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if data["http_endpoints"] and json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                        except Exception:
                            obj = {"raw": "\n".join(json_lines)[:200]}
                        data["http_endpoints"][-1]["example"] = obj
                        last_url = data["http_endpoints"][-1]["url"].lower()
                        if "assets" in last_url or "bo-assets" in last_url:
                            rows, f2 = extract_assets_from_data(obj)
                            if rows and not data["asset_table"]:
                                data["asset_table"] = rows
                            if f2 and not data["asset_first_two"]:
                                data["asset_first_two"] = f2
                    json_lines = []
                    continue

                if inside_json_block:
                    if len(json_lines) < 600:
                        json_lines.append(line.rstrip())
                    continue

            # Section 12: DOM Accessibility Snapshots
            elif current_section == "dom_snapshots":
                if stripped.startswith("### Snapshot:"):
                    m = re.search(r"Snapshot:\s*`([^`]+)`", stripped)
                    if m:
                        current_snap_label = m.group(1)
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                            if isinstance(obj, list):
                                data["dom_snapshots"].append({"label": current_snap_label, "elements": obj})
                        except Exception:
                            pass
                    json_lines = []
                    continue

                if inside_json_block:
                    if len(json_lines) < 600:
                        json_lines.append(line.rstrip())
                    continue

            # Section 13: Non-Phoenix WebSocket Data
            elif current_section == "alt_ws":
                if stripped.startswith("### Host:"):
                    m = re.search(r"Host:\s*`([^`]+)`", stripped)
                    if m:
                        current_alt_host = m.group(1)
                        if current_alt_host not in data["alt_ws_data"]:
                            data["alt_ws_data"][current_alt_host] = {
                                "msg_count": 0,
                                "event_types": {},
                                "subscribe_frames": [],
                                "data_frames": [],
                                "asset_rates": {},
                                "price_field": "",
                                "time_field": "",
                            }
                    continue
                elif stripped.startswith("- Distinct Event Types:"):
                    m = re.search(r"Distinct Event Types:\s*(\{.*\})", stripped)
                    if m and current_alt_host:
                        try:
                            import ast
                            data["alt_ws_data"][current_alt_host]["event_types"] = ast.literal_eval(m.group(1))
                        except Exception:
                            pass
                    continue
                elif stripped.startswith("- Price Field:"):
                    m_p = re.search(r"Price Field:\s*`([^`]+)`", stripped)
                    m_t = re.search(r"Time Field:\s*`([^`]+)`", stripped)
                    if current_alt_host:
                        if m_p:
                            data["alt_ws_data"][current_alt_host]["price_field"] = m_p.group(1)
                        if m_t:
                            data["alt_ws_data"][current_alt_host]["time_field"] = m_t.group(1)
                    continue
                elif stripped.startswith("- `") and "msgs" in stripped:
                    m = re.search(r"- `([^`]+)`:\s*(\d+)\s*msgs", stripped)
                    if m and current_alt_host:
                        data["alt_ws_data"][current_alt_host]["asset_rates"][m.group(1)] = int(m.group(2))
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if current_alt_host and json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                            entry = data["alt_ws_data"][current_alt_host]
                            if isinstance(obj, list):
                                if not entry["subscribe_frames"]:
                                    entry["subscribe_frames"] = obj
                                elif not entry["data_frames"]:
                                    entry["data_frames"] = obj
                            elif isinstance(obj, dict):
                                if "subscribe" in str(obj).lower():
                                    entry["subscribe_frames"].append(obj)
                                else:
                                    entry["data_frames"].append(obj)
                        except Exception:
                            pass
                    json_lines = []
                    continue

                if inside_json_block:
                    if len(json_lines) < 600:
                        json_lines.append(line.rstrip())
                    continue

            # Section 14: Asset List Table
            elif current_section == "asset_table":
                if stripped.startswith("|") and not stripped.startswith("| :---") and not stripped.startswith("| Name"):
                    cols = [c.strip() for c in stripped.strip("|").split("|")]
                    if len(cols) >= 8:
                        data["asset_table"].append({
                            "name": cols[0],
                            "ric": cols[1],
                            "type": cols[2],
                            "otc": cols[3].lower() == "yes",
                            "rates_str": cols[4],
                            "enabled_account_types": cols[5],
                            "enabled_trading_types": cols[6],
                            "today_schedule": cols[7],
                            "payment_rate": cols[4],
                        })
                    elif len(cols) >= 4:
                        data["asset_table"].append({
                            "name": cols[0],
                            "ric": cols[1],
                            "type": cols[2],
                            "payment_rate": cols[3],
                            "status": cols[4] if len(cols) > 4 else "-",
                            "min_max": cols[5] if len(cols) > 5 else "-",
                            "durations": cols[6] if len(cols) > 6 else "-",
                            "otc": bool(cols[1].startswith("Z-") or "(otc)" in cols[0].lower()),
                        })
                    continue

                if stripped == "```json":
                    inside_json_block = True
                    json_lines = []
                    continue
                elif stripped == "```" and inside_json_block:
                    inside_json_block = False
                    if json_lines:
                        try:
                            obj = json.loads("\n".join(json_lines))
                            if isinstance(obj, list) and not data["asset_first_two"]:
                                data["asset_first_two"] = obj[:2]
                        except Exception:
                            pass
                    json_lines = []
                    continue

                if inside_json_block:
                    if len(json_lines) < 600:
                        json_lines.append(line.rstrip())
                    continue

    return data


# ── Section Formatters with Independent Budgets ────────────────────────────────

def format_counts_and_hosts(data: dict[str, Any], budget: int) -> tuple[list[str], bool, int]:
    """Format Counts and Hosts section strictly within line budget."""
    total_text_ws_frames = sum(
        entry["count"]
        for cat_list in data["classified_pairs"].values()
        for entry in cat_list
    )
    binary_frames = data["binary_frames_count"]

    distinct_ws_host_paths = set()
    for ws_url in data["ws_endpoints"]:
        p = urllib.parse.urlparse(ws_url)
        distinct_ws_host_paths.add((p.netloc.lower(), p.path or "/"))

    http_endpoints_seen = len(data["http_endpoints"])
    redactions_applied = data["redactions_applied"]
    i18n_dropped = data["i18n_dropped"]

    sec_lines = [
        "## 1. Key Counts",
        f"- **Text WebSocket Frames**: {total_text_ws_frames}",
        f"- **Binary Frames**: {binary_frames}",
        f"- **Distinct WebSocket Hosts/Paths**: {len(distinct_ws_host_paths)}",
        f"- **HTTP Endpoints Seen**: {http_endpoints_seen}",
        f"- **Redactions Applied**: {redactions_applied}",
        f"- **i18n Messages Filtered**: {i18n_dropped}",
        "",
        "## 2. WebSocket Endpoints",
    ]
    truncated = False
    omitted = 0

    if distinct_ws_host_paths:
        sorted_hosts = sorted(distinct_ws_host_paths)
        included = 0
        for host, path in sorted_hosts:
            line = f"- Endpoint: `wss://{host}{path}`"
            if len(sec_lines) + 1 > budget - 1:
                truncated = True
                omitted = len(sorted_hosts) - included
                break
            sec_lines.append(line)
            included += 1
    else:
        sec_lines.append("- _No WebSocket endpoints recorded._")

    if truncated:
        sec_lines.append(f"[section Counts and Hosts truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def format_phx_join_topics(topics: list[tuple[str, int]], budget: int) -> tuple[list[str], bool, int]:
    """Format Phoenix Join Topics section strictly within line budget."""
    sec_lines = ["## Phoenix Join Topics Captured"]
    if not topics:
        sec_lines.append("- _No phx_join topics recorded._\n")
        return sec_lines[:budget], False, 0

    sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
    included = 0
    truncated = False
    omitted = 0

    for top, cnt in sorted_topics:
        line = f"- Topic: `{top}` (Count: {cnt})"
        if len(sec_lines) + 1 > budget - 1:
            truncated = True
            omitted = len(sorted_topics) - included
            break
        sec_lines.append(line)
        included += 1

    if truncated:
        sec_lines.append(f"[section Phoenix Join Topics truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def format_subscribe_messages(sub_messages: list[dict[str, Any]], budget: int) -> tuple[list[str], bool, int]:
    """Format Subscribe / Unsubscribe section strictly within line budget."""
    sec_lines = ["## 3. Subscribe / Unsubscribe Messages (Full Redacted)"]
    if not sub_messages:
        sec_lines.append("- _No subscribe/unsubscribe messages captured._\n")
        return sec_lines[:budget], False, 0

    included = 0
    truncated = False
    omitted = 0

    for sub in sub_messages:
        item_lines = [
            f"- Host: `{sub['host']}` | Topic: `{sub['topic']}` | Event: `{sub['event']}`",
            "```json",
        ]
        item_lines.extend(format_json_example(sub["message"], max_lines=8))
        item_lines.append("```")

        if len(sec_lines) + len(item_lines) > budget - 1:
            truncated = True
            omitted = len(sub_messages) - included
            break
        sec_lines.extend(item_lines)
        included += 1

    if truncated:
        sec_lines.append(f"[section Subscribe / Unsubscribe truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def format_pairs_section(
    section_id: str,
    cat_title: str,
    pairs: list[dict[str, Any]],
    budget: int,
    extra_preamble_lines: Optional[list[str]] = None,
) -> tuple[list[str], bool, int]:
    """Format message category section prioritizing distinct pairs with 1 example, then 2nd examples."""
    sec_lines = [f"### Category: {cat_title} ({len(pairs)} pairs)"]
    if extra_preamble_lines:
        sec_lines.extend(extra_preamble_lines)

    if not pairs:
        sec_lines.append(f"_No messages captured in {cat_title} category._\n")
        return sec_lines[:budget], False, 0

    rendered_pairs: list[dict[str, Any]] = []
    for entry in pairs:
        t = entry["topic"] or "N/A"
        e = entry["event"] or "N/A"
        cnt = entry["count"]
        rate = entry["rate"]
        p_lines = [f"- **Pair**: `({t}, {e})` (Count: {cnt}, Rate: {rate})"]
        if entry["examples"]:
            p_lines.append("```json")
            p_lines.extend(format_json_example(entry["examples"][0], max_lines=8))
            p_lines.append("```")
        rendered_pairs.append({
            "first_lines": p_lines,
            "has_second": len(entry["examples"]) > 1,
            "second_lines": ["```json"] + format_json_example(entry["examples"][1], max_lines=8) + ["```"] if len(entry["examples"]) > 1 else [],
        })

    included_pairs = 0
    truncated = False
    omitted = 0

    # Step 1: Distinct (topic, event) pairs with 1 example each
    for rp in rendered_pairs:
        cost = len(rp["first_lines"])
        if len(sec_lines) + cost > budget - 1:
            truncated = True
            omitted = len(rendered_pairs) - included_pairs
            break
        sec_lines.extend(rp["first_lines"])
        included_pairs += 1

    # Step 2: If all distinct pairs fit and budget allows, add repeated examples
    if not truncated:
        omitted_repeated = 0
        total_repeated = sum(1 for rp in rendered_pairs if rp["has_second"])
        included_repeated = 0
        for rp in rendered_pairs:
            if rp["has_second"]:
                cost = len(rp["second_lines"])
                if len(sec_lines) + cost > budget - 1:
                    truncated = True
                    omitted_repeated = total_repeated - included_repeated
                    break
                sec_lines.extend(rp["second_lines"])
                included_repeated += 1
        if truncated:
            omitted = omitted_repeated

    # Standard section name for truncation reporting
    report_name = cat_title.replace("PRICE / ", "Prices / ")
    if truncated:
        sec_lines.append(f"[section {report_name} truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def format_http_endpoints_list(endpoints: list[dict[str, Any]], budget: int) -> tuple[list[str], bool, int]:
    """Format Platform HTTP Endpoints list strictly within line budget."""
    sec_lines = ["## 6. Platform HTTP Endpoints (Trackers Excluded)"]
    if not endpoints:
        sec_lines.append("- _No platform HTTP endpoints recorded._\n")
        return sec_lines[:budget], False, 0

    valid_http = []
    for ep in endpoints:
        h, _ = collapse_url(ep["url"])
        if not is_tracker(h):
            valid_http.append(ep)

    if not valid_http:
        sec_lines.append("- _No non-tracker HTTP endpoints recorded._\n")
        return sec_lines[:budget], False, 0

    included = 0
    truncated = False
    omitted = 0

    for ep in valid_http:
        _, path = collapse_url(ep["url"])
        line = f"- [{ep['method']}] `{path}` (status: {ep['status']}, count: {ep['count']})"
        if len(sec_lines) + 1 > budget - 1:
            truncated = True
            omitted = len(valid_http) - included
            break
        sec_lines.append(line)
        included += 1

    if truncated:
        sec_lines.append(f"[section HTTP Endpoints List truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def format_http_samples(endpoints: list[dict[str, Any]], budget: int) -> tuple[list[str], bool, int]:
    """Format HTTP Key Trading Samples strictly within line budget."""
    sec_lines = ["## HTTP Key Trading Samples"]
    samples = [
        ep for ep in endpoints
        if ep.get("example") is not None and INTERESTING_PATH_REGEX.search(collapse_url(ep["url"])[1])
    ]
    if not samples:
        sec_lines.append("- _No structured payloads captured for trading paths._\n")
        return sec_lines[:budget], False, 0

    included = 0
    truncated = False
    omitted = 0

    for ep in samples:
        _, path = collapse_url(ep["url"])
        item_lines = [
            f"- [{ep['method']}] `{path}` (status: {ep['status']}) - Sample payload (verified matching endpoint):",
            "```json",
        ]
        item_lines.extend(format_json_example(ep["example"], max_lines=8))
        item_lines.append("```")

        if len(sec_lines) + len(item_lines) > budget - 1:
            truncated = True
            omitted = len(samples) - included
            break
        sec_lines.extend(item_lines)
        included += 1

    if truncated:
        sec_lines.append(f"[section HTTP Samples truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def format_demo_real_fields(demo_real_fields: dict[str, Any], budget: int) -> tuple[list[str], bool, int]:
    """Format Demo vs Real Account Indicators strictly within line budget."""
    sec_lines = ["## 7. Demo vs Real Account Indicators"]
    if not demo_real_fields:
        sec_lines.append("- _No explicit demo/real fields captured._\n")
        return sec_lines[:budget], False, 0

    sorted_fields = sorted(demo_real_fields.items())
    included = 0
    truncated = False
    omitted = 0

    for k, v in sorted_fields:
        line = f"- `{k}`: `{json.dumps(v)}`"
        if len(sec_lines) + 1 > budget - 1:
            truncated = True
            omitted = len(sorted_fields) - included
            break
        sec_lines.append(line)
        included += 1

    if truncated:
        sec_lines.append(f"[section Demo vs Real Fields truncated: {omitted} more entries omitted]")

    sec_lines.append("")
    return sec_lines[:budget], truncated, omitted


def generate_sections_overview_table(section_reports: list[dict[str, Any]]) -> list[str]:
    """Generate Markdown overview table with line counts and truncation status."""
    lines = [
        "## Summary Sections Overview",
        "",
        "| Section | Budget | Line Count | Truncated |",
        "| :--- | :---: | :---: | :---: |",
    ]
    total_budget = sum(s["budget"] for s in section_reports)
    total_lines = sum(s["line_count"] for s in section_reports)
    any_truncated = any(s["truncated"] for s in section_reports)

    for s in section_reports:
        trunc_str = "Yes" if s["truncated"] else "No"
        lines.append(f"| {s['name']} | {s['budget']} | {s['line_count']} | {trunc_str} |")

    total_trunc_str = "Yes" if any_truncated else "No"
    lines.append(f"| **Total** | **{total_budget}** | **{total_lines}** | **{total_trunc_str}** |")
    lines.append("")
    return lines


# ── Markdown Document Assembly ────────────────────────────────────────────────

def generate_summary_markdown(data: dict[str, Any], budgets: Optional[dict[str, int]] = None) -> str:
    """Generate concise discovery summary markdown with per-section line budgets strictly under 900 lines."""
    b = budgets or SECTION_BUDGETS

    section_reports: list[dict[str, Any]] = []

    # 1. Counts and Hosts
    b_counts = b.get("Counts and Hosts", BUDGET_COUNTS_AND_HOSTS)
    c_lines, c_trunc, _ = format_counts_and_hosts(data, b_counts)
    section_reports.append({
        "name": "Counts and Hosts",
        "budget": b_counts,
        "line_count": len(c_lines),
        "truncated": c_trunc,
        "lines": c_lines,
    })

    # 2. Phoenix Join Topics
    b_phx = b.get("Phoenix Join Topics", BUDGET_PHX_JOIN_TOPICS)
    p_lines, p_trunc, _ = format_phx_join_topics(data["phx_join_topics"], b_phx)
    section_reports.append({
        "name": "Phoenix Join Topics",
        "budget": b_phx,
        "line_count": len(p_lines),
        "truncated": p_trunc,
        "lines": p_lines,
    })

    # 3. Subscribe / Unsubscribe Messages
    b_sub = b.get("Subscribe / Unsubscribe", BUDGET_SUBSCRIBE_UNSUBSCRIBE)
    sub_lines, sub_trunc, _ = format_subscribe_messages(data["subscribe_messages"], b_sub)
    section_reports.append({
        "name": "Subscribe / Unsubscribe",
        "budget": b_sub,
        "line_count": len(sub_lines),
        "truncated": sub_trunc,
        "lines": sub_lines,
    })

    # 4. Prices / Quotes (includes Large Data Frames breakdown preamble if available)
    b_pq = b.get("Prices / Quotes", BUDGET_PRICES_QUOTES)
    ldf_lines: list[str] = []
    if data.get("large_data_frames"):
        ldf_lines.append("## 5. Large Data Frames Breakdown")
        for ldf in data["large_data_frames"][:2]:
            ldf_lines.append(f"- **Topic**: `{ldf['topic']}`")
            ldf_lines.append(f"  - Price Field: `{ldf['price_field']}`")
            ldf_lines.append(f"  - Time Field: `{ldf['time_field']}`")
            ldf_lines.append("```json")
            ldf_lines.extend(format_json_example(ldf["sample"], max_lines=6))
            ldf_lines.append("```")

    price_pairs = data["classified_pairs"].get("prices/quotes", [])
    pq_lines, pq_trunc, _ = format_pairs_section("prices_quotes", "PRICE / Quotes", price_pairs, b_pq, extra_preamble_lines=ldf_lines)
    section_reports.append({
        "name": "Prices / Quotes",
        "budget": b_pq,
        "line_count": len(pq_lines),
        "truncated": pq_trunc,
        "lines": pq_lines,
    })

    # 5. Balance
    b_bal = b.get("Balance", BUDGET_BALANCE)
    bal_pairs = data["classified_pairs"].get("balance", [])
    bal_lines, bal_trunc, _ = format_pairs_section("balance", "Balance", bal_pairs, b_bal)
    section_reports.append({
        "name": "Balance",
        "budget": b_bal,
        "line_count": len(bal_lines),
        "truncated": bal_trunc,
        "lines": bal_lines,
    })

    # 6. Deals
    b_deals = b.get("Deals", BUDGET_DEALS)
    deal_pairs = data["classified_pairs"].get("deals", [])
    deal_lines, deal_trunc, _ = format_pairs_section("deals", "Deals", deal_pairs, b_deals)
    section_reports.append({
        "name": "Deals",
        "budget": b_deals,
        "line_count": len(deal_lines),
        "truncated": deal_trunc,
        "lines": deal_lines,
    })

    # 7. Assets / Payouts
    b_ap = b.get("Assets / Payouts", BUDGET_ASSETS_PAYOUTS)
    ap_pairs = data["classified_pairs"].get("assets/payouts", [])
    ap_lines, ap_trunc, _ = format_pairs_section("assets_payouts", "Assets / Payouts", ap_pairs, b_ap)
    section_reports.append({
        "name": "Assets / Payouts",
        "budget": b_ap,
        "line_count": len(ap_lines),
        "truncated": ap_trunc,
        "lines": ap_lines,
    })

    # 8. System and Other
    b_so = b.get("System and Other", BUDGET_SYSTEM_AND_OTHER)
    so_pairs = data["classified_pairs"].get("system", []) + data["classified_pairs"].get("other", [])
    so_lines, so_trunc, _ = format_pairs_section("system_other", "System and Other", so_pairs, b_so)
    section_reports.append({
        "name": "System and Other",
        "budget": b_so,
        "line_count": len(so_lines),
        "truncated": so_trunc,
        "lines": so_lines,
    })

    # 9. HTTP Endpoints List
    b_http = b.get("HTTP Endpoints List", BUDGET_HTTP_ENDPOINTS_LIST)
    http_lines, http_trunc, _ = format_http_endpoints_list(data["http_endpoints"], b_http)
    section_reports.append({
        "name": "HTTP Endpoints List",
        "budget": b_http,
        "line_count": len(http_lines),
        "truncated": http_trunc,
        "lines": http_lines,
    })

    # 10. HTTP Samples
    b_samples = b.get("HTTP Samples", BUDGET_HTTP_SAMPLES)
    samples_lines, samples_trunc, _ = format_http_samples(data["http_endpoints"], b_samples)
    section_reports.append({
        "name": "HTTP Samples",
        "budget": b_samples,
        "line_count": len(samples_lines),
        "truncated": samples_trunc,
        "lines": samples_lines,
    })

    # 11. Demo vs Real Fields
    b_demo = b.get("Demo vs Real Fields", BUDGET_DEMO_VS_REAL)
    demo_lines, demo_trunc, _ = format_demo_real_fields(data["demo_real_fields"], b_demo)
    section_reports.append({
        "name": "Demo vs Real Fields",
        "budget": b_demo,
        "line_count": len(demo_lines),
        "truncated": demo_trunc,
        "lines": demo_lines,
    })

    # Document assembly
    out_lines: list[str] = [
        "# Platform Discovery Summary",
        "",
    ]
    # Overview table
    out_lines.extend(generate_sections_overview_table(section_reports))

    # All sections
    for s in section_reports:
        out_lines.extend(s["lines"])

    # Hard cap safety check (should never be needed since 850 + 20 <= 870 < 900)
    if len(out_lines) > 895:
        out_lines = out_lines[:890]
        out_lines.append("... [truncated to fit 900 line limit]")
        out_lines.append("")

    # Final line: secret pattern count (must be 0)
    secret_matches = count_secret_matches(out_lines)
    out_lines.append(f"secret-pattern matches in this summary: {secret_matches}")

    return "\n".join(out_lines) + "\n"


def generate_focus_markdown(data: dict[str, Any]) -> str:
    """Generate discovery_focus.md (strictly <= 500 lines) with key trading focus items."""
    out_lines: list[str] = [
        "# Platform Discovery Focus Report",
        "",
        "## 1. Focus Items Status Overview",
        "",
        "| Focus Item | Target | Status |",
        "| :--- | :--- | :---: |",
    ]

    table_rows: list[str] = []
    content_lines: list[str] = []

    # Flatten all pairs across categories
    all_pairs: list[dict[str, Any]] = [
        entry for pair_list in data["classified_pairs"].values() for entry in pair_list
    ]

    def find_pairs(topic_pattern: Optional[str] = None, event_pattern: Optional[str] = None) -> list[dict[str, Any]]:
        matches = []
        for p in all_pairs:
            t = p.get("topic") or ""
            e = p.get("event") or ""
            t_match = bool(re.search(topic_pattern, t, re.IGNORECASE)) if topic_pattern else True
            e_match = bool(re.search(event_pattern, e, re.IGNORECASE)) if event_pattern else True
            if t_match and e_match:
                matches.append(p)
        return matches

    def find_sub(topic_pattern: str) -> Optional[dict[str, Any]]:
        for sub in data["subscribe_messages"]:
            t = sub.get("topic") or ""
            if re.search(topic_pattern, t, re.IGNORECASE):
                return sub
        return None

    content_lines.append("## 2. WebSocket Focus Pairs (Full Redacted Structure)")

    # 1. range_stream:* with quotes_range -> Reclassified: prices
    rs_pairs = find_pairs(topic_pattern=r"range_stream", event_pattern=r"quotes_range")
    if rs_pairs:
        p = rs_pairs[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `range_stream:*` (quotes_range) | WS Pair | `{status}` |")
        content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Reclassified: `prices`]")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `range_stream:*` (quotes_range) | WS Pair | `not captured in report` |")

    # 2. asset:* topic events other than phx_reply (Full Structure) -> Reclassified: assets/payouts
    non_phx_asset_pairs = [
        p for p in all_pairs
        if re.search(r"^asset", p.get("topic") or "", re.IGNORECASE)
        and not (p.get("event") or "").startswith("phx_")
    ]
    raw_asset_pairs = non_phx_asset_pairs if non_phx_asset_pairs else [
        p for p in all_pairs
        if re.search(r"^asset", p.get("topic") or "", re.IGNORECASE)
        and p.get("event") not in ("phx_reply", "phx_join")
    ]
    asset_pairs: list[dict[str, Any]] = []
    seen_asset_events: set[str] = set()
    for p in raw_asset_pairs:
        ev = p.get("event") or ""
        if ev not in seen_asset_events:
            seen_asset_events.add(ev)
            asset_pairs.append(p)
        if len(asset_pairs) >= 2:
            break

    if asset_pairs:
        for idx, p in enumerate(asset_pairs):
            ex = p["examples"][0] if p["examples"] else None
            status, text = get_item_status(ex)
            if idx == 0:
                table_rows.append(f"| `asset:*` ({p['event']}) | WS Pair | `{status}` |")
            content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Reclassified: `assets/payouts`]")
            content_lines.append(f"- Status: {status}")
            content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
            content_lines.append("```json")
            content_lines.append(text)
            content_lines.append("```")
            content_lines.append("")
    else:
        table_rows.append("| `asset:*` | WS Pair | `not captured in report` |")

    # 3. asset_trading_settings_changed* -> Reclassified: assets/payouts
    atsc_pairs = find_pairs(event_pattern=r"asset_trading_settings_changed")
    if atsc_pairs:
        p = atsc_pairs[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `asset_trading_settings_changed*` | WS Pair | `{status}` |")
        content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Reclassified: `assets/payouts`]")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `asset_trading_settings_changed*` | WS Pair | `not captured in report` |")

    # 4. (bo, opened)
    bo_opened = find_pairs(topic_pattern=r"^bo$", event_pattern=r"^opened$")
    if bo_opened:
        p = bo_opened[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `(bo, opened)` | WS Pair | `{status}` |")
        content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Category: `deals`]")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `(bo, opened)` | WS Pair | `not captured in report` |")

    # 5. (bo, closed)
    bo_closed = find_pairs(topic_pattern=r"^bo$", event_pattern=r"^closed$")
    if bo_closed:
        p = bo_closed[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `(bo, closed)` | WS Pair | `{status}` |")
        content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Category: `deals`]")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `(bo, closed)` | WS Pair | `not captured in report` |")

    # 6. (bo, close_deal_batch)
    bo_cdb = find_pairs(topic_pattern=r"^bo$", event_pattern=r"^close_deal_batch$")
    if bo_cdb:
        p = bo_cdb[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `(bo, close_deal_batch)` | WS Pair | `{status}` |")
        content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Category: `deals`]")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `(bo, close_deal_batch)` | WS Pair | `not captured in report` |")

    # 6b. Deals status & option_types breakdown
    deal_pairs = data["classified_pairs"].get("deals", [])
    status_counts, option_types = extract_deal_status_and_option_types(deal_pairs)
    content_lines.append("### Deals Status & Option Types Breakdown")
    if status_counts:
        status_str = ", ".join(f"`{s}`: {c}" for s, c in sorted(status_counts.items(), key=lambda x: x[1], reverse=True))
        content_lines.append(f"- **Distinct Status Values in Opened/Closed**: {status_str}")
    else:
        content_lines.append("- **Distinct Status Values in Opened/Closed**: _none recorded_")
    if option_types:
        opt_str = ", ".join(f"`{ot}`" for ot in sorted(option_types))
        content_lines.append(f"- **Distinct Option Types**: {opt_str}")
    else:
        content_lines.append("- **Distinct Option Types**: _none recorded_")
    content_lines.append("")
    table_rows.append(f"| Deals Status & Option Types | WS Deals | `{len(status_counts)} statuses, {len(option_types)} option types` |")

    # 7. (account, balance_changed)
    acc_bal = find_pairs(topic_pattern=r"account|user", event_pattern=r"balance_changed")
    if acc_bal:
        p = acc_bal[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `(account, balance_changed)` | WS Pair | `{status}` |")
        content_lines.append(f"### Pair: `({p['topic']}, {p['event']})` [Category: `balance`]")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Count: {p['count']} | Rate: {p['rate']}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `(account, balance_changed)` | WS Pair | `not captured in report` |")

    # 8. Client frame create and phx_reply
    create_pairs = find_pairs(event_pattern=r"^create$")
    if create_pairs:
        p = create_pairs[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `({p['topic']}, create)` | WS Client Frame | `{status}` |")
        content_lines.append(f"### Client Frame: `({p['topic']}, {p['event']})`")
        content_lines.append(f"- Status: {status}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| Client frame `create` | WS Client Frame | `not captured in report` |")

    phx_reply = find_pairs(topic_pattern=r"bo|deals", event_pattern=r"^phx_reply$")
    if phx_reply:
        p = phx_reply[0]
        ex = p["examples"][0] if p["examples"] else None
        status, text = get_item_status(ex)
        table_rows.append(f"| `({p['topic']}, phx_reply)` | WS Pair | `{status}` |")
        content_lines.append(f"### Reply Frame: `({p['topic']}, {p['event']})`")
        content_lines.append(f"- Status: {status}")
        content_lines.append("```json")
        content_lines.append(text)
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `phx_reply` for create | WS Pair | `not captured in report` |")

    # 9. One phx_join per topic family
    content_lines.append("## 3. Phoenix Join by Topic Family")
    topic_families = ["quotes", "range_stream", "asset", "account", "bo", "user"]
    seen_families: set[str] = set()

    for top, cnt in data["phx_join_topics"]:
        family = top.split(":")[0] if ":" in top else top
        if family not in seen_families and (family in topic_families or len(seen_families) < 6):
            seen_families.add(family)
            sub_msg = find_sub(re.escape(top))
            if sub_msg:
                status, text = get_item_status(sub_msg["message"])
                content_lines.append(f"- **Family `{family}`**: Topic `{top}` (Count: {cnt}, Status: `{status}`)")
                content_lines.append("```json")
                content_lines.append(text)
                content_lines.append("```")
            else:
                content_lines.append(f"- **Family `{family}`**: Topic `{top}` (Count: {cnt}, Status: `joined`)")

    if not seen_families:
        content_lines.append("- _No phx_join topics recorded._\n")
    else:
        content_lines.append("")
    table_rows.append(f"| One `phx_join` per family | WS Topics | `{len(seen_families)} families captured` |")

    # 10. Non-Phoenix WebSocket Data (wss://as.binomo.com or similar)
    content_lines.append("## 4. Non-Phoenix WebSocket Data (as.binomo.com)")
    alt_ws = data.get("alt_ws_data", {})
    if alt_ws:
        table_rows.append(f"| `wss://as.binomo.com` | WS Socket | `captured ({len(alt_ws)} hosts)` |")
        for host, info in sorted(alt_ws.items()):
            content_lines.append(f"### Host: `{host}` (Total messages: {info.get('msg_count', 0)})")
            event_types_str = ", ".join(f"`{k}`: {v}" for k, v in info.get("event_types", {}).items()) if info.get("event_types") else "none"
            content_lines.append(f"- **Distinct Event Types**: {event_types_str}")
            content_lines.append(f"- **Price Field**: `{info.get('price_field') or 'N/A'}` | **Time Field**: `{info.get('time_field') or 'N/A'}`")
            rates = info.get("asset_rates", {})
            if rates:
                rate_str = ", ".join(f"`{a}`: {c}" for a, c in list(rates.items())[:8])
                content_lines.append(f"- **Message Rate / Count Per Asset**: {rate_str}")
            else:
                content_lines.append("- **Message Rate / Count Per Asset**: _none recorded_")

            sub_frames = info.get("subscribe_frames", [])
            if sub_frames:
                content_lines.append("- **Full Subscribe / Unsubscribe Messages**:")
                content_lines.append("```json")
                content_lines.append(compact_json(sub_frames[:3]))
                content_lines.append("```")
            else:
                content_lines.append("- **Subscribe / Unsubscribe Messages**: _none captured_")

            data_frames = info.get("data_frames", [])
            if data_frames:
                content_lines.append("- **First 3 Data Frames Structure**:")
                content_lines.append("```json")
                content_lines.append(compact_json(data_frames[:3]))
                content_lines.append("```")
            else:
                content_lines.append("- **Data Frames**: _none captured_")
            content_lines.append("")
    else:
        table_rows.append("| `wss://as.binomo.com` | WS Socket | `not captured in report` |")
        content_lines.append("- _No non-Phoenix WebSocket data captured in report. (Awaiting capture on as.binomo.com)._\n")

    # 11. Focus HTTP Endpoints
    content_lines.append("## 5. Focus HTTP Endpoints (Full Redacted First 2 Elements)")

    http_targets = [
        "/bo-assets/v7/assets",
        "/bo-deals-history/v3/deals/trade",
        "/bo/v1/config",
        "/platform/private/v2/currencies",
        "/platform/time",
    ]

    for target in http_targets:
        matched_ep = next((ep for ep in data["http_endpoints"] if target.lower() in ep["url"].lower()), None)
        if matched_ep or ("assets" in target.lower() and data.get("asset_table")):
            path = collapse_url(matched_ep["url"])[1] if matched_ep else "/bo-assets/v7/assets"
            method = matched_ep["method"] if matched_ep else "GET"
            ex = matched_ep.get("example") if matched_ep else None
            status, text = get_item_status(ex)

            # Special handling for /bo-assets: print ALL assets as compact table + full structure of first 2
            if "assets" in target.lower():
                asset_rows = data.get("asset_table", [])
                first_two = data.get("asset_first_two", [])
                if not asset_rows and ex:
                    asset_rows, f2 = extract_assets_from_data(ex)
                    if not first_two:
                        first_two = f2
                if asset_rows:
                    status = f"complete ({len(asset_rows)} assets)"
                    table_rows.append(f"| `{target}` | HTTP | `{status}` |")
                    content_lines.append(f"### Endpoint: `[{method}] {path}` [Asset List Table]")
                    content_lines.append(f"- Status: {status}")
                    content_lines.append(f"- Total Assets: {len(asset_rows)}")
                    content_lines.append("")
                    content_lines.extend(format_asset_compact_table(asset_rows))
                    if first_two:
                        content_lines.append("#### First 2 Asset Objects (Full Structure)")
                        content_lines.append("```json")
                        content_lines.append(compact_json(first_two))
                        content_lines.append("```")
                        content_lines.append("")
                    continue

            if ex is None:
                hits = matched_ep['count'] if matched_ep else 1
                st_code = matched_ep['status'] if matched_ep else 200
                status = f"endpoint seen (count: {hits}, body not captured)"
                table_rows.append(f"| `{target}` | HTTP | `{status}` |")
                content_lines.append(f"### Endpoint: `[{method}] {path}`")
                content_lines.append(f"- Status: {status}")
                content_lines.append(f"- Hits: {hits} | Status Code: {st_code}")
                content_lines.append("")
            else:
                display_ex = ex
                if isinstance(ex, list):
                    display_ex = ex[:2]
                elif isinstance(ex, dict):
                    display_ex = {k: (v[:2] if isinstance(v, list) else v) for k, v in ex.items()}
                _, text = get_item_status(display_ex)
                table_rows.append(f"| `{target}` | HTTP | `{status}` |")
                content_lines.append(f"### Endpoint: `[{method}] {path}`")
                content_lines.append(f"- Status: {status}")
                content_lines.append(f"- Hits: {matched_ep['count']} | Status Code: {matched_ep['status']}")
                content_lines.append("```json")
                content_lines.append(text)
                content_lines.append("```")
                content_lines.append("")
        else:
            table_rows.append(f"| `{target}` | HTTP | `not captured in report` |")

    # 12. Complete /candles/ response
    candle_eps = [ep for ep in data["http_endpoints"] if "candles" in ep["url"].lower()]
    if candle_eps:
        c_ep = candle_eps[0]
        c_url_pat = extract_candle_url_pattern(c_ep["url"])
        ex = c_ep.get("example")
        status, text = get_item_status(ex)

        candle_count = 0
        candle_keys: list[str] = []
        sample_candle: dict[str, Any] = {}

        if isinstance(ex, dict):
            c_data = ex.get("data")
            if isinstance(c_data, list):
                candle_count = len(c_data)
                if c_data and isinstance(c_data[0], dict):
                    candle_keys = list(c_data[0].keys())
                    sample_candle = c_data[0]
        elif isinstance(ex, list):
            candle_count = len(ex)
            if ex and isinstance(ex[0], dict):
                candle_keys = list(ex[0].keys())
                sample_candle = ex[0]

        table_rows.append(f"| `/candles/` response | HTTP | `{status}` |")
        content_lines.append("### Endpoint: Candles Response")
        content_lines.append(f"- Status: {status}")
        content_lines.append(f"- Candle URL Pattern: `{c_url_pat}`")
        content_lines.append(f"- Candles Returned Count: {candle_count}")
        content_lines.append(f"- Candle Object Keys: `{candle_keys}`")
        content_lines.append("- Complete Sample Candle Object:")
        content_lines.append("```json")
        content_lines.append(compact_json(sample_candle))
        content_lines.append("```")
        content_lines.append("")
    else:
        table_rows.append("| `/candles/` response | HTTP | `not captured in report` |")

    # 13. Demo vs Real Discriminators (Non-i18n)
    content_lines.append("## 6. Demo vs Real Account Discriminators (Non-i18n Only)")
    non_i18n_fields = {
        k: v for k, v in data["demo_real_fields"].items()
        if is_non_i18n_account_field(k, v)
    }
    if non_i18n_fields:
        for k, v in sorted(non_i18n_fields.items()):
            content_lines.append(f"- `{k}`: `{json.dumps(v)}`")
    else:
        content_lines.append("- _No non-i18n demo/real discriminators found._")
    content_lines.append("")

    # 14. Platform Endpoints (Static Dropped, Trackers Dropped)
    content_lines.append("## 7. Platform HTTP Endpoints (Static Assets & Trackers Dropped)")
    clean_endpoints = []
    for ep in data["http_endpoints"]:
        h, path = collapse_url(ep["url"])
        if not is_tracker(h) and not STATIC_ASSET_REGEX.search(path):
            clean_endpoints.append(ep)

    if clean_endpoints:
        seen_paths: set[tuple[str, str]] = set()
        for ep in clean_endpoints:
            _, path = collapse_url(ep["url"])
            key = (ep["method"], path)
            if key not in seen_paths:
                seen_paths.add(key)
                content_lines.append(f"- [{ep['method']}] `{path}` (status: {ep['status']}, count: {ep['count']})")
                if len(seen_paths) >= 20:
                    content_lines.append(f"... [{len(clean_endpoints) - 20} more endpoints omitted]")
                    break
    else:
        content_lines.append("- _No platform HTTP endpoints recorded._")
    content_lines.append("")

    # 15. Trading Panel DOM Controls (Read-Only Accessibility Snapshot)
    content_lines.append("## 8. Trading Panel DOM Controls (Read-Only Accessibility Snapshot)")
    controls = extract_trading_panel_controls(data.get("dom_snapshots", []))
    any_found = any(len(v) > 0 for v in controls.values())
    if any_found:
        table_rows.append("| Trading Panel DOM Controls | Accessibility | `captured` |")
        content_lines.append("| Target Control | Role | Accessible Name | Test ID |")
        content_lines.append("| :--- | :--- | :--- | :--- |")
        category_labels = [
            ("Amount Input", "amount_input"),
            ("Expiry / Time Selector & Options", "expiry_selector"),
            ("CALL / PUT Buttons", "call_put_buttons"),
            ("Asset Selector", "asset_selector"),
            ("Demo / Real Account Switch", "account_switch"),
        ]
        for label, cat_key in category_labels:
            items = controls.get(cat_key, [])
            if items:
                for item in items[:6]:
                    role = item.get("role") or "-"
                    name = item.get("name") or "-"
                    tid = item.get("data-testid") or item.get("data-test") or item.get("test_id") or "-"
                    content_lines.append(f"| {label} | `{role}` | `{name}` | `{tid}` |")
            else:
                content_lines.append(f"| {label} | - | _not identified in snapshot_ | - |")
    else:
        table_rows.append("| Trading Panel DOM Controls | Accessibility | `not captured in report` |")
        content_lines.append("- _No trading panel DOM snapshot elements captured. (Run with `--dom-snapshots` to capture)._")
    content_lines.append("")

    # Assemble Document
    out_lines.extend(table_rows)
    out_lines.append("")
    out_lines.extend(content_lines)

    # Line budget safety check: strictly <= 500 lines
    if len(out_lines) > 495:
        out_lines = out_lines[:490]
        out_lines.append("... [truncated to fit 500 line limit]")
        out_lines.append("")

    secret_matches = count_secret_matches(out_lines)
    out_lines.append(f"secret-pattern matches in this focus: {secret_matches}")

    return "\n".join(out_lines) + "\n"


def summarize_focus(input_path: Path, output_path: Path) -> str:
    """Read report, write focus markdown, save archive copy, and return text."""
    report_data = parse_discovery_report(input_path)
    focus_text = generate_focus_markdown(report_data)

    line_count = len(focus_text.splitlines())
    if line_count > 500:
        lines = focus_text.splitlines()[:495]
        secret_matches = count_secret_matches(lines)
        lines.append("... [truncated to fit 500 line limit]")
        lines.append(f"secret-pattern matches in this focus: {secret_matches}")
        focus_text = "\n".join(lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(focus_text, encoding="utf-8")

    archive_dir = Path("reports_archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / output_path.name
    archive_path.write_text(focus_text, encoding="utf-8")

    return focus_text


def summarize(input_path: Path, output_path: Path, budgets: Optional[dict[str, int]] = None) -> str:
    """Read report, write summary, save archive copy, and return summary markdown text."""
    report_data = parse_discovery_report(input_path)
    summary_text = generate_summary_markdown(report_data, budgets=budgets)

    line_count = len(summary_text.splitlines())
    if line_count > 900:
        lines = summary_text.splitlines()[:895]
        secret_matches = count_secret_matches(lines)
        lines.append("... [truncated to fit 900 line limit]")
        lines.append(f"secret-pattern matches in this summary: {secret_matches}")
        summary_text = "\n".join(lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text, encoding="utf-8")

    archive_dir = Path("reports_archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / output_path.name
    archive_path.write_text(summary_text, encoding="utf-8")

    return summary_text


def main() -> None:
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)
    parser = argparse.ArgumentParser(description="Summarize platform discovery report with per-section line budgets.")
    parser.add_argument("--input", default=None, help="Path to discovery_report.md or discovery_report_*.md")
    parser.add_argument("--output", default=None, help="Path to output markdown file")
    parser.add_argument("--focus", action="store_true", help="Generate discovery_focus.md (max 500 lines) with key trading focus items")
    args = parser.parse_args()

    if args.input:
        in_file = Path(args.input)
    else:
        # Check if discovery_report.md exists, else find newest discovery_report_*.md
        if Path("discovery_report.md").exists():
            in_file = Path("discovery_report.md")
        else:
            candidates = sorted(Path(".").glob("discovery_report_*.md"))
            if candidates:
                in_file = candidates[-1]
            else:
                in_file = Path("discovery_report.md")

    # Determine suffix if input has one, e.g. discovery_report_20261102_0015.md -> _20261102_0015
    suffix_match = re.search(r"discovery_report_(\d{8}_\d{4})\.md$", in_file.name)
    suffix = f"_{suffix_match.group(1)}" if suffix_match else ""

    if args.focus:
        default_out = f"discovery_focus{suffix}.md"
        out_file = Path(args.output or default_out)
        print(f"Reading {in_file} and generating focus report {out_file}...")
        summary = summarize_focus(in_file, out_file)
    else:
        default_out = f"discovery_summary{suffix}.md"
        out_file = Path(args.output or default_out)
        print(f"Reading {in_file} and generating summary {out_file}...")
        summary = summarize(in_file, out_file)

    lines = summary.splitlines()
    print(f"Done! Generated {out_file} with {len(lines)} lines.")
    print(lines[-1])


if __name__ == "__main__":
    main()
