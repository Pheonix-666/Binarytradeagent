"""Platform discovery tool — inspects platform traffic and DOM read-only.

Usage:
    python tools/discover_platform.py --url https://platform.example.com [--duration 900]

Safety Rule:
    This tool NEVER clicks, fills, types, presses, checks, or evaluates anything.
    It is strictly READ-ONLY. It launches a persistent Playwright browser profile,
    attaches listeners to all pages, monitors network requests and WebSocket frames,
    redacts all credentials and identifiers, and compiles discovery_report.md.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
import urllib.parse
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)


PROFILE_DIR = Path("state") / "playwright_profile"
REPORT_PATH = Path("discovery_report.md")

# ── Redaction Configuration ───────────────────────────────────────────────────

# Leaf keys that are NEVER masked (string values safe to show)
SAFE_STRING_KEYS = {
    "topic", "event", "event_type", "action", "ric", "asset_ric",
    "asset", "symbol", "currency", "status", "type", "kind", "mode",
    "deal_type", "trend", "direction",
    # Pass 3 additions
    "option_type", "asset_name", "payment_rate", "min_amount", "max_amount",
    "min_bet", "max_bet", "step", "expiration", "expirations", "durations",
    "expire_in", "interval", "timeframe", "trading_status", "is_active",
    "is_open", "schedule",
}

# Numeric keys that keep their values
SAFE_NUMERIC_KEYS = {
    "rate", "price", "quote", "bid", "ask", "open", "high", "low", "close",
    "value", "payout", "payment_rate", "profit", "percent",
    "ts", "time", "created_at", "open_time", "close_time",
    "expire_at", "expiration", "duration",
    # Pass 3 additions
    "payment_rate", "min_amount", "max_amount", "min_bet", "max_bet",
    "step", "expire_in", "expirations", "durations", "interval", "timeframe",
    "is_active", "is_open",
}

# Pass 4 Safe-list: ancestors and key patterns that allow numeric values to show
SAFE_ANCESTOR_REGEX = re.compile(
    r"^(limits|summs|schedule|enabled_.*)$",
    re.IGNORECASE,
)

SAFE_NUMERIC_PATTERN = re.compile(
    r"^(base_payment_rate_.*|payment_rate|min.*|max.*|step|duration.*|expire.*|enabled_.*)$",
    re.IGNORECASE,
)

# Keys that identify demo vs real accounts
DEMO_REAL_KEY_REGEX = re.compile(r"demo|real|account_type", re.IGNORECASE)

# Sensitive keys that trigger whole-subtree masking
ALWAYS_MASK_KEY_REGEX = re.compile(
    r"token|auth|session|cookie|password|secret|key|email|phone|user|uid|ssid|jwt|\baccount\b",
    re.IGNORECASE,
)

ASSET_ANCESTOR_REGEX = re.compile(
    r"asset|instrument|active|pair|symbol|ticker|market",
    re.IGNORECASE,
)

BALANCE_KEY_REGEX = re.compile(r"balance", re.IGNORECASE)

TRACKER_HOSTS_REGEX = re.compile(
    r"doubleclick|google-analytics|googletagmanager|googleads|adservice|"
    r"facebook|hotjar|clarity|appsflyer|adjust|segment|mixpanel|sentry|"
    r"analytics\.google|accounts\.google|criteo|yandex|app-measurement",
    re.IGNORECASE,
)

SKIP_URL_REGEX = re.compile(
    r"login|auth|token|session|signin|password|oauth",
    re.IGNORECASE,
)

I18N_URL_REGEX = re.compile(r"locale|i18n|translations|lang", re.IGNORECASE)

# Secret pattern regexes
JWT_REGEX = re.compile(r"^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}$")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LONG_HEX_B64_REGEX = re.compile(r"^[A-Za-z0-9+/=_-]{25,}$")
UUID_REGEX = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

IDENTIFIER_CHAR_REGEX = re.compile(r"^[A-Za-z0-9_./:-]+$")
SEPARATOR_REGEX = re.compile(r"[_./:-]")


def is_identifier_string(val: str) -> bool:
    """Check if string is a safe identifier with separators (e.g. Z-CRY/IDX, asset_trading_settings_changed_v2)."""
    if not isinstance(val, str) or not val:
        return False
    if not IDENTIFIER_CHAR_REGEX.match(val):
        return False
    if not SEPARATOR_REGEX.search(val):
        return False
    if UUID_REGEX.match(val):
        return False
    if JWT_REGEX.match(val) or EMAIL_REGEX.search(val) or IP_REGEX.search(val):
        return False
    segments = SEPARATOR_REGEX.split(val)
    for seg in segments:
        if len(seg) > 24:
            return False
        if len(seg) >= 16 and re.fullmatch(r"[0-9a-fA-F]{16,}", seg):
            return False
    return True


def is_i18n_payload(data: Any, url: str = "") -> bool:
    """Detect translation/i18n JSON (mostly string leaves, deep nesting of words, URLs with i18n/locale)."""
    if url and I18N_URL_REGEX.search(url):
        return True
    if isinstance(data, dict):
        keys_lower = {str(k).lower() for k in data.keys()}
        if any(k in keys_lower for k in ["i18n", "translations", "locale", "locales", "messages"]):
            return True
        leaves: list[Any] = []
        def get_leaves(d: Any) -> None:
            if isinstance(d, dict):
                for v in d.values():
                    get_leaves(v)
            elif isinstance(d, list):
                for v in d:
                    get_leaves(v)
            else:
                leaves.append(d)
        get_leaves(data)
        if len(leaves) >= 20:
            str_leaves = sum(1 for x in leaves if isinstance(x, str))
            if str_leaves / len(leaves) > 0.85:
                if any(len(str(x).split()) >= 3 for x in leaves if isinstance(x, str)):
                    return True
    return False


def classify_message(topic: str, event: str, is_i18n: bool = False) -> str:
    """Classify (topic, event) pair into 8 canonical categories."""
    if is_i18n:
        return "i18n"

    t_lower = topic.lower()
    e_lower = event.lower()

    # 1. System
    if (
        e_lower.startswith("phx_")
        or e_lower in {"heartbeat", "ping", "pong"}
        or t_lower in {"phoenix", "system"}
    ):
        return "system"

    # 2. Social
    if (
        t_lower.startswith("social")
        or t_lower.startswith("chat")
        or t_lower.startswith("tournament")
        or e_lower in {"social", "chat", "tournament", "leaderboard"}
    ):
        return "social"

    # 3. Prices / Quotes
    if (
        t_lower.startswith("quotes:")
        or t_lower.startswith("quote:")
        or t_lower.startswith("candles:")
        or t_lower.startswith("ticks:")
        or t_lower.startswith("price:")
        or t_lower == "quotes"
        or e_lower in {"quote", "quotes", "tick", "ticks", "candles", "candle", "price", "rates"}
    ):
        return "prices/quotes"

    # 4. Assets / Payouts
    if (
        t_lower in {"assets", "instruments", "payouts"}
        or t_lower.startswith("assets:")
        or t_lower.startswith("payout:")
        or e_lower in {
            "asset_trading_settings_changed", "asset_trading_settings_changed_v2",
            "payout_changed", "assets", "instruments", "payouts", "update_payout",
        }
    ):
        return "assets/payouts"

    # 5. Balance
    if (
        t_lower in {"balance", "user_balance"}
        or t_lower.startswith("balance:")
        or (t_lower.startswith("user:") and "balance" in e_lower)
        or e_lower in {"balance_changed", "balance_update", "balance", "balance_updated"}
    ):
        return "balance"

    # 6. Deals
    if (
        e_lower.startswith("deal_")
        or e_lower.startswith("deals_")
        or e_lower.startswith("close_deal")
        or e_lower.startswith("create_deal")
        or e_lower in {"deal", "deals", "order", "orders", "create_deal", "opened", "closed"}
        or t_lower.startswith("deals")
        or t_lower.startswith("orders")
    ):
        return "deals"

    return "other"


# DOM keywords to match for trading panel elements
DOM_KEYWORDS_REGEX = re.compile(
    r"\b(call|put|up|down|higher|lower|amount|investment|expiry|time|duration|asset|demo|real|payout|profit|balance|trade)\b",
    re.IGNORECASE,
)

PROFILE_MENU_REGEX = re.compile(
    r"profile|avatar|user_menu|main_menu|nav_menu|settings|security",
    re.IGNORECASE,
)

FOCUS_ENDPOINTS_REGEX = re.compile(
    r"/bo-assets/v7/assets|/bo-deals-history/v3/deals/trade|/bo/v1/config|"
    r"/platform/private/v2/currencies|/platform/time|/candles/",
    re.IGNORECASE,
)

FOCUS_TOPICS_REGEX = re.compile(
    r"^(range_stream|asset|bo|account|user|quotes)",
    re.IGNORECASE,
)


def mask_dom_text(text: str) -> str:
    """Mask email, phone, balances, and digit runs longer than 3 while keeping durations like '5 min' or '60 s'."""
    if not text:
        return ""
    text = EMAIL_REGEX.sub("<email>", text)
    text = PHONE_REGEX.sub("<phone>", text)
    # Mask currency amounts like $500, $10,500.00, €50
    text = re.sub(r"[\$€£₹]\s*\d{1,3}(,\d{3})*(\.\d+)?", "<num>", text)
    # Mask formatted numbers like 10,500.00 or 1,000
    text = re.sub(r"\b\d{1,3}(,\d{3})+(\.\d+)?", "<num>", text)
    # Mask any string of digits longer than 3
    text = re.sub(r"\d{4,}", "<num>", text)
    return text


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
        path = path

    query_names = []
    if parsed.query:
        query_names = [k for k, _ in urllib.parse.parse_qsl(parsed.query)]
    q_str = ("?" + "&".join(query_names)) if query_names else ""
    return f"{parsed.scheme or 'https'}://{parsed.netloc or 'api.binomo.com'}{path}{q_str}"


# Backward compatibility helpers
CATEGORY_KEYS = {
    "balance": {"balance", "balances", "demo_balance", "live_balance", "account_balance"},
    "trade/order": {"order", "orders", "trade", "trades", "deal", "deals", "position", "positions"},
    "payout": {"payout", "payouts", "profit_percent", "yield", "rate_payout"},
    "price": {"price", "bid", "ask", "quote", "candles", "candle", "ohlc", "rates", "rate", "tick", "ticks"},
    "assets": {"assets", "instruments", "actives", "pairs", "currencies"},
}


def classify_category_from_keys(keys: set[str]) -> str:
    """Determine message category based on key names (Pass 1 compatibility)."""
    lower_keys = {k.lower() for k in keys}
    for cat in ["balance", "trade/order", "payout", "price", "assets"]:
        if any(ck in lower_keys for ck in CATEGORY_KEYS[cat]):
            return cat
    return "other"


def extract_message_type(data: Any) -> str:
    """Identify message type from common discriminator keys (Pass 1 compatibility)."""
    if isinstance(data, dict):
        for candidate in ["type", "event", "action", "cmd", "topic", "e"]:
            val = data.get(candidate)
            if val is not None and isinstance(val, (str, int)):
                return str(val)
        if data:
            return str(list(data.keys())[0])
    elif isinstance(data, list) and data:
        return extract_message_type(data[0])
    return "unknown"


# ── Payload Redactor ─────────────────────────────────────────────────────────

class PayloadRedactor:
    """Recursively redacts JSON data while preserving safe fields and structural integrity."""

    def __init__(self) -> None:
        self.redaction_count = 0

    def is_sensitive_key(self, key: str) -> bool:
        """Check if key triggers sensitive subtree masking (exempting demo/real indicators)."""
        k_lower = key.lower()
        if DEMO_REAL_KEY_REGEX.search(k_lower):
            return False
        return bool(ALWAYS_MASK_KEY_REGEX.search(k_lower))

    def is_sensitive_value(self, val: str) -> bool:
        """Check if a string value matches sensitive patterns like JWT, email, phone, IP, or long token."""
        if is_identifier_string(val):
            return False
        if JWT_REGEX.match(val):
            return True
        if EMAIL_REGEX.search(val):
            return True
        if IP_REGEX.search(val):
            return True
        if len(val) > 24 and LONG_HEX_B64_REGEX.match(val):
            return True
        if len(val) >= 9 and PHONE_REGEX.search(val) and sum(c.isdigit() for c in val) >= 7:
            return True
        return False

    def redact(
        self,
        data: Any,
        parent_key: str = "",
        ancestors: tuple[str, ...] = (),
        in_sensitive_subtree: bool = False,
        depth: int = 0,
    ) -> Any:
        """Recursively redact JSON data structure up to depth 6, shortening lists to first 2 items."""
        if depth >= 6:
            if isinstance(data, dict):
                return {k: "<obj>" for k in list(data.keys())[:2]}
            if isinstance(data, list):
                return [
                    self.redact(item, parent_key, ancestors, in_sensitive_subtree, depth + 1)
                    for item in data[:2]
                ]
            return self.redact_value(data, parent_key, ancestors, in_sensitive_subtree)

        if isinstance(data, dict):
            redacted_dict: dict[str, Any] = {}
            for k, v in data.items():
                k_str = str(k)
                is_sensitive = self.is_sensitive_key(k_str)
                subtree_sensitive = in_sensitive_subtree or is_sensitive
                child_ancestors = ancestors + (k_str,)

                if isinstance(v, (dict, list)):
                    redacted_dict[k] = self.redact(
                        v,
                        parent_key=k_str,
                        ancestors=child_ancestors,
                        in_sensitive_subtree=subtree_sensitive,
                        depth=depth + 1,
                    )
                else:
                    redacted_dict[k] = self.redact_value(
                        val=v,
                        parent_key=k_str,
                        ancestors=ancestors,
                        in_sensitive_subtree=subtree_sensitive,
                    )
            return redacted_dict

        if isinstance(data, list):
            return [
                self.redact(
                    item,
                    parent_key=parent_key,
                    ancestors=ancestors,
                    in_sensitive_subtree=in_sensitive_subtree,
                    depth=depth + 1,
                )
                for item in data[:2]
            ]

        return self.redact_value(
            val=data,
            parent_key=parent_key,
            ancestors=ancestors,
            in_sensitive_subtree=in_sensitive_subtree,
        )

    def redact_value(
        self,
        val: Any,
        parent_key: str,
        ancestors: tuple[str, ...],
        in_sensitive_subtree: bool,
    ) -> Any:
        """Redact a scalar leaf value based on key rules, ancestors, and sensitive context."""
        k_lower = parent_key.lower()

        # 0. Demo / Real indicators: preserved with values unless it's a secret value
        if DEMO_REAL_KEY_REGEX.search(k_lower):
            if isinstance(val, (int, float)) and not isinstance(val, bool) and BALANCE_KEY_REGEX.search(k_lower):
                self.redaction_count += 1
                return "<num>"
            if isinstance(val, str) and self.is_sensitive_value(val):
                self.redaction_count += 1
                return f"<str:{len(val)}>"
            return val

        # 1. Sensitive subtree (ancestor is sensitive, e.g. user, token, password...)
        if in_sensitive_subtree:
            self.redaction_count += 1
            if isinstance(val, str):
                return f"<str:{len(val)}>"
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                return "<num>"
            elif isinstance(val, bool):
                return "<masked>"
            else:
                return "<masked>"

        # 2. String values
        if isinstance(val, str):
            if self.is_sensitive_value(val):
                self.redaction_count += 1
                return f"<str:{len(val)}>"

            # Leaf keys that are NEVER masked
            if k_lower in SAFE_STRING_KEYS or SAFE_NUMERIC_PATTERN.match(k_lower):
                return val

            # Pass 4: strings under safe ancestors (e.g. schedule status or enabled_account_types)
            if any(SAFE_ANCESTOR_REGEX.match(a.lower()) for a in ancestors):
                return val

            # Key 'name' is only shown when an ancestor key looks like an asset — but
            # asset_name is always safe (it IS an asset name).
            if k_lower == "asset_name":
                return val

            # Key 'name' is only shown when an ancestor key looks like an asset
            if k_lower == "name":
                has_asset_ancestor = any(ASSET_ANCESTOR_REGEX.search(a) for a in ancestors)
                if has_asset_ancestor:
                    return val
                else:
                    self.redaction_count += 1
                    return f"<str:{len(val)}>"

            # Identifier strings (e.g. Z-CRY/IDX, asset_trading_settings_changed_v2)
            if is_identifier_string(val):
                return val

            self.redaction_count += 1
            return f"<str:{len(val)}>"

        # 3. Numeric values
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            # Balance numbers are always masked to protect financial privacy
            if BALANCE_KEY_REGEX.search(k_lower):
                self.redaction_count += 1
                return "<num>"

            # IDs are masked
            if k_lower == "id" or k_lower.endswith("_id") or "uid" in k_lower or k_lower == "uuid":
                self.redaction_count += 1
                return "<num>"

            # Pass 4 Safe Ancestors: limits, summs, schedule, enabled_*
            if any(SAFE_ANCESTOR_REGEX.match(a.lower()) for a in ancestors):
                return val

            # Pass 4 Safe Numeric Patterns: base_payment_rate_*, payment_rate, min*, max*, step, duration*, expire*, enabled_*
            if SAFE_NUMERIC_PATTERN.match(k_lower):
                return val

            # Safe numeric keys (price, rate, timestamps, payout, etc.)
            if k_lower in SAFE_NUMERIC_KEYS:
                return val

            self.redaction_count += 1
            return "<num>"

        # 4. Other types (bool, None)
        return val


# ── Helper for Payload Parsing ───────────────────────────────────────────────

def extract_phoenix_or_json(data: Any) -> tuple[str, str, Any]:
    """Extract (topic, event, payload) from Phoenix array or standard dict."""
    if isinstance(data, list) and len(data) >= 4 and isinstance(data[2], str) and isinstance(data[3], str):
        topic = data[2]
        event = data[3]
        payload = data[4] if len(data) > 4 else {}
        return topic, event, payload
    elif isinstance(data, dict):
        topic = str(data.get("topic") or data.get("channel") or "")
        event = str(data.get("event") or data.get("type") or data.get("action") or "")
        payload = data.get("payload") or data.get("data") or data
        return topic, event, payload
    return "", str(data)[:30], data


# ── Main Discovery Class ──────────────────────────────────────────────────────

class PlatformDiscovery:
    """Strictly read-only platform inspector using Playwright."""

    def __init__(
        self,
        target_url: str,
        duration_seconds: int = 900,
        dom_snapshots: bool = False,
        report_path: Optional[Path | str] = None,
    ) -> None:
        self.target_url = target_url
        self.duration_seconds = duration_seconds
        self.dom_snapshots_enabled = dom_snapshots
        self.redactor = PayloadRedactor()

        # Pass 4 Never Overwrite report filenames
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        if report_path:
            self.report_path = Path(report_path)
            self.archive_path = Path("reports_archive") / self.report_path.name
        else:
            self.report_path = Path(f"discovery_report_{ts}.md")
            self.archive_path = Path("reports_archive") / f"discovery_report_{ts}.md"

        # Pass 4 Login detection indicators
        self.open_websockets: set[str] = set()
        self.seen_passport_or_assets: bool = False
        self.seen_account_join: bool = False
        self.seen_ws_binomo_open: bool = False

        # Pass 4 WebSocket frame counts per socket & iframe tracking
        self.socket_frame_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0, "total": 0})
        self.attached_iframes: set[str] = set()

        # Listeners & Traffic Tracking
        self._attached_pages: set[Any] = set()
        self.ws_endpoints: set[str] = set()
        self.http_endpoints: set[str] = set()

        # Binary frames tracking
        self.binary_frame_count: int = 0
        self.binary_samples: list[dict[str, Any]] = []

        # Message type classification (Pass 1 compatibility)
        self.classified_types: dict[str, dict[str, Any]] = {}

        # Phoenix & Message tracking
        self.phx_join_topics: Counter[str] = Counter()
        self.subscribe_messages: list[dict[str, Any]] = []
        self.pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
        self.outgoing_frames: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.i18n_count: int = 0
        self.social_count: int = 0

        # Large data frames inspection (e.g. candle/tick arrays)
        self.large_data_frames: list[dict[str, Any]] = []

        # Demo vs Real account fields
        self.demo_real_fields: dict[str, Any] = {}

        # DOM snapshots
        self.dom_snapshots: list[dict[str, Any]] = []

        # HTTP Endpoint details: collapsed_path -> {host, method, status, count, example}
        self.http_endpoint_details: dict[str, dict[str, Any]] = {}

        # Dropped tracker and capacity counters
        self.dropped_tracker_count: int = 0
        self.dropped_endpoint_cap_count: int = 0
        self.max_endpoint_cap: int = 2000
        self.max_examples_per_type: int = 3

        # Asset table: list of compact asset rows extracted from /bo-assets or similar
        self.asset_table: list[dict[str, Any]] = []
        self.asset_first_two: list[dict[str, Any]] = []
        # Captured non-Phoenix WS (e.g. wss://as.binomo.com): host -> data
        self.alt_ws_data: dict[str, dict[str, Any]] = {}

    def is_tracker_url(self, url: str) -> bool:
        """Check if URL host matches ad/analytics/tracker domains."""
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc.lower()
            return bool(TRACKER_HOSTS_REGEX.search(host))
        except Exception:
            return False

    def collapse_url_path(self, raw_url: str) -> str:
        """Collapse numbers and UUIDs in path, stripping queries and ';' params,
        preserving final numeric segment (timeframe) and query param names for /candles/."""
        clean = self.strip_url(raw_url)
        parsed = urllib.parse.urlparse(clean)
        path = parsed.path or "/"

        if "/candles/" in path.lower():
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
                raw_parsed = urllib.parse.urlparse(raw_url)
                q_names = [k for k, _ in urllib.parse.parse_qsl(raw_parsed.query)] if raw_parsed.query else []
                q_str = ("?" + "&".join(q_names)) if q_names else ""
                return f"{parsed.scheme}://{parsed.netloc}{prefix}{timeframe}{q_str}"

        path = re.sub(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "{uuid}",
            path,
        )
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def strip_url(self, raw_url: str) -> str:
        """Strip query parameters, fragments, and matrix parameters from URL."""
        return raw_url.split("?")[0].split("#")[0].split(";")[0]

    def check_login_status(self) -> dict[str, bool]:
        """Check the three required login indicators (observation only)."""
        ws_open = any("ws.binomo.com" in u for u in self.open_websockets) or self.seen_ws_binomo_open
        return {
            "ws_binomo_open": ws_open,
            "seen_passport_or_assets": self.seen_passport_or_assets,
            "seen_account_join": self.seen_account_join,
            "logged_in": bool(ws_open and self.seen_passport_or_assets and self.seen_account_join),
        }

    def _extract_asset_table(self, body: Any, url: str) -> None:
        """Extract ALL assets from /bo-assets or JSON arrays of assets with Pass 4 fields."""
        asset_list: list[Any] = []
        if isinstance(body, list):
            asset_list = body
        elif isinstance(body, dict):
            if isinstance(body.get("data"), dict) and isinstance(body["data"].get("assets"), list):
                asset_list = body["data"]["assets"]
            elif isinstance(body.get("data"), list):
                asset_list = body["data"]
            elif isinstance(body.get("assets"), list):
                asset_list = body["assets"]
            else:
                for v in body.values():
                    if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
                        asset_list = v
                        break

        if not asset_list or not isinstance(asset_list[0], dict):
            return

        if not self.asset_first_two:
            self.asset_first_two = [self.redactor.redact(item, depth=0) for item in asset_list[:2]]

        self.asset_table = []
        for item in asset_list:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("asset_name") or "-")
            ric = str(item.get("ric") or item.get("symbol") or "-")
            a_type = str(item.get("type") or item.get("kind") or "-")

            # OTC flag: ric starts with "Z-" or name contains "(OTC)"
            is_otc = bool(ric.startswith("Z-") or "(otc)" in name.lower() or "(OTC)" in name or item.get("is_otc"))

            tts = item.get("trading_tools_settings") if isinstance(item.get("trading_tools_settings"), dict) else {}
            ftt = tts.get("ftt") if isinstance(tts.get("ftt"), dict) else {}

            # enabled_account_types
            enabled_acc = (
                item.get("enabled_account_types")
                or ftt.get("enabled_account_types")
                or []
            )
            if isinstance(enabled_acc, list):
                enabled_acc_str = ", ".join(str(x) for x in enabled_acc) if enabled_acc else "-"
            else:
                enabled_acc_str = str(enabled_acc)

            # enabled_trading_types
            enabled_tt = (
                item.get("enabled_trading_types")
                or list(tts.keys())
                or []
            )
            if isinstance(enabled_tt, list):
                enabled_tt_str = ", ".join(str(x) for x in enabled_tt) if enabled_tt else "-"
            else:
                enabled_tt_str = str(enabled_tt)

            # Three base_payment_rate_* values
            r_turbo = ftt.get("base_payment_rate_turbo") or item.get("payment_rate") or "-"
            r_standard = ftt.get("base_payment_rate_standard") or item.get("base_payment_rate_standard") or "-"
            r_classic = ftt.get("base_payment_rate_classic") or item.get("base_payment_rate_classic") or "-"
            rates_str = f"turbo: {r_turbo} | std: {r_standard} | classic: {r_classic}"

            # Today's schedule
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

            self.asset_table.append({
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
                # Compatibility fields
                "payment_rate": r_turbo if r_turbo != "-" else (item.get("payment_rate") or "-"),
                "status": ftt.get("trading_status") or item.get("trading_status") or item.get("status") or "-",
            })

    def _extract_demo_real_fields(self, data: Any) -> None:
        """Scan data structure for keys matching demo, real, or account_type and record values."""
        if isinstance(data, dict):
            for k, v in data.items():
                k_str = str(k)
                if DEMO_REAL_KEY_REGEX.search(k_str):
                    if not isinstance(v, (dict, list)):
                        safe_val = self.redactor.redact_value(v, k_str, (), False)
                        self.demo_real_fields[k_str] = safe_val
                if isinstance(v, (dict, list)):
                    self._extract_demo_real_fields(v)
        elif isinstance(data, list):
            for item in data:
                self._extract_demo_real_fields(item)

    def _inspect_large_data_frame(self, topic: str, raw_data: Any) -> None:
        """Detect large data frames (e.g. quote/candle arrays) and identify price vs time fields."""
        items: list[Any] = []
        if isinstance(raw_data, list):
            items = raw_data
        elif isinstance(raw_data, dict):
            for val in raw_data.values():
                if isinstance(val, list) and len(val) >= 2:
                    items = val
                    break

        if items and len(items) >= 2 and isinstance(items[0], dict):
            sample_item = items[0]
            price_field = ""
            time_field = ""
            for k in sample_item.keys():
                k_lower = str(k).lower()
                if not price_field and k_lower in {"price", "rate", "quote", "close", "c", "bid", "ask"}:
                    price_field = str(k)
                if not time_field and k_lower in {"time", "timestamp", "ts", "t", "created_at"}:
                    time_field = str(k)

            if price_field or time_field:
                # Store one inspected frame per topic
                if not any(f["topic"] == topic for f in self.large_data_frames):
                    redacted_sample = self.redactor.redact(items[:2])
                    self.large_data_frames.append({
                        "topic": topic,
                        "price_field": price_field or "N/A",
                        "time_field": time_field or "N/A",
                        "structure": f"Array of objects with keys: {list(sample_item.keys())[:8]}",
                        "sample": redacted_sample,
                    })

    def _process_text_frame(self, raw_text: str, source_label: str, clean_url: str) -> None:
        """Parse, redact, and classify a WebSocket or HTTP text payload."""
        try:
            data = json.loads(raw_text)
        except Exception:
            self.redactor.redaction_count += 1
            data = {"raw_text": f"<str:{len(raw_text)}>"}

        # Extract demo/real fields
        self._extract_demo_real_fields(data)

        # Check for i18n JSON noise
        if is_i18n_payload(data, clean_url):
            self.i18n_count += 1
            return

        topic, event, payload = extract_phoenix_or_json(data)
        category = classify_message(topic, event)

        if category == "social":
            self.social_count += 1
            return

        now_t = time.time()

        # Track phx_join topics
        if event == "phx_join" and topic:
            self.phx_join_topics[topic] += 1

        # Track subscribe / unsubscribe messages
        is_sub = (
            event in {"phx_join", "subscribe", "unsubscribe", "sub", "unsub"}
            or event.startswith("sub_")
            or event.startswith("unsub_")
        )
        if is_sub and len(self.subscribe_messages) < 10:
            redacted_sub = self.redactor.redact(data)
            parsed_ws = urllib.parse.urlparse(clean_url)
            ws_host = f"{parsed_ws.scheme}://{parsed_ws.netloc}{parsed_ws.path}"
            self.subscribe_messages.append({
                "host": ws_host,
                "topic": topic,
                "event": event,
                "message": redacted_sub,
            })

        # Inspect large data frames
        self._inspect_large_data_frame(topic, payload)

        # Redact the full structure
        redacted_data = self.redactor.redact(data)

        # Track in pair_stats
        pair_key = (topic, event)
        if pair_key not in self.pair_stats:
            self.pair_stats[pair_key] = {
                "category": category,
                "count": 1,
                "first_ts": now_t,
                "last_ts": now_t,
                "examples": [redacted_data],
            }
        else:
            entry = self.pair_stats[pair_key]
            entry["count"] += 1
            entry["last_ts"] = now_t
            if len(entry["examples"]) < 2:
                entry["examples"].append(redacted_data)

        # Track in classified_types for backward compatibility
        msg_type = extract_message_type(data)
        if msg_type not in self.classified_types:
            self.classified_types[msg_type] = {
                "count": 1,
                "category": category,
                "example": redacted_data,
                "examples": [redacted_data],
                "source": source_label,
            }
        else:
            self.classified_types[msg_type]["count"] += 1
            ex_list = self.classified_types[msg_type].setdefault("examples", [])
            if len(ex_list) < self.max_examples_per_type:
                ex_list.append(redacted_data)

    def _on_response(self, response: Any) -> None:
        """Analyze HTTP responses read-only; drops tracker hosts and enforces 2000 endpoint cap."""
        clean_url = self.strip_url(response.url)
        if self.is_tracker_url(clean_url):
            self.dropped_tracker_count += 1
            return

        collapsed = self.collapse_url_path(clean_url)
        if collapsed not in self.http_endpoints:
            if len(self.http_endpoints) >= self.max_endpoint_cap:
                self.dropped_endpoint_cap_count += 1
                return
            self.http_endpoints.add(collapsed)

        # Record method, status, and count
        method = "GET"
        try:
            if hasattr(response, "request") and hasattr(response.request, "method"):
                method = response.request.method
        except Exception:
            pass

        status = 200
        try:
            if hasattr(response, "status"):
                status = response.status
        except Exception:
            pass

        parsed_url = urllib.parse.urlparse(collapsed)
        host = parsed_url.netloc

        if collapsed not in self.http_endpoint_details:
            self.http_endpoint_details[collapsed] = {
                "host": host,
                "path": parsed_url.path or "/",
                "method": method,
                "status": status,
                "count": 1,
                "example": None,
            }
        else:
            self.http_endpoint_details[collapsed]["count"] += 1

        # Skip all bodies for auth/login/token/password URLs
        if SKIP_URL_REGEX.search(clean_url):
            self.redactor.redaction_count += 1
            return

        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                body_text = response.text()
                try:
                    parsed_body = json.loads(body_text)
                    self._extract_demo_real_fields(parsed_body)

                    # Check login indicators for /passport/ or /bo-assets/
                    url_lower = clean_url.lower()
                    if "/passport/" in url_lower or "/bo-assets/" in url_lower:
                        self.seen_passport_or_assets = True

                    # If endpoint suggests key trading primitives, capture matching sample
                    is_trading = bool(re.search(
                        r"assets?|instruments?|quotes?|payouts?|balance|deals?|orders?|candles?|history|account|currencies",
                        parsed_url.path,
                        re.IGNORECASE,
                    ))
                    if is_trading and self.http_endpoint_details[collapsed]["example"] is None:
                        redacted_body = self.redactor.redact(parsed_body)
                        self.http_endpoint_details[collapsed]["example"] = redacted_body

                    # Asset table extraction for /bo-assets or any array of assets
                    is_asset_ep = bool(re.search(r"bo-assets|/assets", parsed_url.path, re.IGNORECASE))
                    if is_asset_ep:
                        self._extract_asset_table(parsed_body, clean_url)

                    self._process_text_frame(body_text, "HTTP_RESP", clean_url)
                except Exception:
                    pass
            except Exception:
                pass

    def _on_request(self, request: Any) -> None:
        """Observe HTTP requests read-only for login detection."""
        try:
            req_url = self.strip_url(request.url).lower()
            if "/passport/" in req_url or "/bo-assets/" in req_url:
                self.seen_passport_or_assets = True
        except Exception:
            pass

    def _on_websocket(self, ws: Any) -> None:
        """Handle text and binary WebSocket frames, dropping tracker connections."""
        clean_url = self.strip_url(ws.url)
        if self.is_tracker_url(clean_url):
            self.dropped_tracker_count += 1
            return

        parsed = urllib.parse.urlparse(clean_url)
        clean_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.ws_endpoints.add(clean_endpoint)

        # Login observation: track open sockets and ws.binomo.com
        self.open_websockets.add(ws.url)
        if "ws.binomo.com" in ws.url:
            self.seen_ws_binomo_open = True

        def on_close() -> None:
            self.open_websockets.discard(ws.url)

        try:
            ws.on("close", on_close)
        except Exception:
            pass

        # Frame counting per socket
        if clean_endpoint not in self.socket_frame_counts:
            self.socket_frame_counts[clean_endpoint] = {"in": 0, "out": 0, "total": 0}

        # Detect non-Phoenix WS hosts (e.g. wss://as.binomo.com) — not array-framed Phoenix
        is_alt_ws = "as.binomo" in parsed.netloc or "as2.binomo" in parsed.netloc

        def on_frame_received(payload: Any) -> None:
            self.socket_frame_counts[clean_endpoint]["in"] += 1
            self.socket_frame_counts[clean_endpoint]["total"] += 1
            if isinstance(payload, bytes):
                self.binary_frame_count += 1
                if len(self.binary_samples) < 5:
                    self.binary_samples.append({
                        "direction": "in",
                        "len": len(payload),
                        "hex_prefix": payload[:16].hex(),
                    })
            elif isinstance(payload, str):
                if is_alt_ws or payload.lstrip().startswith("{"):
                    self._process_alt_ws_frame(payload, clean_url)
                self._process_text_frame(payload, "WS_IN", clean_url)

        def on_frame_sent(payload: Any) -> None:
            self.socket_frame_counts[clean_endpoint]["out"] += 1
            self.socket_frame_counts[clean_endpoint]["total"] += 1
            if isinstance(payload, bytes):
                self.binary_frame_count += 1
                if len(self.binary_samples) < 5:
                    self.binary_samples.append({
                        "direction": "out",
                        "len": len(payload),
                        "hex_prefix": payload[:16].hex(),
                    })
            elif isinstance(payload, str):
                self._process_outgoing_frame(payload, "WS_OUT", clean_url)

        ws.on("framereceived", on_frame_received)
        ws.on("framesent", on_frame_sent)

    def _process_alt_ws_frame(self, raw_text: str, clean_url: str) -> None:
        """Track non-Phoenix JSON WebSocket frames (e.g. wss://as.binomo.com) for --focus."""
        try:
            data = json.loads(raw_text)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        parsed_ws = urllib.parse.urlparse(clean_url)
        host_key = f"{parsed_ws.scheme}://{parsed_ws.netloc}{parsed_ws.path}"

        if host_key not in self.alt_ws_data:
            self.alt_ws_data[host_key] = {
                "subscribe_frames": [],
                "event_types": Counter(),
                "data_frames": [],     # first 3 data frames
                "msg_count": 0,
                "asset_rates": Counter(),  # asset-key -> message count (rate proxy)
                "price_field": "",
                "time_field": "",
            }
        entry = self.alt_ws_data[host_key]
        entry["msg_count"] += 1

        event_type = str(data.get("event_type") or data.get("type") or data.get("action") or "")
        if event_type:
            entry["event_types"][event_type] += 1

        e_lower = event_type.lower()
        action_lower = str(data.get("action") or "").lower()
        # Subscribe/unsubscribe messages
        if (
            e_lower in {"subscribe", "unsubscribe", "sub", "unsub"}
            or e_lower.startswith("sub")
            or action_lower in {"subscribe", "unsubscribe", "sub", "unsub"}
        ):
            if len(entry["subscribe_frames"]) < 5:
                entry["subscribe_frames"].append(self.redactor.redact(data))

        # Data frames (price ticks etc.)
        raw_data = data.get("data") or data.get("quotes") or data.get("result")
        if raw_data is not None and len(entry["data_frames"]) < 3:
            entry["data_frames"].append(self.redactor.redact(data))
            # Detect price / time fields
            if not entry["price_field"]:
                probe = raw_data if isinstance(raw_data, dict) else (raw_data[0] if isinstance(raw_data, list) and raw_data else {})
                if isinstance(probe, dict):
                    for k in probe:
                        k_l = k.lower()
                        if not entry["price_field"] and k_l in {"price", "rate", "close", "bid", "ask", "c"}:
                            entry["price_field"] = k
                        if not entry["time_field"] and k_l in {"time", "ts", "timestamp", "t", "created_at"}:
                            entry["time_field"] = k

        # Track per-asset message count
        asset_key = str(data.get("asset") or data.get("ric") or data.get("symbol") or "")
        if asset_key:
            entry["asset_rates"][asset_key] += 1

    def _process_outgoing_frame(self, raw_text: str, source_label: str, clean_url: str) -> None:
        """Capture outgoing client-to-server frames for topics bo, asset, range_stream, create events."""
        try:
            data = json.loads(raw_text)
        except Exception:
            return

        topic, event, _ = extract_phoenix_or_json(data)
        t_lower = topic.lower()
        e_lower = event.lower()

        # Login observation: account join
        if (event == "phx_join" or e_lower == "phx_join") and (t_lower == "account" or t_lower.startswith("account:") or t_lower.startswith("user:")):
            self.seen_account_join = True

        is_target = (
            t_lower.startswith("bo")
            or t_lower.startswith("asset")
            or t_lower.startswith("range_stream")
            or e_lower == "create"
            or e_lower.startswith("create_")
        )

        if is_target:
            redacted = self.redactor.redact(data)
            pair_key = (topic, event)
            if pair_key not in self.outgoing_frames:
                self.outgoing_frames[pair_key] = [redacted]
            elif len(self.outgoing_frames[pair_key]) < 3:
                self.outgoing_frames[pair_key].append(redacted)

        # Also process into general message tracking
        self._process_text_frame(raw_text, source_label, clean_url)

    def _take_dom_snapshot(self, page: Any, label: str) -> None:
        """Capture read-only accessibility snapshot of trading panel without clicking or evaluating.

        Uses page.locator('body').aria_snapshot() when available, else CDP Accessibility.getFullAXTree.
        Records errors into the snapshot entry instead of failing silently.
        """
        if not self.dom_snapshots_enabled or not page:
            return

        filtered_items: list[dict[str, Any]] = []
        error_msg: Optional[str] = None

        try:
            # 1. Try page.locator("body").aria_snapshot()
            raw_aria: Optional[str] = None
            if hasattr(page, "locator"):
                try:
                    body_loc = page.locator("body")
                    if hasattr(body_loc, "aria_snapshot") and callable(body_loc.aria_snapshot):
                        raw_aria = body_loc.aria_snapshot()
                except Exception as e_aria:
                    raw_aria = None

            if raw_aria and isinstance(raw_aria, str):
                for line in raw_aria.splitlines():
                    m = re.match(r"^\s*-\s+([a-zA-Z0-9_-]+)(?:\s+\"([^\"]*)\")?(?::\s*(.*))?", line)
                    if m:
                        role = (m.group(1) or "").strip()
                        name = (m.group(2) or "").strip()
                        val = (m.group(3) or "").strip()

                        if PROFILE_MENU_REGEX.search(role) or PROFILE_MENU_REGEX.search(name):
                            continue

                        combined = f"{role} {name} {val}"
                        if DOM_KEYWORDS_REGEX.search(combined):
                            masked_name = mask_dom_text(name) if name else ""
                            masked_val = mask_dom_text(val) if val else ""
                            item: dict[str, Any] = {"role": role, "name": masked_name}
                            if masked_val:
                                item["value"] = masked_val
                            filtered_items.append(item)
            else:
                # 2. Fallback to CDP Accessibility.getFullAXTree
                cdp_nodes: list[dict[str, Any]] = []
                if hasattr(page, "context") and hasattr(page.context, "new_cdp_session"):
                    cdp_client = page.context.new_cdp_session(page)
                    cdp_res = cdp_client.send("Accessibility.getFullAXTree")
                    if isinstance(cdp_res, dict):
                        cdp_nodes = cdp_res.get("nodes", [])

                if cdp_nodes:
                    for node in cdp_nodes:
                        role_obj = node.get("role", {})
                        role = role_obj.get("value", "") if isinstance(role_obj, dict) else str(role_obj or "")
                        name_obj = node.get("name", {})
                        name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj or "")
                        val_obj = node.get("value", {})
                        val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj or "")

                        if PROFILE_MENU_REGEX.search(role) or PROFILE_MENU_REGEX.search(name):
                            continue

                        combined = f"{role} {name} {val}"
                        if DOM_KEYWORDS_REGEX.search(combined):
                            masked_name = mask_dom_text(name.strip()) if name else ""
                            item_dict: dict[str, Any] = {"role": role, "name": masked_name}
                            if val:
                                item_dict["value"] = mask_dom_text(str(val).strip())
                            for k in ("data-testid", "data-test", "test_id"):
                                if k in node:
                                    item_dict[k] = node[k]
                            filtered_items.append(item_dict)
                elif not raw_aria:
                    error_msg = "Neither locator('body').aria_snapshot() nor CDP Accessibility.getFullAXTree returned data"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"

        snap_data: dict[str, Any] = {
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "element_count": len(filtered_items),
            "elements": filtered_items[:40],
        }
        if error_msg:
            snap_data["error"] = error_msg
        self.dom_snapshots.append(snap_data)

    def _attach_page_listeners(self, page: Any) -> None:
        """Attach listeners to a page, popup, or newly opened tab and its frames."""
        if page in self._attached_pages:
            return
        self._attached_pages.add(page)
        page.on("response", self._on_response)
        page.on("request", self._on_request)
        page.on("websocket", self._on_websocket)

        def on_frame(frame: Any) -> None:
            self._on_frame_attached(frame)

        try:
            page.on("frameattached", on_frame)
            for fr in getattr(page, "frames", []):
                self._on_frame_attached(fr)
        except Exception:
            pass

    def _on_frame_attached(self, frame: Any) -> None:
        """Track iframe attachment read-only."""
        try:
            f_url = self.strip_url(frame.url)
            if f_url:
                self.attached_iframes.add(f_url)
        except Exception:
            pass

    def _inspect_page_elements(self, page: Any) -> dict[str, Any]:
        """Read-only inspection of potential DOM elements without any interaction."""
        info: dict[str, Any] = {"title": page.title(), "url": self.strip_url(page.url), "elements_found": []}
        test_selectors = [
            "[class*='balance']",
            "[data-test*='balance']",
            "[class*='payout']",
            "[class*='asset']",
            "[class*='instrument']",
            "[class*='chart']",
        ]
        for sel in test_selectors:
            try:
                matches = page.query_selector_all(sel)
                if matches:
                    info["elements_found"].append({
                        "selector": sel,
                        "count": len(matches),
                        "text_sample": matches[0].inner_text()[:100] if matches else "",
                    })
            except Exception:
                pass
        return info

    def write_report(self, dom_data: Optional[dict[str, Any]] = None) -> None:
        """Compile and write discovery_report.md."""
        now = datetime.now(timezone.utc).isoformat()
        dom = dom_data or {"title": "N/A", "url": "N/A", "elements_found": []}

        # Calculate msg/s for pairs
        dur = max(1, self.duration_seconds)

        def format_pair_section(cat_name: str, cat_key: str) -> str:
            pairs = [
                (p, stats) for p, stats in self.pair_stats.items()
                if stats["category"] == cat_key
            ]
            pairs.sort(key=lambda x: x[1]["count"], reverse=True)
            lines = [f"### Category: {cat_name} ({len(pairs)} pairs captured)"]
            if not pairs:
                lines.append("_No messages captured in this category during monitoring window._\n")
                return "\n".join(lines)
            for (top, ev), details in pairs:
                cnt = details["count"]
                mps = cnt / dur
                lines.append(f"- **Pair**: `({top}, {ev})` (Count: {cnt}, Rate: {mps:.2f} msg/s)")
                for ex in details["examples"][:2]:
                    lines.append("```json")
                    dumped = json.dumps(ex, indent=2)
                    max_chars = 6000 if (FOCUS_TOPICS_REGEX.search(top) or FOCUS_TOPICS_REGEX.search(ev)) else 3000
                    if len(dumped) > max_chars:
                        dumped = dumped[:max_chars] + f"\n... [truncated at {max_chars} chars]"
                    lines.append(dumped)
                    lines.append("```")
            lines.append("")
            return "\n".join(lines)

        logged_in_status = "yes" if self.check_login_status()["logged_in"] else "no"

        # Format WebSocket Endpoints with frame counts if available
        ws_lines: list[str] = []
        if self.socket_frame_counts:
            for sock_url, c in sorted(self.socket_frame_counts.items()):
                ws_lines.append(f"- `{sock_url}`: {c['total']} frames ({c['in']} in, {c['out']} out)")
        elif self.ws_endpoints:
            for ep in sorted(self.ws_endpoints):
                ws_lines.append(f"- `{ep}`")
        ws_section_text = "\n".join(ws_lines) if ws_lines else "- _No WebSocket endpoints recorded._"

        content = f"""# Platform Discovery Report

> [!WARNING]
> ## REVIEW BEFORE SHARING
> Please review this report before sharing externally to verify that no sensitive
> credentials, account IDs, or unintended tokens are present.

- **Generated At**: {now}
- **Target URL**: {self.strip_url(self.target_url)}
- **Monitoring Duration**: {self.duration_seconds}s
- **Logged-in indicators**: {logged_in_status}
- **Total Redactions Applied**: {self.redactor.redaction_count}
- **Binary Frames Count**: {self.binary_frame_count}
- **Dropped Tracker Requests/Frames**: {self.dropped_tracker_count}
- **Dropped Endpoints (Over 2000 Cap)**: {self.dropped_endpoint_cap_count}
- **i18n Messages Dropped**: {self.i18n_count}
- **Social Messages Ignored**: {self.social_count}

## 1. Safety Compliance
- **Read-Only Mode**: Confirmed. Zero clicks, fills, types, presses, checks, or evaluates executed.
- **Credential Safety**: User credentials entered manually in persistent Chromium profile; no request bodies, headers, or cookies stored.
- **URL Sanitation**: All query strings stripped; auth/login endpoints skipped.

## 2. WebSocket Endpoints
{ws_section_text}

## 3. Phoenix Join Topics
{chr(10).join(f"- Topic: `{top}` (Count: {cnt})" for top, cnt in self.phx_join_topics.most_common()) if self.phx_join_topics else "- _No phx_join topics recorded._"}

## 4. Subscribe and Unsubscribe Messages
"""
        if self.subscribe_messages:
            for sub in self.subscribe_messages[:10]:
                content += f"- Host: `{sub['host']}` | Topic: `{sub['topic']}` | Event: `{sub['event']}`\n"
                content += "```json\n" + json.dumps(sub["message"], indent=2) + "\n```\n"
        else:
            content += "- _No subscribe/unsubscribe messages captured._\n"

        content += f"""
## 5. Binary Frames Summary
- Total binary frames: {self.binary_frame_count}
```json
{json.dumps(self.binary_samples, indent=2)}
```

## 6. Message Classification & Redacted Samples

{format_pair_section("Prices / Quotes", "prices/quotes")}
{format_pair_section("Assets / Payouts", "assets/payouts")}
{format_pair_section("Balance", "balance")}
{format_pair_section("Deals", "deals")}
{format_pair_section("System", "system")}
{format_pair_section("Other", "other")}

## 7. Large Data Frames Breakdown
"""
        if self.large_data_frames:
            for ldf in self.large_data_frames:
                content += f"- **Topic**: `{ldf['topic']}`\n"
                content += f"  - Structure: {ldf['structure']}\n"
                content += f"  - Price Field: `{ldf['price_field']}`\n"
                content += f"  - Time Field: `{ldf['time_field']}`\n"
                content += "```json\n" + json.dumps(ldf["sample"], indent=2) + "\n```\n"
        else:
            content += "- _No large data frames identified._\n"

        content += """
## 8. Demo vs Real Account Fields
"""
        if self.demo_real_fields:
            for k, v in sorted(self.demo_real_fields.items()):
                content += f"- `{k}`: `{json.dumps(v)}`\n"
        else:
            content += "- _No explicit demo/real fields captured._\n"

        content += f"""
## 9. Platform HTTP Endpoints
"""
        if self.http_endpoint_details:
            for ep_url, ep_info in sorted(self.http_endpoint_details.items()):
                content += f"- [{ep_info['method']}] `{ep_url}` (status: {ep_info['status']}, count: {ep_info['count']})\n"
                if ep_info.get("example") is not None:
                    content += "  - Sample JSON payload:\n```json\n"
                    dumped = json.dumps(ep_info["example"], indent=2)
                    max_chars = 6000 if FOCUS_ENDPOINTS_REGEX.search(ep_url) else 3000
                    if len(dumped) > max_chars:
                        dumped = dumped[:max_chars] + f"\n... [truncated at {max_chars} chars]"
                    content += dumped + "\n```\n"
        else:
            content += "- _No HTTP endpoints recorded._\n"

        content += f"""
## 10. DOM Elements Identified (Read-Only)
- Page Title: {dom.get('title', 'N/A')}
- Current URL: {dom.get('url', 'N/A')}
- Identified Selectors:
```json
{json.dumps(dom.get('elements_found', []), indent=2)}
```

## 11. Outgoing (Client-to-Server) Frames
"""
        if self.outgoing_frames:
            for (top, ev), ex_list in sorted(self.outgoing_frames.items()):
                content += f"- **Pair**: `({top}, {ev})` ({len(ex_list)} frames captured)\n"
                for ex in ex_list[:3]:
                    content += "```json\n" + json.dumps(ex, indent=2) + "\n```\n"
        else:
            content += "- _No matching outgoing frames captured._\n"

        content += f"""
## 12. DOM Accessibility Snapshots (Read-Only)
"""
        if self.dom_snapshots:
            for snap in self.dom_snapshots:
                content += f"### Snapshot: `{snap['label']}` (Elements: {snap['element_count']}, Time: `{snap['timestamp']}`)\n"
                if snap.get("error"):
                    content += f"- Snapshot Error: `{snap['error']}`\n"
                if snap.get("elements"):
                    content += "```json\n" + json.dumps(snap["elements"], indent=2) + "\n```\n"
        else:
            content += "- _No DOM snapshots recorded._\n"

        content += f"""
## 13. Non-Phoenix WebSocket Data
"""
        if self.alt_ws_data:
            for host, info in sorted(self.alt_ws_data.items()):
                content += f"### Host: `{host}` (Total messages: {info['msg_count']})\n"
                content += f"- Distinct Event Types: {dict(info['event_types'])}\n"
                content += f"- Price Field: `{info.get('price_field', 'N/A')}` | Time Field: `{info.get('time_field', 'N/A')}`\n"
                content += "- Message Rate Per Asset:\n"
                for a_ric, cnt in info["asset_rates"].most_common(10):
                    mps = cnt / dur
                    content += f"  - `{a_ric}`: {cnt} msgs ({mps:.2f} msg/s)\n"
                content += "- Subscribe / Unsubscribe Messages:\n```json\n"
                content += json.dumps(info["subscribe_frames"], indent=2) + "\n```\n"
                content += "- First 3 Data Frames:\n```json\n"
                content += json.dumps(info["data_frames"], indent=2) + "\n```\n"
        else:
            content += "- _No non-Phoenix WebSocket data recorded._\n"

        content += f"""
## 14. Asset List Table (/bo-assets/v7/assets)
"""
        if self.asset_table:
            content += "| Name | RIC | Type | OTC | Base Rates (Turbo/Std/Classic) | Enabled Accounts | Trading Types | Schedule |\n"
            content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            for row in self.asset_table:
                otc_str = "Yes" if row.get("otc") else "No"
                content += f"| {row['name']} | {row['ric']} | {row['type']} | {otc_str} | {row['rates_str']} | {row['enabled_account_types']} | {row['enabled_trading_types']} | {row['today_schedule']} |\n"
            content += "\n"
            if self.asset_first_two:
                content += "### First 2 Asset Objects (Full Structure)\n```json\n"
                content += json.dumps(self.asset_first_two, indent=2) + "\n```\n"
        else:
            content += "- _No asset table recorded._\n"

        target_path = REPORT_PATH if REPORT_PATH != Path("discovery_report.md") else self.report_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        with open(self.archive_path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def run(self) -> None:
        """Launch browser, attach context-level listeners, monitor traffic, and write report."""
        from playwright.sync_api import sync_playwright

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print("PLATFORM DISCOVERY TOOL (STRICTLY READ-ONLY)")
        print("Safety Rule: NEVER clicks, fills, types, presses, checks, or evaluates.")
        print(f"Profile directory: {PROFILE_DIR.resolve()}")
        print(f"Target URL: {self.strip_url(self.target_url)}")
        print("=" * 80)

        print("\nCHECKLIST FOR PLATFORM DISCOVERY:")
        print("  1. Log in manually in the browser window.")
        print("  2. Switch to the DEMO account.")
        print("  3. Open the trading screen.")
        print("  4. Open EUR/USD and two other assets.")
        print("  5. Open the asset list / instrument selector.")
        print("  6. Change the chart timeframe (e.g., 1m, 5m).")
        print("  7. Keep a chart open so real-time price ticks stream in.")
        print("  8. [OPTIONAL] Place ONE minimum-size trade by hand on DEMO account")
        print("     so order/trade messages are captured.")
        print("=" * 80 + "\n")

        dom_data: dict[str, Any] = {}
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR.resolve()),
                headless=False,
                viewport={"width": 1280, "height": 800},
            )

            context.on("page", lambda new_page: self._attach_page_listeners(new_page))
            for pg in context.pages:
                self._attach_page_listeners(pg)

            page = context.pages[0] if context.pages else context.new_page()
            self._attach_page_listeners(page)

            print(f"Navigating to {self.target_url}...")
            try:
                page.goto(self.target_url, timeout=30000)
            except Exception as e:
                print(f"Navigation notice: {e}")

            # Initial DOM snapshot at start
            self._take_dom_snapshot(page, "start")

            print(f"Monitoring network and WebSocket traffic for {self.duration_seconds}s (Press Ctrl+C to finish early)...")

            start = time.time()
            last_reminder = start
            last_login_check = start
            last_snapshot_time = start

            try:
                while time.time() - start < self.duration_seconds:
                    time.sleep(1)
                    now_t = time.time()

                    # 1. Login detection every 30s
                    if now_t - last_login_check >= 30:
                        st = self.check_login_status()
                        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Login Status: ws.binomo.com open={st['ws_binomo_open']}, /passport/ or /bo-assets/ seen={st['seen_passport_or_assets']}, 'account' join seen={st['seen_account_join']}")
                        if not st["logged_in"]:
                            print("  >>> NOT LOGGED IN YET - log in manually in the browser window <<<")
                        last_login_check = now_t

                    # 2. DOM snapshot every 60s while path contains /trading
                    page_url = getattr(page, "url", "")
                    if (now_t - last_snapshot_time >= 60) and ("/trading" in str(page_url).lower()):
                        elapsed_min = int((now_t - start) // 60)
                        self._take_dom_snapshot(page, f"trading_{elapsed_min}m")
                        last_snapshot_time = now_t

                    # 3. Reminder every 120s
                    if now_t - last_reminder >= 120:
                        elapsed_min = int((now_t - start) // 60)
                        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Reminder ({elapsed_min}m elapsed):")
                        print("  - Ensure you are on DEMO account")
                        print("  - View asset list & switch assets")
                        print("  - Change chart timeframe")
                        print("  - Keep a chart open for ticks")
                        last_reminder = now_t
            except KeyboardInterrupt:
                print("\nMonitoring stopped early by user (Ctrl+C). Saving captured data...")

            # Final DOM snapshot at end
            self._take_dom_snapshot(page, "end")

            try:
                dom_data = self._inspect_page_elements(page)
            except Exception as e:
                print(f"DOM inspection note: {e}")

            try:
                context.close()
            except Exception:
                pass

        print(f"Generating report: {self.report_path} (archived to {self.archive_path})...")
        self.write_report(dom_data)
        print(f"Report generated successfully at {self.report_path.resolve()}")

        # Print WebSocket frame summary
        print("\n[WebSocket Frame Summary]")
        if self.socket_frame_counts:
            for sock_url, c in sorted(self.socket_frame_counts.items()):
                print(f"  - {sock_url}: {c['total']} frames ({c['in']} in, {c['out']} out)")
        else:
            print("  - No WebSocket frames captured.")


def main() -> None:
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)
    parser = argparse.ArgumentParser(description="Strictly read-only platform discovery tool")
    parser.add_argument("--url", required=True, help="Target platform URL (required)")
    parser.add_argument("--duration", type=int, default=900, help="Monitoring duration in seconds (default: 900)")
    parser.add_argument("--report", default=None, help="Explicit report path (default: discovery_report_YYYYMMDD_HHMM.md)")
    parser.add_argument(
        "--dom-snapshots",
        action="store_true",
        help="Capture read-only DOM accessibility snapshots",
    )
    args = parser.parse_args()

    discovery = PlatformDiscovery(
        target_url=args.url,
        duration_seconds=args.duration,
        dom_snapshots=args.dom_snapshots,
        report_path=args.report,
    )
    discovery.run()


if __name__ == "__main__":
    main()
