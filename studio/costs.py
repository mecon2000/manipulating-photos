"""Daily cost ledger — same shape as auto_gen_tick's, but on ext4 state.

{"2026-08-07": {"total": 0.12, "by_tool": {"relighting": 0.12}}, ...}
Feeds the UI's daily-cost chip. Estimates come from the registry's
cost_estimate_usd (preview runs cost the same API-wise — smaller pixels,
same calls).
"""
import json
import threading
import time

from .paths import COSTS_FILE, ensure_dirs

_LOCK = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(COSTS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def accrue(tool: str, usd: float) -> None:
    if usd <= 0:
        return
    ensure_dirs()
    day = time.strftime("%Y-%m-%d")
    with _LOCK:
        ledger = _load()
        entry = ledger.setdefault(day, {"total": 0.0, "by_tool": {}})
        entry["total"] = round(entry["total"] + usd, 4)
        entry["by_tool"][tool] = round(entry["by_tool"].get(tool, 0.0) + usd, 4)
        COSTS_FILE.write_text(json.dumps(ledger, indent=2))


def today() -> dict:
    return _load().get(time.strftime("%Y-%m-%d"), {"total": 0.0, "by_tool": {}})
