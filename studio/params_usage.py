"""Params-panel usage learning (§3.3).

Every form render counts an "appearance" per visible param; every interaction
a "touch". A param whose last touch is ≥ HIDE_AFTER appearances ago (or that
was never touched across ≥ HIDE_AFTER appearances) auto-hides into the drawer.
Always manually overridable client-side — this only supplies the hint.
"""
import json
import threading

from .paths import STATE, ensure_dirs

USAGE_FILE = STATE / "params_usage.json"
HIDE_AFTER = 10

_LOCK = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(USAGE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def record(tool: str, param: str, kind: str) -> None:
    ensure_dirs()
    with _LOCK:
        data = _load()
        rec = data.setdefault(tool, {}).setdefault(
            param, {"appearances": 0, "touches": 0, "last_touch_appearance": 0})
        if kind == "appear":
            rec["appearances"] += 1
        elif kind == "touch":
            rec["touches"] += 1
            rec["last_touch_appearance"] = rec["appearances"]
        USAGE_FILE.write_text(json.dumps(data, indent=2))


def hidden_params(tool: str) -> list[str]:
    data = _load().get(tool, {})
    out = []
    for param, rec in data.items():
        if rec["appearances"] - rec["last_touch_appearance"] >= HIDE_AFTER:
            out.append(param)
    return out
