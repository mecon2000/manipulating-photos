"""Per-session activity tracking: powers tab-strip spinners/ready badges and
the ntfy ping when a slow background step finishes.

In-memory (single app process). `ready` sticks until the client marks the
session seen; sessions evaluated synchronously get the same tracking so other
open tabs see the spinner too.
"""
import sys
import threading
import time

from .paths import REPO

_LOCK = threading.Lock()
_ACT: dict[str, dict] = {}   # session_id -> {running, label, started, ready, ok}

NOTIFY_THRESHOLD_SEC = 20


def start(session_id: str, label: str) -> None:
    with _LOCK:
        _ACT[session_id] = {"running": True, "label": label,
                            "started": time.time(), "ready": False, "ok": None}


def finish(session_id: str, ok: bool, notify_label: str = "") -> None:
    took = 0.0
    with _LOCK:
        a = _ACT.get(session_id)
        if a:
            took = time.time() - a["started"]
            a.update(running=False, ready=True, ok=ok)
    if took >= NOTIFY_THRESHOLD_SEC:
        _push(f"Studio: {notify_label or session_id} "
              f"{'ready' if ok else 'FAILED'} ({took:.0f}s)")


def mark_seen(session_id: str) -> None:
    with _LOCK:
        a = _ACT.get(session_id)
        if a and not a["running"]:
            a["ready"] = False


def status() -> dict:
    with _LOCK:
        return {sid: dict(a) for sid, a in _ACT.items()}


def _push(text: str) -> None:
    try:
        wf = str(REPO / "scripts" / "workflows")
        if wf not in sys.path:
            sys.path.insert(0, wf)
        import notify
        notify.push_text(text)
    except Exception:
        pass  # notifications are best-effort
