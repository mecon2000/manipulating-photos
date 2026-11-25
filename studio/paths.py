"""Studio path constants + boundary checks.

Runtime state lives HERE on ext4 (studio/state/, gitignored) — never on the
9p/drvfs `shared/` mount (SQLite WAL corrupts there, small-file churn is slow).
Only human-facing exports (locked finals, favorites, recipes) cross to shared/.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "studio" / "state"
CACHE_DIR = STATE / "cache"
OBJECTS_DIR = CACHE_DIR / "objects"
STEPS_DIR = CACHE_DIR / "steps"
SESSIONS_DIR = STATE / "sessions"
RUNS_DIR = STATE / "runs"
TRASH_DIR = STATE / "trash"
COSTS_FILE = STATE / "costs.json"

SHARED = Path(os.path.expanduser("~/.openclaw/workspace/shared"))
PHOTOS = Path(os.path.expanduser("~/.openclaw/workspace/_photos"))  # READ-ONLY
ARCHIVE = Path(os.path.expanduser("~/.openclaw/workspace/_archive"))  # I:\Photos, 9p mount, READ-ONLY
REGISTRY_FILE = REPO / "manipulating-photos-with-ui" / "tool_registry.json"
VENV_PYTHON = Path(os.path.expanduser("~/openclaw-venv/bin/python3"))

# Roots a client-supplied source path may resolve under (all read access only).
ALLOWED_SOURCE_ROOTS = (SHARED, PHOTOS, STATE, ARCHIVE)

# Windows paths pasted from Explorer ("Copy as path") → where WSL sees them.
WIN_PREFIXES = (
    ("i:\\photos", ARCHIVE),                      # the Lightroom archive (catalog's I:/Photos)
    ("d:\\openclaw\\shared", SHARED),             # the shared folder as Windows sees it
)


def from_windows(path_str: str) -> str:
    """'I:\\Photos\\2026\\x.jpg' (quotes optional, / or \\) → the Linux path, if it's
    under a known mapping; anything else is returned unchanged."""
    s = path_str.strip().strip('"').strip("'")
    if len(s) > 2 and s[1] == ":" and s[0].isalpha():
        low = s.replace("/", "\\").lower()
        for prefix, root in WIN_PREFIXES:
            if low == prefix or low.startswith(prefix + "\\"):
                rest = s.replace("/", "\\")[len(prefix):].strip("\\")
                return str(root.joinpath(*[x for x in rest.split("\\") if x]))
    return s


def ensure_dirs() -> None:
    for d in (OBJECTS_DIR, STEPS_DIR, SESSIONS_DIR, RUNS_DIR, TRASH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def safe_source(path_str: str) -> Path:
    """Resolve a client-supplied path and require it under an allowed root.

    Mirrors the hub's safepath.py contract: resolve symlinks first, then do a
    strict prefix check, so `..` tricks and symlink escapes both fail.
    """
    p = Path(os.path.expanduser(from_windows(path_str))).resolve()
    for root in ALLOWED_SOURCE_ROOTS:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"path outside allowed roots: {p}")


def trash(path: Path) -> Path:
    """Delete = move into studio/state/trash/, never rm."""
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRASH_DIR / path.name
    n = 1
    while dest.exists():
        dest = TRASH_DIR / f"{path.stem}.{n}{path.suffix}"
        n += 1
    path.rename(dest)
    return dest
