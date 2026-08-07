"""Content-addressed cache: objects (images/masks) + step records.

An object ref is the sha256 hex of the file's bytes; the file lives at
objects/<h[:2]>/<h><ext>. A step record maps a step key — sha256 over the
canonical JSON of (tool, params, flags, seed, preview, input_ref, mask_ref) —
to the output object ref it produced. Re-running an identical step is a pure
lookup; changing anything upstream changes input_ref and misses naturally.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

from .paths import OBJECTS_DIR, STEPS_DIR, ensure_dirs


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put_file(path, ref: str | None = None) -> str:
    """Store a file in the object store, return its ref."""
    ensure_dirs()
    path = Path(path)
    ref = ref or file_sha256(path)
    dest = OBJECTS_DIR / ref[:2] / (ref + path.suffix.lower())
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    return ref


def object_path(ref: str) -> Path | None:
    if not ref or len(ref) < 8 or "/" in ref or "." in ref:
        return None
    matches = list((OBJECTS_DIR / ref[:2]).glob(ref + ".*")) if (OBJECTS_DIR / ref[:2]).exists() else []
    return matches[0] if matches else None


def step_key(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_step(key: str) -> dict | None:
    p = STEPS_DIR / (key + ".json")
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except ValueError:
        return None
    # A record is only good if its output object still exists.
    if not object_path(rec.get("output", "")):
        return None
    return rec


def put_step(key: str, output_ref: str, meta: dict | None = None) -> dict:
    ensure_dirs()
    rec = {"output": output_ref, "created": time.time(), **(meta or {})}
    (STEPS_DIR / (key + ".json")).write_text(json.dumps(rec, indent=2))
    return rec
