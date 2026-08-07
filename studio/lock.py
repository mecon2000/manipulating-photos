"""Lock (§3.4): finalize the current draft. Lock = fav.

Two paths:
- upscale (default): Real-ESRGAN 2×/4× on the chosen 1024px draft via the
  existing upscale_replicate.py — look preserved exactly.
- rerender: same graph, same seeds, full-res source (preview=False clones) —
  generative steps may drift; caller shows the warning.

Either way the result lands in shared/favorites/ (THE folder — IG feeds from
it) with a favorites.json entry carrying full reconstruction data, plus the
pinned copy in shared/finals/.
"""
import json
import shutil
import subprocess
import time

from . import cache, runner
from .paths import REPO, RUNS_DIR, SHARED, VENV_PYTHON, ensure_dirs

FAVORITES_DIR = SHARED / "favorites"
FINALS_DIR = SHARED / "finals"
FAVORITES_JSON = FAVORITES_DIR / "favorites.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip()
    except OSError:
        return ""


def _upscale(src, scale: int):
    out_dir = RUNS_DIR / f"lock-upscale-{int(time.time())}"
    out_dir.mkdir(parents=True)
    r = subprocess.run(
        [str(VENV_PYTHON), str(REPO / "scripts/workflows/upscale_replicate.py"),
         "--source", str(src), "--scale", str(scale), "--out-dir", str(out_dir)],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    outs = sorted(out_dir.glob("*.*"), key=lambda p: p.stat().st_mtime)
    if r.returncode != 0 or not outs:
        raise RuntimeError(f"upscale failed: {r.stdout[-300:]} {r.stderr[-300:]}")
    return outs[-1]


def _fav_entry(session, final_name: str, mode: str, scale: int | None) -> dict:
    return {
        "file": final_name,
        "tool": "studio",
        "session_id": session.data["id"],
        "source": session.data["source_path"],
        "chain": [{"tool": n["tool"], "params": n["params"], "flags": n["flags"],
                   "seed": n["seed"]} for n in session.chain()],
        "lock_mode": mode,
        "upscale": scale,
        "git_commit": _git_commit(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": f"reopen: /studio/s/{session.data['id']} — chain re-runs via "
                   "studio tool server (content-addressed, same seeds)",
    }


def _export(session, produced, mode: str, scale: int | None) -> dict:
    ensure_dirs()
    FAVORITES_DIR.mkdir(parents=True, exist_ok=True)
    FINALS_DIR.mkdir(parents=True, exist_ok=True)
    name = (f"studio_{session.data['id']}_{time.strftime('%H%M%S')}"
            f"{produced.suffix.lower()}")
    # copyfile (data only) — copy2's utime and copy's chmod both fail on 9p
    shutil.copyfile(produced, FAVORITES_DIR / name)
    shutil.copyfile(produced, FINALS_DIR / name)
    try:
        favs = json.loads(FAVORITES_JSON.read_text())
    except (OSError, ValueError):
        favs = {"favorites": []}
    favs["favorites"].append(_fav_entry(session, name, mode, scale))
    FAVORITES_JSON.write_text(json.dumps(favs, indent=1))
    return {"file": str(FAVORITES_DIR / name), "final": str(FINALS_DIR / name)}


def lock_upscale(session, scale: int = 4) -> dict:
    """Default path: upscale the exact chosen draft."""
    results = runner.evaluate(session)
    if not results:
        raise RuntimeError("nothing to lock — no steps applied")
    draft = cache.object_path(results[-1]["output"])
    produced = _upscale(draft, scale) if scale in (2, 4) else draft
    out = _export(session, produced, "upscale", scale)
    return {**out, "mode": "upscale", "scale": scale}


def lock_rerender(session) -> dict:
    """Re-render the same graph at full res (new preview=False branch).
    Generative steps may come out different — caller warns."""
    active = list(session.chain())
    if not active:
        raise RuntimeError("nothing to lock — no steps applied")
    parent = None
    for n in active:
        node = session.add_step(n["tool"], n["params"], seed=n["seed"],
                                preview=False, flags=n["flags"],
                                parent=parent["id"] if parent else None)
        parent = node
    results = runner.evaluate(session)
    produced = cache.object_path(results[-1]["output"])
    out = _export(session, produced, "rerender", None)
    return {**out, "mode": "rerender"}
