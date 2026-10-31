"""Compose sessions — Studio's front end for layer_compose.py (plans/deconstruct.md Phase 3).

The engine is a CLI (scripts/workflows/layer_compose.py: init/render/set/options/pick);
this module treats it as the API — subprocess per call, serialized per piece so two
taps can't race on piece.json. Pieces live on shared/ (Windows-visible on purpose,
like recipes/finals); Studio-only runtime state (compose chat transcripts, thumbs)
stays on ext4 under studio/state/compose/.

Pinned canvas comments: <piece>/comments.jsonl, append-only. A status change is a
new line with the same id; readers collapse by id (last wins).
"""
import json
import os
import random
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import activity
from .paths import REPO, SHARED, STATE, VENV_PYTHON

ROOT = SHARED / "deconstruct"
PIECES = ROOT / "pieces"
RECIPES = ROOT / "recipes"
LC = REPO / "scripts" / "workflows" / "layer_compose.py"
COMPOSE_STATE = STATE / "compose"
THUMBS = COMPOSE_STATE / "thumbs"

_NAME_RE = re.compile(r"^[\w.\- ()&,]+$")
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
JOBS: dict[str, dict] = {}   # init jobs (in-memory; the piece dir is the durable result)


def _lock(piece: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(piece, threading.Lock())


def _env() -> dict:
    """Child env: notifications off (Studio pings itself), FAL_KEY from ~/sol/.env,
    no API-billing Anthropic keys (riff.py rides the CLI subscription)."""
    env = {**os.environ, "NOTIFY_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    if not env.get("FAL_KEY"):
        try:
            for line in Path("~/sol/.env").expanduser().read_text().splitlines():
                k, _, v = line.strip().partition("=")
                if k in ("FAL_API_KEY", "FAL_KEY") and v:
                    env["FAL_KEY"] = v.strip().strip('"').strip("'")
                    break
        except OSError:
            pass
    return env


def piece_dir(piece: str) -> Path:
    """Validate a client-supplied piece name → its dir (must exist)."""
    if not piece or "/" in piece or ".." in piece or not _NAME_RE.match(piece):
        raise ValueError(f"bad piece name {piece!r}")
    d = PIECES / piece
    if not (d / "piece.json").is_file():
        raise FileNotFoundError(f"no piece {piece}")
    return d


def run_cli(piece: str | None, args: list[str], timeout: int = 900) -> str:
    """Run layer_compose.py; serialized per piece. Returns stdout, raises on rc!=0."""
    argv = [str(VENV_PYTHON), str(LC), *args]
    lock = _lock(piece) if piece else threading.Lock()
    with lock:
        proc = subprocess.run(argv, cwd=REPO, env=_env(), timeout=timeout,
                              capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout[-600:] + "\n" + proc.stderr[-1200:]).strip()
        raise RuntimeError(f"layer_compose {args[0]} failed: {tail}")
    return proc.stdout


# -- reading state -------------------------------------------------------------

def load_piece(piece: str) -> dict:
    return json.loads((piece_dir(piece) / "piece.json").read_text())


def stack(P: dict) -> list[dict]:
    """Layers in render order: under-layers, the subject, over-layers."""
    def row(L):
        return {"id": L["id"], "kind": L.get("kind"), "z": L.get("z", "under"),
                "enabled": L.get("enabled", True), "riff": bool(L.get("riff")),
                "seed": L.get("seed"), "region": L.get("region"),
                "note": L.get("note", ""), "params": L.get("params", {})}
    under = [row(L) for L in P["layers"] if L.get("z", "under") == "under"]
    over = [row(L) for L in P["layers"] if L.get("z") == "over"]
    return under + [{"id": "subject", "kind": "subject", "z": "subject",
                     "enabled": True, "note": (P.get("base") or {}).get("note", "")}] + over


def questions(piece: str) -> list[dict]:
    qd = piece_dir(piece) / "questions"
    out = []
    for f in sorted(qd.glob("q*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            out.append(json.loads(f.read_text()))
        except ValueError:
            continue
    return out


def comments(piece: str) -> list[dict]:
    p = piece_dir(piece) / "comments.jsonl"
    by_id: dict[str, dict] = {}
    try:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            by_id[c["id"]] = {**by_id.get(c["id"], {}), **c}
    except (OSError, ValueError):
        pass
    return list(by_id.values())


def _append_comment(piece: str, rec: dict) -> None:
    with open(piece_dir(piece) / "comments.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def add_comment(piece: str, x: float, y: float, text: str) -> dict:
    c = {"id": uuid.uuid4().hex[:8], "t": time.strftime("%F %T"),
         "x": round(min(max(float(x), 0.0), 1.0), 4),
         "y": round(min(max(float(y), 0.0), 1.0), 4),
         "text": text.strip()[:500], "status": "open"}
    _append_comment(piece, c)
    return c


def set_comment_status(piece: str, cid: str, status: str, label: str = "") -> dict:
    cur = next((c for c in comments(piece) if c["id"] == cid), None)
    if cur is None:
        raise KeyError(cid)
    rec = {"id": cid, "t": time.strftime("%F %T"), "status": status}
    if label:
        rec["label"] = label
    _append_comment(piece, rec)
    return {**cur, **rec}


def summary(piece: str) -> dict:
    d = piece_dir(piece)
    P = load_piece(piece)
    final = d / "final.jpg"
    return {
        "piece": piece, "photo": P.get("photo"), "recipe": P.get("recipe"),
        "wildness": P.get("wildness"), "seed": P.get("seed"),
        "base": P.get("base", {}), "background": P.get("background", {}),
        "themes": P.get("themes", []), "palette": P.get("palette", []),
        "layers": stack(P), "questions": questions(piece),
        "comments": comments(piece),
        "decisions": P.get("decisions", [])[-30:],
        "final_v": int(final.stat().st_mtime) if final.exists() else 0,
        "has_psd": (d / "piece.psd").exists(),
        "busy": _lock(piece).locked(),
    }


def list_pieces() -> list[dict]:
    out = []
    if not PIECES.exists():
        return out
    for d in PIECES.iterdir():
        pj = d / "piece.json"
        if not pj.is_file():
            continue
        try:
            P = json.loads(pj.read_text())
        except ValueError:
            continue
        final = d / "final.jpg"
        mt = max(pj.stat().st_mtime, final.stat().st_mtime if final.exists() else 0)
        dec = (P.get("decisions") or [{}])[-1]
        last = dec.get("note") or dec.get("question") or ("init" if "init" in dec else "")
        out.append({"piece": d.name, "photo": Path(P.get("photo", "")).name,
                    "recipe": P.get("recipe"), "mtime": mt,
                    "changed": time.strftime("%d %b %H:%M", time.localtime(mt)),
                    "last": last, "final_v": int(mt) if final.exists() else 0,
                    "open_q": sum(1 for q in _safe_questions(d.name)
                                  if q.get("status") == "open")})
    return sorted(out, key=lambda p: -p["mtime"])


def _safe_questions(piece):
    try:
        return questions(piece)
    except (ValueError, FileNotFoundError):
        return []


def list_recipes() -> list[dict]:
    out = []
    for f in sorted(RECIPES.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        out.append({"name": f.stem, "summary": r.get("summary", ""),
                    "layers": len(r.get("layers", [])),
                    "palette": r.get("palette", []),
                    "breakdown": (RECIPES / f"{f.stem}__breakdown.jpg").exists()})
    return out


# -- actions -----------------------------------------------------------------

def _kv(changes) -> list[str]:
    """Accept ['a.b=1', ...] or {'a.b': 1}; values JSON-ish → CLI string form."""
    if isinstance(changes, dict):
        changes = [f"{k}={_fmt(v)}" for k, v in changes.items()]
    out = []
    for kv in changes:
        k, sep, _ = str(kv).partition("=")
        if not sep or not re.match(r"^[\w\-]+\.[\w\-]+$", k):
            raise ValueError(f"bad key=value {kv!r} (want id.param=value)")
        out.append(str(kv))
    return out


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def do_set(piece: str, changes, note: str = "") -> dict:
    piece_dir(piece)
    run_cli(piece, ["set", str(PIECES / piece), *_kv(changes), "--note", note or ""])
    return summary(piece)


def reseed(piece: str, layer_id: str, note: str = "") -> dict:
    return do_set(piece, [f"{layer_id}.seed={random.randint(1, 99_999)}"],
                  note or f"re-seed {layer_id}")


def do_render(piece: str, full: bool = False) -> dict:
    piece_dir(piece)
    run_cli(piece, ["render", str(PIECES / piece)] + (["--full"] if full else []))
    return summary(piece)


def do_options(piece: str, question: str, vary: str) -> dict:
    piece_dir(piece)
    key, sep, vals = vary.partition("=")
    if not sep or len([v for v in vals.split(",") if v]) < 2:
        raise ValueError("vary must be key=a,b[,c] with 2+ values")
    _kv([f"{key}=x"])
    out = run_cli(piece, ["options", str(PIECES / piece), "--question",
                          question, "--vary", vary])
    m = re.search(r"Question (q\d+):", out)
    qs = questions(piece)
    q = next((x for x in qs if m and x["id"] == m.group(1)), qs[-1] if qs else None)
    return {"question": q, "summary": summary(piece)}


def do_pick(piece: str, qid: str, choice: str, comment: str = "") -> dict:
    piece_dir(piece)
    if not re.match(r"^q\d+$", qid) or not re.match(r"^[A-Ha-h]$", choice):
        raise ValueError("bad question/choice")
    run_cli(piece, ["pick", str(PIECES / piece), "--question", qid,
                    "--choice", choice.upper(), "--comment", comment or ""])
    return summary(piece)


def piece_name_for(photo: str, recipe: str) -> str:
    return f"{Path(photo).stem}__{recipe}"


def start_init(photo: str, recipe: str, wildness: float, crop=None,
               custom: str | None = None, extra_set=None) -> dict:
    """Kick `init` in a background thread (riffs take ~1 min). Activity +
    ntfy via activity.py (key 'compose:<piece>')."""
    if not (RECIPES / f"{recipe}.json").is_file():
        raise FileNotFoundError(f"no recipe {recipe}")
    name = piece_name_for(photo, recipe)
    # a piece.json without final.jpg = an init that died mid-render → allow a retry
    if (PIECES / name / "final.jpg").exists():
        raise FileExistsError(name)
    args = ["init", "--photo", photo, "--recipe", recipe,
            "--wildness", f"{min(max(float(wildness), 0.0), 1.0):.2f}"]
    if custom:
        if not re.match(r"^\d+,\d+,\d+,\d+$", custom):
            raise ValueError("custom crop must be x1,y1,x2,y2")
        args += ["--custom", custom]
    elif crop not in (None, ""):
        args += ["--crop", str(int(crop))]
    if extra_set:
        args += ["--set", *_kv(extra_set)]
    job = {"id": uuid.uuid4().hex[:8], "piece": name, "photo": photo,
           "recipe": recipe, "state": "running", "started": time.time(), "error": ""}
    JOBS[job["id"]] = job
    key = f"compose:{name}"
    activity.start(key, f"compose init {name}")

    def _bg():
        try:
            run_cli(name, args, timeout=1800)
            job["state"] = "done"
            activity.finish(key, True, f"compose {name}")
        except Exception as e:
            job.update(state="failed", error=str(e)[-800:])
            activity.finish(key, False, f"compose {name}")
        job["ended"] = time.time()

    threading.Thread(target=_bg, daemon=True).start()
    return job


def thumb(path: Path, w: int = 480) -> Path:
    """Cached JPEG thumbnail on ext4 (phone bandwidth: finals are ~1MB)."""
    from hashlib import sha1

    from PIL import Image, ImageOps
    w = max(64, min(int(w), 1600))
    st = path.stat()
    key = sha1(f"{path}|{st.st_mtime_ns}|{w}".encode()).hexdigest()[:20]
    out = THUMBS / f"{key}.jpg"
    if not out.exists():
        THUMBS.mkdir(parents=True, exist_ok=True)
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((w, w * 2))
        tmp = out.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=82)
        tmp.rename(out)
    return out


def photo_choices(src: str, limit: int = 60) -> list[str]:
    """Photo pickers reused from Studio's existing pools."""
    if src == "favorites":
        try:
            favs = json.loads((SHARED / "favorites" / "favorites.json").read_text()
                              )["favorites"]
        except (OSError, ValueError, KeyError):
            favs = []
        files = [SHARED / "favorites" / f["file"] for f in reversed(favs)
                 if str(f.get("file", "")).lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        base = SHARED / ("finals" if src == "finals" else "candidates")
        files = sorted((p for p in base.glob("*") if p.suffix.lower() in
                        (".jpg", ".jpeg", ".png")),
                       key=lambda p: -p.stat().st_mtime)
    return [str(p) for p in files if p.is_file()][:limit]
