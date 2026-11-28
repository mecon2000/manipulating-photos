"""Fast, hands-on Compose (round 2): no-render edits + debounced background render.

The browser composites layers/manifest.json itself, so edits are instant client-side;
here every quick edit persists with `--no-render` (<1s) and (re)arms a per-piece
debounce timer. When the piece has been idle for DEBOUNCE_SEC a full `render` runs
in a thread so final.jpg / history / layers catch up. Edits that land during a
render bump the generation and trigger one more render afterwards.
"""
import json
import random
import re
import threading
import time
from pathlib import Path

from . import activity, compose

DEBOUNCE_SEC = 0.3
_G = threading.Lock()
_S: dict[str, dict] = {}   # piece -> {timer, rendering, gen, rendered_gen, error}


def _st(piece: str) -> dict:
    with _G:
        return _S.setdefault(piece, {"timer": None, "rendering": False, "gen": 0,
                                     "rendered_gen": 0, "error": ""})


def kick_render(piece: str, delay: float = DEBOUNCE_SEC) -> None:
    st = _st(piece)
    with _G:
        st["gen"] += 1
        if st["timer"]:
            st["timer"].cancel()
        t = threading.Timer(delay, _render_bg, args=(piece,))
        t.daemon = True
        st["timer"] = t
    t.start()


def _render_bg(piece: str) -> None:
    st = _st(piece)
    with _G:
        st["timer"] = None
        if st["rendering"]:
            return  # the running render re-checks gen when it finishes
        st["rendering"] = True
        gen = st["gen"]
    try:
        compose.run_cli(piece, ["render", str(compose.PIECES / piece)])
        st["error"] = ""
    except Exception as e:
        st["error"] = str(e)[-600:]
    with _G:
        st["rendering"] = False
        st["rendered_gen"] = gen
        again = st["gen"] != gen and st["timer"] is None
    if again:
        kick_render(piece, 0.2)


def status(piece: str) -> dict:
    st = _st(piece)
    with _G:
        return {"pending": bool(st["timer"]) or st["rendering"]
                or st["gen"] != st["rendered_gen"],
                "rendering": st["rendering"], "error": st["error"],
                "fav": dict(st["fav"]) if st.get("fav") else None}


# -- reads ---------------------------------------------------------------------

def manifest(piece: str) -> dict | None:
    try:
        return json.loads((compose.piece_dir(piece) / "layers" / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def savepoints(piece: str) -> list[dict]:
    try:
        idx = json.loads((compose.piece_dir(piece) / "savepoints" / "index.json").read_text())
    except (OSError, ValueError):
        return []
    out = []
    for s in idx if isinstance(idx, list) else idx.get("savepoints", []):
        rel = f"savepoints/{Path(str(s.get('dir', ''))).name}"
        out.append({**s, "rel": rel})
    return out


# -- quick edits (no render) ---------------------------------------------------------

def quick_set(piece: str, changes, note: str = "") -> dict:
    compose.piece_dir(piece)
    compose.run_cli(piece, ["set", str(compose.PIECES / piece), *compose._kv(changes),
                            "--note", note or "", "--no-render"], timeout=120)
    kick_render(piece)
    return status(piece)


def order(piece: str, ids: list[str]) -> dict:
    compose.piece_dir(piece)
    ids = [str(i) for i in ids]
    if "subject" not in ids or any(not re.match(r"^[\w\-]+$", i) for i in ids):
        raise ValueError("order must be layer ids incl. 'subject'")
    compose.run_cli(piece, ["order", str(compose.PIECES / piece), ",".join(ids),
                            "--no-render"], timeout=120)
    kick_render(piece)
    return status(piece)


def variant(piece: str, layer: str, seed: int | None = None, count: int = 3) -> dict:
    """Roll `count` new versions of one layer in one engine pass (or one given seed)."""
    compose.piece_dir(piece)
    if not re.match(r"^[\w\-]+$", layer):
        raise ValueError("bad layer id")
    seeds = [int(seed)] if seed is not None else \
        random.sample(range(1, 99_999), max(1, min(6, int(count or 3))))
    out = compose.run_cli(piece, ["variant", str(compose.PIECES / piece), layer,
                                  "--seed", *map(str, seeds)], timeout=600)
    got = [int(x) for x in re.findall(r"Seed:\s*(\d+)", out)] or seeds
    files = [f"layers/variants/{layer}/s{sd}.png" for sd in got]
    return {"layer": layer, "seeds": got, "files": files, "seed": got[0], "file": files[0]}


def save(piece: str, name: str) -> list[dict]:
    compose.piece_dir(piece)
    # no name asked: save points are just numbered (#1, #2 …)
    n = max([int(sp.get("n", 0)) for sp in savepoints(piece)] + [0]) + 1
    name = (name or "").strip()[:80] or f"#{n}"
    compose.run_cli(piece, ["save", str(compose.PIECES / piece), "--name", name], timeout=120)
    return savepoints(piece)


def start_fav(piece: str, name: str = "") -> dict:
    """`fav` (~17s: full-res + PSD + favorites entry + fav savepoint) in a thread;
    activity + ntfy like init. Progress rides status()['fav']."""
    compose.piece_dir(piece)
    st = _st(piece)
    if st.get("fav", {}).get("state") == "running":
        return dict(st["fav"])
    st["fav"] = {"state": "running", "started": time.time(), "jpg": "", "psd": "",
                 "psd_mb": None, "error": ""}
    key = f"compose:{piece}"
    activity.start(key, f"fav {piece}")
    args = ["fav", str(compose.PIECES / piece)] + (["--name", name.strip()[:80]]
                                                    if (name or "").strip() else [])

    def _bg():
        f = st["fav"]
        try:
            out = compose.run_cli(piece, args, timeout=1800)
            jpg = re.search(r"^Fav:\s*(.+)$", out, re.M)
            psd = re.search(r"^PSD:\s*(.+)$", out, re.M)
            f["jpg"] = jpg.group(1).strip() if jpg else ""
            f["psd"] = psd.group(1).strip() if psd else ""
            try:
                f["psd_mb"] = round(Path(f["psd"]).stat().st_size / 1e6, 1)
            except OSError:
                pass
            f["state"] = "done"
            activity.finish(key, True, f"fav {piece}")
        except Exception as e:
            f.update(state="failed", error=str(e)[-600:])
            activity.finish(key, False, f"fav {piece}")
        f["ended"] = time.time()

    threading.Thread(target=_bg, daemon=True).start()
    return dict(st["fav"])


def restore(piece: str, n) -> dict:
    compose.piece_dir(piece)
    compose.run_cli(piece, ["restore", str(compose.PIECES / piece),
                            "--savepoint", str(int(n))], timeout=600)
    return compose.summary(piece)


def zoom(piece: str, factor: float) -> dict:
    """Re-crop the piece from its original photo, wider (factor > 1) or tighter, around the
    same centre. New source / mask / face data; the layers keep their settings."""
    compose.piece_dir(piece)
    f = max(0.5, min(2.0, float(factor)))
    compose.run_cli(piece, ["recrop", str(compose.PIECES / piece), "--zoom", str(f),
                            "--note", f"zoom {'out' if f > 1 else 'in'} x{f:g}"], timeout=900)
    return compose.summary(piece)


def recrop(piece: str, box: str) -> dict:
    """Re-crop the piece from its original photo to `box` (x1,y1,x2,y2 px). Layers keep
    their settings; a hand-placed tear follows the face."""
    compose.piece_dir(piece)
    compose.run_cli(piece, ["recrop", str(compose.PIECES / piece), "--custom", box,
                            "--note", f"crop {box}"], timeout=900)
    return compose.summary(piece)


# ---- live slider preview: one layer at low res, computed IN this process ---------------
# A subprocess per slider move paid ~1-8 s just importing numpy/scipy (cold cache); here
# layer_compose is imported once and kept warm. One worker; only the newest request per
# piece is computed (stale drags are skipped).
_peek_lock = threading.Lock()
_peek_gen: dict = {}
_lc = None


def _lc_mod():
    global _lc
    if _lc is None:
        import importlib, sys as _sys
        wf = str(compose.LC.parent)
        if wf not in _sys.path:
            _sys.path.insert(0, wf)
        _lc = importlib.import_module("layer_compose")
    return _lc


def warm_peek() -> None:
    threading.Thread(target=_lc_mod, daemon=True).start()


def peek(piece: str, layer: str, kv: list, size: int = 560) -> bytes | None:
    """PNG bytes of `layer` re-generated with the kv changes (not saved), or None when a
    newer request for this piece arrived while waiting."""
    import io
    compose.piece_dir(piece)
    with _G:
        g = _peek_gen.get(piece, 0) + 1
        _peek_gen[piece] = g
    with _peek_lock:
        if _peek_gen.get(piece) != g:
            return None
        lc = _lc_mod()
        d, P = lc.load(compose.PIECES / piece)
        lay = lc.peek_layer(d, P, layer, kv, max(240, min(int(size), 900)))
        if lay is None:
            return None
        from PIL import Image
        import numpy as np
        rgba = np.dstack([lay.rgb, lay.alpha])
        buf = io.BytesIO()
        Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(buf, "PNG", compress_level=1)
        return buf.getvalue()
