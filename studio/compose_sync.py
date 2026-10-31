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

DEBOUNCE_SEC = 2.0
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


def variant(piece: str, layer: str, seed: int | None = None) -> dict:
    compose.piece_dir(piece)
    if not re.match(r"^[\w\-]+$", layer):
        raise ValueError("bad layer id")
    seed = int(seed) if seed is not None else random.randint(1, 99_999)
    out = compose.run_cli(piece, ["variant", str(compose.PIECES / piece), layer,
                                  "--seed", str(seed)], timeout=600)
    m = re.search(r"Seed:\s*(\d+)", out)
    p = re.search(r"Variant:\s*(\S.*)$", out, re.M)
    seed = int(m.group(1)) if m else seed
    rel = f"layers/variants/{layer}/s{seed}.png"
    if p:
        try:
            rel = str(Path(p.group(1).strip()).resolve().relative_to(
                compose.piece_dir(piece).resolve()))
        except ValueError:
            pass
    return {"layer": layer, "seed": seed, "file": rel}


def save(piece: str, name: str) -> list[dict]:
    compose.piece_dir(piece)
    name = (name or "").strip()[:80] or time.strftime("save %H:%M")
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
