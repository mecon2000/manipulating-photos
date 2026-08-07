"""SAM 2 tap-to-mask microservice — FastAPI on 127.0.0.1:8703.

Run:  ~/openclaw-venv/bin/python3 -m uvicorn studio.sam_service:app \
          --host 127.0.0.1 --port 8703

Loads sam2.1-hiera-tiny once (~1GB VRAM — chosen over -small because the
Windows desktop already idles ~4.5GB of the 3060 Ti's 8GB and gemma3:4b
vision needs ~3GB when it spins up; measured 2026-08-08). Image embeddings
are cached per object ref (LRU of 4), so repeat taps on the same photo are
sub-second.

POST /segment {ref|path, points: [[x,y]..] normalized, labels: [1|0..], box?}
  → {mask: <object ref>, score} — mask stored as PNG in the studio cache.
POST /embed {ref|path} → prewarm the embedding.
"""
import io
import threading
from collections import OrderedDict

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from PIL import Image

from . import cache
from .paths import RUNS_DIR, ensure_dirs

MODEL_ID = "facebook/sam2.1-hiera-tiny"
EMBED_LRU = 4

app = FastAPI(title="studio-sam")
_lock = threading.Lock()
_predictor = None
_embeds: OrderedDict[str, dict] = OrderedDict()   # ref -> predictor state


def _get_predictor():
    global _predictor
    if _predictor is None:
        import torch
        # WSL: torch 2.12 cu13 wheels hit CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
        # on this driver; native CUDA kernels are plenty for the tiny model
        # (measured: embed 0.23s, predict 0.09s).
        torch.backends.cudnn.enabled = False
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=device)
        print(f"[sam] loaded {MODEL_ID} on {device}")
    return _predictor


def _load_rgb(ref_or_path: str) -> tuple[np.ndarray, str]:
    p = cache.object_path(ref_or_path)
    if p is None:
        raise HTTPException(404, f"unknown object ref {ref_or_path!r} "
                                 "(pass a studio cache ref)")
    img = Image.open(p).convert("RGB")
    return np.array(img), ref_or_path


def _set_image(ref_or_path: str):
    """Predictor with the image embedded; embeddings LRU-cached per ref."""
    predictor = _get_predictor()
    arr, key = _load_rgb(ref_or_path)
    state = _embeds.get(key)
    if state is not None:
        _embeds.move_to_end(key)
        predictor._features = state["features"]
        predictor._orig_hw = state["orig_hw"]
        predictor._is_image_set = True
    else:
        predictor.set_image(arr)
        _embeds[key] = {"features": predictor._features,
                        "orig_hw": predictor._orig_hw}
        while len(_embeds) > EMBED_LRU:
            _embeds.popitem(last=False)
    return predictor, arr.shape[:2]


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "loaded": _predictor is not None}


@app.post("/embed")
def embed(body: dict = Body(...)):
    with _lock:
        _, (h, w) = _set_image(body["ref"])
    return {"ok": True, "size": [w, h]}


@app.post("/segment")
def segment(body: dict = Body(...)):
    points = body.get("points") or []
    box = body.get("box")
    if not points and not box:
        raise HTTPException(400, "need points or box")
    labels = body.get("labels") or [1] * len(points)

    with _lock:
        predictor, (h, w) = _set_image(body["ref"])
        pts = np.array([[x * w, y * h] for x, y in points]) if points else None
        lbl = np.array(labels) if points else None
        bx = np.array([box[0] * w, box[1] * h, box[2] * w, box[3] * h]) \
            if box else None
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=lbl, box=bx, multimask_output=True)

    best = int(np.argmax(scores))
    mask = (masks[best].astype(np.uint8)) * 255
    ensure_dirs()
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, "PNG")
    tmp = RUNS_DIR / "sam-mask.png"
    tmp.write_bytes(buf.getvalue())
    ref = cache.put_file(tmp)
    tmp.unlink()
    return {"mask": ref, "score": float(scores[best]),
            "coverage_pct": round(float(mask.mean()) / 2.55, 1)}
