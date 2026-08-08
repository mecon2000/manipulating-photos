"""Built-in generative inpaint step — "add flowers at #1"-type edits.

Not a registry tool: a Studio-native step so the buddy isn't boxed into the
pre-made pipelines. Flux inpainting on fal (the repo's proven endpoint), mask
from Ronnie's brush/Select mask, a SAM point-mask, or a marker region.

Known limit (documented in CLAUDE.md): Flux scans INPUT images — explicit
frames come back black. We detect that (brightness) and raise a clear error
so the agent says so instead of showing a black hole.
"""
import io
import os
import time
import uuid

import requests
from PIL import Image, ImageFilter

from . import cache
from .paths import RUNS_DIR, ensure_dirs

FAL_MODEL = "fal-ai/flux-general/inpainting"
BLACK_MEAN_THRESHOLD = 10


def _ensure_fal_key() -> None:
    if os.environ.get("FAL_KEY"):
        return
    env_file = os.path.expanduser("~/sol/.env")
    try:
        for line in open(env_file):
            line = line.strip()
            if line.startswith("FAL_API_KEY=") or line.startswith("FAL_KEY="):
                os.environ.setdefault("FAL_API_KEY", line.split("=", 1)[1])
    except OSError:
        pass
    key = os.environ.get("FAL_API_KEY")
    if not key:
        raise RuntimeError("FAL_API_KEY not set (checked env and ~/sol/.env)")
    os.environ["FAL_KEY"] = key   # fal_client wants FAL_KEY, env has FAL_API_KEY


def style_suffix(input_ref: str) -> str:
    """Auto style-match: describe the image's medium/palette/lighting so a bare
    'add flowers here' prompt inherits the photo's look (realistic vs anime vs
    painterly). Local VLM; cached per input ref; empty string on failure."""
    key = cache.step_key({"step": "__style_desc__", "input": input_ref})
    rec = cache.get_step(key)
    if rec is not None:
        return rec.get("text", "")
    try:
        from . import eyes
        text = eyes.describe(
            str(cache.object_path(input_ref)),
            question=("In under 15 words, name this image's medium and look — "
                      "e.g. 'realistic low-key photograph, teal-tinted, soft "
                      "window light' or 'anime illustration, pastel'. "
                      "Just the phrase."))["text"].strip().strip(".")
    except Exception:
        text = ""
    # store alongside a tiny placeholder object so the step-record cache works
    rec = cache.put_step(key, input_ref, {"step": "__style_desc__", "text": text})
    return text


def run_inpaint(input_ref: str, mask_ref: str, prompt: str,
                strength: float = 0.95, grow_mask_pct: float = 2.0,
                seed: int | None = None, match_style: bool = True) -> str:
    """Inpaint the masked region; returns the output object ref."""
    _ensure_fal_key()
    import fal_client

    if match_style:
        suffix = style_suffix(input_ref)
        if suffix:
            prompt = f"{prompt}, seamlessly matching the image style: {suffix}"

    src_path = cache.object_path(input_ref)
    mask_path = cache.object_path(mask_ref)
    if src_path is None or mask_path is None:
        raise RuntimeError("input or mask ref not in object store")

    img = Image.open(src_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(img.size)
    if grow_mask_pct > 0:   # feather+grow so the seam blends (scaled to image)
        radius = max(2, int(min(img.size) * grow_mask_pct / 100))
        mask = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
        mask = mask.filter(ImageFilter.GaussianBlur(radius // 2))

    ensure_dirs()
    tmp_img = RUNS_DIR / f"inpaint-src-{uuid.uuid4().hex[:8]}.jpg"
    tmp_mask = RUNS_DIR / f"inpaint-mask-{uuid.uuid4().hex[:8]}.png"
    img.save(tmp_img, "JPEG", quality=95)
    mask.save(tmp_mask, "PNG")
    try:
        args = {"image_url": fal_client.upload_file(str(tmp_img)),
                "mask_url": fal_client.upload_file(str(tmp_mask)),
                "prompt": prompt, "strength": float(strength),
                "num_images": 1, "output_format": "jpeg",
                "enable_safety_checker": False}
        if seed is not None:
            args["seed"] = int(seed)
        result = fal_client.submit(FAL_MODEL, arguments=args).get()
    finally:
        tmp_img.unlink(missing_ok=True)
        tmp_mask.unlink(missing_ok=True)

    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"inpaint returned no image: {str(result)[:200]}")
    out = Image.open(io.BytesIO(
        requests.get(images[0]["url"], timeout=120).content)).convert("RGB")
    if sum(out.resize((64, 64)).convert("L").getdata()) / (64 * 64) \
            < BLACK_MEAN_THRESHOLD:
        raise RuntimeError(
            "inpaint came back black — Flux blocks explicit input images; "
            "this frame is too explicit for the inpaint model")

    out_path = RUNS_DIR / f"inpaint-{int(time.time())}-{uuid.uuid4().hex[:6]}.jpg"
    out.save(out_path, "JPEG", quality=95)
    ref = cache.put_file(out_path)
    out_path.unlink()
    return ref


META = {
    "label": "Inpaint (generative edit)",
    "params": {
        "prompt": {"type": "string", "default": "",
                   "description": "What to paint into the masked region"},
        "mask": {"type": "string", "default": "",
                 "description": "Mask object ref (brush / Select / SAM)"},
        "strength": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.95,
                     "description": "How fully to repaint the region"},
        "grow-mask-pct": {"type": "float", "min": 0.0, "max": 8.0, "default": 2.0,
                          "description": "Grow+feather mask, % of short edge"},
        "match-style": {"type": "int", "min": 0, "max": 1, "default": 1,
                        "description": "Auto-append the image's style/medium to "
                                       "the prompt (bare prompts inherit the look)"},
    },
    "presets": None, "artifacts": None, "flags": [], "flag_descriptions": {},
    "deterministic": False, "output_kind": "image", "builtin": True,
    "cost_estimate_usd": 0.03, "wall_time_estimate_sec": 20,
    "needs_style_ref": False,
}
