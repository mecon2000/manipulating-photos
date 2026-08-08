"""Semantic mask building — thin wrapper over scripts/workflows/masking.py.

Runs in-process (MediaPipe is ~0.5s; BiRefNet hits fal.ai) and stores the
resulting mask PNG in the object cache, keyed by (input, affect, exclude,
feather, cleanup) so repeat requests are free.
"""
import sys

from . import cache
from .paths import REPO, RUNS_DIR, ensure_dirs

_WORKFLOWS = REPO / "scripts" / "workflows"
if str(_WORKFLOWS) not in sys.path:
    sys.path.insert(0, str(_WORKFLOWS))

import masking  # noqa: E402  (scripts/workflows/masking.py)


def region_mask(input_ref: str, x: float, y: float,
                rx: float = 0.15, ry: float = 0.15) -> dict:
    """Soft elliptical blob mask centered at normalized (x, y) — for ADDING
    new content at a spot (SAM segments existing objects; an empty patch of
    background would segment as the whole background)."""
    from PIL import Image, ImageDraw, ImageFilter
    key = cache.step_key({"step": "__region_mask__", "input": input_ref,
                          "x": round(x, 4), "y": round(y, 4),
                          "rx": round(rx, 4), "ry": round(ry, 4)})
    rec = cache.get_step(key)
    if rec:
        return {"mask": rec["output"], "cache_hit": True}
    src = cache.object_path(input_ref)
    if src is None:
        raise FileNotFoundError(f"input ref {input_ref} not in object store")
    W, H = Image.open(src).size
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([x * W - rx * W, y * H - ry * H, x * W + rx * W, y * H + ry * H],
              fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(int(min(W, H) * 0.02)))
    ensure_dirs()
    tmp = RUNS_DIR / f"region-{key[:12]}.png"
    mask.save(tmp)
    ref = cache.put_file(tmp)
    tmp.unlink()
    cache.put_step(key, ref, {"step": "__region_mask__"})
    return {"mask": ref, "cache_hit": False}


def build_mask(input_ref: str, affect: str, exclude: str = "",
               rope_color: str = "auto", feather: float = 0.5,
               cleanup: str = "smooth") -> dict:
    key = cache.step_key({"step": "__mask__", "input": input_ref,
                          "affect": affect, "exclude": exclude,
                          "rope_color": rope_color, "feather": feather,
                          "cleanup": cleanup})
    rec = cache.get_step(key)
    if rec:
        return {"mask": rec["output"], "cache_hit": True,
                "info": rec.get("info", {})}

    src = cache.object_path(input_ref)
    if src is None:
        raise FileNotFoundError(f"input ref {input_ref} not in object store")
    mask_pil, info = masking.build_mask(str(src), affect, exclude,
                                        rope_color=rope_color, feather=feather,
                                        cleanup=cleanup)
    ensure_dirs()
    tmp = RUNS_DIR / f"mask-{key[:12]}.png"
    mask_pil.save(tmp)
    ref = cache.put_file(tmp)
    tmp.unlink()
    cache.put_step(key, ref, {"step": "__mask__", "info": info})
    return {"mask": ref, "cache_hit": False, "info": info}
