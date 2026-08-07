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
