#!/home/rong/openclaw-venv/bin/python3
"""
Polish — standalone "un-AI-ify" pass on any finished photo.

Runs:
  1. Tensor Art ControlNet Tile refine (low-denoise re-texture)
  2. fal.ai face-swap (restore identity if denoise >= 0.2)
  3. Grain match from original source (if --grain-source provided)

Usage:
    ./polish.py --source photo.jpg
    ./polish.py --source photo.jpg --denoise 0.35
    ./polish.py --source-dir ~/path/to/favs/ --denoise 0.3
    ./polish.py --source photo.jpg --grain-source original.jpg  # match grain from original
"""

import os
import sys

# Load env
_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())
os.environ['FAL_KEY'] = os.environ.get('FAL_API_KEY', '')

import argparse
import time
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

from PIL import Image

# Import helpers from baroque-surround (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "baroque_surround",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "baroque-surround.py"),
)
_bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bs)

sys.stdout.reconfigure(line_buffering=True)


def log(msg, level="INFO"):
    ts = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def polish_one(source_path, denoise, prompt, grain_source, grain_strength, output_dir):
    """Polish a single image. Returns path to polished output."""
    src_name = os.path.splitext(os.path.basename(source_path))[0]
    log(f"Polishing: {src_name} (denoise={denoise})")
    img = Image.open(source_path).convert("RGB")

    # 1. Tile-refine
    t0 = time.time()
    log(f"  Tile-refine (denoise={denoise})...")
    refined = _bs.tensor_tile_refine(img, denoise, prompt, output_dir)
    if refined is None:
        log(f"  Tile-refine failed for {src_name}", "ERROR")
        return None
    log(f"  Tile-refine done ({time.time()-t0:.1f}s)")

    # 2. Face-swap if denoise high enough
    if denoise >= 0.2:
        t0 = time.time()
        log("  Face-swap...")
        swapped = _bs.fal_face_swap(source_path, refined, output_dir)
        if swapped is not None:
            refined = swapped
            log(f"  Face-swap done ({time.time()-t0:.1f}s)")
        else:
            log("  Face-swap failed — keeping tile-refined without swap", "WARN")

    # 3. Grain match (from source or provided original)
    grain_ref = Image.open(grain_source).convert("RGB") if grain_source else img
    if grain_ref.size != refined.size:
        grain_ref = grain_ref.resize(refined.size, Image.LANCZOS)
    if grain_strength > 0:
        log(f"  Grain match (strength={grain_strength})")
        refined = _bs.apply_grain_match(refined, grain_ref, grain_strength, output_dir)

    # Save
    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    out_name = f"{src_name}__polished_d{int(denoise*100):02d}_{timestamp}.jpg"
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    out_path = os.path.join(finals_dir, out_name)
    refined.save(out_path, "JPEG", quality=95)
    log(f"  Saved: {out_path}")

    # Push to phone unless disabled
    if not os.environ.get("NOTIFY_DISABLE"):
        try:
            from notify import push_image
            push_image(out_path, title=f"Polish d{int(denoise*100):02d}", body=src_name)
        except Exception as e:
            log(f"  Push failed: {e}", "WARN")

    return out_path


def main():
    p = argparse.ArgumentParser(description="Polish — standalone un-AI-ify pass")
    p.add_argument("--source", help="Single image to polish")
    p.add_argument("--source-dir", help="Directory of images to polish (batch)")
    p.add_argument("--denoise", type=float, default=0.3,
                   help="Tile-refine denoise 0.15-0.4 (default 0.3). Higher = more re-texture but more face drift.")
    p.add_argument("--prompt", default="raw photo, detailed skin texture, photographic, film grain, natural lighting, sharp focus, masterpiece, best quality",
                   help="Positive prompt for tile-refine pass")
    p.add_argument("--grain-source", default=None,
                   help="Path to original photo to source grain from. If omitted, grain is taken from the input itself (less useful — pass original portrait).")
    p.add_argument("--grain-strength", type=float, default=0.3,
                   help="Grain match strength (default 0.3). Lower than baroque's default because tile-refine already added some texture.")
    p.add_argument("--no-faceswap", action="store_true",
                   help="Skip face-swap even if denoise >= 0.2")
    args = p.parse_args()

    if not args.source and not args.source_dir:
        p.error("need --source or --source-dir")

    # Disable face-swap by setting denoise threshold high if asked
    denoise = args.denoise
    if args.no_faceswap:
        # Force denoise below 0.2 threshold so no swap runs, but keep user's intent
        # Actually simpler: monkey-patch by passing to polish_one via different path.
        # Here we just bypass face-swap in polish_one — but polish_one uses denoise >=0.2 check.
        # So: temporarily bump denoise to a value that polishes but skips face swap.
        # Cleanest: add a flag through. Quick fix: lower denoise threshold in polish_one.
        pass

    sources = []
    if args.source:
        sources = [os.path.expanduser(args.source)]
    if args.source_dir:
        d = os.path.expanduser(args.source_dir)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                sources.append(os.path.join(d, f))

    log(f"Polishing {len(sources)} image(s)")
    # Use shared finals dir as scratch for intermediate/debug
    work_dir = os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates/polish")
    os.makedirs(work_dir, exist_ok=True)

    done = 0
    for src in sources:
        # Skip face-swap cleanly by temporarily zeroing the threshold
        original_denoise = denoise
        if args.no_faceswap:
            # Run with a threshold-dodging technique — simplest: call helpers directly
            img = Image.open(src).convert("RGB")
            refined = _bs.tensor_tile_refine(img, denoise, args.prompt, work_dir)
            if refined is None:
                continue
            grain_ref = Image.open(args.grain_source).convert("RGB") if args.grain_source else img
            if grain_ref.size != refined.size:
                grain_ref = grain_ref.resize(refined.size, Image.LANCZOS)
            if args.grain_strength > 0:
                refined = _bs.apply_grain_match(refined, grain_ref, args.grain_strength, work_dir)
            ts = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
            out = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared/finals"),
                               f"{os.path.splitext(os.path.basename(src))[0]}__polished_d{int(denoise*100):02d}_noswap_{ts}.jpg")
            refined.save(out, "JPEG", quality=95)
            log(f"  Saved (no-swap): {out}")
            done += 1
        else:
            out = polish_one(src, denoise, args.prompt, args.grain_source, args.grain_strength, work_dir)
            if out:
                done += 1
    log(f"Done: {done}/{len(sources)} polished")


if __name__ == "__main__":
    main()
