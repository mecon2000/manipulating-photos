#!/home/rong/openclaw-venv/bin/python3
"""Color Bath — dominant single-color scene wash via LAB a*/b* channel shift.

Pure local, zero-cost. Preserves luminance structure, bathes the scene in
one dominant color (red-film, ochre, teal-moody, amber, etc.).
"""

import os
import sys
import argparse
import random
import shutil
import threading
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone
from PIL import Image

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

sys.stdout.reconfigure(line_buffering=True)
_log_lock = threading.Lock()


def log(level, msg):
    ts = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        print(f"[{ts}] [{level}] {msg}", flush=True)


# RGB target colors per preset. Algorithm converts these to LAB and pulls
# the image's a*/b* channels toward the target's a*/b* by `strength`.
PRESETS = {
    "red-film":    {"rgb": (200,  35,  35), "desc": "80s/90s Japanese red monochrome"},
    "ochre":       {"rgb": (210, 150,  55), "desc": "gold/yellow painterly wall"},
    "teal-moody":  {"rgb": ( 40, 115, 130), "desc": "cool teal/blue room"},
    "amber":       {"rgb": (215, 140,  55), "desc": "warm honey tone"},
    "blue-hour":   {"rgb": ( 55,  95, 170), "desc": "cold twilight"},
    "rose":        {"rgb": (220, 130, 150), "desc": "soft pink"},
    "sepia":       {"rgb": (180, 140,  95), "desc": "classic vintage"},
    "emerald":     {"rgb": ( 40, 130,  75), "desc": "deep green"},
    "magenta-dusk":{"rgb": (180,  70, 140), "desc": "neon magenta dusk"},
    "cyan-ice":    {"rgb": ( 80, 180, 200), "desc": "icy cyan"},
}


def rgb_to_lab_target(rgb):
    """Convert a single RGB triplet to LAB (OpenCV uint8 LAB space)."""
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def apply_color_bath(img_rgb, target_rgb, strength=0.75, preserve_shadows=False):
    """Shift a*/b* channels toward target color; keep L* intact."""
    src = img_rgb.astype(np.uint8)
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    _, ta, tb = rgb_to_lab_target(target_rgb)

    # Per-pixel blend weight (uniform unless preserve_shadows)
    if preserve_shadows:
        # L is 0..255 in OpenCV LAB. Shadows (L<80) get progressively less wash.
        shadow_factor = np.clip((L - 30.0) / 60.0, 0.0, 1.0)  # 0 at L=30, 1 at L=90
        w = strength * shadow_factor
    else:
        w = np.full_like(L, strength, dtype=np.float32)

    new_a = a * (1.0 - w) + ta * w
    new_b = b * (1.0 - w) + tb * w

    out_lab = np.stack([L, new_a, new_b], axis=-1)
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    return out_rgb


def parse_rgb(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--custom-hue must be R,G,B")
    return tuple(max(0, min(255, int(p))) for p in parts)


def find_model_name(source_path):
    """Extract model name from _photos/ path, fallback to 'Unknown'."""
    parts = os.path.abspath(source_path).split(os.sep)
    try:
        i = parts.index("_photos")
        return parts[i + 1]
    except (ValueError, IndexError):
        return "Unknown"


def run(args):
    src = os.path.abspath(os.path.expanduser(args.source))
    if not os.path.isfile(src):
        log("ERROR", f"Source not found: {src}")
        sys.exit(1)

    # Resolve target color
    if args.custom_hue:
        target = args.custom_hue
        preset_label = "custom"
    else:
        if args.preset not in PRESETS:
            log("ERROR", f"Unknown preset '{args.preset}'. Available: {list(PRESETS)}")
            sys.exit(1)
        target = PRESETS[args.preset]["rgb"]
        preset_label = args.preset

    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    log("INFO", f"Loading {src}")
    img = Image.open(src).convert("RGB")
    w, h = img.size
    log("INFO", f"Image {w}x{h} — preset={preset_label} target_rgb={target} strength={args.strength} preserve_shadows={args.preserve_shadows}")

    arr = np.asarray(img)
    log("INFO", "Converting RGB->LAB and shifting a*/b* toward target...")
    result_arr = apply_color_bath(arr, target, strength=args.strength, preserve_shadows=args.preserve_shadows)
    result = Image.fromarray(result_arr)

    # Output folder: {Model}_{srcname}_{timestamp}_color-bath_{preset}_{seed%100:02d}
    model_name = find_model_name(src)
    src_stem = os.path.splitext(os.path.basename(src))[0]
    ts = datetime.now(ISRAEL_TZ).strftime("%Y%m%d_%H%M%S")
    folder = f"{model_name}_{src_stem}_{ts}_color-bath_{preset_label}_{seed % 100:02d}"
    out_root = os.path.expanduser(args.local_output_dir)
    out_dir = os.path.join(out_root, folder)
    os.makedirs(out_dir, exist_ok=True)

    orig_path = os.path.join(out_dir, "0_original.jpg")
    img.save(orig_path, quality=95)
    log("INFO", f"Saved original: {orig_path}")

    final_name = f"{src_stem}_color-bath_{preset_label}.jpg"
    final_path = os.path.join(out_dir, final_name)
    result.save(final_path, quality=95)
    log("INFO", f"Saved final: {final_path}")

    # Copy to finals/
    try:
        finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
        os.makedirs(finals_dir, exist_ok=True)
        finals_path = os.path.join(finals_dir, f"{folder}.jpg")
        shutil.copy2(final_path, finals_path)
        log("INFO", f"Copied to finals: {finals_path}")
    except Exception as e:
        log("WARN", f"Finals copy failed: {e}")

    # Push to phone
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from notify import push_image
        push_image(final_path, title=f"color-bath {preset_label}", body=f"{src_stem} strength={args.strength}")
        log("INFO", "Pushed to phone")
    except Exception as e:
        log("WARN", f"Push failed: {e}")

    return final_path


def main():
    p = argparse.ArgumentParser(description="Color Bath — single-dominant-color scene wash (LAB a*/b*)")
    p.add_argument("--source")
    p.add_argument("--preset", default="red-film")
    p.add_argument("--strength", type=float, default=0.75, help="0-1 blend weight toward target (default 0.75)")
    p.add_argument("--custom-hue", type=parse_rgb, default=None, help="R,G,B overrides preset")
    p.add_argument("--preserve-shadows", action="store_true", help="Keep deep shadows neutral (chiaroscuro)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-to", default="local", choices=["local", "gdrive", "both"])
    p.add_argument("--local-output-dir", default=os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"))
    p.add_argument("--list-presets", action="store_true")

    args = p.parse_args()

    if args.list_presets:
        for name, info in PRESETS.items():
            print(f"  {name:14s} rgb={info['rgb']}  — {info['desc']}")
        return

    if not args.source:
        p.error("--source is required")
    run(args)


if __name__ == "__main__":
    main()
