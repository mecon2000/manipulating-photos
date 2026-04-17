#!/home/rong/openclaw-venv/bin/python3
"""
Ink Splash — zero-cost local body ink-ification.

Converts the subject's skin/body into painted ink tones with procedural
splash/drip texture. Pure PIL + numpy + scipy + MediaPipe (no API calls).
Designed to chain with baroque-surround --use-cached-bg for unified
ink-body-on-ink-BG composites.

Pipeline:
  1. Body-skin mask (MediaPipe via masking.build_mask, local)
  2. Grayscale → ink palette remap (2-3 tones)
  3. Procedural noise overlay (ink-wash texture)
  4. Vertical drips at mask bottom edge
  5. Edge splatter (mask jitter)
  6. Composite onto original BG

Usage:
  ./ink-splash.py --source photo.jpg                          # default: indigo palette
  ./ink-splash.py --source photo.jpg --palette crimson --drip 0.7
  ./ink-splash.py --source photo.jpg --palette black --affect skin --exclude hands
  ./ink-splash.py --list-palettes
"""
import os, sys, argparse, random, re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

PALETTES = {
    "indigo":  [(10, 15, 40), (30, 55, 110), (150, 170, 210)],
    "black":   [(8, 8, 10),   (40, 40, 45),  (180, 180, 185)],
    "crimson": [(35, 5, 10),  (110, 25, 35), (220, 180, 180)],
    "sepia":   [(35, 25, 15), (110, 80, 45), (230, 215, 180)],
    "jade":    [(8, 30, 25),  (35, 90, 70),  (190, 220, 205)],
    "wine":    [(25, 5, 30),  (80, 25, 70),  (210, 180, 210)],
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def remap_to_palette(gray, palette):
    """Map a grayscale array (H,W) 0-255 to an RGB ink palette via 3 anchors."""
    dark, mid, light = [np.array(c, dtype=np.float32) for c in palette]
    g = gray.astype(np.float32) / 255.0
    out = np.zeros((*gray.shape, 3), dtype=np.float32)
    lo = g < 0.5
    hi = ~lo
    # dark→mid in [0,0.5]
    t = (g[lo] * 2.0)[..., None]
    out[lo] = dark * (1 - t) + mid * t
    # mid→light in [0.5,1]
    t = ((g[hi] - 0.5) * 2.0)[..., None]
    out[hi] = mid * (1 - t) + light * t
    return np.clip(out, 0, 255).astype(np.uint8)

def procedural_noise(h, w, scale, seed):
    """Low-frequency noise (0..1) via Gaussian-blurred random field."""
    rng = np.random.default_rng(seed)
    n = rng.random((h, w)).astype(np.float32)
    n = gaussian_filter(n, sigma=scale)
    n -= n.min(); n /= (n.max() + 1e-6)
    return n

def add_drips(mask_bin, strength, short_edge, seed):
    """Return a drip-extension mask (0-1 float) extending downward from mask bottom."""
    h, w = mask_bin.shape
    drips = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(seed + 1)
    # detect bottom edge per column
    any_mask = mask_bin.any(axis=0)
    max_drip_px = int(short_edge * 0.12 * strength)
    if max_drip_px < 5:
        return drips
    # find lowest y per column
    for x in range(w):
        if not any_mask[x]:
            continue
        col = mask_bin[:, x]
        ys = np.where(col)[0]
        if ys.size == 0:
            continue
        y_bot = ys.max()
        if rng.random() > 0.35:  # 35% of columns get a drip
            continue
        length = int(rng.integers(max_drip_px // 3, max_drip_px + 1))
        thickness = int(rng.integers(1, max(2, int(short_edge * 0.004))))
        for dy in range(length):
            y = y_bot + dy
            if y >= h: break
            alpha = 1.0 - (dy / length) ** 1.5
            x0 = max(0, x - thickness); x1 = min(w, x + thickness + 1)
            drips[y, x0:x1] = np.maximum(drips[y, x0:x1], alpha)
    # slight blur for softness
    drips = gaussian_filter(drips, sigma=max(1.0, short_edge * 0.0015))
    drips = np.clip(drips, 0, 1)
    return drips

def splatter_mask(mask_bin, short_edge, seed):
    """Jitter the mask edge slightly via noise-driven dilation/erosion to look hand-painted."""
    h, w = mask_bin.shape
    # Perlin-ish field
    n = procedural_noise(h, w, scale=short_edge * 0.008, seed=seed + 7)
    # threshold-jitter: per-pixel threshold wave
    threshold = 0.5 + (n - 0.5) * 0.3
    # soft-mask from distance transform
    from scipy.ndimage import distance_transform_edt
    inside = distance_transform_edt(mask_bin)
    outside = distance_transform_edt(~mask_bin.astype(bool))
    sdf = inside - outside
    sdf_norm = sdf / (short_edge * 0.01)
    soft = 1.0 / (1.0 + np.exp(-sdf_norm))
    # modulate with noise
    modulated = soft + (n - 0.5) * 0.25
    return np.clip(modulated, 0, 1)

def main():
    ap = argparse.ArgumentParser(description="Ink Splash — zero-cost body ink-ification")
    ap.add_argument("--source", required=True)
    ap.add_argument("--palette", default="indigo", help="|".join(PALETTES.keys()))
    ap.add_argument("--affect", default="skin", help="Mask target (skin, body-skin, hair, subject, etc.)")
    ap.add_argument("--exclude", default="hands", help="Mask exclude (hands, ropes, ...)")
    ap.add_argument("--noise-strength", type=float, default=0.35, help="Ink texture noise (0-1)")
    ap.add_argument("--drip", type=float, default=0.5, help="Drip strength (0-1, 0=none)")
    ap.add_argument("--splatter", type=float, default=0.7, help="Edge splatter roughness (0-1)")
    ap.add_argument("--contrast", type=float, default=1.2, help="Grayscale contrast before palette map")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bg", default="keep", choices=["keep", "black", "white"], help="What to do with BG")
    ap.add_argument("--output-to", default="local")
    ap.add_argument("--local-output-dir", default=None)
    ap.add_argument("--list-palettes", action="store_true")
    args = ap.parse_args()

    if args.list_palettes:
        for name, cols in PALETTES.items():
            print(f"  {name:10s} dark={cols[0]}  mid={cols[1]}  light={cols[2]}")
        return

    if args.palette not in PALETTES:
        print(f"Unknown palette '{args.palette}'. Use --list-palettes."); sys.exit(1)

    src_path = os.path.expanduser(args.source)
    if not os.path.isfile(src_path):
        print(f"Source not found: {src_path}"); sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    # output folder
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    path_parts = os.path.normpath(src_path).split(os.sep)
    model = "Unknown"
    for i, p in enumerate(path_parts):
        if p == "_photos" and i + 1 < len(path_parts):
            model = path_parts[i + 1]; break
    ts = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    folder = f"{model}_{src_name}_{ts}_ink-splash_{args.palette}_{seed % 100:02d}"
    folder = re.sub(r'[<>:"/\\|?*]', '_', folder)
    out_root = os.path.expanduser(args.local_output_dir or "~/.openclaw/workspace/shared")
    out_dir = os.path.join(out_root, folder)
    os.makedirs(out_dir, exist_ok=True)

    log(f"Source:   {src_path}")
    log(f"Palette:  {args.palette}  seed={seed}")
    log(f"Output:   {out_dir}")

    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    short_edge = min(w, h)
    img.save(os.path.join(out_dir, "0_original.jpg"), "JPEG", quality=95)

    log("Step 1: body mask via MediaPipe...")
    mask_pil, info = build_mask(img, affect=args.affect, exclude=args.exclude,
                                output_dir=out_dir, feather=0.0, cleanup="smooth")
    mask_pil.save(os.path.join(out_dir, "1_mask.png"))
    log(f"  engine={info.get('engine')} coverage={info.get('coverage_pct'):.1f}%")

    mask_bin = np.array(mask_pil) > 127
    mask_soft = splatter_mask(mask_bin, short_edge, seed)
    if args.splatter < 0.7:
        # less jitter → blend toward clean mask
        clean = mask_bin.astype(np.float32)
        mask_soft = mask_soft * args.splatter / 0.7 + clean * (1 - args.splatter / 0.7)

    # Step 2: ink-remap body
    log("Step 2: grayscale → palette remap...")
    rgb = np.array(img).astype(np.float32)
    gray = rgb.mean(axis=2)
    # contrast stretch
    m = gray.mean()
    gray = np.clip((gray - m) * args.contrast + m, 0, 255)
    ink = remap_to_palette(gray.astype(np.uint8), PALETTES[args.palette]).astype(np.float32)

    # Step 3: procedural noise texture overlay
    log("Step 3: ink-wash texture...")
    noise = procedural_noise(h, w, scale=short_edge * 0.006, seed=seed)
    # multiplicative modulation in range [1-s, 1+s/2]
    s = args.noise_strength
    mod = 1.0 + (noise - 0.5) * s
    ink = np.clip(ink * mod[..., None], 0, 255)

    # Step 4: drips
    log("Step 4: drips...")
    drip_alpha = add_drips(mask_bin, args.drip, short_edge, seed) if args.drip > 0.01 else np.zeros((h, w), dtype=np.float32)

    # Step 5: composite
    log("Step 5: compositing...")
    if args.bg == "keep":
        bg = rgb.copy()
    elif args.bg == "black":
        bg = np.zeros_like(rgb)
    else:  # white
        bg = np.full_like(rgb, 255)

    # combined alpha: body-soft-mask OR drip alpha
    drip_color = np.array(PALETTES[args.palette][0], dtype=np.float32)  # deepest ink
    alpha = np.maximum(mask_soft, drip_alpha)[..., None]

    # Where drips exist and body doesn't, use drip_color (dark)
    drip_only = np.clip(drip_alpha - mask_soft, 0, 1)[..., None]
    ink_with_drips = ink * (1 - drip_only) + drip_color * drip_only

    result = (bg * (1 - alpha) + ink_with_drips * alpha).astype(np.uint8)
    out_img = Image.fromarray(result)
    final_path = os.path.join(out_dir, f"{folder}.jpg")
    out_img.save(final_path, "JPEG", quality=94)
    log(f"Final: {final_path}")

    # Copy to finals/
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_path = os.path.join(finals_dir, f"{folder}.jpg")
    try:
        import shutil
        shutil.copy(final_path, finals_path)
        log(f"Copied to: {finals_path}")
    except Exception as e:
        log(f"finals copy failed: {e}")

    # Pushbullet
    try:
        from notify import push_image
        push_image(final_path, title=f"ink-splash {args.palette}")
        log("Pushed to phone.")
    except Exception as e:
        log(f"push failed: {e}")

    print(f"\nOutput: {final_path}")

if __name__ == "__main__":
    main()
