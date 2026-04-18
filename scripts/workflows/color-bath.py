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
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

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


def face_center_and_radius(img_pil):
    """Return ((cx, cy), radius_px) of face-skin region, or mask centroid fallback."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from masking import build_mask
        face_mask, _ = build_mask(img_pil, affect="face-skin", output_dir=None, feather=0.0)
        arr = np.array(face_mask) > 127
        if arr.sum() > 200:
            ys, xs = np.where(arr)
            cx, cy = float(xs.mean()), float(ys.mean())
            # radius ≈ face extent
            r = float(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0)
            return (cx, cy), r
    except Exception as e:
        log("WARN", f"face detection fallback: {e}")
    h, w = np.array(img_pil).shape[:2]
    return (w / 2.0, h / 3.0), min(w, h) * 0.12


def radial_falloff(w, h, center, inner_r, outer_mult=3.5):
    """Return (H,W) float in [0,1] — 0 at center (preserved), 1 past outer."""
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outer = inner_r * outer_mult
    norm = np.clip((dist - inner_r) / max(1.0, outer - inner_r), 0.0, 1.0)
    return norm


def add_grain(img_arr, falloff, strength, seed):
    """Gradient grain: 0 at face, up to ±strength*40 at edges."""
    h, w = img_arr.shape[:2]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=0.7)  # keep sharp-ish
    mod = noise[..., None] * (falloff[..., None] * strength * 40.0)
    return np.clip(img_arr.astype(np.float32) + mod, 0, 255).astype(np.uint8)


SCRATCHES_DIR = os.path.expanduser("~/.openclaw/workspace/shared/overlay_assets/scratches")
SCRATCH_PROMPTS = [
    "pure black background covered with white film scratches, thin irregular hair-line scratches with varying thickness and breaks, analog film damage texture, no subject, monochrome, high contrast",
    "black background with random white vertical and diagonal scratches, torn film emulsion, dust and hair-line marks, monochrome texture",
    "scratched film overlay, white scratches hairs dust and streaks on pure black, varying angles and lengths, film grain damage, monochrome",
    "old film scratches pattern, white irregular lines with forks and breaks on black, damaged celluloid, monochrome texture",
]


def ensure_scratch_overlays():
    """Generate scratch PNGs via Flux Schnell on first use, cache to disk."""
    os.makedirs(SCRATCHES_DIR, exist_ok=True)
    existing = sorted([f for f in os.listdir(SCRATCHES_DIR) if f.endswith(".png")])
    if len(existing) >= len(SCRATCH_PROMPTS):
        return [os.path.join(SCRATCHES_DIR, f) for f in existing[: len(SCRATCH_PROMPTS)]]
    log("INFO", f"Generating {len(SCRATCH_PROMPTS) - len(existing)} scratch overlays via Flux Schnell...")
    _env = os.path.expanduser("~/sol/.env")
    if os.path.isfile(_env):
        for ln in open(_env):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")
    import fal_client, requests
    from io import BytesIO
    for i, prompt in enumerate(SCRATCH_PROMPTS):
        fname = f"scratch_{i:02d}.png"
        fpath = os.path.join(SCRATCHES_DIR, fname)
        if os.path.exists(fpath):
            continue
        handle = fal_client.submit("fal-ai/flux/schnell", arguments={
            "prompt": prompt, "image_size": {"width": 1024, "height": 1024},
            "num_inference_steps": 4, "num_images": 1, "output_format": "png",
            "enable_safety_checker": False, "seed": 1000 + i,
        })
        url = handle.get()["images"][0]["url"]
        r = requests.get(url, timeout=60); r.raise_for_status()
        Image.open(BytesIO(r.content)).convert("RGB").save(fpath, "PNG")
        log("INFO", f"  cached: {fname}")
    return [os.path.join(SCRATCHES_DIR, f) for f in sorted(os.listdir(SCRATCHES_DIR)) if f.endswith(".png")]


def add_scratches(img_arr, strength, seed, subject_mask=None):
    """Composite one scratch-texture overlay. Avoids subject, low opacity by default."""
    h, w = img_arr.shape[:2]
    rng = np.random.default_rng(seed + 7)
    try:
        overlays = ensure_scratch_overlays()
    except Exception as e:
        log("WARN", f"scratch overlay generation failed: {e} — skipping")
        return img_arr
    if not overlays:
        return img_arr
    pick = str(rng.choice(overlays))
    tex = Image.open(pick).convert("L")
    tw, th = tex.size
    cs = int(min(tw, th) * rng.uniform(0.55, 1.0))
    cx0 = int(rng.integers(0, tw - cs + 1))
    cy0 = int(rng.integers(0, th - cs + 1))
    tex = tex.crop((cx0, cy0, cx0 + cs, cy0 + cs))
    scale = max(w, h) * 1.3 / cs
    tex = tex.resize((int(cs * scale), int(cs * scale)), Image.LANCZOS)
    tex = tex.rotate(float(rng.uniform(0, 360)), resample=Image.BILINEAR, expand=False)
    ctx, cty = tex.size
    x0 = (ctx - w) // 2; y0 = (cty - h) // 2
    tex = tex.crop((x0, y0, x0 + w, y0 + h))
    tex_arr = np.array(tex).astype(np.float32) / 255.0
    # High threshold + low gain → only the brightest scratches
    tex_arr = np.clip((tex_arr - 0.65) * 1.6, 0, 1)
    alpha = tex_arr * strength * float(rng.uniform(0.11, 0.19))
    # Mask out subject
    if subject_mask is not None:
        alpha = alpha * (1.0 - subject_mask)
    scratch_rgb = np.array([238, 232, 218], dtype=np.float32)
    arr = img_arr.astype(np.float32)
    arr = arr * (1 - alpha[..., None]) + scratch_rgb * alpha[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def get_subject_mask(img_pil):
    """Return HxW float mask [0,1] of the subject, or None on failure."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from masking import build_mask
        mask_pil, info = build_mask(img_pil, affect="subject", output_dir=None, feather=0.0)
        arr = (np.array(mask_pil) > 127).astype(np.float32)
        return arr
    except Exception as e:
        log("WARN", f"subject mask fetch failed: {e}")
        return None


def complementary_rgb(rgb, shift_deg=150):
    """Hue-shift an RGB triplet by N degrees (default 150 for 'near-complement')."""
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    hsv[0] = (hsv[0] + (shift_deg / 2.0)) % 180  # OpenCV H is 0-179
    hsv = hsv.astype(np.uint8)[None, None]
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(v) for v in out)


def add_color_strip(img_arr, comp_rgb, seed, subject_mask=None):
    """Near-vertical complementary-hue strip, partial length, sharp falloff.

    - angle: 80-100° from horizontal (±10° from vertical)
    - length: 20-80% of image height, starting anywhere
    - width: 5-15% of image width
    - falloff: power curve (1 - d^2.5) → ~65% plateau + 35% fade
    - avoids subject: places strip x-center outside subject x-bbox if possible
    """
    h, w = img_arr.shape[:2]
    rng = np.random.default_rng(seed + 13)

    # Pick x-center that avoids the subject
    x_center_frac = None
    if subject_mask is not None and subject_mask.sum() > 100:
        cols_with_subject = subject_mask.sum(axis=0) > (h * 0.03)
        xs = np.where(cols_with_subject)[0]
        if xs.size > 0:
            s_lo, s_hi = int(xs.min()), int(xs.max())
            # Candidate zones: left of subject, right of subject
            margin = int(w * 0.08)
            zones = []
            if s_lo - margin > int(w * 0.05):
                zones.append((int(w * 0.05), s_lo - margin))
            if s_hi + margin < int(w * 0.95):
                zones.append((s_hi + margin, int(w * 0.95)))
            if zones:
                lo, hi = zones[int(rng.integers(0, len(zones)))]
                x_center_frac = rng.uniform(lo / w, hi / w)
    if x_center_frac is None:
        x_center_frac = rng.uniform(0.05, 0.95)
    # Build strip on an oversized canvas then rotate+translate+crop
    max_side = int(np.sqrt(w * w + h * h)) + 40
    canvas = np.zeros((max_side, max_side), dtype=np.float32)

    strip_w = int(w * rng.uniform(0.05, 0.15))
    length_frac = rng.uniform(0.20, 0.80)
    strip_len = int(h * length_frac)
    # Anchor to top or bottom edge of final image (not floating)
    canvas_image_top = (max_side - h) // 2
    from_top = rng.random() < 0.5
    if from_top:
        y_start = canvas_image_top  # strip begins at y=0 in final image
    else:
        y_start = canvas_image_top + (h - strip_len)  # strip ends at y=h
    x_offset_in_final = int(x_center_frac * w)
    x_center = (max_side - w) // 2 + x_offset_in_final

    # Horizontal falloff: slightly softer (2.0 exponent, was 2.5)
    half = strip_w / 2.0
    for dx in range(-strip_w // 2, strip_w // 2 + 1):
        d_norm = abs(dx) / max(1.0, half)
        a = max(0.0, 1.0 - d_norm ** 2.0)
        col = x_center + dx
        if 0 <= col < max_side:
            canvas[y_start:y_start + strip_len, col] = a

    # Vertical fade: only on the *inner* end (edge-anchored end is hard)
    fade_end = max(1, int(strip_len * 0.18))  # slightly longer fade
    for i in range(fade_end):
        f = i / fade_end
        if from_top:
            # hard at top, fade at bottom
            canvas[y_start + strip_len - 1 - i, :] *= f
        else:
            # fade at top, hard at bottom
            canvas[y_start + i, :] *= f

    # Rotate canvas by 80-100° from horizontal = (rotation from vertical axis) ±10°
    angle_from_vertical = rng.uniform(-10.0, 10.0)
    canvas_img = Image.fromarray((canvas * 255).astype(np.uint8), "L")
    rotated = canvas_img.rotate(angle_from_vertical, resample=Image.BILINEAR, expand=False)
    alpha_big = np.array(rotated).astype(np.float32) / 255.0
    # Center-crop to image dims
    y0 = (max_side - h) // 2; x0 = (max_side - w) // 2
    alpha_map = alpha_big[y0:y0 + h, x0:x0 + w]
    alpha_map *= float(rng.uniform(0.22, 0.45))  # peak alpha (slightly less visible)

    # Defense-in-depth: zero alpha where subject sits, in case rotation clipped into them
    if subject_mask is not None:
        alpha_map = alpha_map * (1.0 - subject_mask)

    comp = np.array(comp_rgb, dtype=np.float32)
    result = img_arr.astype(np.float32) * (1 - alpha_map[..., None]) + comp * alpha_map[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def add_light_leak(img_arr, intensity, seed):
    """Radial gradient warm/cold tint from one corner."""
    h, w = img_arr.shape[:2]
    rng = np.random.default_rng(seed + 21)
    corner = rng.integers(0, 4)
    cx = 0 if corner in (0, 2) else w
    cy = 0 if corner in (0, 1) else h
    warm = rng.random() < 0.6
    color = np.array([255, 195, 120] if warm else [100, 160, 220], dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_d = np.sqrt(w * w + h * h) * rng.uniform(0.45, 0.65)
    alpha = np.clip(1.0 - dist / max_d, 0.0, 1.0) * intensity
    result = img_arr.astype(np.float32) * (1 - alpha[..., None]) + color * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


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

    # Analog imperfections (off by default; --analog enables all at defaults)
    apply_analog = args.analog or args.grain > 0 or args.scratches > 0 or args.color_strips > 0 or args.light_leak > 0
    if apply_analog:
        grain_s = args.grain if args.grain > 0 else (0.45 if args.analog else 0)
        scratch_s = args.scratches if args.scratches > 0 else (0.55 if args.analog else 0)
        strips_n = args.color_strips if args.color_strips > 0 else (1 if args.analog else 0)
        leak_s = args.light_leak if args.light_leak > 0 else (0.25 if args.analog else 0)

        # Fetch subject mask once, reused by scratches + strips to keep them off the model
        subj_mask = None
        if scratch_s > 0 or strips_n > 0:
            log("INFO", "Fetching subject mask (BiRefNet) to keep artifacts off the model...")
            subj_mask = get_subject_mask(img)

        if grain_s > 0:
            log("INFO", "Locating face for grain face-preserve radius...")
            face_center, face_r = face_center_and_radius(img)
            falloff = radial_falloff(w, h, face_center, face_r * args.face_preserve_mult)
            log("INFO", f"Adding gradient grain (strength={grain_s}, face=({face_center[0]:.0f},{face_center[1]:.0f}), r={face_r:.0f})")
            result_arr = add_grain(result_arr, falloff, grain_s, seed)
        if scratch_s > 0:
            log("INFO", f"Adding scratch overlay (strength={scratch_s}, subject-masked={subj_mask is not None})")
            result_arr = add_scratches(result_arr, scratch_s, seed, subject_mask=subj_mask)
        if strips_n > 0:
            comp = complementary_rgb(target, shift_deg=150)
            log("INFO", f"Adding {strips_n} complementary color strip(s) ({comp})")
            for i in range(strips_n):
                result_arr = add_color_strip(result_arr, comp, seed + i * 17, subject_mask=subj_mask)
        if leak_s > 0:
            log("INFO", f"Adding light leak (intensity={leak_s})")
            result_arr = add_light_leak(result_arr, leak_s, seed)

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
    # Analog imperfections
    p.add_argument("--analog", action="store_true", help="Enable all analog-photo imperfections at defaults")
    p.add_argument("--grain", type=float, default=0.0, help="Gradient grain strength 0-1 (face stays sharp, edges grainier)")
    p.add_argument("--face-preserve-mult", type=float, default=2.5, help="Grain preserve radius = face_radius × mult (default 2.5)")
    p.add_argument("--scratches", type=float, default=0.0, help="Scratch overlay opacity 0-1 (0=off, uses cached Flux-generated textures)")
    p.add_argument("--color-strips", type=int, default=0, help="Number of complementary-hue light leak strips (0=off)")
    p.add_argument("--light-leak", type=float, default=0.0, help="Corner light leak intensity 0-1 (0=off)")

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
