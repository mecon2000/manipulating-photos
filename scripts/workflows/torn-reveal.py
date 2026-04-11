#!/home/rong/openclaw-venv/bin/python3
"""
Torn Reveal — Two-Layer Portrait Composite with Paper Tear

Layers two portraits of the same person:
  - Top layer: Color photo (the "public mask")
  - Bottom layer: High-contrast B&W (the raw emotional truth)
  - Connection: A horizontal paper tear across the eye area reveals B&W eyes beneath

Uses MediaPipe face mesh to align eyes between photos, generates realistic torn-paper
edges with fibers and drop shadows, applies film grain.

Pure PIL/numpy/scipy/cv2/mediapipe — no API calls needed.

Usage:
    ./torn-reveal.py --top color.jpg --bottom bw.jpg
    ./torn-reveal.py --top photo.jpg --bottom photo.jpg --bw-contrast 1.8
    ./torn-reveal.py --top photo.jpg --bottom photo.jpg --tear-height 0.12 --tear-jitter 0.7
"""

import os
import sys

# Auto-load env vars from ~/sol/.env if not already set
_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import json
import math
import random
import shutil
import argparse
import threading
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageStat
from scipy.ndimage import gaussian_filter1d
import mediapipe as mp

sys.stdout.reconfigure(line_buffering=True)

_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    with _log_lock:
        log_path = os.path.join(output_dir, "workflow.log")
        with open(log_path, "a") as f:
            f.write(formatted + "\n")


# ---------------------------------------------------------------------------
# Eye Detection via MediaPipe Face Mesh
# ---------------------------------------------------------------------------
# Iris center landmarks (refined landmarks must be enabled)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Fallback: average of eye contour landmarks
LEFT_EYE_CONTOUR = [33, 133, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [263, 362, 382, 381, 380, 374, 373, 386]


def detect_eyes(img_pil, output_dir, label=""):
    """Detect left and right eye centers using MediaPipe face mesh.

    Returns ((lx, ly), (rx, ry)) in pixel coords, or None if detection fails.
    """
    img_np = np.array(img_pil)
    # MediaPipe expects RGB
    if img_np.ndim == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_np

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    result = face_mesh.process(img_rgb)
    face_mesh.close()

    if not result.multi_face_landmarks:
        log(output_dir, f"Face mesh detection failed for {label} — no faces found", "WARN")
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    h, w = img_np.shape[:2]

    # Try iris centers first (refined landmarks)
    try:
        left_iris = landmarks[LEFT_IRIS_CENTER]
        right_iris = landmarks[RIGHT_IRIS_CENTER]
        lx, ly = left_iris.x * w, left_iris.y * h
        rx, ry = right_iris.x * w, right_iris.y * h
        log(output_dir, f"Eyes detected [{label}] via iris landmarks: L=({lx:.0f},{ly:.0f}) R=({rx:.0f},{ry:.0f})")
        return ((lx, ly), (rx, ry))
    except (IndexError, AttributeError):
        pass

    # Fallback to eye contour averages
    try:
        lx = np.mean([landmarks[i].x for i in LEFT_EYE_CONTOUR]) * w
        ly = np.mean([landmarks[i].y for i in LEFT_EYE_CONTOUR]) * h
        rx = np.mean([landmarks[i].x for i in RIGHT_EYE_CONTOUR]) * w
        ry = np.mean([landmarks[i].y for i in RIGHT_EYE_CONTOUR]) * h
        log(output_dir, f"Eyes detected [{label}] via contour avg: L=({lx:.0f},{ly:.0f}) R=({rx:.0f},{ry:.0f})")
        return ((lx, ly), (rx, ry))
    except (IndexError, AttributeError):
        log(output_dir, f"Eye contour detection also failed for {label}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Eye Alignment — Affine Warp
# ---------------------------------------------------------------------------
def align_eyes(img_bottom, eyes_top, eyes_bottom, output_dir):
    """Warp img_bottom so its eyes align with eyes_top positions.

    Uses affine transform: translation + rotation + uniform scale.
    Returns a PIL Image the same size as img_bottom, warped.
    """
    (tl_x, tl_y), (tr_x, tr_y) = eyes_top      # top image eye centers
    (bl_x, bl_y), (br_x, br_y) = eyes_bottom    # bottom image eye centers

    # Midpoint and angle for each pair
    top_mid = np.array([(tl_x + tr_x) / 2, (tl_y + tr_y) / 2])
    bot_mid = np.array([(bl_x + br_x) / 2, (bl_y + br_y) / 2])

    top_dx, top_dy = tr_x - tl_x, tr_y - tl_y
    bot_dx, bot_dy = br_x - bl_x, br_y - bl_y

    top_dist = math.hypot(top_dx, top_dy)
    bot_dist = math.hypot(bot_dx, bot_dy)

    if bot_dist < 1:
        log(output_dir, "Bottom eye distance too small for alignment — skipping warp", "WARN")
        return img_bottom

    scale = top_dist / bot_dist
    top_angle = math.atan2(top_dy, top_dx)
    bot_angle = math.atan2(bot_dy, bot_dx)
    rotation = top_angle - bot_angle

    log(output_dir, f"Alignment: scale={scale:.3f}, rotation={math.degrees(rotation):.1f}deg, "
                     f"shift=({top_mid[0]-bot_mid[0]:.0f},{top_mid[1]-bot_mid[1]:.0f})")

    # Build affine matrix: rotate+scale around bot_mid, then translate to top_mid
    cos_r = math.cos(rotation) * scale
    sin_r = math.sin(rotation) * scale

    # Affine: dest = M * src
    # We want: top_mid = M * bot_mid
    # M = [[cos_r, -sin_r, tx], [sin_r, cos_r, ty]]
    tx = top_mid[0] - cos_r * bot_mid[0] + sin_r * bot_mid[1]
    ty = top_mid[1] - sin_r * bot_mid[0] - cos_r * bot_mid[1]

    M = np.array([[cos_r, -sin_r, tx],
                   [sin_r,  cos_r, ty]], dtype=np.float64)

    img_np = np.array(img_bottom)
    h, w = img_np.shape[:2]
    warped = cv2.warpAffine(img_np, M, (w, h),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT_101)

    return Image.fromarray(warped)


# ---------------------------------------------------------------------------
# High-Contrast B&W Conversion
# ---------------------------------------------------------------------------
def make_high_contrast_bw(img_pil, contrast_boost=1.5, output_dir=None):
    """Convert to dramatic high-contrast B&W with S-curve.

    contrast_boost: multiplier for the S-curve steepness (1.0 = moderate, 2.0 = extreme).
    """
    gray = img_pil.convert("L")
    gray_np = np.array(gray, dtype=np.float64) / 255.0

    # S-curve: sigmoid centered at 0.5, steepness controlled by contrast_boost
    # Higher contrast_boost = steeper curve = more crushed shadows + blown highlights
    steepness = 5.0 + contrast_boost * 5.0  # range ~10-15 for typical values
    midpoint = 0.5
    curved = 1.0 / (1.0 + np.exp(-steepness * (gray_np - midpoint)))

    # Normalize to full range
    c_min, c_max = curved.min(), curved.max()
    if c_max > c_min:
        curved = (curved - c_min) / (c_max - c_min)

    result = np.clip(curved * 255, 0, 255).astype(np.uint8)
    bw_img = Image.fromarray(result, mode="L").convert("RGB")

    if output_dir:
        log(output_dir, f"B&W conversion: steepness={steepness:.1f}, "
                         f"input range=[{gray_np.min():.2f},{gray_np.max():.2f}]")

    return bw_img


# ---------------------------------------------------------------------------
# Tear Path Generation
# ---------------------------------------------------------------------------
def generate_tear_edge(width, base_y, jitter_amplitude, seed, num_low_freq=4, num_high_freq=15):
    """Generate a jagged tear edge as an array of y-values across image width.

    Combines low-frequency waves (overall tear shape) with high-frequency
    jaggedness (paper fiber randomness).
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, width)
    y = np.full(width, base_y, dtype=np.float64)

    # Low-frequency displacement (broad waves, 3-5 cycles)
    for _ in range(num_low_freq):
        freq = rng.uniform(2.0, 6.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0) * jitter_amplitude
        y += amp * np.sin(2 * np.pi * freq * x + phase)

    # High-frequency jaggedness (many small tears)
    for _ in range(num_high_freq):
        freq = rng.uniform(15.0, 50.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.3) * jitter_amplitude
        y += amp * np.sin(2 * np.pi * freq * x + phase)

    # Additional sharp notches at random positions
    num_notches = rng.randint(5, 15)
    for _ in range(num_notches):
        cx = rng.randint(0, width)
        notch_w = rng.randint(max(1, width // 200), max(2, width // 50))
        notch_h = rng.uniform(0.2, 0.8) * jitter_amplitude
        direction = rng.choice([-1, 1])
        left = max(0, cx - notch_w // 2)
        right = min(width, cx + notch_w // 2)
        # Triangular notch
        for xi in range(left, right):
            dist_from_center = abs(xi - cx) / max(1, notch_w // 2)
            y[xi] += direction * notch_h * (1.0 - dist_from_center)

    return y.astype(np.float64)


def build_tear_mask(width, height, top_edge_y, bottom_edge_y):
    """Build a mask where the tear gap is white (255) and everything else is black (0).

    This mask determines where the bottom layer shows through.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        y_top = int(np.clip(top_edge_y[x], 0, height - 1))
        y_bot = int(np.clip(bottom_edge_y[x], 0, height - 1))
        if y_bot > y_top:
            mask[y_top:y_bot, x] = 255
    return mask


# ---------------------------------------------------------------------------
# Paper Fibers
# ---------------------------------------------------------------------------
def draw_paper_fibers(img_pil, edge_y, fiber_count, seed, side="top", short_edge=1000):
    """Draw wispy white paper fibers along a torn edge.

    side: "top" means fibers hang DOWN from the top-layer's torn bottom edge.
          "bottom" means fibers poke UP from the bottom of the tear gap.
    """
    rng = np.random.RandomState(seed)
    draw = ImageDraw.Draw(img_pil, "RGBA")
    w, h = img_pil.size

    # Scale fiber dimensions to image size
    min_len = max(2, int(short_edge * 0.005))   # ~0.5% of short edge
    max_len = max(4, int(short_edge * 0.02))    # ~2% of short edge
    fiber_width = max(1, int(short_edge * 0.001))  # ~0.1% of short edge

    # Compute local jitter magnitude for density weighting
    edge_diff = np.abs(np.gradient(edge_y))
    edge_diff_smooth = gaussian_filter1d(edge_diff, sigma=max(1, w // 50))
    if edge_diff_smooth.max() > 0:
        density_weight = edge_diff_smooth / edge_diff_smooth.max()
    else:
        density_weight = np.ones(w)
    # Minimum density everywhere, boosted where jitter is high
    density_weight = 0.3 + 0.7 * density_weight

    # Generate fiber positions weighted by density
    positions = []
    for _ in range(fiber_count * 3):  # oversample, then filter
        x = rng.randint(0, w)
        if rng.random() < density_weight[x]:
            positions.append(x)
        if len(positions) >= fiber_count:
            break

    for x in positions:
        base_y = edge_y[x]
        length = rng.randint(min_len, max_len + 1)
        angle_offset = rng.uniform(-0.4, 0.4)  # slight angle variation (radians)
        alpha = rng.randint(75, 180)  # semi-transparent

        if side == "top":
            # Fibers hang down from top layer's torn edge
            direction = 1
        else:
            # Fibers poke up from bottom edge
            direction = -1

        end_x = x + int(length * math.sin(angle_offset))
        end_y = int(base_y) + direction * length

        # Clamp
        end_x = max(0, min(w - 1, end_x))
        end_y = max(0, min(h - 1, end_y))
        start_y = max(0, min(h - 1, int(base_y)))

        color = (255, 255, 255, alpha)
        draw.line([(x, start_y), (end_x, end_y)], fill=color, width=fiber_width)

    return img_pil


# ---------------------------------------------------------------------------
# Drop Shadow
# ---------------------------------------------------------------------------
def add_inner_shadow(composite_np, top_edge_y, bottom_edge_y, shadow_height_frac, short_edge):
    """Add inner drop shadow along tear edges on the top layer side.

    Shadow falls INTO the tear gap, suggesting the top paper curls slightly.
    """
    h, w = composite_np.shape[:2]
    shadow_h = max(3, int(short_edge * shadow_height_frac))

    shadow_layer = np.zeros((h, w), dtype=np.float64)

    for x in range(w):
        # Shadow below top edge (falls into gap from top layer)
        y_top = int(np.clip(top_edge_y[x], 0, h - 1))
        for dy in range(shadow_h):
            y = y_top + dy
            if 0 <= y < h:
                # Linear falloff from 0.4 opacity to 0
                opacity = 0.4 * (1.0 - dy / shadow_h)
                shadow_layer[y, x] = max(shadow_layer[y, x], opacity)

        # Shadow above bottom edge (falls into gap from bottom of tear)
        y_bot = int(np.clip(bottom_edge_y[x], 0, h - 1))
        for dy in range(shadow_h):
            y = y_bot - dy
            if 0 <= y < h:
                opacity = 0.25 * (1.0 - dy / shadow_h)  # slightly lighter
                shadow_layer[y, x] = max(shadow_layer[y, x], opacity)

    # Apply shadow: darken the composite
    shadow_3ch = np.stack([shadow_layer] * 3, axis=-1)
    result = composite_np.astype(np.float64) * (1.0 - shadow_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Film Grain
# ---------------------------------------------------------------------------
def apply_film_grain(img_pil, strength=0.04, seed=None):
    """Apply realistic film grain — stronger in shadows, weaker in highlights."""
    rng = np.random.RandomState(seed)
    img_np = np.array(img_pil, dtype=np.float64)

    # Base noise standard deviation in pixel values
    noise_std = strength * 255  # e.g. 0.04 * 255 ~= 10

    noise = rng.normal(0, noise_std, img_np.shape)

    # Shadow-weighted: more grain in dark areas
    # Luminance normalized to 0-1
    gray = np.mean(img_np, axis=2, keepdims=True) / 255.0
    # Weight: 1.5x in pure black, 0.5x in pure white
    weight = 1.5 - gray * 1.0
    noise *= weight

    result = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# Comparison Panel
# ---------------------------------------------------------------------------
def make_comparison(top_img, bottom_img, result_img, short_edge):
    """Create a 3-panel comparison: top | bottom | result."""
    w, h = top_img.size
    gap = max(2, int(short_edge * 0.005))
    panel_w = w
    total_w = panel_w * 3 + gap * 2
    canvas = Image.new("RGB", (total_w, h), (32, 32, 32))

    # Resize all to same dimensions
    canvas.paste(top_img.resize((panel_w, h), Image.LANCZOS), (0, 0))
    canvas.paste(bottom_img.resize((panel_w, h), Image.LANCZOS), (panel_w + gap, 0))
    canvas.paste(result_img.resize((panel_w, h), Image.LANCZOS), (2 * (panel_w + gap), 0))

    return canvas


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args, output_dir):
    log(output_dir, "=== Torn Reveal Pipeline ===")

    # --- Load images ---
    log(output_dir, "--- Step 1: Load images ---")
    top_img = Image.open(args.top).convert("RGB")
    log(output_dir, f"Top image: {top_img.size[0]}x{top_img.size[1]} — {args.top}")

    same_photo = os.path.abspath(args.top) == os.path.abspath(args.bottom)

    bottom_img = Image.open(args.bottom).convert("RGB")
    log(output_dir, f"Bottom image: {bottom_img.size[0]}x{bottom_img.size[1]} — {args.bottom}"
                     f"{' (same as top — will auto-convert to B&W)' if same_photo else ''}")

    # Resize bottom to match top if different
    if bottom_img.size != top_img.size:
        log(output_dir, f"Resizing bottom from {bottom_img.size} to {top_img.size}")
        bottom_img = bottom_img.resize(top_img.size, Image.LANCZOS)

    w, h = top_img.size
    short_edge = min(w, h)

    # Save originals
    top_img.save(os.path.join(output_dir, "01_top_original.jpg"), quality=95)

    # Seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    log(output_dir, f"Seed: {seed}")

    # --- Eye detection ---
    log(output_dir, "--- Step 2: Detect eyes ---")
    eyes_top = detect_eyes(top_img, output_dir, label="top")
    eyes_bottom = detect_eyes(bottom_img, output_dir, label="bottom")

    if eyes_top is None:
        log(output_dir, "FALLBACK: No eyes in top image — using center horizontal tear", "WARN")
        eye_center_y = h * 0.4  # slightly above center, typical eye position
    else:
        # Eye center Y is average of both eyes
        (tl_x, tl_y), (tr_x, tr_y) = eyes_top
        eye_center_y = (tl_y + tr_y) / 2

    # --- Align bottom to top ---
    log(output_dir, "--- Step 3: Align bottom image ---")
    if eyes_top is not None and eyes_bottom is not None:
        bottom_aligned = align_eyes(bottom_img, eyes_top, eyes_bottom, output_dir)
        log(output_dir, "Bottom image warped to align eyes with top")
    else:
        bottom_aligned = bottom_img
        if not same_photo:
            log(output_dir, "Skipping alignment — eye detection failed on one or both images", "WARN")
        else:
            log(output_dir, "Same photo — no alignment needed")

    # --- B&W conversion ---
    log(output_dir, "--- Step 4: High-contrast B&W ---")
    bottom_bw = make_high_contrast_bw(bottom_aligned, contrast_boost=args.bw_contrast,
                                       output_dir=output_dir)
    bottom_bw.save(os.path.join(output_dir, "02_bottom_bw.jpg"), quality=95)

    # --- Generate tear path ---
    log(output_dir, "--- Step 5: Generate tear path ---")
    tear_h = args.tear_height * h  # total tear height in pixels
    jitter_amp = args.tear_jitter * h * 0.025  # max displacement (~2.5% at jitter=1.0)

    tear_center_y = eye_center_y
    top_base_y = tear_center_y - tear_h / 2
    bot_base_y = tear_center_y + tear_h / 2

    log(output_dir, f"Tear: center_y={tear_center_y:.0f}, height={tear_h:.0f}px "
                     f"({args.tear_height:.0%} of image), jitter_amp={jitter_amp:.1f}px")

    top_edge_y = generate_tear_edge(w, top_base_y, jitter_amp, seed=seed,
                                     num_low_freq=4, num_high_freq=15)
    bottom_edge_y = generate_tear_edge(w, bot_base_y, jitter_amp, seed=seed + 1,
                                        num_low_freq=4, num_high_freq=15)

    # Ensure top edge is always above bottom edge
    for x in range(w):
        if top_edge_y[x] >= bottom_edge_y[x]:
            mid = (top_edge_y[x] + bottom_edge_y[x]) / 2
            top_edge_y[x] = mid - 2
            bottom_edge_y[x] = mid + 2

    # Build tear mask
    tear_mask = build_tear_mask(w, h, top_edge_y, bottom_edge_y)
    tear_mask_img = Image.fromarray(tear_mask, mode="L")
    tear_mask_img.save(os.path.join(output_dir, "03_tear_mask.png"))
    log(output_dir, f"Tear mask: {np.sum(tear_mask > 0)} pixels revealed "
                     f"({100 * np.mean(tear_mask > 0):.1f}% of image)")

    # --- Composite ---
    log(output_dir, "--- Step 6: Composite layers ---")
    # Start with bottom (B&W) as base, paste top layer where mask is black
    top_np = np.array(top_img)
    bottom_np = np.array(bottom_bw)
    mask_3ch = np.stack([tear_mask] * 3, axis=-1) / 255.0

    composite_np = (top_np * (1.0 - mask_3ch) + bottom_np * mask_3ch).astype(np.uint8)

    # --- Drop shadow ---
    log(output_dir, "--- Step 7: Inner drop shadow ---")
    shadow_h_frac = 0.008  # ~0.8% of short edge for shadow depth
    composite_np = add_inner_shadow(composite_np, top_edge_y, bottom_edge_y,
                                     shadow_h_frac, short_edge)

    composite_img = Image.fromarray(composite_np).convert("RGBA")

    # --- Paper fibers ---
    log(output_dir, "--- Step 8: Paper fibers ---")
    # Fibers on top edge (hanging down into the gap)
    composite_img = draw_paper_fibers(composite_img, top_edge_y,
                                       fiber_count=args.fiber_count, seed=seed + 10,
                                       side="top", short_edge=short_edge)
    # Fibers on bottom edge (poking up into the gap)
    composite_img = draw_paper_fibers(composite_img, bottom_edge_y,
                                       fiber_count=args.fiber_count, seed=seed + 20,
                                       side="bottom", short_edge=short_edge)
    log(output_dir, f"Drew {args.fiber_count} fibers per edge")

    # Convert back to RGB
    composite_img = composite_img.convert("RGB")
    composite_img.save(os.path.join(output_dir, "04_composite_pre_grain.jpg"), quality=95)

    # --- Film grain ---
    log(output_dir, "--- Step 9: Film grain ---")
    result = apply_film_grain(composite_img, strength=args.grain, seed=seed + 100)
    log(output_dir, f"Film grain applied: strength={args.grain}")

    result.save(os.path.join(output_dir, "05_final.jpg"), quality=95)

    # --- Comparison panel ---
    log(output_dir, "--- Step 10: Output ---")
    comparison = make_comparison(top_img, bottom_bw, result, short_edge)
    comparison.save(os.path.join(output_dir, "06_comparison.jpg"), quality=90)

    return result, comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Torn Reveal — Two-Layer Portrait Composite with Paper Tear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top", required=True, help="Path to top layer (color photo)")
    parser.add_argument("--bottom", required=True,
                        help="Path to bottom layer (B&W photo, or same photo for auto-convert)")
    parser.add_argument("--tear-height", type=float, default=0.10,
                        help="Height of tear as fraction of image (default: 0.10)")
    parser.add_argument("--tear-jitter", type=float, default=0.5,
                        help="Jaggedness of tear edges, 0=smooth 1=very rough (default: 0.5)")
    parser.add_argument("--fiber-count", type=int, default=150,
                        help="Number of paper fibers per torn edge (default: 150)")
    parser.add_argument("--grain", type=float, default=0.04,
                        help="Film grain strength (default: 0.04)")
    parser.add_argument("--bw-contrast", type=float, default=1.5,
                        help="Contrast boost for B&W layer (default: 1.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-to", default="local", choices=["local", "gdrive", "both"],
                        help="Output destination (default: local)")
    parser.add_argument("--local-output-dir", default=os.path.expanduser("~/.openclaw/workspace/shared"),
                        help="Local output directory (default: ~/.openclaw/workspace/shared)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.top):
        print(f"ERROR: Top image not found: {args.top}")
        sys.exit(1)
    if not os.path.isfile(args.bottom):
        print(f"ERROR: Bottom image not found: {args.bottom}")
        sys.exit(1)

    # Resolve names
    basename = os.path.basename(args.top)
    photo_name = os.path.splitext(basename)[0]
    model_name = ""
    match = re.search(r"(Original_|BLD_|Rong_IMG_)(.*?)_(.*)\.(jpg|png|jpeg|JPG)", basename, re.IGNORECASE)
    if match:
        model_name = match.group(2).replace(" ", "_")
        photo_name = match.group(3).replace(" ", "_")
    else:
        source_abs = os.path.abspath(args.top)
        parts = source_abs.split(os.sep)
        try:
            photos_idx = parts.index("_photos")
            if photos_idx + 1 < len(parts):
                model_name = parts[photos_idx + 1].replace(" ", "_")
        except ValueError:
            model_name = "Unknown"

    # Output directory
    israel_dt = datetime.now(ISRAEL_TZ)
    timestamp = israel_dt.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name}_{photo_name}_{timestamp}_torn_reveal_{random.randint(10, 99)}"
    if args.local_output_dir:
        output_dir = os.path.join(args.local_output_dir, folder_name)
    else:
        output_dir = os.path.join("outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save script copy for reproducibility
    try:
        with open(__file__, "r") as src:
            with open(os.path.join(output_dir, f"torn_reveal_script_{timestamp}.py"), "w") as dst:
                dst.write(src.read())
    except OSError:
        pass

    log(output_dir, f"Output directory: {output_dir}")
    log(output_dir, f"Args: {vars(args)}")

    # Run pipeline
    result, comparison = run_pipeline(args, output_dir)

    # Copy final to shared finals/ folder
    if args.local_output_dir:
        finals_dir = os.path.join(args.local_output_dir, "finals")
    else:
        finals_dir = os.path.join(output_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)

    finals_name = os.path.basename(output_dir) + ".jpg"
    finals_path = os.path.join(finals_dir, finals_name)
    result.save(finals_path, quality=95)
    log(output_dir, f"Final saved to: {finals_path}")

    # Save comparison to finals
    comparison_name = os.path.basename(output_dir) + "_comparison.jpg"
    comparison_path = os.path.join(finals_dir, comparison_name)
    comparison.save(comparison_path, quality=90)
    log(output_dir, f"Comparison saved to: {comparison_path}")

    # Save metadata
    metadata = {
        "tool": "torn-reveal",
        "top": os.path.abspath(args.top),
        "bottom": os.path.abspath(args.bottom),
        "tear_height": args.tear_height,
        "tear_jitter": args.tear_jitter,
        "fiber_count": args.fiber_count,
        "grain": args.grain,
        "bw_contrast": args.bw_contrast,
        "seed": args.seed,
        "timestamp": timestamp,
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Push to phone
    try:
        from notify import push_image
        src_name = os.path.splitext(os.path.basename(args.top))[0]
        push_image(finals_path, title=f"Torn Reveal — {src_name}",
                   body=f"tear={args.tear_height:.0%}, contrast={args.bw_contrast}")
        log(output_dir, "Pushed to phone")
    except Exception as e:
        log(output_dir, f"Push notification failed: {e}", "WARN")

    log(output_dir, "=== Torn Reveal complete ===")
    return finals_path


if __name__ == "__main__":
    main()
