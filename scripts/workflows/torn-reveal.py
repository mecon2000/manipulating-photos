#!/home/rong/openclaw-venv/bin/python3
"""
Torn Reveal — Two-Layer Portrait Composite with Paper Tear

Layers two portraits of the same person:
  - Top layer: Color photo (the "public mask")
  - Bottom layer: High-contrast B&W (the raw emotional truth)
  - Connection: A paper tear across the eye area reveals B&W eyes beneath

Uses MediaPipe face mesh to align eyes between photos, generates realistic torn-paper
edges with fiber zones and drop shadows, applies film grain.

For best results, use two photos of the same person from the same shoot (adjacent file
numbers like BLD_5004.jpg and BLD_5039.jpg). The top photo should be color, the bottom
will be auto-converted to high-contrast B&W.

Pure PIL/numpy/scipy/cv2/mediapipe — no API calls needed.

Usage:
    ./torn-reveal.py --top color.jpg --bottom bw.jpg
    ./torn-reveal.py --top photo.jpg --bottom photo.jpg --bw-contrast 1.8
    ./torn-reveal.py --top photo.jpg --bottom photo.jpg --tear-height 0.12 --tear-jitter 0.7
    ./torn-reveal.py --top photo.jpg --bottom photo.jpg --tear-angle 15
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

import gc
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


def _segment_face_skin(img_rgb):
    """Run MediaPipe selfie segmentation and return face-skin boolean mask.

    Uses the multiclass selfie segmenter directly (no importlib).
    Category 3 = face-skin in MediaPipe's multiclass model.
    Returns a boolean numpy array, or None if model not found.
    """
    model_path = os.path.expanduser("~/openclaw-venv/mediapipe_models/selfie_multiclass.tflite")
    if not os.path.exists(model_path):
        return None

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    base = mp.tasks.BaseOptions(model_asset_path=model_path)
    opts = mp.tasks.vision.ImageSegmenterOptions(
        base_options=base,
        output_category_mask=True,
    )
    segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(opts)
    result = segmenter.segment(mp_img)
    # CRITICAL: .copy() before closing — numpy_view() is a view into native memory
    cat_mask = result.category_mask.numpy_view().squeeze().copy()
    segmenter.close()
    return cat_mask == 3  # face-skin


def _detect_eyes_via_body_segment(img_pil, output_dir, label=""):
    """Fallback eye detection using MediaPipe body segmentation.

    Uses the face-skin category (index 3) to find the face region,
    then estimates eye positions from its centroid.
    Returns ((lx, ly), (rx, ry)) or None.
    """
    try:
        img_np = np.array(img_pil)
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_np

        face_mask = _segment_face_skin(img_rgb)
        if face_mask is None:
            log(output_dir, "Selfie segmenter model not found", "WARN")
            return None

        if not np.any(face_mask):
            log(output_dir, f"Body segment fallback [{label}]: no face-skin pixels found", "WARN")
            return None

        h, w = img_np.shape[:2]
        ys, xs = np.where(face_mask)
        # Face centroid
        cx = np.mean(xs)
        cy = np.mean(ys)

        # Face bounding box for estimating eye separation
        face_left, face_right = np.min(xs), np.max(xs)
        face_top, face_bottom = np.min(ys), np.max(ys)
        face_w = face_right - face_left
        face_h = face_bottom - face_top

        # Eyes are typically in the upper 40% of the face region
        eye_y = face_top + face_h * 0.35

        # Eye separation is roughly 35% of face width
        eye_sep = face_w * 0.35
        lx = cx - eye_sep / 2
        rx = cx + eye_sep / 2

        log(output_dir, f"Eyes estimated [{label}] via body-segment face-skin: "
                         f"L=({lx:.0f},{eye_y:.0f}) R=({rx:.0f},{eye_y:.0f}) "
                         f"(face bbox: {face_w:.0f}x{face_h:.0f}px)")
        return ((lx, eye_y), (rx, eye_y))

    except Exception as e:
        log(output_dir, f"Body segment fallback failed [{label}]: {e}", "WARN")
        return None


def detect_eyes(img_pil, output_dir, label=""):
    """Detect left and right eye centers using MediaPipe face mesh.

    Falls back to body segmentation (face-skin centroid) if face mesh fails.
    Returns (((lx, ly), (rx, ry)), method) where method is 'iris'/'contour'/'segment'/'none'.
    Returns (None, 'none') if all methods fail.
    """
    img_np = np.array(img_pil)
    # MediaPipe expects RGB
    if img_np.ndim == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_np

    h, w = img_np.shape[:2]
    FACE_MODEL = os.path.expanduser("~/openclaw-venv/mediapipe_models/face_landmarker.task")

    # --- Primary: FaceLandmarker ---
    if os.path.exists(FACE_MODEL):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        base_opts = mp.tasks.BaseOptions(model_asset_path=FACE_MODEL)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
        )
        detector = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        result = detector.detect(mp_img)
        detector.close()

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Try iris centers first (landmarks 468=left iris, 473=right iris)
            try:
                left_iris = landmarks[LEFT_IRIS_CENTER]
                right_iris = landmarks[RIGHT_IRIS_CENTER]
                lx, ly = left_iris.x * w, left_iris.y * h
                rx, ry = right_iris.x * w, right_iris.y * h
                log(output_dir, f"Eyes detected [{label}] via iris landmarks: L=({lx:.0f},{ly:.0f}) R=({rx:.0f},{ry:.0f})")
                return ((lx, ly), (rx, ry)), "iris"
            except (IndexError, AttributeError):
                pass

            # Fallback to eye contour averages
            try:
                lx = np.mean([landmarks[i].x for i in LEFT_EYE_CONTOUR]) * w
                ly = np.mean([landmarks[i].y for i in LEFT_EYE_CONTOUR]) * h
                rx = np.mean([landmarks[i].x for i in RIGHT_EYE_CONTOUR]) * w
                ry = np.mean([landmarks[i].y for i in RIGHT_EYE_CONTOUR]) * h
                log(output_dir, f"Eyes detected [{label}] via contour avg: L=({lx:.0f},{ly:.0f}) R=({rx:.0f},{ry:.0f})")
                return ((lx, ly), (rx, ry)), "contour"
            except (IndexError, AttributeError):
                log(output_dir, f"Eye contour extraction failed for {label}", "WARN")
        else:
            log(output_dir, f"FaceLandmarker found no faces for {label} — trying body-segment fallback", "WARN")
            # Force cleanup of native MediaPipe resources before loading another model
            del detector, result, mp_img
            gc.collect()
    else:
        log(output_dir, f"Face landmarker model not found: {FACE_MODEL} — trying body-segment fallback", "WARN")

    # --- Fallback: body segmentation face-skin ---
    result_bs = _detect_eyes_via_body_segment(img_pil, output_dir, label)
    if result_bs is not None:
        return result_bs, "segment"

    log(output_dir, f"All eye detection methods failed for {label}", "WARN")
    return None, "none"


# ---------------------------------------------------------------------------
# Eye Alignment — Affine Warp
# ---------------------------------------------------------------------------
def align_eyes(img_bottom, eyes_top, eyes_bottom, output_dir, translation_only=False):
    """Warp img_bottom so its eyes align with eyes_top positions.

    Uses affine transform: translation + rotation + uniform scale.
    If translation_only=True, only shifts the midpoint (no rotation/scale) —
    safer when one eye detection used an imprecise fallback method.
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

    if translation_only:
        # Only shift midpoints — no rotation or scale
        tx = top_mid[0] - bot_mid[0]
        ty = top_mid[1] - bot_mid[1]
        log(output_dir, f"Translation-only alignment: shift=({tx:.0f},{ty:.0f})")
        M = np.array([[1, 0, tx],
                       [0, 1, ty]], dtype=np.float64)
        img_np = np.array(img_bottom)
        h, w = img_np.shape[:2]
        warped = cv2.warpAffine(img_np, M, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(warped)

    scale = top_dist / bot_dist
    top_angle = math.atan2(top_dy, top_dx)
    bot_angle = math.atan2(bot_dy, bot_dx)
    rotation = top_angle - bot_angle

    # Sanity caps: if alignment is too extreme, the eye detection was probably wrong
    MAX_ROTATION_DEG = 15.0
    MAX_SCALE_DEVIATION = 0.4  # allow 0.6x to 1.4x
    rot_deg = math.degrees(rotation)
    if abs(rot_deg) > MAX_ROTATION_DEG:
        log(output_dir, f"Alignment rotation {rot_deg:.1f}deg exceeds ±{MAX_ROTATION_DEG}deg — "
                         f"clamping (eye detection likely inaccurate)", "WARN")
        rotation = math.radians(max(-MAX_ROTATION_DEG, min(MAX_ROTATION_DEG, rot_deg)))
    if abs(scale - 1.0) > MAX_SCALE_DEVIATION:
        log(output_dir, f"Alignment scale {scale:.3f} too extreme — clamping to ±{MAX_SCALE_DEVIATION}", "WARN")
        scale = max(1.0 - MAX_SCALE_DEVIATION, min(1.0 + MAX_SCALE_DEVIATION, scale))

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
def _midpoint_displace(points, rng, amplitude, levels):
    """Fractal midpoint displacement on a list of (x, y) control points.

    At each level, insert a midpoint between every adjacent pair, displaced
    vertically by a random amount within [-amplitude, +amplitude].
    Amplitude halves each level (like fractal terrain generation).
    """
    for _ in range(levels):
        new_pts = [points[0]]
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2 + rng.uniform(-amplitude, amplitude)
            new_pts.append((mx, my))
            new_pts.append((x1, y1))
        points = new_pts
        amplitude *= 0.5  # halve displacement range each recursion
    return points


def generate_tear_edge(width, base_y_array, jitter_amplitude, seed, num_low_freq=4, num_high_freq=15):
    """Generate a jagged tear edge as an array of y-values across image width.

    base_y_array: array of per-column base y values (supports angled tears).
    Uses fractal midpoint displacement for natural paper-tear jaggedness:
    sharp, irregular direction changes instead of smooth sine waves.
    """
    rng = np.random.RandomState(seed)
    y = base_y_array.copy().astype(np.float64)

    # --- Fractal midpoint displacement ---
    # Start with ~20 anchor points along the edge, sampled from base_y_array
    num_anchors = 20
    anchor_xs = np.linspace(0, width - 1, num_anchors).astype(int)
    anchor_pts = [(int(ax), float(y[ax])) for ax in anchor_xs]

    # 9 levels of recursive subdivision gives ~10k points from 20 anchors
    fractal_pts = _midpoint_displace(anchor_pts, rng, jitter_amplitude, levels=9)

    # Interpolate fractal points back to per-pixel y values
    fx = np.array([p[0] for p in fractal_pts])
    fy = np.array([p[1] for p in fractal_pts])
    # Sort by x (should already be sorted, but ensure it)
    sort_idx = np.argsort(fx)
    fx, fy = fx[sort_idx], fy[sort_idx]
    # Linear interpolation to full width
    y = np.interp(np.arange(width, dtype=np.float64), fx, fy)

    # --- Sharp V-notches at random positions (5-10 per edge) ---
    num_notches = rng.randint(5, 11)
    for _ in range(num_notches):
        cx = rng.randint(0, width)
        # Notch width: 0.5-2% of width — narrow and sharp
        notch_w = rng.randint(max(1, width // 200), max(2, width // 50))
        notch_h = rng.uniform(0.5, 1.5) * jitter_amplitude
        direction = rng.choice([-1, 1])
        left = max(0, cx - notch_w // 2)
        right = min(width, cx + notch_w // 2)
        # Triangular (sharp V) notch — linear sides meeting at a point
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
def draw_fiber_zone(img_np, edge_y, seed, side="top", short_edge=1000, tear_angle_rad=0.0):
    """Draw a zone of white fibrous texture along a torn edge.

    Realistic torn paper fiber zone with:
    - Mostly thin base width (~0.8% of short edge)
    - 2-3 random "burst" peaks where fibers splay wide (3-5% of short edge)
    - Clean break segments with nearly zero fibers between bursts
    - Taper/fade on the outer side (away from tear gap)
    - More chaotic texture within burst zones

    side: "top" means fiber zone extends INTO the gap (downward from top edge).
          "bottom" means fiber zone extends INTO the gap (upward from bottom edge).
    """
    rng = np.random.RandomState(seed)
    h, w = img_np.shape[:2]

    # Base fiber zone width: ~0.8% of short edge (thin default)
    base_width = max(2, int(short_edge * 0.008))
    # Burst width: 3-5% of short edge
    burst_width = max(6, int(short_edge * rng.uniform(0.03, 0.05)))

    # --- Build width profile: base + gaussian burst peaks ---
    xs = np.linspace(0, 1, w)
    width_profile = np.ones(w, dtype=np.float64) * base_width

    # 2-3 burst peaks at random positions
    num_bursts = rng.randint(2, 4)
    for _ in range(num_bursts):
        peak_x = rng.uniform(0.05, 0.95)  # avoid extreme edges
        # Each burst spans 5-15% of image width
        sigma = rng.uniform(0.05, 0.15) / 2.35  # FWHM -> sigma
        bump = (burst_width - base_width) * np.exp(-0.5 * ((xs - peak_x) / sigma) ** 2)
        width_profile += bump

    fiber_widths = np.clip(width_profile, 0, burst_width * 1.2).astype(np.int32)

    # --- Clean break segments (nearly zero fibers) ---
    # Between bursts, some segments should have almost no fibers
    gap_mask = np.ones(w, dtype=np.float64)  # 1.0 = full fibers, 0.0 = clean break
    num_clean = rng.randint(6, 15)
    for _ in range(num_clean):
        gap_cx = rng.uniform(0, 1)
        gap_w = rng.uniform(0.005, 0.025)  # 0.5-2.5% of width
        # Only suppress fibers where width is near base (don't cut through bursts)
        left_x = max(0, gap_cx - gap_w / 2)
        right_x = min(1, gap_cx + gap_w / 2)
        left_i = int(left_x * w)
        right_i = int(right_x * w)
        for xi in range(left_i, right_i):
            if xi < w and fiber_widths[xi] < base_width * 1.5:
                gap_mask[xi] = 0.0

    # Perpendicular direction for fiber texture: perpendicular to tear angle
    perp_angle = tear_angle_rad + math.pi / 2

    # Create the fiber zone overlay (RGBA)
    fiber_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for x in range(w):
        if gap_mask[x] < 0.01:
            continue

        ey = int(np.clip(edge_y[x], 0, h - 1))
        fw = fiber_widths[x]
        is_burst = fw > base_width * 2  # inside a burst zone

        for dy in range(fw):
            if side == "top":
                y = ey + dy  # extend downward into gap
            else:
                y = ey - dy  # extend upward into gap

            if y < 0 or y >= h:
                continue

            # Taper: outer 40% of zone fades out
            inner_frac = dy / max(1, fw)
            if inner_frac > 0.6:
                alpha_taper = 1.0 - (inner_frac - 0.6) / 0.4
            else:
                alpha_taper = 1.0

            # Base alpha: 0.7-0.95 mostly opaque
            base_alpha = rng.uniform(0.7, 0.95)

            # Fine fiber texture: random brightness variation per pixel
            if is_burst:
                # More chaotic in burst zones: wider brightness range, more gaps
                brightness = rng.uniform(0.65, 1.0)
                if rng.random() < 0.15:
                    brightness *= 0.2
                    base_alpha *= 0.3
            else:
                brightness = rng.uniform(0.85, 1.0)
                # Occasional thin dark streaks perpendicular to tear (fiber gaps)
                if rng.random() < 0.08:
                    brightness *= 0.3
                    base_alpha *= 0.5

            final_alpha = int(np.clip(base_alpha * alpha_taper * gap_mask[x] * 255, 0, 255))
            pixel_val = int(np.clip(brightness * 255, 180, 255))

            fiber_layer[y, x] = [pixel_val, pixel_val, pixel_val, final_alpha]

    # Composite fiber layer onto image
    alpha = fiber_layer[:, :, 3:4].astype(np.float64) / 255.0
    rgb = fiber_layer[:, :, :3].astype(np.float64)
    img_float = img_np.astype(np.float64)
    result = img_float * (1.0 - alpha) + rgb * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Drop Shadow
# ---------------------------------------------------------------------------
def add_tear_shadow(composite_np, tear_mask, top_edge_y, shadow_height_frac, short_edge):
    """Add drop shadow cast BY the top layer ONTO the revealed bottom layer in the tear gap.

    The shadow sits just below the top torn edge, on the B&W image visible in the gap.
    This simulates the top sheet being slightly lifted, casting a shadow downward.
    Shadow is only applied within the tear gap (where the bottom layer is visible).
    """
    h, w = composite_np.shape[:2]
    shadow_h = max(3, int(short_edge * shadow_height_frac))

    shadow_layer = np.zeros((h, w), dtype=np.float64)

    for x in range(w):
        # Shadow falls below the top edge, INTO the gap
        y_top = int(np.clip(top_edge_y[x], 0, h - 1))
        for dy in range(shadow_h):
            y = y_top + dy
            if 0 <= y < h:
                # Only apply shadow within the tear gap
                if tear_mask[y, x] > 0:
                    # Quadratic falloff: stronger near edge, fades quickly
                    t = dy / shadow_h
                    opacity = 0.5 * (1.0 - t) ** 2
                    shadow_layer[y, x] = max(shadow_layer[y, x], opacity)

    # Slight horizontal blur to soften shadow edge
    shadow_layer = gaussian_filter1d(shadow_layer, sigma=max(1, int(short_edge * 0.002)), axis=1)

    # Apply shadow: darken only where shadow exists
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
    eyes_top, method_top = detect_eyes(top_img, output_dir, label="top")
    eyes_bottom, method_bottom = detect_eyes(bottom_img, output_dir, label="bottom")

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
        # If one used precise detection and the other used segment fallback,
        # only do translation (shift midpoints), skip rotation/scale — too inaccurate
        precise_methods = {"iris", "contour"}
        both_precise = method_top in precise_methods and method_bottom in precise_methods
        if not both_precise:
            log(output_dir, f"Mixed detection methods (top={method_top}, bottom={method_bottom}) "
                             f"— using translation-only alignment (no rotation/scale)")
        bottom_aligned = align_eyes(bottom_img, eyes_top, eyes_bottom, output_dir,
                                     translation_only=not both_precise)
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
    jitter_amp = args.tear_jitter * h * 0.025  # max displacement (~2.5% at jitter=1.0)

    # Tear angle: random ±20 degrees from horizontal if "auto"
    rng_angle = random.Random(seed)
    if args.tear_angle == "auto":
        tear_angle_deg = rng_angle.uniform(-20, 20)
    else:
        tear_angle_deg = float(args.tear_angle)
    tear_angle_rad = math.radians(tear_angle_deg)

    # Cone-shaped tear: varying width across image
    # Scale cone dimensions by --tear-height (default 0.10 maps to ~12-15% wide, ~3-5% narrow)
    scale = args.tear_height / 0.10  # 1.0 at default
    tear_h_max = h * rng_angle.uniform(0.12, 0.15) * scale
    tear_h_min = h * rng_angle.uniform(0.03, 0.05) * scale
    # Randomly pick which side is wider
    cone_left_wide = rng_angle.random() < 0.5

    # Compute per-column tear height and angled base y
    tear_center_y = eye_center_y
    x_arr = np.arange(w, dtype=np.float64)

    # Angled center line: base_y(x) = center_y + (x - w/2) * tan(angle)
    center_y_arr = tear_center_y + (x_arr - w / 2) * math.tan(tear_angle_rad)

    # Varying tear height (cone shape)
    t = x_arr / max(1, w - 1)  # 0 to 1 across width
    if cone_left_wide:
        tear_h_arr = tear_h_max + (tear_h_min - tear_h_max) * t  # wide left, narrow right
    else:
        tear_h_arr = tear_h_min + (tear_h_max - tear_h_min) * t  # narrow left, wide right

    top_base_y_arr = center_y_arr - tear_h_arr / 2
    bot_base_y_arr = center_y_arr + tear_h_arr / 2

    log(output_dir, f"Tear: center_y={tear_center_y:.0f}, angle={tear_angle_deg:.1f}deg, "
                     f"cone={'L-wide' if cone_left_wide else 'R-wide'}, "
                     f"h_range={tear_h_min:.0f}-{tear_h_max:.0f}px, jitter_amp={jitter_amp:.1f}px")

    top_edge_y = generate_tear_edge(w, top_base_y_arr, jitter_amp, seed=seed,
                                     num_low_freq=4, num_high_freq=15)
    bottom_edge_y = generate_tear_edge(w, bot_base_y_arr, jitter_amp, seed=seed + 1,
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
    log(output_dir, "--- Step 7: Drop shadow (top layer onto gap) ---")
    shadow_h_frac = 0.012  # ~1.2% of short edge for shadow depth
    composite_np = add_tear_shadow(composite_np, tear_mask, top_edge_y,
                                    shadow_h_frac, short_edge)

    # --- Paper fiber zones ---
    log(output_dir, "--- Step 8: Paper fiber zones ---")
    # Fiber zone on top edge (extending down into the gap)
    composite_np = draw_fiber_zone(composite_np, top_edge_y, seed=seed + 10,
                                    side="top", short_edge=short_edge,
                                    tear_angle_rad=tear_angle_rad)
    # Fiber zone on bottom edge (extending up into the gap)
    composite_np = draw_fiber_zone(composite_np, bottom_edge_y, seed=seed + 20,
                                    side="bottom", short_edge=short_edge,
                                    tear_angle_rad=tear_angle_rad)
    log(output_dir, f"Fiber zones drawn (base width ~{int(short_edge * 0.008)}px, burst up to ~{int(short_edge * 0.05)}px)")

    # --- Extra tear (censor/artistic second tear) ---
    if args.extra_tear_y is not None:
        log(output_dir, "--- Step 8b: Extra tear ---")
        extra_center_y = h * args.extra_tear_y
        extra_rng = random.Random(seed + 500)
        if args.extra_tear_angle is not None:
            extra_angle_deg = args.extra_tear_angle
        else:
            extra_angle_deg = extra_rng.uniform(-15, 15)
        extra_angle_rad = math.radians(extra_angle_deg)

        extra_scale = args.extra_tear_height / 0.10
        extra_h_max = h * extra_rng.uniform(0.10, 0.13) * extra_scale
        extra_h_min = h * extra_rng.uniform(0.03, 0.05) * extra_scale
        extra_left_wide = extra_rng.random() < 0.5

        extra_x_arr = np.arange(w, dtype=np.float64)
        extra_center_arr = extra_center_y + (extra_x_arr - w / 2) * math.tan(extra_angle_rad)
        extra_t = extra_x_arr / max(1, w - 1)
        if extra_left_wide:
            extra_h_arr = extra_h_max + (extra_h_min - extra_h_max) * extra_t
        else:
            extra_h_arr = extra_h_min + (extra_h_max - extra_h_min) * extra_t

        extra_top_base = extra_center_arr - extra_h_arr / 2
        extra_bot_base = extra_center_arr + extra_h_arr / 2
        extra_jitter = args.tear_jitter * h * 0.025

        extra_top_edge = generate_tear_edge(w, extra_top_base, extra_jitter, seed=seed + 600)
        extra_bot_edge = generate_tear_edge(w, extra_bot_base, extra_jitter, seed=seed + 601)
        for x in range(w):
            if extra_top_edge[x] >= extra_bot_edge[x]:
                mid = (extra_top_edge[x] + extra_bot_edge[x]) / 2
                extra_top_edge[x] = mid - 2
                extra_bot_edge[x] = mid + 2

        extra_mask = build_tear_mask(w, h, extra_top_edge, extra_bot_edge)

        # Build the fill layer for the extra tear
        if args.extra_tear_fill == "black":
            fill_layer = np.zeros_like(composite_np)
        elif args.extra_tear_fill == "bw":
            fill_layer = np.array(bottom_bw)
        else:  # "dark" — very dark, blurred version of the original
            dark_layer = np.array(top_img).astype(np.float64) * 0.08  # 8% brightness
            dark_layer = np.clip(dark_layer, 0, 255).astype(np.uint8)
            dark_pil = Image.fromarray(dark_layer)
            blur_r = max(10, int(short_edge * 0.05))
            dark_pil = dark_pil.filter(ImageFilter.GaussianBlur(radius=blur_r))
            fill_layer = np.array(dark_pil)

        extra_mask_3ch = np.stack([extra_mask] * 3, axis=-1) / 255.0
        composite_np = (composite_np * (1.0 - extra_mask_3ch) + fill_layer * extra_mask_3ch).astype(np.uint8)

        # Shadow + fibers for extra tear
        composite_np = add_tear_shadow(composite_np, extra_mask, extra_top_edge, shadow_h_frac, short_edge)
        composite_np = draw_fiber_zone(composite_np, extra_top_edge, seed=seed + 610,
                                        side="top", short_edge=short_edge, tear_angle_rad=extra_angle_rad)
        composite_np = draw_fiber_zone(composite_np, extra_bot_edge, seed=seed + 620,
                                        side="bottom", short_edge=short_edge, tear_angle_rad=extra_angle_rad)

        log(output_dir, f"Extra tear: y={args.extra_tear_y:.0%}, angle={extra_angle_deg:.1f}deg, "
                         f"fill={args.extra_tear_fill}, h_range={extra_h_min:.0f}-{extra_h_max:.0f}px")

    composite_img = Image.fromarray(composite_np).convert("RGB")
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
    parser.add_argument("--tear-angle", default="auto",
                        help="Tear angle in degrees from horizontal, or 'auto' for random ±20deg (default: auto)")
    parser.add_argument("--tear-jitter", type=float, default=0.5,
                        help="Jaggedness of tear edges, 0=smooth 1=very rough (default: 0.5)")
    parser.add_argument("--grain", type=float, default=0.04,
                        help="Film grain strength (default: 0.04)")
    parser.add_argument("--bw-contrast", type=float, default=1.5,
                        help="Contrast boost for B&W layer (default: 1.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--extra-tear-y", type=float, default=None,
                        help="Y position for extra tear as fraction of image height (e.g. 0.7 for lower body)")
    parser.add_argument("--extra-tear-angle", type=float, default=None,
                        help="Angle of extra tear in degrees (default: random ±15)")
    parser.add_argument("--extra-tear-height", type=float, default=0.08,
                        help="Height of extra tear as fraction of image (default: 0.08)")
    parser.add_argument("--extra-tear-fill", default="dark",
                        help="Fill for extra tear: 'dark' (very dark blur), 'bw' (B&W of same area), 'black' (default: dark)")
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
        "tear_angle": args.tear_angle,
        "tear_jitter": args.tear_jitter,
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
