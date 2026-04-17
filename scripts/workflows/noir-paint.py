#!/home/rong/openclaw-venv/bin/python3
"""
Noir Paint — Pulpbrother-style high-contrast painterly effect.

Takes a photo and transforms it into a bold, posterized gouache/acrylic painting
with dramatic directional lighting, 2-3 tone palette, and visible paint texture.

Pipeline:
  1. Extract subject (BiRefNet) → place on black BG
  2. Detect body axis (MediaPipe pose) → compute perpendicular light direction
  3. Relight with IC-Light (hard directional, single source, far outside frame)
  4. Posterize to 2-3 tones (histogram-aware thresholds)
  5. Remap to palette (cool blue-grey or warm skin tones)
  6. Edge roughening (displacement noise at tone boundaries)
  7. Paint texture pass (Tensor Art img2img, strength 0.15-0.20)
  8. Canvas texture overlay
  9. Evaluate + output

Inspired by @pulpbrother's high-contrast painted figure studies.

Usage:
    python noir-paint.py --source photo.jpg
    python noir-paint.py --source photo.jpg --tones warm --light-angle 45
    python noir-paint.py --source photo.jpg --num-tones 3 --tones cool
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

import re
import json
import math
import time
import uuid
import random
import base64
import argparse
import shutil
import threading
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import requests
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageDraw, ImageEnhance
from scipy.ndimage import sobel
import mediapipe as mp

# Use shared masking module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
PALETTES = {
    "cool": {
        "bg": (80, 95, 110),           # grey-blue background
        "black": (15, 18, 22),         # near-black with blue tint
        "shadow": (45, 55, 65),        # dark blue-grey
        "mid": (107, 125, 138),        # #6B7D8A
        "highlight": (175, 188, 198),  # blue-grey highlight
        "shadow_cast": (40, 50, 62),   # cast shadow tone
        "description": "Cool blue-grey monochrome (classic pulpbrother)",
    },
    "warm": {
        "bg": (75, 65, 55),
        "black": (15, 10, 8),
        "shadow": (70, 45, 35),
        "mid": (160, 120, 100),
        "highlight": (220, 195, 175),
        "shadow_cast": (50, 40, 32),
        "description": "Warm skin tones on dark",
    },
    "cold": {
        "bg": (60, 72, 90),
        "black": (10, 14, 20),
        "shadow": (30, 40, 55),
        "mid": (80, 100, 130),
        "highlight": (170, 185, 210),
        "shadow_cast": (35, 45, 60),
        "description": "Cold steel-blue tones",
    },
    "sepia": {
        "bg": (70, 60, 48),
        "black": (12, 10, 8),
        "shadow": (55, 40, 30),
        "mid": (130, 100, 70),
        "highlight": (210, 190, 160),
        "shadow_cast": (45, 35, 25),
        "description": "Sepia/vintage tone",
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()


def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    if output_dir:
        with _log_lock:
            log_path = os.path.join(output_dir, "workflow.log")
            with open(log_path, "a") as f:
                f.write(formatted + "\n")


# ---------------------------------------------------------------------------
# Helper: image to base64
# ---------------------------------------------------------------------------
def _img_to_b64(img, max_size=2048):
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_fal_key():
    return os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY", "")


# ---------------------------------------------------------------------------
# Step 1: Extract subject on black background
# ---------------------------------------------------------------------------
def extract_subject_on_black(source, output_dir):
    """Extract subject via BiRefNet, place on black background."""
    log(output_dir, "Extracting subject mask (BiRefNet)...")
    mask, mask_info = build_mask(source, affect="subject", exclude="", output_dir=output_dir)
    log(output_dir, f"Subject mask: {mask_info['coverage_pct']:.1f}%")

    img = Image.open(source).convert("RGB")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.LANCZOS)

    # Subject on black
    black = Image.new("RGB", img.size, (0, 0, 0))
    black.paste(img, mask=mask)

    return black, mask, img


# ---------------------------------------------------------------------------
# Step 2: Detect body axis → compute light direction
# ---------------------------------------------------------------------------
POSE_MODEL = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")

# MediaPipe pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def detect_body_axis(img_array, subject_mask, output_dir):
    """Detect body's major axis using MediaPipe pose landmarks.

    Returns light angle in degrees (0=right, 90=top, etc.) that is
    perpendicular to the body axis, biased toward the side with more
    empty space in the image.
    """
    h, w = img_array.shape[:2]

    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array.copy())
        base = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
        opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
        detector = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        result = detector.detect(mp_img)
        detector.close()
    except Exception as e:
        log(output_dir, f"Pose detection failed: {e}", "WARN")
        return None

    if not result.pose_landmarks:
        log(output_dir, "No pose detected — cannot determine body axis", "WARN")
        return None

    landmarks = result.pose_landmarks[0]

    # Get key points
    def lm_px(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    # Body axis: midpoint of shoulders → midpoint of hips
    ls, rs = lm_px(LEFT_SHOULDER), lm_px(RIGHT_SHOULDER)
    lh, rh = lm_px(LEFT_HIP), lm_px(RIGHT_HIP)

    shoulders_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hips_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    # Body axis vector (shoulders → hips)
    axis_dx = hips_mid[0] - shoulders_mid[0]
    axis_dy = hips_mid[1] - shoulders_mid[1]

    # Body axis angle
    axis_angle = math.degrees(math.atan2(-axis_dy, axis_dx))  # -dy because y increases downward

    # Perpendicular: two options (±90°)
    perp1 = axis_angle + 90
    perp2 = axis_angle - 90

    log(output_dir, f"Body axis: ({axis_dx:.0f}, {axis_dy:.0f}), angle={axis_angle:.0f}°")
    log(output_dir, f"Perpendicular options: {perp1:.0f}° and {perp2:.0f}°")

    # Choose the perpendicular that points toward more empty space
    # (so shadows fall into background, not into the frame center)
    mask_arr = np.array(subject_mask) > 127
    subject_cx = shoulders_mid[0]  # use shoulders as subject center x

    # Which side has more empty space?
    left_empty = 1.0 - mask_arr[:, :w // 2].mean()
    right_empty = 1.0 - mask_arr[:, w // 2:].mean()

    log(output_dir, f"Empty space: left={left_empty:.2f}, right={right_empty:.2f}")

    # Convert perp angles to unit vectors, see which points toward empty side
    def angle_to_vec(deg):
        rad = math.radians(deg)
        return (math.cos(rad), -math.sin(rad))  # -sin because y-down

    v1 = angle_to_vec(perp1)
    v2 = angle_to_vec(perp2)

    # Score: positive x = points right, negative x = points left
    # Prefer the one pointing toward the emptier side
    if right_empty > left_empty:
        # Want light from right (positive x)
        score1 = v1[0]
        score2 = v2[0]
    else:
        # Want light from left (negative x)
        score1 = -v1[0]
        score2 = -v2[0]

    light_angle = perp1 if score1 >= score2 else perp2

    # Normalize to 0-360
    light_angle = light_angle % 360

    log(output_dir, f"Chosen light direction: {light_angle:.0f}° (0=right, 90=up, 180=left, 270=down)")

    return light_angle


def angle_to_light_prompt(angle_deg):
    """Convert angle (0=right, 90=up, 180=left, 270=down) to IC-Light prompt."""
    # Normalize
    angle = angle_deg % 360

    # Map to clock positions and prompt descriptions
    if 337.5 <= angle or angle < 22.5:
        direction = "from the far right, far outside the right edge of frame"
    elif 22.5 <= angle < 67.5:
        direction = "from the upper right, far outside the upper right corner"
    elif 67.5 <= angle < 112.5:
        direction = "from directly above, far outside the top of frame"
    elif 112.5 <= angle < 157.5:
        direction = "from the upper left, far outside the upper left corner"
    elif 157.5 <= angle < 202.5:
        direction = "from the far left, far outside the left edge of frame"
    elif 202.5 <= angle < 247.5:
        direction = "from the lower left, far outside the lower left corner"
    elif 247.5 <= angle < 292.5:
        direction = "from directly below, far outside the bottom of frame"
    else:
        direction = "from the lower right, far outside the lower right corner"

    prompt = (
        f"single hard directional light {direction}, "
        f"harsh shadows like direct sunlight, no fill light, "
        f"stark contrast between light and shadow, "
        f"grid modifier for hard directional shadows, "
        f"professional studio portrait, monochromatic lighting"
    )
    negative = (
        "no halo, no corona, no glow, no lens flare, no bloom, "
        "no soft light, no ambient light, no fill light, no rim light, "
        "no colored gels, no warm tones"
    )
    return prompt, negative


# ---------------------------------------------------------------------------
# Step 3: Relight with IC-Light
# ---------------------------------------------------------------------------
def relight_subject(subject_on_black, light_prompt, light_negative, output_dir,
                    seed=None, highres_denoise=0.45, guidance_scale=2.5, num_steps=28):
    """Relight using IC-Light V2 with hard directional light."""
    log(output_dir, f"Relighting: '{light_prompt[:80]}...'")

    img_b64 = _img_to_b64(subject_on_black, max_size=1536)

    payload = {
        "prompt": light_prompt,
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "negative_prompt": light_negative,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "lowres_denoise": 0.9,  # High — we want strong re-lighting
        "highres_denoise": highres_denoise,
        "enable_hr_fix": True,
        "output_format": "jpeg",
        "num_images": 1,
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {
        "Authorization": f"Key {_get_fal_key()}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post("https://fal.run/fal-ai/iclight-v2",
                                 headers=headers, json=payload, timeout=600)
    except requests.RequestException as e:
        log(output_dir, f"IC-Light failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"IC-Light failed ({response.status_code}): {response.text[:200]}", "ERROR")
        return None

    data = response.json()
    images = data.get("images", [])
    if not images:
        log(output_dir, "IC-Light returned no images", "ERROR")
        return None

    result_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
    log(output_dir, f"Relit CDN: {result_url}")

    try:
        result_img = Image.open(BytesIO(requests.get(result_url, timeout=60).content)).convert("RGB")
        return result_img
    except Exception as e:
        log(output_dir, f"Failed to download relit image: {e}", "ERROR")
        return None


# ---------------------------------------------------------------------------
# Step 4: Posterize (histogram-aware)
# ---------------------------------------------------------------------------
def presmooth_bilateral(img, output_dir=None):
    """Edge-preserving smooth before posterization.

    Smooths gradients into flat regions while keeping real edges sharp.
    This prevents jittery/dithered boundaries when posterizing.
    """
    from scipy.ndimage import sobel
    gray = np.array(img.convert("L")).astype(np.float32)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=12))
    edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
    edges = np.clip(edges / max(edges.max(), 1), 0, 1)[:, :, np.newaxis]
    arr = np.array(blurred).astype(np.float32) * (1 - edges) + np.array(img).astype(np.float32) * edges
    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    log(output_dir, "Bilateral pre-smooth applied")
    return result


def cast_shadow(subject_mask, light_angle_deg, sun_elevation=20, output_dir=None):
    """Cast a floor-plane shadow from the subject silhouette.

    Projects the shadow as if cast onto a floor plane receding in Z (depth),
    not on the flat image plane. This gives perspective foreshortening —
    the shadow is elongated and narrow, stretching away from the subject's
    feet/contact point.

    light_angle_deg: direction light comes FROM (0=right, 90=up, 180=left)
    sun_elevation: degrees above horizon (lower = longer shadow)
    """
    W, H = subject_mask.size
    mask_arr = np.array(subject_mask) > 127
    if not mask_arr.any():
        return Image.new("L", (W, H), 0)

    # Find contact point (bottom of subject = feet/base)
    ys = np.where(mask_arr.any(axis=1))[0]
    floor_y = int(ys.max())

    # Shadow direction on floor (horizontal component of light)
    shadow_h_angle = (light_angle_deg + 180) % 360
    shadow_rad = math.radians(shadow_h_angle)
    sdx = math.cos(shadow_rad)  # horizontal shadow direction

    # Shadow length factor
    stretch = min(1.0 / math.tan(math.radians(max(5, sun_elevation))), 4.0)

    shadow = np.zeros((H, W), dtype=np.float32)

    # For each row of the subject above the floor, project it onto the floor
    # plane. Rows closer to the top of the subject cast shadow further away.
    # On the floor plane, "further away" = further below floor_y (receding in Z)
    # and shifted horizontally.
    for y in range(0, floor_y + 1):
        row = mask_arr[y, :]
        if not row.any():
            continue

        # Height above floor
        height_above = floor_y - y
        if height_above <= 0:
            continue

        # Shadow projects DOWN from floor (further below = further in Z)
        # Vertical offset on image: proportional to height × stretch
        # But with perspective foreshortening: compress as it gets further
        v_offset = int(height_above * stretch * 0.35)
        shadow_y = floor_y + v_offset

        if shadow_y >= H:
            continue

        # Horizontal offset: proportional to height
        h_offset = int(sdx * height_above * stretch * 0.25)

        # Place the row at shadow position
        xs = np.where(row)[0] + h_offset
        valid = (xs >= 0) & (xs < W)

        # Fade with distance (further = lighter shadow)
        fade = max(0.3, 1.0 - (v_offset / max(1, H - floor_y)) * 0.7)
        shadow[shadow_y, xs[valid]] = max(shadow[shadow_y, xs[valid]].max(), fade) if valid.any() else 0

    # Remove shadow under subject
    shadow[mask_arr] = 0

    # Fill gaps with morphological close
    sp = Image.fromarray((shadow * 255).astype(np.uint8), "L")
    sp = sp.filter(ImageFilter.MaxFilter(7))
    sp = sp.filter(ImageFilter.MinFilter(3))

    # Light blur (not too much — shadow should be fairly defined)
    blur_r = max(3, int(min(W, H) * 0.006))
    sp = sp.filter(ImageFilter.GaussianBlur(radius=blur_r))

    coverage = np.array(sp).mean() / 255.0 * 100
    log(output_dir, f"Shadow cast: light={light_angle_deg}°, elev={sun_elevation}°, "
        f"floor_y={floor_y}, stretch={stretch:.1f}x, blur={blur_r}px, "
        f"coverage={coverage:.1f}%")
    return sp


def detect_scene_context(source, output_dir=None):
    """Ask Gemini for scene context: surface shapes + subject count."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping scene context")
        return None
    try:
        img = Image.open(source).convert("RGB")
        r = 512 / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            'Photo for noir painting. Subject will be extracted onto a flat colored background. '
            'To prevent the subject from "floating", I need ONE simple shape representing '
            'the ground/surface they are on (floor, bed, sofa seat, table — just the surface plane). '
            'NOT objects or furniture around them — only the surface touching the subject. '
            'Use a simple trapezoid or rectangle. Also count the subjects. '
            'Return JSON with normalized coordinates (0.0-1.0, top-left origin): '
            '{"scene":"description","num_subjects":N,'
            '"shapes":[{"points":[[x,y],...],"tone":"darker","description":"..."}]}'
        )
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json={"contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": prompt},
            ]}], "generationConfig": {
                "temperature": 0.3, "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            }},
            timeout=60,
        )
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(text)
            log(output_dir, f"Scene context: {result.get('scene', '?')}, "
                f"{result.get('num_subjects', '?')} subjects, "
                f"{len(result.get('shapes', []))} shapes")
            return result
    except Exception as e:
        log(output_dir, f"Scene context failed: {e}", "WARN")
    return None


def posterize_histogram(img, num_tones=2, output_dir=None):
    """Smart posterization using luminance histogram to find natural breaks.

    Instead of even splits, finds the thresholds that best separate the
    image's natural shadow/mid/highlight zones.
    """
    gray = np.array(img.convert("L")).astype(np.float32)

    # Build cumulative histogram
    hist, _ = np.histogram(gray[gray > 5], bins=256, range=(0, 256))  # ignore pure black
    cumsum = np.cumsum(hist).astype(np.float64)
    total = cumsum[-1]

    if total < 100:
        log(output_dir, "Image is nearly all black — posterizing evenly", "WARN")
        thresholds = [int(256 * (i + 1) / (num_tones + 1)) for i in range(num_tones - 1)]
    elif num_tones == 2:
        # Otsu's method — find threshold that maximizes inter-class variance
        best_thresh = 128
        best_var = 0
        for t in range(10, 246):
            w0 = cumsum[t]
            w1 = total - w0
            if w0 == 0 or w1 == 0:
                continue
            # Mean of each class
            sum0 = np.sum(np.arange(t + 1) * hist[:t + 1])
            sum1 = np.sum(np.arange(t + 1, 256) * hist[t + 1:])
            mean0 = sum0 / w0
            mean1 = sum1 / w1
            var = w0 * w1 * (mean0 - mean1) ** 2
            if var > best_var:
                best_var = var
                best_thresh = t
        thresholds = [best_thresh]
        log(output_dir, f"Otsu threshold: {best_thresh}")
    elif num_tones == 3:
        # Two thresholds: find via percentile approach
        # Dark/mid split ~40th percentile, mid/light split ~75th percentile
        dark_pct = 0.40
        light_pct = 0.75
        t1 = np.searchsorted(cumsum, total * dark_pct)
        t2 = np.searchsorted(cumsum, total * light_pct)
        thresholds = [int(t1), int(t2)]
        log(output_dir, f"3-tone thresholds: {t1}, {t2}")
    else:
        # Even percentile split for 4+ tones
        thresholds = []
        for i in range(1, num_tones):
            pct = i / num_tones
            t = np.searchsorted(cumsum, total * pct)
            thresholds.append(int(t))
        log(output_dir, f"{num_tones}-tone thresholds: {thresholds}")

    # Create tone map: each pixel gets a tone index (0 = darkest)
    tone_map = np.zeros_like(gray, dtype=np.uint8)
    for i, t in enumerate(thresholds):
        tone_map[gray >= t] = i + 1

    log(output_dir, f"Posterized to {num_tones} tones, thresholds={thresholds}")
    return tone_map, thresholds


# ---------------------------------------------------------------------------
# Step 5: Palette remap
# ---------------------------------------------------------------------------
def remap_to_palette(tone_map, palette_name, num_tones, subject_mask=None,
                     shadow_mask=None, scene_context=None, output_dir=None):
    """Map tone indices to palette colors, with BG, context shapes, and cast shadow."""
    palette = PALETTES[palette_name]
    h, w = tone_map.shape

    if num_tones == 2:
        colors = [palette["black"], palette["highlight"]]
    elif num_tones == 3:
        colors = [palette["black"], palette["mid"], palette["highlight"]]
    else:
        all_colors = [palette["black"], palette["shadow"], palette["mid"], palette["highlight"]]
        colors = []
        for i in range(num_tones):
            t = i / max(1, num_tones - 1) * (len(all_colors) - 1)
            idx = int(t)
            frac = t - idx
            if idx >= len(all_colors) - 1:
                colors.append(all_colors[-1])
            else:
                c0 = np.array(all_colors[idx])
                c1 = np.array(all_colors[idx + 1])
                colors.append(tuple((c0 + frac * (c1 - c0)).astype(int)))

    # Start with BG color (not black)
    result = np.zeros((h, w, 3), dtype=np.uint8)
    bg_color = palette.get("bg", (80, 95, 110))
    result[:] = bg_color

    # Draw context shapes in BG
    if scene_context and "shapes" in scene_context:
        bg_img = Image.fromarray(result)
        draw = ImageDraw.Draw(bg_img)
        for shape in scene_context["shapes"]:
            raw_pts = shape["points"]
            # Auto-detect if coords are normalized (0-1) or pixel-like (0-999+)
            max_coord = max(max(abs(x), abs(y)) for x, y in raw_pts)
            if max_coord > 1.0:
                # Pixel-like coords — normalize by dividing by max
                norm = max(max(x for x, _ in raw_pts), max(y for _, y in raw_pts))
                if norm > 0:
                    pts = [(int(x / norm * w), int(y / norm * h)) for x, y in raw_pts]
                else:
                    continue
            else:
                pts = [(int(x * w), int(y * h)) for x, y in raw_pts]
            tone = shape.get("tone", "darker")
            if tone == "darker":
                color = tuple(max(0, c - 45) for c in bg_color)  # stronger contrast
            else:
                color = tuple(min(255, c + 35) for c in bg_color)
            draw.polygon(pts, fill=color)
            log(output_dir, f"Context shape: {shape.get('description', '?')} at {pts[:3]}...")
        result = np.array(bg_img)
        log(output_dir, f"Drew {len(scene_context['shapes'])} context shapes")

    # Apply cast shadow in BG
    if shadow_mask is not None:
        sh_arr = np.array(shadow_mask.resize((w, h))).astype(np.float32) / 255.0
        shadow_color = palette.get("shadow_cast", (40, 50, 62))
        for c in range(3):
            result[:, :, c] = (
                result[:, :, c].astype(np.float32) * (1 - sh_arr * 0.7) +
                shadow_color[c] * sh_arr * 0.7
            ).astype(np.uint8)
        log(output_dir, f"Applied cast shadow (70% opacity)")

    # Apply subject tones ONLY where subject exists
    subject_arr = np.array(subject_mask.resize((w, h))) > 30 if subject_mask is not None else np.ones((h, w), dtype=bool)
    for i, color in enumerate(colors):
        mask = (tone_map == i) & subject_arr
        result[mask] = color

    # Post-smooth boundaries
    result_img = Image.fromarray(result, "RGB")
    result_img = result_img.filter(ImageFilter.MedianFilter(size=9))

    log(output_dir, f"Remapped to '{palette_name}' palette with {len(colors)} colors: {colors}")
    return result_img


# ---------------------------------------------------------------------------
# Step 6: Edge roughening
# ---------------------------------------------------------------------------
def roughen_edges(img, tone_map, strength=0.5, output_dir=None):
    """Displace tone boundaries with low-frequency noise for painterly feel.

    Real paint strokes don't have pixel-perfect boundaries between tones.
    """
    h, w = tone_map.shape
    arr = np.array(img)

    # Find edges between tone zones
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[1:, :] |= tone_map[1:, :] != tone_map[:-1, :]
    edge_mask[:, 1:] |= tone_map[:, 1:] != tone_map[:, :-1]

    # Dilate edge region
    short_edge = min(w, h)
    dilate_r = max(2, int(short_edge * 0.008 * (1 + strength)))
    edge_pil = Image.fromarray(edge_mask.astype(np.uint8) * 255, "L")
    edge_pil = edge_pil.filter(ImageFilter.MaxFilter(dilate_r * 2 + 1))
    edge_arr = np.array(edge_pil) > 127

    # Generate low-frequency displacement noise
    noise_scale = max(4, int(short_edge * 0.03))
    small_h, small_w = max(2, h // noise_scale), max(2, w // noise_scale)
    dx_small = np.random.uniform(-1, 1, (small_h, small_w)).astype(np.float32)
    dy_small = np.random.uniform(-1, 1, (small_h, small_w)).astype(np.float32)

    # Upscale noise to full resolution (this makes it low-frequency)
    dx = np.array(Image.fromarray(dx_small).resize((w, h), Image.BILINEAR))
    dy = np.array(Image.fromarray(dy_small).resize((w, h), Image.BILINEAR))

    # Displacement magnitude scales to image size
    mag = short_edge * 0.006 * (1 + strength)

    # Apply displacement only at edges
    coords_y, coords_x = np.mgrid[0:h, 0:w]
    src_x = np.clip((coords_x + dx * mag * edge_arr).astype(int), 0, w - 1)
    src_y = np.clip((coords_y + dy * mag * edge_arr).astype(int), 0, h - 1)

    result = arr[src_y, src_x]

    num_displaced = edge_arr.sum()
    log(output_dir, f"Edge roughening: {num_displaced} pixels displaced, mag={mag:.1f}px, dilate={dilate_r}px")
    return Image.fromarray(result, "RGB")


# ---------------------------------------------------------------------------
# Step 6b: Smooth contour vectorization
# ---------------------------------------------------------------------------
def _slight_curve_between(p1, p2, bow=0.025):
    """Create a slightly curved path between two points (quadratic bezier).

    Each segment bows a tiny bit to one side, like a painter's hand naturally
    wobbles. Much more natural than ruler-straight polygon edges.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 3:
        return [p1, p2]
    nx, ny = -dy / seg_len, dx / seg_len
    bow_amt = seg_len * bow * random.choice([-1, 1]) * random.uniform(0.5, 1.5)
    mid_x = (p1[0] + p2[0]) / 2 + nx * bow_amt
    mid_y = (p1[1] + p2[1]) / 2 + ny * bow_amt
    return [(int((1 - t)**2 * p1[0] + 2 * (1 - t) * t * mid_x + t**2 * p2[0]),
             int((1 - t)**2 * p1[1] + 2 * (1 - t) * t * mid_y + t**2 * p2[1]))
            for t in np.linspace(0, 1, 10)]


def _smooth_contour(cnt, bow=0.025):
    """Simplify contour with Douglas-Peucker, then add slight curves between points.

    Converts jagged pixel-level posterization boundaries into smooth, decisive
    strokes like a painter would draw.
    """
    epsilon = max(3, int(cv2.arcLength(cnt, True) * 0.006))
    simplified = cv2.approxPolyDP(cnt, epsilon, True).squeeze()
    if len(simplified) < 3:
        return simplified
    curved = []
    for i in range(len(simplified)):
        seg = _slight_curve_between(simplified[i], simplified[(i + 1) % len(simplified)], bow)
        curved.extend(seg[:-1])
    return np.array(curved, dtype=np.int32)


def vectorize_tones(tone_map, subject_mask, palette, output_dir=None):
    """Convert posterized tone map into smooth vectorized contours.

    Instead of pixel-level tone boundaries, uses Douglas-Peucker simplification
    + slight bezier curves for painterly edges. Returns RGB PIL Image.
    """
    h, w = tone_map.shape
    pal = PALETTES[palette]
    ma = np.array(subject_mask.resize((w, h))) > 30

    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:] = pal.get("bg", (80, 95, 110))

    # Subject silhouette with smooth edges
    subj_uint8 = (ma.astype(np.uint8) * 255)
    subj_contours, _ = cv2.findContours(subj_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in subj_contours:
        curved = _smooth_contour(cnt, bow=0.015)
        cv2.fillPoly(result, [curved.reshape(-1, 1, 2)], pal["black"])

    # Highlight zones with smooth edges
    gray = np.array(Image.fromarray(tone_map).convert("L"))
    # Find the highlight tone (highest index)
    max_tone = tone_map.max()
    for tone_idx in range(1, max_tone + 1):
        zone = (tone_map == tone_idx) & ma
        contours, _ = cv2.findContours(
            (zone.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        # Pick color for this tone
        if max_tone == 1:
            color = pal["highlight"]
        elif tone_idx == max_tone:
            color = pal["highlight"]
        elif tone_idx == 1 and max_tone >= 2:
            color = pal.get("mid", pal["shadow"])
        else:
            color = pal.get("shadow", pal["black"])

        drawn = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            curved = _smooth_contour(cnt, bow=0.025)
            cv2.fillPoly(result, [curved.reshape(-1, 1, 2)], color)
            drawn += 1

    result_img = Image.fromarray(result, "RGB")
    # Light anti-aliasing
    result_img = result_img.filter(ImageFilter.GaussianBlur(radius=1.0))

    log(output_dir, f"Vectorized tones with smooth curved contours")
    return result_img


# ---------------------------------------------------------------------------
# Step 7: Paint texture via img2img (optional, uses Tensor Art)
# ---------------------------------------------------------------------------
def add_paint_texture(img, output_dir, seed=None, strength=0.18):
    """Light img2img pass to add organic paint/brush texture.

    Uses Tensor Art SDXL img2img at very low strength (0.15-0.20) with
    a painterly prompt. This adds brush stroke texture without changing
    the composition.
    """
    tensor_key = os.environ.get("TENSOR_API_KEY", "")
    if not tensor_key:
        log(output_dir, "TENSOR_API_KEY not set — skipping paint texture pass", "WARN")
        return img

    img_b64 = _img_to_b64(img, max_size=1536)

    prompt = (
        "gouache painting on canvas, visible brush strokes, acrylic paint texture, "
        "thick impasto paint application, canvas weave visible, hand-painted artwork, "
        "fine art painting, artistic brush work"
    )
    negative = (
        "photo, photograph, digital art, smooth, blurry, soft focus, "
        "3d render, cartoon, anime"
    )

    payload = {
        "request_id": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": seed or random.randint(0, 2**32 - 1),
                    "count": 1,
                },
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": min(img.width, 1536),
                    "height": min(img.height, 1536),
                    "prompts": [{"text": prompt}],
                    "negativePrompts": [{"text": negative}],
                    "sdModel": "600423083519508503",  # SDXL base
                    "sampler": "DPM++ 2M Karras",
                    "steps": 25,
                    "cfgScale": 7.0,
                    "denoisingStrength": strength,
                    "initImage": f"data:image/jpeg;base64,{img_b64}",
                    "sdVae": "Automatic",
                    "clipSkip": 2,
                },
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {tensor_key}",
        "Content-Type": "application/json",
    }

    try:
        # Create job
        response = requests.post("https://api.tensor.art/works/v1/works/task",
                                 headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            log(output_dir, f"Tensor Art create failed ({response.status_code}): {response.text[:200]}", "WARN")
            return img

        job_data = response.json()
        job_id = job_data.get("job", {}).get("id")
        if not job_id:
            log(output_dir, f"Tensor Art no job ID: {json.dumps(job_data)[:200]}", "WARN")
            return img

        log(output_dir, f"Tensor Art job: {job_id}, strength={strength}")

        # Poll for completion
        for attempt in range(60):
            time.sleep(3)
            check = requests.get(f"https://api.tensor.art/works/v1/works/task?ids={job_id}",
                                 headers=headers, timeout=15)
            if check.status_code != 200:
                continue
            tasks = check.json().get("tasks", [])
            if not tasks:
                continue
            status = tasks[0].get("status")
            if status == "COMPLETED":
                images = tasks[0].get("result", {}).get("images", [])
                if images:
                    result_url = images[0].get("url")
                    log(output_dir, f"Paint texture CDN: {result_url}")
                    result_img = Image.open(BytesIO(requests.get(result_url, timeout=30).content)).convert("RGB")
                    return result_img.resize(img.size, Image.LANCZOS)
                break
            elif status in ("FAILED", "CANCELLED"):
                log(output_dir, f"Tensor Art job {status}", "WARN")
                break

        log(output_dir, "Tensor Art timed out — using untextured version", "WARN")
        return img

    except Exception as e:
        log(output_dir, f"Paint texture failed: {e}", "WARN")
        return img


# ---------------------------------------------------------------------------
# Step 8: Canvas texture overlay
# ---------------------------------------------------------------------------
def add_canvas_texture(img, strength=0.15, output_dir=None, canvas_path=None):
    """Overlay a real canvas texture using seamless tiling.

    Uses a coarse burlap/canvas texture image, tiles it with overlapping
    feathered edges (no visible seams), random flips for variety, and
    multiply-blends onto the painting.

    If no canvas_path is provided, generates one via Flux Schnell.
    The texture is cached at ~/.openclaw/workspace/shared/canvas_texture.jpg.
    """
    w, h = img.size

    # Load or generate canvas texture
    cache_path = os.path.expanduser("~/.openclaw/workspace/shared/canvas_texture.jpg")
    if canvas_path and os.path.exists(canvas_path):
        tex_src = canvas_path
    elif os.path.exists(cache_path):
        tex_src = cache_path
    else:
        # Generate coarse canvas texture
        log(output_dir, "Generating canvas texture via Flux Schnell...")
        headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
        payload = {
            "prompt": "extreme close-up of coarse burlap sack texture, thick rough jute weave, "
                      "large visible threads, heavy woven fabric, neutral grey-brown, even flat lighting",
            "image_size": {"width": 512, "height": 512},
            "num_images": 1,
        }
        try:
            resp = requests.post("https://fal.run/fal-ai/flux/schnell",
                                 headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                url = data["images"][0]["url"] if isinstance(data["images"][0], dict) else data["images"][0]
                tex_img = Image.open(BytesIO(requests.get(url, timeout=30).content)).convert("L")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                tex_img.save(cache_path, quality=95)
                log(output_dir, f"Canvas texture cached at {cache_path}")
            else:
                log(output_dir, f"Canvas generation failed ({resp.status_code}) — using procedural fallback", "WARN")
                # Procedural fallback
                tex_img = Image.fromarray(
                    (np.random.uniform(0.85, 1.0, (200, 200)) * 255).astype(np.uint8), "L")
                tex_img.save(cache_path, quality=95)
        except Exception as e:
            log(output_dir, f"Canvas generation failed: {e} — using procedural fallback", "WARN")
            tex_img = Image.fromarray(
                (np.random.uniform(0.85, 1.0, (200, 200)) * 255).astype(np.uint8), "L")
            tex_img.save(cache_path, quality=95)
        tex_src = cache_path

    canvas_tex = Image.open(tex_src).convert("L")

    # Scale up 1.5x for coarser, more visible threads
    canvas_tex = canvas_tex.resize(
        (int(canvas_tex.width * 1.5), int(canvas_tex.height * 1.5)), Image.LANCZOS)
    cw, ch = canvas_tex.size
    canvas_arr = np.array(canvas_tex).astype(np.float32) / 255.0

    # Seamless tiling with overlapping feathered edges + random flips
    big = np.zeros((h + ch, w + cw), dtype=np.float32)
    weight = np.zeros_like(big)

    feather = np.ones((ch, cw), dtype=np.float32)
    fade = min(ch, cw) // 4
    for i in range(fade):
        f = i / fade
        feather[i, :] *= f
        feather[-(i + 1), :] *= f
        feather[:, i] *= f
        feather[:, -(i + 1)] *= f

    step_x = int(cw * 0.6)  # 40% overlap
    step_y = int(ch * 0.6)

    for ty in range(-1, (h // step_y) + 2):
        for tx in range(-1, (w // step_x) + 2):
            ox = tx * step_x + random.randint(-step_x // 4, step_x // 4)
            oy = ty * step_y + random.randint(-step_y // 4, step_y // 4)
            tile = canvas_arr.copy()
            if random.random() > 0.5:
                tile = tile[::-1, :]
            if random.random() > 0.5:
                tile = tile[:, ::-1]
            y1 = max(0, oy)
            y2 = min(h + ch, oy + ch)
            x1 = max(0, ox)
            x2 = min(w + cw, ox + cw)
            ty1, tx1 = y1 - oy, x1 - ox
            ty2 = ty1 + (y2 - y1)
            tx2 = tx1 + (x2 - x1)
            if ty2 <= ty1 or tx2 <= tx1:
                continue
            big[y1:y2, x1:x2] += tile[ty1:ty2, tx1:tx2] * feather[ty1:ty2, tx1:tx2]
            weight[y1:y2, x1:x2] += feather[ty1:ty2, tx1:tx2]

    weight = np.maximum(weight, 0.001)
    tiled = (big / weight)[:h, :w]

    # Multiply blend: strength controls how visible the texture is
    tex_norm = (1.0 - strength) + (2.0 * strength) * tiled  # e.g. 0.78-1.22 at strength=0.22

    img_arr = np.array(img).astype(np.float32)
    result = np.clip(img_arr * tex_norm[:, :, np.newaxis], 0, 255).astype(np.uint8)

    log(output_dir, f"Canvas texture: coarse seamless tiling, strength={strength:.2f}")
    return Image.fromarray(result, "RGB")


# ---------------------------------------------------------------------------
# Gemini Evaluation
# ---------------------------------------------------------------------------
_EVAL_PROMPT = """\
You are an art director evaluating a painted portrait in the style of high-contrast \
gouache/acrylic paintings (similar to pulpbrother's style).

Evaluate the image on:
1. Is the tonal separation (posterization) clean and dramatic?
2. Do the shadows create interesting shapes that define the figure?
3. Does it have visible paint texture / brushwork feel?
4. Is the directional lighting effective — does it sculpt the form?
5. Does the palette work (monochrome consistency)?
6. Overall artistic impact and graphic boldness.

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "flat_lighting", "muddy_tones", "too_many_tones", \
"no_texture", "palette_wrong", "subject_lost", "too_dark", "too_bright">]
}"""


def evaluate_with_gemini(img, output_dir):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping evaluation")
        return None
    try:
        img_b64 = _img_to_b64(img, max_size=1024)
        parts = [
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            {"text": _EVAL_PROMPT},
        ]
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json=payload, timeout=60,
        )
        if response.status_code != 200:
            log(output_dir, f"Gemini returned {response.status_code}", "WARN")
            return None
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except Exception as e:
        log(output_dir, f"Gemini evaluation failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Noir Paint — high-contrast painterly effect")
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--tones", default="cool", choices=list(PALETTES.keys()),
                        help=f"Color palette (default: cool). Options: {', '.join(PALETTES.keys())}")
    parser.add_argument("--num-tones", type=int, default=2, choices=[2, 3, 4],
                        help="Number of tone levels (default: 2)")
    parser.add_argument("--light-angle", type=float, default=None,
                        help="Override light direction in degrees (0=right, 90=up, 180=left). Default: auto from body axis")
    parser.add_argument("--highres-denoise", type=float, default=0.45,
                        help="IC-Light highres denoise (default: 0.45, lower=more faithful)")
    parser.add_argument("--paint-strength", type=float, default=0.18,
                        help="Img2img paint texture strength (default: 0.18, 0=skip)")
    parser.add_argument("--canvas-strength", type=float, default=0.22,
                        help="Canvas weave texture strength (default: 0.22, 0=skip)")
    parser.add_argument("--edge-roughness", type=float, default=0.5,
                        help="Edge displacement strength (default: 0.5, 0=skip)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-correct", action="store_true")
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None)
    parser.add_argument("--list-palettes", action="store_true")
    args = parser.parse_args()

    if args.list_palettes:
        print(f"\n{'Palette':<10} Description")
        print("=" * 50)
        for name, pal in PALETTES.items():
            print(f"  {name:<8} {pal['description']}")
        sys.exit(0)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Output directory
    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name}_{source_basename}_{timestamp}_noir_{args.tones}_{seed % 100:02d}"
    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    timings = {}

    # ========================================================================
    # Step 1: Extract subject + scene context (parallel)
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 1/9: Extract subject + scene context ---")
    subject_on_black, subject_mask, img_orig = extract_subject_on_black(source, output_dir)
    subject_on_black.save(os.path.join(output_dir, "1_subject_on_black.jpg"), "JPEG", quality=95)
    subject_mask.save(os.path.join(output_dir, "1_subject_mask.png"))

    # Scene context (Gemini) — surface shapes + subject count
    scene_context = detect_scene_context(source, output_dir)

    timings["extract"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['extract']:.1f}s)")

    # ========================================================================
    # Step 2: Detect body axis → light direction
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 2/9: Body axis → light direction ---")

    if args.light_angle is not None:
        light_angle = args.light_angle
        log(output_dir, f"Light angle override: {light_angle:.0f}°")
    else:
        img_array = np.array(img_orig)
        light_angle = detect_body_axis(img_array, subject_mask, output_dir)
        if light_angle is None:
            light_angle = 135.0  # Default: upper-left
            log(output_dir, f"Fallback light angle: {light_angle:.0f}°")

    light_prompt, light_negative = angle_to_light_prompt(light_angle)
    log(output_dir, f"Light prompt: '{light_prompt[:80]}...'")
    timings["axis"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['axis']:.1f}s)")

    # ========================================================================
    # Step 3: Relight
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 3/9: Relight (IC-Light) ---")
    relit = relight_subject(
        subject_on_black, light_prompt, light_negative, output_dir,
        seed=seed, highres_denoise=args.highres_denoise,
    )
    if relit is None:
        log(output_dir, "Relighting failed — using subject on black as fallback", "WARN")
        relit = subject_on_black

    # Resize to match original
    if relit.size != img_orig.size:
        relit = relit.resize(img_orig.size, Image.LANCZOS)

    relit.save(os.path.join(output_dir, "3_relit.jpg"), "JPEG", quality=95)
    timings["relight"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['relight']:.1f}s)")

    # ========================================================================
    # Step 4: Cast shadow (disabled — parked for manual direction later)
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 4/9: Cast shadow (skipped) ---")
    shadow_mask = None  # Shadow casting parked — procedural approach didn't generalize
    timings["shadow"] = time.time() - t0

    # ========================================================================
    # Step 5: Bilateral presmooth + Posterize
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 5/9: Presmooth + Posterize ---")

    # Mask the relit image to subject only (force BG to black)
    relit_arr = np.array(relit)
    mask_arr = np.array(subject_mask.resize(relit.size)) < 30
    relit_arr[mask_arr] = 0
    relit_masked = Image.fromarray(relit_arr, "RGB")

    # Bilateral presmooth to eliminate gradient jitter
    smoothed = presmooth_bilateral(relit_masked, output_dir)

    tone_map, thresholds = posterize_histogram(smoothed, args.num_tones, output_dir)
    timings["posterize"] = time.time() - t0
    log(output_dir, f"Step 5 done ({timings['posterize']:.1f}s)")

    # ========================================================================
    # Step 6: Vectorize tones (smooth contours with slight curves)
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 6/9: Vectorize tones ---")
    random.seed(seed)  # deterministic curves
    vectorized = vectorize_tones(tone_map, subject_mask, args.tones, output_dir)
    vectorized.save(os.path.join(output_dir, "6_vectorized.jpg"), "JPEG", quality=95)
    timings["remap"] = time.time() - t0
    log(output_dir, f"Step 6 done ({timings['remap']:.1f}s)")

    # ========================================================================
    # Step 7: Edge roughening (optional, on top of vectorized contours)
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 7/9: Edge roughening ---")
    if args.edge_roughness > 0:
        roughened = roughen_edges(vectorized, tone_map, args.edge_roughness, output_dir)
    else:
        roughened = vectorized
        log(output_dir, "Edge roughening skipped (strength=0)")
    roughened.save(os.path.join(output_dir, "7_roughened.jpg"), "JPEG", quality=95)
    timings["roughen"] = time.time() - t0
    log(output_dir, f"Step 7 done ({timings['roughen']:.1f}s)")

    # ========================================================================
    # Step 8: Paint texture (img2img)
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 8/9: Paint texture ---")
    if args.paint_strength > 0:
        painted = add_paint_texture(roughened, output_dir, seed=seed, strength=args.paint_strength)
    else:
        painted = roughened
        log(output_dir, "Paint texture skipped (strength=0)")
    painted.save(os.path.join(output_dir, "8_painted.jpg"), "JPEG", quality=95)
    timings["paint"] = time.time() - t0
    log(output_dir, f"Step 8 done ({timings['paint']:.1f}s)")

    # ========================================================================
    # Step 9: Canvas texture + output
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 9/9: Canvas texture + output ---")
    if args.canvas_strength > 0:
        final = add_canvas_texture(painted, args.canvas_strength, output_dir)
    else:
        final = painted
        log(output_dir, "Canvas texture skipped (strength=0)")

    final_path = os.path.join(output_dir, "9_noir_final.jpg")
    final.save(final_path, "JPEG", quality=95)

    # Evaluate
    eval_result = evaluate_with_gemini(final, output_dir)

    # Copy to finals
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates")
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_name = os.path.basename(output_dir) + ".jpg"
    finals_dest = os.path.join(finals_dir, finals_name)
    with open(final_path, "rb") as f_in:
        with open(finals_dest, "wb") as f_out:
            f_out.write(f_in.read())
    log(output_dir, f"Final copied to: {finals_dest}")

    # Push to phone
    try:
        from notify import push_image
        src_name = os.path.splitext(os.path.basename(args.source))[0]
        push_image(finals_dest, title=f"Noir Paint — {src_name}", body=f"{args.tones} palette, {args.num_tones} tones")
        log(output_dir, "Pushed to phone")
    except Exception as e:
        log(output_dir, f"Push notification failed: {e}", "WARN")

    # Copy script
    try:
        shutil.copy2(os.path.abspath(__file__), os.path.join(output_dir, f"workflow_script_{os.path.basename(__file__)}"))
    except Exception:
        pass

    timings["output"] = time.time() - t0
    log(output_dir, f"Step 8 done ({timings['output']:.1f}s)")

    # --- Summary ---
    total = sum(timings.values())
    score_str = f"{eval_result['score']}/10 — {eval_result.get('critique', '')}" if eval_result else "N/A"

    print(f"""
============================================================
  NOIR PAINT SUMMARY
============================================================
  Source:          {source}
  Palette:         {args.tones} ({args.num_tones} tones)
  Light angle:     {light_angle:.0f}°
  Posterize:       {args.num_tones} tones, thresholds={thresholds}
  Paint texture:   {args.paint_strength}
  Canvas texture:  {args.canvas_strength}
  Edge roughness:  {args.edge_roughness}
  Seed:            {seed}

  Step Timings:
    1. Extract + context     {timings.get('extract', 0):>8.1f}s
    2. Body axis detection   {timings.get('axis', 0):>8.1f}s
    3. Relight (IC-Light)    {timings.get('relight', 0):>8.1f}s
    4. Cast shadow           {timings.get('shadow', 0):>8.1f}s
    5. Presmooth + posterize {timings.get('posterize', 0):>8.1f}s
    6. Palette remap         {timings.get('remap', 0):>8.1f}s
    7. Edge roughening       {timings.get('roughen', 0):>8.1f}s
    8. Paint texture         {timings.get('paint', 0):>8.1f}s
    9. Canvas + output       {timings.get('output', 0):>8.1f}s
    TOTAL                    {total:>8.1f}s

  Quality Report:
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}
============================================================""")


if __name__ == "__main__":
    main()
