#!/home/rong/openclaw-venv/bin/python3
"""
Foreground Framing Workflow

Adds blurry foreground elements to the edges of a photo, simulating the
"shoot-through" technique (shooting through foliage, doorframes, fabric, etc.)
at shallow depth of field.

Pipeline:
  1. Analyze photo (EXIF, scene via Gemini, subject mask, depth map)
  2. Generate foreground element via text-to-image (Flux Schnell on black BG)
  3. Extract element alpha from black background
  4. Depth-aware DOF blur (physically-based circle of confusion)
  5. Color match + darken
  6. Composite over original (respecting subject mask + edge mask)
  7. Evaluate + output

Usage:
    python foreground-framing.py --source photo.jpg --framing "foliage"
    python foreground-framing.py --source photo.jpg --framing "doorframe" --coverage 0.25
    python foreground-framing.py --list-presets
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
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageDraw, ImageEnhance

# Use shared masking module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Framing Presets — generation-style prompts (element on black background)
# ---------------------------------------------------------------------------
FRAMING_PRESETS = {
    "foliage": {
        "prompt": "green oak leaves and small twigs with dappled light, natural organic shapes, lush vegetation, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, colorful background, bright background, white background",
        "description": "Blurry green leaves/branches framing the shot",
    },
    "warm foliage": {
        "prompt": "warm autumn leaves in golden brown and orange tones, dried twigs, fall foliage, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, green, bright background, white background",
        "description": "Warm autumn-toned blurry leaves",
    },
    "doorframe": {
        "prompt": "dark wooden doorframe edge with warm wood grain texture, architectural element, aged wood, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, bright background, white background, full door",
        "description": "Dark wooden doorframe edges",
    },
    "curtain": {
        "prompt": "sheer white curtain fabric, soft translucent flowing textile, delicate draping, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, opaque, bright background, colorful",
        "description": "Soft sheer curtain fabric",
    },
    "dark curtain": {
        "prompt": "dark velvet curtain fabric, rich heavy draping textile, theatrical deep tones, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, bright, white, colorful background",
        "description": "Dark velvet curtain draping",
    },
    "flowers": {
        "prompt": "colorful flower petals and blossoms, soft delicate petals in pink and white, romantic floral arrangement, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, stems, bright background, white background",
        "description": "Blurry flower petals framing",
    },
    "fairy lights": {
        "prompt": "warm golden fairy lights, string of glowing bokeh orbs, warm light circles, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, daylight, bright background",
        "description": "Warm bokeh light circles",
    },
    "metal": {
        "prompt": "dark iron railing and metal bars, industrial metalwork, aged dark metal with patina, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, bright, shiny, chrome, white background",
        "description": "Dark metal railing/bars",
    },
    "smoke": {
        "prompt": "wispy tendrils of white and grey smoke, ethereal fog wisps, atmospheric haze, delicate swirling patterns, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, colorful, bright background, fire",
        "description": "Ethereal smoke/haze framing",
    },
    "brick": {
        "prompt": "red brick wall corner edge, warm masonry texture, urban architectural element, rough textured bricks, against solid black background, isolated object, no other objects",
        "negative": "person, face, text, bright background, white background, full wall",
        "description": "Blurry brick wall edge",
    },
}

_log_lock = threading.Lock()


def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    with _log_lock:
        log_path = os.path.join(output_dir, "workflow.log")
        with open(log_path, "a") as f:
            f.write(formatted + "\n")


def check_image_quality(img, label, output_dir):
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    entropy = gray.entropy()
    reasons = []
    if brightness < 10:
        reasons.append(f"nearly black (brightness={brightness:.1f})")
    elif brightness > 245:
        reasons.append(f"nearly white (brightness={brightness:.1f})")
    if contrast < 5:
        reasons.append(f"flat/uniform (contrast={contrast:.1f})")
    if entropy < 1.0:
        reasons.append(f"zero-entropy (entropy={entropy:.2f})")
    ok = len(reasons) == 0
    if not ok:
        log(output_dir, f"QUALITY FAIL [{label}]: {'; '.join(reasons)}", "WARN")
    else:
        log(output_dir, f"Quality OK [{label}]: brightness={brightness:.1f} contrast={contrast:.1f} entropy={entropy:.2f}")
    return {"ok": ok, "brightness": round(brightness, 1), "contrast": round(contrast, 1), "entropy": round(entropy, 2)}


def _get_fal_key():
    key = os.environ.get("FAL_API_KEY")
    if not key:
        raise EnvironmentError("FAL_API_KEY not set")
    return key


def _img_to_b64_simple(img, fmt="JPEG", quality=90):
    buf = BytesIO()
    if img.mode == "RGBA" and fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _img_to_b64(img, max_size=None, fmt="JPEG", quality=85):
    """Encode image to base64. Optionally downscale."""
    img_out = img.copy()
    if max_size:
        img_out.thumbnail((max_size, max_size), Image.LANCZOS)
    if img_out.mode == "RGBA" and fmt == "JPEG":
        img_out = img_out.convert("RGB")
    buf = BytesIO()
    img_out.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------
# EXIF tag IDs
_EXIF_FOCAL_LENGTH = 37386
_EXIF_FNUMBER = 33437
_EXIF_LENS_MODEL = 42036
_EXIF_EXPOSURE_TIME = 33434
_EXIF_ISO = 34855


def extract_exif(image_path, output_dir):
    """Extract photographic EXIF data from source image.

    Returns dict with focal_length_mm, aperture, lens_model, exposure_time, iso.
    Uses defaults (50mm, f/2.0) if EXIF is missing.
    """
    defaults = {
        "focal_length_mm": 50.0,
        "aperture": 2.0,
        "lens_model": None,
        "exposure_time": None,
        "iso": None,
    }

    try:
        img = Image.open(image_path)
        exif_data = img.getexif()
        if not exif_data:
            log(output_dir, "No EXIF data found — using defaults (50mm f/2.0)")
            return defaults

        result = dict(defaults)

        # Focal length
        fl = exif_data.get(_EXIF_FOCAL_LENGTH)
        if fl is not None:
            # May be IFDRational or tuple
            if hasattr(fl, 'numerator'):
                result["focal_length_mm"] = float(fl.numerator) / float(fl.denominator) if fl.denominator else 50.0
            elif isinstance(fl, tuple):
                result["focal_length_mm"] = float(fl[0]) / float(fl[1]) if fl[1] else 50.0
            else:
                result["focal_length_mm"] = float(fl)

        # Aperture (f-number)
        fn = exif_data.get(_EXIF_FNUMBER)
        if fn is not None:
            if hasattr(fn, 'numerator'):
                result["aperture"] = float(fn.numerator) / float(fn.denominator) if fn.denominator else 2.0
            elif isinstance(fn, tuple):
                result["aperture"] = float(fn[0]) / float(fn[1]) if fn[1] else 2.0
            else:
                result["aperture"] = float(fn)

        # Lens model
        lm = exif_data.get(_EXIF_LENS_MODEL)
        if lm is not None:
            result["lens_model"] = str(lm)

        # Exposure time
        et = exif_data.get(_EXIF_EXPOSURE_TIME)
        if et is not None:
            if hasattr(et, 'numerator'):
                result["exposure_time"] = f"{et.numerator}/{et.denominator}" if et.denominator else str(et)
            elif isinstance(et, tuple):
                result["exposure_time"] = f"{et[0]}/{et[1]}" if et[1] else str(et[0])
            else:
                result["exposure_time"] = str(et)

        # ISO
        iso = exif_data.get(_EXIF_ISO)
        if iso is not None:
            result["iso"] = int(iso)

        log(output_dir, f"EXIF: focal={result['focal_length_mm']:.1f}mm f/{result['aperture']:.1f}"
            f" lens={result['lens_model'] or 'unknown'}"
            f" exposure={result['exposure_time'] or 'unknown'}"
            f" ISO={result['iso'] or 'unknown'}")

        return result

    except Exception as e:
        log(output_dir, f"EXIF extraction failed: {e} — using defaults", "WARN")
        return defaults


# ---------------------------------------------------------------------------
# Depth estimation via fal.ai Depth Anything V2
# ---------------------------------------------------------------------------
def run_depth_estimation(image_path, output_dir):
    """Get depth map from fal.ai Depth Anything V2.

    Returns PIL L-mode image (depth map) or None on failure.
    """
    log(output_dir, "Running depth estimation (Depth Anything V2)...")
    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(
            "https://fal.run/fal-ai/imageutils/depth",
            headers=headers,
            json={"image_url": f"data:image/jpeg;base64,{img_b64}"},
            timeout=120,
        )
    except requests.RequestException as e:
        log(output_dir, f"Depth estimation request failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"Depth estimation failed ({response.status_code}): {response.text[:300]}", "ERROR")
        return None

    data = response.json()
    depth_url = data.get("image", {}).get("url")
    if not depth_url:
        log(output_dir, f"Depth estimation returned no image URL. Keys: {list(data.keys())}", "ERROR")
        return None

    log(output_dir, f"Depth map CDN URL: {depth_url}")
    depth_img = Image.open(requests.get(depth_url, stream=True, timeout=30).raw).convert("L")
    log(output_dir, f"Depth map size: {depth_img.size[0]}x{depth_img.size[1]}")
    return depth_img


def get_focus_distance_from_depth(depth_map, subject_mask, output_dir):
    """Sample depth at subject centroid to determine focus plane.

    Returns normalised depth value 0-1 (0=near, 1=far in depth map).
    """
    depth_arr = np.array(depth_map).astype(np.float32) / 255.0
    mask_arr = np.array(subject_mask.resize(depth_map.size, Image.LANCZOS))
    binary = mask_arr > 127

    if binary.sum() < 100:
        log(output_dir, "Subject mask too small for depth sampling — using center", "WARN")
        cy, cx = depth_arr.shape[0] // 2, depth_arr.shape[1] // 2
    else:
        ys, xs = np.where(binary)
        cy, cx = int(ys.mean()), int(xs.mean())

    # Sample a small region around centroid for stability
    r = max(5, min(depth_arr.shape) // 40)
    y0, y1 = max(0, cy - r), min(depth_arr.shape[0], cy + r)
    x0, x1 = max(0, cx - r), min(depth_arr.shape[1], cx + r)
    focus_depth = float(np.mean(depth_arr[y0:y1, x0:x1]))

    log(output_dir, f"Focus plane depth: {focus_depth:.3f} (sampled at centroid [{cx}, {cy}])")
    return focus_depth


# ---------------------------------------------------------------------------
# Subject detection for smart side selection
# ---------------------------------------------------------------------------
def detect_smart_sides(subject_mask, img_width, img_height, output_dir):
    """Analyze subject position to pick 2 adjacent sides for L-shaped framing."""
    mask_np = np.array(subject_mask)
    binary = (mask_np > 127).astype(np.float32)

    if binary.sum() < 100:
        log(output_dir, "Subject mask too small for smart detection — falling back to auto", "WARN")
        return None

    ys, xs = np.where(binary > 0)
    cx = xs.mean() / img_width
    cy = ys.mean() / img_height

    log(output_dir, f"Subject centroid: x={cx:.2f}, y={cy:.2f} (0=left/top, 1=right/bottom)")

    h_side = "left" if cx > 0.5 else "right"
    h_offset = abs(cx - 0.5)
    v_side = "top" if cy > 0.5 else "bottom"
    v_offset = abs(cy - 0.5)

    if h_offset >= v_offset:
        primary, secondary = h_side, v_side
        primary_mult, secondary_mult = 1.0, 0.6
    else:
        primary, secondary = v_side, h_side
        primary_mult, secondary_mult = 1.0, 0.6

    log(output_dir, f"Smart framing: primary={primary} (x{primary_mult}), secondary={secondary} (x{secondary_mult})")

    return {
        "primary": primary,
        "secondary": secondary,
        "primary_mult": primary_mult,
        "secondary_mult": secondary_mult,
    }


# ---------------------------------------------------------------------------
# Scene-aware framing prompt via Gemini
# ---------------------------------------------------------------------------
_SCENE_PROMPT = """\
You are a professional photographer planning a "shoot-through" foreground framing element for this photo.

Look at the scene: the environment, setting, objects present, indoor/outdoor, lighting direction, \
color temperature, and mood.

Suggest ONE specific foreground object that would look natural if it were very close to the camera lens \
and heavily out of focus. This object should:
1. Be something that BELONGS in this scene (e.g., leaves for outdoor, curtain for indoor, bottle for bathroom)
2. Be dark or semi-transparent when blurred
3. Add depth without distracting from the subject

Also describe the scene lighting direction (e.g., "warm light from upper left") and color temperature \
(e.g., "warm/golden", "cool/blue", "neutral").

Respond ONLY with valid JSON:
{
  "object": "<the object, e.g. 'green leaves and small twigs', 'dark wooden doorframe', 'frosted glass bottle'>",
  "prompt": "<generation prompt: '[object] against solid black background, isolated object, no other objects, [details]'>",
  "negative": "<what to avoid: 'person, face, text, bright background, [scene-specific]'>",
  "lighting_direction": "<e.g. 'warm light from upper left'>",
  "color_temperature": "<e.g. 'warm golden'>",
  "reasoning": "<one sentence why this fits the scene>"
}"""


def suggest_framing_prompt(image_path, output_dir):
    """Use Gemini to analyze the scene and suggest a contextual foreground element."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — cannot auto-detect framing", "WARN")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                {"text": _SCENE_PROMPT},
            ]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
            },
        }

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json=payload, timeout=30)

        if response.status_code != 200:
            log(output_dir, f"Gemini scene analysis failed ({response.status_code})", "WARN")
            return None

        resp_json = response.json()
        candidates = resp_json.get("candidates", [])
        if not candidates:
            return None

        raw = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                return None
            result = json.loads(raw[start:end + 1])

        log(output_dir, f"Scene analysis: {result.get('object', '?')} — {result.get('reasoning', '')}")
        if result.get("lighting_direction"):
            log(output_dir, f"Scene lighting: {result['lighting_direction']}, temperature: {result.get('color_temperature', '?')}")
        return result

    except Exception as e:
        log(output_dir, f"Scene analysis failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Edge mask generation
# ---------------------------------------------------------------------------
def create_edge_mask(width, height, coverage=0.20, sides="auto", irregularity=0.4, smart_sides=None):
    """Create an organic-looking edge mask for framing.

    White = areas where foreground element should appear.
    Black = areas to keep clear (subject region).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if smart_sides is not None:
        edge_px_x = int(width * coverage)
        edge_px_y = int(height * coverage)
        rects = []
        for role in ("primary", "secondary"):
            side = smart_sides[role]
            mult = smart_sides[f"{role}_mult"]
            if side == "left":
                w = int(edge_px_x * mult)
                rects.append(("left", 0, 0, w, height))
            elif side == "right":
                w = int(edge_px_x * mult)
                rects.append(("right", width - w, 0, width, height))
            elif side == "top":
                h = int(edge_px_y * mult)
                rects.append(("top", 0, 0, width, h))
            elif side == "bottom":
                h = int(edge_px_y * mult)
                rects.append(("bottom", 0, height - h, width, height))
    else:
        if sides == "auto":
            aspect = width / height
            if aspect > 1.3:
                sides = "left-right"
            elif aspect < 0.77:
                sides = "top-bottom"
            else:
                sides = "left-right"

        edge_px_x = int(width * coverage)
        edge_px_y = int(height * coverage)

        rects = []
        if sides in ("left-right", "all"):
            rects.append(("left", 0, 0, edge_px_x, height))
            rects.append(("right", width - edge_px_x, 0, width, height))
        if sides in ("top-bottom", "all"):
            rects.append(("top", 0, 0, width, edge_px_y))
            rects.append(("bottom", 0, height - edge_px_y, width, height))

    for side, x1, y1, x2, y2 in rects:
        draw.rectangle([x1, y1, x2, y2], fill=255)

    if irregularity > 0:
        np.random.seed(42)
        num_blobs = int(30 * (1 + irregularity))
        blob_size_range = (int(min(width, height) * 0.02), int(min(width, height) * 0.08 * (1 + irregularity)))

        for side, x1, y1, x2, y2 in rects:
            for _ in range(num_blobs):
                r = np.random.randint(blob_size_range[0], max(blob_size_range[1], blob_size_range[0] + 1))
                if side == "left":
                    cx = x2 + np.random.randint(-r, r)
                    cy = np.random.randint(0, height)
                elif side == "right":
                    cx = x1 + np.random.randint(-r, r)
                    cy = np.random.randint(0, height)
                elif side == "top":
                    cx = np.random.randint(0, width)
                    cy = y2 + np.random.randint(-r, r)
                elif side == "bottom":
                    cx = np.random.randint(0, width)
                    cy = y1 + np.random.randint(-r, r)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)

        for side, x1, y1, x2, y2 in rects:
            for _ in range(num_blobs // 3):
                r = np.random.randint(blob_size_range[0], max(blob_size_range[1] // 2, blob_size_range[0] + 1))
                if side == "left":
                    cx = x2 + np.random.randint(0, r * 2)
                    cy = np.random.randint(0, height)
                elif side == "right":
                    cx = x1 - np.random.randint(0, r * 2)
                    cy = np.random.randint(0, height)
                elif side == "top":
                    cx = np.random.randint(0, width)
                    cy = y2 + np.random.randint(0, r * 2)
                elif side == "bottom":
                    cx = np.random.randint(0, width)
                    cy = y1 - np.random.randint(0, r * 2)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.015))
    mask = mask.point(lambda p: 255 if p > 100 else (int(p * 2.55) if p > 40 else 0))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.01))

    return mask


# ---------------------------------------------------------------------------
# Text-to-image generation via fal.ai Flux Schnell
# ---------------------------------------------------------------------------
def generate_element(prompt, negative_prompt, width, height, output_dir,
                     seed=None, scene_context=None, smart_sides=None, exif_info=None):
    """Generate a foreground element on a black background using Flux Schnell.

    Returns PIL RGB image or None on failure.
    """
    # Build the full prompt with side placement, scene context, and camera info
    side_desc = ""
    if smart_sides:
        sides = []
        for role in ("primary", "secondary"):
            sides.append(smart_sides[role])
        side_desc = f", entering from the {' and '.join(sides)} edge of the frame"

    scene_desc = ""
    if scene_context:
        lighting = scene_context.get("lighting_direction", "")
        temperature = scene_context.get("color_temperature", "")
        if lighting or temperature:
            parts = []
            if lighting:
                parts.append(lighting)
            if temperature:
                parts.append(f"{temperature} tones")
            scene_desc = f", matching {' with '.join(parts)}"

    camera_desc = ""
    if exif_info:
        fl = exif_info.get("focal_length_mm", 50)
        ap = exif_info.get("aperture", 2.0)
        camera_desc = f", as seen through a {fl:.0f}mm f/{ap:.1f} lens"

    full_prompt = f"{prompt}{side_desc}{scene_desc}{camera_desc}"

    log(output_dir, f"Generating element: '{full_prompt[:120]}...'")

    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}

    payload = {
        "prompt": full_prompt,
        "image_size": {"width": width, "height": height},
        "num_images": 1,
    }
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt:
        # Flux Schnell does not have a dedicated negative prompt field but
        # we append negative guidance to the prompt itself
        payload["prompt"] = full_prompt + f". NOT: {negative_prompt}"

    try:
        response = requests.post(
            "https://fal.run/fal-ai/flux/schnell",
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        log(output_dir, f"Element generation request failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"Element generation failed ({response.status_code}): {response.text[:300]}", "ERROR")
        return None

    data = response.json()
    images = data.get("images", [])
    if not images:
        log(output_dir, f"Element generation returned no images. Keys: {list(data.keys())}", "ERROR")
        return None

    result_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
    if not result_url:
        log(output_dir, "Element generation returned no image URL", "ERROR")
        return None

    log(output_dir, f"Element CDN URL: {result_url}")
    result_img = Image.open(requests.get(result_url, stream=True, timeout=30).raw).convert("RGB")
    log(output_dir, f"Generated element: {result_img.size[0]}x{result_img.size[1]}")
    return result_img


# ---------------------------------------------------------------------------
# Extract alpha from black background
# ---------------------------------------------------------------------------
def extract_element_alpha(element_img, output_dir, threshold=20, feather_px=None):
    """Extract alpha mask from an element generated on a black background.

    Pixels brighter than threshold are considered part of the element.
    Returns L-mode PIL Image (alpha mask).
    """
    gray = element_img.convert("L")
    arr = np.array(gray).astype(np.float32)

    # Create soft alpha: ramp from 0 at threshold to 255 at threshold+40
    alpha = np.clip((arr - threshold) * (255.0 / 40.0), 0, 255).astype(np.uint8)

    alpha_img = Image.fromarray(alpha, "L")

    # Feather edges for smooth blending
    if feather_px is None:
        short_edge = min(element_img.size)
        feather_px = max(3, int(short_edge * 0.01))

    if feather_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather_px))

    coverage = np.mean(np.array(alpha_img) > 30) * 100
    log(output_dir, f"Element alpha: {coverage:.1f}% non-zero (threshold={threshold}, feather={feather_px}px)")

    return alpha_img


# ---------------------------------------------------------------------------
# Depth-aware DOF blur
# ---------------------------------------------------------------------------
def calculate_dof_blur(focal_length_mm, aperture, focus_distance_m, fg_distance_m,
                       image_width, sensor_width_mm=36.0):
    """Calculate physically-based defocus blur radius in pixels.

    Uses circle of confusion formula:
      CoC = |f^2 * (D_focus - D_fg)| / (D_fg * (D_focus - f) * N)

    where f=focal length, N=f-number, D=distances in meters.

    Returns blur radius in pixels (capped at 80).
    """
    f = focal_length_mm / 1000.0  # Convert to meters
    N = aperture
    D_focus = max(focus_distance_m, f + 0.001)  # Must be > focal length
    D_fg = max(fg_distance_m, f + 0.001)

    # Circle of confusion in meters
    numerator = abs(f * f * (D_focus - D_fg))
    denominator = D_fg * (D_focus - f) * N

    if denominator < 1e-10:
        coc_m = 0.001  # Fallback
    else:
        coc_m = numerator / denominator

    # Convert CoC from meters to pixels
    coc_px = coc_m * (image_width / (sensor_width_mm / 1000.0))

    # Cap to prevent abstract blobs
    blur_px = int(min(80, max(5, coc_px)))

    return blur_px, coc_m * 1000  # Return (blur_px, coc_mm)


def apply_dof_blur(element_img, element_alpha, depth_map, focus_depth,
                   exif_info, output_dir, blur_override=None):
    """Apply depth-aware defocus blur to the foreground element.

    The blur varies across the element based on the depth map — parts closer
    to camera get more blur.

    Returns blurred RGB image.
    """
    focal_length = exif_info.get("focal_length_mm", 50.0)
    aperture = exif_info.get("aperture", 2.0)

    # Estimate real-world distances from normalised depth
    # Depth map: 0=near, 1=far (or vice versa depending on model)
    # For foreground framing, the element is very close (~0.3-0.5m)
    fg_distance_m = 0.35  # Foreground element is very close to lens

    # Map focus_depth (0-1) to approximate real distance (1-10m range)
    # Higher depth value = farther away
    focus_distance_m = 1.0 + focus_depth * 9.0  # 1m to 10m

    base_blur, coc_mm = calculate_dof_blur(
        focal_length, aperture, focus_distance_m, fg_distance_m,
        element_img.width,
    )

    if blur_override is not None:
        base_blur = blur_override

    log(output_dir, f"DOF blur: focal={focal_length:.0f}mm f/{aperture:.1f} "
        f"focus={focus_distance_m:.1f}m fg={fg_distance_m:.1f}m "
        f"CoC={coc_mm:.2f}mm blur={base_blur}px")

    # Resize depth map to match element
    depth_resized = depth_map.resize(element_img.size, Image.LANCZOS)
    depth_arr = np.array(depth_resized).astype(np.float32) / 255.0

    # The element is in the foreground, so parts with lower depth values
    # (closer to camera) should get MORE blur. We modulate blur radius
    # based on how far each pixel is from the focus plane.
    # For simplicity and quality, apply blur in 3 passes at different radii
    alpha_arr = np.array(element_alpha).astype(np.float32) / 255.0

    # Determine depth range within the element area
    element_pixels = alpha_arr > 0.1
    if element_pixels.sum() > 0:
        mean_depth = float(np.mean(depth_arr[element_pixels]))
    else:
        mean_depth = 0.2

    # Distance from focus plane determines blur strength
    # Foreground is closer than focus → large blur
    depth_diff = abs(focus_depth - mean_depth)
    # Scale: further from focus = more blur
    blur_scale = max(0.5, min(2.0, 0.5 + depth_diff * 3.0))

    final_blur = int(min(80, max(5, base_blur * blur_scale)))
    log(output_dir, f"Depth-modulated blur: mean_element_depth={mean_depth:.2f} "
        f"focus_depth={focus_depth:.2f} scale={blur_scale:.2f} final={final_blur}px")

    # Apply multi-pass blur for smoother result
    blurred = element_img.copy()
    # First pass: main DOF blur
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=final_blur))
    # Second lighter pass for extra softness (mimics lens aberration)
    extra = max(1, final_blur // 4)
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=extra))

    return blurred, final_blur


# ---------------------------------------------------------------------------
# Color matching for generated element
# ---------------------------------------------------------------------------
def match_element_colors(original, element_img, element_alpha, edge_mask, darken, output_dir):
    """Shift the generated element's colors toward the original photo's edge tones.

    Also applies darkening to keep the framing subtle.
    Returns color-matched + darkened RGB image.
    """
    # Sample average color from original's edge regions
    sample_mask = create_edge_mask(original.width, original.height,
                                   coverage=0.15, sides="all", irregularity=0)
    edge_stat = ImageStat.Stat(original, mask=sample_mask)
    target_mean = edge_stat.mean[:3]

    # Get mean of element (only where alpha > 0)
    alpha_arr = np.array(element_alpha)
    element_mask_pil = Image.fromarray((alpha_arr > 30).astype(np.uint8) * 255, "L")

    try:
        elem_stat = ImageStat.Stat(element_img, mask=element_mask_pil)
        current_mean = elem_stat.mean[:3]
    except Exception:
        current_mean = [128, 128, 128]

    # Shift channels 30% toward scene edge colors
    result = element_img.copy()
    r, g, b = result.split()

    def shift_channel(ch, current, target):
        diff = target - current
        shift = int(diff * 0.3)
        return ch.point(lambda p: max(0, min(255, p + shift)))

    r = shift_channel(r, current_mean[0], target_mean[0])
    g = shift_channel(g, current_mean[1], target_mean[1])
    b = shift_channel(b, current_mean[2], target_mean[2])
    result = Image.merge("RGB", (r, g, b))

    log(output_dir, f"Color shift: element [{current_mean[0]:.0f},{current_mean[1]:.0f},{current_mean[2]:.0f}] "
        f"-> scene [{target_mean[0]:.0f},{target_mean[1]:.0f},{target_mean[2]:.0f}] (30% blend)")

    # Darken
    result = ImageEnhance.Brightness(result).enhance(darken)
    log(output_dir, f"Darkened element by factor {darken:.2f}")

    return result


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def composite_element(original, element_img, element_alpha, subject_mask, edge_mask, output_dir):
    """Composite the foreground element over the original photo.

    The element only appears where:
      - edge_mask is white (edge regions)
      - subject_mask is black (not covering the subject)
    """
    w, h = original.size

    # Ensure all images are same size
    if element_img.size != (w, h):
        element_img = element_img.resize((w, h), Image.LANCZOS)
    if element_alpha.size != (w, h):
        element_alpha = element_alpha.resize((w, h), Image.LANCZOS)
    if subject_mask.size != (w, h):
        subject_mask = subject_mask.resize((w, h), Image.LANCZOS)
    if edge_mask.size != (w, h):
        edge_mask = edge_mask.resize((w, h), Image.LANCZOS)

    # Combine masks: element_alpha AND edge_mask AND NOT subject_mask
    alpha_arr = np.array(element_alpha).astype(np.float32) / 255.0
    edge_arr = np.array(edge_mask).astype(np.float32) / 255.0
    subject_arr = np.array(subject_mask).astype(np.float32) / 255.0

    # Invert subject mask (we want to place element where subject is NOT)
    not_subject = 1.0 - subject_arr

    # Combine: element visible where it has alpha AND we want framing AND subject is absent
    final_alpha = alpha_arr * edge_arr * not_subject
    final_alpha = np.clip(final_alpha, 0, 1)

    # Soften the composite mask edges
    soft_blur = max(3, int(min(w, h) * 0.005))
    final_alpha_img = Image.fromarray((final_alpha * 255).astype(np.uint8), "L")
    final_alpha_img = final_alpha_img.filter(ImageFilter.GaussianBlur(radius=soft_blur))

    # Composite
    result = original.copy()
    result.paste(element_img, mask=final_alpha_img)

    composite_coverage = np.mean(np.array(final_alpha_img) > 30) * 100
    log(output_dir, f"Composite coverage: {composite_coverage:.1f}% of image")

    return result, final_alpha_img


# ---------------------------------------------------------------------------
# Gemini Evaluation
# ---------------------------------------------------------------------------
_EVAL_PROMPT = """\
You are a professional photography director evaluating a photo with added foreground framing.
If you see TWO images, the first is the ORIGINAL and the second has FOREGROUND FRAMING added.

Evaluate the FRAMED image on these criteria:
1. Does the foreground framing look natural, like something genuinely close to the camera lens?
2. Is the blur convincing — does it look like real shallow depth-of-field bokeh?
3. Does the framing enhance the composition by drawing attention to the subject?
4. Is the color palette of the framing consistent with the overall photo?
5. Does the framing add genuine depth to the image?
6. Is the subject still clearly visible and not obscured by the framing?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "framing_too_sharp", "framing_too_thick", "subject_obscured", \
"color_mismatch", "looks_artificial", "too_dark", "too_bright", "wrong_perspective", "no_depth_effect">],
  "adjustments": {
    "coverage": <null or suggested float 0.1-0.4>,
    "blur_more": <true if framing needs more blur>,
    "darken_more": <true if framing should be darker>,
    "try_different_seed": <true/false>,
    "suggestion": "<one sentence about what to change>"
  }
}"""


def evaluate_with_gemini(img, output_dir, original_img=None):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping Gemini evaluation")
        return None
    try:
        img_b64 = _img_to_b64(img, max_size=1024)
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]
        if original_img is not None:
            orig_b64 = _img_to_b64(original_img, max_size=1024)
            parts.insert(0, {"text": "ORIGINAL (no framing):"})
            parts.insert(1, {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"text": "WITH FOREGROUND FRAMING:\n\n" + _EVAL_PROMPT})
        else:
            parts.append({"text": _EVAL_PROMPT})

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
            json=payload, timeout=60)

        if response.status_code != 200:
            log(output_dir, f"Gemini API error ({response.status_code}): {response.text[:200]}", "WARN")
            return None

        resp_json = response.json()
        candidates = resp_json.get("candidates", [])
        if not candidates:
            reason = resp_json.get("promptFeedback", {}).get("blockReason", "unknown")
            log(output_dir, f"Gemini returned no candidates (reason: {reason})", "WARN")
            return None

        finish_reason = candidates[0].get("finishReason", "")
        parts_out = candidates[0].get("content", {}).get("parts", [])
        if not parts_out:
            log(output_dir, f"Gemini has no content parts (finishReason: {finish_reason})", "WARN")
            return None

        raw = parts_out[0].get("text", "").strip()
        log(output_dir, f"Gemini raw ({len(raw)} chars, finishReason={finish_reason}): {raw[:500]}")

        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                log(output_dir, f"Gemini: no JSON object: {raw[:200]}", "WARN")
                return None
            try:
                result = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as e:
                log(output_dir, f"Gemini JSON parse failed: {e}", "WARN")
                return None

        score = result.get("score", "?")
        critique = result.get("critique", "")
        issues = result.get("issues", [])
        log(output_dir, f"Gemini score: {score}/10 — {critique}")
        if issues:
            log(output_dir, f"Gemini issues: {', '.join(issues)}")
        adjustments = result.get("adjustments", {})
        if adjustments.get("suggestion"):
            log(output_dir, f"Gemini suggests: {adjustments['suggestion']}")
        return result
    except Exception as e:
        log(output_dir, f"Gemini evaluation failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Foreground Framing — add blurry foreground depth")
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--framing", default="auto", help="Framing preset name, or 'auto' for Gemini scene detection (default: auto)")
    parser.add_argument("--prompt", default=None, help="Custom framing prompt (overrides preset)")
    parser.add_argument("--negative", default=None, help="Custom negative prompt")
    parser.add_argument("--coverage", type=float, default=0.20, help="How much of the edge is framed (0.1-0.4, default: 0.20)")
    parser.add_argument("--sides", choices=["left-right", "top-bottom", "all", "auto", "smart"], default="smart",
                        help="Which sides to frame (default: smart — L-shaped based on subject position)")
    parser.add_argument("--blur-radius", type=int, default=None, help="Override DOF blur radius (default: auto from EXIF + depth)")
    parser.add_argument("--darken", type=float, default=0.55, help="Darken framing factor (0.0=black, 1.0=no darkening, default: 0.55)")
    parser.add_argument("--irregularity", type=float, default=0.5, help="Edge irregularity (0=straight, 1=very jagged, default: 0.5)")
    parser.add_argument("--focal-length", type=float, default=None, help="Override focal length in mm (default: from EXIF or 50mm)")
    parser.add_argument("--aperture", type=float, default=None, help="Override aperture f-number (default: from EXIF or f/2.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--auto-correct", action="store_true", help="Enable Gemini evaluation")
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")
    parser.add_argument("--list-presets", action="store_true", help="List all framing presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{'Preset':<18} Description")
        print("=" * 65)
        for name, preset in FRAMING_PRESETS.items():
            print(f"  {name:<16} {preset['description']}")
        print(f"\nTotal: {len(FRAMING_PRESETS)} presets")
        sys.exit(0)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Resolve prompt
    framing_prompt = None
    framing_negative = None
    framing_name = None

    if args.prompt:
        framing_prompt = args.prompt
        framing_negative = args.negative or "person, face, text, bright background, white background"
        framing_name = "Custom"
    elif args.framing and args.framing != "auto":
        if args.framing not in FRAMING_PRESETS:
            print(f"ERROR: Unknown preset '{args.framing}'. Use --list-presets.")
            sys.exit(1)
        preset = FRAMING_PRESETS[args.framing]
        framing_prompt = preset["prompt"]
        framing_negative = args.negative or preset.get("negative", "")
        framing_name = args.framing
    elif args.framing == "auto" or args.framing is None:
        framing_name = "auto"
    else:
        print("ERROR: Must specify --framing <preset>, --framing auto, or --prompt '<custom>'")
        sys.exit(1)

    # Derive names
    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    framing_tag = (framing_name or "auto").replace(" ", "_")[:20]
    suffix = random.randint(10, 99)
    folder_name = f"{model_name}_{source_basename}_{timestamp}_frame_{framing_tag}_{suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    timings = {}

    log(output_dir, "=" * 60)
    log(output_dir, "FOREGROUND FRAMING WORKFLOW START")
    log(output_dir, f"Source:         {source}")
    log(output_dir, f"Framing:        {framing_name}")
    log(output_dir, f"Coverage:       {args.coverage}")
    log(output_dir, f"Sides:          {args.sides}")
    log(output_dir, f"Darken:         {args.darken}")
    log(output_dir, f"Irregularity:   {args.irregularity}")
    log(output_dir, f"Seed:           {seed}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    img_orig = Image.open(source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), "JPEG", quality=95)

    # ========================================================================
    # Step 1: Analyze photo (EXIF, scene, subject mask, depth map)
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 1/6: Analyze photo ---")

    # 1a. EXIF extraction
    exif_info = extract_exif(source, output_dir)
    if args.focal_length is not None:
        exif_info["focal_length_mm"] = args.focal_length
        log(output_dir, f"Focal length overridden to {args.focal_length:.0f}mm")
    if args.aperture is not None:
        exif_info["aperture"] = args.aperture
        log(output_dir, f"Aperture overridden to f/{args.aperture:.1f}")

    # 1b. Auto-detect framing prompt if needed
    scene_context = None
    if framing_name == "auto":
        log(output_dir, "Auto-detecting scene for framing prompt (Gemini)...")
        scene_context = suggest_framing_prompt(source, output_dir)
        if scene_context and scene_context.get("prompt"):
            framing_prompt = scene_context["prompt"]
            framing_negative = args.negative or scene_context.get("negative", "person, face, text, bright background")
            framing_name = scene_context.get("object", "auto")[:20]
            log(output_dir, f"Auto framing: '{framing_name}' — {framing_prompt[:80]}")
        else:
            log(output_dir, "Scene detection failed — falling back to 'foliage' preset", "WARN")
            preset = FRAMING_PRESETS["foliage"]
            framing_prompt = preset["prompt"]
            framing_negative = preset["negative"]
            framing_name = "foliage (fallback)"

    # 1c. Subject mask via shared masking module (BiRefNet)
    log(output_dir, "Extracting subject mask (BiRefNet via masking module)...")
    subject_mask, mask_info = build_mask(
        source, affect="subject", exclude="", output_dir=output_dir,
    )
    subject_mask.save(os.path.join(output_dir, "1_subject_mask.png"))
    log(output_dir, f"Subject mask: {mask_info['engine']}, coverage={mask_info['coverage_pct']:.1f}%")

    # 1d. Smart side detection
    smart_sides = None
    if args.sides == "smart":
        smart_sides = detect_smart_sides(subject_mask, img_orig.width, img_orig.height, output_dir)
        if smart_sides is None:
            log(output_dir, "Smart detection failed — falling back to auto sides", "WARN")

    # 1e. Depth estimation
    depth_map = run_depth_estimation(source, output_dir)
    focus_depth = 0.5  # default mid-range
    if depth_map is not None:
        if depth_map.size != img_orig.size:
            depth_map = depth_map.resize(img_orig.size, Image.LANCZOS)
        depth_map.save(os.path.join(output_dir, "1_depth.png"))
        focus_depth = get_focus_distance_from_depth(depth_map, subject_mask, output_dir)
    else:
        log(output_dir, "Depth estimation failed — using default focus depth 0.5", "WARN")
        # Create a flat depth map as fallback
        depth_map = Image.new("L", img_orig.size, 128)

    timings["analyze"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['analyze']:.1f}s)")

    # ========================================================================
    # Step 2: Generate foreground element
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 2/6: Generate foreground element ---")

    element_raw = generate_element(
        framing_prompt, framing_negative,
        img_orig.width, img_orig.height, output_dir,
        seed=seed, scene_context=scene_context,
        smart_sides=smart_sides, exif_info=exif_info,
    )
    if element_raw is None:
        log(output_dir, "Element generation failed — cannot proceed", "ERROR")
        sys.exit(1)

    if element_raw.size != img_orig.size:
        log(output_dir, f"Resizing element {element_raw.size} -> {img_orig.size}")
        element_raw = element_raw.resize(img_orig.size, Image.LANCZOS)

    element_raw.save(os.path.join(output_dir, "2_element_raw.jpg"), "JPEG", quality=95)
    timings["generate"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['generate']:.1f}s)")

    # ========================================================================
    # Step 3: Extract element alpha
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 3/6: Extract element alpha ---")

    element_alpha = extract_element_alpha(element_raw, output_dir)
    element_alpha.save(os.path.join(output_dir, "2_element_alpha.png"))

    timings["alpha"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['alpha']:.1f}s)")

    # ========================================================================
    # Step 4: Depth-aware DOF blur
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 4/6: Depth-aware DOF blur ---")

    element_blurred, blur_radius = apply_dof_blur(
        element_raw, element_alpha, depth_map, focus_depth,
        exif_info, output_dir, blur_override=args.blur_radius,
    )
    element_blurred.save(os.path.join(output_dir, "3_element_blurred.jpg"), "JPEG", quality=95)

    timings["blur"] = time.time() - t0
    log(output_dir, f"Step 4 done ({timings['blur']:.1f}s)")

    # ========================================================================
    # Step 5: Color match + darken
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 5/6: Color match + darken ---")

    edge_mask = create_edge_mask(
        img_orig.width, img_orig.height,
        coverage=args.coverage,
        sides=args.sides if args.sides != "smart" else "auto",
        irregularity=args.irregularity,
        smart_sides=smart_sides,
    )
    edge_mask.save(os.path.join(output_dir, "1_edge_mask.png"))
    mask_coverage = np.array(edge_mask).mean() / 255.0
    log(output_dir, f"Edge mask coverage: {mask_coverage*100:.1f}% of image")

    element_colored = match_element_colors(
        img_orig, element_blurred, element_alpha, edge_mask,
        args.darken, output_dir,
    )
    element_colored.save(os.path.join(output_dir, "4_element_colored.jpg"), "JPEG", quality=95)

    timings["color"] = time.time() - t0
    log(output_dir, f"Step 5 done ({timings['color']:.1f}s)")

    # ========================================================================
    # Step 6: Composite + evaluate + output
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 6/6: Composite + evaluate + output ---")

    # Also blur the alpha to match the element blur
    alpha_blurred = element_alpha.filter(ImageFilter.GaussianBlur(radius=max(3, blur_radius // 2)))

    final_img, composite_mask = composite_element(
        img_orig, element_colored, alpha_blurred,
        subject_mask, edge_mask, output_dir,
    )
    composite_mask.save(os.path.join(output_dir, "5_composite_mask.png"))

    final_path = os.path.join(output_dir, "6_framed_final.jpg")
    final_img.save(final_path, "JPEG", quality=95)

    quality_final = check_image_quality(final_img, "FINAL", output_dir)

    # Evaluate
    eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

    # Copy to finals
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared")
    finals_dir = os.path.join(local_out, "finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_name = os.path.basename(output_dir) + ".jpg"
    finals_dest = os.path.join(finals_dir, finals_name)
    with open(final_path, "rb") as f_in:
        with open(finals_dest, "wb") as f_out:
            f_out.write(f_in.read())
    log(output_dir, f"Final copied to: {finals_dest}")

    # Copy script for reproducibility
    try:
        shutil.copy2(os.path.abspath(__file__), os.path.join(output_dir, f"workflow_script_{os.path.basename(__file__)}"))
    except Exception:
        pass

    timings["output"] = time.time() - t0
    log(output_dir, f"Step 6 done ({timings['output']:.1f}s)")

    # --- Summary ---
    total = sum(timings.values())
    score_str = f"{eval_result['score']}/10 — {eval_result.get('critique', '')}" if eval_result else "N/A"

    print(f"""
============================================================
  FOREGROUND FRAMING SUMMARY
============================================================
  Source:          {source}
  Framing:         {framing_name}
  Coverage:        {args.coverage}
  Sides:           {args.sides}
  DOF Blur:        {blur_radius}px (focal={exif_info['focal_length_mm']:.0f}mm f/{exif_info['aperture']:.1f})
  Darken:          {args.darken}
  Seed:            {seed}

  Step Timings:
    1. Analyze photo          {timings.get('analyze', 0):>8.1f}s
    2. Generate element       {timings.get('generate', 0):>8.1f}s
    3. Extract alpha          {timings.get('alpha', 0):>8.1f}s
    4. DOF blur               {timings.get('blur', 0):>8.1f}s
    5. Color match + darken   {timings.get('color', 0):>8.1f}s
    6. Composite + output     {timings.get('output', 0):>8.1f}s
    TOTAL                     {total:>8.1f}s

  Quality Report:
    Final image:   brightness={quality_final['brightness']}  contrast={quality_final['contrast']}  entropy={quality_final['entropy']}
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}
============================================================""")


if __name__ == "__main__":
    main()
