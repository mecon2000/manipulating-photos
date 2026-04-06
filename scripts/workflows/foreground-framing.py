#!/home/rong/openclaw-venv/bin/python3
"""
Foreground Framing Workflow

Adds blurry foreground elements to the edges of a photo, simulating the
"shoot-through" technique (shooting through foliage, doorframes, fabric, etc.)
at shallow depth of field (f/1.4-2.8, 35-50mm lens).

Creates 3D depth: sharp subject in the middle, blurry abstract foreground
framing the edges. The foreground color palette is matched to the photo.

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

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Framing Presets
# ---------------------------------------------------------------------------
FRAMING_PRESETS = {
    "foliage": {
        "prompt": "out of focus green leaves and branches very close to camera, soft bokeh foliage, natural organic shapes, dappled light through leaves",
        "negative": "sharp, in focus, text, face, person, hand",
        "description": "Blurry green leaves/branches framing the shot",
    },
    "warm foliage": {
        "prompt": "out of focus warm autumn leaves close to camera, golden brown orange foliage bokeh, soft organic shapes, warm tones",
        "negative": "sharp, in focus, text, face, person, green",
        "description": "Warm autumn-toned blurry leaves",
    },
    "doorframe": {
        "prompt": "out of focus dark wooden doorframe very close to camera, warm wood grain texture, architectural framing element, shallow depth of field",
        "negative": "sharp, in focus, text, face, person, hand",
        "description": "Dark wooden doorframe edges",
    },
    "curtain": {
        "prompt": "out of focus sheer curtain fabric very close to camera, soft translucent white fabric, flowing textile, shallow depth of field, dreamy",
        "negative": "sharp, in focus, text, face, person, opaque",
        "description": "Soft sheer curtain fabric",
    },
    "dark curtain": {
        "prompt": "out of focus dark velvet curtain fabric very close to camera, rich dark textile, heavy draping, theatrical, shallow depth of field",
        "negative": "sharp, in focus, text, face, person, bright",
        "description": "Dark velvet curtain draping",
    },
    "flowers": {
        "prompt": "out of focus colorful flower petals very close to camera, soft bokeh blossoms, delicate petals, shallow depth of field, romantic",
        "negative": "sharp, in focus, text, face, person, stem",
        "description": "Blurry flower petals framing",
    },
    "fairy lights": {
        "prompt": "out of focus warm fairy lights bokeh circles very close to camera, golden bokeh balls, string lights, warm glowing orbs, shallow depth of field",
        "negative": "sharp, in focus, text, face, person",
        "description": "Warm bokeh light circles",
    },
    "metal": {
        "prompt": "out of focus dark iron railing or metal bars very close to camera, industrial metal framing, shallow depth of field, dark tones",
        "negative": "sharp, in focus, text, face, person, bright",
        "description": "Dark metal railing/bars",
    },
    "smoke": {
        "prompt": "out of focus wispy smoke or haze very close to camera, ethereal fog, atmospheric mist, shallow depth of field, mysterious",
        "negative": "sharp, in focus, text, face, person, fire",
        "description": "Ethereal smoke/haze framing",
    },
    "brick": {
        "prompt": "out of focus red brick wall very close to camera, warm masonry texture, urban architectural element, shallow depth of field",
        "negative": "sharp, in focus, text, face, person",
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


# ---------------------------------------------------------------------------
# Subject detection for smart side selection
# ---------------------------------------------------------------------------
def run_fal_birefnet(image_path, output_dir):
    """Extract foreground mask using BiRefNet."""
    log(output_dir, "Detecting subject position (BiRefNet)...")
    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    try:
        response = requests.post("https://fal.run/fal-ai/birefnet", headers=headers,
            json={"image_url": f"data:image/jpeg;base64,{img_b64}"}, timeout=180)
    except requests.RequestException as e:
        log(output_dir, f"BiRefNet request failed: {e}", "WARN")
        return None
    if response.status_code != 200:
        log(output_dir, f"BiRefNet failed ({response.status_code})", "WARN")
        return None
    data = response.json()
    result_url = data["image"]["url"]
    result_img = Image.open(requests.get(result_url, stream=True, timeout=30).raw)
    if result_img.mode == "RGBA":
        return result_img.split()[3]
    return result_img.convert("L")


def detect_smart_sides(subject_mask, img_width, img_height, output_dir):
    """Analyze subject position to pick 2 adjacent sides for L-shaped framing.

    Returns a dict with: primary side (more coverage), secondary side (less coverage),
    and coverage multipliers for each.

    Logic:
    - Subject right of center → frame LEFT (more lead room on left)
    - Subject above center → frame BOTTOM
    - Pick the 2 adjacent sides with the most space
    - Primary side gets full coverage, secondary gets 60% coverage
    """
    mask_np = np.array(subject_mask)
    # Threshold
    binary = (mask_np > 127).astype(np.float32)

    if binary.sum() < 100:
        log(output_dir, "Subject mask too small for smart detection — falling back to auto", "WARN")
        return None

    # Find centroid
    ys, xs = np.where(binary > 0)
    cx = xs.mean() / img_width   # 0-1, 0=left, 1=right
    cy = ys.mean() / img_height  # 0-1, 0=top, 1=bottom

    log(output_dir, f"Subject centroid: x={cx:.2f}, y={cy:.2f} (0=left/top, 1=right/bottom)")

    # Determine horizontal framing side (opposite to where subject is pushed)
    # Subject is right → more space on left → frame left
    h_side = "left" if cx > 0.5 else "right"
    h_offset = abs(cx - 0.5)  # How far off-center (0=centered, 0.5=at edge)

    # Determine vertical framing side
    v_side = "top" if cy > 0.5 else "bottom"
    v_offset = abs(cy - 0.5)

    # Primary side = the one with more space (larger offset)
    if h_offset >= v_offset:
        primary, secondary = h_side, v_side
        primary_mult, secondary_mult = 1.0, 0.6
    else:
        primary, secondary = v_side, h_side
        primary_mult, secondary_mult = 1.0, 0.6

    log(output_dir, f"Smart framing: primary={primary} (×{primary_mult}), secondary={secondary} (×{secondary_mult})")

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

Look at the scene: the environment, setting, objects present, indoor/outdoor, lighting, mood.

Suggest ONE specific foreground object that would look natural if it were very close to the camera lens \
and heavily out of focus. This object should:
1. Be something that BELONGS in this scene (e.g., leaves for outdoor, curtain for indoor, bottle for bathroom)
2. Be dark or semi-transparent when blurred
3. Add depth without distracting from the subject

Respond ONLY with valid JSON:
{
  "object": "<the object, e.g. 'green leaves and small twigs', 'dark wooden doorframe', 'frosted glass bottle'>",
  "prompt": "<inpainting prompt: 'out of focus [object] very close to camera, shallow depth of field, [details]'>",
  "negative": "<what to avoid: 'sharp, in focus, text, face, person, [scene-specific]'>",
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
        # Parse JSON
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
        return result

    except Exception as e:
        log(output_dir, f"Scene analysis failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Edge mask generation
# ---------------------------------------------------------------------------
def create_edge_mask(width, height, coverage=0.20, sides="auto", irregularity=0.4, smart_sides=None):
    """Create an organic-looking edge mask for framing.

    White = areas to inpaint (foreground framing).
    Black = areas to keep (subject region).

    Args:
        coverage: fraction of image covered by framing (0.1-0.4)
        sides: "left-right", "top-bottom", "all", "auto", or "smart"
        irregularity: how jagged the inner edge is (0=straight, 1=very jagged)
        smart_sides: dict from detect_smart_sides() with primary/secondary sides + multipliers
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Smart mode: L-shaped framing on 2 adjacent sides
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
        # Draw base rectangle
        draw.rectangle([x1, y1, x2, y2], fill=255)

    # Add irregular inner edge by drawing random black circles along the border
    # This makes the framing look organic, not like a perfect rectangle
    if irregularity > 0:
        np.random.seed(42)  # Reproducible but organic
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
                # Erase (black) circles to create irregular edge
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)

        # Also add some white blobs extending inward for organic feel
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

    # Soften the mask edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.015))
    # Re-threshold to keep it mostly binary but with soft edges
    mask = mask.point(lambda p: 255 if p > 100 else (int(p * 2.55) if p > 40 else 0))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.01))

    return mask


# ---------------------------------------------------------------------------
# Color matching
# ---------------------------------------------------------------------------
def match_framing_colors(original, inpainted, mask, darken=0.6):
    """Match the inpainted framing's color palette to the original photo's edges.

    Samples colors from the original's border regions and shifts the inpainted
    areas to match. Also darkens the framing to keep focus on the subject.
    """
    # Sample average color from original's edge regions
    edge_mask = create_edge_mask(original.width, original.height, coverage=0.15, sides="all", irregularity=0)
    edge_stat = ImageStat.Stat(original, mask=edge_mask)
    target_mean = edge_stat.mean[:3]  # RGB

    # Get current mean of inpainted areas
    inpaint_stat = ImageStat.Stat(inpainted, mask=mask)
    current_mean = inpaint_stat.mean[:3]

    # Calculate shift
    result = inpainted.copy()
    r, g, b = result.split()

    def shift_channel(ch, current, target):
        diff = target - current
        # Light blend toward scene colors — keep most of the generated character
        # (was 0.6 but that washed out doorframe/wood tones into generic scene colors)
        shift = int(diff * 0.3)
        return ch.point(lambda p: max(0, min(255, p + shift)))

    r = shift_channel(r, current_mean[0], target_mean[0])
    g = shift_channel(g, current_mean[1], target_mean[1])
    b = shift_channel(b, current_mean[2], target_mean[2])
    result = Image.merge("RGB", (r, g, b))

    # Darken the framing to keep it subtle
    darkened = ImageEnhance.Brightness(result).enhance(darken)

    # Only apply darkening in the masked area
    soft_mask = mask.convert("L")
    final = original.copy()
    final.paste(darkened, mask=soft_mask)

    return final


# ---------------------------------------------------------------------------
# Inpainting via fal.ai
# ---------------------------------------------------------------------------
def run_inpaint(image, mask, prompt, negative_prompt, output_dir,
                guidance_scale=9.0, steps=30, seed=None, strength=0.95):
    """Inpaint masked areas with text-guided content using fal.ai SDXL inpainting."""
    log(output_dir, f"Inpainting: '{prompt[:80]}...' (guidance={guidance_scale}, steps={steps}, strength={strength})")

    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}

    img_b64 = _img_to_b64_simple(image)
    mask_b64 = _img_to_b64_simple(mask.convert("RGB"), fmt="PNG", quality=100)

    payload = {
        "model_name": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "prompt": prompt,
        "negative_prompt": negative_prompt or "sharp, in focus, text, watermark",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "mask_url": f"data:image/png;base64,{mask_b64}",
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "strength": strength,  # How much to replace (0=keep original, 1=fully replace)
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post("https://fal.run/fal-ai/inpaint", headers=headers,
                                 json=payload, timeout=300)
    except requests.RequestException as e:
        log(output_dir, f"Inpainting request failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"Inpainting failed ({response.status_code}): {response.text[:300]}", "ERROR")
        return None

    data = response.json()
    log(output_dir, f"Inpaint response keys: {list(data.keys())}")

    # fal-ai/inpaint may return "image" (singular) or "images" (plural)
    images = data.get("images", [])
    if not images and "image" in data:
        images = [data["image"]]
    if not images:
        log(output_dir, f"Inpainting returned no images. Response: {json.dumps(data)[:500]}", "ERROR")
        return None

    result_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
    if not result_url:
        log(output_dir, "Inpainting returned no image URL", "ERROR")
        return None

    log(output_dir, f"Inpaint CDN URL: {result_url}")
    result_img = Image.open(requests.get(result_url, stream=True, timeout=30).raw).convert("RGB")
    log(output_dir, f"Inpaint result: {result_img.size[0]}x{result_img.size[1]}")
    return result_img


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
    parser.add_argument("--blur-radius", type=int, default=None, help="Gaussian blur radius for framing (default: auto based on image size)")
    parser.add_argument("--darken", type=float, default=0.55, help="Darken framing factor (0.0=black, 1.0=no darkening, default: 0.55)")
    parser.add_argument("--irregularity", type=float, default=0.5, help="Edge irregularity (0=straight, 1=very jagged, default: 0.5)")
    parser.add_argument("--guidance-scale", type=float, default=9.0, help="Inpainting guidance scale (default: 9.0)")
    parser.add_argument("--steps", type=int, default=30, help="Inpainting steps (default: 30)")
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
        framing_negative = args.negative or "sharp, in focus, text, face, person"
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
        # Will be resolved after output_dir is created (needs Gemini)
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
    blur_radius = args.blur_radius
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

    # --- Step 0: Auto-detect framing prompt if needed ---
    if framing_name == "auto":
        log(output_dir, "--- Auto-detecting scene for framing prompt (Gemini) ---")
        scene = suggest_framing_prompt(source, output_dir)
        if scene and scene.get("prompt"):
            framing_prompt = scene["prompt"]
            framing_negative = args.negative or scene.get("negative", "sharp, in focus, text, face, person")
            framing_name = scene.get("object", "auto")[:20]
            log(output_dir, f"Auto framing: '{framing_name}' — {framing_prompt[:80]}")
        else:
            log(output_dir, "Scene detection failed — falling back to 'foliage' preset", "WARN")
            preset = FRAMING_PRESETS["foliage"]
            framing_prompt = preset["prompt"]
            framing_negative = preset["negative"]
            framing_name = "foliage (fallback)"

    # --- Step 1: Create edge mask ---
    t0 = time.time()
    log(output_dir, "--- Step 1/4: Create edge mask ---")

    # Smart side detection using BiRefNet
    smart_sides = None
    if args.sides == "smart":
        subject_mask = run_fal_birefnet(source, output_dir)
        if subject_mask is not None:
            # Resize mask to match image
            if subject_mask.size != img_orig.size:
                subject_mask = subject_mask.resize(img_orig.size, Image.LANCZOS)
            subject_mask.save(os.path.join(output_dir, "1_subject_mask.png"))
            smart_sides = detect_smart_sides(subject_mask, img_orig.width, img_orig.height, output_dir)
        if smart_sides is None:
            log(output_dir, "Smart detection failed — falling back to auto sides", "WARN")

    mask = create_edge_mask(img_orig.width, img_orig.height,
                            coverage=args.coverage,
                            sides=args.sides if args.sides != "smart" else "auto",
                            irregularity=args.irregularity,
                            smart_sides=smart_sides)
    mask.save(os.path.join(output_dir, "1_edge_mask.png"))
    mask_coverage = np.array(mask).mean() / 255.0
    log(output_dir, f"Edge mask coverage: {mask_coverage*100:.1f}% of image")
    timings["mask"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['mask']:.1f}s)")

    # --- Step 2: Inpaint foreground elements ---
    t0 = time.time()
    log(output_dir, "--- Step 2/4: Inpaint foreground ---")
    inpainted = run_inpaint(img_orig, mask, framing_prompt, framing_negative,
                            output_dir, guidance_scale=args.guidance_scale,
                            steps=args.steps, seed=seed)
    if inpainted is None:
        log(output_dir, "Inpainting failed — cannot proceed", "ERROR")
        sys.exit(1)

    # Resize if needed
    if inpainted.size != img_orig.size:
        log(output_dir, f"Resizing inpainted {inpainted.size} -> {img_orig.size}")
        inpainted = inpainted.resize(img_orig.size, Image.LANCZOS)

    inpainted.save(os.path.join(output_dir, "2_inpainted_raw.jpg"), "JPEG", quality=95)
    timings["inpaint"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['inpaint']:.1f}s)")

    # --- Step 3: Blur + color match + composite ---
    t0 = time.time()
    log(output_dir, "--- Step 3/4: Blur + color match + composite ---")

    # Auto blur radius: ~2-3% of the longest edge, capped at 60px
    # Too much blur makes shapes look like abstract blobs rather than real objects
    if blur_radius is None:
        blur_radius = max(15, min(60, int(max(img_orig.width, img_orig.height) * 0.025)))
    log(output_dir, f"Blur radius: {blur_radius}px")

    # Heavy blur on the inpainted result (only the framing areas will be used)
    blurred = inpainted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred.save(os.path.join(output_dir, "3a_blurred.jpg"), "JPEG", quality=95)

    # Color match + darken the framing
    final_img = match_framing_colors(img_orig, blurred, mask, darken=args.darken)
    final_path = os.path.join(output_dir, "3b_framed_final.jpg")
    final_img.save(final_path, "JPEG", quality=95)

    quality_final = check_image_quality(final_img, "FINAL", output_dir)
    timings["composite"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['composite']:.1f}s)")

    # --- Step 4: Evaluate + output ---
    t0 = time.time()
    log(output_dir, "--- Step 4/4: Evaluate + output ---")

    eval_result = None
    if args.auto_correct:
        eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)
    else:
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
    log(output_dir, f"Step 4 done ({timings['output']:.1f}s)")

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
  Blur:            {blur_radius}px
  Darken:          {args.darken}
  Seed:            {seed}

  Step Timings:
    1. Create edge mask       {timings.get('mask', 0):>8.1f}s
    2. Inpaint foreground     {timings.get('inpaint', 0):>8.1f}s
    3. Blur + composite       {timings.get('composite', 0):>8.1f}s
    4. Evaluate + output      {timings.get('output', 0):>8.1f}s
    TOTAL                     {total:>8.1f}s

  Quality Report:
    Final image:   brightness={quality_final['brightness']}  contrast={quality_final['contrast']}  entropy={quality_final['entropy']}
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}
============================================================""")


if __name__ == "__main__":
    main()
