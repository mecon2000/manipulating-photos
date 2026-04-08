#!/home/rong/openclaw-venv/bin/python3
"""
Time Corruption — Temporal Decay Effects for Art Photography

Simulates "time decay" / "temporal corruption" on photos: frozen motion,
image decomposition through time, ghosting, melting, motion trails, glitch.

Effects are applied to a body-part target specified by --affect:
  skin     — face-skin + body-skin (default; ropes/clothes excluded — shibari)
  subject  — whole foreground subject (BiRefNet)
  bg       — background only (subject stays sharp)
  all      — entire image
  face-skin, body-skin, hair, clothes, others  — MediaPipe fine-grained parts

Presets:
    ghost   — Multiple-exposure ghosting (arc offsets on skin targets)
    melt    — Diffusion melting / dissolving through a gradient
    trails  — Directional motion blur trailing from the subject
    glitch  — Chromatic aberration / channel shift
    full    — All effects layered: channel shift -> ghosting -> trails -> light melt

Usage:
    python time-corruption.py --source photo.jpg --effect ghost
    python time-corruption.py --source photo.jpg --effect ghost --affect skin
    python time-corruption.py --source photo.jpg --effect full --affect subject --intensity 0.5
    python time-corruption.py --source photo.jpg --effect melt --affect bg --direction 90
    python time-corruption.py --source photo.jpg --effect ghost --affect skin --exclude hands
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
import math
import shutil
import random
import base64
import argparse
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
from scipy.ndimage import convolve

# Shared masking module (same directory)
import importlib.util as _ilu
_masking_spec = _ilu.spec_from_file_location(
    "masking", os.path.join(os.path.dirname(os.path.abspath(__file__)), "masking.py")
)
masking = _ilu.module_from_spec(_masking_spec)
_masking_spec.loader.exec_module(masking)

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EFFECTS = ["ghost", "melt", "trails", "glitch", "echo", "full"]
MODES = ["normal", "dissolve", "float"]  # kept for backward-compat deprecation mapping
# Deprecated mode → affect mapping
_MODE_TO_AFFECT = {
    "normal":  "subject",
    "dissolve": "skin",
    "float":   "bg",
}

# Parts that need BiRefNet (API call, ~5s, excellent edges)
BIREFNET_PARTS = {"bg", "subject"}
# Parts that come from MediaPipe body-segment (includes "skin" shortcut)
BODY_SEGMENT_PARTS = {"face-skin", "body-skin", "skin", "hair", "clothes", "others"}

STEP_NAMES = {
    1: "Build affect mask",
    2: "Apply time-corruption effects",
    3: "Quality evaluation (Gemini)",
    4: "Output / upload",
}

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
# API helpers (kept for non-masking API calls, e.g. Gemini)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Quality utilities
# ---------------------------------------------------------------------------
def check_image_quality(img, label, output_dir):
    """Check if an image is degenerate (black, white, flat, zero-entropy)."""
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
    result = {
        "ok": ok,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "entropy": round(entropy, 2),
        "reason": "; ".join(reasons) if reasons else None,
    }
    if not ok:
        log(output_dir, f"QUALITY FAIL [{label}]: {result['reason']}", "WARN")
    else:
        log(output_dir, f"Quality OK [{label}]: brightness={brightness:.1f} contrast={contrast:.1f} entropy={entropy:.2f}")
    return result


def _img_to_b64(img, max_size=1024):
    """Downscale and encode image to base64 JPEG."""
    img_resized = img.copy()
    img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_EVAL_PROMPT = """\
You are an art director evaluating a photo that has been processed with temporal/motion \
art effects (ghosting, motion trails, chromatic aberration, diffusion melting). \
The goal is artistic expression — frozen motion, time decay, temporal corruption.

If you see TWO images, the first is the ORIGINAL and the second is the PROCESSED result.

Evaluate the PROCESSED image on:
1. Overall artistic impact — does the temporal effect create a compelling visual?
2. Subject readability — is the subject still recognizable and visually anchored?
3. Effect balance — are the effects noticeable but not destroying the image? Too subtle? Too much?
4. Color harmony — do the chromatic shifts (if any) add to the aesthetic or create ugly artifacts?
5. Composition — does the motion/decay direction enhance or fight the composition?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "too_subtle", "too_extreme", "subject_lost", \
"ugly_artifacts", "colors_clash", "too_dark", "too_bright", "low_contrast", \
"direction_wrong", "effect_uneven">],
  "adjustments": {
    "intensity": <null or suggested float 0.3-1.0>,
    "try_different_direction": <true/false>,
    "reduce_ghosting": <true/false>,
    "reduce_melt": <true/false>,
    "suggestion": "<one sentence about what to change>"
  }
}"""


def evaluate_with_gemini(img, output_dir, original_img=None):
    """Evaluate using Google Gemini Vision API."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping Gemini evaluation")
        return None

    try:
        img_b64 = _img_to_b64(img)

        parts = []
        if original_img is not None:
            orig_b64 = _img_to_b64(original_img)
            parts.append({"text": "ORIGINAL (before effects):"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
            parts.append({"text": "PROCESSED (after temporal effects):\n\n" + _EVAL_PROMPT})
        else:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
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
            json=payload,
            timeout=60,
        )

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
        content = candidates[0].get("content", {})
        parts_out = content.get("parts", [])
        if not parts_out:
            log(output_dir, f"Gemini candidate has no content parts (finishReason: {finish_reason})", "WARN")
            return None

        raw = parts_out[0].get("text", "").strip()
        log(output_dir, f"Gemini raw response ({len(raw)} chars): {raw[:500]}")

        # Strip markdown fences
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                log(output_dir, f"Gemini response contains no JSON object: {raw[:200]}", "WARN")
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
# Time Corruption Effects
# ---------------------------------------------------------------------------

def effect_ghosting(img_arr, mask_arr, intensity, direction_deg, rng, output_dir):
    """Create 3-5 ghostly copies of the subject, offset with decreasing opacity.

    Only the subject (defined by mask) is ghosted; the background stays clean.
    """
    log(output_dir, f"Applying ghosting effect (intensity={intensity:.2f}, direction={direction_deg} deg)")

    h, w = img_arr.shape[:2]
    result = img_arr.astype(np.float64)

    # Direction vector from angle
    rad = math.radians(direction_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    # Number of ghost copies scales with intensity (3-5)
    n_ghosts = int(3 + intensity * 2)
    opacities = [0.7, 0.5, 0.3, 0.15, 0.08][:n_ghosts]
    # Offset range scales with intensity AND image size
    short_edge = min(h, w)
    base_offset = short_edge * (0.01 + intensity * 0.02)  # 1-3% of image per step

    # Extract subject pixels using the mask
    mask_norm = mask_arr.astype(np.float64) / 255.0  # 0-1
    mask_3ch = mask_norm[:, :, np.newaxis]
    subject = img_arr.astype(np.float64) * mask_3ch

    for i, opacity in enumerate(opacities):
        offset_px = base_offset * (i + 1)
        # Add slight randomness to offset direction
        jitter_deg = rng.uniform(-15, 15)
        jitter_rad = math.radians(direction_deg + jitter_deg)
        ox = int(round(math.cos(jitter_rad) * offset_px))
        oy = int(round(math.sin(jitter_rad) * offset_px))

        # Shift subject and its mask
        shifted_subject = np.zeros_like(subject)
        shifted_mask = np.zeros_like(mask_norm)

        # Compute source and destination slices for the shift
        src_y_start = max(0, -oy)
        src_y_end = min(h, h - oy)
        src_x_start = max(0, -ox)
        src_x_end = min(w, w - ox)
        dst_y_start = max(0, oy)
        dst_y_end = min(h, h + oy)
        dst_x_start = max(0, ox)
        dst_x_end = min(w, w + ox)

        # Ensure matching dimensions
        copy_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        copy_w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        if copy_h <= 0 or copy_w <= 0:
            continue

        shifted_subject[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
            subject[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
        shifted_mask[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
            mask_norm[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]

        # Blend ghost onto result where shifted mask is active
        blend_mask = shifted_mask[:, :, np.newaxis] * opacity * intensity
        result = result * (1 - blend_mask) + shifted_subject * blend_mask + result * blend_mask * (1 - opacity)

    return np.clip(result, 0, 255).astype(np.uint8)


def effect_motion_trails(img_arr, mask_arr, intensity, direction_deg, output_dir):
    """Apply directional motion blur, stronger at edges via gradient mask.

    The motion blur is computed on the full image, then blended using a gradient
    mask that transitions from 0 (center) to 1 (edges) so trails appear to
    emanate outward.
    """
    log(output_dir, f"Applying motion trails (intensity={intensity:.2f}, direction={direction_deg} deg)")

    h, w = img_arr.shape[:2]

    # Create a directional motion blur kernel — scaled to image size
    short_edge = min(h, w)
    kernel_size = max(15, int(short_edge * (0.02 + intensity * 0.04)))  # 2-6% of image
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    # Draw a line through the center at the given angle
    cx, cy = kernel_size // 2, kernel_size // 2
    rad = math.radians(direction_deg)
    for i in range(kernel_size):
        t = i - cx
        kx = int(round(cx + t * math.cos(rad)))
        ky = int(round(cy + t * math.sin(rad)))
        if 0 <= kx < kernel_size and 0 <= ky < kernel_size:
            kernel[ky, kx] = 1.0

    # Normalize
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    # Apply motion blur to each channel
    blurred = np.zeros_like(img_arr, dtype=np.float64)
    for c in range(3):
        blurred[:, :, c] = convolve(img_arr[:, :, c].astype(np.float64), kernel, mode='reflect')

    # Create edge-weighted gradient mask: stronger at edges of frame
    # Distance from center, normalized to 0-1
    yy, xx = np.mgrid[0:h, 0:w]
    center_y, center_x = h / 2.0, w / 2.0
    dist = np.sqrt(((yy - center_y) / center_y) ** 2 + ((xx - center_x) / center_x) ** 2)
    # Normalize so center=0, corners~=1.0
    edge_mask = np.clip(dist / dist.max(), 0, 1)

    # Also weight by subject mask — trails extend from the subject area
    mask_norm = mask_arr.astype(np.float64) / 255.0
    # Dilate the mask substantially to create the trail area
    dilated = Image.fromarray((mask_norm * 255).astype(np.uint8))
    dilation_size = int(20 + intensity * 40)
    if dilation_size % 2 == 0:
        dilation_size += 1
    dilated = dilated.filter(ImageFilter.MaxFilter(min(dilation_size, 99)))
    dilated_arr = np.array(dilated).astype(np.float64) / 255.0

    # Combine: trail where dilated mask is active but original mask is not (the "trail zone")
    trail_zone = np.clip(dilated_arr - mask_norm * 0.3, 0, 1)
    blend_mask = (trail_zone * edge_mask * intensity)[:, :, np.newaxis]

    result = img_arr.astype(np.float64) * (1 - blend_mask) + blurred * blend_mask
    return np.clip(result, 0, 255).astype(np.uint8)


def effect_diffusion_melt(img_arr, mask_arr, intensity, direction_deg, output_dir):
    """Apply increasingly strong Gaussian blur through a directional gradient.

    The gradient controls where melting is strongest. Default (0 deg) = sharp at top,
    melting at bottom. 90 = sharp at left, melting at right. Etc.
    """
    log(output_dir, f"Applying diffusion melt (intensity={intensity:.2f}, direction={direction_deg} deg)")

    h, w = img_arr.shape[:2]

    # Create a directional gradient 0->1
    # direction_deg: 0 = melt toward bottom, 90 = melt toward right, 180 = melt toward top, 270 = melt toward left
    rad = math.radians(direction_deg)
    yy, xx = np.mgrid[0:h, 0:w]
    # Normalized coords centered at image center
    ny = (yy - h / 2.0) / (h / 2.0)  # -1 to 1
    nx = (xx - w / 2.0) / (w / 2.0)
    # Project onto direction vector: how far along the melt direction
    gradient = ny * math.cos(rad) + nx * math.sin(rad)
    # Shift to 0-1 range
    gradient = (gradient + 1.0) / 2.0
    gradient = np.clip(gradient, 0, 1)

    # Combine gradient with subject mask — melt the subject, not the BG
    mask_norm = mask_arr.astype(np.float64) / 255.0
    # Weight: the melt effect is strongest where both gradient and mask are high
    # But also let it bleed slightly beyond the mask
    dilated = Image.fromarray((mask_norm * 255).astype(np.uint8))
    dilated = dilated.filter(ImageFilter.GaussianBlur(radius=int(10 + intensity * 15)))
    dilated_arr = np.array(dilated).astype(np.float64) / 255.0
    melt_weight = gradient * np.maximum(mask_norm, dilated_arr * 0.5)

    # Apply multi-level Gaussian blur, weighted by gradient
    # We'll create several blur levels and blend based on the gradient
    result = img_arr.astype(np.float64)
    pil_img = Image.fromarray(img_arr)

    blur_levels = [
        (0.3, int(3 + intensity * 5)),
        (0.5, int(8 + intensity * 12)),
        (0.7, int(15 + intensity * 25)),
        (0.9, int(25 + intensity * 40)),
    ]

    for threshold, radius in blur_levels:
        blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float64)
        # Activate this blur level where gradient exceeds threshold
        level_mask = np.clip((melt_weight - threshold) / 0.2, 0, 1) * intensity
        level_mask_3ch = level_mask[:, :, np.newaxis]
        result = result * (1 - level_mask_3ch) + blurred * level_mask_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def effect_channel_shift(img_arr, mask_arr, intensity, direction_deg, output_dir):
    """Offset R, G, B channels by different amounts for chromatic aberration.

    R shifts in the direction, B shifts opposite, G stays. The shift amount
    scales with intensity.
    """
    log(output_dir, f"Applying channel shift (intensity={intensity:.2f}, direction={direction_deg} deg)")

    h, w = img_arr.shape[:2]
    result = img_arr.copy()

    # Shift amount: scales with image size (0.3-1% of short edge)
    short_edge = min(h, w)
    shift_px = max(3, int(short_edge * (0.005 + intensity * 0.025)))

    rad = math.radians(direction_deg)
    dx = int(round(math.cos(rad) * shift_px))
    dy = int(round(math.sin(rad) * shift_px))

    # Apply shift mainly in the subject area (with soft edges)
    mask_norm = mask_arr.astype(np.float64) / 255.0
    # Soften the mask so the effect bleeds slightly
    soft_mask = Image.fromarray((mask_norm * 255).astype(np.uint8))
    soft_mask = soft_mask.filter(ImageFilter.GaussianBlur(radius=8))
    blend = np.array(soft_mask).astype(np.float64) / 255.0 * intensity
    blend_3ch = blend[:, :, np.newaxis]

    # Shift red channel in direction
    r_shifted = np.zeros_like(img_arr[:, :, 0], dtype=np.float64)
    r_src_y = slice(max(0, -dy), min(h, h - dy))
    r_src_x = slice(max(0, -dx), min(w, w - dx))
    r_dst_y = slice(max(0, dy), min(h, h + dy))
    r_dst_x = slice(max(0, dx), min(w, w + dx))
    copy_h = min(r_src_y.stop - r_src_y.start, r_dst_y.stop - r_dst_y.start)
    copy_w = min(r_src_x.stop - r_src_x.start, r_dst_x.stop - r_dst_x.start)
    if copy_h > 0 and copy_w > 0:
        r_shifted[r_dst_y.start:r_dst_y.start+copy_h, r_dst_x.start:r_dst_x.start+copy_w] = \
            img_arr[r_src_y.start:r_src_y.start+copy_h, r_src_x.start:r_src_x.start+copy_w, 0]
    else:
        r_shifted = img_arr[:, :, 0].astype(np.float64)

    # Shift blue channel in opposite direction
    b_shifted = np.zeros_like(img_arr[:, :, 2], dtype=np.float64)
    b_src_y = slice(max(0, dy), min(h, h + dy))
    b_src_x = slice(max(0, dx), min(w, w + dx))
    b_dst_y = slice(max(0, -dy), min(h, h - dy))
    b_dst_x = slice(max(0, -dx), min(w, w - dx))
    copy_h_b = min(b_src_y.stop - b_src_y.start, b_dst_y.stop - b_dst_y.start)
    copy_w_b = min(b_src_x.stop - b_src_x.start, b_dst_x.stop - b_dst_x.start)
    if copy_h_b > 0 and copy_w_b > 0:
        b_shifted[b_dst_y.start:b_dst_y.start+copy_h_b, b_dst_x.start:b_dst_x.start+copy_w_b] = \
            img_arr[b_src_y.start:b_src_y.start+copy_h_b, b_src_x.start:b_src_x.start+copy_w_b, 2]
    else:
        b_shifted = img_arr[:, :, 2].astype(np.float64)

    # Blend shifted channels with original using the mask
    result_f = result.astype(np.float64)
    result_f[:, :, 0] = result_f[:, :, 0] * (1 - blend) + r_shifted * blend
    result_f[:, :, 2] = result_f[:, :, 2] * (1 - blend) + b_shifted * blend

    return np.clip(result_f, 0, 255).astype(np.uint8)



def effect_arc_ghosting(img_arr, mask_arr, intensity, direction_deg, rng, output_dir, arc_angle=30):
    """Exponential-offset arc ghosting: copies at 3, 6, 12, 24, 48px along a curved path.

    The mask controls which part of the image is ghosted (body without ropes).
    Arc_angle controls how much the path curves over the sequence of copies.
    """
    log(output_dir, f"Applying arc ghosting (intensity={intensity:.2f}, dir={direction_deg}°, arc={arc_angle}°)")

    h, w = img_arr.shape[:2]
    result = img_arr.astype(np.float64)

    mask_norm = mask_arr.astype(np.float64) / 255.0
    mask_3ch = mask_norm[:, :, np.newaxis]
    subject = img_arr.astype(np.float64) * mask_3ch

    # Exponential offsets scaled to image size (% of shorter edge)
    short_edge = min(w, h)
    base = max(5, int(short_edge * 0.01))  # ~1% of image as base unit
    offsets = [base * 1, base * 2, base * 4, base * 8, base * 16]
    opacities = [0.7, 0.55, 0.4, 0.25, 0.12]
    log(output_dir, f"Arc offsets (base={base}px): {offsets}")

    # Arc: each copy curves slightly more
    n_copies = len(offsets)
    arc_step = arc_angle / max(n_copies - 1, 1)

    for i, (offset_px, opacity) in enumerate(zip(offsets, opacities)):
        # Current angle along the arc
        current_angle = direction_deg + arc_step * i
        rad = math.radians(current_angle)
        ox = int(round(math.cos(rad) * offset_px * intensity))
        oy = int(round(math.sin(rad) * offset_px * intensity))

        if ox == 0 and oy == 0:
            continue

        shifted_subject = np.zeros_like(subject)
        shifted_mask = np.zeros_like(mask_norm)

        src_y_start = max(0, -oy)
        src_y_end = min(h, h - oy)
        src_x_start = max(0, -ox)
        src_x_end = min(w, w - ox)
        dst_y_start = max(0, oy)
        dst_y_end = min(h, h + oy)
        dst_x_start = max(0, ox)
        dst_x_end = min(w, w + ox)

        copy_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        copy_w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        if copy_h <= 0 or copy_w <= 0:
            continue

        shifted_subject[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
            subject[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
        shifted_mask[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
            mask_norm[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]

        blend_mask = shifted_mask[:, :, np.newaxis] * opacity
        result = result * (1 - blend_mask) + shifted_subject * blend_mask + result * blend_mask * (1 - opacity)

    return np.clip(result, 0, 255).astype(np.uint8)



def effect_echo(img_arr, mask_arr, intensity, direction_deg, rng, output_dir, n_echoes=4, arc_angle=20):
    """Silhouette echo: offset copies of the subject placed OUTSIDE the mask in the BG.

    The subject stays perfectly sharp. Semi-transparent copies of her silhouette
    radiate outward in 1-2 directions, creating a vibration/afterimage feel.
    Echoes only appear in the background — they don't cover ropes, gags, or
    anything else that's outside the affect mask.

    Uses the FULL subject (not just skin) for the echo shape, but places echoes
    only where the original image has background (not on non-affected foreground
    elements like ropes).
    """
    log(output_dir, f"Applying echo effect (intensity={intensity:.2f}, dir={direction_deg}°, n={n_echoes})")

    h, w = img_arr.shape[:2]
    result = img_arr.astype(np.float64)
    short_edge = min(h, w)

    # The mask defines what to AFFECT — but for echo, we use it differently:
    # - Subject stays sharp (mask area untouched)
    # - Echoes are placed OUTSIDE the mask (in background)
    mask_norm = mask_arr.astype(np.float64) / 255.0
    mask_3ch = mask_norm[:, :, np.newaxis]

    # Extract the subject pixels for making echo copies
    subject = img_arr.astype(np.float64) * mask_3ch

    # Exponential offsets scaled to image size
    base_offset = max(5, int(short_edge * 0.015))  # ~1.5% of image
    offsets = [base_offset * (2 ** i) for i in range(n_echoes)]  # 1x, 2x, 4x, 8x
    opacities = [0.5, 0.35, 0.2, 0.1][:n_echoes]

    # Color tint for echoes — slight shift toward a complementary tone
    # Creates a more artistic, less "copy-paste" feel
    tint_strength = 0.15 * intensity
    rad = math.radians(direction_deg)

    # Arc: each echo curves slightly
    arc_step = arc_angle / max(n_echoes - 1, 1)

    for i, (offset_px, opacity) in enumerate(zip(offsets, opacities)):
        current_angle = direction_deg + arc_step * i
        echo_rad = math.radians(current_angle)
        ox = int(round(math.cos(echo_rad) * offset_px * intensity))
        oy = int(round(math.sin(echo_rad) * offset_px * intensity))

        if ox == 0 and oy == 0:
            continue

        # Shift the subject
        shifted_subject = np.zeros_like(subject)
        shifted_mask = np.zeros_like(mask_norm)

        src_y_start = max(0, -oy)
        src_y_end = min(h, h - oy)
        src_x_start = max(0, -ox)
        src_x_end = min(w, w - ox)
        dst_y_start = max(0, oy)
        dst_y_end = min(h, h + oy)
        dst_x_start = max(0, ox)
        dst_x_end = min(w, w + ox)

        copy_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        copy_w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        if copy_h <= 0 or copy_w <= 0:
            continue

        shifted_subject[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
            subject[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
        shifted_mask[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
            mask_norm[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]

        # KEY: only place echo where the ORIGINAL mask is NOT active
        # This means echoes go into background, never over ropes/accessories/the model herself
        bg_zone = 1.0 - mask_norm  # background = where mask is 0
        echo_alpha = shifted_mask * opacity * intensity * bg_zone

        # Slight color shift for artistic feel (warmer for forward echoes, cooler for further)
        tinted = shifted_subject.copy()
        warmth = 1.0 - (i / n_echoes)  # first echo = warm, last = cool
        tinted[:, :, 0] = tinted[:, :, 0] * (1 + tint_strength * warmth)      # red boost
        tinted[:, :, 2] = tinted[:, :, 2] * (1 + tint_strength * (1 - warmth)) # blue boost on far echoes

        echo_alpha_3ch = echo_alpha[:, :, np.newaxis]
        result = result * (1 - echo_alpha_3ch) + tinted * echo_alpha_3ch

    # Subject stays perfectly sharp — paste original back on top
    result = result * (1 - mask_3ch) + img_arr.astype(np.float64) * mask_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_effects(img, mask, effect, intensity, direction_deg, seed, output_dir, **kwargs):
    """Apply the requested effect preset to the region defined by mask.

    mask: PIL L-mode image — white (255) = apply effects here.
    The mask is produced by build_affect_mask() which handles all targeting logic.

    For skin/body-segment affect targets, intensity is boosted slightly because
    the affected region is smaller than the full subject.
    For arc ghosting (ghost effect on skin targets), uses exponential arc offsets.

    kwargs:
      arc_angle      — arc curve for ghost effect (default 30)
      is_skin_target — True when affect contains skin/body-segment parts (enables arc ghosting + boost)
    """
    rng = np.random.default_rng(seed)
    img_arr = np.array(img)
    mask_arr = np.array(mask.resize(img.size, Image.LANCZOS))
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = np.where(mask_arr > 127, 255, 0).astype(np.uint8)

    arc_angle_arg = kwargs.get("arc_angle", 30)
    is_skin_target = kwargs.get("is_skin_target", False)

    # Boost intensity when effects target a sub-region (skin, face, clothes, etc.)
    # so the effect is as visible as if it were on the full subject
    effective_intensity = min(1.0, intensity * 1.5) if is_skin_target else intensity

    if is_skin_target and effect == "ghost":
        # Arc ghosting: exponential offsets give more visible body echoes
        log(output_dir, f"Using arc ghosting for skin target (arc_angle={arc_angle_arg})")
        result = effect_arc_ghosting(img_arr, mask_arr, effective_intensity, direction_deg,
                                     rng, output_dir, arc_angle_arg)
    else:
        result = _apply_single_effect(effect, img_arr, mask_arr, effective_intensity,
                                      direction_deg, rng, output_dir)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_single_effect(effect, img_arr, mask_arr, intensity, direction_deg, rng, output_dir):
    """Apply a single effect to the given mask area. Returns numpy array."""
    if effect == "ghost":
        return effect_ghosting(img_arr, mask_arr, intensity, direction_deg, rng, output_dir)
    elif effect == "melt":
        return effect_diffusion_melt(img_arr, mask_arr, intensity, direction_deg, output_dir)
    elif effect == "trails":
        return effect_motion_trails(img_arr, mask_arr, intensity, direction_deg, output_dir)
    elif effect == "glitch":
        return effect_channel_shift(img_arr, mask_arr, intensity, direction_deg, output_dir)
    elif effect == "echo":
        return effect_echo(img_arr, mask_arr, intensity, direction_deg, rng, output_dir)
    elif effect == "full":
        log(output_dir, "Full preset: layering all effects...")
        result = effect_channel_shift(img_arr, mask_arr, intensity * 0.8, direction_deg, output_dir)
        result = effect_ghosting(result, mask_arr, intensity * 0.7, direction_deg, rng, output_dir)
        result = effect_motion_trails(result, mask_arr, intensity * 0.6, direction_deg, output_dir)
        result = effect_diffusion_melt(result, mask_arr, intensity * 0.4, direction_deg, output_dir)
        return result
    else:
        return img_arr


# ---------------------------------------------------------------------------
# Auto-correction
# ---------------------------------------------------------------------------
def apply_corrections(args, eval_result, output_dir):
    """Parse Gemini feedback and return adjusted parameters, or None if good enough."""
    if not eval_result:
        return None
    score = eval_result.get("score", 10)
    if score >= 7:
        return None

    adjustments = eval_result.get("adjustments", {})
    issues = eval_result.get("issues", [])
    changes = {}

    if "too_extreme" in issues or "subject_lost" in issues:
        changes["intensity"] = max(0.3, args.intensity - 0.2)
        log(output_dir, f"Auto-correct: reducing intensity to {changes['intensity']:.2f}")

    if "too_subtle" in issues:
        changes["intensity"] = min(1.0, args.intensity + 0.15)
        log(output_dir, f"Auto-correct: increasing intensity to {changes['intensity']:.2f}")

    if "direction_wrong" in issues or adjustments.get("try_different_direction"):
        changes["direction"] = (args.direction + 90) % 360
        log(output_dir, f"Auto-correct: rotating direction to {changes['direction']} deg")

    if adjustments.get("intensity") is not None:
        changes["intensity"] = float(adjustments["intensity"])
        log(output_dir, f"Auto-correct: Gemini suggests intensity={changes['intensity']:.2f}")

    if "ugly_artifacts" in issues or "colors_clash" in issues:
        # Reduce intensity for channel shift heavy effects
        changes["intensity"] = max(0.3, args.intensity - 0.15)
        log(output_dir, f"Auto-correct: reducing intensity for artifacts: {changes['intensity']:.2f}")

    return changes if changes else None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def upload_to_gdrive(local_dir, model_name, photo_name, timestamp, output_dir):
    import subprocess
    gdrive_path = f"gdrive:_photos from openclaw/daily_game/public/{model_name}_{photo_name}_{timestamp}"
    try:
        subprocess.run(["rclone", "copy", local_dir, gdrive_path], check=True, timeout=120)
        res = subprocess.run(["rclone", "link", gdrive_path], capture_output=True, text=True, timeout=30)
        link = res.stdout.strip()
        log(output_dir, f"GDrive upload OK: {link}")
        return link
    except subprocess.TimeoutExpired:
        log(output_dir, "GDrive upload timed out", "ERROR")
        return None
    except Exception as e:
        log(output_dir, f"GDrive upload failed: {e}", "ERROR")
        return None


def copy_to_local(output_dir, local_dest):
    try:
        if os.path.exists(local_dest):
            for f in os.listdir(output_dir):
                src = os.path.join(output_dir, f)
                dst = os.path.join(local_dest, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            shutil.copytree(output_dir, local_dest)
        return local_dest
    except Exception as e:
        log(output_dir, f"Local copy failed: {e}", "ERROR")
        return None


def _print_summary(args, output_dir, effect, intensity, direction, seed, timings,
                   quality_report, gdrive_link, local_path):
    lines = [
        "",
        "=" * 60,
        "  TIME CORRUPTION — Summary",
        "=" * 60,
        "",
        "  Config:",
        f"    Source:         {args.source}",
        f"    Effect:         {effect}",
        f"    Affect:         {args.affect}",
        f"    Intensity:      {intensity}",
        f"    Direction:      {direction} deg",
        f"    Seed:           {seed}",
        "",
        "  Timings:",
    ]
    for step, elapsed in sorted(timings.items()):
        lines.append(f"    Step {step} ({STEP_NAMES.get(step, '?')}): {elapsed:.1f}s")
    lines.append(f"    Total: {sum(timings.values()):.1f}s")

    if quality_report:
        lines.append("")
        lines.append("  Quality:")
        if "final" in quality_report:
            qc = quality_report["final"]
            status = "OK" if qc["ok"] else f"FAIL: {qc['reason']}"
            lines.append(f"    Final image:   brightness={qc['brightness']}  contrast={qc['contrast']}  entropy={qc['entropy']}  {status}")
        if "aesthetic" in quality_report:
            ae = quality_report["aesthetic"]
            lines.append(f"    Aesthetic:     {ae.get('score', '?')}/10 — {ae.get('critique', 'N/A')}")
            if ae.get("issues"):
                lines.append(f"    Issues:        {', '.join(ae['issues'])}")
        if "aesthetic_retry" in quality_report:
            ar = quality_report["aesthetic_retry"]
            lines.append(f"    After fix:     {ar.get('score', '?')}/10 — {ar.get('critique', 'N/A')}")

    lines.append("")
    lines.append("  Output:")
    lines.append(f"    Working dir:   {output_dir}")
    if gdrive_link:
        lines.append(f"    GDrive:        {gdrive_link}")
    if local_path:
        lines.append(f"    Local:         {local_path}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)
    with _log_lock:
        with open(os.path.join(output_dir, "workflow.log"), "a") as f:
            f.write(summary + "\n")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def run_workflow(args):
    effect = args.effect
    intensity = args.intensity
    direction = args.direction
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Resolve model/photo names from filename
    basename = os.path.basename(args.source)
    photo_name = os.path.splitext(basename)[0]
    model_name = "Unknown"
    match = re.search(r"(Original_|BLD_|Rong_IMG_)(.*?)_(.*)\.(jpg|png|jpeg|JPG)", basename, re.IGNORECASE)
    if match:
        model_name = match.group(2).replace(" ", "_")
        photo_name = match.group(3).replace(" ", "_")
    else:
        source_abs = os.path.abspath(args.source)
        parts = source_abs.replace("\\", "/").split("/")
        try:
            photos_idx = parts.index("_photos")
            if photos_idx + 1 < len(parts):
                model_name = parts[photos_idx + 1].replace(" ", "_")
        except ValueError:
            pass

    # Output directory
    israel_dt = datetime.now(ISRAEL_TZ)
    timestamp = israel_dt.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name}_{photo_name}_{timestamp}_tc-{effect}_{random.randint(10,99)}"
    if args.local_output_dir:
        output_dir = os.path.join(args.local_output_dir, folder_name)
    else:
        output_dir = os.path.join("outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save a copy of this script for reproducibility
    try:
        with open(__file__, "r") as src, open(os.path.join(output_dir, f"time_corruption_script_{timestamp}.py"), "w") as dst:
            dst.write(src.read())
    except OSError:
        log(output_dir, "Could not save script copy (non-critical)", "WARN")

    timings = {}
    quality_report = {}

    # Resolve --affect (and handle deprecated --mode)
    affect = args.affect
    exclude = args.exclude

    # Determine is_skin_target: True when affect contains body-segment parts
    affect_parts = {p.strip().lower() for p in affect.split(",") if p.strip()}
    is_skin_target = bool(affect_parts & BODY_SEGMENT_PARTS)

    # Log config
    log(output_dir, "=" * 60)
    log(output_dir, "TIME CORRUPTION — Start")
    log(output_dir, f"Source:         {args.source}")
    log(output_dir, f"Effect:         {effect}")
    log(output_dir, f"Affect:         {affect}")
    if exclude:
        log(output_dir, f"Exclude:        {exclude}")
    log(output_dir, f"Intensity:      {intensity}")
    log(output_dir, f"Direction:      {direction} deg")
    log(output_dir, f"Seed:           {seed}")
    log(output_dir, f"Auto-correct:   {args.auto_correct}")
    log(output_dir, f"Output to:      {args.output_to}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    # Load source image
    original = Image.open(args.source).convert("RGB")
    original.save(os.path.join(output_dir, "00_original.jpg"), quality=95)
    log(output_dir, f"Loaded source: {original.size[0]}x{original.size[1]}")

    # --- Step 1: Build affect mask ---
    t0 = time.time()
    log(output_dir, f"--- Step 1/4: {STEP_NAMES[1]} ---")
    log(output_dir, f"Building mask for affect='{affect}' (exclude='{exclude}')")

    mask, mask_info = masking.build_mask(
        original, affect=affect, exclude=exclude,
        output_dir=output_dir, rope_color=args.rope_color,
    )
    log(output_dir, f"Mask engine: {mask_info['engine']}")

    # Resize mask to match source (build_mask should already return correct size, but be safe)
    mask = mask.resize(original.size, Image.LANCZOS)
    mask.save(os.path.join(output_dir, "01_mask.png"))

    # Check mask coverage
    mask_arr = np.array(mask)
    mask_coverage = (mask_arr > 127).sum() / (mask_arr.shape[0] * mask_arr.shape[1]) * 100
    log(output_dir, f"Mask coverage: {mask_coverage:.1f}%")
    if mask_coverage < 3:
        log(output_dir, "Mask too small (<3%) — effects will be applied to entire image", "WARN")
        mask = Image.new("L", original.size, 255)

    timings[1] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings[1]:.1f}s)")

    # --- Step 2: Apply effects ---
    t0 = time.time()
    log(output_dir, f"--- Step 2/4: {STEP_NAMES[2]} ---")

    result_img = apply_effects(original, mask, effect, intensity, direction, seed, output_dir,
                               arc_angle=args.arc_angle, is_skin_target=is_skin_target)
    result_img.save(os.path.join(output_dir, "02_time_corrupted.jpg"), quality=95)

    timings[2] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings[2]:.1f}s)")

    final_path = os.path.join(output_dir, "02_time_corrupted.jpg")

    # --- Step 3: Quality evaluation ---
    t0 = time.time()
    log(output_dir, f"--- Step 3/4: {STEP_NAMES[3]} ---")

    qc = check_image_quality(result_img, "time_corrupted", output_dir)
    quality_report["final"] = qc

    if not qc["ok"]:
        log(output_dir, f"Quality gate failed: {qc['reason']}. Reducing intensity and retrying.", "WARN")
        reduced_intensity = max(0.3, intensity * 0.6)
        result_img = apply_effects(original, mask, effect, reduced_intensity, direction, seed, output_dir,
                                   arc_angle=args.arc_angle, is_skin_target=is_skin_target)
        result_img.save(os.path.join(output_dir, "02_time_corrupted_retry.jpg"), quality=95)
        final_path = os.path.join(output_dir, "02_time_corrupted_retry.jpg")
        qc = check_image_quality(result_img, "time_corrupted_retry", output_dir)
        quality_report["final"] = qc

    # Gemini evaluation
    eval_result = evaluate_with_gemini(result_img, output_dir, original_img=original)
    if eval_result:
        quality_report["aesthetic"] = eval_result

    # Auto-correction loop
    if args.auto_correct and eval_result:
        corrections_done = 0
        current_intensity = intensity
        current_direction = direction

        while corrections_done < args.max_corrections:
            corrections = apply_corrections(args, eval_result, output_dir)
            if corrections is None:
                log(output_dir, "No corrections needed (score >= 7 or no actionable feedback)")
                break

            corrections_done += 1
            log(output_dir, f"Auto-correction round {corrections_done}/{args.max_corrections}")

            if "intensity" in corrections:
                current_intensity = corrections["intensity"]
                args.intensity = current_intensity
            if "direction" in corrections:
                current_direction = corrections["direction"]
                args.direction = current_direction

            result_img = apply_effects(original, mask, effect, current_intensity, current_direction,
                                       seed + corrections_done, output_dir,
                                       arc_angle=args.arc_angle, is_skin_target=is_skin_target)
            retry_path = os.path.join(output_dir, f"02_time_corrupted_correction_{corrections_done}.jpg")
            result_img.save(retry_path, quality=95)
            final_path = retry_path

            eval_result = evaluate_with_gemini(result_img, output_dir, original_img=original)
            if eval_result:
                quality_report["aesthetic_retry"] = eval_result

    timings[3] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings[3]:.1f}s)")

    # Save final
    final_output = os.path.join(output_dir, "final.jpg")
    result_img.save(final_output, quality=95)
    log(output_dir, f"Final image saved: {final_output}")

    # --- Step 4: Output / upload ---
    t0 = time.time()
    log(output_dir, f"--- Step 4/4: {STEP_NAMES[4]} ---")

    gdrive_link = None
    local_path = None

    if args.output_to in ("gdrive", "both"):
        gdrive_link = upload_to_gdrive(output_dir, model_name, photo_name, timestamp, output_dir)

    if args.output_to in ("local", "both"):
        if args.local_output_dir and os.path.abspath(output_dir).startswith(os.path.abspath(args.local_output_dir)):
            local_path = output_dir
            log(output_dir, f"Output already in local dir: {local_path}")
        else:
            default_local = os.path.expanduser(f"~/openclaw-outputs/{model_name}_{photo_name}_{timestamp}")
            dest = args.local_output_dir or default_local
            local_path = copy_to_local(output_dir, dest)
            if local_path:
                log(output_dir, f"Local copy: {local_path}")

    # Copy final to finals/ folder
    if args.local_output_dir and final_path and os.path.exists(final_path):
        finals_dir = os.path.join(args.local_output_dir, "finals")
        os.makedirs(finals_dir, exist_ok=True)
        finals_name = os.path.basename(output_dir) + ".jpg"
        finals_dest = os.path.join(finals_dir, finals_name)
        with open(final_path, "rb") as f_in:
            with open(finals_dest, "wb") as f_out:
                f_out.write(f_in.read())
        log(output_dir, f"Final copied to: {finals_dest}")

    timings[4] = time.time() - t0
    log(output_dir, f"Step 4 done ({timings[4]:.1f}s)")

    _print_summary(args, output_dir, effect, intensity, direction, seed, timings,
                   quality_report, gdrive_link, local_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Time Corruption — Temporal Decay Effects for Art Photography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Effects:
  ghost   — Multiple-exposure ghosting (arc offsets when targeting skin)
  melt    — Diffusion melting / dissolving through a gradient
  trails  — Directional motion blur trailing
  glitch  — Chromatic aberration / channel shift
  full    — All effects layered together

Affect targets (--affect):
  BiRefNet engine (API, ~5s, best edges):
    subject        — whole subject / foreground
    bg             — background only (subject stays sharp)

  MediaPipe body-segment engine (local, ~0.5s):
    skin           — face-skin + body-skin (= old dissolve mode; ropes/clothes excluded)
    face-skin      — face only
    body-skin      — neck, shoulders, torso, arms
    hair           — hair
    clothes        — clothing, accessories
    others         — miscellaneous foreground items

  Compound (comma-separated):
    face-skin,body-skin   — same as skin
    hair,clothes          — everything except bare skin

  Special:
    all            — entire image, no masking

Exclude (--exclude, body-segment targets only):
  hands, ropes, hair, clothes, others, background

Examples:
  %(prog)s --source photo.jpg --effect ghost --affect skin
  %(prog)s --source photo.jpg --effect melt --affect skin --intensity 0.8
  %(prog)s --source photo.jpg --effect ghost --affect bg
  %(prog)s --source photo.jpg --effect full --affect subject
  %(prog)s --source photo.jpg --effect ghost --affect skin --exclude hands
  %(prog)s --source photo.jpg --effect glitch --affect all

Deprecated (still works, prints warning):
  --mode dissolve  =>  --affect skin
  --mode normal    =>  --affect subject
  --mode float     =>  --affect bg
        """,
    )
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--effect", choices=EFFECTS, default="ghost",
                        help="Effect preset (default: ghost)")
    masking.add_affect_args(parser)
    # Override the masking default: time-corruption defaults to "skin" (shibari-safe)
    parser.set_defaults(affect="skin")
    parser.add_argument("--arc-angle", type=float, default=30,
                        help="Arc curve angle for ghost effect on skin targets (default: 30 degrees)")
    # Deprecated --mode flag (kept for backward compatibility)
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="DEPRECATED: use --affect instead. "
                             "normal => subject, dissolve => skin, float => bg")
    parser.add_argument("--rope-color", choices=["auto", "red", "beige", "black", "white"], default="auto",
                        help="Rope color hint for HSV detection when using --exclude ropes (default: auto)")
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Effect intensity 0.3-1.0 (default: 0.7)")
    parser.add_argument("--direction", type=float, default=0,
                        help="Direction angle in degrees (default: 0 = melt downward, trails rightward)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (random if not set)")
    parser.add_argument("--auto-correct", action="store_true", default=False,
                        help="Use Gemini feedback to auto-adjust parameters and retry")
    parser.add_argument("--max-corrections", type=int, default=2,
                        help="Max auto-correction rounds (default: 2)")
    parser.add_argument("--output-to", choices=["gdrive", "local", "both"], default="both",
                        help="Where to output results (default: both)")
    parser.add_argument("--local-output-dir", default=None,
                        help="Custom local output directory")

    args = parser.parse_args()

    # Handle deprecated --mode flag
    if args.mode is not None:
        mapped = _MODE_TO_AFFECT.get(args.mode, "skin")
        print(f"WARNING: --mode is deprecated. Use --affect {mapped} instead. "
              f"(--mode {args.mode} => --affect {mapped})")
        # Only override --affect if user didn't also explicitly set --affect
        # (if both are set, --affect takes precedence)
        if args.affect == "skin":  # still at default
            args.affect = mapped

    # Validate
    if not os.path.isfile(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        sys.exit(1)

    if not (0.3 <= args.intensity <= 1.0):
        print(f"ERROR: Intensity must be between 0.3 and 1.0 (got {args.intensity})")
        sys.exit(1)

    # Validate --affect parts
    affect_parts = {p.strip().lower() for p in args.affect.split(",") if p.strip()}
    valid_affect = BIREFNET_PARTS | BODY_SEGMENT_PARTS | {"all"}
    unknown = affect_parts - valid_affect
    if unknown:
        print(f"ERROR: Unknown --affect parts: {', '.join(sorted(unknown))}")
        print(f"Valid parts: {', '.join(sorted(valid_affect))}")
        sys.exit(1)

    run_workflow(args)


if __name__ == "__main__":
    main()
