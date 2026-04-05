#!/home/rong/openclaw-venv/bin/python3
"""
Stylized Photo Workflow — Separate BG/Model Stylization

Takes an input photo and produces stylized fine-art output.
In 'separate' mode (default): extracts subject, cleans BG, stylizes BG and model
in parallel with different styles, composites back, face-swaps, evaluates quality.
In 'whole' mode (--no-separate): stylizes the entire image as one piece.

Quality gates catch degenerate outputs (black frames, blown-out images, zero-entropy
results) and auto-retry with different seeds. Optional Claude Vision aesthetic scoring
if ANTHROPIC_API_KEY is set.

Usage:
    python stylizing-bg-model-separately.py --source photo.jpg
    python stylizing-bg-model-separately.py --source photo.jpg --bg-style "Baroque Chiaroscuro" --model-style "Oil Impasto"
    python stylizing-bg-model-separately.py --source photo.jpg --no-separate --style "Indigo Wash"
    python stylizing-bg-model-separately.py --source photo.jpg --up-to-step 3 --output-to local
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
import shutil
import random
import base64
import argparse
import subprocess
import threading
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageDraw, ImageEnhance

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
MODEL_DEFAULT = "965126062386242266"  # Z-Image-Uncensored-fp16-v3

STYLE_PROMPTS = {
    "Baroque":     "dramatic Caravaggio-style lighting, intense contrast, theatrical, moody",
    "Oil Impasto": "thick paint, heavy texture, visible brushstrokes, rich colors, artistic",
    "Indigo":      "deep indigo wash, watercolor stains, bleeding ink, ethereal, monochromatic blue",
    "Smoke":       "ethereal smoke swirls, wispy textures, atmospheric, moody, dreamlike",
    "Cyberpunk":   "neon lights, futuristic, dark streets, glitch effects, high contrast",
    "Watercolor":  "soft watercolor washes, wet-on-wet blending, delicate, translucent layers",
    "Renaissance": "classical Renaissance painting, sfumato, warm earth tones, dignified",
    "Noir":        "high contrast black and white, deep shadows, film noir, cinematic grain",
}

# Load extended styles from styles.json if available
_styles_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.json")
if os.path.isfile(_styles_json):
    try:
        with open(_styles_json) as _f:
            for _s in json.load(_f):
                STYLE_PROMPTS[_s["name"]] = _s["prompt"]
    except Exception:
        pass  # Fall back to built-in styles

STEP_NAMES = {
    1: "Extract mask",
    2: "Clean background (LaMa)",
    3: "Stylize (parallel BG + Model)" ,
    4: "Composite",
    5: "Face swap",
    6: "Quality evaluation",
    7: "Upload",
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
# Quality Gate Utilities
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


def ssim_simple(img_a, img_b, size=256):
    """Simplified SSIM on downsampled grayscale. Returns float 0-1."""
    a = np.array(img_a.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    b = np.array(img_b.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2, sigma_b2 = a.var(), b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2))
    return float(ssim)


def evaluate_aesthetic(img, output_dir, original_img=None):
    """Aesthetic evaluation using Gemini Vision (free tier) or Claude Vision (fallback).

    Returns dict with:
        score: int 1-10
        critique: str
        issues: list of issue codes for auto-correction
        adjustments: dict of suggested parameter changes
    """
    # Try Gemini first (free), then Claude as fallback
    result = _evaluate_with_gemini(img, output_dir, original_img)
    if result is None:
        result = _evaluate_with_claude(img, output_dir)
    return result


def _img_to_b64(img, max_size=1024):
    """Downscale and encode image to base64 JPEG."""
    img_resized = img.copy()
    img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_EVAL_PROMPT = """\
You are an art director evaluating a stylized photograph for social media and fine art prints.
If you see TWO images, the first is the ORIGINAL and the second is the STYLIZED result — compare them.

Evaluate the STYLIZED image on these criteria:
1. Overall aesthetic appeal (composition, color harmony, visual impact)
2. Subject integrity: does the person's face/body look anatomically correct? Look for warped faces, \
extra/missing fingers, melted features, backwards-facing heads, unnatural skin
3. Style coherence: is the artistic style consistent across the whole image, or do BG and subject clash?
4. Background quality: is there a visible "hole" or silhouette artifact where the subject was removed? \
Does the BG look like it belongs with the subject?
5. Background-subject balance: does the BG overwhelm or overpower the subject?
6. Face orientation: is the subject's face clearly visible and facing the camera, or turned away/profile?
7. Brightness/contrast: is the subject too dark, washed out, or lacking definition?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "subject_distorted", "face_warped", "too_dark", "too_bright", \
"bg_overwhelms", "bg_has_hole", "bg_unrelated", "style_inconsistent", "low_contrast", "artifacts", \
"colors_clash", "subject_lost", "too_blurry", "face_sideways", "model_too_dark", "anatomy_wrong">],
  "adjustments": {
    "bg_strength": <null or suggested float 0.1-0.9>,
    "model_strength": <null or suggested float 0.1-0.9>,
    "cfg_scale": <null or suggested float 3-15>,
    "try_different_seed": <true/false>,
    "skip_faceswap": <true if face is sideways or distorted>,
    "try_different_style": <true if style fundamentally doesn't work>,
    "brighten_model": <true if subject is too dark after stylization>,
    "increase_lama_dilation": <true if BG has a visible hole/silhouette>,
    "suggestion": "<one sentence about what to change>"
  }
}"""


def _evaluate_with_gemini(img, output_dir, original_img=None):
    """Evaluate using Google Gemini Vision API (free tier)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping Gemini evaluation")
        return None

    try:
        img_b64 = _img_to_b64(img)

        parts = [
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        ]
        # If we have the original, send it too for comparison
        if original_img is not None:
            orig_b64 = _img_to_b64(original_img)
            parts.insert(0, {"text": "ORIGINAL (before stylization):"})
            parts.insert(1, {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"text": "STYLIZED (after processing):\n\n" + _EVAL_PROMPT})
        else:
            parts.append({"text": _EVAL_PROMPT})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048,
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
        # Handle missing/empty candidates (safety blocks, quota errors, etc.)
        candidates = resp_json.get("candidates", [])
        if not candidates:
            reason = resp_json.get("promptFeedback", {}).get("blockReason", "unknown")
            log(output_dir, f"Gemini returned no candidates (reason: {reason})", "WARN")
            return None
        # Handle finishReason != STOP (e.g. SAFETY, MAX_TOKENS)
        finish_reason = candidates[0].get("finishReason", "")
        content = candidates[0].get("content", {})
        parts_out = content.get("parts", [])
        if not parts_out:
            log(output_dir, f"Gemini candidate has no content parts (finishReason: {finish_reason})", "WARN")
            return None
        raw = parts_out[0].get("text", "").strip()
        log(output_dir, f"Gemini raw response ({len(raw)} chars, finishReason={finish_reason}): {raw[:500]}")
        # Strip markdown fences line by line
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
        # Find outermost JSON object (handle nested braces)
        # First try parsing the whole thing
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Find first { and last } to extract the JSON object
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                log(output_dir, f"Gemini response contains no JSON object: {raw[:200]}", "WARN")
                return None
            try:
                result = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as e:
                log(output_dir, f"Gemini JSON parse failed: {e}. Raw: {raw[start:start+300]}", "WARN")
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


def _evaluate_with_claude(img, output_dir):
    """Fallback: evaluate using Claude Vision API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    try:
        img_b64 = _img_to_b64(img)
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": _EVAL_PROMPT},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        log(output_dir, f"Claude Vision score: {result.get('score')}/10 — {result.get('critique')}")
        return result
    except Exception as e:
        log(output_dir, f"Claude Vision evaluation failed: {e}", "WARN")
        return None


def apply_adjustments(args, eval_result, output_dir):
    """Parse evaluation result and return correction strategy.

    Returns dict with:
        - Numeric param adjustments (bg_strength, model_strength, cfg_scale)
        - Strategy flags (skip_faceswap, try_different_style, brighten_model,
          increase_lama_dilation, new_seed)
    Or None if no correction needed.
    """
    if not eval_result:
        return None

    score = eval_result.get("score", 10)
    if score >= 7:
        return None  # Good enough

    adjustments = eval_result.get("adjustments", {})
    issues = eval_result.get("issues", [])
    changes = {}

    # --- Strategy: BG has hole/silhouette artifact → re-run LaMa with bigger dilation ---
    if "bg_has_hole" in issues or adjustments.get("increase_lama_dilation"):
        changes["increase_lama_dilation"] = True

    # --- Strategy: BG completely unrelated or style doesn't work → try different style ---
    if "bg_unrelated" in issues or "style_inconsistent" in issues or adjustments.get("try_different_style"):
        changes["try_different_style"] = True

    # --- Strategy: face issues → skip face swap + lower model strength ---
    if "face_warped" in issues or "face_sideways" in issues or adjustments.get("skip_faceswap"):
        changes["skip_faceswap"] = True
    if "subject_distorted" in issues or "face_warped" in issues or "anatomy_wrong" in issues:
        changes["model_strength"] = max(0.15, args.model_strength - 0.15)

    # --- Strategy: model too dark → brighten subject post-stylization ---
    if "model_too_dark" in issues or "too_dark" in issues or adjustments.get("brighten_model"):
        changes["brighten_model"] = True

    # --- Numeric adjustments ---
    if "bg_overwhelms" in issues or "subject_lost" in issues:
        changes["bg_strength"] = max(0.2, args.bg_strength - 0.15)
        changes["model_strength"] = min(0.7, args.model_strength + 0.1)

    if "too_bright" in issues:
        changes["cfg_scale"] = min(12.0, args.cfg_scale + 1.0)

    if "colors_clash" in issues:
        changes["color_match"] = True  # post-process: match model colors to BG

    if "low_contrast" in issues:
        changes["bg_strength"] = min(0.8, args.bg_strength + 0.1)

    # Use explicit numeric adjustments from the evaluator if provided
    if adjustments.get("bg_strength") is not None:
        changes["bg_strength"] = adjustments["bg_strength"]
    if adjustments.get("model_strength") is not None:
        changes["model_strength"] = adjustments["model_strength"]
    if adjustments.get("cfg_scale") is not None:
        changes["cfg_scale"] = adjustments["cfg_scale"]
    if adjustments.get("try_different_seed"):
        changes["new_seed"] = True

    if not changes:
        return None

    log(output_dir, f"Auto-correction strategy: {changes}")
    return changes


def pick_alternative_style(current_style):
    """Pick a random different style from the available styles."""
    available = list(STYLE_PROMPTS.keys())
    # Filter out current style
    alternatives = [s for s in available if s.lower() not in current_style.lower()]
    if not alternatives:
        return current_style
    return random.choice(alternatives)


def brighten_subject(img, mask, factor=1.3):
    """Brighten the subject area (within mask) without touching the BG."""
    brightened = ImageEnhance.Brightness(img).enhance(factor)
    # Also boost contrast slightly to avoid washing out
    brightened = ImageEnhance.Contrast(brightened).enhance(1.05)
    # Blend: only apply brightening within the mask area
    soft_mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=5))
    result = img.copy()
    result.paste(brightened, mask=soft_mask)
    return result


def color_match_to_bg(model_img, bg_img, mask):
    """Shift model's color palette toward the BG's palette for coherence."""
    # Get average color of BG (outside mask area)
    bg_stat = ImageStat.Stat(bg_img, mask=ImageOps.invert(mask.convert("L")))
    bg_mean = bg_stat.mean  # [R, G, B]

    # Get average color of model (inside mask area)
    model_stat = ImageStat.Stat(model_img, mask=mask.convert("L"))
    model_mean = model_stat.mean

    # Compute per-channel shift (blend 30% toward BG palette)
    blend = 0.3
    result = model_img.copy()
    r, g, b = result.split()
    r = r.point(lambda p: int(p + blend * (bg_mean[0] - model_mean[0])))
    g = g.point(lambda p: int(p + blend * (bg_mean[1] - model_mean[1])))
    b = b.point(lambda p: int(p + blend * (bg_mean[2] - model_mean[2])))
    return Image.merge("RGB", (r, g, b))


# ---------------------------------------------------------------------------
# API Wrappers
# ---------------------------------------------------------------------------
def _get_fal_key():
    key = os.environ.get("FAL_API_KEY")
    if not key:
        raise EnvironmentError("FAL_API_KEY not set")
    return key


def _get_tensor_key():
    key = os.environ.get("TENSOR_API_KEY")
    if not key:
        raise EnvironmentError("TENSOR_API_KEY not set")
    return key


def run_fal_rembg(image_path, output_dir):
    """Extract foreground mask using Fal.ai rembg (basic, fast)."""
    log(output_dir, "Extracting mask using rembg...")
    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post("https://fal.run/fal-ai/rembg", headers=headers,
            json={"image_url": f"data:image/jpeg;base64,{img_b64}"}, timeout=180)
    except requests.RequestException as e:
        log(output_dir, f"rembg request failed: {e}", "ERROR")
        return None
    if response.status_code != 200:
        log(output_dir, f"rembg failed ({response.status_code}): {response.text}", "ERROR")
        return None

    mask_url = response.json()["image"]["url"]
    mask_img = Image.open(requests.get(mask_url, stream=True, timeout=30).raw).split()[3]
    return mask_img


def run_fal_birefnet(image_path, output_dir):
    """Extract foreground mask using BiRefNet (better edges, catches hands/limbs)."""
    log(output_dir, "Extracting mask using BiRefNet (high quality)...")
    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post("https://fal.run/fal-ai/birefnet", headers=headers,
            json={"image_url": f"data:image/jpeg;base64,{img_b64}"}, timeout=180)
    except requests.RequestException as e:
        log(output_dir, f"BiRefNet request failed: {e}", "ERROR")
        return None
    if response.status_code != 200:
        log(output_dir, f"BiRefNet failed ({response.status_code}): {response.text}", "ERROR")
        return None

    data = response.json()
    result_url = data["image"]["url"]
    result_img = Image.open(requests.get(result_url, stream=True, timeout=30).raw)
    if result_img.mode == "RGBA":
        return result_img.split()[3]
    else:
        return result_img.convert("L")


def extract_mask(image_path, output_dir, model="birefnet"):
    """Extract foreground mask using the specified model. Falls back on failure."""
    if model == "birefnet":
        mask = run_fal_birefnet(image_path, output_dir)
        if mask is not None:
            return mask
        log(output_dir, "BiRefNet failed, falling back to rembg", "WARN")
    return run_fal_rembg(image_path, output_dir)


def run_fal_lama(image_pil, mask_pil, output_dir, dilation_px=25, margin_px=96):
    """Step 2: Remove subject from background using LaMa inpainting."""
    log(output_dir, "Cleaning background using LaMa inpainting...")

    mask = mask_pil.convert("L").point(lambda p: 255 if p > 127 else 0)
    k = dilation_px if dilation_px % 2 != 0 else dilation_px + 1
    mask = mask.filter(ImageFilter.MaxFilter(k)).point(lambda p: 255 if p > 127 else 0)

    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        log(output_dir, "Mask is empty — returning original image", "WARN")
        return image_pil

    x_min = max(0, int(xs.min()) - margin_px)
    x_max = min(image_pil.width, int(xs.max()) + margin_px)
    y_min = max(0, int(ys.min()) - margin_px)
    y_max = min(image_pil.height, int(ys.max()) + margin_px)

    image_crop = image_pil.crop((x_min, y_min, x_max, y_max))
    mask_crop = mask.crop((x_min, y_min, x_max, y_max))

    orig_w, orig_h = image_crop.size
    new_w, new_h = (orig_w // 8) * 8, (orig_h // 8) * 8
    if max(new_w, new_h) > 1024:
        scale = 1024 / max(new_w, new_h)
        new_w, new_h = (int(new_w * scale) // 8) * 8, (int(new_h * scale) // 8) * 8

    image_crop_send = image_crop.resize((new_w, new_h), Image.LANCZOS)
    mask_crop_send = mask_crop.resize((new_w, new_h), Image.LANCZOS)

    def to_b64(pil_img, fmt="JPEG"):
        buf = BytesIO()
        pil_img.save(buf, format=fmt, quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
    payload = {
        "image_url": f"data:image/jpeg;base64,{to_b64(image_crop_send)}",
        "mask_image_url": f"data:image/jpeg;base64,{to_b64(mask_crop_send)}",
    }

    try:
        response = requests.post("https://fal.run/fal-ai/lama", headers=headers, json=payload, timeout=180)
    except requests.RequestException as e:
        log(output_dir, f"LaMa request failed: {e}", "ERROR")
        return None
    if response.status_code != 200:
        log(output_dir, f"LaMa API failed ({response.status_code}): {response.text}", "ERROR")
        return None

    res_img_url = response.json()["image"]["url"]
    result_crop = Image.open(requests.get(res_img_url, stream=True, timeout=30).raw).convert("RGB")
    result_crop = result_crop.resize((x_max - x_min, y_max - y_min), Image.LANCZOS)

    # Feathered blending back into original
    mask_blur = mask_crop.filter(ImageFilter.GaussianBlur(radius=10))
    final = image_pil.copy()
    final.paste(result_crop, (x_min, y_min), mask=mask_blur)
    return final


def upload_to_tensor(image_pil, output_dir):
    """Upload a PIL image to Tensor Art and return (resource_id, width, height)."""
    w, h = image_pil.size
    MAX_PIXELS = 2073600
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
        image_pil = image_pil.resize((w, h), Image.LANCZOS)

    w, h = (w // 8) * 8, (h // 8) * 8
    image_pil = image_pil.resize((w, h), Image.LANCZOS)

    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    headers = {"Authorization": f"Bearer {_get_tensor_key()}", "Content-Type": "application/json"}

    try:
        res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers, timeout=30)
    except requests.RequestException as e:
        log(output_dir, f"Tensor upload init failed: {e}", "ERROR")
        return None, w, h
    if res.status_code != 200:
        log(output_dir, f"Tensor upload init failed ({res.status_code}): {res.text}", "ERROR")
        return None, w, h
    data = res.json()

    try:
        put_resp = requests.put(data["putUrl"], data=buf.getvalue(), headers=data["headers"], timeout=120)
    except requests.RequestException as e:
        log(output_dir, f"Tensor upload PUT failed: {e}", "ERROR")
        return None, w, h
    if put_resp.status_code not in (200, 201):
        log(output_dir, f"Tensor upload PUT failed ({put_resp.status_code})", "ERROR")
        return None, w, h

    return data["resourceId"], w, h


def run_tensor_job(payload, output_dir):
    """Submit a Tensor Art job and poll until completion. Returns image URL or None."""
    headers = {"Authorization": f"Bearer {_get_tensor_key()}", "Content-Type": "application/json"}

    response = requests.post(f"{TENSOR_BASE_URL}/jobs", headers=headers, json=payload, timeout=30)
    if response.status_code == 429:
        log(output_dir, "Tensor Art rate limited (429) — waiting 15s before retry", "WARN")
        time.sleep(15)
        response = requests.post(f"{TENSOR_BASE_URL}/jobs", headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        log(output_dir, f"Tensor job creation failed ({response.status_code}): {response.text}", "ERROR")
        return None

    job_id = response.json().get("job", {}).get("id")
    if not job_id:
        log(output_dir, "Tensor job response missing job ID", "ERROR")
        return None

    for attempt in range(60):
        time.sleep(5)
        try:
            res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers, timeout=15).json()
        except requests.RequestException as e:
            log(output_dir, f"Tensor poll error (attempt {attempt}): {e}", "WARN")
            continue
        status = res.get("job", {}).get("status")
        if status == "SUCCESS":
            images = res["job"].get("successInfo", {}).get("images", [])
            if images:
                return images[0]["url"]
            log(output_dir, "Tensor job succeeded but returned no images", "ERROR")
            return None
        elif status == "FAILED":
            log(output_dir, f"Tensor job failed: {json.dumps(res.get('job', {}), indent=2)}", "ERROR")
            return None

    log(output_dir, "Tensor job timed out after 5 minutes of polling", "ERROR")
    return None


def tensor_stylize(image_pil, prompt, strength, cfg_scale, output_dir, model_id, seed):
    """Single Tensor Art img2img stylization pass."""
    resource_id, w, h = upload_to_tensor(image_pil, output_dir)
    if resource_id is None:
        return None

    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {"image_resource_id": resource_id, "count": 1, "seed": seed},
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": w, "height": h,
                    "prompts": [{"text": prompt, "weight": 1.0}],
                    "sdModel": model_id, "steps": 30, "cfgScale": cfg_scale,
                    "denoisingStrength": strength, "sampler": "Euler a",
                },
            },
        ],
    }
    img_url = run_tensor_job(payload, output_dir)
    if img_url:
        return Image.open(requests.get(img_url, stream=True, timeout=30).raw).convert("RGB")
    return None


def tensor_stylize_with_retry(image_pil, prompt, strength, cfg_scale, output_dir,
                               label, model_id, base_seed, max_retries):
    """Stylize with quality-gated retries."""
    best = None
    for attempt in range(1 + max_retries):
        seed = base_seed + attempt
        tag = f"{label} (attempt {attempt + 1}/{1 + max_retries}, seed={seed})"
        log(output_dir, f"Stylizing: {tag}")

        result = tensor_stylize(image_pil, prompt, strength, cfg_scale, output_dir, model_id, seed)
        if result is None:
            log(output_dir, f"API returned nothing for {tag}", "WARN")
            continue

        qc = check_image_quality(result, tag, output_dir)
        if qc["ok"]:
            return result
        best = result  # keep last non-None even if quality failed

    if best is not None:
        log(output_dir, f"All retries exhausted for [{label}] — returning best available result", "WARN")
    else:
        log(output_dir, f"All retries exhausted for [{label}] — no usable result", "ERROR")
    return best


def run_fal_faceswap(source_path, target_path, output_dir):
    """Face-swap original face onto stylized result."""
    log(output_dir, "Running face swap...")
    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
    with open(source_path, "rb") as f:
        source_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(target_path, "rb") as f:
        target_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "base_image_url": f"data:image/jpeg;base64,{target_b64}",
        "swap_image_url": f"data:image/jpeg;base64,{source_b64}",
    }
    try:
        response = requests.post("https://fal.run/fal-ai/face-swap", headers=headers, json=payload, timeout=180)
    except requests.RequestException as e:
        log(output_dir, f"Face swap request failed: {e}", "ERROR")
        return None
    if response.status_code == 200:
        img_url = response.json().get("image", {}).get("url")
        if img_url:
            return Image.open(requests.get(img_url, stream=True, timeout=30).raw).convert("RGB")
    log(output_dir, f"Face swap failed ({response.status_code}): {response.text}", "ERROR")
    return None


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------
def upload_to_gdrive(local_dir, model_name, photo_name, timestamp, output_dir):
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
            # Merge into existing dir
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


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt_addition(style):
    for key, addition in STYLE_PROMPTS.items():
        if key.lower() in style.lower():
            return addition
    return "fine art, detailed, artistic"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_workflow(args):
    # Resolve styles
    bg_style = args.bg_style or args.style
    model_style = args.model_style or args.style
    bg_prompt_add = args.prompt_add or build_prompt_addition(bg_style)
    model_prompt_add = args.prompt_add or build_prompt_addition(model_style)

    extra = f", {args.prompt_extra}" if args.prompt_extra else ""
    bg_prompt = f"An abstract fine art {bg_style} background, {bg_prompt_add}, moody, cinematic, painterly textures{extra}"
    model_prompt = f"A fine art portrait, {model_style} style, {model_prompt_add}, high detail, realistic skin texture{extra}"

    # Resolve model/photo names from filename
    basename = os.path.basename(args.source)
    photo_name = os.path.splitext(basename)[0]
    model_name = args.model_name
    if not model_name:
        # Try to extract from filename pattern (e.g. BLD_Michaela_001.jpg)
        match = re.search(r"(Original_|BLD_|Rong_IMG_)(.*?)_(.*)\.(jpg|png|jpeg|JPG)", basename, re.IGNORECASE)
        if match:
            model_name = match.group(2).replace(" ", "_")
            photo_name = match.group(3).replace(" ", "_")
        else:
            # Try to detect from folder path (e.g. _photos/Michaela/Processed/file.jpg)
            source_abs = os.path.abspath(args.source)
            parts = source_abs.replace("\\", "/").split("/")
            try:
                photos_idx = parts.index("_photos")
                if photos_idx + 1 < len(parts):
                    model_name = parts[photos_idx + 1].replace(" ", "_")
            except ValueError:
                model_name = "Unknown"

    # Seed
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Output directory — <model_name>_<original_filename>_<timestamp>
    israel_dt = datetime.now(ISRAEL_TZ)
    timestamp = israel_dt.strftime("%Y-%m-%d_%H-%M-%S")
    # Short style tag for folder name (first 2-3 words, cleaned)
    style_tag = bg_style.replace(" ", "_")[:20]
    folder_name = f"{model_name}_{photo_name}_{timestamp}_{style_tag}_{random.randint(10,99)}"
    if args.local_output_dir:
        output_dir = os.path.join(args.local_output_dir, folder_name)
    else:
        output_dir = os.path.join("outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save a copy of this script for reproducibility
    # Save a copy of this script for reproducibility (use raw copy to avoid permission issues on shared folders)
    try:
        with open(__file__, "r") as src, open(os.path.join(output_dir, f"workflow_script_{timestamp}.py"), "w") as dst:
            dst.write(src.read())
    except OSError:
        log(output_dir, "Could not save script copy (permission issue, non-critical)", "WARN")

    # Log configuration
    mode = "separate" if args.separate else "whole-image"
    log(output_dir, "=" * 60)
    log(output_dir, f"WORKFLOW START — Mode: {mode}")
    log(output_dir, f"Source:         {args.source}")
    log(output_dir, f"BG Style:       {bg_style}")
    log(output_dir, f"Model Style:    {model_style}")
    log(output_dir, f"BG Strength:    {args.bg_strength}")
    log(output_dir, f"Model Strength: {args.model_strength}")
    log(output_dir, f"CFG Scale:      {args.cfg_scale}")
    log(output_dir, f"Tensor Model:   {args.tensor_model}")
    log(output_dir, f"Seed:           {base_seed}")
    log(output_dir, f"Max retries:    {args.max_retries}")
    log(output_dir, f"Face swap:      {args.faceswap}")
    log(output_dir, f"Output to:      {args.output_to}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    img_orig = Image.open(args.source).convert("RGB")
    orig_full_size = img_orig.size  # Save for final upscale
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), quality=95)

    # Keep a pristine copy for model compositing (before posterize/downscale)
    img_pristine = img_orig.copy()

    # Pre-processing (order matters: directional blur → posterize → downscale)
    if args.stroke_angle is not None:
        import math
        from scipy.ndimage import convolve
        length = args.stroke_length or 20
        angle = args.stroke_angle
        log(output_dir, f"Pre-process: directional blur angle={angle}° length={length}px (guides AI stroke direction)")
        # Create a motion blur kernel as a numpy array
        kernel = np.zeros((length, length), dtype=np.float64)
        cx, cy = length // 2, length // 2
        rad = math.radians(angle)
        for i in range(length):
            t = i - length // 2
            x = int(cx + t * math.cos(rad))
            y = int(cy - t * math.sin(rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() or 1
        # Apply to each channel
        img_arr = np.array(img_orig, dtype=np.float64)
        for c in range(3):
            img_arr[:, :, c] = convolve(img_arr[:, :, c], kernel, mode='reflect')
        img_orig = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img_orig.save(os.path.join(output_dir, "0_directional_blur.jpg"), quality=95)

    if args.posterize:
        bits = args.posterize
        img_orig = ImageOps.posterize(img_orig, bits)
        log(output_dir, f"Pre-process: posterized to {bits} bits ({2**bits} tones per channel)")
        img_orig.save(os.path.join(output_dir, "0_preprocessed.jpg"), quality=95)

    if args.downscale:
        long_edge = max(img_orig.size)
        if long_edge > args.downscale:
            scale = args.downscale / long_edge
            new_size = (int(img_orig.size[0] * scale), int(img_orig.size[1] * scale))
            img_orig = img_orig.resize(new_size, Image.LANCZOS)
            log(output_dir, f"Pre-process: downscaled {orig_full_size} -> {new_size} (strokes will appear larger)")
            img_orig.save(os.path.join(output_dir, "0_downscaled.jpg"), quality=95)

    timings = {}
    up_to = args.up_to_step or 7

    # Track intermediate results for summary
    mask = None
    bg_clean = None
    bg_stylized = None
    model_stylized = None
    final_img = None
    final_path = None
    quality_report = {}

    # -----------------------------------------------------------------------
    # STEP 1: MASK (separate mode only, but may auto-switch to whole-image)
    # -----------------------------------------------------------------------
    use_separate = args.separate
    if use_separate:
        # --- Step 1: Mask ---
        if up_to >= 1:
            t0 = time.time()
            log(output_dir, f"--- Step 1/7: {STEP_NAMES[1]} ---")
            mask = extract_mask(args.source, output_dir, model=args.mask_model)
            timings[1] = time.time() - t0
            if mask is None:
                log(output_dir, "Mask extraction failed — falling back to whole-image mode", "WARN")
                use_separate = False
                mode = "whole-image (auto: mask failed)"
            else:
                mask.save(os.path.join(output_dir, "1_mask_raw.png"))

                # Expand mask to catch nearby body parts (hands, arms) misclassified as BG
                if args.mask_expand > 0:
                    k = args.mask_expand if args.mask_expand % 2 != 0 else args.mask_expand + 1
                    mask = mask.filter(ImageFilter.MaxFilter(k))
                    log(output_dir, f"Mask expanded by {args.mask_expand}px to catch nearby body parts")

                # Resize mask to match img_orig if it was downscaled
                if mask.size != img_orig.size:
                    log(output_dir, f"Resizing mask {mask.size} -> {img_orig.size} to match (pre-processed) image")
                    mask = mask.resize(img_orig.size, Image.LANCZOS)

                mask.save(os.path.join(output_dir, "1_mask.png"))

                # Check mask coverage — if subject fills most of the frame, separation is pointless
                mask_np = np.array(mask.convert("L"))
                mask_coverage = (mask_np > 127).sum() / mask_np.size
                log(output_dir, f"Mask coverage: {mask_coverage:.1%} of image")
                if mask_coverage > 0.70:
                    log(output_dir, f"Subject fills {mask_coverage:.0%} of the frame — switching to whole-image mode (separation would remove most of the image)", "WARN")
                    use_separate = False
                    mode = "whole-image (auto: subject too large)"
                elif mask_coverage < 0.05:
                    log(output_dir, f"Mask covers only {mask_coverage:.1%} — rembg couldn't find the subject. Switching to whole-image mode", "WARN")
                    use_separate = False
                    mode = "whole-image (auto: no subject found)"

            log(output_dir, f"Step 1 done ({timings[1]:.1f}s)")
        if up_to < 2:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

    # -----------------------------------------------------------------------
    # SEPARATE MODE: Steps 2-4 (only if still using separate after mask check)
    # -----------------------------------------------------------------------
    if use_separate:
        # --- Step 2: Clean BG ---
        if up_to >= 2:
            t0 = time.time()
            log(output_dir, f"--- Step 2/7: {STEP_NAMES[2]} (method: {args.bg_fill}) ---")

            if args.bg_fill == "blur":
                # Blur the model area in the original — no LaMa API call needed.
                # Produces uniform texture so stylization is even across the whole BG.
                log(output_dir, "Filling model area with heavy blur (no LaMa)")
                mask_dilated = mask.convert("L").point(lambda p: 255 if p > 127 else 0)
                k = args.dilation if args.dilation % 2 != 0 else args.dilation + 1
                mask_dilated = mask_dilated.filter(ImageFilter.MaxFilter(k))
                # Heavy blur of original to fill the hole
                blurred_fill = img_orig.filter(ImageFilter.GaussianBlur(radius=40))
                bg_clean = img_orig.copy()
                # Feathered paste of blurred area over the model region
                mask_feathered = mask_dilated.filter(ImageFilter.GaussianBlur(radius=15))
                bg_clean.paste(blurred_fill, mask=mask_feathered)
            else:
                # LaMa inpainting — produces cleaner fill but different texture than real BG
                bg_clean = run_fal_lama(img_orig, mask, output_dir, args.dilation)
                if bg_clean is None:
                    log(output_dir, "LaMa cleanup failed — falling back to blur fill", "WARN")
                    blurred_fill = img_orig.filter(ImageFilter.GaussianBlur(radius=40))
                    mask_l = mask.convert("L").point(lambda p: 255 if p > 127 else 0)
                    mask_f = mask_l.filter(ImageFilter.GaussianBlur(radius=15))
                    bg_clean = img_orig.copy()
                    bg_clean.paste(blurred_fill, mask=mask_f)

            timings[2] = time.time() - t0
            bg_clean.save(os.path.join(output_dir, "2_bg_clean.jpg"), quality=95)
            check_image_quality(bg_clean, "cleaned BG", output_dir)
            log(output_dir, f"Step 2 done ({timings[2]:.1f}s)")
        if up_to < 3:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

        # --- Step 3: Parallel Stylization ---
        if up_to >= 3:
            t0 = time.time()
            log(output_dir, f"--- Step 3/7: {STEP_NAMES[3]} ---")

            # Skip model stylization entirely if strength is 0 — use pristine original pixels
            if args.model_strength == 0.0:
                log(output_dir, "Model strength=0.0 — skipping model stylization, using pristine original subject")
                model_only = img_pristine.copy()
                model_stylized = img_pristine.copy()

                # Stylize BG only
                bg_stylized = tensor_stylize_with_retry(
                    bg_clean, bg_prompt, args.bg_strength, args.cfg_scale,
                    output_dir, "BG", args.tensor_model, base_seed, args.max_retries,
                )
            else:
                # Prepare model-only image: subject on heavily blurred original BG
                blurred_bg = img_orig.filter(ImageFilter.GaussianBlur(radius=30))
                blurred_bg = ImageEnhance.Color(blurred_bg).enhance(0.3)
                model_only = blurred_bg.copy()
                model_only.paste(img_orig, mask=mask)
                model_only.save(os.path.join(output_dir, "3_model_input.jpg"), quality=95)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    future_bg = pool.submit(
                        tensor_stylize_with_retry,
                        bg_clean, bg_prompt, args.bg_strength, args.cfg_scale,
                        output_dir, "BG", args.tensor_model, base_seed, args.max_retries,
                    )
                    future_model = pool.submit(
                        tensor_stylize_with_retry,
                        model_only, model_prompt, args.model_strength, args.cfg_scale,
                        output_dir, "Model", args.tensor_model, base_seed, args.max_retries,
                    )
                    bg_stylized = future_bg.result()
                    model_stylized = future_model.result()

            timings[3] = time.time() - t0

            if bg_stylized is not None:
                bg_stylized.save(os.path.join(output_dir, "3a_bg_stylized.jpg"), quality=95)
                ssim_val = ssim_simple(bg_clean, bg_stylized)
                quality_report["bg_ssim"] = round(ssim_val, 3)
                if ssim_val > 0.95:
                    log(output_dir, f"BG SSIM={ssim_val:.3f} — style had almost no effect (too similar)", "WARN")
                elif ssim_val < 0.2:
                    log(output_dir, f"BG SSIM={ssim_val:.3f} — stylized BG is radically different from original (may look disconnected)", "WARN")
                else:
                    log(output_dir, f"BG SSIM={ssim_val:.3f} — within acceptable range")
            else:
                log(output_dir, "BG stylization failed entirely", "ERROR")

            if model_stylized is not None:
                model_stylized.save(os.path.join(output_dir, "3b_model_stylized.jpg"), quality=95)
            else:
                log(output_dir, "Model stylization failed entirely", "ERROR")

            if bg_stylized is None or model_stylized is None:
                log(output_dir, "Cannot proceed to composite — one or both stylizations failed", "ERROR")
                _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
                return

            log(output_dir, f"Step 3 done ({timings[3]:.1f}s)")
        if up_to < 4:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

        # --- Step 4: Composite ---
        if up_to >= 4:
            t0 = time.time()
            log(output_dir, f"--- Step 4/7: {STEP_NAMES[4]} ---")
            # Resize everything to original dimensions (Tensor Art may return different sizes)
            orig_size = img_orig.size
            if bg_stylized.size != orig_size:
                log(output_dir, f"Resizing BG {bg_stylized.size} -> {orig_size}")
                bg_stylized = bg_stylized.resize(orig_size, Image.LANCZOS)
            if model_stylized.size != orig_size:
                log(output_dir, f"Resizing Model {model_stylized.size} -> {orig_size}")
                model_stylized = model_stylized.resize(orig_size, Image.LANCZOS)
            soft_mask = mask.resize(orig_size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=3))
            final_img = Image.composite(model_stylized, bg_stylized, soft_mask)
            final_path = os.path.join(output_dir, "4_composite.jpg")
            final_img.save(final_path, "JPEG", quality=95)
            check_image_quality(final_img, "composite", output_dir)
            timings[4] = time.time() - t0
            log(output_dir, f"Step 4 done ({timings[4]:.1f}s)")
        if up_to < 5:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

    # -----------------------------------------------------------------------
    # WHOLE-IMAGE MODE (--no-separate or auto-switched)
    # -----------------------------------------------------------------------
    if not use_separate:
        # Skip steps 1, 2, 4 — just stylize the whole image
        if up_to >= 3:
            t0 = time.time()
            log(output_dir, f"--- Step 3/7: Stylize whole image ---")
            whole_prompt = f"A fine art {bg_style} photograph, {bg_prompt_add}, moody, cinematic, painterly textures, high detail"
            final_img = tensor_stylize_with_retry(
                img_orig, whole_prompt, args.bg_strength, args.cfg_scale,
                output_dir, "Whole", args.tensor_model, base_seed, args.max_retries,
            )
            timings[3] = time.time() - t0
            if final_img is None:
                log(output_dir, "Whole-image stylization failed", "ERROR")
                _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
                return
            # Upscale back to original dimensions if we downscaled
            if args.downscale and final_img.size != orig_full_size:
                log(output_dir, f"Upscaling {final_img.size} -> {orig_full_size}")
                final_img = final_img.resize(orig_full_size, Image.LANCZOS)

            final_path = os.path.join(output_dir, "3_stylized_whole.jpg")
            final_img.save(final_path, "JPEG", quality=95)
            log(output_dir, f"Step 3 done ({timings[3]:.1f}s)")
        if up_to < 5:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

    # -----------------------------------------------------------------------
    # SHARED STEPS (both modes)
    # -----------------------------------------------------------------------

    # --- Step 5: Face swap ---
    if up_to >= 5 and args.faceswap and final_path:
        t0 = time.time()
        log(output_dir, f"--- Step 5/7: {STEP_NAMES[5]} ---")
        swapped = run_fal_faceswap(args.source, final_path, output_dir)
        timings[5] = time.time() - t0
        if swapped is not None:
            qc = check_image_quality(swapped, "face-swapped", output_dir)
            if qc["ok"]:
                final_img = swapped
                final_path = os.path.join(output_dir, "5_faceswapped.jpg")
                final_img.save(final_path, "JPEG", quality=95)
            else:
                log(output_dir, "Face-swapped image failed quality check — keeping pre-swap version", "WARN")
        else:
            log(output_dir, "Face swap failed — keeping pre-swap version", "WARN")
        log(output_dir, f"Step 5 done ({timings.get(5, 0):.1f}s)")
    elif up_to >= 5 and not args.faceswap:
        log(output_dir, "Step 5: Face swap skipped (--no-faceswap)")
    if up_to < 6:
        _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
        return

    # --- Step 6: Quality evaluation + auto-correction ---
    if up_to >= 6 and final_img:
        t0 = time.time()
        log(output_dir, f"--- Step 6/7: {STEP_NAMES[6]} ---")
        qc_final = check_image_quality(final_img, "FINAL", output_dir)
        quality_report["final"] = qc_final

        # Aesthetic evaluation (Gemini free tier, Claude fallback)
        eval_result = evaluate_aesthetic(final_img, output_dir, original_img=img_orig)
        if eval_result:
            quality_report["aesthetic"] = eval_result

            # Auto-correction loop: retry with adjusted strategies if score < 7
            max_corrections = args.max_corrections
            correction_round = 0
            current_eval = eval_result
            current_final = final_img
            current_path = final_path

            while (args.auto_correct
                   and correction_round < max_corrections
                   and current_eval
                   and current_eval.get("score", 10) < 7):

                correction_round += 1
                changes = apply_adjustments(args, current_eval, output_dir)
                if not changes:
                    log(output_dir, "No actionable corrections identified — stopping")
                    break

                log(output_dir, f"--- Auto-correction round {correction_round}/{max_corrections} (score={current_eval['score']}/10) ---")
                correction_seed = base_seed + (100 * correction_round) if changes.get("new_seed") else base_seed + correction_round

                # --- Strategy: re-run LaMa with bigger dilation ---
                retry_bg_clean = bg_clean
                if changes.get("increase_lama_dilation") and args.separate and mask is not None:
                    new_dilation = args.dilation + 20 * correction_round
                    log(output_dir, f"Re-running LaMa with dilation={new_dilation}px")
                    retry_bg_clean = run_fal_lama(img_orig, mask, output_dir, dilation_px=new_dilation)
                    if retry_bg_clean is None:
                        retry_bg_clean = bg_clean
                    else:
                        retry_bg_clean.save(os.path.join(output_dir, f"6_bg_clean_retry{correction_round}.jpg"), quality=95)

                # --- Strategy: try a different style ---
                retry_bg_style = bg_style
                retry_model_style = model_style
                retry_bg_prompt = bg_prompt
                retry_model_prompt = model_prompt
                if changes.get("try_different_style"):
                    retry_bg_style = pick_alternative_style(bg_style)
                    retry_model_style = retry_bg_style  # keep consistent
                    retry_bg_add = build_prompt_addition(retry_bg_style)
                    retry_model_add = build_prompt_addition(retry_model_style)
                    retry_bg_prompt = f"An abstract fine art {retry_bg_style} background, {retry_bg_add}, moody, cinematic, painterly textures"
                    retry_model_prompt = f"A fine art portrait, {retry_model_style} style, {retry_model_add}, high detail, realistic skin texture"
                    log(output_dir, f"Switching style: {bg_style} -> {retry_bg_style}")

                adj_bg_str = changes.get("bg_strength", args.bg_strength)
                adj_model_str = changes.get("model_strength", args.model_strength)
                adj_cfg = changes.get("cfg_scale", args.cfg_scale)

                if args.separate and retry_bg_clean is not None and mask is not None:
                    log(output_dir, f"Re-stylizing with bg_str={adj_bg_str}, model_str={adj_model_str}, cfg={adj_cfg}")

                    blurred_bg = img_orig.filter(ImageFilter.GaussianBlur(radius=30))
                    blurred_bg = ImageEnhance.Color(blurred_bg).enhance(0.3)
                    model_only_retry = blurred_bg.copy()
                    model_only_retry.paste(img_orig, mask=mask)

                    with ThreadPoolExecutor(max_workers=2) as pool:
                        f_bg = pool.submit(tensor_stylize_with_retry,
                            retry_bg_clean, retry_bg_prompt, adj_bg_str, adj_cfg,
                            output_dir, f"BG-r{correction_round}", args.tensor_model, correction_seed, 1)
                        f_model = pool.submit(tensor_stylize_with_retry,
                            model_only_retry, retry_model_prompt, adj_model_str, adj_cfg,
                            output_dir, f"Model-r{correction_round}", args.tensor_model, correction_seed, 1)
                        bg_retry = f_bg.result()
                        model_retry = f_model.result()

                    if not bg_retry or not model_retry:
                        log(output_dir, "Correction stylization failed — stopping", "WARN")
                        break

                    # --- Post-processing strategies ---
                    if changes.get("brighten_model"):
                        log(output_dir, "Brightening subject")
                        model_retry = brighten_subject(model_retry,
                            mask.resize(model_retry.size, Image.LANCZOS))

                    if changes.get("color_match"):
                        log(output_dir, "Color-matching model to BG")
                        model_retry = color_match_to_bg(model_retry, bg_retry,
                            mask.resize(model_retry.size, Image.LANCZOS))

                    soft_mask = mask.resize(model_retry.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=3))
                    retry_composite = Image.composite(model_retry, bg_retry, soft_mask)

                    # --- Strategy: skip face swap for this round ---
                    if not changes.get("skip_faceswap") and args.faceswap:
                        retry_comp_path = os.path.join(output_dir, f"6_corrected_r{correction_round}_pre_swap.jpg")
                        retry_composite.save(retry_comp_path, "JPEG", quality=95)
                        swapped = run_fal_faceswap(args.source, retry_comp_path, output_dir)
                        if swapped and check_image_quality(swapped, f"faceswap-r{correction_round}", output_dir)["ok"]:
                            retry_composite = swapped
                    elif changes.get("skip_faceswap"):
                        log(output_dir, "Skipping face swap per correction strategy")

                    retry_path = os.path.join(output_dir, f"6_corrected_r{correction_round}.jpg")
                    retry_composite.save(retry_path, "JPEG", quality=95)

                elif not args.separate:
                    retry_prompt = f"A fine art {retry_bg_style} photograph, {build_prompt_addition(retry_bg_style)}, moody, cinematic, painterly textures, high detail"
                    retry_img = tensor_stylize_with_retry(
                        img_orig, retry_prompt, adj_bg_str, adj_cfg,
                        output_dir, f"Whole-r{correction_round}", args.tensor_model, correction_seed, 1)
                    if not retry_img:
                        log(output_dir, "Correction stylization failed — stopping", "WARN")
                        break
                    if changes.get("brighten_model"):
                        retry_img = ImageEnhance.Brightness(retry_img).enhance(1.3)
                    retry_composite = retry_img
                    retry_path = os.path.join(output_dir, f"6_corrected_r{correction_round}.jpg")
                    retry_composite.save(retry_path, "JPEG", quality=95)
                else:
                    break

                # Re-evaluate the corrected version
                retry_eval = evaluate_aesthetic(retry_composite, output_dir, original_img=img_orig)
                retry_score = retry_eval.get("score", 0) if retry_eval else 0
                original_score = current_eval.get("score", 0)

                if retry_score > original_score:
                    log(output_dir, f"Correction round {correction_round} improved: {original_score} -> {retry_score}")
                    current_final = retry_composite
                    current_path = retry_path
                    current_eval = retry_eval
                    quality_report[f"aesthetic_r{correction_round}"] = retry_eval
                else:
                    log(output_dir, f"Correction round {correction_round} did not improve ({original_score} -> {retry_score}) — stopping", "WARN")
                    break

            # Use best result
            if current_final is not final_img:
                final_img = current_final
                final_path = current_path

        timings[6] = time.time() - t0
        log(output_dir, f"Step 6 done ({timings[6]:.1f}s)")
    if up_to < 7:
        _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
        return

    # --- Step 7: Upload ---
    if up_to >= 7:
        t0 = time.time()
        log(output_dir, f"--- Step 7/7: {STEP_NAMES[7]} ---")
        gdrive_link = None
        local_path = None

        if args.output_to in ("gdrive", "both"):
            gdrive_link = upload_to_gdrive(output_dir, model_name, photo_name, timestamp, output_dir)

        if args.output_to in ("local", "both"):
            if args.local_output_dir and os.path.abspath(output_dir).startswith(os.path.abspath(args.local_output_dir)):
                # output_dir is already inside local_output_dir — no copy needed
                local_path = output_dir
                log(output_dir, f"Output already in local dir: {local_path}")
            else:
                default_local = os.path.expanduser(f"~/openclaw-outputs/{model_name}_{photo_name}_{timestamp}")
                dest = args.local_output_dir or default_local
                local_path = copy_to_local(output_dir, dest)
                if local_path:
                    log(output_dir, f"Local copy: {local_path}")

        timings[7] = time.time() - t0
        log(output_dir, f"Step 7 done ({timings[7]:.1f}s)")

        _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, gdrive_link, local_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(args, output_dir, mode, bg_style, model_style, seed, timings, quality_report, gdrive_link, local_path):
    total = sum(timings.values())
    lines = [
        "",
        "=" * 60,
        "  WORKFLOW SUMMARY",
        "=" * 60,
        f"  Source:          {args.source}",
        f"  Mode:            {mode}",
        f"  BG Style:        {bg_style}",
        f"  Model Style:     {model_style}",
        f"  BG Strength:     {args.bg_strength}",
        f"  Model Strength:  {args.model_strength}",
        f"  Seed:            {seed}",
        f"  Tensor Model:    {args.tensor_model}",
        "",
        "  Step Timings:",
    ]
    for step_num in sorted(timings):
        name = STEP_NAMES.get(step_num, f"Step {step_num}")
        lines.append(f"    {step_num}. {name:<35} {timings[step_num]:>6.1f}s")
    lines.append(f"    {'TOTAL':<38} {total:>6.1f}s")

    lines.append("")
    lines.append("  Quality Report:")
    if "final" in quality_report:
        qc = quality_report["final"]
        status = "OK" if qc["ok"] else f"FAIL: {qc['reason']}"
        lines.append(f"    Final image:   brightness={qc['brightness']}  contrast={qc['contrast']}  entropy={qc['entropy']}  {status}")
    if "bg_ssim" in quality_report:
        lines.append(f"    BG SSIM:       {quality_report['bg_ssim']}")
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
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stylized Photo Workflow — Separate BG/Model Stylization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--model-name", default="", help="Model/subject name (auto-detected from filename if empty)")

    # Style
    parser.add_argument("--style", default="Baroque Chiaroscuro", help="Style for both BG and model (default: Baroque Chiaroscuro)")
    parser.add_argument("--bg-style", default=None, help="Override style for background only")
    parser.add_argument("--model-style", default=None, help="Override style for model/subject only")
    parser.add_argument("--prompt-add", default="", help="Extra prompt text (overrides auto style-based addition)")
    parser.add_argument("--prompt-extra", default="", help="Additional prompt text appended to the style prompt (e.g. 'long brush strokes, bold palette knife')")
    parser.add_argument("--posterize", type=int, default=None, choices=[2, 3, 4, 5],
                        help="Pre-process: reduce image to N bits per channel (2=4 tones, 3=8 tones, 4=16 tones)")
    parser.add_argument("--downscale", type=int, default=None,
                        help="Pre-process: downscale long edge to N pixels before stylizing (e.g. 512, 768). Makes brush strokes appear larger. Result is upscaled back.")
    parser.add_argument("--stroke-angle", type=int, default=None,
                        help="Pre-process: apply directional blur at N degrees before stylizing (0=horizontal, 45=diagonal, 90=vertical). Guides AI to follow stroke direction.")
    parser.add_argument("--stroke-length", type=int, default=20,
                        help="Length of directional blur in pixels (default: 20). Larger = more pronounced direction.")

    # Strengths
    parser.add_argument("--bg-strength", type=float, default=0.6, help="Denoising strength for BG (default: 0.6)")
    parser.add_argument("--model-strength", type=float, default=0.4, help="Denoising strength for model (default: 0.4)")
    parser.add_argument("--cfg-scale", type=float, default=6.5, help="CFG scale for Tensor Art (default: 6.5)")

    # Mode
    parser.add_argument("--separate", action="store_true", default=True, help="Separate BG/model stylization (default)")
    parser.add_argument("--no-separate", dest="separate", action="store_false", help="Stylize whole image without separation")
    parser.add_argument("--up-to-step", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Run only up to this step number (1-7)")

    # Face swap
    parser.add_argument("--faceswap", action="store_true", default=True)
    parser.add_argument("--no-faceswap", dest="faceswap", action="store_false", help="Skip face swap step")

    # Tensor Art
    parser.add_argument("--tensor-model", default=MODEL_DEFAULT, help=f"Tensor Art model ID (default: {MODEL_DEFAULT})")
    parser.add_argument("--dilation", type=int, default=25, help="Mask dilation in pixels for LaMa (default: 25)")
    parser.add_argument("--bg-fill", choices=["lama", "blur"], default="blur",
                        help="How to fill the model hole in BG: 'blur' (uniform texture, default) or 'lama' (AI inpainting)")
    parser.add_argument("--mask-expand", type=int, default=0, help="Expand foreground mask by N pixels to catch nearby body parts (default: 0)")
    parser.add_argument("--mask-model", choices=["birefnet", "rembg"], default="birefnet",
                        help="Segmentation model for mask extraction (default: birefnet — better edges, catches hands)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed (random if not set)")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per stylization on quality failure (default: 2)")
    parser.add_argument("--auto-correct", action="store_true", default=False,
                        help="If aesthetic score < 7, auto-adjust params and retry stylization")
    parser.add_argument("--max-corrections", type=int, default=2,
                        help="Max auto-correction rounds (default: 2)")

    # Output
    parser.add_argument("--output-to", choices=["gdrive", "local", "both"], default="both",
                        help="Where to upload results (default: both)")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")

    parser.add_argument("--list-styles", action="store_true", help="List all available styles and exit")

    args = parser.parse_args()

    if args.list_styles:
        print(f"\n{len(STYLE_PROMPTS)} available styles:\n")
        for name, prompt in sorted(STYLE_PROMPTS.items()):
            print(f"  {name:<30} {prompt[:70]}")
        sys.exit(0)

    # Validate source exists
    if not os.path.isfile(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        sys.exit(1)

    run_workflow(args)


if __name__ == "__main__":
    main()
