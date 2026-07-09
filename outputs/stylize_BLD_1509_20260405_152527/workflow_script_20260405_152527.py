#!/usr/bin/env python3
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
from datetime import datetime, timedelta
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
    israel_time = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
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


def evaluate_with_claude_vision(img, output_dir):
    """Optional aesthetic evaluation using Claude Vision API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log(output_dir, "ANTHROPIC_API_KEY not set — skipping Claude Vision evaluation")
        return None

    try:
        import anthropic
    except ImportError:
        log(output_dir, "anthropic package not installed — skipping Claude Vision evaluation")
        return None

    try:
        buf = BytesIO()
        img_resized = img.copy()
        # Downscale for API cost/speed — 1024px long edge is plenty for evaluation
        img_resized.thumbnail((1024, 1024), Image.LANCZOS)
        img_resized.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an art director evaluating a stylized photo for social media / fine art print. "
                            "Rate it 1-10 on aesthetic appeal. Consider: composition, color harmony, "
                            "style coherence, whether the subject looks natural vs. distorted, overall visual impact. "
                            "Respond ONLY with valid JSON: {\"score\": <int 1-10>, \"critique\": \"<2-3 sentences>\"}"
                        ),
                    },
                ],
            }],
        )
        raw = response.content[0].text.strip()
        # Try to parse JSON, tolerating markdown fences
        raw = re.sub(r"^```json\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        log(output_dir, f"Claude Vision score: {result.get('score')}/10 — {result.get('critique')}")
        return result
    except Exception as e:
        log(output_dir, f"Claude Vision evaluation failed: {e}", "WARN")
        return None


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
    """Step 1: Extract foreground mask using Fal.ai rembg."""
    log(output_dir, "Extracting mask using Fal.ai rembg...")
    url = "https://fal.run/fal-ai/rembg"
    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(url, headers=headers, json={"image_url": f"data:image/jpeg;base64,{img_b64}"}, timeout=60)
    if response.status_code != 200:
        log(output_dir, f"rembg failed ({response.status_code}): {response.text}", "ERROR")
        return None

    mask_url = response.json()["image"]["url"]
    mask_img = Image.open(requests.get(mask_url, stream=True, timeout=30).raw).split()[3]
    return mask_img


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

    response = requests.post("https://fal.run/fal-ai/lama", headers=headers, json=payload, timeout=120)
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

    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers, timeout=30)
    if res.status_code != 200:
        log(output_dir, f"Tensor upload init failed ({res.status_code}): {res.text}", "ERROR")
        return None, w, h
    data = res.json()

    put_resp = requests.put(data["putUrl"], data=buf.getvalue(), headers=data["headers"], timeout=60)
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
    response = requests.post("https://fal.run/fal-ai/face-swap", headers=headers, json=payload, timeout=120)
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

    bg_prompt = f"An abstract fine art {bg_style} background, {bg_prompt_add}, moody, cinematic, painterly textures"
    model_prompt = f"A fine art portrait, {model_style} style, {model_prompt_add}, high detail, realistic skin texture"

    # Resolve model/photo names from filename
    basename = os.path.basename(args.source)
    photo_name = os.path.splitext(basename)[0]
    model_name = args.model_name
    if not model_name:
        match = re.search(r"(Original_|BLD_|Rong_IMG_)(.*?)_(.*)\.(jpg|png|jpeg|JPG)", basename, re.IGNORECASE)
        if match:
            model_name = match.group(2).replace(" ", "_")
            photo_name = match.group(3).replace(" ", "_")
        else:
            model_name = "Unknown"

    # Seed
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Output directory
    israel_dt = datetime.utcnow() + timedelta(hours=3)
    timestamp = israel_dt.strftime("%Y%m%d_%H%M%S")
    if args.local_output_dir:
        output_dir = args.local_output_dir
    else:
        output_dir = f"outputs/stylize_{photo_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save a copy of this script for reproducibility
    shutil.copy(__file__, os.path.join(output_dir, f"workflow_script_{timestamp}.py"))

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
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), quality=95)

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
    # SEPARATE MODE
    # -----------------------------------------------------------------------
    if args.separate:
        # --- Step 1: Mask ---
        if up_to >= 1:
            t0 = time.time()
            log(output_dir, f"--- Step 1/7: {STEP_NAMES[1]} ---")
            mask = run_fal_rembg(args.source, output_dir)
            timings[1] = time.time() - t0
            if mask is None:
                log(output_dir, "FATAL: Mask extraction failed — cannot continue in separate mode", "ERROR")
                _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
                return
            mask.save(os.path.join(output_dir, "1_mask.png"))
            log(output_dir, f"Step 1 done ({timings[1]:.1f}s)")
        if up_to < 2:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

        # --- Step 2: Clean BG ---
        if up_to >= 2:
            t0 = time.time()
            log(output_dir, f"--- Step 2/7: {STEP_NAMES[2]} ---")
            bg_clean = run_fal_lama(img_orig, mask, output_dir, args.dilation)
            timings[2] = time.time() - t0
            if bg_clean is None:
                log(output_dir, "LaMa cleanup failed — falling back to original image as BG", "WARN")
                bg_clean = img_orig.copy()
            bg_clean.save(os.path.join(output_dir, "2_bg_clean.jpg"), quality=95)
            # Quality check on cleaned BG
            check_image_quality(bg_clean, "cleaned BG", output_dir)
            log(output_dir, f"Step 2 done ({timings[2]:.1f}s)")
        if up_to < 3:
            _print_summary(args, output_dir, mode, bg_style, model_style, base_seed, timings, quality_report, None, None)
            return

        # --- Step 3: Parallel Stylization ---
        if up_to >= 3:
            t0 = time.time()
            log(output_dir, f"--- Step 3/7: {STEP_NAMES[3]} ---")

            # Prepare model-only image (subject on black BG)
            model_only = Image.new("RGB", img_orig.size, (0, 0, 0))
            model_only.paste(img_orig, mask=mask)
            model_only.save(os.path.join(output_dir, "3_model_only.jpg"), quality=95)

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
                # SSIM check: is stylized BG reasonable vs cleaned BG?
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
            soft_mask = mask.resize(model_stylized.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=3))
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
    # WHOLE-IMAGE MODE (--no-separate)
    # -----------------------------------------------------------------------
    else:
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

    # --- Step 6: Quality evaluation ---
    if up_to >= 6 and final_img:
        t0 = time.time()
        log(output_dir, f"--- Step 6/7: {STEP_NAMES[6]} ---")
        qc_final = check_image_quality(final_img, "FINAL", output_dir)
        quality_report["final"] = qc_final

        # Claude Vision aesthetic evaluation (optional)
        vision_result = evaluate_with_claude_vision(final_img, output_dir)
        if vision_result:
            quality_report["claude_vision"] = vision_result

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
    if "claude_vision" in quality_report:
        cv = quality_report["claude_vision"]
        lines.append(f"    Claude Vision: {cv.get('score', '?')}/10 — {cv.get('critique', 'N/A')}")

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
    parser.add_argument("--seed", type=int, default=None, help="Base seed (random if not set)")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per stylization on quality failure (default: 2)")

    # Output
    parser.add_argument("--output-to", choices=["gdrive", "local", "both"], default="both",
                        help="Where to upload results (default: both)")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")

    args = parser.parse_args()

    # Validate source exists
    if not os.path.isfile(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        sys.exit(1)

    run_workflow(args)


if __name__ == "__main__":
    main()
