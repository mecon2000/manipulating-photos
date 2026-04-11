#!/home/rong/openclaw-venv/bin/python3
"""
Material Swap Workflow — Transform Subject Skin into Glass, Marble, Metal, etc.

Takes a portrait photo, extracts the subject, transforms the subject's material/texture
using Tensor Art img2img with a material-specific prompt, then composites the
material-swapped subject back onto the pristine original background.

Especially suited for shibari photography where a fragile glass effect contrasts
with the ropes.

Usage:
    python material-swap.py --source photo.jpg --material "wet glass"
    python material-swap.py --source photo.jpg --material "marble" --strength 0.5
    python material-swap.py --source photo.jpg --prompt "custom material prompt" --strength 0.4
    python material-swap.py --source photo.jpg --list-presets
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
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageEnhance

# Shared masking module (sibling script)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from masking import build_mask, add_affect_args

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
MODEL_DEFAULT = "965126062386242266"  # Z-Image-Uncensored-fp16-v3

# ---------------------------------------------------------------------------
# Material Presets
# ---------------------------------------------------------------------------
MATERIAL_PRESETS = {
    "wet glass": "translucent wet glass skin, refractive, water droplets on glass surface, see-through, fragile, crystalline",
    "cracked glass": "cracked stained glass skin, shattered fracture lines, colorful glass fragments, mosaic, fragile beauty",
    "oily glass": "oily iridescent glass skin, rainbow oil slick surface, prismatic reflections, smooth transparent",
    "frosted glass": "frosted etched glass skin, translucent milky white, diffused light through glass, delicate",
    "marble": "white Carrara marble skin, smooth polished stone, subtle grey veins, classical sculpture",
    "liquid metal": "liquid mercury skin, chrome mirror surface, flowing metallic, T-1000 style",
    "porcelain": "fine porcelain skin, delicate ceramic, hairline cracks, glazed smooth surface",
    "ice": "frozen ice skin, translucent blue-white, crystalline frost patterns, cracking ice",
    "gold": "hammered gold leaf skin, burnished metallic gold, warm reflections, Byzantine icon",
    "obsidian": "polished obsidian volcanic glass skin, deep black reflective, sharp edges",
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
# Quality checks
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# API Key Helpers
# ---------------------------------------------------------------------------
def _get_tensor_key():
    key = os.environ.get("TENSOR_API_KEY")
    if not key:
        raise EnvironmentError("TENSOR_API_KEY not set")
    return key


# ---------------------------------------------------------------------------
# Tensor Art Upload / Job
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Gemini Evaluation (material-specific)
# ---------------------------------------------------------------------------
def _img_to_b64(img, max_size=1024):
    img_resized = img.copy()
    img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_EVAL_PROMPT = """\
You are an art director evaluating a material-swap photograph where the subject's skin \
has been transformed into a non-human material (glass, marble, metal, etc.).
If you see TWO images, the first is the ORIGINAL and the second is the MATERIAL-SWAPPED result — compare them.

Evaluate the MATERIAL-SWAPPED image on these criteria:
1. Material believability: does the skin convincingly look like the target material? Are there \
appropriate reflections, refractions, texture, transparency, or surface qualities?
2. Subject integrity: is the body shape and pose preserved? No warped limbs, missing body parts, \
or distorted anatomy. The form should be recognizable as the original subject.
3. Material consistency: is the material effect consistent across the entire subject, or are \
there patches that look like normal skin mixed with the material?
4. Background preservation: the background should be untouched/pristine — any bleed of the \
material effect into the background is a defect.
5. Edge quality: are the edges where material-subject meets background clean and natural, \
or are there halos, fringing, or rough cuts?
6. Overall artistic impact: does the material transformation create a striking, surreal, \
fine-art quality image?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "material_unconvincing", "subject_distorted", "face_warped", \
"inconsistent_material", "bg_contaminated", "bad_edges", "too_dark", "too_bright", \
"artifacts", "anatomy_wrong", "too_blurry", "material_patchy", "lost_form">],
  "adjustments": {
    "strength": <null or suggested float 0.1-0.8>,
    "cfg_scale": <null or suggested float 3-15>,
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
        img_b64 = _img_to_b64(img)
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]
        if original_img is not None:
            orig_b64 = _img_to_b64(original_img)
            parts.insert(0, {"text": "ORIGINAL (before material swap):"})
            parts.insert(1, {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"text": "MATERIAL-SWAPPED (after processing):\n\n" + _EVAL_PROMPT})
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
        content = candidates[0].get("content", {})
        parts_out = content.get("parts", [])
        if not parts_out:
            log(output_dir, f"Gemini candidate has no content parts (finishReason: {finish_reason})", "WARN")
            return None

        raw = parts_out[0].get("text", "").strip()
        log(output_dir, f"Gemini raw response ({len(raw)} chars, finishReason={finish_reason}): {raw[:500]}")

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


# ---------------------------------------------------------------------------
# Upload helpers
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


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Material Swap Workflow — Transform subject skin into glass, marble, metal, etc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--material", default=None, help="Material preset name (use --list-presets to see all)")
    parser.add_argument("--prompt", default=None, help="Custom material prompt (overrides preset)")
    parser.add_argument("--strength", type=float, default=0.4, help="Denoising strength (default: 0.4). Higher = more material, less recognizable form")
    parser.add_argument("--cfg-scale", type=float, default=7, help="CFG scale (default: 7)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--tensor-model", default=MODEL_DEFAULT, help=f"Tensor Art model ID (default: {MODEL_DEFAULT})")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per stylization on quality failure (default: 2)")
    parser.add_argument("--auto-correct", action="store_true", default=False,
                        help="Enable Gemini evaluation + auto-correction loop")
    parser.add_argument("--max-corrections", type=int, default=2, help="Max auto-correction rounds (default: 2)")
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local",
                        help="Where to output results (default: local)")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")
    parser.add_argument("--list-presets", action="store_true", help="List all material presets and exit")
    add_affect_args(parser)  # adds --affect (default: subject) and --exclude
    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{'Material Preset':<20} Prompt")
        print("=" * 90)
        for name, prompt in MATERIAL_PRESETS.items():
            desc = prompt[:65] + "..." if len(prompt) > 65 else prompt
            print(f"  {name:<18} {desc}")
        print(f"\nTotal: {len(MATERIAL_PRESETS)} presets")
        sys.exit(0)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Resolve material prompt
    if args.prompt:
        material_prompt_addition = args.prompt
        material_name = "Custom"
    elif args.material:
        material_key = args.material.lower()
        if material_key not in MATERIAL_PRESETS:
            print(f"ERROR: Unknown material preset '{args.material}'. Use --list-presets to see available presets.")
            sys.exit(1)
        material_prompt_addition = MATERIAL_PRESETS[material_key]
        material_name = args.material
    else:
        print("ERROR: Must specify --material <preset> or --prompt '<custom prompt>'")
        sys.exit(1)

    # Build the full prompt: material description applied to a person
    material_prompt = f"A portrait of a person made entirely of {material_name.lower()}, {material_prompt_addition}, photorealistic, fine art photography, dramatic lighting, high detail"

    # Derive model/photo names from path
    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    # Output directory
    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    material_tag = material_name.replace(" ", "_")[:25]
    suffix = random.randint(10, 99)
    folder_name = f"{model_name}_{source_basename}_{timestamp}_mat_{material_tag}_{suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    timings = {}

    log(output_dir, "=" * 60)
    log(output_dir, "MATERIAL SWAP WORKFLOW START")
    log(output_dir, f"Source:         {source}")
    log(output_dir, f"Material:       {material_name}")
    log(output_dir, f"Prompt:         {material_prompt[:100]}")
    log(output_dir, f"Strength:       {args.strength}")
    log(output_dir, f"CFG Scale:      {args.cfg_scale}")
    log(output_dir, f"Tensor Model:   {args.tensor_model}")
    log(output_dir, f"Seed:           {seed}")
    log(output_dir, f"Max retries:    {args.max_retries}")
    log(output_dir, f"Auto-correct:   {args.auto_correct}")
    log(output_dir, f"Affect:         {args.affect}")
    log(output_dir, f"Exclude:        {args.exclude or '(none)'}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    # Save original
    img_orig = Image.open(source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), "JPEG", quality=95)

    # Save script copy for reproducibility
    try:
        with open(__file__, "r") as src_f, open(os.path.join(output_dir, f"workflow_script_{timestamp}.py"), "w") as dst_f:
            dst_f.write(src_f.read())
    except OSError:
        log(output_dir, "Could not save script copy (permission issue, non-critical)", "WARN")

    # -----------------------------------------------------------------------
    # STEP 1: Extract subject mask using shared masking module
    # -----------------------------------------------------------------------
    t0 = time.time()
    log(output_dir, f"--- Step 1/4: Extract mask (affect={args.affect}, exclude='{args.exclude}') ---")
    mask, mask_info = build_mask(
        source,
        affect=args.affect,
        exclude=args.exclude,
        output_dir=output_dir,
    )

    # Resize mask to match image if needed
    if mask.size != img_orig.size:
        log(output_dir, f"Resizing mask {mask.size} -> {img_orig.size}")
        mask = mask.resize(img_orig.size, Image.LANCZOS)

    mask.save(os.path.join(output_dir, "1_mask.png"))

    mask_np = np.array(mask)
    mask_coverage = (mask_np > 127).sum() / mask_np.size
    log(output_dir, f"Mask coverage: {mask_coverage:.1%} of image (engine: {mask_info['engine']})")

    if mask_coverage < 0.03:
        log(output_dir, f"Mask covers only {mask_coverage:.1%} — could not find subject. Cannot proceed.", "ERROR")
        sys.exit(1)

    timings["extract"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['extract']:.1f}s)")

    # -----------------------------------------------------------------------
    # STEP 2: Stylize subject with material prompt via Tensor Art
    # -----------------------------------------------------------------------
    t0 = time.time()
    log(output_dir, "--- Step 2/4: Material-swap subject (Tensor Art img2img) ---")

    # Create subject on blurred+desaturated BG (avoid black background issues)
    blurred_bg = img_orig.filter(ImageFilter.GaussianBlur(radius=30))
    blurred_bg = ImageEnhance.Color(blurred_bg).enhance(0.3)
    subject_on_blurred = blurred_bg.copy()
    subject_on_blurred.paste(img_orig, mask=mask)
    subject_on_blurred.save(os.path.join(output_dir, "2_subject_input.jpg"), "JPEG", quality=95)

    # Run Tensor Art img2img with material prompt
    material_result = tensor_stylize_with_retry(
        subject_on_blurred, material_prompt, args.strength, args.cfg_scale,
        output_dir, "Material", args.tensor_model, seed, args.max_retries,
    )

    if material_result is None:
        log(output_dir, "Material stylization failed — cannot proceed", "ERROR")
        sys.exit(1)

    # Resize to match original if Tensor Art returned different dimensions
    if material_result.size != img_orig.size:
        log(output_dir, f"Resizing material result {material_result.size} -> {img_orig.size}")
        material_result = material_result.resize(img_orig.size, Image.LANCZOS)

    material_result.save(os.path.join(output_dir, "2_material_stylized.jpg"), "JPEG", quality=95)
    timings["stylize"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['stylize']:.1f}s)")

    # -----------------------------------------------------------------------
    # STEP 3: Composite — material subject onto pristine original BG
    # -----------------------------------------------------------------------
    t0 = time.time()
    log(output_dir, "--- Step 3/4: Composite (material subject + original BG) ---")

    # Soft-edge mask for clean blending
    soft_mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=3))

    # Composite: material-swapped subject over original background
    final_img = Image.composite(material_result, img_orig, soft_mask)
    final_path = os.path.join(output_dir, "3_composite.jpg")
    final_img.save(final_path, "JPEG", quality=95)

    quality_final = check_image_quality(final_img, "composite", output_dir)
    timings["composite"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['composite']:.1f}s)")

    # -----------------------------------------------------------------------
    # STEP 4: Evaluate + auto-correct + output
    # -----------------------------------------------------------------------
    t0 = time.time()
    log(output_dir, "--- Step 4/4: Quality evaluation + output ---")

    eval_result = None
    if args.auto_correct:
        eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

        # Auto-correction loop
        if eval_result and eval_result.get("score", 10) < 7:
            for correction_round in range(1, args.max_corrections + 1):
                prev_score = eval_result.get("score", 0)
                adjustments = eval_result.get("adjustments", {})
                log(output_dir, f"--- Auto-correction round {correction_round}/{args.max_corrections} (score={prev_score}/10) ---")

                # Apply adjustments from Gemini
                new_strength = adjustments.get("strength") or args.strength
                new_cfg = adjustments.get("cfg_scale") or args.cfg_scale
                new_seed = seed + correction_round if adjustments.get("try_different_seed") else seed

                log(output_dir, f"Re-stylizing with strength={new_strength}, cfg={new_cfg}, seed={new_seed}")

                retry_result = tensor_stylize_with_retry(
                    subject_on_blurred, material_prompt, new_strength, new_cfg,
                    output_dir, f"Material-r{correction_round}", args.tensor_model, new_seed, 1,
                )
                if retry_result is None:
                    log(output_dir, "Correction stylization failed — keeping previous result", "WARN")
                    break

                if retry_result.size != img_orig.size:
                    retry_result = retry_result.resize(img_orig.size, Image.LANCZOS)

                retry_composite = Image.composite(retry_result, img_orig, soft_mask)
                retry_path = os.path.join(output_dir, f"3_composite_r{correction_round}.jpg")
                retry_composite.save(retry_path, "JPEG", quality=95)

                retry_eval = evaluate_with_gemini(retry_composite, output_dir, original_img=img_orig)
                retry_score = retry_eval.get("score", 0) if retry_eval else 0

                if retry_score > prev_score:
                    log(output_dir, f"Correction improved: {prev_score} -> {retry_score}")
                    final_img = retry_composite
                    final_path = retry_path
                    eval_result = retry_eval
                    if retry_score >= 7:
                        break
                else:
                    log(output_dir, f"Correction round {correction_round} did not improve ({prev_score} -> {retry_score}) — stopping", "WARN")
                    break
    else:
        eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

    # Copy final to finals folder
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared")
    if os.path.exists(final_path):
        finals_dir = os.path.join(local_out, "finals")
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
            push_image(finals_dest, title=f"Material — {src_name}", body=f"{args.material}")
            log(output_dir, "Pushed to phone")
        except Exception as e:
            log(output_dir, f"Push notification failed: {e}", "WARN")

    # GDrive / local output
    gdrive_link = None
    local_path = None

    if args.output_to in ("gdrive", "both"):
        gdrive_link = upload_to_gdrive(output_dir, model_name, source_basename, timestamp, output_dir)

    if args.output_to in ("local", "both"):
        if args.local_output_dir and os.path.abspath(output_dir).startswith(os.path.abspath(args.local_output_dir)):
            local_path = output_dir
            log(output_dir, f"Output already in local dir: {local_path}")
        else:
            default_local = os.path.expanduser(f"~/openclaw-outputs/{model_name}_{source_basename}_{timestamp}")
            dest = args.local_output_dir or default_local
            local_path = copy_to_local(output_dir, dest)
            if local_path:
                log(output_dir, f"Local copy: {local_path}")

    timings["evaluate_output"] = time.time() - t0
    log(output_dir, f"Step 4 done ({timings['evaluate_output']:.1f}s)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = sum(timings.values())
    score_str = f"{eval_result['score']}/10 — {eval_result.get('critique', '')}" if eval_result else "N/A"

    summary = f"""
============================================================
  MATERIAL SWAP SUMMARY
============================================================
  Source:          {source}
  Material:        {material_name}
  Strength:        {args.strength}
  CFG Scale:       {args.cfg_scale}
  Seed:            {seed}
  Mask coverage:   {mask_coverage:.1%}

  Step Timings:
    1. Extract mask           {timings.get('extract', 0):>8.1f}s
    2. Material stylize       {timings.get('stylize', 0):>8.1f}s
    3. Composite              {timings.get('composite', 0):>8.1f}s
    4. Evaluate + output      {timings.get('evaluate_output', 0):>8.1f}s
    TOTAL                     {total:>8.1f}s

  Quality Report:
    Final image:   brightness={quality_final['brightness']}  contrast={quality_final['contrast']}  entropy={quality_final['entropy']}
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}"""

    if gdrive_link:
        summary += f"\n    GDrive:        {gdrive_link}"
    if local_path:
        summary += f"\n    Local:         {local_path}"

    summary += "\n============================================================"

    print(summary)
    with _log_lock:
        with open(os.path.join(output_dir, "workflow.log"), "a") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
