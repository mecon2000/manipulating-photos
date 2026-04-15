#!/home/rong/openclaw-venv/bin/python3
"""
Baroque Surround — Generative Painterly Background

Creates a "baroque oil painting surround" effect: photorealistic subject floating
in flowing, painterly, baroque-style forms. The background is GENERATIVE (wholly
reimagined via inpainting), not transformative. Subject stays completely untouched.

Usage:
    python baroque-surround.py --source photo.jpg --preset baroque
    python baroque-surround.py --source photo.jpg --preset dark-romantic
    python baroque-surround.py --source photo.jpg --prompt "custom prompt" --strength 0.95
    python baroque-surround.py --list-presets
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

# fal_client expects FAL_KEY, but env has FAL_API_KEY
os.environ['FAL_KEY'] = os.environ.get('FAL_API_KEY', '')

import re
import json
import time
import random
import base64
import argparse
import shutil
import tempfile
import threading
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat
from scipy import ndimage
import fal_client

# Shared masking module (BiRefNet / body-segment)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask, add_affect_args

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "baroque": {
        "prompt": "large flowing amorphous organic shapes and billowing drapery surrounding the subject, baroque oil painting, dramatic chiaroscuro, luminous glazing, Bouguereau and Caravaggio inspired, warm ochre and cool blue-grey and cream, smooth blended brushwork, sweeping abstract undulating forms radiating outward from center, masterpiece classical painting",
        "negative": "modern, digital, sharp edges, text, watermark, flat colors, cartoon, solid color background, plain background",
        "strength": 0.95,
    },
    "renaissance": {
        "prompt": "large soft amorphous forms of golden light and flowing draped silk fabric, sfumato Renaissance oil painting, Raphael and da Vinci inspired, abstract sweeping shapes in olive and warm brown and soft blue, luminous atmospheric depth, billowing organic forms emanating from the figure",
        "negative": "modern, digital, harsh lighting, text, watermark, flat background, solid color",
        "strength": 0.92,
    },
    "dark-romantic": {
        "prompt": "large swirling amorphous storm forms and turbulent abstract shapes, dark romantic oil painting, Delacroix and Turner inspired, dark blue and warm amber and charcoal and copper, flowing organic masses radiating from center, dramatic atmospheric turbulence, visible sweeping brushwork",
        "negative": "bright, cheerful, flat, text, watermark, cartoon, solid background, empty background",
        "strength": 0.95,
    },
    "ethereal": {
        "prompt": "large flowing amorphous luminous forms and soft ethereal mist, dreamy angelic oil painting, abstract billowing organic shapes in pearl and ivory and pale gold and soft blue, divine radiance emanating outward, sweeping undulating cloud-like masses surrounding figure",
        "negative": "dark, gritty, harsh, text, watermark, modern, flat background",
        "strength": 0.93,
    },
    "smoke": {
        "prompt": "large visible swirling smoke plumes and flowing amorphous grey forms against dark background, tenebrist oil painting, dramatic single light source illuminating billowing smoke shapes, abstract organic masses in warm grey and amber and cream emerging from shadows, Caravaggio chiaroscuro, NOT solid black",
        "negative": "flat black, solid black, empty background, text, watermark, all dark, plain background",
        "strength": 0.93,
    },
}

_log_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    if output_dir:
        with _log_lock:
            log_path = os.path.join(output_dir, "workflow.log")
            try:
                with open(log_path, "a") as f:
                    f.write(formatted + "\n")
            except OSError:
                pass


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
# Gemini Evaluation
# ---------------------------------------------------------------------------
_EVAL_PROMPT = """\
You are a professional art director evaluating a composite photograph where the subject is photographic \
and the background has been replaced with generative painterly forms (baroque/classical oil painting style).

If you see TWO images, the first is the ORIGINAL and the second is the RESULT — compare them.

Evaluate the RESULT image on these criteria:
1. Subject integrity: does the person look untouched, photorealistic, and anatomically correct?
2. Surround quality: does the painterly background look like convincing oil painting? Rich textures, proper brushwork?
3. Transition: is the blend between photographic subject and painted surround smooth and natural?
4. Composition: does the overall image work as a coherent piece of art?
5. Color harmony: do the surround colors complement the subject?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "subject_altered", "harsh_transition", "flat_background", \
"color_clash", "artifacts", "too_dark", "too_bright", "incoherent", "repetitive_pattern">]
}"""


def evaluate_with_gemini(img, output_dir, original_img=None):
    import requests
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping Gemini evaluation")
        return None
    try:
        def _img_to_b64(im, max_size=1024):
            im_resized = im.copy()
            im_resized.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = BytesIO()
            im_resized.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        img_b64 = _img_to_b64(img)
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]
        if original_img is not None:
            orig_b64 = _img_to_b64(original_img)
            parts.insert(0, {"text": "ORIGINAL:"})
            parts.insert(1, {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"text": "RESULT (baroque surround applied):\n\n" + _EVAL_PROMPT})
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
        log(output_dir, f"Gemini raw ({len(raw)} chars, finishReason={finish_reason}): {raw[:500]}")

        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                log(output_dir, f"Gemini response contains no JSON: {raw[:200]}", "WARN")
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
        return result
    except Exception as e:
        log(output_dir, f"Gemini evaluation failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Inpainting via fal.ai
# ---------------------------------------------------------------------------
def upload_pil_image(img, fmt="PNG"):
    """Upload a PIL image to fal.ai CDN and return the URL."""
    with tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False) as tmp:
        img.save(tmp, format=fmt, quality=95)
        tmp_path = tmp.name
    try:
        url = fal_client.upload_file(tmp_path)
        return url
    finally:
        os.unlink(tmp_path)


def generate_bg(prompt, width, height, output_dir, seed=None):
    """Generate a standalone baroque BG from scratch via text-to-image."""
    log(output_dir, f"Generating BG: {width}x{height}, prompt='{prompt[:80]}...'")
    import requests as req_lib

    payload = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        handle = fal_client.submit("fal-ai/flux/dev", arguments=payload)
        result = handle.get()
        images = result.get("images", [])
        if not images:
            log(output_dir, "BG generation returned no images", "ERROR")
            return None
        bg_url = images[0].get("url", "")
        log(output_dir, f"BG CDN URL: {bg_url}")
        resp = req_lib.get(bg_url, timeout=60)
        bg_img = Image.open(BytesIO(resp.content)).convert("RGB")
        log(output_dir, f"Generated BG: {bg_img.size[0]}x{bg_img.size[1]}")
        return bg_img
    except Exception as e:
        log(output_dir, f"BG generation failed: {e}", "ERROR")
        return None


def run_inpainting(image_url, mask_url, prompt, negative, strength, output_dir, seed=None):
    """Inpaint background using fal.ai Flux inpainting (legacy fallback)."""
    log(output_dir, f"Inpainting: strength={strength}, prompt='{prompt[:80]}...'")

    payload = {
        "image_url": image_url,
        "mask_url": mask_url,
        "prompt": prompt,
        "strength": strength,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    }
    if negative:
        payload["negative_prompt"] = negative
    if seed is not None:
        payload["seed"] = seed

    try:
        log(output_dir, "Submitting to fal-ai/flux-general/inpainting...")
        handle = fal_client.submit("fal-ai/flux-general/inpainting", arguments=payload)
        result = handle.get()
    except Exception as e:
        log(output_dir, f"Inpainting failed: {e}", "ERROR")
        return None

    images = result.get("images", [])
    if not images:
        log(output_dir, "Inpainting returned no images", "ERROR")
        return None

    result_url = images[0].get("url")
    if not result_url:
        log(output_dir, "Inpainting returned no URL", "ERROR")
        return None

    log(output_dir, f"Inpainting CDN URL: {result_url}")
    import requests
    result_img = Image.open(requests.get(result_url, stream=True, timeout=60).raw).convert("RGB")
    log(output_dir, f"Inpainting result: {result_img.size[0]}x{result_img.size[1]}")
    return result_img


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baroque Surround — Generative Painterly Background")
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--preset", default="baroque", help="Preset name (default: baroque)")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides preset)")
    parser.add_argument("--negative", default=None, help="Custom negative prompt")
    parser.add_argument("--strength", type=float, default=None, help="Inpainting strength (default: from preset)")
    parser.add_argument("--transition", type=float, default=0.04,
                        help="Blend zone as fraction of short edge (default: 0.04)")
    parser.add_argument("--method", default="generate", choices=["generate", "inpaint"],
                        help="BG method: 'generate' (from scratch, default) or 'inpaint' (replace existing BG)")
    parser.add_argument("--noise", action="store_true",
                        help="Add noise/pixelation to BG before inpainting (only for --method inpaint)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--auto-correct", action="store_true", help="Enable Gemini evaluation")
    parser.add_argument("--max-corrections", type=int, default=2, help="Max auto-correction rounds")
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")
    parser.add_argument("--list-presets", action="store_true", help="List all presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{'Preset':<20} Strength  Description")
        print("=" * 90)
        for name, p in PRESETS.items():
            desc = p["prompt"][:65] + "..." if len(p["prompt"]) > 65 else p["prompt"]
            print(f"  {name:<18} {p['strength']:.2f}     {desc}")
        print(f"\nTotal: {len(PRESETS)} presets")
        sys.exit(0)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Resolve preset / custom prompt
    if args.prompt:
        inpaint_prompt = args.prompt
        inpaint_negative = args.negative or "modern, digital, text, watermark"
        inpaint_strength = args.strength or 0.95
        preset_name = "Custom"
    else:
        if args.preset not in PRESETS:
            print(f"ERROR: Unknown preset '{args.preset}'. Use --list-presets.")
            sys.exit(1)
        preset = PRESETS[args.preset]
        inpaint_prompt = preset["prompt"]
        inpaint_negative = args.negative or preset.get("negative", "")
        inpaint_strength = args.strength if args.strength is not None else preset["strength"]
        preset_name = args.preset

    # Derive names
    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    preset_tag = preset_name.replace(" ", "_")[:25]
    suffix = random.randint(10, 99)
    folder_name = f"{model_name}_{source_basename}_{timestamp}_baroque_{preset_tag}_{suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    timings = {}

    log(output_dir, "=" * 60)
    log(output_dir, "BAROQUE SURROUND WORKFLOW START")
    log(output_dir, f"Source:         {source}")
    log(output_dir, f"Preset:         {preset_name}")
    log(output_dir, f"Prompt:         {inpaint_prompt[:100]}")
    log(output_dir, f"Strength:       {inpaint_strength}")
    log(output_dir, f"Transition:     {args.transition}")
    log(output_dir, f"Seed:           {seed}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    # Load original
    img_orig = Image.open(source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), "JPEG", quality=95)
    w, h = img_orig.size
    short_edge = min(w, h)

    # --- Step 1: Extract subject mask ---
    t0 = time.time()
    log(output_dir, "--- Step 1/5: Extract subject mask (BiRefNet) ---")
    mask, mask_info = build_mask(img_orig, affect="subject", exclude="", output_dir=output_dir, feather=0)
    if mask is None:
        log(output_dir, "Subject extraction failed — cannot proceed", "ERROR")
        sys.exit(1)
    # Ensure mask is same size as original
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.LANCZOS)
    log(output_dir, f"Mask: engine={mask_info['engine']}, coverage={mask_info['coverage_pct']}%")
    mask.save(os.path.join(output_dir, "1_mask_raw.png"))
    timings["extract"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['extract']:.1f}s)")

    # --- Step 2: Build masks ---
    t0 = time.time()
    log(output_dir, "--- Step 2/5: Build masks (tight + bleed) ---")

    mask_arr = np.array(mask)
    mask_binary = (mask_arr > 127).astype(np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)

    # TIGHT inpainting mask: only expand by ~0.5% — let inpainting come RIGHT UP to the subject
    # This prevents the "cutout with 20px gap" look
    tight_expand = max(1, int(short_edge * 0.005))
    mask_tight = ndimage.binary_dilation(mask_binary, structure=struct, iterations=tight_expand)
    mask_tight_pil = Image.fromarray((mask_tight.astype(np.uint8) * 255), "L")
    inpaint_mask = ImageOps.invert(mask_tight_pil)
    inpaint_mask.save(os.path.join(output_dir, "2_inpaint_mask.png"))
    log(output_dir, f"Tight inpaint mask: expand={tight_expand}px (subject hugging)")

    # BLEED mask: where the BG forms can bleed INTO the subject
    # Concentrated on lower body — uses vertical gradient to weight bleed toward bottom
    bleed_depth = max(3, int(short_edge * 0.03))  # how far BG bleeds into subject
    mask_eroded = ndimage.binary_erosion(mask_binary, structure=struct, iterations=bleed_depth)
    # Bleed zone = original mask minus eroded mask (narrow band inside subject edge)
    bleed_zone = mask_binary.astype(np.float32) - mask_eroded.astype(np.float32)
    bleed_zone = np.clip(bleed_zone, 0, 1)

    # Weight bleed toward lower body: linear gradient 0 at top → 1 at bottom
    yy = np.linspace(0, 1, h)[:, np.newaxis]  # (h, 1)
    bleed_gradient = np.clip(yy * 1.5 - 0.3, 0, 1)  # starts at ~20% height, full at ~87%
    bleed_zone *= bleed_gradient
    # Add randomness so bleed isn't uniform — perlin-like noise
    rng_bleed = np.random.RandomState(seed)
    noise_bleed = ndimage.gaussian_filter(rng_bleed.randn(h, w), sigma=max(10, int(short_edge * 0.04)))
    noise_bleed = (noise_bleed - noise_bleed.min()) / (noise_bleed.max() - noise_bleed.min() + 1e-8)
    bleed_zone *= noise_bleed  # zero out ~half the bleed randomly
    bleed_zone = ndimage.gaussian_filter(bleed_zone, sigma=max(2, int(short_edge * 0.01)))
    bleed_zone = np.clip(bleed_zone, 0, 1)
    log(output_dir, f"Bleed zone: depth={bleed_depth}px, gradient bottom-weighted, noisy")

    # COMPOSITE mask: tight protection with bleed holes
    # Where bleed_zone > 0, reduce subject protection so BG shows through
    composite_mask_arr = mask_arr.astype(np.float32) / 255.0
    # Feather the raw mask edges
    feather_px = max(3, int(short_edge * args.transition))
    composite_mask_pil = Image.fromarray(mask_arr).filter(ImageFilter.GaussianBlur(radius=feather_px))
    composite_mask_arr = np.array(composite_mask_pil).astype(np.float32) / 255.0
    # Punch bleed holes: reduce mask where bleed_zone is active
    bleed_strength = 0.6  # how much BG can show through in bleed zones (0=none, 1=full)
    composite_mask_arr = composite_mask_arr * (1.0 - bleed_zone * bleed_strength)
    composite_mask_arr = np.clip(composite_mask_arr, 0, 1)
    mask_feathered = Image.fromarray((composite_mask_arr * 255).astype(np.uint8), "L")
    mask_feathered.save(os.path.join(output_dir, "2_composite_mask.png"))
    # Also save bleed zone for debugging
    Image.fromarray((bleed_zone * 255).astype(np.uint8), "L").save(
        os.path.join(output_dir, "2_bleed_zone.png"))

    # Keep expanded mask for BG prep step (needed later)
    mask_expanded_pil = mask_tight_pil

    bg_coverage = np.mean(np.array(inpaint_mask) > 127) * 100
    log(output_dir, f"BG area to inpaint: {bg_coverage:.1f}%")
    timings["mask_prep"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['mask_prep']:.1f}s)")

    # --- Step 3: Generate or inpaint background ---
    t0 = time.time()
    import requests as req_lib

    if args.method == "generate":
        # GENERATE BG FROM SCRATCH — independent of source photo's darkness
        log(output_dir, "--- Step 3/5: Generate BG from scratch (text-to-image) ---")
        # Add "NO person" to prevent generating figures in the BG
        bg_prompt = inpaint_prompt + ", NO person, NO figure, NO face, just abstract painterly forms and shapes"
        bg_result = generate_bg(bg_prompt, w, h, output_dir, seed=seed)
        if bg_result is None:
            log(output_dir, "BG generation failed — cannot proceed", "ERROR")
            sys.exit(1)
        if bg_result.size != (w, h):
            bg_result = bg_result.resize((w, h), Image.LANCZOS)
        inpainted = bg_result
        inpainted.save(os.path.join(output_dir, "3_generated_bg.jpg"), "JPEG", quality=95)
    else:
        # INPAINT — replace existing BG via Flux inpainting (works better on light BGs)
        log(output_dir, "--- Step 3/5: Inpaint surround via fal.ai ---")
        log(output_dir, "Uploading image to fal CDN...")
        image_url = upload_pil_image(img_orig, fmt="JPEG")
        log(output_dir, "Uploading inpainting mask to fal CDN...")
        mask_url = upload_pil_image(inpaint_mask, fmt="PNG")
        inpainted = run_inpainting(
            image_url, mask_url, inpaint_prompt, inpaint_negative,
            inpaint_strength, output_dir, seed=seed,
        )
        if inpainted is None:
            log(output_dir, "Inpainting failed — cannot proceed", "ERROR")
            sys.exit(1)
        if inpainted.size != (w, h):
            inpainted = inpainted.resize((w, h), Image.LANCZOS)
        inpainted.save(os.path.join(output_dir, "3_inpainted_raw.jpg"), "JPEG", quality=95)

    timings["bg"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['bg']:.1f}s)")

    # --- Step 4: Composite subject back ---
    t0 = time.time()
    log(output_dir, "--- Step 4/5: Composite subject onto inpainted surround ---")

    # Optional: light bilateral filter on surround area for oil-painting smoothness
    # Apply bilateral only to the background region
    try:
        import cv2
        inpainted_arr = np.array(inpainted)
        # Bilateral filter: d=9, sigmaColor=75, sigmaSpace=75
        smoothed_arr = cv2.bilateralFilter(inpainted_arr, d=9, sigmaColor=75, sigmaSpace=75)
        # Apply smoothing only to background (where inpaint_mask is white / subject mask is black)
        bg_weight = np.array(inpaint_mask).astype(np.float32) / 255.0
        bg_weight_3ch = bg_weight[:, :, np.newaxis]
        blended_bg = (smoothed_arr * bg_weight_3ch + inpainted_arr * (1 - bg_weight_3ch)).astype(np.uint8)
        inpainted_smoothed = Image.fromarray(blended_bg)
        log(output_dir, "Applied bilateral filter to surround area")
    except ImportError:
        log(output_dir, "cv2 not available — skipping bilateral filter", "WARN")
        inpainted_smoothed = inpainted

    # --- Color harmonization: match subject edge tones to BG ---
    # Sample BG colors near the subject boundary and shift subject edges slightly toward them
    orig_arr = np.array(img_orig).astype(np.float32)
    inpainted_arr = np.array(inpainted_smoothed).astype(np.float32)
    mask_arr_f = np.array(mask_feathered).astype(np.float32) / 255.0

    # Create a narrow edge band (subject boundary zone)
    edge_inner = (mask_arr_f > 0.3) & (mask_arr_f < 0.8)  # transition zone
    if edge_inner.any():
        # Average BG color in the zone just outside the subject
        bg_zone = mask_arr_f < 0.4
        if bg_zone.any():
            bg_mean = inpainted_arr[bg_zone].mean(axis=0)  # (3,)
            subj_zone = mask_arr_f > 0.6
            if subj_zone.any():
                subj_mean = orig_arr[subj_zone].mean(axis=0)
                # Shift subject edge colors 15% toward BG average (gentle color grading)
                color_shift = (bg_mean - subj_mean) * 0.15
                # Apply shift only in the transition zone, fading with mask
                edge_weight = np.clip((0.7 - mask_arr_f) / 0.4, 0, 1)  # 1 at mask=0.3, 0 at mask=0.7
                edge_weight_3ch = edge_weight[:, :, np.newaxis]
                orig_arr = orig_arr + color_shift[np.newaxis, np.newaxis, :] * edge_weight_3ch
                orig_arr = np.clip(orig_arr, 0, 255)
                log(output_dir, f"Color harmonization: shifted edges by {color_shift.astype(int)} toward BG")

    # --- Light wrap: bleed BG light/color into subject edges ---
    # Heavily blur the inpainted BG and let it spill slightly into the subject
    light_wrap_radius = max(5, int(short_edge * 0.04))
    bg_blurred = inpainted_smoothed.filter(ImageFilter.GaussianBlur(radius=light_wrap_radius))
    bg_blurred_arr = np.array(bg_blurred).astype(np.float32)

    # Light wrap zone: narrow band just inside the subject edge
    # Where mask is 0.5-0.85, blend 20% of blurred BG into subject
    wrap_weight = np.clip((0.85 - mask_arr_f) / 0.35, 0, 1) * np.clip(mask_arr_f / 0.5, 0, 1)
    wrap_strength = 0.2  # how much BG light bleeds in
    wrap_weight_3ch = (wrap_weight * wrap_strength)[:, :, np.newaxis]
    orig_arr = orig_arr * (1.0 - wrap_weight_3ch) + bg_blurred_arr * wrap_weight_3ch
    orig_arr = np.clip(orig_arr, 0, 255)
    log(output_dir, f"Light wrap applied: radius={light_wrap_radius}px, strength={wrap_strength}")

    # Composite: use feathered mask to blend (color-matched) subject onto inpainted background
    blend_weight_3ch = mask_arr_f[:, :, np.newaxis]
    composite_arr = orig_arr * blend_weight_3ch + inpainted_arr * (1 - blend_weight_3ch)
    composite = Image.fromarray(np.clip(composite_arr, 0, 255).astype(np.uint8))
    composite.save(os.path.join(output_dir, "4_composite.jpg"), "JPEG", quality=95)

    timings["composite"] = time.time() - t0
    log(output_dir, f"Step 4 done ({timings['composite']:.1f}s)")

    # --- Step 5: Evaluate & output ---
    t0 = time.time()
    log(output_dir, "--- Step 5/5: Quality evaluation & output ---")

    quality_final = check_image_quality(composite, "FINAL", output_dir)

    eval_result = None
    if args.auto_correct:
        eval_result = evaluate_with_gemini(composite, output_dir, original_img=img_orig)
    else:
        eval_result = evaluate_with_gemini(composite, output_dir, original_img=img_orig)

    final_img = composite
    final_path = os.path.join(output_dir, "4_composite.jpg")

    # Copy final to finals folder
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared")
    finals_dir = os.path.join(os.path.expanduser(local_out), "finals")
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
        push_image(finals_dest, title=f"Baroque — {src_name}", body=f"{preset_name}")
        log(output_dir, "Pushed to phone")
    except Exception as e:
        log(output_dir, f"Push notification failed: {e}", "WARN")

    # Copy script for reproducibility
    try:
        script_path = os.path.abspath(__file__)
        shutil.copy2(script_path, os.path.join(output_dir, f"workflow_script_{os.path.basename(script_path)}"))
    except Exception:
        pass

    timings["output"] = time.time() - t0
    log(output_dir, f"Step 5 done ({timings['output']:.1f}s)")

    # --- Summary ---
    total = sum(timings.values())
    score_str = f"{eval_result['score']}/10 — {eval_result.get('critique', '')}" if eval_result else "N/A"

    print(f"""
============================================================
  BAROQUE SURROUND SUMMARY
============================================================
  Source:          {source}
  Preset:          {preset_name}
  Strength:        {inpaint_strength}
  Transition:      {args.transition}
  Seed:            {seed}

  Step Timings:
    1. Extract subject        {timings.get('extract', 0):>8.1f}s
    2. Mask expand/feather    {timings.get('mask_prep', 0):>8.1f}s
    3. Generate/inpaint BG     {timings.get('bg', 0):>8.1f}s
    4. Composite              {timings.get('composite', 0):>8.1f}s
    5. Evaluate & output      {timings.get('output', 0):>8.1f}s
    TOTAL                     {total:>8.1f}s

  Quality Report:
    Final image:   brightness={quality_final['brightness']}  contrast={quality_final['contrast']}  entropy={quality_final['entropy']}
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}
============================================================""")


if __name__ == "__main__":
    main()
