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
        "prompt": "large flowing amorphous organic shapes and billowing drapery, baroque oil painting, dramatic chiaroscuro, luminous glazing, Bouguereau and Caravaggio, warm ochre cool blue-grey cream, smooth blended brushwork, sweeping undulating forms radiating from center",
        "negative": "modern, digital, sharp edges, text, watermark, flat colors, cartoon, solid color background",
        "strength": 0.95,
    },
    "renaissance": {
        "prompt": "large soft amorphous forms of golden light and flowing draped silk fabric, sfumato Renaissance oil painting, Raphael da Vinci, olive warm brown soft blue, luminous atmospheric depth, billowing organic forms",
        "negative": "modern, digital, harsh lighting, text, watermark, flat background, solid color",
        "strength": 0.92,
    },
    "dark-romantic": {
        "prompt": "large swirling amorphous storm forms and turbulent abstract shapes, dark romantic oil painting, Delacroix Turner, dark blue warm amber charcoal copper, flowing organic masses, dramatic atmospheric turbulence",
        "negative": "bright, cheerful, flat, text, watermark, cartoon, solid background",
        "strength": 0.95,
    },
    "ethereal": {
        "prompt": "large flowing amorphous luminous cloud forms and soft ethereal mist, dreamy angelic, billowing organic shapes in pearl ivory pale gold soft blue, divine radiance, sweeping undulating cloud-like masses",
        "negative": "dark, gritty, harsh, text, watermark, modern, flat background",
        "strength": 0.93,
    },
    "smoke": {
        "prompt": "large visible swirling smoke plumes and flowing amorphous grey volumetric forms, dramatic single light source illuminating billowing smoke, warm grey amber cream emerging from shadows, Caravaggio chiaroscuro, dense volumetric smoke clouds",
        "negative": "flat black, solid black, empty background, text, watermark, plain background",
        "strength": 0.93,
    },
    "underwater": {
        "prompt": "deep underwater scene with volumetric light rays penetrating dark ocean water, large flowing organic jellyfish-like forms and bioluminescent particles, swirling ocean currents carrying soft blue green teal glowing shapes, deep sea atmosphere",
        "negative": "text, watermark, surface, sky, dry, land, flat",
        "strength": 0.93,
    },
    "ink-water": {
        "prompt": "large flowing ink drops dissolving in water, organic amorphous spreading ink forms in deep indigo black and warm sienna, mesmerizing fluid dynamics, billowing ink tendrils and blooming clouds of pigment in clear water",
        "negative": "text, watermark, flat, solid color, dry, paper",
        "strength": 0.93,
    },
    "aurora": {
        "prompt": "sweeping northern lights aurora borealis forms, large flowing luminous curtains of green teal purple pink light against dark starry sky, organic undulating ribbons of light, atmospheric glow",
        "negative": "text, watermark, flat, daylight, sun, bright",
        "strength": 0.93,
    },
    "silk": {
        "prompt": "large flowing luxurious silk fabric forms billowing in wind, organic draping shapes in rich burgundy gold ivory, volumetric folds catching dramatic light, Renaissance drapery study, sensual flowing textile",
        "negative": "text, watermark, flat, modern, digital, hard edges",
        "strength": 0.93,
    },
    "embers": {
        "prompt": "swirling embers and warm smoke forms rising in dramatic updraft, glowing orange sparks and flowing ash shapes against dark background, volumetric fire glow, warm amber red black, cinematic atmosphere",
        "negative": "text, watermark, flat, bright, daylight, cold",
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
    log(output_dir, "--- Step 2/6: Build masks (tight edge + spot bleeds) ---")

    mask_arr = np.array(mask).astype(np.float32) / 255.0
    mask_binary = (mask_arr > 0.5).astype(np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)

    # TIGHT feather: 1-2 pixels only — no soft halo
    feather_px = max(1, min(2, int(short_edge * 0.002)))
    mask_feathered_arr = np.array(
        Image.fromarray((mask_binary * 255).astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(radius=feather_px))
    ).astype(np.float32) / 255.0

    # SPOT BLEEDS: 2-3 locations on the body outline where BG "engulfs" the subject
    # These are large, organic blobs that eat into the subject edge
    rng_bleed = np.random.RandomState(seed + 33)
    # Find contour points (edge of subject mask)
    edge = ndimage.binary_dilation(mask_binary, struct, 1).astype(np.float32) - mask_binary.astype(np.float32)
    edge_ys, edge_xs = np.where(edge > 0.5)

    bleed_mask = np.zeros((h, w), dtype=np.float32)
    if len(edge_ys) > 0:
        num_spots = rng_bleed.randint(2, 4)  # 2-3 spots
        # Pick spots biased toward lower body (y > 40% of image height)
        lower_idx = edge_ys > h * 0.4
        if lower_idx.any():
            candidate_ys = edge_ys[lower_idx]
            candidate_xs = edge_xs[lower_idx]
        else:
            candidate_ys, candidate_xs = edge_ys, edge_xs

        for _ in range(num_spots):
            idx = rng_bleed.randint(0, len(candidate_ys))
            cy, cx = int(candidate_ys[idx]), int(candidate_xs[idx])
            # Blob radius: 3-8% of short edge
            blob_r = int(short_edge * rng_bleed.uniform(0.03, 0.08))
            # Gaussian blob centered on the edge point, extending INTO the subject
            yy_b, xx_b = np.ogrid[0:h, 0:w]
            dist_sq = (yy_b - cy) ** 2 + (xx_b - cx) ** 2
            blob = np.exp(-dist_sq / (2 * (blob_r * 0.5) ** 2))
            # Only apply where currently inside subject (engulf effect)
            blob *= mask_binary
            bleed_mask = np.maximum(bleed_mask, blob)
            log(output_dir, f"  Bleed spot at ({cx},{cy}), radius={blob_r}px")

        # Smooth and cap
        bleed_mask = ndimage.gaussian_filter(bleed_mask, sigma=max(2, int(short_edge * 0.008)))
        bleed_mask = np.clip(bleed_mask, 0, 1)

    # Punch bleed holes into the composite mask
    bleed_strength = 0.7
    composite_mask_arr = mask_feathered_arr * (1.0 - bleed_mask * bleed_strength)
    composite_mask_arr = np.clip(composite_mask_arr, 0, 1)

    mask_feathered = Image.fromarray((composite_mask_arr * 255).astype(np.uint8), "L")
    mask_feathered.save(os.path.join(output_dir, "2_composite_mask.png"))
    Image.fromarray((bleed_mask * 255).astype(np.uint8), "L").save(
        os.path.join(output_dir, "2_bleed_spots.png"))

    # Inpaint mask for the generate path (not used in generate mode, but needed for inpaint fallback)
    tight_expand = max(1, int(short_edge * 0.003))
    mask_tight = ndimage.binary_dilation(mask_binary, structure=struct, iterations=tight_expand)
    mask_tight_pil = Image.fromarray((mask_tight.astype(np.uint8) * 255), "L")
    inpaint_mask = ImageOps.invert(mask_tight_pil)
    mask_expanded_pil = mask_tight_pil

    bg_coverage = np.mean(np.array(inpaint_mask) > 127) * 100
    log(output_dir, f"Masks: feather={feather_px}px, {num_spots if len(edge_ys) > 0 else 0} bleed spots, BG={bg_coverage:.1f}%")
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

    # --- Step 4: Color-match subject to BG ---
    t0 = time.time()
    log(output_dir, "--- Step 4/6: Color-match subject to BG ---")
    import cv2

    bg_arr = np.array(inpainted).astype(np.float32)
    orig_arr = np.array(img_orig).astype(np.float32)
    mask_arr_f = np.array(mask_feathered).astype(np.float32) / 255.0

    # LAB color transfer: shift subject colors toward BG color distribution
    # This is more perceptually uniform than RGB shifting
    try:
        orig_lab = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(np.array(inpainted), cv2.COLOR_RGB2LAB).astype(np.float32)
        subj_pixels = mask_arr_f > 0.5
        bg_pixels_m = mask_arr_f < 0.3
        if subj_pixels.any() and bg_pixels_m.any():
            for ch in range(3):
                s_mean = orig_lab[:, :, ch][subj_pixels].mean()
                s_std = orig_lab[:, :, ch][subj_pixels].std() + 1e-8
                b_mean = bg_lab[:, :, ch][bg_pixels_m].mean()
                b_std = bg_lab[:, :, ch][bg_pixels_m].std() + 1e-8
                # Partial transfer: 25% shift toward BG distribution
                shift = 0.25
                new_mean = s_mean + (b_mean - s_mean) * shift
                new_std = s_std + (b_std - s_std) * shift * 0.5
                # Apply only to subject area
                shifted = (orig_lab[:, :, ch] - s_mean) * (new_std / s_std) + new_mean
                # Blend: full effect in transition zone, zero deep inside subject
                edge_w = np.clip(1.0 - (mask_arr_f - 0.3) / 0.5, 0, 1)  # 1 at edge, 0 deep inside
                orig_lab[:, :, ch] = orig_lab[:, :, ch] * (1 - edge_w) + shifted * edge_w
            orig_lab = np.clip(orig_lab, 0, 255).astype(np.uint8)
            orig_arr = cv2.cvtColor(orig_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
            log(output_dir, "LAB color transfer: 25% shift toward BG colors at subject edges")
    except Exception as e:
        log(output_dir, f"LAB color transfer failed: {e}", "WARN")

    timings["color"] = time.time() - t0
    log(output_dir, f"Step 4 done ({timings['color']:.1f}s)")

    # --- Step 5: Composite + foreground overlay ---
    t0 = time.time()
    log(output_dir, "--- Step 5/6: Composite + foreground overlay ---")

    # Main composite: subject over BG using the tight mask with bleed spots
    mask_3ch = mask_arr_f[:, :, np.newaxis]
    composite_arr = orig_arr * mask_3ch + bg_arr * (1 - mask_3ch)

    # --- FOREGROUND OVERLAY: generate a second BG and overlay on lower body ---
    # This creates the "engulfing" effect where forms wrap AROUND the subject
    if args.method == "generate":
        log(output_dir, "Generating foreground overlay (same style, placed over lower body)...")
        fg_prompt = inpaint_prompt + ", NO person, NO figure, NO face, wispy foreground elements, semi-transparent flowing forms"
        fg_result = generate_bg(fg_prompt, w, h, output_dir, seed=seed + 999)
        if fg_result is not None:
            if fg_result.size != (w, h):
                fg_result = fg_result.resize((w, h), Image.LANCZOS)
            fg_arr = np.array(fg_result).astype(np.float32)

            # Foreground mask: only show on lower 60% of image, NOT on face/upper body
            yy_fg = np.linspace(0, 1, h)[:, np.newaxis]
            fg_visibility = np.clip((yy_fg - 0.4) / 0.3, 0, 1)  # 0 above 40%, ramps to 1 by 70%
            fg_visibility = np.broadcast_to(fg_visibility, (h, w)).copy()
            # Reduce visibility where subject face is (upper part of subject)
            yy_full = np.broadcast_to(yy_fg, (h, w))
            face_protection = np.clip(mask_arr_f * (1.0 - yy_full * 0.5), 0, 1)
            fg_visibility = fg_visibility * (1.0 - face_protection * 0.8)
            # Make it patchy — only some areas show through
            rng_fg = np.random.RandomState(seed + 777)
            fg_noise = ndimage.gaussian_filter(rng_fg.randn(h, w), sigma=max(15, int(short_edge * 0.06)))
            fg_noise = (fg_noise - fg_noise.min()) / (fg_noise.max() - fg_noise.min() + 1e-8)
            fg_visibility *= np.clip(fg_noise * 2 - 0.5, 0, 1)  # threshold: ~50% coverage
            fg_visibility = ndimage.gaussian_filter(fg_visibility, sigma=max(3, int(short_edge * 0.015)))

            # Use brightness of FG to modulate opacity (brighter = more visible, dark = transparent)
            fg_brightness = np.mean(fg_arr, axis=2) / 255.0
            fg_visibility *= np.clip(fg_brightness * 1.5, 0, 1)

            fg_opacity = 0.5  # max overlay strength
            fg_weight = np.clip(fg_visibility * fg_opacity, 0, 1)[:, :, np.newaxis]
            composite_arr = composite_arr * (1 - fg_weight) + fg_arr * fg_weight
            log(output_dir, f"Foreground overlay applied (opacity={fg_opacity}, lower body)")
        else:
            log(output_dir, "Foreground generation failed — skipping overlay", "WARN")

    composite = Image.fromarray(np.clip(composite_arr, 0, 255).astype(np.uint8))
    composite.save(os.path.join(output_dir, "5_composite.jpg"), "JPEG", quality=95)

    timings["composite"] = time.time() - t0
    log(output_dir, f"Step 5 done ({timings['composite']:.1f}s)")

    # --- Step 6: Evaluate & output ---
    t0 = time.time()
    log(output_dir, "--- Step 6/6: Quality evaluation & output ---")

    quality_final = check_image_quality(composite, "FINAL", output_dir)

    eval_result = None
    if args.auto_correct:
        eval_result = evaluate_with_gemini(composite, output_dir, original_img=img_orig)
    else:
        eval_result = evaluate_with_gemini(composite, output_dir, original_img=img_orig)

    final_img = composite
    final_path = os.path.join(output_dir, "5_composite.jpg")

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
    4. Color match            {timings.get('color', 0):>8.1f}s
    5. Composite + FG overlay {timings.get('composite', 0):>8.1f}s
    6. Evaluate & output      {timings.get('output', 0):>8.1f}s
    TOTAL                     {total:>8.1f}s

  Quality Report:
    Final image:   brightness={quality_final['brightness']}  contrast={quality_final['contrast']}  entropy={quality_final['entropy']}
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}
============================================================""")


if __name__ == "__main__":
    main()
