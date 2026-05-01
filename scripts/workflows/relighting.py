#!/home/rong/openclaw-venv/bin/python3
"""
Lighting Re-imagination Workflow

Instead of stylizing, this script re-lights photographs using IC-Light V2.
Takes a portrait photo, extracts the subject (BiRefNet), then uses IC-Light
to relight with dramatic setups: rim light, spotlight, colored gels, etc.

The result looks photographic (not painterly), just with completely new lighting.

Usage:
    python relighting.py --source photo.jpg --lighting "Dramatic Rim"
    python relighting.py --source photo.jpg --lighting "Neon Gels" --strength 0.9
    python relighting.py --source photo.jpg --list-presets
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
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageEnhance

# Shared masking module (BiRefNet / body-segment)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask, add_affect_args

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Lighting Presets
# ---------------------------------------------------------------------------
LIGHTING_PRESETS = {
    # Dramatic single-source
    "Dramatic Rim": {
        "prompt": "dramatic rim light from behind the subject, strong backlight silhouette edge glow, dark moody front, cinematic studio lighting, professional portrait photography",
        "negative": "flat lighting, overexposed, washed out",
    },
    "Spotlight": {
        "prompt": "single dramatic spotlight from above, sharp focused beam, deep shadows, theatrical stage lighting, Rembrandt triangle on cheek, professional portrait",
        "negative": "flat lighting, multiple light sources, overexposed",
    },
    "Low Key": {
        "prompt": "low key lighting, single side light source, deep rich shadows, minimal fill, dramatic chiaroscuro, fine art portrait photography",
        "negative": "bright, high key, flat lighting, overexposed",
    },
    "High Key": {
        "prompt": "high key lighting, bright even illumination, soft wraparound light, minimal shadows, clean bright background, fashion editorial portrait",
        "negative": "dark, moody, heavy shadows, underexposed",
    },

    # Colored gels / creative
    "Neon Gels": {
        "prompt": "split lighting with neon blue gel on left and neon pink/magenta gel on right, vivid color contrast, cyberpunk portrait photography, night club atmosphere",
        "negative": "natural lighting, warm tones, daylight",
    },
    "Teal & Orange": {
        "prompt": "cinematic color grading, warm orange key light from one side, cool teal fill light from opposite side, Hollywood color contrast, portrait photography",
        "negative": "flat colors, monochromatic, dull",
    },
    "Red Drama": {
        "prompt": "dramatic red lighting, deep crimson illumination, dark shadows, intense moody atmosphere, theatrical red spotlight, portrait photography",
        "negative": "natural colors, daylight, bright",
    },
    "Golden Hour": {
        "prompt": "warm golden hour sunlight, soft directional warm light, long shadows, golden skin tones, magic hour glow, natural portrait photography",
        "negative": "cold lighting, blue tones, artificial light",
    },

    # Natural / environmental
    "Window Light": {
        "prompt": "soft natural window light from the side, gentle fall-off into shadow, Vermeer-style lighting, intimate indoor portrait, diffused daylight",
        "negative": "harsh shadows, artificial lighting, studio strobes",
    },
    "Overcast Soft": {
        "prompt": "soft overcast daylight, even diffused illumination, gentle shadows, beauty portrait lighting, cloudy day natural light",
        "negative": "harsh sun, hard shadows, dramatic contrast",
    },
    "Candlelight": {
        "prompt": "warm candlelight illumination, flickering warm glow, intimate low-light atmosphere, romantic portrait, warm skin tones, dark background",
        "negative": "daylight, cool tones, bright, modern lighting",
    },

    # Studio setups
    "Butterfly": {
        "prompt": "butterfly lighting setup, key light directly above camera pointing down, small shadow under nose, glamour portrait, fashion editorial lighting",
        "negative": "side lighting, flat, unflattering",
    },
    "Split Light": {
        "prompt": "dramatic split lighting, half face illuminated half in deep shadow, single hard side light, artistic portrait, strong contrast",
        "negative": "flat lighting, both sides even, fill light",
    },
    "Beauty Dish": {
        "prompt": "beauty dish lighting, soft specular wrap-around light, gentle catch light in eyes, fashion beauty portrait, smooth even illumination, professional studio",
        "negative": "harsh shadows, dramatic, moody, underexposed",
    },

    # Atmospheric / creative
    "Underwater Caustics": {
        "prompt": "underwater light caustics, dappled light patterns through water surface, blue-green aquatic glow, ethereal submerged lighting, swimming pool light",
        "negative": "dry environment, harsh sun, artificial studio",
    },
    "Moonlight": {
        "prompt": "cool blue moonlight, nocturnal illumination, subtle silver-blue glow, night portrait, mysterious ambient light, soft shadows",
        "negative": "warm tones, daylight, bright, golden",
    },
    "Neon Signs": {
        "prompt": "colored neon sign reflections on skin, urban night portrait, multiple colored light sources from neon signs, city lights bokeh, wet streets reflecting neon",
        "negative": "natural lighting, daylight, clean studio",
    },
    "Firelight": {
        "prompt": "warm firelight from below, campfire orange glow, flickering warm illumination, dark surroundings, intimate atmosphere, upward shadows",
        "negative": "cool lighting, daylight, even illumination",
    },
    "Laser": {
        "prompt": "thin laser beam lines of red and green light cutting through haze, concert laser lighting, sharp light rays through fog, futuristic portrait",
        "negative": "natural light, soft diffused, daylight",
    },

    # New additions
    "Hard Midday Sun": {
        "prompt": "harsh overhead midday sunlight, hard direct sun, sharp short shadows under brow and nose, high contrast outdoor portrait, bright daylight, clear sky",
        "negative": "soft diffused light, overcast, golden hour, indoor",
    },
    "Stage Backlight": {
        "prompt": "powerful stage backlight from behind, strong rim glow outlining hair and shoulders, silhouette edge light, dark front, smoky concert stage atmosphere",
        "negative": "front light, flat lighting, daylight, even illumination",
    },
    "Blue Hour": {
        "prompt": "cool blue hour twilight, soft cobalt-blue ambient light just after sunset, gentle gradient sky illumination, calm cinematic dusk portrait",
        "negative": "warm tones, golden hour, harsh sun, daytime",
    },
    "Projector Patterns": {
        "prompt": "patterned shadows from a window blind or leaf gobo cast across the subject, slatted light stripes, dappled shadow patterns on skin, cinematic noir lighting",
        "negative": "flat even lighting, no patterns, soft diffused",
    },
    "Lightning Flash": {
        "prompt": "sudden lightning flash illumination, brief intense cool blue-white burst light from one side, deep dark surroundings, dramatic stormy atmosphere portrait",
        "negative": "warm tones, soft, daylight, even illumination",
    },
    "TV Glow": {
        "prompt": "flickering TV glow lighting the face, cool blue cathode-ray tint, dim dark room, late-night ambient screen light, intimate nocturnal portrait",
        "negative": "daylight, warm tones, bright, studio lighting",
    },
    "Stained Glass": {
        "prompt": "colored light streaming through stained glass window, multicolored projected patches of red blue and gold light on the subject and surroundings, cathedral atmosphere",
        "negative": "monochromatic light, flat lighting, harsh shadows, modern",
    },
    "Practical Bulb": {
        "prompt": "single warm tungsten practical bulb light source visible in scene, intimate lamp-lit interior, warm fall-off shadows, cozy domestic portrait, hard small light source",
        "negative": "studio strobe, daylight, soft wraparound, cool tones",
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
# API Wrappers
# ---------------------------------------------------------------------------
def _get_fal_key():
    key = os.environ.get("FAL_API_KEY")
    if not key:
        raise EnvironmentError("FAL_API_KEY not set")
    return key


def extract_subject_on_black(img, mask):
    """Composite subject onto black background for IC-Light input."""
    subject_on_black = Image.new("RGB", img.size, (0, 0, 0))
    subject_on_black.paste(img, mask=mask)
    return subject_on_black


def run_iclight(subject_img, prompt, negative_prompt, output_dir, seed=None,
                lowres_denoise=0.85, highres_denoise=0.5, guidance_scale=2.5,
                num_steps=28, enable_hr=True):
    """Relight the subject using IC-Light V2 on fal.ai."""
    log(output_dir, f"IC-Light V2: '{prompt[:80]}...' (denoise_lr={lowres_denoise}, denoise_hr={highres_denoise})")

    headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}

    # Encode subject image
    buf = BytesIO()
    subject_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "prompt": prompt,
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "negative_prompt": negative_prompt or "lowres, bad anatomy, bad hands, cropped, worst quality",
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "lowres_denoise": lowres_denoise,
        "highres_denoise": highres_denoise,
        "enable_hr_fix": enable_hr,
        "output_format": "jpeg",
        "num_images": 1,
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post("https://fal.run/fal-ai/iclight-v2", headers=headers,
                                 json=payload, timeout=600)
    except requests.RequestException as e:
        log(output_dir, f"IC-Light request failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"IC-Light failed ({response.status_code}): {response.text[:300]}", "ERROR")
        return None

    data = response.json()
    images = data.get("images", [])
    if not images:
        log(output_dir, "IC-Light returned no images", "ERROR")
        return None

    result_url = images[0].get("url")
    if not result_url:
        log(output_dir, "IC-Light returned no image URL", "ERROR")
        return None

    log(output_dir, f"IC-Light CDN URL: {result_url}")
    result_img = Image.open(requests.get(result_url, stream=True, timeout=30).raw).convert("RGB")
    log(output_dir, f"IC-Light result: {result_img.size[0]}x{result_img.size[1]}")
    return result_img


# ---------------------------------------------------------------------------
# Gemini Evaluation (same as main script)
# ---------------------------------------------------------------------------
def _img_to_b64(img, max_size=1024):
    img_resized = img.copy()
    img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_EVAL_PROMPT = """\
You are a professional photography lighting director evaluating a re-lit photograph.
If you see TWO images, the first is the ORIGINAL and the second is the RE-LIT result — compare them.

Evaluate the RE-LIT image on these criteria:
1. Lighting quality: does the new lighting look natural and physically plausible?
2. Subject integrity: does the person look anatomically correct? No warped faces, extra fingers, melted features
3. Light consistency: are shadows, highlights, and reflections consistent with a single lighting setup?
4. Skin quality: does the skin look natural, not plastic/waxy/over-smoothed?
5. Color coherence: do the colors work together? No unnatural color casts or banding?
6. Overall impact: does the re-lighting improve the photo or make it worse?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "subject_distorted", "face_warped", "too_dark", "too_bright", \
"unnatural_shadows", "plastic_skin", "color_banding", "artifacts", "lighting_inconsistent", \
"anatomy_wrong", "over_smoothed", "too_blurry", "face_sideways">],
  "adjustments": {
    "lowres_denoise": <null or suggested float 0.1-1.0>,
    "highres_denoise": <null or suggested float 0.1-1.0>,
    "guidance_scale": <null or suggested float 1-10>,
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
            parts.insert(0, {"text": "ORIGINAL (before relighting):"})
            parts.insert(1, {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"text": "RE-LIT (after processing):\n\n" + _EVAL_PROMPT})
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
# Main Workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lighting Re-imagination Workflow using IC-Light V2")
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--lighting", required=False, help="Lighting preset name (use --list-presets to see all)")
    parser.add_argument("--prompt", default=None, help="Custom lighting prompt (overrides preset)")
    parser.add_argument("--negative", default=None, help="Custom negative prompt")
    parser.add_argument("--lowres-denoise", type=float, default=0.85, help="Low-res denoising strength (default: 0.85)")
    parser.add_argument("--highres-denoise", type=float, default=0.5, help="High-res denoising strength (default: 0.5) — lower = more faithful to original")
    parser.add_argument("--guidance-scale", type=float, default=2.5, help="Guidance scale (default: 2.5)")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps (default: 28)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-hr", action="store_true", help="Disable high-res fix")
    parser.add_argument("--bg-blend", type=float, default=0.0,
                        help="Blend original BG back at this opacity (0.0=fully relit BG, 0.5=50%% original BG, 1.0=original BG). Default: 0.0")
    parser.add_argument("--bg-blend-blur", type=int, default=None,
                        help="Mask blur radius for BG blend edge (default: auto 2%% of image, 0=no feathering)")
    parser.add_argument("--auto-correct", action="store_true", help="Enable Gemini evaluation + auto-correction loop")
    parser.add_argument("--max-corrections", type=int, default=2, help="Max auto-correction rounds (default: 2)")
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")
    parser.add_argument("--list-presets", action="store_true", help="List all lighting presets and exit")
    parser.add_argument("--save-stack", action="store_true",
                        help="export pipeline stages as a multi-page TIFF (<finals>__stack.tif)")
    add_affect_args(parser)
    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{'Preset Name':<25} Description")
        print("=" * 80)
        for name, preset in LIGHTING_PRESETS.items():
            desc = preset["prompt"][:70] + "..." if len(preset["prompt"]) > 70 else preset["prompt"]
            print(f"  {name:<23} {desc}")
        print(f"\nTotal: {len(LIGHTING_PRESETS)} presets")
        sys.exit(0)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Resolve lighting prompt
    if args.prompt:
        lighting_prompt = args.prompt
        lighting_negative = args.negative or "lowres, bad anatomy, bad hands, cropped, worst quality"
        lighting_name = "Custom"
    elif args.lighting:
        if args.lighting not in LIGHTING_PRESETS:
            print(f"ERROR: Unknown preset '{args.lighting}'. Use --list-presets to see available presets.")
            sys.exit(1)
        preset = LIGHTING_PRESETS[args.lighting]
        lighting_prompt = preset["prompt"]
        lighting_negative = args.negative or preset.get("negative", "")
        lighting_name = args.lighting
    else:
        print("ERROR: Must specify --lighting <preset> or --prompt '<custom prompt>'")
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
    lighting_tag = lighting_name.replace(" ", "_")[:25]
    suffix = random.randint(10, 99)
    folder_name = f"{model_name}_{source_basename}_{timestamp}_{lighting_tag}_{suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    timings = {}

    log(output_dir, "=" * 60)
    log(output_dir, "RELIGHTING WORKFLOW START")
    log(output_dir, f"Source:         {source}")
    log(output_dir, f"Lighting:       {lighting_name}")
    log(output_dir, f"Prompt:         {lighting_prompt[:100]}")
    log(output_dir, f"LR Denoise:     {args.lowres_denoise}")
    log(output_dir, f"HR Denoise:     {args.highres_denoise}")
    log(output_dir, f"Guidance:       {args.guidance_scale}")
    log(output_dir, f"Steps:          {args.steps}")
    log(output_dir, f"Seed:           {seed}")
    log(output_dir, f"HR Fix:         {not args.no_hr}")
    log(output_dir, f"Affect:         {args.affect}")
    if args.exclude:
        log(output_dir, f"Exclude:        {args.exclude}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    # Save original
    img_orig = Image.open(source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), "JPEG", quality=95)

    # --- Step 1: Extract subject ---
    t0 = time.time()
    log(output_dir, "--- Step 1/4: Extract subject ---")
    mask, mask_info = build_mask(img_orig, affect=args.affect, exclude=args.exclude, output_dir=output_dir)
    if mask is None:
        log(output_dir, "Subject extraction failed — cannot proceed", "ERROR")
        sys.exit(1)
    log(output_dir, f"Mask: engine={mask_info['engine']}, coverage={mask_info['coverage_pct']}%")
    subject_on_black = extract_subject_on_black(img_orig, mask)
    subject_on_black.save(os.path.join(output_dir, "1_subject_on_black.jpg"), "JPEG", quality=95)
    mask.save(os.path.join(output_dir, "1_mask.png"))
    timings["extract"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['extract']:.1f}s)")

    # --- Step 2: Relight with IC-Light V2 ---
    t0 = time.time()
    log(output_dir, "--- Step 2/4: Relight (IC-Light V2) ---")
    relit = run_iclight(
        subject_on_black, lighting_prompt, lighting_negative, output_dir,
        seed=seed, lowres_denoise=args.lowres_denoise, highres_denoise=args.highres_denoise,
        guidance_scale=args.guidance_scale, num_steps=args.steps, enable_hr=not args.no_hr,
    )
    if relit is None:
        log(output_dir, "IC-Light failed — cannot proceed", "ERROR")
        sys.exit(1)

    # Resize to match original if needed
    if relit.size != img_orig.size:
        log(output_dir, f"Resizing relit {relit.size} -> {img_orig.size}")
        relit = relit.resize(img_orig.size, Image.LANCZOS)

    relit.save(os.path.join(output_dir, "2_relit_raw.jpg"), "JPEG", quality=95)

    # Blend original BG back if requested
    if args.bg_blend > 0 and mask is not None:
        log(output_dir, f"Blending original BG back at {args.bg_blend*100:.0f}% opacity")
        # Where mask is LOW (background), blend original back
        # Blur on mask edge for blending — configurable, 0 = hard edge
        if args.bg_blend_blur is not None:
            blur_r = args.bg_blend_blur
        else:
            blur_r = max(10, int(min(img_orig.width, img_orig.height) * 0.02))
        if blur_r > 0:
            mask_soft = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
        else:
            mask_soft = mask
        log(output_dir, f"BG blend mask blur: {blur_r}px")
        mask_arr = np.array(mask_soft).astype(np.float64) / 255.0
        # Invert: 1 = background, 0 = subject
        bg_weight = (1.0 - mask_arr) * args.bg_blend
        bg_weight_3ch = bg_weight[:, :, np.newaxis]
        relit_arr = np.array(relit).astype(np.float64)
        orig_arr = np.array(img_orig).astype(np.float64)
        blended = relit_arr * (1 - bg_weight_3ch) + orig_arr * bg_weight_3ch
        relit = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        relit.save(os.path.join(output_dir, "2_relit_bg_blended.jpg"), "JPEG", quality=95)

    relit.save(os.path.join(output_dir, "2_relit.jpg"), "JPEG", quality=95)
    quality = check_image_quality(relit, "relit", output_dir)
    timings["relight"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['relight']:.1f}s)")

    final_img = relit
    final_path = os.path.join(output_dir, "2_relit.jpg")

    # --- Step 3: Evaluate (Gemini) ---
    t0 = time.time()
    log(output_dir, "--- Step 3/4: Quality evaluation ---")
    quality_final = check_image_quality(final_img, "FINAL", output_dir)

    eval_result = None
    if args.auto_correct:
        eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

        # Auto-correction loop
        if eval_result and eval_result.get("score", 10) < 7:
            for correction_round in range(1, args.max_corrections + 1):
                prev_score = eval_result.get("score", 0)
                adjustments = eval_result.get("adjustments", {})
                log(output_dir, f"--- Auto-correction round {correction_round}/{args.max_corrections} (score={prev_score}/10) ---")

                # Apply adjustments
                lr_denoise = adjustments.get("lowres_denoise") or args.lowres_denoise
                hr_denoise = adjustments.get("highres_denoise") or args.highres_denoise
                guidance = adjustments.get("guidance_scale") or args.guidance_scale
                new_seed = seed + correction_round if adjustments.get("try_different_seed") else seed

                log(output_dir, f"Re-lighting with lr_denoise={lr_denoise}, hr_denoise={hr_denoise}, guidance={guidance}, seed={new_seed}")

                retry_relit = run_iclight(
                    subject_on_black, lighting_prompt, lighting_negative, output_dir,
                    seed=new_seed, lowres_denoise=lr_denoise, highres_denoise=hr_denoise,
                    guidance_scale=guidance, num_steps=args.steps, enable_hr=not args.no_hr,
                )
                if retry_relit is None:
                    log(output_dir, "Correction re-light failed — keeping previous result", "WARN")
                    break

                if retry_relit.size != img_orig.size:
                    retry_relit = retry_relit.resize(img_orig.size, Image.LANCZOS)

                retry_path = os.path.join(output_dir, f"2_relit_r{correction_round}.jpg")
                retry_relit.save(retry_path, "JPEG", quality=95)

                retry_eval = evaluate_with_gemini(retry_relit, output_dir, original_img=img_orig)
                if retry_eval and retry_eval.get("score", 0) > prev_score:
                    log(output_dir, f"Correction improved: {prev_score} -> {retry_eval['score']}")
                    final_img = retry_relit
                    final_path = retry_path
                    eval_result = retry_eval
                    if retry_eval.get("score", 0) >= 7:
                        break
                else:
                    new_score = retry_eval.get("score", "?") if retry_eval else "N/A"
                    log(output_dir, f"Correction round {correction_round} did not improve ({prev_score} -> {new_score}) — stopping", "WARN")
                    break
    else:
        eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

    timings["evaluate"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['evaluate']:.1f}s)")

    # --- Step 4: Output ---
    t0 = time.time()
    log(output_dir, "--- Step 4/4: Output ---")

    # Copy final to finals folder
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates")
    if os.path.exists(final_path):
        finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
        os.makedirs(finals_dir, exist_ok=True)
        finals_name = os.path.basename(output_dir) + ".jpg"
        finals_dest = os.path.join(finals_dir, finals_name)
        with open(final_path, "rb") as f_in:
            with open(finals_dest, "wb") as f_out:
                f_out.write(f_in.read())
        log(output_dir, f"Final copied to: {finals_dest}")

        # --save-stack: aggregate intermediates into a multi-page TIFF
        if args.save_stack:
            try:
                from _layered_tiff import save_stack
                stage_files = [
                    ("00_original",         "0_original.jpg"),
                    ("01_mask",             "1_mask.png"),
                    ("02_subject_on_black", "1_subject_on_black.jpg"),
                    ("03_relit_raw",        "2_relit_raw.jpg"),
                    ("04_relit_bg_blended", "2_relit_bg_blended.jpg"),
                ]
                layers = []
                for name, fname in stage_files:
                    fp = os.path.join(output_dir, fname)
                    if os.path.isfile(fp):
                        layers.append((name, Image.open(fp)))
                layers.append(("99_final", final_img))
                stack_path = os.path.join(finals_dir, os.path.basename(output_dir) + "__stack.tif")
                save_stack(stack_path, layers)
                log(output_dir, f"Stack: {stack_path} ({len(layers)} layers)")
            except Exception as e:
                log(output_dir, f"save-stack failed: {e}", "WARN")

        # Push to phone
        try:
            from notify import push_image
            src_name = os.path.splitext(os.path.basename(args.source))[0]
            push_image(finals_dest, title=f"Relight — {src_name}", body=f"{args.lighting or 'custom'}")
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
    log(output_dir, f"Step 4 done ({timings['output']:.1f}s)")

    # --- Summary ---
    total = sum(timings.values())
    score_str = f"{eval_result['score']}/10 — {eval_result.get('critique', '')}" if eval_result else "N/A"

    print(f"""
============================================================
  RELIGHTING SUMMARY
============================================================
  Source:          {source}
  Lighting:        {lighting_name}
  LR Denoise:      {args.lowres_denoise}
  HR Denoise:      {args.highres_denoise}
  Guidance:        {args.guidance_scale}
  Seed:            {seed}

  Step Timings:
    1. Extract subject        {timings.get('extract', 0):>8.1f}s
    2. Relight (IC-Light)     {timings.get('relight', 0):>8.1f}s
    3. Quality evaluation     {timings.get('evaluate', 0):>8.1f}s
    4. Output                 {timings.get('output', 0):>8.1f}s
    TOTAL                     {total:>8.1f}s

  Quality Report:
    Final image:   brightness={quality_final['brightness']}  contrast={quality_final['contrast']}  entropy={quality_final['entropy']}
    Aesthetic:     {score_str}

  Output:
    Working dir:   {output_dir}
============================================================""")


if __name__ == "__main__":
    main()
