#!/home/rong/openclaw-venv/bin/python3
"""
Anime Stylize — Cinematic semi-realistic anime/manga version of a photo.

Full-image img2img via Tensor Art using an UNCENSORED checkpoint
(Z-Image-Uncensored by default), so NSFW input is not rejected the way
ChatGPT / Flux Kontext / Qwen-Image-Edit / nano-banana reject it.

Preserves pose, camera angle, background layout and composition by running
img2img at a moderate denoise strength (no ControlNet wired here — structure
fidelity is controlled purely via --strength; lower = more faithful, higher =
more fully anime).

Usage:
    python anime_stylize.py --source photo.jpg
    python anime_stylize.py --source a.jpg b.jpg c.jpg          # batch
    python anime_stylize.py --source photo.jpg --strength 0.7   # more anime
    python anime_stylize.py --source photo.jpg --tensor-model <ID>  # try an anime checkpoint
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

import json
import time
import uuid
import shutil
import argparse
import threading
from io import BytesIO
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import requests
from PIL import Image, ImageOps

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
MODEL_DEFAULT = "965126062386242266"  # Z-Image-Uncensored-fp16-v3 (no input filter)
TENSOR_COST = 0.02  # rough per-image estimate

DEFAULT_PROMPT = (
    "Transform the reference photo into a polished cinematic anime illustration while "
    "preserving the exact person's identity, facial structure, proportions, hairstyle, "
    "expression, pose, camera angle, background layout, and overall composition from the "
    "reference photo. "
    "Style: elegant semi-realistic anime / manga illustration, soft painterly rendering, "
    "warm cinematic lighting, refined clean linework, delicate facial features, expressive "
    "eyes, smooth skin shading, subtle blush, natural but idealized anatomy, detailed hair "
    "strands, glossy highlights, soft ambient shadows, rich warm color palette, cozy interior "
    "atmosphere, slightly vintage film grain, high-end illustrated poster look. "
    "Keep the person highly recognizable from the reference photo: same face shape, eye "
    "spacing, nose, mouth, eyebrows, hairline, hairstyle, and expression. Do not beautify them "
    "into a different person. Do not change age, ethnicity, body type, or facial identity. "
    "Use soft golden indoor lighting, gentle rim light, muted background detail, shallow depth "
    "of field, painterly texture, cinematic composition, high detail, tasteful stylization, "
    "elegant mood, natural proportions, realistic fabric/material rendering, subtle specular "
    "highlights, atmospheric warmth."
)

DEFAULT_NEGATIVE = (
    "plastic skin, oversexualized anatomy, exaggerated proportions, distorted face, generic "
    "anime face, changing the person's identity, extra fingers, warped hands, messy linework, "
    "low detail, harsh shadows, uncanny eyes, watermark, text, signature, lowres, bad anatomy, "
    "deformed, blurry, jpeg artifacts"
)

_log_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------
PIPELINE_COST = {"date": "", "cost_today": 0.0}
PIPELINE_COST_LOCK = threading.RLock()


def _pipeline_accrue(amount):
    today = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d")
    with PIPELINE_COST_LOCK:
        if PIPELINE_COST["date"] != today:
            PIPELINE_COST["date"] = today
            PIPELINE_COST["cost_today"] = 0.0
        PIPELINE_COST["cost_today"] = round(PIPELINE_COST["cost_today"] + amount, 4)


# ---------------------------------------------------------------------------
# Logging / keys
# ---------------------------------------------------------------------------
def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    if output_dir:
        with _log_lock:
            with open(os.path.join(output_dir, "workflow.log"), "a") as f:
                f.write(formatted + "\n")


def _get_tensor_key():
    key = os.environ.get("TENSOR_API_KEY")
    if not key:
        raise EnvironmentError("TENSOR_API_KEY not set")
    return key


# ---------------------------------------------------------------------------
# Tensor Art upload / job / stylize
# ---------------------------------------------------------------------------
def upload_to_tensor(image_pil, output_dir):
    """Upload a PIL image to Tensor Art; return (resource_id, width, height)."""
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


def anime_stylize(image_pil, prompt, negative, strength, cfg_scale, steps,
                  output_dir, model_id, seed):
    """Single full-image Tensor Art img2img anime pass."""
    resource_id, w, h = upload_to_tensor(image_pil, output_dir)
    if resource_id is None:
        return None

    diffusion = {
        "width": w, "height": h,
        "prompts": [{"text": prompt, "weight": 1.0}],
        "sdModel": model_id, "steps": steps, "cfgScale": cfg_scale,
        "denoisingStrength": strength, "sampler": "Euler a",
    }
    if negative:
        diffusion["negativePrompts"] = [{"text": negative, "weight": 1.0}]

    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {"type": "INPUT_INITIALIZE",
             "inputInitialize": {"image_resource_id": resource_id, "count": 1, "seed": seed}},
            {"type": "DIFFUSION", "diffusion": diffusion},
        ],
    }
    img_url = run_tensor_job(payload, output_dir)
    if img_url:
        return Image.open(requests.get(img_url, stream=True, timeout=60).raw).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------
def process_one(src, args):
    src = os.path.abspath(os.path.expanduser(src))
    if not os.path.isfile(src):
        print(f"[ERROR] not a file: {src}")
        return None

    src_name = os.path.splitext(os.path.basename(src))[0]
    stamp = datetime.now(ISRAEL_TZ).strftime("%Y%m%d_%H%M%S")
    sfx = uuid.uuid4().hex[:6]
    tag = f"anime_{src_name}_s{args.strength}_{stamp}_{sfx}"
    base = os.path.expanduser(args.local_output_dir or "~/.openclaw/workspace/shared")
    output_dir = os.path.join(base, "tool-outputs-intermediates", tag)
    os.makedirs(output_dir, exist_ok=True)

    log(output_dir, f"Source: {src}")
    log(output_dir, f"model={args.tensor_model} strength={args.strength} cfg={args.cfg_scale} "
                    f"steps={args.steps} seed={args.seed}")

    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")

    result = anime_stylize(
        img, args.prompt, args.negative, args.strength, args.cfg_scale,
        args.steps, output_dir, args.tensor_model, args.seed,
    )
    if result is None:
        log(output_dir, "Stylization returned no image", "ERROR")
        return None

    _pipeline_accrue(TENSOR_COST)

    final_path = os.path.join(output_dir, f"{tag}__final.jpg")
    result.save(final_path, quality=95)
    log(output_dir, f"Saved: {final_path}")

    # Copy a self-reference of the script for reproducibility
    try:
        shutil.copyfile(os.path.abspath(__file__), os.path.join(output_dir, os.path.basename(__file__)))
    except OSError:
        pass

    # Always copy final to shared/finals/ (pinned) and shared/anime/ (this tool's gallery)
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    anime_dir = os.path.expanduser("~/.openclaw/workspace/shared/anime")
    if getattr(args, "anime_subdir", None):
        anime_dir = os.path.join(anime_dir, args.anime_subdir)
    os.makedirs(finals_dir, exist_ok=True)
    os.makedirs(anime_dir, exist_ok=True)
    finals_dest = os.path.join(finals_dir, f"{tag}.jpg")
    shutil.copyfile(final_path, finals_dest)
    shutil.copyfile(final_path, os.path.join(anime_dir, f"{tag}.jpg"))
    log(output_dir, f"Final copied to: {finals_dest} (+ shared/anime/)")

    # Push to phone
    try:
        from notify import push_image
        push_image(finals_dest, title=f"Anime — {src_name}",
                   body=f"strength={args.strength}")
    except Exception as e:
        log(output_dir, f"Push failed (non-fatal): {e}", "WARN")

    return finals_dest


def main():
    parser = argparse.ArgumentParser(
        description="Cinematic semi-realistic anime version of a photo (uncensored Tensor Art img2img).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", nargs="+", required=True, help="Input photo path(s)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Override the anime prompt")
    parser.add_argument("--prompt-extra", default=None, help="Text appended to the prompt (e.g. clothing for an SFW variant)")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE, help="Override the negative prompt")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Denoising strength (default 0.5: preserves pose/composition while reading as anime). "
                             "Lower=more faithful to photo, higher=more fully anime but drifts pose")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale (default 5.0)")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps (default 30)")
    parser.add_argument("--tensor-model", default=MODEL_DEFAULT,
                        help=f"Tensor Art model ID (default uncensored Z-Image: {MODEL_DEFAULT})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--local-output-dir", default=None, help="Base output dir (default ~/.openclaw/workspace/shared)")
    parser.add_argument("--anime-subdir", default=None, help="Subfolder under shared/anime/ to collect these results (e.g. 'inbal alafi')")
    args = parser.parse_args()

    if args.prompt_extra:
        args.prompt = f"{args.prompt} {args.prompt_extra}"

    finals = []
    for src in args.source:
        print(f"\n=== {src} ===")
        dest = process_one(src, args)
        if dest:
            finals.append(dest)

    print(f"\nDone. {len(finals)}/{len(args.source)} succeeded.")
    for f in finals:
        print(f"  {f}")
    if PIPELINE_COST["cost_today"]:
        print(f"Approx cost this run: ${PIPELINE_COST['cost_today']:.2f}")


if __name__ == "__main__":
    main()
