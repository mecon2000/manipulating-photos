#!/home/rong/openclaw-venv/bin/python3
"""
Stills to Video — animate a single still using fal image-to-video models.

Default engine: fal-ai/wan/v2.2-a14b/image-to-video/turbo (~$0.30/5s clip).
Alt engine: fal-ai/kling-video/v2/master/image-to-video (~$0.50).

Usage:
    python stills_to_video.py --source photo.jpg --preset subtle-breath
    python stills_to_video.py --source photo.jpg --preset hair-wind --engine kling
    python stills_to_video.py --list-presets
"""

import os
import sys

_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# fal_client SDK reads FAL_KEY at import; env has FAL_API_KEY
os.environ.setdefault("FAL_KEY", os.environ.get("FAL_API_KEY", ""))

import re
import json
import time
import random
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

import requests
from PIL import Image
import fal_client

sys.stdout.reconfigure(line_buffering=True)

PRESETS = {
    "subtle-breath": "the subject breathes gently, very subtle chest and shoulder movement, eyes blink occasionally, photorealistic, no camera movement",
    "hair-wind": "the subject's hair moves softly in a light breeze, gentle strands flowing, subtle clothing motion, photorealistic, locked camera",
    "smoke-drift": "soft smoke or mist drifts slowly across the scene, atmospheric particles floating, subject mostly still, photorealistic",
    "water-ripple": "gentle water ripples and reflections across the scene, subtle liquid motion, subject still, cinemagraph style",
    "eye-blink": "the subject blinks slowly and naturally a few times, very subtle micro-expressions, otherwise still, photorealistic portrait",
    "full-cinemagraph": "cinemagraph: subject is still and frozen, but one element in the scene moves continuously — hair, smoke, fabric, or water — photorealistic loop",
}

ENGINES = {
    "wan": {
        "model_id": "fal-ai/wan/v2.2-a14b/image-to-video/turbo",
        "cost": 0.30,
    },
    "kling": {
        "model_id": "fal-ai/kling-video/v2/master/image-to-video",
        "cost": 0.50,
    },
}

_log_lock = threading.Lock()


def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    with _log_lock:
        log_path = os.path.join(output_dir, "workflow.log")
        try:
            with open(log_path, "a") as f:
                f.write(formatted + "\n")
        except OSError:
            pass


def run_video(image_path, prompt, engine, output_dir, seed=None):
    spec = ENGINES[engine]
    log(output_dir, f"Submitting {spec['model_id']} prompt='{prompt[:80]}'")
    image_url = fal_client.upload_file(image_path)
    log(output_dir, f"Uploaded source: {image_url}")
    args = {"prompt": prompt, "image_url": image_url}
    if seed is not None:
        args["seed"] = seed
    handle = fal_client.submit(spec["model_id"], arguments=args)
    res = handle.get()
    video = res.get("video") or {}
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        # some engines return at top-level
        url = res.get("url")
    if not url:
        log(output_dir, f"No video URL in response: {json.dumps(res)[:300]}", "ERROR")
        return None
    log(output_dir, f"Video CDN URL: {url}")
    return url


def download(url, dest_path, output_dir):
    log(output_dir, f"Downloading -> {dest_path}")
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    sz = os.path.getsize(dest_path)
    log(output_dir, f"Saved {sz/1024:.1f} KB")
    return dest_path


def first_frame(mp4_path, png_path, output_dir):
    try:
        import cv2
        cap = cv2.VideoCapture(mp4_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            log(output_dir, "Could not read first frame", "WARN")
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(png_path, "PNG")
        log(output_dir, f"First frame: {png_path}")
        return png_path
    except Exception as e:
        log(output_dir, f"First frame extract failed: {e}", "WARN")
        return None


# Cost tracking helper (copied from batch-runner.py pattern)
PIPELINE_COST = {"date": "", "cost_today": 0.0}
PIPELINE_COST_LOCK = threading.RLock()


def _pipeline_accrue(amount):
    today = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d")
    with PIPELINE_COST_LOCK:
        if PIPELINE_COST["date"] != today:
            PIPELINE_COST["date"] = today
            PIPELINE_COST["cost_today"] = 0.0
        PIPELINE_COST["cost_today"] = round(
            PIPELINE_COST["cost_today"] + amount, 4)


def main():
    parser = argparse.ArgumentParser(description="Animate a still via fal image-to-video")
    parser.add_argument("--source", required=False, help="Input photo path")
    parser.add_argument("--preset", default="subtle-breath", help="Motion preset")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides preset)")
    parser.add_argument("--engine", choices=list(ENGINES.keys()), default="wan", help="Video model engine")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None)
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{'Preset':<22} Prompt")
        print("=" * 90)
        for name, p in PRESETS.items():
            print(f"  {name:<20} {p[:65]}...")
        sys.exit(0)

    if not args.source:
        print("ERROR: --source required")
        sys.exit(1)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    if args.prompt:
        prompt = args.prompt
        preset_name = "custom"
    else:
        if args.preset not in PRESETS:
            print(f"ERROR: Unknown preset '{args.preset}'. Use --list-presets.")
            sys.exit(1)
        prompt = PRESETS[args.preset]
        preset_name = args.preset

    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    tag = preset_name.replace(" ", "_")[:25]
    suffix = random.randint(10, 99)
    folder_name = f"{model_name}_{source_basename}_{timestamp}_video_{args.engine}_{tag}_{suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    log(output_dir, "=" * 60)
    log(output_dir, "STILLS-TO-VIDEO START")
    log(output_dir, f"Source:   {source}")
    log(output_dir, f"Engine:   {args.engine} ({ENGINES[args.engine]['model_id']})")
    log(output_dir, f"Preset:   {preset_name}")
    log(output_dir, f"Prompt:   {prompt[:100]}")
    log(output_dir, f"Seed:     {seed}")
    log(output_dir, f"Output:   {output_dir}")
    log(output_dir, "=" * 60)

    t0 = time.time()
    url = run_video(source, prompt, args.engine, output_dir, seed=seed)
    if not url:
        log(output_dir, "Video generation failed", "ERROR")
        sys.exit(1)

    mp4_path = os.path.join(output_dir, "video.mp4")
    download(url, mp4_path, output_dir)
    _pipeline_accrue(ENGINES[args.engine]["cost"])
    log(output_dir, f"Cost accrued: ${ENGINES[args.engine]['cost']:.3f}")

    frame_png = os.path.join(output_dir, "first_frame.png")
    first_frame(mp4_path, frame_png, output_dir)

    # Copy to finals
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_mp4 = os.path.join(finals_dir, folder_name + ".mp4")
    finals_png = os.path.join(finals_dir, folder_name + ".png")
    shutil.copyfile(mp4_path, finals_mp4)
    if os.path.isfile(frame_png):
        shutil.copyfile(frame_png, finals_png)
    log(output_dir, f"Finals: {finals_mp4}")

    # Push first frame
    try:
        from notify import push_image
        if os.path.isfile(finals_png):
            push_image(finals_png, title=f"Video — {source_basename}",
                       body=f"{args.engine} / {preset_name}")
            log(output_dir, "Pushed first frame to phone")
    except Exception as e:
        log(output_dir, f"Push failed: {e}", "WARN")

    try:
        shutil.copy2(os.path.abspath(__file__),
                     os.path.join(output_dir, f"workflow_script_{os.path.basename(__file__)}"))
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"""
============================================================
  STILLS-TO-VIDEO SUMMARY
============================================================
  Source:        {source}
  Engine:        {args.engine}
  Preset:        {preset_name}
  Seed:          {seed}
  Time:          {elapsed:.1f}s
  Cost:          ${ENGINES[args.engine]['cost']:.3f}
  MP4:           {finals_mp4}
  First frame:   {finals_png}
============================================================""")


if __name__ == "__main__":
    main()
