#!/home/rong/openclaw-venv/bin/python3
"""
Parallax 3D — animate a still as a 3D-parallax MP4.

Default DIY pipeline:
  1. depth via fal-ai/imageutils/depth-anything-v2 (~$0.005)
  2. quantize depth into FG/MID/BG tiers
  3. translate each tier per frame, composite, write MP4 with cv2

Optional --inpaint flag uses Replicate pollinations/3d-photo-inpainting,
falls through to DIY if unavailable.

Motion presets: dolly-left, dolly-right, zoom-in, dutch.

Usage:
    python parallax_3d.py --source photo.jpg --motion dolly-left
    python parallax_3d.py --source photo.jpg --motion zoom-in --strength 0.7
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

os.environ.setdefault("FAL_KEY", os.environ.get("FAL_API_KEY", ""))

import re
import math
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

import numpy as np
import requests
from PIL import Image
import fal_client
import cv2

sys.stdout.reconfigure(line_buffering=True)

MOTIONS = ["dolly-left", "dolly-right", "zoom-in", "dutch"]

_log_lock = threading.Lock()


def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    with _log_lock:
        try:
            with open(os.path.join(output_dir, "workflow.log"), "a") as f:
                f.write(formatted + "\n")
        except OSError:
            pass


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


def get_depth(image_path, output_dir):
    """Return a single-channel depth array (uint8, near=255 / far=0)."""
    log(output_dir, "Requesting depth: fal-ai/imageutils/depth")
    image_url = fal_client.upload_file(image_path)
    handle = fal_client.submit("fal-ai/imageutils/depth",
                               arguments={"image_url": image_url})
    res = handle.get()
    img = res.get("image") or {}
    url = img.get("url") if isinstance(img, dict) else None
    if not url:
        log(output_dir, f"Bad depth response: {json.dumps(res)[:300]}", "ERROR")
        return None
    log(output_dir, f"Depth CDN: {url}")
    r = requests.get(url, timeout=120); r.raise_for_status()
    depth_img = Image.open(BytesIO(r.content)).convert("L")
    return np.array(depth_img)


def try_inpaint_replicate(image_path, motion, duration, fps, output_dir):
    """Best-effort Replicate 3d-photo-inpainting; return mp4 path or None."""
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        log(output_dir, "REPLICATE_API_TOKEN missing — skipping inpaint path", "WARN")
        return None
    try:
        import replicate
    except ImportError:
        log(output_dir, "replicate SDK not installed — skipping inpaint", "WARN")
        return None
    try:
        with open(image_path, "rb") as f:
            output = replicate.run(
                "pollinations/3d-photo-inpainting",
                input={"image": f},
            )
        # Output expected to be a video URL or list
        url = None
        if isinstance(output, str):
            url = output
        elif isinstance(output, (list, tuple)) and output:
            url = output[0]
        if not url:
            return None
        dest = os.path.join(output_dir, "inpaint_video.mp4")
        r = requests.get(url, stream=True, timeout=600); r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        log(output_dir, f"Inpaint MP4: {dest}")
        _pipeline_accrue(0.05)  # rough estimate
        return dest
    except Exception as e:
        log(output_dir, f"Inpaint path failed: {e} — falling back to DIY", "WARN")
        return None


def render_diy(image_bgr, depth, motion, strength, duration, fps, output_dir, mp4_path):
    """Tier-based parallax render. depth: HxW uint8 (near=255)."""
    h, w = image_bgr.shape[:2]
    n_frames = int(duration * fps)

    # 3 tiers via depth quantiles
    d = depth.astype(np.float32)
    q33, q66 = np.percentile(d, [33, 66])
    fg_mask = (d >= q66).astype(np.uint8) * 255
    mid_mask = ((d >= q33) & (d < q66)).astype(np.uint8) * 255
    bg_mask = (d < q33).astype(np.uint8) * 255

    # Feather masks
    k = max(3, int(min(h, w) * 0.005) | 1)
    fg_mask = cv2.GaussianBlur(fg_mask, (k, k), 0)
    mid_mask = cv2.GaussianBlur(mid_mask, (k, k), 0)
    bg_mask = cv2.GaussianBlur(bg_mask, (k, k), 0)

    # Per-tier offset multipliers (FG moves most opposite the camera)
    # Max pixel offset scales with strength + image size
    base_offset = int(min(h, w) * 0.04 * strength)  # ~4% of short edge at strength=1
    fg_mult, mid_mult, bg_mult = 1.0, 0.45, 0.1

    # Inpaint occluded edges via cv2.inpaint on background tier
    bg_only = cv2.bitwise_and(image_bgr, image_bgr, mask=(bg_mask > 64).astype(np.uint8) * 255)
    fg_hole_mask = ((fg_mask > 64) | (mid_mask > 64)).astype(np.uint8) * 255
    bg_filled = cv2.inpaint(image_bgr, fg_hole_mask, 5, cv2.INPAINT_TELEA)
    mid_only = image_bgr
    fg_only = image_bgr

    # Pre-convert masks to 3-channel float
    fg_a = (fg_mask.astype(np.float32) / 255.0)[:, :, None]
    mid_a = (mid_mask.astype(np.float32) / 255.0)[:, :, None]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        log(output_dir, "VideoWriter failed to open", "ERROR")
        return False

    log(output_dir, f"Rendering {n_frames} frames at {fps}fps, base_offset={base_offset}px, motion={motion}")

    for fi in range(n_frames):
        t = fi / max(1, n_frames - 1)  # 0..1
        # Smooth ease in-out
        e = 0.5 - 0.5 * math.cos(t * math.pi)

        if motion == "dolly-left":
            dx, dy, scale = -e, 0.0, 1.0
        elif motion == "dolly-right":
            dx, dy, scale = e, 0.0, 1.0
        elif motion == "zoom-in":
            dx, dy = 0.0, 0.0
            scale = 1.0 + 0.06 * e * strength
        elif motion == "dutch":
            angle = 1.5 * e * strength  # subtle rotate
            dx, dy, scale = 0.0, 0.0, None
        else:
            dx, dy, scale = -e, 0.0, 1.0

        def shift_layer(img, mult):
            if motion == "dutch":
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle * mult, 1.0)
                return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            ox = int(dx * base_offset * mult)
            oy = int(dy * base_offset * mult)
            if motion == "zoom-in":
                s = 1.0 + (scale - 1.0) * mult
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
                return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        bg_layer = shift_layer(bg_filled, bg_mult)
        mid_layer = shift_layer(mid_only, mid_mult)
        fg_layer = shift_layer(fg_only, fg_mult)

        # Shift masks too
        if motion == "dutch":
            M_mid = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5 * e * strength * mid_mult, 1.0)
            M_fg = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5 * e * strength * fg_mult, 1.0)
            mid_a_s = cv2.warpAffine(mid_a, M_mid, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            fg_a_s = cv2.warpAffine(fg_a, M_fg, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif motion == "zoom-in":
            s_mid = 1.0 + (scale - 1.0) * mid_mult
            s_fg = 1.0 + (scale - 1.0) * fg_mult
            M_mid = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s_mid)
            M_fg = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s_fg)
            mid_a_s = cv2.warpAffine(mid_a, M_mid, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            fg_a_s = cv2.warpAffine(fg_a, M_fg, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            M_mid = np.float32([[1, 0, int(dx * base_offset * mid_mult)], [0, 1, int(dy * base_offset * mid_mult)]])
            M_fg = np.float32([[1, 0, int(dx * base_offset * fg_mult)], [0, 1, int(dy * base_offset * fg_mult)]])
            mid_a_s = cv2.warpAffine(mid_a, M_mid, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            fg_a_s = cv2.warpAffine(fg_a, M_fg, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if mid_a_s.ndim == 2:
            mid_a_s = mid_a_s[:, :, None]
        if fg_a_s.ndim == 2:
            fg_a_s = fg_a_s[:, :, None]

        comp = bg_layer.astype(np.float32) * (1 - mid_a_s) + mid_layer.astype(np.float32) * mid_a_s
        comp = comp * (1 - fg_a_s) + fg_layer.astype(np.float32) * fg_a_s
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        writer.write(comp)

    writer.release()
    log(output_dir, f"Wrote {mp4_path} ({os.path.getsize(mp4_path)/1024:.1f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="3D parallax video from a still")
    parser.add_argument("--source", required=True)
    parser.add_argument("--motion", choices=MOTIONS, default="dolly-left")
    parser.add_argument("--strength", type=float, default=0.6, help="0.3-1.0, scales pixel offsets")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--inpaint", action="store_true", help="Try Replicate 3d-photo-inpainting")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None)
    args = parser.parse_args()

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: source not found: {source}")
        sys.exit(1)

    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    suffix = random.randint(10, 99)
    folder_name = f"{model_name}_{source_basename}_{timestamp}_parallax_{args.motion}_{suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    log(output_dir, "=" * 60)
    log(output_dir, "PARALLAX 3D START")
    log(output_dir, f"Source:    {source}")
    log(output_dir, f"Motion:    {args.motion} strength={args.strength}")
    log(output_dir, f"Duration:  {args.duration}s @ {args.fps}fps")
    log(output_dir, f"Inpaint:   {args.inpaint}")
    log(output_dir, f"Output:    {output_dir}")
    log(output_dir, "=" * 60)

    t0 = time.time()
    mp4_path = os.path.join(output_dir, "video.mp4")
    used_inpaint = False

    if args.inpaint:
        ip = try_inpaint_replicate(source, args.motion, args.duration, args.fps, output_dir)
        if ip:
            shutil.copy2(ip, mp4_path)
            used_inpaint = True

    if not used_inpaint:
        depth = get_depth(source, output_dir)
        if depth is None:
            log(output_dir, "Depth failed", "ERROR")
            sys.exit(1)
        _pipeline_accrue(0.005)
        # Save depth for debugging
        Image.fromarray(depth).save(os.path.join(output_dir, "depth.png"))

        img = cv2.imread(source, cv2.IMREAD_COLOR)
        if img is None:
            log(output_dir, "cv2.imread failed", "ERROR")
            sys.exit(1)
        # Match depth size to image
        if depth.shape[:2] != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        ok = render_diy(img, depth, args.motion, args.strength,
                        args.duration, args.fps, output_dir, mp4_path)
        if not ok:
            sys.exit(1)

    if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
        log(output_dir, "MP4 missing/empty", "ERROR")
        sys.exit(1)

    # First frame
    frame_png = os.path.join(output_dir, "first_frame.png")
    try:
        cap = cv2.VideoCapture(mp4_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_png)
    except Exception as e:
        log(output_dir, f"first frame failed: {e}", "WARN")

    # Finals
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_mp4 = os.path.join(finals_dir, folder_name + ".mp4")
    finals_png = os.path.join(finals_dir, folder_name + ".png")
    shutil.copyfile(mp4_path, finals_mp4)
    if os.path.isfile(frame_png):
        shutil.copyfile(frame_png, finals_png)
    log(output_dir, f"Finals: {finals_mp4}")

    try:
        from notify import push_image
        if os.path.isfile(finals_png):
            push_image(finals_png, title=f"Parallax — {source_basename}",
                       body=f"{args.motion} s={args.strength}")
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
  PARALLAX 3D SUMMARY
============================================================
  Source:    {source}
  Motion:    {args.motion}  strength={args.strength}
  Duration:  {args.duration}s @ {args.fps}fps
  Time:      {elapsed:.1f}s
  MP4:       {finals_mp4}
  Frame:     {finals_png}
============================================================""")


if __name__ == "__main__":
    main()
