#!/home/rong/openclaw-venv/bin/python3
"""
Depth-framing physics demo.

Synthetic scene:
  - Canvas: 1920x1080, camera 85mm f/2.8 focused at 2m on full-frame sensor
  - Plane 3m: cloud (BG, slight blur)
  - Plane 2m: subject circle (head+shoulder placeholder, in focus)
  - Plane 1m: cloud (FG, moderate blur)
  - Plane 0.5m: cloud (near-FG, heavy blur)

No subject extraction, no EXIF, no face protection — pure DoF physics + scale.
Generates 3 distinct Flux cloud images (parallel), mattes to alpha via luminance,
scales by 1/d, Gaussian blurs by computed CoC, composites back-to-front.

Usage:
    ./depth-framing-demo.py [--seed N] [--focal 85] [--fnum 2.8] [--focus 2.0]
"""

import os
import sys

_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                k, v = _line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
os.environ['FAL_KEY'] = os.environ.get('FAL_API_KEY', '')

import argparse
import random
import threading
import time
from io import BytesIO
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageDraw
import fal_client
import requests

sys.stdout.reconfigure(line_buffering=True)


# --- Camera params ---
CANVAS_W, CANVAS_H = 1920, 1080
SENSOR_W_MM = 36.0  # full-frame
PX_PER_MM = CANVAS_W / SENSOR_W_MM


def coc_pixels(dist_m, focus_m, focal_mm, fnum):
    """Circle-of-confusion diameter in pixels on the image.

    C_mm = (f² / (N * (s - f))) * |s - a| / a    (distances in mm)
    """
    if abs(dist_m - focus_m) < 1e-6:
        return 0.0
    f = focal_mm
    N = fnum
    s_mm = focus_m * 1000.0
    a_mm = dist_m * 1000.0
    c_mm = (f * f) / (N * (s_mm - f)) * abs(s_mm - a_mm) / a_mm
    return c_mm * PX_PER_MM


def generate_cloud(seed, prompt, result_dict, key):
    """Generate one cloud image via Flux schnell on black background."""
    try:
        handle = fal_client.submit("fal-ai/flux/schnell", arguments={
            "prompt": prompt,
            "image_size": {"width": 1024, "height": 1024},
            "num_inference_steps": 4,
            "num_images": 1,
            "output_format": "jpeg",
            "enable_safety_checker": False,
            "seed": seed,
        })
        result = handle.get()
        url = result["images"][0]["url"]
        img = Image.open(BytesIO(requests.get(url, timeout=60).content)).convert("RGB")
        result_dict[key] = img
    except Exception as e:
        result_dict[key] = None
        result_dict[key + "_err"] = str(e)


def matte_to_alpha(img_rgb):
    """Use luminance as alpha: bright = opaque cloud, dark = transparent."""
    gray = img_rgb.convert("L")
    a = np.array(gray).astype(np.float32)
    # Lift blacks to make darker cloud parts still visible
    # pure black (bg) stays transparent, mid-grey partial, bright fully opaque
    a = np.clip((a - 10) / 245.0 * 255.0, 0, 255).astype(np.uint8)
    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(a))
    return rgba


def place_plane(cloud_rgba, dist_m, canvas_w, canvas_h, jitter_seed=0):
    """Scale cloud by 1/d (treating native 1024 image as 1m reference filling frame),
    Gaussian blur by CoC, place on a transparent canvas of target size."""
    # Reference: at 1m distance, cloud image spans roughly the canvas width
    scale = canvas_w / 1024.0 / dist_m  # smaller when farther
    new_w = int(1024 * scale)
    new_h = int(1024 * scale)
    if new_w < 8 or new_h < 8:
        new_w = new_h = 8
    scaled = cloud_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Apply Gaussian blur matching CoC
    blur_r = coc_pixels(dist_m, FOCUS_M, FOCAL, FNUM)
    if blur_r > 0.5:
        # Gaussian sigma ~ blur_r / 3 (roughly matches disk blur visually)
        sigma = max(1.0, blur_r / 3.0)
        arr = np.array(scaled).astype(np.float32)
        # Blur RGB and alpha separately
        for c in range(4):
            arr[:, :, c] = cv2.GaussianBlur(arr[:, :, c], (0, 0), sigma)
        scaled = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Place centered with small random offset
    rng = random.Random(jitter_seed)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    cx = canvas_w // 2 + rng.randint(-canvas_w // 8, canvas_w // 8)
    cy = canvas_h // 2 + rng.randint(-canvas_h // 8, canvas_h // 8)
    canvas.paste(scaled, (cx - new_w // 2, cy - new_h // 2), scaled)
    return canvas, blur_r


def draw_subject_circle(canvas_w, canvas_h):
    """Head+shoulders placeholder at plane 2m (in focus)."""
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Head: 200x240 ellipse, shoulders below
    head_w, head_h = 200, 240
    shoulder_w, shoulder_h = 500, 280
    cx = canvas_w // 2
    cy = canvas_h // 2
    # Head
    draw.ellipse([cx - head_w // 2, cy - head_h // 2 - 100,
                  cx + head_w // 2, cy + head_h // 2 - 100],
                 fill=(220, 180, 160, 255))
    # Shoulders (wider rounded rect)
    draw.ellipse([cx - shoulder_w // 2, cy + 30,
                  cx + shoulder_w // 2, cy + 30 + shoulder_h],
                 fill=(60, 60, 80, 255))
    return img


def main():
    global FOCUS_M, FOCAL, FNUM
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=random.randint(1, 10**9))
    ap.add_argument("--focal", type=float, default=85.0)
    ap.add_argument("--fnum", type=float, default=2.8)
    ap.add_argument("--focus", type=float, default=2.0)
    args = ap.parse_args()

    FOCAL = args.focal
    FNUM = args.fnum
    FOCUS_M = args.focus
    seed = args.seed

    print(f"[cfg] {FOCAL}mm f/{FNUM} focus @ {FOCUS_M}m, seed={seed}")
    print(f"[cfg] Canvas {CANVAS_W}x{CANVAS_H}, px/mm_sensor={PX_PER_MM:.1f}")

    planes = [3.0, 1.0, 0.5]
    for d in planes:
        print(f"[coc] {d}m  blur = {coc_pixels(d, FOCUS_M, FOCAL, FNUM):.1f}px")

    # Generate 3 cloud images in parallel
    prompt = "thick photorealistic cumulus clouds, dense puffy white and grey clouds, soft volumetric lighting, strong shape definition, isolated floating on pure black background, cinematic"
    results = {}
    threads = []
    for i, d in enumerate(planes):
        t = threading.Thread(target=generate_cloud,
                             args=(seed + i * 100, prompt, results, f"cloud_{i}"))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Verify all generated
    for i in range(len(planes)):
        if results.get(f"cloud_{i}") is None:
            print(f"[fail] cloud {i} generation failed: {results.get(f'cloud_{i}_err')}")
            sys.exit(1)

    # Build scene
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (30, 30, 40))  # dark navy BG

    # Back-to-front composite
    # 1. 3m cloud (back)
    c3, b3 = place_plane(matte_to_alpha(results["cloud_0"]), 3.0, CANVAS_W, CANVAS_H, seed)
    canvas.paste(c3, (0, 0), c3)
    # 2. 2m subject (sharp)
    subj = draw_subject_circle(CANVAS_W, CANVAS_H)
    canvas.paste(subj, (0, 0), subj)
    # 3. 1m cloud (FG 1)
    c1, b1 = place_plane(matte_to_alpha(results["cloud_1"]), 1.0, CANVAS_W, CANVAS_H, seed + 50)
    canvas.paste(c1, (0, 0), c1)
    # 4. 0.5m cloud (FG 2, heaviest blur)
    c05, b05 = place_plane(matte_to_alpha(results["cloud_2"]), 0.5, CANVAS_W, CANVAS_H, seed + 100)
    canvas.paste(c05, (0, 0), c05)

    # Label
    draw = ImageDraw.Draw(canvas)
    label = f"{FOCAL}mm f/{FNUM} focus@{FOCUS_M}m | planes: 3m(blur {b3:.0f}px) 2m(sharp) 1m({b1:.0f}px) 0.5m({b05:.0f}px)"
    draw.text((20, 20), label, fill=(255, 255, 180))

    # Save
    out_dir = os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates/depth_framing_demo")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"demo_seed{seed}_{ts}.jpg")
    canvas.save(out_path, "JPEG", quality=95)
    # Also save individual layers for inspection
    results["cloud_0"].save(os.path.join(out_dir, f"native_3m_{ts}.jpg"), quality=90)
    results["cloud_1"].save(os.path.join(out_dir, f"native_1m_{ts}.jpg"), quality=90)
    results["cloud_2"].save(os.path.join(out_dir, f"native_0.5m_{ts}.jpg"), quality=90)

    # Copy to finals for phone viewing
    finals = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    finals_out = os.path.join(finals, f"depth_framing_demo_{ts}.jpg")
    canvas.save(finals_out, "JPEG", quality=95)
    print(f"[out] {finals_out}")
    try:
        from notify import push_image
        push_image(finals_out, title="Depth framing demo",
                   body=f"{FOCAL}mm f/{FNUM} focus@{FOCUS_M}m")
    except Exception:
        pass


if __name__ == "__main__":
    main()
