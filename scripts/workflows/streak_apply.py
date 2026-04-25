#!/usr/bin/env python3
"""Apply cv2 directional motion blur to one limb of an existing image.

Skips BW/mask/face/smear — assumes the input is already a finished photo.
Use to iterate fast on streak parameters.

Usage:
  streak_apply.py --source path/to/image.jpg \\
      --limb right-arm --length-pct 40 --decay 4.0 [--flip]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

OUT = os.path.expanduser("~/.openclaw/workspace/shared/motion-streak-finals")

POSE_MODEL = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")
LIMB_LANDMARKS = {
    "left-arm":  (11, 13, 15),
    "right-arm": (12, 14, 16),
    "left-leg":  (23, 25, 27),
    "right-leg": (24, 26, 28),
}


def detect_limb_points(img_arr, limb):
    import mediapipe as mp
    idxs = LIMB_LANDMARKS[limb]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr.astype(np.uint8))
    base = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
    opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
    det = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    res = det.detect(mp_img); det.close()
    if not res.pose_landmarks: return None
    lms = res.pose_landmarks[0]
    h, w = img_arr.shape[:2]
    return [(lms[i].x * w, lms[i].y * h) for i in idxs]


def limb_mask_and_tangent(img_size, pts, radius_px):
    w, h = img_size
    root, mid, tip = [np.array(p, dtype=np.float32) for p in pts]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    def seg(a, b):
        ab = b - a
        L2 = float(ab[0]**2 + ab[1]**2) or 1.0
        t = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2
        t = np.clip(t, 0, 1)
        d = np.sqrt((xx - (a[0] + t*ab[0]))**2 + (yy - (a[1] + t*ab[1]))**2)
        return d, t
    d1, t1 = seg(root, mid); d2, t2 = seg(mid, tip)
    use2 = d2 < d1
    d = np.where(use2, d2, d1)
    param = np.where(use2, 0.5 + 0.5*t2, 0.5*t1)
    mask = np.clip(1.0 - d / radius_px, 0, 1).astype(np.float32)
    strength = (mask * param).astype(np.float32)
    tangent = float(np.degrees(np.arctan2(-(root[1]-tip[1]), root[0]-tip[0])))
    return mask, strength, tangent


def streak(img_arr, strength, tangent_deg, length_px, decay=4.0, gain=2.5):
    L = max(3, int(length_px))
    K = L*2 + 1
    kernel = np.zeros((K, K), dtype=np.float32)
    cx = cy = K // 2
    rad = np.deg2rad(tangent_deg + 180)   # cv2 correlation flip
    for t in range(L):
        x = int(round(cx + np.cos(rad)*t))
        y = int(round(cy - np.sin(rad)*t))
        if 0 <= x < K and 0 <= y < K:
            kernel[y, x] = np.exp(-decay * t / L)
    source = (img_arr.astype(np.float32) * gain) * strength[..., None]
    blurred = cv2.filter2D(source, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    out = np.maximum(img_arr.astype(np.float32), blurred)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--limb", default="right-arm", choices=list(LIMB_LANDMARKS))
    p.add_argument("--length-pct", type=float, default=40.0,
                   help="trail length as %% of short edge")
    p.add_argument("--decay", type=float, default=4.0)
    p.add_argument("--gain", type=float, default=2.5)
    p.add_argument("--limb-radius-pct", type=float, default=10.0)
    p.add_argument("--flip", action="store_true",
                   help="flip trail direction 180°")
    p.add_argument("--suffix", default=None)
    args = p.parse_args()

    img = Image.open(args.source).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    short = min(w, h)
    length_px = short * args.length_pct / 100.0
    radius_px = short * args.limb_radius_pct / 100.0

    pts = detect_limb_points(arr, args.limb)
    if pts is None:
        print("[err] pose not detected"); sys.exit(2)
    mask, strength, tangent = limb_mask_and_tangent((w, h), pts, radius_px)
    if args.flip: tangent += 180
    print(f"limb={args.limb} tangent={tangent:.0f}° L={length_px:.0f}px decay={args.decay} gain={args.gain}")

    out = streak(arr, strength, tangent, length_px, decay=args.decay, gain=args.gain)
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.source))[0]
    suf = args.suffix or f"apply_{args.limb}_L{int(args.length_pct)}_d{args.decay:.0f}_g{args.gain:.1f}{'_flip' if args.flip else ''}"
    out_path = os.path.join(OUT, f"{name}_{suf}.jpg")
    Image.fromarray(out).save(out_path, quality=92)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
