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
    # Anti-aliased kernel: bilinear-splat each weight to 4 surrounding pixels.
    # Off-axis angles (e.g. 190°) used to stair-step and lose energy; this
    # gives sub-pixel coverage so every angle has the same total weight.
    for t in range(L):
        fx = cx + np.cos(rad) * t
        fy = cy - np.sin(rad) * t
        if not (0 <= fx < K - 1 and 0 <= fy < K - 1):
            continue
        x0, y0 = int(fx), int(fy)
        dx, dy = fx - x0, fy - y0
        w = np.exp(-decay * t / L)
        kernel[y0,     x0    ] += w * (1 - dx) * (1 - dy)
        kernel[y0,     x0 + 1] += w * dx       * (1 - dy)
        kernel[y0 + 1, x0    ] += w * (1 - dx) * dy
        kernel[y0 + 1, x0 + 1] += w * dx       * dy
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
    p.add_argument("--angle", type=float, default=None,
                   help="override detected tangent — 0=right, 90=up, 180=left, 270=down")
    p.add_argument("--black-marks", type=int, default=0,
                   help="add N small black bars perpendicular to the limb before blurring")
    p.add_argument("--mark-width-pct", type=float, default=1.5,
                   help="black mark length (perpendicular to arm), %% of short edge")
    p.add_argument("--dot-clusters", type=int, default=0,
                   help="N clusters of 4-5 white pixels along arm; blurred (same params),"
                        " inverted, and subtracted → dark streaks alongside the bright ones")
    p.add_argument("--dot-cluster-radius-pct", type=float, default=0.5,
                   help="radius (%% of short edge) within which the cluster pixels scatter")
    p.add_argument("--dot-strength", type=float, default=1.0,
                   help="how strongly the inverted dark streaks darken the result (0-1.5)")
    p.add_argument("--dot-radius-px", type=int, default=3,
                   help="radius (px) of each anti-aliased ball planted in the cluster")
    p.add_argument("--suffix", default=None)
    args = p.parse_args()

    img = Image.open(args.source).convert("RGB")
    arr = np.array(img)   # writable copy
    h, w = arr.shape[:2]
    short = min(w, h)
    length_px = short * args.length_pct / 100.0
    radius_px = short * args.limb_radius_pct / 100.0

    pts = detect_limb_points(arr, args.limb)
    if pts is None:
        print("[err] pose not detected"); sys.exit(2)
    mask, strength, tangent = limb_mask_and_tangent((w, h), pts, radius_px)
    if args.angle is not None: tangent = args.angle
    if args.flip: tangent += 180
    print(f"limb={args.limb} tangent={tangent:.0f}° L={length_px:.0f}px decay={args.decay} gain={args.gain}")

    # Optional black marks across the arm (perpendicular bars)
    bar_endpoints = []
    bar_thick = 0
    if args.black_marks > 0:
        rng = np.random.default_rng(42)
        root, mid, tip = [np.array(p, dtype=np.float32) for p in pts]
        ts = rng.uniform(0.1, 0.95, args.black_marks)
        bar_len = int(short * args.mark_width_pct / 100.0)
        bar_thick = max(2, int(short * 0.003))
        for t in ts:
            if t < 0.5:
                p = root + (mid - root) * (t / 0.5)
            else:
                p = mid + (tip - mid) * ((t - 0.5) / 0.5)
            seg = (mid - root) if t < 0.5 else (tip - mid)
            n = np.array([-seg[1], seg[0]])
            n = n / (np.linalg.norm(n) or 1)
            a = (int(p[0] - n[0]*bar_len/2), int(p[1] - n[1]*bar_len/2))
            b = (int(p[0] + n[0]*bar_len/2), int(p[1] + n[1]*bar_len/2))
            bar_endpoints.append((a, b))
            cv2.line(arr, a, b, (0, 0, 0), bar_thick)
        print(f"  drew {args.black_marks} black bars (len={bar_len}px, thick={bar_thick}px)")

    out = streak(arr, strength, tangent, length_px, decay=args.decay, gain=args.gain)

    # Re-draw the bars on top of the streaked result so they show unblurred
    for a, b in bar_endpoints:
        cv2.line(out, a, b, (0, 0, 0), bar_thick)

    # Dual-blur dark streaks: scatter white pixels along the arm on a black
    # canvas, blur them with the same kernel, invert → dark trails, subtract.
    if args.dot_clusters > 0:
        rng3 = np.random.default_rng(43)
        root, mid, tip = [np.array(p, dtype=np.float32) for p in pts]
        ts = rng3.uniform(0.1, 0.95, args.dot_clusters)
        cluster_r = max(2, int(short * args.dot_cluster_radius_pct / 100.0))
        dots = np.zeros_like(arr)
        for t in ts:
            if t < 0.5:
                p = root + (mid - root) * (t / 0.5)
            else:
                p = mid + (tip - mid) * ((t - 0.5) / 0.5)
            for _ in range(int(rng3.integers(4, 6))):
                ox = int(rng3.integers(-cluster_r, cluster_r + 1))
                oy = int(rng3.integers(-cluster_r, cluster_r + 1))
                px, py = int(p[0]) + ox, int(p[1]) + oy
                cv2.circle(dots, (px, py), args.dot_radius_px,
                           (255, 255, 255), -1, cv2.LINE_AA)
        # blur the dots with same kernel, gain=1.0 (already white)
        dot_blurred = streak(dots, strength, tangent, length_px,
                             decay=args.decay, gain=1.0)
        # invert white-on-black → bright trails treated as darken-amount
        out = np.clip(out.astype(np.float32) -
                      dot_blurred.astype(np.float32) * args.dot_strength,
                      0, 255).astype(np.uint8)
        print(f"  added {args.dot_clusters} dot clusters (radius={cluster_r}px)")
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.source))[0]
    suf = args.suffix or f"apply_{args.limb}_L{int(args.length_pct)}_d{args.decay:.0f}_g{args.gain:.1f}{'_flip' if args.flip else ''}"
    out_path = os.path.join(OUT, f"{name}_{suf}.jpg")
    Image.fromarray(out).save(out_path, quality=92)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
