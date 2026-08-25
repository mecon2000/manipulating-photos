#!/usr/bin/env python3
"""Destroy the area around a portrait while the eyes survive untouched.

Seven local, free destructions — strip slicing, leftward colour bleed, inverted
polygons, channel tearing, and so on. All are geometric/pixel operations; nothing
is generated, so identity cannot drift.

Two protection levels:
  face  — the whole face is restored afterwards (destruction is background-only)
  eyes  — only the eyes are restored, so the face gets wrecked along with the rest

The eye mask is never optional. It is built from the 468-point face mesh and
feathered, so the restore has no visible cut line.

  wreck-surround.py --source PHOTO.jpg --all --out-dir DIR
  wreck-surround.py --source PHOTO.jpg --effect strips --protect eyes
"""
import argparse, os, sys, math
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_align as FA

# 468-mesh: eye outlines, so the guard covers lids and lashes, not just the corners
EYE_L = [33, 133, 159, 145, 160, 144, 158, 153]
EYE_R = [263, 362, 386, 374, 387, 373, 385, 380]


def _mesh(img):
    """Full landmark array (not just face_align's named subset)."""
    import mediapipe as mp
    H, W = img.shape[:2]
    base = mp.tasks.BaseOptions(model_asset_path=str(FA.FACE_MODEL))
    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    det = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(img).astype(np.uint8)))
    det.close()
    if not res.face_landmarks:
        return None
    return np.array([[l.x * W, l.y * H] for l in res.face_landmarks[0]], dtype=np.float32)


def eye_mask(img, pts, pad=2.2, feather=0.45):
    """1 = keep original. An ellipse per eye, sized from the eye's own span."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.float32)
    for idx in (EYE_L, EYE_R):
        p = pts[idx]
        c = p.mean(axis=0)
        rx = max(np.ptp(p[:, 0]) * pad * 0.5, W * 0.02)
        ry = max(np.ptp(p[:, 1]) * pad * 0.9, H * 0.012)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    k = int(max(rx, ry) * feather) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


def face_mask(img, pts, grow=1.25, feather=0.25):
    """1 = keep original, over the whole face via its convex hull."""
    H, W = img.shape[:2]
    hull = cv2.convexHull(pts.astype(np.int32))
    c = hull.mean(axis=0)
    hull = ((hull - c) * grow + c).astype(np.int32)
    m = np.zeros((H, W), np.float32)
    cv2.fillConvexPoly(m, hull, 1.0)
    k = int(min(H, W) * feather * 0.25) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


# --- the seven destructions ------------------------------------------------

def fx_strips(a, rng):
    """Cut vertical strips out and slide them; some go to black."""
    out = a.copy(); H, W = a.shape[:2]
    x = 0
    while x < W:
        w = rng.integers(int(W * 0.015), int(W * 0.06))
        roll = int(rng.normal(0, H * 0.06))
        if rng.random() < 0.13:
            out[:, x:x + w] = 0
        else:
            out[:, x:x + w] = np.roll(out[:, x:x + w], roll, axis=0)
        x += w + rng.integers(0, int(W * 0.03))
    return out


def fx_bleed(a, rng):
    """Drag colour leftward in bands, so the pixels visibly run off the subject.

    A max-accumulate alone barely changes a photo whose left side is already bright;
    this instead motion-smears the band and pushes its hue, which reads as bleeding
    at any brightness.
    """
    out = a.astype(np.float32).copy()
    H, W = a.shape[:2]
    y = 0
    while y < H:
        h = int(rng.integers(int(H * .03), int(H * .13)))
        if rng.random() < 0.72:
            band = out[y:y + h].copy()
            length = int(rng.uniform(W * .06, W * .28))
            k = np.zeros((1, length * 2 + 1), np.float32)
            k[0, :length + 1] = 1.0          # weight to the LEFT only
            k /= k.sum()
            smear = cv2.filter2D(band, -1, k, borderType=cv2.BORDER_REPLICATE)
            # let the streak keep the band's brightest values so it reads as a drag
            smear = np.maximum(smear, cv2.filter2D(band, -1, k * 1.6,
                                                   borderType=cv2.BORDER_REPLICATE) * .75)
            ch = int(rng.integers(0, 3))
            smear[..., ch] = np.clip(smear[..., ch] * rng.uniform(1.15, 1.5), 0, 255)
            ramp = np.linspace(1.0, 0.15, band.shape[1], dtype=np.float32)[None, :, None]
            out[y:y + h] = band * (1 - ramp) + smear * ramp
        y += h
    return np.clip(out, 0, 255).astype(np.uint8)


def fx_polygon(a, rng):
    """Long thin polygons; colours inside are inverted."""
    out = a.copy(); H, W = a.shape[:2]
    for _ in range(rng.integers(3, 6)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        L, T = rng.uniform(W * .5, W * 1.4), rng.uniform(H * .03, H * .11)
        ang = rng.uniform(-70, 70)
        box = cv2.boxPoints(((cx, cy), (L, T), ang)).astype(np.int32)
        m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, box, 1)
        sel = m.astype(bool)
        out[sel] = 255 - out[sel]
    return out


def fx_channels(a, rng):
    """Tear the RGB channels apart, per horizontal band."""
    out = a.copy(); H, W = a.shape[:2]
    y = 0
    while y < H:
        h = rng.integers(int(H * .02), int(H * .1))
        for ch in range(3):
            s = int(rng.normal(0, W * 0.035))
            out[y:y + h, :, ch] = np.roll(out[y:y + h, :, ch], s, axis=1)
        y += h
    return out


def fx_solarize(a, rng):
    """Solarise + posterise into hard colour plateaus."""
    x = a.astype(np.float32) / 255.0
    thr = rng.uniform(0.35, 0.6)
    x = np.where(x > thr, 1.0 - x, x)
    lv = int(rng.integers(3, 6))
    x = np.round(x * lv) / lv
    x = x / max(x.max(), 1e-6)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def fx_shards(a, rng):
    """Shatter into triangles; each shard shifts and some inverts."""
    out = a.copy(); H, W = a.shape[:2]
    pts = np.column_stack([rng.uniform(0, W, 90), rng.uniform(0, H, 90)]).astype(np.float32)
    sub = cv2.Subdiv2D((0, 0, W, H))
    for p in pts: sub.insert((float(p[0]), float(p[1])))
    for t in sub.getTriangleList():
        tri = t.reshape(3, 2)
        if tri.min() < 0 or tri[:, 0].max() > W or tri[:, 1].max() > H: continue
        m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, tri.astype(np.int32), 1)
        dx, dy = int(rng.normal(0, W * .012)), int(rng.normal(0, H * .012))
        piece = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        if rng.random() < 0.22: piece = 255 - piece
        sel = m.astype(bool); out[sel] = piece[sel]
    return out


def fx_scanlines(a, rng):
    """Displace scanlines and crush them toward a two-tone palette."""
    out = a.copy(); H, W = a.shape[:2]
    for y in range(H):
        if rng.random() < 0.5:
            out[y] = np.roll(out[y], int(rng.normal(0, W * .02)), axis=0)
    lut = np.clip(np.linspace(-40, 295, 256), 0, 255).astype(np.uint8)
    out = cv2.LUT(out, lut)
    tint = np.array([1.15, 0.92, 1.25], np.float32)
    return np.clip(out.astype(np.float32) * tint, 0, 255).astype(np.uint8)


EFFECTS = {"strips": (fx_strips, "face"), "bleed": (fx_bleed, "face"),
           "polygon": (fx_polygon, "face"), "channels": (fx_channels, "eyes"),
           "solarize": (fx_solarize, "eyes"), "shards": (fx_shards, "eyes"),
           "scanlines": (fx_scanlines, "eyes")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--effect", choices=list(EFFECTS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--protect", choices=["face", "eyes"],
                   help="override the effect's default protection level")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    img = np.asarray(Image.open(args.source).convert("RGB"))
    pts = _mesh(img)
    if pts is None:
        sys.exit("no face found — cannot guarantee the eyes are protected, refusing")
    em, fm = eye_mask(img, pts), face_mask(img, pts)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem

    names = list(EFFECTS) if args.all else [args.effect]
    for i, name in enumerate(names):
        fn, default = EFFECTS[name]
        level = args.protect or default
        rng = np.random.default_rng(args.seed + i)
        wrecked = fn(img, rng).astype(np.float32)
        keep = fm if level == "face" else np.zeros_like(em)
        keep = np.maximum(keep, em)          # eyes are ALWAYS restored
        k = keep[..., None]
        final = np.clip(img * k + wrecked * (1 - k), 0, 255).astype(np.uint8)
        out = out_dir / f"{stem}__{i+1}_{name}_{level}.jpg"
        Image.fromarray(final).save(out, quality=94)
        print(f"  {out.name}  (protected: {level})")


if __name__ == "__main__":
    main()
