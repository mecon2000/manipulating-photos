#!/usr/bin/env python3
"""LAB color grading post-step. Three modes:

  warm-cool   — radial: warm subject (face ellipse), cool BG. Uses MediaPipe
                pose to place the ellipse; falls back to top-third center.
  split       — luminance-driven split-tone (shadows→teal, highlights→orange).
  wash:<color> — single global LAB tint (red, teal, ochre, amber, sepia,
                rose, cyan, blue-hour, magenta, emerald). Same palette as
                color-bath.py.

Pure local. No API cost. ~1 second.

Usage:
  color_grade.py --source PHOTO --mode warm-cool [--strength 0.2]
  color_grade.py --source PHOTO --mode split
  color_grade.py --source PHOTO --mode wash:teal
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

POSE_MODEL = Path("~/openclaw-venv/mediapipe_models/pose_landmarker.task").expanduser()


# ---- LAB helpers (cv2 LAB: L 0-255, a/b 0-255 centered at 128) ------------

def to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def from_lab(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# ---- Face ellipse mask (lifted from surreal_with_face.py) ----------------

def face_ellipse_mask_biased(img_size, source_arr, inner_mult=1.0, outer_mult=6.0,
                              vertical_bias=1.6):
    """Same as face_ellipse_mask but with a downward-only stretch — the outer
    ellipse extends further DOWN (toward shoulders/torso) than UP, so warmth
    naturally hits face + upper body instead of being a symmetric blob."""
    import mediapipe as mp
    W, H = img_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, a, b, angle_deg = W*0.5, H*0.33, min(W,H)*0.13, min(W,H)*0.13, 0.0
    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=source_arr.astype(np.uint8))
        base = mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL))
        opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
        det = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        res = det.detect(mp_img); det.close()
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            def P(i): return np.array([lms[i].x*W, lms[i].y*H], np.float32)
            nose, le, re, lear, rear, lm, rm = P(0),P(2),P(5),P(7),P(8),P(9),P(10)
            eye_mid   = (le+re)/2
            mouth_mid = (lm+rm)/2
            cx, cy = np.stack([nose, le, re, lm, rm]).mean(axis=0)
            axis = eye_mid - mouth_mid
            an = axis / (np.linalg.norm(axis) + 1e-6)
            angle_deg = float(np.degrees(np.arctan2(-an[1], an[0]))) - 90.0
            face_h = float(np.linalg.norm(eye_mid - mouth_mid))
            face_w = float(np.linalg.norm(lear - rear)) * 0.4
            a, b = face_w, face_h
    except Exception:
        pass
    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(-rad), np.sin(-rad)
    dx, dy = xx - cx, yy - cy
    # rotate into face-local frame
    lx_unscaled = dx*cos_t - dy*sin_t
    ly_unscaled = dx*sin_t + dy*cos_t
    # vertical bias: when the local-y is positive (DOWN in face frame, toward
    # body), divide by a larger semi-axis so the ellipse reaches further down
    b_eff = np.where(ly_unscaled > 0, b * vertical_bias, b)
    lx = lx_unscaled / a
    ly = ly_unscaled / b_eff
    d = np.sqrt(lx**2 + ly**2)
    t = np.clip((d - inner_mult) / max(outer_mult - inner_mult, 1e-3), 0, 1)
    return (1.0 - t).astype(np.float32)


def face_ellipse_mask(img_size, source_arr, inner_mult=1.0, outer_mult=3.5):
    import mediapipe as mp
    W, H = img_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, a, b, angle_deg = W*0.5, H*0.33, min(W,H)*0.13, min(W,H)*0.13, 0.0
    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=source_arr.astype(np.uint8))
        base = mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL))
        opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
        det = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        res = det.detect(mp_img); det.close()
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            def P(i): return np.array([lms[i].x*W, lms[i].y*H], np.float32)
            nose, le, re, lear, rear, lm, rm = P(0),P(2),P(5),P(7),P(8),P(9),P(10)
            eye_mid   = (le+re)/2
            mouth_mid = (lm+rm)/2
            cx, cy = np.stack([nose, le, re, lm, rm]).mean(axis=0)
            axis = eye_mid - mouth_mid
            an = axis / (np.linalg.norm(axis) + 1e-6)
            angle_deg = float(np.degrees(np.arctan2(-an[1], an[0]))) - 90.0
            face_h = float(np.linalg.norm(eye_mid - mouth_mid))
            face_w = float(np.linalg.norm(lear - rear)) * 0.4
            a, b = face_w, face_h
    except Exception:
        pass
    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(-rad), np.sin(-rad)
    dx, dy = xx - cx, yy - cy
    lx = (dx*cos_t - dy*sin_t) / a
    ly = (dx*sin_t + dy*cos_t) / b
    d  = np.sqrt(lx**2 + ly**2)
    t  = np.clip((d - inner_mult) / max(outer_mult - inner_mult, 1e-3), 0, 1)
    return (1.0 - t).astype(np.float32)


# ---- Mode A: radial warm-cool --------------------------------------------

def grade_warm_cool(rgb_arr, mask, strength=0.4, subject_mask=None):
    """Inside mask: warm.  Outside (still on subject): cold-blue.

    If `subject_mask` is given (0-1, H×W), the entire warm-cool delta is gated
    by it — BG is left UNTOUCHED, which is what we want: the body gets a
    warm-to-cool gradient, surroundings stay as the surreal stylization made them.

    Warm and cool are kept closer in LAB space (smaller deltas) so the gradient
    is subtle, not a hard hot/cold split.
    """
    lab = to_lab(rgb_arr)
    # Closer pair: less orange warm, less blue cool. Per-channel tweaked 2026-04-26.
    warm = np.array([0, +9, +9],  dtype=np.float32) * (strength * 5)
    cool = np.array([0, +2, -7],  dtype=np.float32) * (strength * 5)
    m3 = mask[..., None]
    delta = warm * m3 + cool * (1.0 - m3)
    if subject_mask is not None:
        delta = delta * subject_mask[..., None]
    return from_lab(lab + delta)


# ---- Mode B: split-tone teal-orange --------------------------------------

def grade_split_tone(rgb_arr, strength=0.25):
    """Shadows → teal (-a -b on darker), highlights → orange (+a +b on lighter)."""
    lab = to_lab(rgb_arr)
    L = lab[..., 0:1]
    # weights: 1 at L=0 (full shadow) → 0 at L=128, then 0 → 1 at L=255 (full highlight)
    w_shadow = np.clip(1.0 - L / 128.0, 0, 1)
    w_high   = np.clip((L - 128.0) / 127.0, 0, 1)
    teal   = np.array([0, -10, -10], dtype=np.float32) * (strength * 5)
    orange = np.array([0, +14, +18], dtype=np.float32) * (strength * 5)
    delta = teal * w_shadow + orange * w_high
    return from_lab(lab + delta)


# ---- Mode C: single global wash ------------------------------------------

WASH_PALETTE = {
    # name → (Δa, Δb)  — same color choices as color-bath.py
    "red":         (+30, +20),
    "ochre":       (+12, +35),
    "teal":        (-25, -10),
    "amber":       (+18, +30),
    "blue-hour":   (-15, -30),
    "rose":        (+25, +5),
    "sepia":       (+10, +20),
    "emerald":     (-20, +15),
    "magenta":     (+25, -10),
    "cyan":        (-25, -5),
}

def grade_wash(rgb_arr, color, strength=0.22):
    if color not in WASH_PALETTE:
        raise ValueError(f"unknown wash color {color!r}; pick from {sorted(WASH_PALETTE)}")
    da, db = WASH_PALETTE[color]
    lab = to_lab(rgb_arr)
    lab[..., 1] += da * strength
    lab[..., 2] += db * strength
    return from_lab(lab)


# ---- Driver --------------------------------------------------------------

def grade(rgb_arr, mode, strength=0.4, mask_inner=0.2, mask_outer=6.0,
          warm_vertical_bias=1.6):
    """Single entry point — used both as CLI and as a library import from
    surreal_with_face.py.

    For warm-cool mode, `mask_outer` controls falloff size as multiple of the
    face ellipse, and `warm_vertical_bias` extends the ellipse downward more
    than sideways (so warm light reaches upper torso, not just face).
    Defaults: outer_mult=6, vertical bias 1.6×.
    """
    if mode == "off":
        return rgb_arr
    if mode == "warm-cool":
        mask = face_ellipse_mask_biased(
            (rgb_arr.shape[1], rgb_arr.shape[0]), rgb_arr,
            inner_mult=mask_inner, outer_mult=mask_outer,
            vertical_bias=warm_vertical_bias)
        # subject mask via BiRefNet — keeps BG untouched
        try:
            from masking import build_mask
            from PIL import Image as _PI
            pil = _PI.fromarray(rgb_arr)
            sm_pil, _ = build_mask(pil, affect="subject", output_dir=None, feather=2.0)
            subject_mask = (np.asarray(sm_pil).astype(np.float32) / 255.0)
            if subject_mask.ndim == 3: subject_mask = subject_mask[..., 0]
            print(f"  [grade] subject mask coverage {subject_mask.mean()*100:.1f}%")
        except Exception as e:
            print(f"  [grade] subject mask failed ({e}) — falling back to ellipse-only")
            subject_mask = None
        return grade_warm_cool(rgb_arr, mask, strength, subject_mask=subject_mask)
    if mode == "split":
        return grade_split_tone(rgb_arr, strength)
    if mode.startswith("wash:"):
        return grade_wash(rgb_arr, mode.split(":", 1)[1], strength)
    raise ValueError(f"unknown grade mode {mode!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mode", required=True,
                   help="warm-cool | split | wash:<color> | off")
    p.add_argument("--strength", type=float, default=0.25)
    p.add_argument("--mask-inner", type=float, default=1.0)
    p.add_argument("--mask-outer", type=float, default=3.5)
    p.add_argument("--out", default=None,
                   help="output path (default: source__graded.jpg)")
    args = p.parse_args()

    img = Image.open(args.source).convert("RGB")
    arr = np.asarray(img)
    t0 = time.time()
    out = grade(arr, args.mode, strength=args.strength,
                mask_inner=args.mask_inner, mask_outer=args.mask_outer)
    out_path = args.out or str(Path(args.source).with_suffix(
        f".__graded_{args.mode.replace(':','-')}.jpg"))
    Image.fromarray(out).save(out_path, quality=92)
    print(f"  → {out_path}  ({time.time()-t0:.2f}s)")


if __name__ == "__main__":
    main()
