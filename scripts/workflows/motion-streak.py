#!/usr/bin/env python3
"""Motion-streak pipeline — 0010x0010 "Infused" aesthetic.

B&W portrait with hard side light; body dissolves into directional streaks
that extend beyond the silhouette via long-exposure stacking; face stays sharp.

Steps:
  0  Relight (not run here — run relighting.py first with --lighting "Window Light")
  1  B&W with crushed blacks + lifted midtones
  2  BiRefNet subject mask
  3  Face-skin mask + radial falloff (preserve zone)
  4  Directional smear (1D motion blur kernel)
  5  Long-exposure stacking (ghost copies screen-blended along motion vector)
  6  Face punch-back (sharp original face composited over streaks)
  7  Subtle grain

Usage:
  motion-streak.py --source photo.jpg                       # all steps
  motion-streak.py --source photo.jpg --up-to-step 4        # stop after smear
  motion-streak.py --source photo.jpg --angle 15 --length-pct 12 --ghosts 10
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter
from masking import build_mask

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/motion-streak-finals")
OUT_DIR = os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates")


def bw_with_curve(img_pil):
    """Crushed blacks + lifted midtones — matches 0010x0010 tone curve."""
    arr = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    x = lum / 255.0
    y = np.where(x < 0.15, x * 0.4, 0.06 + (x - 0.15) * (1.05 / 0.85))
    y = np.clip(y, 0, 1)
    out = (y * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([out, out, out], axis=-1))


def motion_blur_kernel(length_px, angle_deg):
    length_px = max(3, int(length_px))
    k = np.zeros((length_px, length_px), dtype=np.float32)
    k[length_px // 2, :] = 1.0
    ki = Image.fromarray((k * 255).astype(np.uint8)).rotate(angle_deg, resample=Image.BILINEAR)
    k = np.asarray(ki, dtype=np.float32) / 255.0
    s = k.sum()
    return k / s if s > 0 else k


def apply_motion_blur(img_arr_f, kernel):
    out = np.empty_like(img_arr_f)
    for c in range(img_arr_f.shape[2]):
        out[..., c] = convolve(img_arr_f[..., c], kernel, mode="reflect")
    return out


def face_falloff(img_size, face_mask_arr, fade_mult=2.5):
    w, h = img_size
    ys, xs = np.where(face_mask_arr > 127)
    if ys.size < 50:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 3
        r = min(w, h) * 0.12
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return np.clip(1.0 - (d - r) / (r * fade_mult), 0, 1).astype(np.float32)
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0)
    r_outer = r * fade_mult
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - (d - r) / max(1.0, (r_outer - r)), 0, 1).astype(np.float32)


def long_exposure_stack(body_arr, body_alpha, angle_deg, max_shift_px, ghosts=10):
    """Simulate long exposure: shift body along motion vector with decaying alpha, screen-blend.

    body_arr: HxWx3 luminance of body pixels (0 outside body)
    body_alpha: HxW alpha (0-1) of body region
    Returns stacked HxWx3 luminance.
    """
    h, w = body_alpha.shape
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), -np.sin(rad)
    accum = body_arr.copy()
    # progressively longer offsets: 0.1, 0.22, 0.36, 0.52, 0.7, 0.9, ... of max_shift
    for i in range(1, ghosts + 1):
        frac = (i / ghosts) ** 0.85
        shift_px = frac * max_shift_px
        tx, ty = int(round(dx * shift_px)), int(round(dy * shift_px))
        shifted = np.roll(body_arr, (ty, tx), axis=(0, 1))
        shifted_alpha = np.roll(body_alpha, (ty, tx), axis=(0, 1))
        # zero out the wrap-around region
        if tx > 0: shifted[:, :tx] = 0; shifted_alpha[:, :tx] = 0
        elif tx < 0: shifted[:, tx:] = 0; shifted_alpha[:, tx:] = 0
        if ty > 0: shifted[:ty, :] = 0; shifted_alpha[:ty, :] = 0
        elif ty < 0: shifted[ty:, :] = 0; shifted_alpha[ty:, :] = 0
        fade = (1.0 - i / (ghosts + 1)) ** 1.3
        ghost = shifted * fade * shifted_alpha[..., None]
        # screen blend: out = 1 - (1-a)(1-b)
        accum = 255.0 - (255.0 - accum) * (1.0 - ghost / 255.0)
    return accum


POSE_MODEL = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")
LIMB_LANDMARKS = {
    "left-arm":  (11, 13, 15),  # shoulder, elbow, wrist (model's left = viewer's right)
    "right-arm": (12, 14, 16),
    "left-leg":  (23, 25, 27),  # hip, knee, ankle
    "right-leg": (24, 26, 28),
}


def detect_limb_points(img_arr, limb_name):
    """Run MediaPipe pose, return pixel coords of (root, mid, tip) for chosen limb."""
    import mediapipe as mp
    idxs = LIMB_LANDMARKS[limb_name]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr.astype(np.uint8))
    base = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
    opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    result = detector.detect(mp_img)
    detector.close()
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    h, w = img_arr.shape[:2]
    return [(lms[i].x * w, lms[i].y * h) for i in idxs]


def limb_mask_and_gradient(img_size, pts, radius_px):
    """Build a soft mask along polyline (root→mid→tip) with gradient from tip (1.0) to root (0).

    Returns (mask, strength, tangent_deg).
      mask: HxW float 0-1 — which pixels are "in the limb"
      strength: HxW float 0-1 — how much streak to apply (max near tip)
      tangent_deg: unit direction root→tip in degrees (for streak angle)
    """
    w, h = img_size
    root, mid, tip = [np.array(p, dtype=np.float32) for p in pts]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def seg_dist_and_t(a, b):
        ab = b - a
        L2 = float(ab[0] ** 2 + ab[1] ** 2) or 1.0
        t = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2
        t = np.clip(t, 0, 1)
        closest_x = a[0] + t * ab[0]
        closest_y = a[1] + t * ab[1]
        d = np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)
        return d, t

    d1, t1 = seg_dist_and_t(root, mid)
    d2, t2 = seg_dist_and_t(mid, tip)
    # for each pixel pick the closer segment; compute param-along-limb in [0,1] (0=root, 1=tip)
    use_seg2 = d2 < d1
    d = np.where(use_seg2, d2, d1)
    param = np.where(use_seg2, 0.5 + 0.5 * t2, 0.5 * t1)  # 0 root → 1 tip

    mask = np.clip(1.0 - d / radius_px, 0, 1).astype(np.float32)
    strength = (mask * param).astype(np.float32)  # gradient: 0 near root, max near tip

    # streak direction: tip → root (trail behind as limb swept outward)
    dx, dy = float(root[0] - tip[0]), float(root[1] - tip[1])
    tangent_deg = float(np.degrees(np.arctan2(-dy, dx)))  # 0=right, 90=up
    return mask, strength, tangent_deg


def limb_streak_effect(bw_arr, limb_mask, limb_strength, tangent_deg, length_px,
                        ghosts, bright_threshold=80, gain=2.5):
    """Smear the limb's brightest pixels in a straight line across the image.

    Isolates highlights inside the limb (pixels >= bright_threshold), then
    projects them along tangent direction with slow alpha decay. Shifted
    copies are NOT re-gated by the limb mask, so streaks extend beyond the
    limb silhouette — mimicking long-exposure motion trails on highlights.
    """
    h, w, _ = bw_arr.shape
    rad = np.deg2rad(tangent_deg)
    dx, dy = np.cos(rad), -np.sin(rad)
    lum = bw_arr.mean(axis=2)
    # saturate quickly above threshold (80-unit window), so mid-grays streak too
    soft = np.clip((lum - bright_threshold) / 80.0, 0, 1)
    source_alpha = (soft * limb_strength).astype(np.float32)
    source_rgb = np.clip(bw_arr * gain, 0, 255) * source_alpha[..., None]
    accum = bw_arr.copy()
    for i in range(1, ghosts + 1):
        frac = (i / ghosts) ** 0.85
        shift_px = frac * length_px
        tx, ty = int(round(dx * shift_px)), int(round(dy * shift_px))
        shifted = np.roll(source_rgb, (ty, tx), axis=(0, 1))
        if tx > 0: shifted[:, :tx] = 0
        elif tx < 0: shifted[:, tx:] = 0
        if ty > 0: shifted[:ty, :] = 0
        elif ty < 0: shifted[ty:, :] = 0
        fade = (1.0 - i / (ghosts + 1)) ** 0.35   # very slow falloff
        ghost = shifted * fade
        # max blend: streak wins wherever it's brighter than existing pixel
        accum = np.maximum(accum, ghost)
    return accum


def slitscan_warp(img_arr, body_alpha, n_slices=6, max_jitter_pct=0.12, seed=None):
    """Fake slit-scan: each column sampled from a different Y offset, stacked.

    Mimics the temporal-compression look where X-axis becomes time — body
    stretches into vertical striations.
    """
    h, w, _ = img_arr.shape
    rng = np.random.default_rng(seed)
    max_jitter = h * max_jitter_pct
    yy = np.arange(h, dtype=np.int32)
    xs = np.linspace(0, 1, w, dtype=np.float32)
    accum = img_arr.copy()
    for i in range(n_slices):
        # smooth per-column offset: sum of a few sine components
        off = np.zeros(w, dtype=np.float32)
        for k in range(1, 5):
            amp = rng.standard_normal() * max_jitter / k
            phase = rng.uniform(0, 2 * np.pi)
            off += amp * np.sin(k * np.pi * xs + phase)
        ys = np.clip(yy[:, None] - off[None, :].astype(np.int32), 0, h - 1)
        # sample per-column
        warped = np.take_along_axis(img_arr, ys[..., None].repeat(3, axis=-1), axis=0)
        alpha = np.take_along_axis(body_alpha, ys, axis=0)
        fade = (1.0 - i / (n_slices + 1)) ** 1.2
        ghost = warped * alpha[..., None] * fade
        accum = 255.0 - (255.0 - accum) * (1.0 - ghost / 255.0)
    return accum


def add_grain(img_arr, strength=0.025, seed=None):
    rng = np.random.default_rng(seed)
    h, w = img_arr.shape[:2]
    noise = rng.standard_normal((h, w)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=0.7)
    mod = noise[..., None] * (strength * 255.0)
    return np.clip(img_arr.astype(np.float32) + mod, 0, 255).astype(np.uint8)


def run(source, angle, length_pct, ghosts, up_to_step, seed, out_suffix, mode="streak", slitscan_slices=6, slitscan_jitter=0.12, limb="right-arm", limb_radius_pct=6.0):
    t0 = time.time()
    name = os.path.splitext(os.path.basename(source))[0]
    tag = f"streak_a{int(angle)}_l{int(length_pct)}_g{ghosts}_s{up_to_step}"
    if out_suffix: tag += f"_{out_suffix}"
    out_dir = os.path.join(OUT_DIR, f"{name}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] out={out_dir}  up_to={up_to_step}")

    img = Image.open(source).convert("RGB")
    w, h = img.size

    # Step 1: B&W
    bw = bw_with_curve(img)
    bw.save(os.path.join(out_dir, "1_bw.jpg"), quality=92)
    bw_arr = np.asarray(bw, dtype=np.float32)
    print(f"  step1 bw done")
    if up_to_step == 1:
        return _finalize(out_dir, name, tag, np.asarray(bw))

    # Step 2: subject mask
    subj_mask, _ = build_mask(bw, affect="subject", output_dir=out_dir, feather=1.0)
    subj_arr = np.asarray(subj_mask, dtype=np.float32) / 255.0
    print(f"  step2 subject mask (coverage={subj_arr.mean()*100:.1f}%)")
    if up_to_step == 2:
        vis = (bw_arr * subj_arr[..., None]).astype(np.uint8)
        return _finalize(out_dir, name, tag, vis)

    # Step 3: face falloff
    try:
        face_mask, _ = build_mask(bw, affect="face-skin", output_dir=None, feather=0.5)
        fall = face_falloff((w, h), np.asarray(face_mask), fade_mult=2.5)
    except Exception as e:
        print(f"  face-skin failed: {e}, using top-third fallback")
        fall = face_falloff((w, h), np.zeros((h, w), dtype=np.uint8), fade_mult=2.5)
    Image.fromarray((fall * 255).astype(np.uint8)).save(os.path.join(out_dir, "3_face_falloff.png"))
    print(f"  step3 face falloff done")
    if up_to_step == 3:
        # show falloff as cyan tint on bw
        vis = bw_arr.copy()
        vis[..., 2] = np.clip(vis[..., 2] + fall * 60, 0, 255)
        return _finalize(out_dir, name, tag, vis.astype(np.uint8))

    # Step 4: directional smear
    length_px = int(min(w, h) * length_pct / 100.0)
    kernel = motion_blur_kernel(length_px, angle)
    smeared = apply_motion_blur(bw_arr, kernel)
    print(f"  step4 smear: {length_px}px @ {angle}°")
    # body = subject minus face protection
    body_w = (subj_arr * (1.0 - fall))[..., None]
    face_w = fall[..., None]
    composite_4 = (bw_arr * (1.0 - subj_arr[..., None])
                   + bw_arr * face_w * subj_arr[..., None]
                   + smeared * body_w)
    if up_to_step == 4:
        return _finalize(out_dir, name, tag, np.clip(composite_4, 0, 255).astype(np.uint8))

    # Step 5: long-exposure stacking (or slitscan if mode=slitscan)
    body_layer = bw_arr * subj_arr[..., None]  # body luminance on black
    body_alpha = subj_arr * (1.0 - fall)        # fade body alpha near face
    if mode == "slitscan":
        stacked = slitscan_warp(body_layer, body_alpha,
                                n_slices=slitscan_slices,
                                max_jitter_pct=slitscan_jitter, seed=seed)
        print(f"  step5 slitscan: {slitscan_slices} slices, jitter={slitscan_jitter*100:.0f}%")
    elif mode == "limb-streak":
        pts = detect_limb_points(np.asarray(img), limb)
        if pts is None:
            print(f"  limb-streak: pose not detected, falling back to full-body stacking")
            stacked = long_exposure_stack(body_layer, body_alpha, angle,
                                          max_shift_px=length_px * 2.5, ghosts=ghosts)
        else:
            radius_px = min(w, h) * limb_radius_pct / 100.0
            lmask, lstrength, tangent = limb_mask_and_gradient((w, h), pts, radius_px)
            # save debug viz
            Image.fromarray((lstrength * 255).astype(np.uint8)).save(
                os.path.join(out_dir, "4_limb_strength.png"))
            # intersect with subject mask to not streak outside body
            lstrength = lstrength * subj_arr
            lmask = lmask * subj_arr
            stacked = limb_streak_effect(bw_arr, lmask, lstrength, tangent,
                                         length_px * 10.0, max(ghosts, 20))
            print(f"  step5 limb-streak: limb={limb} tangent={tangent:.0f}° radius={radius_px:.0f}px")
    else:
        stacked = long_exposure_stack(body_layer, body_alpha, angle,
                                      max_shift_px=length_px * 2.5, ghosts=ghosts)
        print(f"  step5 stacking: {ghosts} ghosts, max_shift={length_px * 2.5:.0f}px")
    # blend stacked with smear for soft seams
    effect = 0.65 * stacked + 0.35 * smeared * body_w
    # composite: stacked body over BG, sharp face punched back
    composite_5 = (bw_arr * (1.0 - subj_arr[..., None])  # BG preserved
                   + effect                                 # stacked body
                   + bw_arr * face_w * subj_arr[..., None]) # face on top (additive-ish)
    composite_5 = np.clip(composite_5, 0, 255)
    if up_to_step == 5:
        return _finalize(out_dir, name, tag, composite_5.astype(np.uint8))

    # Step 6: face punch-back (already partially applied; strengthen)
    sharp_face_zone = (fall > 0.5)[..., None].astype(np.float32)
    composite_6 = composite_5 * (1 - sharp_face_zone) + bw_arr * sharp_face_zone
    print(f"  step6 face punch-back")
    if up_to_step == 6:
        return _finalize(out_dir, name, tag, composite_6.astype(np.uint8))

    # Step 7: grain
    final = add_grain(composite_6.astype(np.uint8), strength=0.025, seed=seed)
    print(f"  step7 grain  ({time.time()-t0:.1f}s total)")
    return _finalize(out_dir, name, tag, final)


def _finalize(out_dir, name, tag, arr):
    step_path = os.path.join(out_dir, f"final_{tag}.jpg")
    Image.fromarray(arr).save(step_path, quality=92)
    os.makedirs(FINALS, exist_ok=True)
    final_path = os.path.join(FINALS, f"{name}_{tag}.jpg")
    Image.fromarray(arr).save(final_path, quality=92)
    print(f"  → {final_path}")
    # push to phone (best-effort)
    try:
        from notify import push_image
        push_image(final_path, title=f"motion-streak {tag}", body=name)
    except Exception:
        pass
    return final_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="relit source photo (run relighting.py first)")
    p.add_argument("--angle", type=float, default=0.0, help="motion angle degrees (0=horizontal, 90=up)")
    p.add_argument("--length-pct", type=float, default=10.0, help="smear kernel length as % of min(w,h)")
    p.add_argument("--ghosts", type=int, default=10, help="number of long-exposure ghost copies")
    p.add_argument("--up-to-step", type=int, default=7, choices=range(1, 8),
                   help="stop after this step (1=bw, 2=mask, 3=face, 4=smear, 5=stack, 6=face-punch, 7=grain)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--suffix", default="", help="extra tag for output filename")
    p.add_argument("--mode", choices=["streak", "slitscan", "limb-streak"], default="streak",
                   help="streak=full-body stacking; slitscan=vertical time-stripes; limb-streak=one limb only")
    p.add_argument("--slitscan-slices", type=int, default=6)
    p.add_argument("--slitscan-jitter", type=float, default=0.12, help="max Y jitter as fraction of image height")
    p.add_argument("--limb", choices=list(LIMB_LANDMARKS.keys()), default="right-arm")
    p.add_argument("--limb-radius-pct", type=float, default=6.0,
                   help="limb mask radius as %% of min(w,h)")
    a = p.parse_args()
    run(a.source, a.angle, a.length_pct, a.ghosts, a.up_to_step, a.seed, a.suffix,
        mode=a.mode, slitscan_slices=a.slitscan_slices, slitscan_jitter=a.slitscan_jitter,
        limb=a.limb, limb_radius_pct=a.limb_radius_pct)
