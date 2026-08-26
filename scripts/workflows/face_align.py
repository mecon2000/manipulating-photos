#!/usr/bin/env python3
"""Face landmark detection + source→target alignment for face compositing.

Why this exists: `surreal_with_face.py` used to build its face ellipse from the
SOURCE photo and composite it onto the STYLIZED output. become-image moves and
rotates the head, so the oval landed off-face (and dragged the wrong tones with
it). Here we detect the face in BOTH images, warp the source onto the target's
face with a similarity transform, and derive the ellipse from the TARGET.

Uses MediaPipe face_landmarker (468 pts) — far more reliable than the pose
landmarker's 7 face points, which miss entirely on some portraits.
"""
from pathlib import Path
import numpy as np
import cv2

FACE_MODEL = Path("~/openclaw-venv/mediapipe_models/face_landmarker.task").expanduser()

# 468-point canonical face mesh indices
IDX = {
    "forehead": 10, "chin": 152,
    "eye_l": 33, "eye_r": 263,
    "nose": 1, "mouth_l": 61, "mouth_r": 291,
    "cheek_l": 234, "cheek_r": 454,
}
# points used to solve the source→target transform
ALIGN_KEYS = ["forehead", "chin", "eye_l", "eye_r", "nose", "mouth_l", "mouth_r",
              "cheek_l", "cheek_r"]


def _normalize_for_detection(img_arr):
    """Neutralize colour cast + lift shadows, for detection only.

    Heavily gelled low-key portraits (a face lit by one orange flare against
    black) read as "no face" to MediaPipe. Gray-world white balance plus CLAHE
    on L recovers them; the transform is never applied to output pixels.
    """
    a = img_arr.astype(np.float32)
    means = a.reshape(-1, 3).mean(axis=0) + 1e-6
    a = np.clip(a * (means.mean() / means), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def landmarks(img_arr, _retry=True, max_edge=1600):
    """Return {name: (x, y)} in pixel coords, or None if no face found.

    Detection runs on a downscaled copy and the result is scaled back. MediaPipe
    misses small faces in very large frames — a 3078x5472 crop with the subject a
    few percent of the height detected nothing at full size and fine at 1600px —
    and every caller that worked was already downscaling by accident.
    """
    import mediapipe as mp
    H, W = img_arr.shape[:2]
    scale = 1.0
    if max(H, W) > max_edge:
        scale = max_edge / max(H, W)
        img_arr = cv2.resize(img_arr, (int(W * scale), int(H * scale)),
                             interpolation=cv2.INTER_AREA)
        H, W = img_arr.shape[:2]
    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(img_arr).astype(np.uint8))
        base = mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL))
        opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
        det = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        res = det.detect(mp_img)
        det.close()
    except Exception as e:
        print(f"  face_landmarker error: {e}")
        return None
    if not res.face_landmarks:
        if _retry:
            print("  no face — retrying on normalized copy")
            got = landmarks(_normalize_for_detection(img_arr), _retry=False,
                            max_edge=max_edge)
            if got is not None and scale != 1.0:
                got = {k: v / scale for k, v in got.items()}   # retry saw the small copy
            return got
        return None
    lms = res.face_landmarks[0]
    return {k: np.array([lms[i].x * W / scale, lms[i].y * H / scale], dtype=np.float32)
            for k, i in IDX.items()}


def similarity_transform(src_pts, dst_pts):
    """2x3 affine (rotation+uniform scale+translation) mapping src onto dst."""
    s = np.stack([src_pts[k] for k in ALIGN_KEYS])
    d = np.stack([dst_pts[k] for k in ALIGN_KEYS])
    M, _ = cv2.estimateAffinePartial2D(s, d, method=cv2.LMEDS)
    return M


def warp(img_arr, M, size_wh):
    return cv2.warpAffine(img_arr, M, size_wh, flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)


def ellipse_from_landmarks(pts, width_mult=1.0, height_mult=1.0):
    """Face ellipse geometry (center, semi-axes, angle°) from landmarks.

    Spans forehead→chin vertically and cheek→cheek horizontally, so the whole
    face is inside — unlike the old eye-to-mouth box, which covered only the
    middle third and read as an oddly rotated patch on tilted heads.
    """
    forehead, chin = pts["forehead"], pts["chin"]
    axis = forehead - chin
    b = float(np.linalg.norm(axis)) * 0.5 * height_mult          # semi, along face axis
    a = float(np.linalg.norm(pts["cheek_l"] - pts["cheek_r"])) * 0.5 * width_mult
    cx, cy = (forehead + chin) / 2.0
    n = axis / (np.linalg.norm(axis) + 1e-6)
    angle_deg = float(np.degrees(np.arctan2(-n[1], n[0]))) - 90.0   # 0 = upright
    return (float(cx), float(cy)), (a, b), angle_deg


def radial_mask(size_wh, center, semi, angle_deg,
                inner_mult=1.0, outer_mult=2.2, falloff_power=1.0):
    """Soft elliptical mask: 1 inside inner_mult, ramping to 0 at outer_mult."""
    W, H = size_wh
    a, b = semi
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(-rad), np.sin(-rad)
    dx, dy = xx - cx, yy - cy
    lx = (dx * cos_t - dy * sin_t) / max(a, 1e-6)
    ly = (dx * sin_t + dy * cos_t) / max(b, 1e-6)
    d = np.sqrt(lx**2 + ly**2)
    t = np.clip((d - inner_mult) / max(outer_mult - inner_mult, 1e-3), 0, 1)
    return (1.0 - t ** falloff_power).astype(np.float32)


def match_in_region(src, dst, mask, thresh=0.5, strength=1.0, scope="hybrid"):
    """Reinhard mean/std color transfer in LAB, using ONLY true face pixels.

    Two earlier attempts failed here. Matching histograms over the whole frame
    turned faces black — a mostly-dark stylized canvas dragged skin down to its
    global histogram. Matching over the mask's bounding box turned them red —
    the box still contained mostly background, so per-channel RGB mapping sent
    skin somewhere absurd. Sampling only mask>0.5 means the stats come from the
    stylized FACE, which is skin, so skin gets matched to skin. LAB keeps
    luminance and chroma independent, and mean/std is stable on small samples
    where full histogram matching is not.

    `scope` controls where the TARGET stats come from:
      "face"   — the stylized face only. Safest, but leaves the head looking
                 dull and muted next to a punchy high-contrast scene.
      "frame"  — the whole stylized image. Picks up its contrast and palette,
                 but a dark canvas drags face brightness down with it.
      "hybrid" — (default) brightness anchored on the face so exposure stays
                 correct, while contrast (L spread) and colour (a/b) come from
                 the frame, so the head belongs to the picture.
    """
    sel = mask > thresh
    if sel.sum() < 500:
        sel = mask > 0.05
    if sel.sum() < 500:
        return src
    if src.ndim == 2:
        s32, d32 = src.astype(np.float32), dst.astype(np.float32)
        sm, ss = s32[sel].mean(), s32[sel].std() + 1e-6
        dm, ds = d32[sel].mean(), d32[sel].std() + 1e-6
        out = (s32 - sm) * (ds / ss) + dm
        out = src.astype(np.float32) * (1 - strength) + out * strength
        return np.clip(out, 0, 255).astype(np.uint8)

    s_lab = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    d_lab = cv2.cvtColor(dst.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    frame = np.ones(mask.shape, dtype=bool)
    out = s_lab.copy()
    for c in range(3):
        if scope == "face":
            stat_sel, use_frame_mean = sel, False
        elif scope == "frame":
            stat_sel, use_frame_mean = frame, True
        else:                       # hybrid
            stat_sel = frame if (c > 0) else frame
            use_frame_mean = (c > 0)          # keep L mean from the face
        sm, ss = s_lab[..., c][sel].mean(), s_lab[..., c][sel].std() + 1e-6
        dm_face = d_lab[..., c][sel].mean()
        dm_frame = d_lab[..., c][stat_sel].mean()
        ds = d_lab[..., c][stat_sel].std() + 1e-6
        dm = dm_frame if use_frame_mean else dm_face
        # cap the stretch so a low-variance target can't posterize the face
        gain = float(np.clip(ds / ss, 0.5, 3.0))
        out[..., c] = (s_lab[..., c] - sm) * gain + dm
    out = s_lab * (1 - strength) + out * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
