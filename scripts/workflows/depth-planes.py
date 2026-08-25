#!/usr/bin/env python3
"""Composite a portrait into 5 depth planes with lens-like defocus.

    plane 1  nearest camera, heaviest blur   decorations only
    plane 2  near,           strong blur     decorations only
    plane 3  THE PORTRAIT,   sharp
    plane 4  behind,         light blur      decorations only
    plane 5  farthest,       moderate blur   decorations, partly hidden by her

Planes 4 and 5 only read if the portrait has an alpha channel, so the subject is
cut out with MediaPipe's selfie segmenter — BiRefNet would be better but it is a
fal.ai call and that balance is currently exhausted.

Decoration plates arrive as flat art on a cream field; the cream is keyed to
transparent so the plates can stack. The eyes are never covered: the front planes
have an eye-shaped hole punched in their alpha.

  depth-planes.py --portrait P.jpg --plates p1.jpg p2.jpg p4.jpg p5.jpg --out OUT.jpg
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_align as FA

BLUR = {1: 24, 2: 10, 4: 5, 5: 12}          # near blurs harder than far
SELFIE = Path("~/openclaw-venv/mediapipe_models/selfie_multiclass.tflite").expanduser()
EYE_L = [33, 133, 159, 145, 160, 144, 158, 153]
EYE_R = [263, 362, 386, 374, 387, 373, 385, 380]


def subject_alpha(img):
    """Alpha for the person. Categories 1-5 are all body parts; 0 is background."""
    import mediapipe as mp
    base = mp.tasks.BaseOptions(model_asset_path=str(SELFIE))
    opts = mp.tasks.vision.ImageSegmenterOptions(base_options=base,
                                                 output_category_mask=True)
    with mp.tasks.vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(img).astype(np.uint8)))
    cat = res.category_mask.numpy_view()
    a = (cat > 0).astype(np.float32)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.GaussianBlur(a, (9, 9), 0)


def mesh(img):
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
    return np.array([[l.x * W, l.y * H] for l in res.face_landmarks[0]], np.float32)


def eye_hole(shape, pts, pad=3.0):
    """0 inside the eyes, 1 elsewhere — multiplied into the FRONT planes' alpha."""
    H, W = shape
    m = np.ones((H, W), np.float32)
    if pts is None:
        return m
    r = 0
    for idx in (EYE_L, EYE_R):
        p = pts[idx]
        c = p.mean(axis=0)
        rx = max(np.ptp(p[:, 0]) * pad * 0.5, W * .03)
        ry = max(np.ptp(p[:, 1]) * pad * 1.1, H * .02)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(rx), int(ry)), 0, 0, 360, 0.0, -1)
        r = max(r, rx)
    k = int(r * 0.5) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


def key_cream(img, tol=26.0, soft=16.0):
    """Alpha from distance to the plate's dominant colour.

    Sampling the corners is wrong for these plates: the decorations are deliberately
    driven to the edges, so a corner sample returns burnt orange and the whole plate
    keys opaque. The paper ground is instead the most COMMON colour by area, found
    with a coarse histogram, which holds regardless of where the art sits.
    """
    a = img.astype(np.float32)
    q = (a // 16).astype(np.int32)
    flat = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    mode = np.bincount(flat.ravel()).argmax()
    sel = flat == mode
    bg = np.median(a[sel].reshape(-1, 3), axis=0)
    d = np.linalg.norm(a - bg, axis=2)
    return np.clip((d - tol) / soft, 0, 1).astype(np.float32)


def face_relief(shape, pts, strength, spread=1.9):
    """Thin out the FRONT planes near the face.

    "Decorations anywhere, eyes guaranteed" still looks wrong when a large opaque
    shape parks on the mouth. This is a soft radial reduction centred on the face,
    so shapes thin out as they approach her but are never forbidden — the hard
    guarantee stays with the eye hole.
    """
    H, W = shape
    if pts is None or strength <= 0:
        return np.ones((H, W), np.float32)
    c = pts.mean(axis=0)
    sigma = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * spread
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
    return (1.0 - strength * np.exp(-d2 / (2 * sigma * sigma))).astype(np.float32)


def over(dst, src_rgb, src_a):
    a = src_a[..., None]
    return dst * (1 - a) + src_rgb * a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--portrait", required=True)
    p.add_argument("--plates", nargs=4, required=True,
                   help="plates for planes 1 2 4 5, in that order")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0,
                   help="shrink the portrait inside the frame (0.8 = 10%% clear each side), "
                        "so the side planes have room to read")
    p.add_argument("--mask-from", default=None,
                   help="segment THIS image instead of the portrait. A stylized frame with "
                        "flat graphic clothing barely reads as a person (27%% coverage vs 58%% "
                        "on the photo it came from), so the cut-out tears. Defaults to the "
                        "matching frame under the reel's _sources/crop_916/ when present")
    p.add_argument("--subject-relief", type=float, default=0.85,
                   help="0-1: how much the near planes thin out over her whole silhouette. "
                        "The face relief alone only clears a disc around the head, which "
                        "leaves the torso to be swallowed by front shapes")
    p.add_argument("--front-relief", type=float, default=0.75,
                   help="0-1: how much the near planes thin out over her face")
    args = p.parse_args()

    port = np.asarray(Image.open(args.portrait).convert("RGB"))
    H, W = port.shape[:2]
    pts = mesh(port)
    if pts is None:
        print("  no face found — front planes will not be eye-masked", file=sys.stderr)
    mask_src = args.mask_from
    if not mask_src:                       # sibling original from the same reel, same crop
        stem = Path(args.portrait).stem.split("__style")[0]
        cand = Path(args.portrait).parent.parent / "_sources" / "crop_916" / f"{stem}.jpg"
        if cand.is_file():
            mask_src = str(cand)
            print(f"  masking from the source crop: {cand.name}")
    if mask_src:
        m = np.asarray(Image.open(mask_src).convert("RGB"))
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LANCZOS4)
        subj = subject_alpha(m)
    else:
        subj = subject_alpha(port)

    if args.scale != 1.0:
        # Inset the subject so the near planes have margin to work in. The landmarks are
        # moved by the same transform rather than re-detected: re-running the mesh on a
        # resampled image would shift the eye guard by a pixel or two for no gain.
        sw, sh = int(W * args.scale), int(H * args.scale)
        small = cv2.resize(port, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
        small_a = cv2.resize(subj, (sw, sh), interpolation=cv2.INTER_LINEAR)
        ox = (W - sw) // 2
        oy = H - sh                        # sit her on the bottom edge: only the SIDES inset,
                                           # so her body runs off-frame as it did before
        port = np.zeros_like(port)
        subj = np.zeros((H, W), np.float32)
        # Her silhouette was clipped by the original crop, so insetting turns those clips
        # into straight cut lines floating in the collage. Fade the alpha at the inset
        # border; interior edges are untouched because the mask is already 0 there.
        fade = max(6, int(min(sw, sh) * 0.03))
        ramp = np.ones((sh, sw), np.float32)
        g = np.linspace(0, 1, fade, dtype=np.float32)
        ramp[:, :fade] *= g; ramp[:, -fade:] *= g[::-1]
        ramp[:fade, :] *= g[:, None]
        small_a = small_a * ramp
        port[oy:oy + sh, ox:ox + sw] = small
        subj[oy:oy + sh, ox:ox + sw] = small_a
        if pts is not None:
            pts = pts * args.scale + np.array([ox, oy], np.float32)
        print(f"  portrait inset to {args.scale:.2f} "
              f"({ox}px clear each side)")
    hole = eye_hole((H, W), pts)
    relief = face_relief((H, W), pts, args.front_relief)
    # Thin the near planes across the subject too, so decorations gather at the sides
    # rather than sitting on her body. Blurred so shapes still graze hair and shoulders.
    body = cv2.GaussianBlur(subj, (0, 0), max(W, H) * 0.02)
    front = hole * relief * (1.0 - args.subject_relief * np.clip(body, 0, 1))

    def plate(path, plane):
        im = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS))
        a = key_cream(im)
        k = BLUR[plane] | 1
        im = cv2.GaussianBlur(im.astype(np.float32), (k * 2 + 1, k * 2 + 1), k / 2)
        a = cv2.GaussianBlur(a, (k * 2 + 1, k * 2 + 1), k / 2)
        return im, a

    p1, p2, p4, p5 = args.plates
    canvas = np.full((H, W, 3), 238, np.float32)          # paper ground

    im, a = plate(p5, 5); canvas = over(canvas, im, a)     # far
    im, a = plate(p4, 4); canvas = over(canvas, im, a)
    canvas = over(canvas, port.astype(np.float32), subj)   # plane 3 — sharp portrait
    im, a = plate(p2, 2); canvas = over(canvas, im, a * front)  # near, eyes protected
    im, a = plate(p1, 1); canvas = over(canvas, im, a * front)  # nearest

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(args.out, quality=95)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
