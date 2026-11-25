#!/home/rong/openclaw-venv/bin/python3
"""Red/blue dramatic paint — local reconstruction of a graphic-novel poster look.

Reverse-engineered from shared/refs/'red blue dramatic paint.jpeg':
  1. Crushed blacks    — everything under ~20% luminance goes to pure black.
  2. Blue gradient map — luminance -> black / navy / cobalt / pale cyan.
  3. Red rim light     — a band just inside the subject's outline, on the sides
                         facing the two "lights", broken up by dry-brush noise.
  4. Dry-brush grunge  — streaky black knock-outs in the shadows, red + blue
                         spatter in the background around the figure.
  5. Smoky background  — the original BG crushed to near-black with a faint
                         low-frequency blue haze.
Pure local (MediaPipe mask), ~$0.

  red-blue-paint.py --source photo.jpg [--rim 0.8 --grunge 0.6 --crush 0.2]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path.home() / ".openclaw/workspace/shared/finals"

BLUE_STOPS = [(0.00, (0, 0, 0)), (0.30, (8, 30, 66)), (0.62, (28, 104, 168)),
              (0.88, (88, 182, 226)), (1.00, (170, 225, 245))]
RED = np.array([235, 28, 44]) / 255.0


def gradient_map(t, stops):
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], float) / 255.0
    return np.stack([np.interp(t, xs, cols[:, c]) for c in range(3)], -1)


def streak_noise(shape, s, rng, angle_len=(0.004, 0.05)):
    """Dry-brush noise: white noise smeared along one axis, then normalised."""
    n = rng.random(shape)
    n = ndi.gaussian_filter(n, (angle_len[1] * s, angle_len[0] * s))
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    return n


def subject_mask(src, img):
    import masking
    m, info = masking.build_mask(src, "face-skin,body-skin,hair,clothes,others",
                                 feather=0.3)
    return np.asarray(m, float) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--crush", type=float, default=0.20, help="black point (0-0.5)")
    p.add_argument("--gamma", type=float, default=1.25)
    p.add_argument("--rim", type=float, default=1.0, help="red rim strength")
    p.add_argument("--rim-width", type=float, default=0.02, help="fraction of short edge")
    p.add_argument("--rim-dirs", default="-1,-0.2;1,0.4", help="light dirs x,y;x,y (screen space)")
    p.add_argument("--grunge", type=float, default=0.6)
    p.add_argument("--keep-reds", type=float, default=1.0, help="keep red clothing red (0-1)")
    p.add_argument("--levels", type=int, default=5, help="posterize levels, 0=off")
    p.add_argument("--ink", type=float, default=1.0, help="ink line strength")
    p.add_argument("--bg-keep", type=float, default=0.18, help="how much original BG survives")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    img = Image.open(a.source).convert("RGB")
    rgb = np.asarray(img, float) / 255.0
    H, W = rgb.shape[:2]
    s = min(H, W)
    m = subject_mask(a.source, img)

    # 1-2. tone: crush + gamma, BG suppressed, then blue map
    L = rgb @ [0.299, 0.587, 0.114]
    import cv2
    Lu = (L * 255).astype(np.uint8)
    for _ in range(3):                                      # paint-flat skin, keep edges
        Lu = cv2.bilateralFilter(Lu, 0, 18, max(3, 0.004 * s))
    L = Lu / 255.0
    t = np.clip((L - a.crush) / (0.92 - a.crush), 0, 1) ** a.gamma
    if a.levels > 0:                                        # soft posterize
        q = t * a.levels
        f = q - np.floor(q)
        t = (np.floor(q) + np.clip((f - 0.5) * 3 + 0.5, 0, 1)) / a.levels
    # ink: dark DoG lines on strong edges (graphic-novel linework)
    dog = ndi.gaussian_filter(L, 0.0012 * s) - ndi.gaussian_filter(L, 0.004 * s)
    ink = np.clip(-dog * a.ink * 25, 0, 1)
    t = t * (1 - ink)
    haze = ndi.gaussian_filter(rng.random((H, W)), 0.06 * s)
    haze = np.clip((haze - haze.mean()) / (haze.std() + 1e-6) * 0.5 + 0.5, 0, 1)
    t_bg = t * a.bg_keep + haze * 0.22
    t = t * m + t_bg * (1 - m)
    out = gradient_map(t, BLUE_STOPS)

    # 3. red rim: pixels inside her where a step toward the light leaves the
    #    silhouette (classic shifted-mask rim) -> one-sided, follows the form.
    mb = ndi.gaussian_filter(m, 0.004 * s)
    inside = (mb > 0.5).astype(float)
    rimraw = np.zeros_like(m)
    for d in a.rim_dirs.split(";"):
        dx, dy = map(float, d.split(","))
        k = np.hypot(dx, dy)
        sh = ndi.shift(inside, (-dy / k * a.rim_width * s, -dx / k * a.rim_width * s), order=0, mode="nearest")
        rimraw = np.maximum(rimraw, inside * (1 - sh))
    rimraw = ndi.gaussian_filter(rimraw, 0.003 * s) * inside
    brush = streak_noise((H, W), s, rng, (0.003, 0.03))
    rim = rimraw * np.clip((brush - 0.3) * 3.5, 0, 1) * a.rim
    rim = np.clip(rim * 1.5, 0, 1) * np.clip(0.4 + t * 1.2, 0, 1)
    out = out * (1 - rim[..., None]) + RED * rim[..., None]
    # keep strongly red garments/props red (the ref's only accent colour)
    if a.keep_reds > 0:
        import colorsys
        r, g, bl = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx, mn = rgb.max(-1), rgb.min(-1)
        sat = (mx - mn) / (mx + 1e-6)
        redness = np.clip((r - np.maximum(g, bl)) / (mx + 1e-6), 0, 1) * np.clip((sat - 0.35) * 3, 0, 1)
        redness = ndi.gaussian_filter(redness, 0.002 * s) * a.keep_reds
        red_tone = RED[None, None, :] * np.clip(t * 1.4 + 0.05, 0, 1)[..., None]
        out = out * (1 - redness[..., None]) + red_tone * redness[..., None]

    # 4. grunge: black dry-brush knockouts in shadows + spatter around figure
    if a.grunge > 0:
        kn = streak_noise((H, W), s, rng, (0.0015, 0.02))
        kn = ndi.rotate(kn, -20, reshape=False, mode="reflect")
        knock = (kn < 0.18 + 0.15 * a.grunge) & (t < 0.3) & (mb > 0.5)
        out[knock] *= 0.1
        # BG dry-brush: rotated streak noise, faint blue strokes
        st = streak_noise((H, W), s, rng, (0.003, 0.06))
        st = ndi.rotate(st, 35, reshape=False, mode="reflect")
        near = ndi.gaussian_filter(mb, 0.08 * s) * (1 - mb)
        stroke = np.clip((st - 0.6) * 4, 0, 1) * np.clip(near * 3, 0, 1) * a.grunge
        out += stroke[..., None] * (gradient_map(np.full((H, W), 0.55), BLUE_STOPS) - out) * 0.8
        # red spatter clustered where the rim is strong
        rimzone = ndi.gaussian_filter(rim, 0.03 * s) * (1 - mb)
        sp = ndi.gaussian_filter(rng.random((H, W)), 0.0018 * s)
        sp = (sp - sp.mean()) / sp.std()
        red_sp = (sp > 2.2) & (rimzone > 0.02 / max(a.grunge, 0.1))
        out[red_sp] = RED * 0.8

    # 5. finish: grain + soft vignette
    yy, xx = np.mgrid[0:H, 0:W]
    v = 1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2
    out *= np.clip(v, 0.2, 1)[..., None]
    out += (rng.normal(0, 0.025, (H, W)))[..., None]
    out = np.clip(out, 0, 1)

    dst = Path(a.out) if a.out else OUT / f"{Path(a.source).stem}__redblue_{time.strftime('%H%M%S')}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out * 255).astype(np.uint8)).save(dst, quality=93)
    print(dst)


if __name__ == "__main__":
    main()
