#!/home/rong/openclaw-venv/bin/python3
"""
botanical-overlay.py — Scatter small procedural botanical elements (flowers,
petals, leaves) onto the subject's skin. Zero-cost, pure local (MediaPipe +
PIL drawing). Delicate, romantic-realist — not baroque.

Usage:
    ./botanical-overlay.py --source photo.jpg --preset spine-flowers
    ./botanical-overlay.py --source photo.jpg --preset torso-petals --color pink
    ./botanical-overlay.py --list-presets
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from masking import build_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Color palettes (RGB tuples)
# ---------------------------------------------------------------------------
COLORS = {
    "white":  [(252, 250, 245), (245, 240, 230), (255, 253, 248)],
    "cream":  [(245, 232, 210), (238, 220, 195), (252, 240, 220)],
    "pink":   [(245, 200, 210), (235, 175, 195), (250, 215, 225)],
    "green":  [(170, 195, 155), (150, 180, 140), (190, 210, 170)],
}
CENTER_COLOR = (220, 190, 90)  # soft yellow flower center
LEAF_VEIN = (110, 135, 95)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "spine-flowers": {
        "element": "flower",
        "color": "white",
        "n": 13,
        "size_range": (0.018, 0.032),  # % of short edge
        "placement": "spine",
        "jitter": 0.015,  # lateral jitter as % of short edge
    },
    "torso-petals": {
        "element": "petal",
        "color": "cream",
        "n": 36,
        "size_range": (0.015, 0.028),
        "placement": "torso",
        "jitter": 0.0,
    },
    "covering-leaves": {
        "element": "leaf",
        "color": "green",
        "n": 4,
        "size_range": (0.09, 0.14),
        "placement": "covering",
        "jitter": 0.0,
    },
    "shoulder-trail": {
        "element": "flower",
        "color": "white",
        "n": 18,
        "size_range": (0.014, 0.032),
        "placement": "diagonal",
        "jitter": 0.02,
    },
    "constellation": {
        "element": "flower",
        "color": "white",
        "n": 45,
        "size_range": (0.008, 0.016),
        "placement": "scatter",
        "jitter": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# ---------------------------------------------------------------------------
# Pose detection (spine axis)
# ---------------------------------------------------------------------------
POSE_MODEL = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")


def detect_spine(img_pil):
    """Return ((shoulders_mid_xy), (hips_mid_xy)) in pixels or None."""
    try:
        import mediapipe as mp
        arr = np.array(img_pil)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr.copy())
        base = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
        opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
        det = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        res = det.detect(mp_img)
        det.close()
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        w, h = img_pil.size
        ls, rs = lms[11], lms[12]
        lh, rh = lms[23], lms[24]
        sh_mid = ((ls.x + rs.x) / 2 * w, (ls.y + rs.y) / 2 * h)
        hp_mid = ((lh.x + rh.x) / 2 * w, (lh.y + rh.y) / 2 * h)
        return sh_mid, hp_mid
    except Exception as e:
        log("WARN", f"Pose detection failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Procedural sprite drawing
# ---------------------------------------------------------------------------
def _drop_shadow(sprite, offset_px=3, darkness=0.35, blur=2.0):
    """Return a new RGBA image = shadow layer + sprite, same size as sprite upsized."""
    w, h = sprite.size
    pad = int(offset_px + blur * 2 + 2)
    canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    # Shadow = alpha of sprite, dark, blurred, offset
    alpha = sprite.split()[-1]
    shadow = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda v: int(v * darkness)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    canvas.paste(shadow, (pad + offset_px, pad + offset_px), shadow)
    canvas.paste(sprite, (pad, pad), sprite)
    return canvas


def draw_flower(size_px, color_rgb, rng):
    """5-petal flower sprite as RGBA."""
    s = max(8, int(size_px))
    pad = s * 2
    img = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = pad // 2, pad // 2
    petal_r = s * 0.45
    petal_dist = s * 0.35
    n_petals = 5
    rot0 = rng.uniform(0, math.pi * 2)
    # slight tonal variation per petal
    for i in range(n_petals):
        a = rot0 + i * (2 * math.pi / n_petals)
        px = cx + math.cos(a) * petal_dist
        py = cy + math.sin(a) * petal_dist
        # vary petal color slightly
        dr = rng.randint(-8, 8)
        c = tuple(max(0, min(255, v + dr)) for v in color_rgb) + (240,)
        bbox = (px - petal_r, py - petal_r * 0.8,
                px + petal_r, py + petal_r * 0.8)
        d.ellipse(bbox, fill=c)
    # center
    cr = s * 0.12
    d.ellipse((cx - cr, cy - cr, cx + cr, cy + cr),
              fill=CENTER_COLOR + (255,))
    # tiny soft blur to lose pixel edges
    img = img.filter(ImageFilter.GaussianBlur(max(0.5, s * 0.02)))
    return _drop_shadow(img, offset_px=max(2, s // 12),
                        darkness=0.30, blur=max(1.5, s * 0.05))


def draw_petal(size_px, color_rgb, rng):
    """Single teardrop / elongated oval petal."""
    s = max(8, int(size_px))
    pad = s * 2
    img = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = pad // 2, pad // 2
    # elongated oval — width < height
    rw, rh = s * 0.35, s * 0.7
    c = tuple(color_rgb) + (235,)
    d.ellipse((cx - rw, cy - rh, cx + rw, cy + rh), fill=c)
    # Subtle inner highlight
    hw, hh = rw * 0.5, rh * 0.6
    hc = tuple(min(255, v + 15) for v in color_rgb) + (120,)
    d.ellipse((cx - hw, cy - hh * 1.1, cx + hw, cy + hh * 0.4), fill=hc)
    img = img.filter(ImageFilter.GaussianBlur(max(0.5, s * 0.02)))
    # random rotation
    img = img.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=False)
    return _drop_shadow(img, offset_px=max(2, s // 14),
                        darkness=0.25, blur=max(1.2, s * 0.04))


def draw_leaf(size_px, color_rgb, rng):
    """Pointed ellipse with center vein."""
    s = max(16, int(size_px))
    pad = s * 2
    img = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = pad // 2, pad // 2
    rw, rh = s * 0.32, s * 0.75
    c = tuple(color_rgb) + (225,)
    # draw leaf as ellipse then mask top/bottom to points via polygon overlay
    d.ellipse((cx - rw, cy - rh, cx + rw, cy + rh), fill=c)
    # taper tips by drawing triangles at top and bottom in transparent (skip — ellipse reads as leaf shape fine)
    # center vein
    vein_c = LEAF_VEIN + (200,)
    d.line([(cx, cy - rh * 0.85), (cx, cy + rh * 0.85)],
           fill=vein_c, width=max(1, s // 40))
    # side veins
    n_veins = 4
    for i in range(1, n_veins + 1):
        t = i / (n_veins + 1)
        y = cy - rh * 0.7 + t * rh * 1.4
        vw = rw * (1 - abs(t - 0.5) * 1.4) * 0.85
        d.line([(cx, y), (cx - vw, y + rh * 0.08)],
               fill=vein_c, width=max(1, s // 60))
        d.line([(cx, y), (cx + vw, y + rh * 0.08)],
               fill=vein_c, width=max(1, s // 60))
    img = img.filter(ImageFilter.GaussianBlur(max(0.5, s * 0.015)))
    img = img.rotate(rng.uniform(-30, 30), resample=Image.BICUBIC, expand=False)
    return _drop_shadow(img, offset_px=max(3, s // 18),
                        darkness=0.35, blur=max(2.0, s * 0.05))


SPRITE_DRAW = {"flower": draw_flower, "petal": draw_petal, "leaf": draw_leaf}


# ---------------------------------------------------------------------------
# Placement generators
# ---------------------------------------------------------------------------
def _sample_points_on_mask(mask_arr, n, rng, max_tries=200):
    """Sample n random points where mask_arr > 127."""
    h, w = mask_arr.shape
    ys, xs = np.where(mask_arr > 127)
    if len(xs) == 0:
        return []
    idxs = rng.sample(range(len(xs)), min(n, len(xs)))
    return [(int(xs[i]), int(ys[i])) for i in idxs]


def placement_spine(mask_arr, spine, n, jitter_px, rng):
    """Points along spine line (shoulders→hips extended), jittered laterally."""
    h, w = mask_arr.shape
    if spine is None:
        # fallback: vertical axis through mask centroid, top→bottom
        ys, xs = np.where(mask_arr > 127)
        if len(xs) == 0:
            return []
        cx = int(xs.mean())
        y_top = int(np.percentile(ys, 10))
        y_bot = int(np.percentile(ys, 85))
        sh_mid = (cx, y_top)
        hp_mid = (cx, y_bot)
    else:
        sh_mid, hp_mid = spine

    # Extend line a bit past hips for tailing flowers
    dx = hp_mid[0] - sh_mid[0]
    dy = hp_mid[1] - sh_mid[1]
    points = []
    # Perpendicular unit for jitter
    seg_len = max(1.0, math.hypot(dx, dy))
    px_u, py_u = -dy / seg_len, dx / seg_len
    # Distribute across t in [0.0, 1.15]
    for i in range(n):
        t = (i / max(1, n - 1)) * 1.15
        x = sh_mid[0] + dx * t
        y = sh_mid[1] + dy * t
        # lateral jitter
        j = rng.uniform(-jitter_px, jitter_px)
        x += px_u * j
        y += py_u * j
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h and mask_arr[yi, xi] > 127:
            points.append((xi, yi))
        else:
            # nudge: search in expanding radius for nearest mask pixel
            found = False
            radius_cap = int(max(w, h) * 0.08)
            for r in range(10, radius_cap, 10):
                for _ in range(12):
                    xi2 = xi + rng.randint(-r, r)
                    yi2 = yi + rng.randint(-r, r)
                    if 0 <= xi2 < w and 0 <= yi2 < h and mask_arr[yi2, xi2] > 127:
                        points.append((xi2, yi2))
                        found = True
                        break
                if found:
                    break
    return points


def placement_torso(mask_arr, spine, n, rng):
    """Scatter across torso region (between shoulders and hips band)."""
    h, w = mask_arr.shape
    if spine is not None:
        sh_mid, hp_mid = spine
        y_min = int(min(sh_mid[1], hp_mid[1]) - 0.05 * h)
        y_max = int(max(sh_mid[1], hp_mid[1]) + 0.05 * h)
    else:
        ys, xs = np.where(mask_arr > 127)
        y_min = int(np.percentile(ys, 15))
        y_max = int(np.percentile(ys, 70))
    band = mask_arr.copy()
    band[:max(0, y_min), :] = 0
    band[min(h, y_max):, :] = 0
    return _sample_points_on_mask(band, n, rng)


def placement_covering(mask_arr, spine, n, rng):
    """Large leaves at chest + hip region — strategic modesty placement."""
    h, w = mask_arr.shape
    points = []
    if spine is not None:
        sh_mid, hp_mid = spine
        # chest (just below shoulders)
        t = 0.25
        chest = (int(sh_mid[0] + (hp_mid[0] - sh_mid[0]) * t),
                 int(sh_mid[1] + (hp_mid[1] - sh_mid[1]) * t))
        # hip (slightly above hip mid)
        hip = (int(hp_mid[0]), int(hp_mid[1]))
        # If n==3: chest + hip + one extra; if n==4: add shoulder
        candidates = [chest, hip]
        if n >= 3:
            # second chest leaf offset
            candidates.append((chest[0] + int(w * 0.04), chest[1] + int(h * 0.03)))
        if n >= 4:
            candidates.append((int(sh_mid[0] - w * 0.05), int(sh_mid[1] + h * 0.02)))
        if n >= 5:
            candidates.append((hip[0] - int(w * 0.05), hip[1] - int(h * 0.02)))
        for cx, cy in candidates[:n]:
            if 0 <= cx < w and 0 <= cy < h and mask_arr[cy, cx] > 127:
                points.append((cx, cy))
            else:
                # snap to nearest mask
                pts = _sample_points_on_mask(mask_arr, 1, rng)
                if pts:
                    points.append(pts[0])
    else:
        points = _sample_points_on_mask(mask_arr, n, rng)
    return points


def placement_diagonal(mask_arr, spine, n, jitter_px, rng):
    """Shoulder → opposite hip diagonal trail."""
    h, w = mask_arr.shape
    if spine is not None:
        sh_mid, hp_mid = spine
        # Pick one shoulder side and opposite hip side: use axis-aware offsets
        start = (sh_mid[0] - w * 0.06, sh_mid[1])  # left shoulder-ish
        end = (hp_mid[0] + w * 0.05, hp_mid[1] + h * 0.05)  # right hip
    else:
        ys, xs = np.where(mask_arr > 127)
        start = (int(np.percentile(xs, 20)), int(np.percentile(ys, 15)))
        end = (int(np.percentile(xs, 80)), int(np.percentile(ys, 75)))
    points = []
    for i in range(n):
        t = i / max(1, n - 1)
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        x += rng.uniform(-jitter_px, jitter_px)
        y += rng.uniform(-jitter_px, jitter_px)
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h and mask_arr[yi, xi] > 127:
            points.append((xi, yi))
    return points


def placement_scatter(mask_arr, n, rng):
    return _sample_points_on_mask(mask_arr, n, rng)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def composite_elements(base_rgb, mask_arr, preset, element, color, density,
                       size_mult, seed):
    rng = random.Random(seed)
    w, h = base_rgb.size
    short_edge = min(w, h)
    n = max(1, int(preset["n"] * density))
    s_min = preset["size_range"][0] * short_edge * size_mult
    s_max = preset["size_range"][1] * short_edge * size_mult
    jitter_px = preset["jitter"] * short_edge

    # Spine detection (needed for most placements)
    spine = detect_spine(base_rgb)
    if spine is None:
        log("WARN", "No pose — using mask-based fallback")

    # Generate placement points
    pm = preset["placement"]
    if pm == "spine":
        pts = placement_spine(mask_arr, spine, n, jitter_px, rng)
    elif pm == "torso":
        pts = placement_torso(mask_arr, spine, n, rng)
    elif pm == "covering":
        pts = placement_covering(mask_arr, spine, n, rng)
    elif pm == "diagonal":
        pts = placement_diagonal(mask_arr, spine, n, jitter_px, rng)
    elif pm == "scatter":
        pts = placement_scatter(mask_arr, n, rng)
    else:
        pts = placement_scatter(mask_arr, n, rng)

    log("INFO", f"Placed {len(pts)} / {n} target elements ({pm})")

    # Sort back-to-front by y (upper z first, then paste lower on top)
    pts.sort(key=lambda p: p[1])

    color_pool = COLORS[color]
    out = base_rgb.convert("RGBA")

    for (x, y) in pts:
        size_px = rng.uniform(s_min, s_max)
        c = rng.choice(color_pool)
        sprite = SPRITE_DRAW[element](size_px, c, rng)
        sw, sh = sprite.size
        px = x - sw // 2
        py = y - sh // 2
        out.alpha_composite(sprite, (px, py))

    return out.convert("RGB")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(args):
    log("INFO", f"Loading {args.source}")
    img = Image.open(args.source).convert("RGB")
    w, h = img.size
    log("INFO", f"Image size: {w}x{h}")

    preset = PRESETS[args.preset]
    element = args.element or preset["element"]
    color = args.color or preset["color"]
    seed = args.seed if args.seed is not None else random.randint(0, 999999)

    log("INFO", f"Preset={args.preset} element={element} color={color} seed={seed}")

    # Skin mask
    log("INFO", "Building skin mask (MediaPipe)...")
    mask_pil = None
    try:
        mask_pil, info = build_mask(img, affect="skin", exclude="hands",
                                    feather=0.3, cleanup="smooth")
        cov = info.get("coverage_pct", 0)
        log("INFO", f"Skin mask coverage: {cov}%")
        # If skin is too small (back-facing, heavily clothed/bound), widen
        if float(cov) < 6.0:
            log("WARN", f"Skin coverage low ({cov}%) — widening to subject mask")
            try:
                mask_pil, info2 = build_mask(img, affect="subject")
                log("INFO", f"Subject mask coverage: {info2.get('coverage_pct', '?')}%")
            except Exception as e:
                log("WARN", f"Subject fallback failed: {e}")
    except Exception as e:
        log("ERROR", f"Mask failed: {e} — falling back to subject")
        try:
            mask_pil, _ = build_mask(img, affect="subject")
        except Exception:
            mask_pil = Image.new("L", img.size, 255)

    mask_arr = np.array(mask_pil)

    # Compose
    log("INFO", "Compositing botanical elements...")
    result = composite_elements(img, mask_arr, preset, element, color,
                                args.density, args.size, seed)

    # Output path
    src_name = os.path.splitext(os.path.basename(args.source))[0]
    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.source))))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{model_name}_{src_name}_{ts}_botanical_{args.preset}_{seed % 100:02d}"
    out_dir = os.path.join(args.local_output_dir, folder)
    os.makedirs(out_dir, exist_ok=True)

    out_name = f"{src_name}_botanical_{args.preset}.jpg"
    out_path = os.path.join(out_dir, out_name)
    result.save(out_path, quality=95)
    log("INFO", f"Saved: {out_path}")

    # Copy to finals
    try:
        finals_dir = os.path.join(args.local_output_dir, "finals")
        os.makedirs(finals_dir, exist_ok=True)
        finals_path = os.path.join(finals_dir, f"{src_name}_botanical_{args.preset}_{seed % 100:02d}.jpg")
        result.save(finals_path, quality=95)
        log("INFO", f"Finals: {finals_path}")
    except Exception as e:
        log("WARN", f"Finals copy failed: {e}")
        finals_path = out_path

    # Push
    try:
        from notify import push_image
        push_image(finals_path,
                   title=f"Botanical — {src_name}",
                   body=f"{args.preset} / {element} / {color}")
        log("INFO", "Pushed to phone")
    except Exception as e:
        log("WARN", f"Push failed: {e}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Botanical overlay tool")
    p.add_argument("--source", help="Source photo path")
    p.add_argument("--preset", default="spine-flowers",
                   choices=list(PRESETS.keys()))
    p.add_argument("--element", choices=["flower", "petal", "leaf"],
                   help="Override preset element kind")
    p.add_argument("--color", choices=list(COLORS.keys()),
                   help="Override preset color")
    p.add_argument("--density", type=float, default=1.0,
                   help="Multiplier for element count (0-2)")
    p.add_argument("--size", type=float, default=1.0,
                   help="Multiplier for element size")
    p.add_argument("--seed", type=int)
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--output-to", default="local")
    p.add_argument("--local-output-dir",
                   default=os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"))
    args = p.parse_args()

    if args.list_presets:
        print("\nBotanical overlay presets:\n")
        for name, cfg in PRESETS.items():
            print(f"  {name:20s}  element={cfg['element']:8s} color={cfg['color']:6s} n={cfg['n']:3d}  placement={cfg['placement']}")
        return

    if not args.source:
        p.error("--source is required")

    run(args)


if __name__ == "__main__":
    main()
