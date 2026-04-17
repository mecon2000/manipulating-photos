#!/home/rong/openclaw-venv/bin/python3
"""silhouette-backdrop — Reduce subject to graphic silhouette on a clean backdrop."""

import os
import sys
import re
import shutil
import random
import argparse
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

sys.stdout.reconfigure(line_buffering=True)


PRESETS = {
    "moon": {
        "backdrop": (14, 20, 38),
        "element": "moon",
        "element_color": (235, 230, 215),
        "pos": (0.5, 0.42),
        "size": 0.55,
        "desc": "Large pale moon on deep night-blue",
    },
    "spotlight": {
        "backdrop": (210, 208, 202),
        "element": "halo",
        "element_color": (255, 252, 240),
        "pos": (0.5, 0.55),
        "size": 0.85,
        "desc": "Radial halo on warm light-grey",
    },
    "pedestal": {
        "backdrop": (232, 222, 205),
        "element": "pedestal",
        "element_color": (150, 142, 130),
        "pos": (0.5, 0.88),
        "size": 0.55,
        "desc": "Grey pedestal rectangle on cream",
    },
    "sunset": {
        "backdrop": "gradient:(245,120,60)->(120,25,45)",
        "element": "circle",
        "element_color": (255, 220, 150),
        "pos": (0.5, 0.62),
        "size": 0.75,
        "desc": "Large warm sun on orange-to-red gradient",
    },
    "triangle": {
        "backdrop": (238, 232, 222),
        "element": "triangle",
        "element_color": (190, 60, 55),
        "pos": (0.5, 0.58),
        "size": 0.9,
        "desc": "Bold red triangle on off-white",
    },
    "red-wall": {
        "backdrop": (140, 30, 32),
        "element": "none",
        "element_color": (0, 0, 0),
        "pos": (0.5, 0.5),
        "size": 0.0,
        "desc": "Pure silhouette on deep red",
    },
    "arch": {
        "backdrop": (225, 215, 200),
        "element": "arch",
        "element_color": (90, 80, 75),
        "pos": (0.5, 0.58),
        "size": 0.75,
        "desc": "Dark arch/doorway on warm cream",
    },
}


def check_suitability(source, img, output_dir, min_extension, max_clothing):
    """Return (ok, reason). Clothing check only — loose clothes make bad silhouettes."""
    try:
        clothes_pil, _ = build_mask(img, affect="clothes", output_dir=None, feather=0.0)
        skin_pil, _ = build_mask(img, affect="skin", output_dir=None, feather=0.0)
        clothes_px = int((np.array(clothes_pil) > 127).sum())
        skin_px = int((np.array(skin_pil) > 127).sum())
        total = clothes_px + skin_px
        if total > 0:
            fraction = clothes_px / total
            log(output_dir, f"Clothes/(clothes+skin) ratio: {fraction:.2%}")
            if fraction > max_clothing:
                return False, f"too much clothing ({fraction:.0%}, max {max_clothing:.0%}) — silhouette needs nude or tight"
    except Exception as e:
        log(output_dir, f"clothing check failed: {e} — skipping clothing gate", "WARN")
    return True, "ok"


def log(output_dir, msg, level="INFO"):
    t = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] [{level}] {msg}"
    print(line)
    if output_dir:
        with open(os.path.join(output_dir, "workflow.log"), "a") as f:
            f.write(line + "\n")


def parse_rgb(s):
    if s is None:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B got {s}")
    return tuple(parts)


def make_backdrop(w, h, spec):
    if isinstance(spec, tuple):
        img = Image.new("RGB", (w, h), spec)
        return img
    if isinstance(spec, str) and spec.startswith("gradient:"):
        m = re.match(r"gradient:\((\d+),(\d+),(\d+)\)->\((\d+),(\d+),(\d+)\)", spec)
        c1 = tuple(int(m.group(i)) for i in (1, 2, 3))
        c2 = tuple(int(m.group(i)) for i in (4, 5, 6))
        arr = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(3):
            col = np.linspace(c1[i], c2[i], h).astype(np.float32)
            arr[:, :, i] = col[:, None]
        return Image.fromarray(arr.astype(np.uint8), "RGB")
    return Image.new("RGB", (w, h), (200, 200, 200))


def draw_moon(canvas, color, pos, size, w, h, seed=0):
    """Textured moon: base disc + crater noise + terminator shading + outer glow."""
    cx, cy = int(pos[0] * w), int(pos[1] * h)
    short = min(w, h)
    r = int(size * short) // 2
    # Outer glow layer
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(30, 0, -1):
        a = int(70 * (i / 30) ** 2)
        rr = int(r * (1 + i * 0.05))
        gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr),
                   fill=color + (a,))
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.12))
    canvas.paste(glow, (0, 0), glow)
    # Base moon disc with radial shading (darker on one side → 3D terminator)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / max(r, 1)
    dy = (yy - cy) / max(r, 1)
    dist = np.sqrt(dx * dx + dy * dy)
    inside = dist <= 1.0
    # Terminator: cosine shading with light from upper-left
    light = np.clip(0.65 + 0.45 * (-dx * 0.6 + -dy * 0.6), 0.35, 1.10)
    # Crater noise — blurred random field
    from scipy.ndimage import gaussian_filter
    noise = rng.random((h, w)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=max(3, r * 0.015))
    noise -= noise.mean()
    # stronger crater noise at larger scale too
    noise2 = rng.random((h, w)).astype(np.float32)
    noise2 = gaussian_filter(noise2, sigma=max(6, r * 0.05))
    noise2 -= noise2.mean()
    tex = 1.0 + noise * 2.5 + noise2 * 1.5
    shading = light * tex
    disc_rgb = np.array(color, dtype=np.float32)[None, None, :] * shading[:, :, None]
    disc_rgb = np.clip(disc_rgb, 0, 255)
    disc_rgba = np.concatenate([disc_rgb, np.where(inside, 255, 0)[..., None]], axis=-1).astype(np.uint8)
    moon_layer = Image.fromarray(disc_rgba, "RGBA")
    # soft edge so it isn't pixelated
    edge = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(edge).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    edge = edge.filter(ImageFilter.GaussianBlur(max(1.5, r * 0.008)))
    moon_layer.putalpha(edge)
    canvas.paste(moon_layer, (0, 0), moon_layer)
    return canvas


def draw_element(canvas, element, color, pos, size, w, h, seed=0):
    if element == "none":
        return canvas
    if element == "moon":
        return draw_moon(canvas, color, pos, size, w, h, seed)
    draw = ImageDraw.Draw(canvas, "RGBA")
    cx, cy = int(pos[0] * w), int(pos[1] * h)
    short = min(w, h)
    s = int(size * short)

    if element == "circle":
        r = s // 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif element == "halo":
        # radial gradient from element_color (center) to backdrop (transparent outer)
        r = s // 2
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        steps = 40
        for i in range(steps, 0, -1):
            a = int(255 * (i / steps) ** 2 * 0.5)
            rr = int(r * (i / steps))
            ld.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color + (a,))
        layer = layer.filter(ImageFilter.GaussianBlur(r * 0.08))
        canvas.paste(layer, (0, 0), layer)
    elif element == "pedestal":
        pw = int(s * 1.2)
        ph = int(s * 0.35)
        x0 = cx - pw // 2
        y0 = cy - ph // 2
        draw.rectangle((x0, y0, x0 + pw, h), fill=color)
    elif element == "triangle":
        half = s // 2
        apex = (cx, cy - int(s * 0.55))
        left = (cx - half, cy + int(s * 0.35))
        right = (cx + half, cy + int(s * 0.35))
        draw.polygon([apex, left, right], fill=color)
    elif element == "arch":
        aw = int(s * 0.9)
        ah = int(s * 1.25)
        x0 = cx - aw // 2
        y0 = cy - ah // 2
        draw.rectangle((x0, y0 + aw // 2, x0 + aw, h), fill=color)
        draw.ellipse((x0, y0, x0 + aw, y0 + aw), fill=color)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source")
    p.add_argument("--preset", default="moon")
    p.add_argument("--backdrop-color")
    p.add_argument("--silhouette-color", default="0,0,0")
    p.add_argument("--element-color")
    p.add_argument("--detail", type=float, default=0.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Skip pose+clothing suitability check")
    p.add_argument("--min-extension", type=float, default=1.6,
                   help="Min arm-span/shoulder-width ratio OR leg-span/hip-width for pose to count as stretched (default 1.6)")
    p.add_argument("--max-clothing", type=float, default=0.25,
                   help="Max clothes fraction of subject before skipping (default 0.25)")
    p.add_argument("--output-to", default="local", choices=["local", "gdrive", "both"])
    p.add_argument("--local-output-dir", default="~/.openclaw/workspace/shared/tool-outputs-intermediates")
    args = p.parse_args()

    if args.list_presets:
        for k, v in PRESETS.items():
            print(f"  {k:12s} — {v['desc']}")
        return

    if not args.source:
        p.error("--source required")

    preset = PRESETS.get(args.preset)
    if not preset:
        p.error(f"Unknown preset {args.preset}. Available: {list(PRESETS)}")

    source = os.path.expanduser(args.source)
    img = Image.open(source).convert("RGB")
    w, h = img.size

    src_base = os.path.splitext(os.path.basename(source))[0]
    parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, pp in enumerate(parts):
        if pp == "_photos" and i + 1 < len(parts):
            model_name = parts[i + 1]
            break

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)
    ts = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    folder = f"{model_name}_{src_base}_{ts}_silhouette_{args.preset}_{seed % 100:02d}"
    folder = re.sub(r'[<>:"/\\|?*]', "_", folder)
    output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder)
    os.makedirs(output_dir, exist_ok=True)

    log(output_dir, f"Source: {source}")
    log(output_dir, f"Preset: {args.preset} ({preset['desc']})")
    log(output_dir, f"Size: {w}x{h}  Seed: {seed}")

    log(output_dir, "Extracting subject mask via BiRefNet...")
    mask_pil, mask_info = build_mask(source, affect="subject", output_dir=output_dir, feather=0.0)
    mask = mask_pil.resize((w, h), Image.LANCZOS)
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    # Hard-binary at 0.5, then tiny 1px feather for anti-aliased edge only
    mask_arr = (mask_arr > 0.5).astype(np.float32)
    mask_arr = np.array(Image.fromarray((mask_arr * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(0.8, min(w, h) * 0.0008)))).astype(np.float32) / 255.0
    log(output_dir, f"Mask coverage: {mask_info.get('coverage_pct', '?')}%")

    # Suitability checks: stretched pose + minimal clothing
    if not args.force:
        suitable, reason = check_suitability(source, img, output_dir,
                                             args.min_extension, args.max_clothing)
        if not suitable:
            log(output_dir, f"Skip: {reason} (use --force to override)", "WARN")
            print(f"\n[SKIPPED] {reason}")
            # write a sentinel so batch runner can see why
            with open(os.path.join(output_dir, "SKIPPED.txt"), "w") as f:
                f.write(reason + "\n")
            sys.exit(2)

    backdrop_spec = parse_rgb(args.backdrop_color) if args.backdrop_color else preset["backdrop"]
    element_color = parse_rgb(args.element_color) if args.element_color else preset["element_color"]
    silhouette_color = parse_rgb(args.silhouette_color)

    canvas = make_backdrop(w, h, backdrop_spec)
    canvas = draw_element(canvas, preset["element"], element_color,
                          preset["pos"], preset["size"], w, h, seed=seed)

    sil_layer = Image.new("RGB", (w, h), silhouette_color)

    detail = max(0.0, min(0.3, args.detail))
    if detail > 0:
        orig_arr = np.array(img).astype(np.float32)
        gray = np.dot(orig_arr[..., :3], [0.299, 0.587, 0.114])
        gray_norm = (gray - gray.min()) / max(1.0, gray.max() - gray.min())
        gray_rgb = np.stack([gray_norm] * 3, axis=-1) * 255.0
        sil_arr = np.array(sil_layer).astype(np.float32)
        sil_arr = sil_arr * (1 - detail) + gray_rgb * detail
        sil_layer = Image.fromarray(sil_arr.astype(np.uint8), "RGB")

    canvas_arr = np.array(canvas).astype(np.float32)
    sil_arr = np.array(sil_layer).astype(np.float32)
    m3 = mask_arr[..., None]
    out_arr = canvas_arr * (1 - m3) + sil_arr * m3
    final = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), "RGB")

    final_path = os.path.join(output_dir, "final.jpg")
    final.save(final_path, "JPEG", quality=95)
    log(output_dir, f"Saved: {final_path}")

    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_dest = os.path.join(finals_dir, folder + ".jpg")
    try:
        final.save(finals_dest, "JPEG", quality=95)
        log(output_dir, f"Finals: {finals_dest}")
    except Exception as e:
        log(output_dir, f"Finals copy failed: {e}", "WARN")

    try:
        from notify import push_image
        push_image(finals_dest, title=f"Silhouette {args.preset}", body=src_base)
    except Exception as e:
        log(output_dir, f"Push failed: {e}", "WARN")

    try:
        shutil.copy2(os.path.abspath(__file__), os.path.join(output_dir, "silhouette-backdrop.py"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
