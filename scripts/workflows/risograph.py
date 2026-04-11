#!/home/rong/openclaw-venv/bin/python3
"""
Risograph — Halftone Print Art from Portrait Photography

Transforms portrait photos into risograph/screen-print style art with coarse
halftone dot patterns, spot color separation, screen-angle moiré, slight
misregistration, and kraft paper substrate.

The halftone dots are deliberately LARGE (8-15 lpi) — visible as design
elements, not fine printing dots.

Usage:
    ./risograph.py --source photo.jpg
    ./risograph.py --source photo.jpg --palette punk --lpi 10
    ./risograph.py --source photo.jpg --palette retro --misregistration 0.8
    ./risograph.py --list-palettes
"""

import os
import sys

# Auto-load env vars from ~/sol/.env if not already set
_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# Ensure masking.py (sibling) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask

import json
import math
import time
import shutil
import random
import argparse
import threading
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALETTES = {
    "neon": {
        "description": "Fluorescent pink + teal + gold — vivid riso pop",
        "colors": [(255, 50, 120), (0, 180, 170), (255, 200, 50)],
        "paper": (210, 190, 160),       # warm kraft
        "thresholds": [0.35, 0.60, 0.85],  # dark, mid, light channel breakpoints
    },
    "retro": {
        "description": "Burnt orange + navy + cream — 1970s screen print",
        "colors": [(200, 80, 30), (30, 50, 100), (240, 220, 180)],
        "paper": (225, 210, 185),
        "thresholds": [0.30, 0.58, 0.82],
    },
    "punk": {
        "description": "Hot pink + black + electric blue — zine aesthetic",
        "colors": [(255, 20, 80), (20, 20, 20), (0, 100, 255)],
        "paper": (235, 230, 220),       # off-white newsprint
        "thresholds": [0.30, 0.55, 0.80],
    },
    "earth": {
        "description": "Terracotta + olive + sand — organic warmth",
        "colors": [(180, 80, 50), (80, 100, 40), (220, 200, 160)],
        "paper": (200, 185, 160),
        "thresholds": [0.32, 0.58, 0.82],
    },
    "mono": {
        "description": "Single dark ink on kraft — classic one-color riso",
        "colors": [(30, 30, 50)],
        "paper": (215, 195, 165),
        "thresholds": [0.50],
    },
    "coral": {
        "description": "Coral + mint + lavender — soft pastel riso",
        "colors": [(240, 100, 100), (100, 200, 180), (180, 150, 220)],
        "paper": (240, 235, 225),
        "thresholds": [0.35, 0.60, 0.85],
    },
}

# Screen angles for each channel (degrees). Classic print angles avoid moiré
# between channels while creating the characteristic rosette pattern.
SCREEN_ANGLES = [0, 15, 30, 45, 60, 75]

_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    with _log_lock:
        log_path = os.path.join(output_dir, "workflow.log")
        try:
            with open(log_path, "a") as f:
                f.write(formatted + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Halftone rendering
# ---------------------------------------------------------------------------

def render_halftone_channel(gray_arr, cell_size, angle_deg, rng):
    """
    Render a single halftone channel.

    Args:
        gray_arr: numpy float32 array [0..1], where 0=dark, 1=light
        cell_size: pixel size of each halftone cell
        angle_deg: screen rotation angle in degrees
        rng: numpy random generator

    Returns:
        numpy float32 array [0..1] — the dot mask (1 = ink, 0 = no ink)
    """
    h, w = gray_arr.shape
    # We render into a larger rotated canvas, then crop back.
    # Diagonal of the image — we need this much canvas to cover after rotation.
    diag = int(math.ceil(math.sqrt(h * h + w * w)))
    # Pad to ensure full coverage
    canvas_size = diag + cell_size * 4

    # Build a grid of dot radii on the (rotated) canvas
    n_cells_x = canvas_size // cell_size + 2
    n_cells_y = canvas_size // cell_size + 2

    # Create the dot pattern image at the rotated orientation
    # We'll work in PIL for circle drawing, then convert
    dot_img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(dot_img)

    # Rotation transform: to sample the gray image at rotated grid positions
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Center of the canvas maps to center of the original image
    cx_canvas = canvas_size / 2.0
    cy_canvas = canvas_size / 2.0
    cx_img = w / 2.0
    cy_img = h / 2.0

    max_radius = cell_size * 0.48  # max dot radius — nearly touching adjacent cells

    for iy in range(n_cells_y):
        for ix in range(n_cells_x):
            # Position in canvas space (center of this cell)
            px = ix * cell_size + cell_size / 2.0
            py = iy * cell_size + cell_size / 2.0

            # Rotate back to find corresponding position in original image
            dx = px - cx_canvas
            dy = py - cy_canvas
            img_x = cos_a * dx + sin_a * dy + cx_img
            img_y = -sin_a * dx + cos_a * dy + cy_img

            # Sample the gray value (bilinear would be nice but nearest is fine)
            ix_img = int(round(img_x))
            iy_img = int(round(img_y))

            if 0 <= ix_img < w and 0 <= iy_img < h:
                tone = gray_arr[iy_img, ix_img]  # 0=dark, 1=light
                # In riso, dark areas get BIG dots. Invert: darkness = 1-tone
                darkness = 1.0 - tone
            else:
                darkness = 0.0  # outside image — no ink

            if darkness < 0.03:
                continue  # skip near-white cells

            radius = max_radius * math.sqrt(darkness)  # sqrt for perceptual linearity
            if radius < 0.5:
                continue

            x0 = px - radius
            y0 = py - radius
            x1 = px + radius
            y1 = py + radius
            draw.ellipse([x0, y0, x1, y1], fill=255)

    # Now rotate the dot image back so dots align with the original image orientation
    # We rotate by -angle so the grid appears at +angle relative to the image
    dot_img_rotated = dot_img.rotate(angle_deg, resample=Image.BICUBIC, expand=False)

    # Crop center region matching original image size
    left = (canvas_size - w) // 2
    top = (canvas_size - h) // 2
    dot_crop = dot_img_rotated.crop((left, top, left + w, top + h))

    return np.array(dot_crop, dtype=np.float32) / 255.0


def generate_kraft_paper(w, h, color, rng):
    """Generate a kraft paper background with subtle fiber texture."""
    paper = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        paper[:, :, c] = color[c] / 255.0

    # Add noise for fiber texture
    noise = rng.normal(0, 0.03, (h, w))
    for c in range(3):
        paper[:, :, c] = np.clip(paper[:, :, c] + noise, 0, 1)

    # Add subtle horizontal streaks (paper grain)
    streak_noise = rng.normal(0, 0.015, (h, 1))
    streak_noise = np.broadcast_to(streak_noise, (h, w))
    for c in range(3):
        paper[:, :, c] = np.clip(paper[:, :, c] + streak_noise, 0, 1)

    return paper


def apply_misregistration(channel_arr, offset_x, offset_y):
    """Shift a channel by (offset_x, offset_y) pixels via numpy roll + zero-fill edges."""
    shifted = np.roll(channel_arr, int(round(offset_y)), axis=0)
    shifted = np.roll(shifted, int(round(offset_x)), axis=1)
    # Zero out wrapped edges
    oy = int(round(offset_y))
    ox = int(round(offset_x))
    if oy > 0:
        shifted[:oy, :] = 0
    elif oy < 0:
        shifted[oy:, :] = 0
    if ox > 0:
        shifted[:, :ox] = 0
    elif ox < 0:
        shifted[:, ox:] = 0
    return shifted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def risograph_transform(img, mask, palette_name, lpi, misreg_strength, seed, output_dir):
    """
    Full risograph transformation pipeline.

    Args:
        img: PIL Image (RGB)
        mask: PIL Image (L mode, 255=subject) or None
        palette_name: key into PALETTES
        lpi: lines per inch (really lines per short-edge)
        misreg_strength: 0-1, controls channel misregistration amount
        seed: random seed
        output_dir: for saving intermediates

    Returns:
        PIL Image (RGB) — final risograph result
    """
    rng = np.random.default_rng(seed)
    palette = PALETTES[palette_name]
    colors = palette["colors"]
    thresholds = palette["thresholds"]
    paper_color = palette["paper"]

    w, h = img.size
    short_edge = min(w, h)

    # Cell size in pixels — derived from lpi relative to short edge
    cell_size = max(4, int(round(short_edge / lpi)))
    log(output_dir, f"Cell size: {cell_size}px (short edge {short_edge}px / {lpi} lpi)")

    # Convert to grayscale float [0..1]
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    # Boost contrast slightly for punchier riso look
    gray = np.clip((gray - 0.5) * 1.3 + 0.5, 0, 1)

    # If we have a subject mask, boost subject brightness slightly and darken BG
    if mask is not None:
        mask_arr = np.array(mask.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
        # Slight subject boost (riso often has subject "pop")
        gray = gray * (1.0 + 0.1 * mask_arr)
        gray = np.clip(gray, 0, 1)
    else:
        mask_arr = np.ones((h, w), dtype=np.float32)

    # Build per-channel tone separations — spot color ink layers.
    #
    # Approach: each channel is a continuous-tone "ink density" map [0..1].
    # The halftone renderer converts density to dot size (bigger dot = more ink).
    #
    # Key insight: the COMBINED channels must reproduce the full tonal range
    # of the original image. Each channel covers a tonal band:
    #   Ch0 (darkest ink, e.g. pink):  responds to shadows — big dots in dark areas
    #   Ch1 (mid ink, e.g. teal):      responds to midtones — big dots in mid areas
    #   Ch2 (lightest ink, e.g. gold): responds to highlights — big dots in light areas
    #
    # Within each band, density varies from 0 (no dot) to 1 (max dot).

    n_channels = len(colors)
    channel_grays = []
    darkness = 1.0 - gray  # 0=bright, 1=deep shadow

    if n_channels == 1:
        channel_grays.append(darkness)
    else:
        # Spread channels evenly across the tonal range.
        # Each channel has a center and responds in a bell-curve around it.
        # Channel 0 centered at high darkness (shadows), last at low darkness (highlights).
        for i in range(n_channels):
            # Center positions: evenly spaced from 0.75 (shadows) to 0.25 (highlights)
            center = 0.80 - i * (0.60 / max(n_channels - 1, 1))
            # Width of response — generous overlap so no tonal gaps
            sigma = 0.30

            # Gaussian response: how much this pixel's darkness matches this channel
            response = np.exp(-0.5 * ((darkness - center) / sigma) ** 2)

            # Scale: darker areas in this band get more ink (bigger dots)
            # The response already peaks at the right tonal zone, but we also
            # want the dot SIZE to vary with darkness within the band.
            # Multiply by the actual darkness value mapped through the band.
            density = response * np.clip(darkness * 1.2, 0, 1)

            # For the highlight channel: invert — light areas get big dots
            if i == n_channels - 1:
                # Highlight ink: density is high where image is BRIGHT
                density = response * np.clip((1.0 - darkness) * 1.0, 0, 1)

            # Normalize to use full range
            d_max = np.percentile(density, 99) if np.max(density) > 0 else 1.0
            density = np.clip(density / max(d_max, 0.01), 0, 1)

            # Suppress very faint dots (below 8% density)
            density[density < 0.08] = 0

            channel_grays.append(density)

    # Save channel separations for debug
    for i, cg in enumerate(channel_grays):
        ch_debug = Image.fromarray((cg * 255).astype(np.uint8), "L")
        ch_debug.save(os.path.join(output_dir, f"channel_{i}_tone.png"))

    # Render halftone dots for each channel
    log(output_dir, f"Rendering {n_channels} halftone channels...")
    halftone_channels = []
    for i, cg in enumerate(channel_grays):
        angle = SCREEN_ANGLES[i % len(SCREEN_ANGLES)]
        log(output_dir, f"  Channel {i} ({colors[i]}): angle={angle}deg")
        ht = render_halftone_channel(cg, cell_size, angle, rng)
        halftone_channels.append(ht)

        # Save intermediate
        ht_debug = Image.fromarray((ht * 255).astype(np.uint8), "L")
        ht_debug.save(os.path.join(output_dir, f"channel_{i}_halftone.png"))

    # Apply misregistration offsets
    # Max offset = 0.5% of short edge at full misregistration strength
    max_offset_px = short_edge * 0.005 * misreg_strength
    log(output_dir, f"Misregistration: max offset {max_offset_px:.1f}px (strength={misreg_strength})")

    offset_channels = []
    for i, ht in enumerate(halftone_channels):
        if i == 0:
            # First channel is the "key" — no offset
            offset_channels.append(ht)
        else:
            ox = rng.uniform(-max_offset_px, max_offset_px)
            oy = rng.uniform(-max_offset_px, max_offset_px)
            log(output_dir, f"  Channel {i} offset: ({ox:.1f}, {oy:.1f})px")
            shifted = apply_misregistration(ht, ox, oy)
            offset_channels.append(shifted)

    # Generate kraft paper background
    log(output_dir, "Generating kraft paper background...")
    paper = generate_kraft_paper(w, h, paper_color, rng)

    # Composite: risograph inks are semi-opaque spot colors printed on paper.
    # Each ink layer partially covers the previous layers. Where two inks
    # overlap, the top ink dominates but the bottom bleeds through slightly.
    #
    # We use alpha-over compositing with semi-transparent ink (~75% opacity).
    # This gives the characteristic risograph look: distinct spot colors with
    # subtle color mixing at overlaps.

    result = paper.copy()

    # Print lightest ink first, darkest last (on top) — like real riso print order.
    # Reverse so channel 0 (darkest) is on top for visual punch.
    print_order = list(range(len(offset_channels)))[::-1]

    for i in print_order:
        ht = offset_channels[i]
        color = np.array(colors[i], dtype=np.float32) / 255.0
        ink_opacity = 0.80  # riso ink is semi-opaque

        for c in range(3):
            # Alpha-over: where dot exists, blend toward ink color
            alpha = ht * ink_opacity
            result[:, :, c] = result[:, :, c] * (1.0 - alpha) + color[c] * alpha

    # Add very subtle paper grain overlay
    grain = rng.normal(0, 0.012, (h, w, 3)).astype(np.float32)
    result = np.clip(result + grain, 0, 1)

    # Convert to uint8
    result_uint8 = (result * 255).astype(np.uint8)
    result_img = Image.fromarray(result_uint8, "RGB")

    return result_img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Risograph — Halftone Print Art from Portrait Photography"
    )
    parser.add_argument("--source", type=str, help="Source photo path")
    parser.add_argument(
        "--palette", type=str, default="neon",
        help=f"Color palette (default: neon). Options: {', '.join(PALETTES.keys())}"
    )
    parser.add_argument(
        "--lpi", type=int, default=12,
        help="Lines per inch / dot density (8-20, default 12). Lower = bigger dots."
    )
    parser.add_argument(
        "--misregistration", type=float, default=0.5,
        help="Channel misregistration amount (0-1, default 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--affect", type=str, default="subject",
        help="Mask target for subject extraction (default: subject)"
    )
    parser.add_argument(
        "--exclude", type=str, default="",
        help="Parts to exclude from mask (e.g., hands,ropes)"
    )
    parser.add_argument(
        "--output-to", type=str, default="local",
        choices=["local", "gdrive", "both"],
        help="Where to save output (default: local)"
    )
    parser.add_argument(
        "--local-output-dir", type=str,
        default=os.path.expanduser("~/.openclaw/workspace/shared"),
        help="Local output directory"
    )
    parser.add_argument(
        "--list-palettes", action="store_true",
        help="List available palettes and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # List palettes and exit
    if args.list_palettes:
        print("\nAvailable risograph palettes:\n")
        for name, pal in PALETTES.items():
            color_strs = [f"({r},{g},{b})" for r, g, b in pal["colors"]]
            print(f"  {name:10s} — {pal['description']}")
            print(f"             Colors: {' + '.join(color_strs)}")
            print()
        return

    if not args.source:
        print("ERROR: --source is required (use --list-palettes to see options)")
        sys.exit(1)

    if not os.path.isfile(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        sys.exit(1)

    if args.palette not in PALETTES:
        print(f"ERROR: Unknown palette '{args.palette}'. Use --list-palettes to see options.")
        sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)

    # Create output directory
    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y%m%d_%H%M%S")
    src_name = os.path.splitext(os.path.basename(args.source))[0]
    run_name = f"risograph_{src_name}_{args.palette}_{timestamp}"
    output_dir = os.path.join(args.local_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log(output_dir, f"=== Risograph Transform ===")
    log(output_dir, f"Source: {args.source}")
    log(output_dir, f"Palette: {args.palette}")
    log(output_dir, f"LPI: {args.lpi}")
    log(output_dir, f"Misregistration: {args.misregistration}")
    log(output_dir, f"Seed: {seed}")

    t_start = time.time()

    # Load image
    img = Image.open(args.source).convert("RGB")
    w, h = img.size
    log(output_dir, f"Image size: {w}x{h}")

    # Save original to output dir
    img.save(os.path.join(output_dir, "0_original.jpg"), quality=95)

    # --- Step 1: Extract subject mask ---
    log(output_dir, "--- Step 1: Subject mask ---")
    t0 = time.time()
    try:
        mask, mask_info = build_mask(
            args.source,
            affect=args.affect,
            exclude=args.exclude,
            output_dir=output_dir,
        )
        log(output_dir, f"Mask: {mask_info['engine']}, coverage={mask_info['coverage_pct']:.1f}%")
        mask.save(os.path.join(output_dir, "1_mask.png"))
    except Exception as e:
        log(output_dir, f"Mask extraction failed ({e}), proceeding without mask", "WARN")
        mask = None
    t_mask = time.time() - t0
    log(output_dir, f"Step 1 done ({t_mask:.1f}s)")

    # --- Step 2: Risograph transform ---
    log(output_dir, "--- Step 2: Risograph transform ---")
    t0 = time.time()
    result = risograph_transform(
        img=img,
        mask=mask,
        palette_name=args.palette,
        lpi=args.lpi,
        misreg_strength=args.misregistration,
        seed=seed,
        output_dir=output_dir,
    )
    result.save(os.path.join(output_dir, "2_risograph.jpg"), quality=95)
    t_riso = time.time() - t0
    log(output_dir, f"Step 2 done ({t_riso:.1f}s)")

    # --- Step 3: Output ---
    log(output_dir, "--- Step 3: Output ---")

    # Copy to finals
    finals_dir = os.path.join(args.local_output_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_name = f"{run_name}.jpg"
    finals_path = os.path.join(finals_dir, finals_name)
    result.save(finals_path, quality=95)
    log(output_dir, f"Final: {finals_path}")

    # Save side-by-side comparison
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(img, (0, 0))
    comparison.paste(result, (w, 0))
    comp_path = os.path.join(finals_dir, f"{run_name}_comparison.jpg")
    comparison.save(comp_path, quality=92)
    log(output_dir, f"Comparison: {comp_path}")

    # Push to phone
    try:
        from notify import push_image
        push_image(
            finals_path,
            title=f"Risograph — {src_name}",
            body=f"{args.palette} palette, {args.lpi} lpi"
        )
        log(output_dir, "Pushed to phone")
    except Exception as e:
        log(output_dir, f"Push notification failed: {e}", "WARN")

    # Copy script for reproducibility
    try:
        shutil.copy2(
            os.path.abspath(__file__),
            os.path.join(output_dir, f"workflow_script_{os.path.basename(__file__)}")
        )
    except Exception:
        pass

    # Save metadata
    metadata = {
        "source": os.path.abspath(args.source),
        "palette": args.palette,
        "lpi": args.lpi,
        "misregistration": args.misregistration,
        "seed": seed,
        "affect": args.affect,
        "exclude": args.exclude,
        "timestamp": timestamp,
        "timings": {
            "mask": round(t_mask, 1),
            "transform": round(t_riso, 1),
            "total": round(time.time() - t_start, 1),
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t_start
    print(f"""
============================================================
  RISOGRAPH SUMMARY
============================================================
  Source:          {args.source}
  Palette:         {args.palette} ({len(PALETTES[args.palette]['colors'])} colors)
  LPI:             {args.lpi} (cell size ~{min(w,h)//args.lpi}px)
  Misregistration: {args.misregistration}
  Seed:            {seed}
  Total time:      {total_time:.1f}s
  Final:           {finals_path}
  Comparison:      {comp_path}
============================================================
""")


if __name__ == "__main__":
    main()
