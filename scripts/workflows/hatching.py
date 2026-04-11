#!/home/rong/openclaw-venv/bin/python3
"""
Hatching — Cross-hatched illustration from portrait photography.

Transforms a portrait photo into a cross-hatched pen-and-ink illustration
where hatching lines follow the body's 3D surface curvature (derived from
a depth map). Darker regions get denser hatching; the face gets finer strokes;
the background gets sparse or no hatching.

Pipeline:
  1. Extract subject mask (BiRefNet via masking.py)
  2. Estimate depth map (fal.ai depth endpoint)
  3. Compute surface normals from depth gradients
  4. Body-segment for face vs body density control
  5. Generate two-pass cross-hatching following surface normals
  6. Composite on desaturated/tinted base
  7. Output + push notification

Usage:
    python hatching.py --source photo.jpg
    python hatching.py --source photo.jpg --style fine --density 1.5
    python hatching.py --source photo.jpg --style sepia --density 0.8
    python hatching.py --list-styles
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

# fal_client needs FAL_KEY
os.environ.setdefault("FAL_KEY", os.environ.get("FAL_API_KEY", ""))

import math
import argparse
import shutil
import random
import threading
import importlib.util
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageEnhance
from scipy.ndimage import gaussian_filter, uniform_filter
import fal_client

# Use shared masking module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from masking import build_mask

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Styles (hatching presets)
# ---------------------------------------------------------------------------
STYLES = {
    "classic": {
        "ink_color": (25, 22, 18),          # near-black warm ink
        "paper_color": (235, 225, 210),     # warm off-white paper
        "base_tint": (0.15, 0.10, 0.05),   # slight warm desaturation tint (added to grey)
        "line_width_scale": 1.0,            # multiplier on base line width
        "cross_angle_offset": 60,           # degrees between hatching passes
        "second_pass_opacity": 0.55,        # opacity of cross-hatch pass
        "description": "Black ink on warm paper, medium density",
    },
    "fine": {
        "ink_color": (15, 12, 10),
        "paper_color": (240, 235, 228),
        "base_tint": (0.08, 0.06, 0.03),
        "line_width_scale": 0.6,
        "cross_angle_offset": 45,
        "second_pass_opacity": 0.65,
        "description": "Very dense, thin lines — master draftsman feel",
    },
    "bold": {
        "ink_color": (10, 8, 5),
        "paper_color": (230, 220, 205),
        "base_tint": (0.12, 0.08, 0.04),
        "line_width_scale": 1.6,
        "cross_angle_offset": 70,
        "second_pass_opacity": 0.45,
        "description": "Thick lines, high contrast, woodcut-adjacent",
    },
    "sepia": {
        "ink_color": (80, 50, 25),
        "paper_color": (235, 220, 190),
        "base_tint": (0.25, 0.15, 0.05),
        "line_width_scale": 1.0,
        "cross_angle_offset": 55,
        "second_pass_opacity": 0.50,
        "description": "Brown ink on cream paper, vintage",
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()


def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    if output_dir:
        with _log_lock:
            log_path = os.path.join(output_dir, "workflow.log")
            try:
                with open(log_path, "a") as f:
                    f.write(formatted + "\n")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Step 1: Depth estimation via fal.ai
# ---------------------------------------------------------------------------
def estimate_depth(source_path, output_dir):
    """Get depth map from fal.ai depth endpoint. Returns numpy float32 array [0,1]."""
    log(output_dir, "Uploading image for depth estimation...")
    image_url = fal_client.upload_file(source_path)

    log(output_dir, "Estimating depth map (fal-ai/imageutils/depth)...")
    result = fal_client.subscribe(
        "fal-ai/imageutils/depth",
        arguments={"image_url": image_url},
    )

    depth_url = result["image"]["url"]
    log(output_dir, f"Depth map received: {depth_url}")

    import requests
    resp = requests.get(depth_url, timeout=60)
    resp.raise_for_status()
    depth_img = Image.open(BytesIO(resp.content)).convert("L")

    depth_arr = np.array(depth_img, dtype=np.float32) / 255.0
    return depth_arr, depth_img


# ---------------------------------------------------------------------------
# Step 2: Surface normals from depth
# ---------------------------------------------------------------------------
def compute_surface_normals(depth_arr, smooth_sigma_pct=0.008):
    """Compute surface normal direction from depth gradients.

    Returns angle array in radians (same shape as depth_arr).
    smooth_sigma_pct: gaussian smoothing sigma as fraction of short edge.
    """
    h, w = depth_arr.shape
    short_edge = min(h, w)
    sigma = max(2, int(short_edge * smooth_sigma_pct))

    # Smooth depth to reduce noise before gradient
    smoothed = gaussian_filter(depth_arr, sigma=sigma)

    # Compute gradients
    dy, dx = np.gradient(smoothed)

    # Surface normal direction: perpendicular to gradient = hatching follows contours
    # Gradient points in steepest-ascent direction; hatching should go perpendicular
    angle = np.arctan2(dx, -dy)  # perpendicular to gradient direction

    return angle


# ---------------------------------------------------------------------------
# Step 3: Body segmentation for region-specific density
# ---------------------------------------------------------------------------
def get_body_segments(img_array, output_dir):
    """Load body-segment module and get category mask.

    Returns category mask: 0=bg, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
    """
    log(output_dir, "Running body segmentation (MediaPipe)...")
    spec = importlib.util.spec_from_file_location(
        "body_segment", os.path.join(script_dir, "body-segment.py")
    )
    bs_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs_mod)
    cat_mask = bs_mod.segment_body(img_array)
    return cat_mask


# ---------------------------------------------------------------------------
# Step 4: Hatching generation
# ---------------------------------------------------------------------------
def generate_hatching(
    img, luminance_arr, angle_arr, subject_mask_arr, cat_mask,
    style_params, density=1.0, line_width_override=None, seed=None,
    output_dir=None,
):
    """Generate cross-hatched illustration.

    Args:
        img: original PIL Image (for sizing)
        luminance_arr: float32 [0,1] luminance of original, shape (H, W)
        angle_arr: float32 radians, surface normal angle per pixel, shape (H, W)
        subject_mask_arr: bool array, True=subject
        cat_mask: int array, body segment categories (0=bg, 1=hair, 2=body, 3=face, 4=clothes, 5=others)
        style_params: dict from STYLES
        density: global density multiplier
        line_width_override: override auto line width (scaled to image)
        seed: random seed
        output_dir: for logging

    Returns:
        PIL Image with hatching on paper background
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    w, h = img.size
    short_edge = min(w, h)

    ink_color = style_params["ink_color"]
    paper_color = style_params["paper_color"]
    lw_scale = style_params["line_width_scale"]
    cross_angle = math.radians(style_params["cross_angle_offset"])
    second_opacity = style_params["second_pass_opacity"]

    # --- Line width: scale to image size ---
    # Base: ~0.08% of short edge, clamped to [1, 4]
    if line_width_override is not None:
        base_lw = line_width_override
    else:
        base_lw = max(1, min(4, short_edge * 0.0008 * lw_scale))
    base_lw = max(1, base_lw)

    # Face gets finer lines (60% of base)
    face_lw = max(1, base_lw * 0.6)

    log(output_dir, f"Line width: base={base_lw:.1f}px, face={face_lw:.1f}px "
        f"(image short edge={short_edge}px)")

    # --- Cell size: determines hatching resolution ---
    # Smaller cells = more lines, finer detail
    # Base cell: ~1.2% of short edge for normal density
    base_cell = max(4, int(short_edge * 0.012 / density))
    face_cell = max(3, int(base_cell * 0.55))  # face gets ~55% cell size (denser)
    bg_cell = int(base_cell * 2.5)  # background gets very sparse

    log(output_dir, f"Cell sizes: base={base_cell}px, face={face_cell}px, bg={bg_cell}px")

    # --- Build region maps ---
    is_face = (cat_mask == 3)
    is_body_skin = (cat_mask == 2)
    is_hair = (cat_mask == 1)
    is_clothes = (cat_mask == 4)
    is_bg = ~subject_mask_arr

    # --- Smooth the angle field for coherent strokes ---
    # Use a circular mean via sin/cos components
    sin_arr = np.sin(angle_arr)
    cos_arr = np.cos(angle_arr)
    smooth_kernel = max(3, int(short_edge * 0.015))
    sin_smooth = uniform_filter(sin_arr, size=smooth_kernel)
    cos_smooth = uniform_filter(cos_arr, size=smooth_kernel)
    angle_smooth = np.arctan2(sin_smooth, cos_smooth)

    # --- Create hatching canvas (white/paper) ---
    hatch_pass1 = Image.new("L", (w, h), 255)
    hatch_pass2 = Image.new("L", (w, h), 255)
    draw1 = ImageDraw.Draw(hatch_pass1)
    draw2 = ImageDraw.Draw(hatch_pass2)

    def _draw_hatch_cell(draw, cx, cy, cell_sz, angle_rad, darkness, lw):
        """Draw parallel hatch lines in a cell based on darkness level.

        darkness: 0=white (no lines), 1=black (max lines)
        angle_rad: direction of lines
        """
        if darkness < 0.08:
            return  # too light, skip

        # Number of lines scales with darkness and density
        # At max darkness, fill cell with lines spaced ~lw*1.5 apart
        max_lines = max(1, int(cell_sz / max(1.5, lw * 1.8)))
        num_lines = max(1, int(max_lines * darkness * density))

        # Line spacing within cell
        if num_lines <= 1:
            spacing = 0
        else:
            spacing = cell_sz / num_lines

        # Direction vectors
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Perpendicular for spacing offset
        perp_x = -sin_a
        perp_y = cos_a

        half_len = cell_sz * 0.75  # lines extend beyond cell for overlap

        # Draw parallel lines centered on cell
        for i in range(num_lines):
            offset = (i - (num_lines - 1) / 2.0) * spacing
            ox = cx + perp_x * offset
            oy = cy + perp_y * offset

            x0 = ox - cos_a * half_len
            y0 = oy - sin_a * half_len
            x1 = ox + cos_a * half_len
            y1 = oy + sin_a * half_len

            # Opacity based on darkness (darker = more opaque lines)
            ink_val = int(255 * (1.0 - min(1.0, darkness * 1.1)))
            draw.line([(x0, y0), (x1, y1)], fill=ink_val, width=max(1, int(round(lw))))

    # --- Process cells in different regions ---
    # We use different cell sizes for face, body, and background
    def _hatch_region(draw, mask, cell_sz, lw, angle_offset=0.0, label=""):
        """Fill a region with hatching."""
        ys = np.arange(0, h, cell_sz)
        xs = np.arange(0, w, cell_sz)

        cells_drawn = 0
        for cy_start in ys:
            for cx_start in xs:
                cy = int(cy_start + cell_sz // 2)
                cx = int(cx_start + cell_sz // 2)

                if cy >= h or cx >= w:
                    continue

                # Check if cell center is in region
                if not mask[min(cy, h - 1), min(cx, w - 1)]:
                    continue

                # Sample angle from smoothed field
                angle_val = angle_smooth[min(cy, h - 1), min(cx, w - 1)] + angle_offset

                # Sample darkness from luminance (invert: dark areas = high value)
                # Average luminance in cell neighbourhood
                y0c = max(0, cy - cell_sz // 2)
                y1c = min(h, cy + cell_sz // 2)
                x0c = max(0, cx - cell_sz // 2)
                x1c = min(w, cx + cell_sz // 2)
                local_lum = luminance_arr[y0c:y1c, x0c:x1c]
                if local_lum.size == 0:
                    continue
                avg_lum = np.mean(local_lum)
                darkness = 1.0 - avg_lum  # invert: dark image = dense hatching

                # Add slight randomness to angle for organic feel
                angle_val += (random.random() - 0.5) * 0.15

                _draw_hatch_cell(draw, cx, cy, cell_sz, angle_val, darkness, lw)
                cells_drawn += 1

        if label:
            log(output_dir, f"  {label}: {cells_drawn} cells hatched (cell={cell_sz}px, lw={lw:.1f}px)")

    log(output_dir, "Generating hatching pass 1 (primary strokes)...")

    # Face: fine, dense hatching
    _hatch_region(draw1, is_face, face_cell, face_lw, angle_offset=0.0, label="Face")
    # Body skin: medium hatching
    _hatch_region(draw1, is_body_skin, base_cell, base_lw, angle_offset=0.0, label="Body skin")
    # Hair: medium-dense hatching
    hair_cell = max(3, int(base_cell * 0.7))
    _hatch_region(draw1, is_hair, hair_cell, base_lw * 0.8, angle_offset=0.0, label="Hair")
    # Clothes: medium hatching
    _hatch_region(draw1, is_clothes, base_cell, base_lw, angle_offset=0.0, label="Clothes")
    # Others (on subject): medium
    is_others_subject = (cat_mask == 5) & subject_mask_arr
    _hatch_region(draw1, is_others_subject, base_cell, base_lw, angle_offset=0.0, label="Others")
    # Background: sparse
    _hatch_region(draw1, is_bg, bg_cell, base_lw * 0.7, angle_offset=0.0, label="Background")

    log(output_dir, "Generating hatching pass 2 (cross-hatching)...")

    # Second pass: cross-hatch at offset angle, slightly sparser
    cross_density_mult = 0.7  # fewer lines in cross pass

    # Only cross-hatch areas that are reasonably dark (shadows/midtones)
    # Face cross-hatching
    _hatch_region(draw2, is_face, int(face_cell * 1.2), face_lw,
                  angle_offset=cross_angle, label="Face cross")
    # Body cross-hatching
    _hatch_region(draw2, is_body_skin, int(base_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Body cross")
    # Hair cross-hatching (denser since hair is dark)
    _hatch_region(draw2, is_hair, int(hair_cell * 1.1), base_lw * 0.8,
                  angle_offset=cross_angle, label="Hair cross")
    # Clothes cross-hatching
    _hatch_region(draw2, is_clothes, int(base_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Clothes cross")
    # Others on subject
    _hatch_region(draw2, is_others_subject, int(base_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Others cross")
    # No cross-hatch for background (keeps it airy)

    # --- Composite the two passes ---
    log(output_dir, "Compositing hatching passes...")

    # Convert to arrays for blending
    p1 = np.array(hatch_pass1, dtype=np.float32) / 255.0
    p2 = np.array(hatch_pass2, dtype=np.float32) / 255.0

    # Multiply blend: both passes darken
    # Pass 2 at reduced opacity
    p2_blended = 1.0 - (1.0 - p2) * second_opacity
    combined = p1 * p2_blended

    # --- Create final image ---
    # Paper background with ink-colored hatching
    paper = Image.new("RGB", (w, h), paper_color)
    paper_arr = np.array(paper, dtype=np.float32) / 255.0

    ink_arr = np.array(ink_color, dtype=np.float32) / 255.0

    # Where combined is dark (low values), use ink; where bright, use paper
    combined_3d = combined[:, :, np.newaxis]
    final_arr = paper_arr * combined_3d + ink_arr * (1.0 - combined_3d)

    # --- Overlay subtle original image tones for depth ---
    # Desaturated, low-opacity original underneath gives tonal variation
    grey = ImageOps.grayscale(img)
    grey_arr = np.array(grey, dtype=np.float32) / 255.0

    # Apply base tint
    tint = np.array(style_params["base_tint"], dtype=np.float32)
    tinted = grey_arr[:, :, np.newaxis] * (1.0 - tint) + tint

    # Blend: mostly hatching, subtle tonal underlay (15% opacity)
    tone_opacity = 0.15
    # Only where subject is present, let some tone through
    subj_f = subject_mask_arr.astype(np.float32)[:, :, np.newaxis]
    tone_mask = subj_f * tone_opacity

    # Resize tinted if dimensions differ (depth map might differ)
    if tinted.shape[:2] != final_arr.shape[:2]:
        tinted_img = Image.fromarray((tinted * 255).astype(np.uint8))
        tinted_img = tinted_img.resize((w, h), Image.LANCZOS)
        tinted = np.array(tinted_img, dtype=np.float32) / 255.0

    final_arr = final_arr * (1.0 - tone_mask) + tinted * tone_mask

    final_arr = np.clip(final_arr * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(final_arr)

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    """Run the hatching pipeline."""

    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source file not found: {source}")
        sys.exit(1)

    # Output directory
    out_base = os.path.expanduser(args.local_output_dir)
    src_name = os.path.splitext(os.path.basename(source))[0]
    ts = datetime.now(ISRAEL_TZ).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(out_base, f"hatching_{src_name}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    log(output_dir, f"=== Hatching Pipeline ===")
    log(output_dir, f"Source: {source}")
    log(output_dir, f"Style: {args.style}")
    log(output_dir, f"Density: {args.density}")
    log(output_dir, f"Output dir: {output_dir}")

    style_params = STYLES[args.style]

    # Load image
    img = Image.open(source).convert("RGB")
    w, h = img.size
    log(output_dir, f"Image size: {w}x{h}")
    img_arr = np.array(img)

    # Step 1: Subject mask
    log(output_dir, "Step 1: Extracting subject mask (BiRefNet)...")
    subject_mask, mask_info = build_mask(
        source, affect="subject", exclude="", output_dir=output_dir
    )
    log(output_dir, f"Subject coverage: {mask_info['coverage_pct']:.1f}%")

    if subject_mask.size != (w, h):
        subject_mask = subject_mask.resize((w, h), Image.LANCZOS)
    subject_mask.save(os.path.join(output_dir, "subject_mask.png"))
    subject_mask_arr = np.array(subject_mask) > 127

    # Step 2: Depth estimation
    log(output_dir, "Step 2: Estimating depth map...")
    depth_arr, depth_img = estimate_depth(source, output_dir)

    # Resize depth to match image if needed
    if depth_arr.shape[:2] != (h, w):
        depth_pil = Image.fromarray((depth_arr * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((w, h), Image.LANCZOS)
        depth_arr = np.array(depth_pil, dtype=np.float32) / 255.0
        depth_img = depth_pil

    depth_img.save(os.path.join(output_dir, "depth_map.png"))
    log(output_dir, f"Depth range: {depth_arr.min():.3f} - {depth_arr.max():.3f}")

    # Step 3: Surface normals
    log(output_dir, "Step 3: Computing surface normals from depth...")
    angle_arr = compute_surface_normals(depth_arr)

    # Save angle visualization (hue-mapped)
    angle_norm = ((angle_arr + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    angle_vis = Image.fromarray(angle_norm, mode="L")
    angle_vis.save(os.path.join(output_dir, "surface_normals.png"))

    # Step 4: Body segmentation
    log(output_dir, "Step 4: Body segmentation for region density...")
    cat_mask = get_body_segments(img_arr, output_dir)

    if cat_mask.shape[:2] != (h, w):
        cat_pil = Image.fromarray(cat_mask.astype(np.uint8))
        cat_pil = cat_pil.resize((w, h), Image.NEAREST)
        cat_mask = np.array(cat_pil)

    # Save category visualization
    cat_vis = (cat_mask.astype(np.float32) / 5.0 * 255).astype(np.uint8)
    Image.fromarray(cat_vis).save(os.path.join(output_dir, "body_segments.png"))

    face_pct = np.sum(cat_mask == 3) / cat_mask.size * 100
    body_pct = np.sum(cat_mask == 2) / cat_mask.size * 100
    hair_pct = np.sum(cat_mask == 1) / cat_mask.size * 100
    log(output_dir, f"Segments: face={face_pct:.1f}%, body={body_pct:.1f}%, hair={hair_pct:.1f}%")

    # Step 5: Luminance
    luminance_arr = np.array(ImageOps.grayscale(img), dtype=np.float32) / 255.0

    # Step 6: Generate hatching
    log(output_dir, "Step 5: Generating cross-hatching...")
    result = generate_hatching(
        img=img,
        luminance_arr=luminance_arr,
        angle_arr=angle_arr,
        subject_mask_arr=subject_mask_arr,
        cat_mask=cat_mask,
        style_params=style_params,
        density=args.density,
        line_width_override=args.line_width,
        seed=args.seed,
        output_dir=output_dir,
    )

    # Save result
    out_name = f"{src_name}_hatching_{args.style}.jpg"
    out_path = os.path.join(output_dir, out_name)
    result.save(out_path, quality=95)
    log(output_dir, f"Saved: {out_path}")

    # Copy to finals
    finals_dir = os.path.join(out_base, "finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_dest = os.path.join(finals_dir, out_name)
    result.save(finals_dest, quality=95)
    log(output_dir, f"Finals: {finals_dest}")

    # Side-by-side comparison
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(img, (0, 0))
    comparison.paste(result, (w, 0))
    comp_name = f"{src_name}_hatching_{args.style}_comparison.jpg"
    comp_path = os.path.join(finals_dir, comp_name)
    comparison.save(comp_path, quality=92)
    log(output_dir, f"Comparison: {comp_path}")

    # Push to phone
    try:
        from notify import push_image
        push_image(finals_dest, title=f"Hatching — {src_name}", body=f"{args.style}, density={args.density}")
        log(output_dir, "Pushed to phone")
    except Exception as e:
        log(output_dir, f"Push notification failed: {e}", "WARN")

    log(output_dir, "=== Done ===")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hatching — cross-hatched illustration from portrait photography"
    )
    parser.add_argument("--source", help="Source photo path")
    parser.add_argument(
        "--style", default="classic", choices=list(STYLES.keys()),
        help="Hatching style preset (default: classic)"
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Hatching density multiplier, 0.5-2.0 (default: 1.0)"
    )
    parser.add_argument(
        "--line-width", type=float, default=None,
        help="Override line width in pixels (default: auto-scaled to image)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output-to", default="local", choices=["local", "gdrive", "both"]
    )
    parser.add_argument(
        "--local-output-dir",
        default=os.path.expanduser("~/.openclaw/workspace/shared"),
    )
    parser.add_argument(
        "--list-styles", action="store_true", help="List available hatching styles"
    )

    args = parser.parse_args()

    if args.list_styles:
        print("\nAvailable hatching styles:")
        for name, params in STYLES.items():
            print(f"  {name:12s} — {params['description']}")
        print()
        return

    if not args.source:
        parser.error("--source is required (unless using --list-styles)")

    # Clamp density
    args.density = max(0.3, min(3.0, args.density))

    run_pipeline(args)


if __name__ == "__main__":
    main()
