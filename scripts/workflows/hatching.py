#!/home/rong/openclaw-venv/bin/python3
"""
Hatching — Cross-hatched illustration from portrait photography.

Overlays depth-following cross-hatching on the original photo with region-
aware opacity: the face stays nearly photographic, the body gets subtle
hatching texture, and the background receives full hatching that fades out
(fewer lines, lower alpha) the farther you get from the subject.

Pipeline:
  1. Extract subject mask (BiRefNet via masking.py)
  2. Estimate depth map (fal.ai depth endpoint)
  3. Compute surface normals from depth gradients
  4. Body-segment for face vs body density control
  5. Distance transform from subject edge for BG fade
  6. Generate two-pass cross-hatching following surface normals
  7. Composite hatching ON TOP of original photo with region-varying opacity
  8. Output + push notification

Usage:
    python hatching.py --source photo.jpg
    python hatching.py --source photo.jpg --style fine --density 1.5
    python hatching.py --source photo.jpg --style sepia --bg-desat 0.7
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
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt
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
        "paper_color": (235, 225, 210),     # warm off-white (used as BG tint when bg_desat > 0)
        "line_width_scale": 1.0,            # multiplier on base line width
        "cross_angle_offset": 60,           # degrees between hatching passes
        "second_pass_opacity": 0.55,        # opacity of cross-hatch pass
        "face_opacity": 0.12,              # hatching overlay opacity on face
        "body_opacity": 0.35,              # hatching overlay opacity on body/subject
        "bg_max_opacity": 0.85,            # hatching overlay opacity on nearest BG
        "description": "Black ink on warm paper, medium density",
    },
    "fine": {
        "ink_color": (15, 12, 10),
        "paper_color": (240, 235, 228),
        "line_width_scale": 0.6,
        "cross_angle_offset": 45,
        "second_pass_opacity": 0.65,
        "face_opacity": 0.10,
        "body_opacity": 0.30,
        "bg_max_opacity": 0.80,
        "description": "Very dense, thin lines — master draftsman feel",
    },
    "bold": {
        "ink_color": (10, 8, 5),
        "paper_color": (230, 220, 205),
        "line_width_scale": 1.6,
        "cross_angle_offset": 70,
        "second_pass_opacity": 0.45,
        "face_opacity": 0.15,
        "body_opacity": 0.40,
        "bg_max_opacity": 0.90,
        "description": "Thick lines, high contrast, woodcut-adjacent",
    },
    "sepia": {
        "ink_color": (80, 50, 25),
        "paper_color": (235, 220, 190),
        "line_width_scale": 1.0,
        "cross_angle_offset": 55,
        "second_pass_opacity": 0.50,
        "face_opacity": 0.12,
        "body_opacity": 0.35,
        "bg_max_opacity": 0.85,
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
    bg_desat=0.6, output_dir=None,
):
    """Generate cross-hatching overlaid on original photo with region-varying opacity.

    The base image is the actual photo (not paper). Hatching is drawn on top:
      - Face: very low opacity (~12%) — face stays photographic
      - Body/subject: low-medium opacity (~35%) overlay on photo
      - Background: hatching density and alpha fade with distance from subject

    Args:
        img: original PIL Image
        luminance_arr: float32 [0,1] luminance of original, shape (H, W)
        angle_arr: float32 radians, surface normal angle per pixel, shape (H, W)
        subject_mask_arr: bool array, True=subject
        cat_mask: int array, body segment categories (0=bg, 1=hair, 2=body, 3=face, 4=clothes, 5=others)
        style_params: dict from STYLES
        density: global density multiplier
        line_width_override: override auto line width (scaled to image)
        seed: random seed
        bg_desat: desaturation strength for background (0=none, 1=fully desaturated)
        output_dir: for logging

    Returns:
        PIL Image — photo with hatching overlay
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
    face_opacity = style_params["face_opacity"]
    body_opacity = style_params["body_opacity"]
    bg_max_opacity = style_params["bg_max_opacity"]

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
    base_cell = max(4, int(short_edge * 0.012 / density))
    face_cell = max(3, int(base_cell * 0.55))
    bg_cell = max(4, int(base_cell * 0.8))  # BG cells similar to body — density
                                              # fade handled via alpha, not cell size

    log(output_dir, f"Cell sizes: base={base_cell}px, face={face_cell}px, bg={bg_cell}px")

    # --- Build region maps ---
    is_face = (cat_mask == 3)
    is_body_skin = (cat_mask == 2)
    is_hair = (cat_mask == 1)
    is_clothes = (cat_mask == 4)
    is_bg = ~subject_mask_arr

    # --- Distance transform from subject boundary for BG fade ---
    # Distance in pixels from each BG pixel to nearest subject pixel
    log(output_dir, "Computing distance field from subject boundary...")
    dist_from_subject = distance_transform_edt(~subject_mask_arr)
    # Normalize: 0 at subject boundary, 1 at max distance
    # Use a fade range of ~25% of short edge — beyond that, hatching is minimal
    fade_range = short_edge * 0.25
    dist_norm = np.clip(dist_from_subject / max(1, fade_range), 0.0, 1.0)
    # BG opacity: max near subject, fading to near-zero far away
    # Use a curve that starts high and drops off: (1 - dist)^1.5 for gradual fade
    bg_opacity_map = np.power(1.0 - dist_norm, 1.5) * bg_max_opacity
    # Only apply to BG pixels
    bg_opacity_map[subject_mask_arr] = 0.0

    log(output_dir, f"BG fade range: {fade_range:.0f}px, max dist: {dist_from_subject.max():.0f}px")

    # --- Smooth the angle field for coherent strokes ---
    sin_arr = np.sin(angle_arr)
    cos_arr = np.cos(angle_arr)
    smooth_kernel = max(3, int(short_edge * 0.015))
    sin_smooth = uniform_filter(sin_arr, size=smooth_kernel)
    cos_smooth = uniform_filter(cos_arr, size=smooth_kernel)
    angle_smooth = np.arctan2(sin_smooth, cos_smooth)

    # --- Create hatching canvas (white = no ink, dark = ink) ---
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

        max_lines = max(1, int(cell_sz / max(1.5, lw * 1.8)))
        num_lines = max(1, int(max_lines * darkness * density))

        if num_lines <= 1:
            spacing = 0
        else:
            spacing = cell_sz / num_lines

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        perp_x = -sin_a
        perp_y = cos_a
        half_len = cell_sz * 0.75

        for i in range(num_lines):
            offset = (i - (num_lines - 1) / 2.0) * spacing
            ox = cx + perp_x * offset
            oy = cy + perp_y * offset
            x0 = ox - cos_a * half_len
            y0 = oy - sin_a * half_len
            x1 = ox + cos_a * half_len
            y1 = oy + sin_a * half_len
            ink_val = int(255 * (1.0 - min(1.0, darkness * 1.1)))
            draw.line([(x0, y0), (x1, y1)], fill=ink_val, width=max(1, int(round(lw))))

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
                if not mask[min(cy, h - 1), min(cx, w - 1)]:
                    continue

                angle_val = angle_smooth[min(cy, h - 1), min(cx, w - 1)] + angle_offset

                y0c = max(0, cy - cell_sz // 2)
                y1c = min(h, cy + cell_sz // 2)
                x0c = max(0, cx - cell_sz // 2)
                x1c = min(w, cx + cell_sz // 2)
                local_lum = luminance_arr[y0c:y1c, x0c:x1c]
                if local_lum.size == 0:
                    continue
                avg_lum = np.mean(local_lum)
                darkness = 1.0 - avg_lum

                angle_val += (random.random() - 0.5) * 0.15
                _draw_hatch_cell(draw, cx, cy, cell_sz, angle_val, darkness, lw)
                cells_drawn += 1

        if label:
            log(output_dir, f"  {label}: {cells_drawn} cells hatched (cell={cell_sz}px, lw={lw:.1f}px)")

    # --- Pass 1: primary strokes everywhere ---
    log(output_dir, "Generating hatching pass 1 (primary strokes)...")

    # Face
    _hatch_region(draw1, is_face, face_cell, face_lw, angle_offset=0.0, label="Face")
    # Body skin
    _hatch_region(draw1, is_body_skin, base_cell, base_lw, angle_offset=0.0, label="Body skin")
    # Hair
    hair_cell = max(3, int(base_cell * 0.7))
    _hatch_region(draw1, is_hair, hair_cell, base_lw * 0.8, angle_offset=0.0, label="Hair")
    # Clothes
    _hatch_region(draw1, is_clothes, base_cell, base_lw, angle_offset=0.0, label="Clothes")
    # Others on subject
    is_others_subject = (cat_mask == 5) & subject_mask_arr
    _hatch_region(draw1, is_others_subject, base_cell, base_lw, angle_offset=0.0, label="Others")
    # Background — same density as body, fade handled in compositing
    _hatch_region(draw1, is_bg, bg_cell, base_lw, angle_offset=0.0, label="Background")

    # --- Pass 2: cross-hatching ---
    log(output_dir, "Generating hatching pass 2 (cross-hatching)...")

    _hatch_region(draw2, is_face, int(face_cell * 1.2), face_lw,
                  angle_offset=cross_angle, label="Face cross")
    _hatch_region(draw2, is_body_skin, int(base_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Body cross")
    _hatch_region(draw2, is_hair, int(hair_cell * 1.1), base_lw * 0.8,
                  angle_offset=cross_angle, label="Hair cross")
    _hatch_region(draw2, is_clothes, int(base_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Clothes cross")
    _hatch_region(draw2, is_others_subject, int(base_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Others cross")
    # BG cross-hatching too (will fade with distance)
    _hatch_region(draw2, is_bg, int(bg_cell * 1.3), base_lw,
                  angle_offset=cross_angle, label="Background cross")

    # --- Combine two hatching passes into a single ink layer ---
    log(output_dir, "Compositing hatching passes...")

    p1 = np.array(hatch_pass1, dtype=np.float32) / 255.0
    p2 = np.array(hatch_pass2, dtype=np.float32) / 255.0

    # Multiply blend: pass 2 at reduced opacity
    p2_blended = 1.0 - (1.0 - p2) * second_opacity
    combined = p1 * p2_blended  # 0=full ink, 1=no ink

    # Convert to ink intensity: 0=no hatching, 1=dense hatching
    ink_intensity = 1.0 - combined

    # --- Build the ink-colored hatching layer (RGB) ---
    ink_arr = np.array(ink_color, dtype=np.float32) / 255.0
    # hatching_rgb: everywhere ink color, alpha varies
    hatching_rgb = np.full((h, w, 3), ink_arr, dtype=np.float32)

    # --- Build per-pixel alpha map for compositing ---
    # Region-based opacity modulated by ink intensity
    alpha_map = np.zeros((h, w), dtype=np.float32)

    # Face: very low opacity
    alpha_map[is_face] = face_opacity
    # Body skin
    alpha_map[is_body_skin] = body_opacity
    # Hair: slightly higher than body (hair is dark, hatching reads well)
    alpha_map[is_hair] = body_opacity * 1.1
    # Clothes: same as body
    alpha_map[is_clothes] = body_opacity
    # Others on subject
    alpha_map[(cat_mask == 5) & subject_mask_arr] = body_opacity
    # Background: distance-faded opacity
    alpha_map[is_bg] = bg_opacity_map[is_bg]

    # Final alpha = region opacity * ink intensity (no ink = no overlay)
    alpha_map = alpha_map * ink_intensity

    log(output_dir, f"Overlay alpha — face: {face_opacity:.2f}, body: {body_opacity:.2f}, "
        f"bg max: {bg_max_opacity:.2f}")

    # --- Prepare base image ---
    base_arr = np.array(img, dtype=np.float32) / 255.0

    # Optionally desaturate + lighten background to make hatching more visible
    if bg_desat > 0:
        grey = np.mean(base_arr, axis=2, keepdims=True)
        paper_arr = np.array(paper_color, dtype=np.float32) / 255.0
        # Blend toward paper color for a slight tint
        desat_bg = grey * (1.0 - 0.3 * bg_desat) + paper_arr * 0.3 * bg_desat
        # Lighten slightly
        desat_bg = desat_bg + (1.0 - desat_bg) * 0.15 * bg_desat
        desat_bg = np.clip(desat_bg, 0.0, 1.0)

        # Apply only to BG, with feathered transition
        # Feather the subject mask for smooth BG treatment transition
        feather_radius = max(1, int(short_edge * 0.015))
        subj_f = gaussian_filter(subject_mask_arr.astype(np.float32), sigma=feather_radius)
        subj_f = subj_f[:, :, np.newaxis]
        base_arr = base_arr * subj_f + desat_bg * (1.0 - subj_f)
        log(output_dir, f"BG desaturation: {bg_desat:.1%}")

    # --- Composite: base photo + hatching overlay ---
    alpha_3d = alpha_map[:, :, np.newaxis]
    final_arr = base_arr * (1.0 - alpha_3d) + hatching_rgb * alpha_3d

    final_arr = np.clip(final_arr * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(final_arr)

    # --- Gouache contour strokes: wet paint marks along offset silhouette ---
    # Curved, color-sampled, with wet brush texture (tapered width, translucent edges)
    base_photo_arr = np.array(img, dtype=np.float32)  # original photo for color sampling
    bold_rng = np.random.RandomState(seed if seed else 42)

    dist_field = distance_transform_edt(1 - subject_mask_arr)
    max_dist = dist_field.max() + 1e-8

    # We'll paint gouache strokes onto an RGBA overlay, then composite
    gouache_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(gouache_layer)

    # 4 echo rings at increasing distances
    ring_distances = [0.03, 0.07, 0.13, 0.22]
    total_strokes = 0

    # First ring gets 4-5 big decisive strokes, rest get smaller ones
    for ring_i, ring_frac in enumerate(ring_distances):
        ring_dist_px = int(short_edge * ring_frac)
        band_lo = ring_dist_px - max(2, int(short_edge * 0.006))
        band_hi = ring_dist_px + max(2, int(short_edge * 0.006))
        ring_mask = (dist_field >= band_lo) & (dist_field <= band_hi)

        ring_pixels = np.where(ring_mask)
        if len(ring_pixels[0]) < 10:
            continue

        fade = 1.0 - ring_i * 0.25

        if ring_i == 0:
            # Inner ring: big decisive gouache strokes
            stroke_len = int(short_edge * 0.10)  # 10% of short edge — bold
            n_strokes = 5
            base_opacity = 0.8
            base_width = max(5, int(short_edge * 0.008))  # thick
        elif ring_i == 1:
            # Second ring: medium strokes
            stroke_len = int(short_edge * 0.06)
            n_strokes = 6
            base_opacity = 0.6
            base_width = max(4, int(short_edge * 0.005))
        else:
            # Outer rings: smaller, fading
            stroke_len = int(short_edge * 0.03 * fade)
            n_strokes = max(3, int(8 * fade))
            base_opacity = max(0.15, fade * 0.5)
            base_width = max(3, int(short_edge * 0.003 * fade))

        indices = bold_rng.choice(len(ring_pixels[0]),
                                   size=min(n_strokes, len(ring_pixels[0])), replace=False)

        for idx in indices:
            by = ring_pixels[0][idx]
            bx = ring_pixels[1][idx]

            # Tangent direction (along contour)
            gy = dist_field[min(by + 1, h - 1), bx] - dist_field[max(by - 1, 0), bx]
            gx = dist_field[by, min(bx + 1, w - 1)] - dist_field[by, max(bx - 1, 0)]
            tang_angle = math.atan2(gx, -gy)

            # Sample color from photo at this position — darken + saturate it
            sample_y = max(0, min(h - 1, by))
            sample_x = max(0, min(w - 1, bx))
            photo_rgb = base_photo_arr[sample_y, sample_x, :]  # 0-255
            # Darken by ~40%, boost saturation
            stroke_r = int(photo_rgb[0] * 0.55)
            stroke_g = int(photo_rgb[1] * 0.50)
            stroke_b = int(photo_rgb[2] * 0.55)

            # Length variation
            this_len = stroke_len * bold_rng.uniform(0.6, 1.4)

            # Generate curved stroke as chain of points with quadratic bezier bow
            n_pts = max(8, int(this_len / 3))
            cos_a = math.cos(tang_angle)
            sin_a = math.sin(tang_angle)

            # Bezier control point: perpendicular offset for curve
            # Bigger strokes get more dramatic curves
            bow_range = 0.20 if base_width >= int(short_edge * 0.006) else 0.12
            bow_amount = this_len * bold_rng.uniform(-bow_range, bow_range)
            perp_x = -sin_a * bow_amount
            perp_y = cos_a * bow_amount

            pts = []
            for ti in range(n_pts):
                t = ti / (n_pts - 1)  # 0 to 1
                # Quadratic bezier: P0 → P1(control) → P2
                p0x = bx - cos_a * this_len
                p0y = by - sin_a * this_len
                p2x = bx + cos_a * this_len
                p2y = by + sin_a * this_len
                # Control point at midpoint + perpendicular bow
                p1x = bx + perp_x
                p1y = by + perp_y
                # Bezier formula
                px = (1 - t)**2 * p0x + 2 * (1 - t) * t * p1x + t**2 * p2x
                py = (1 - t)**2 * p0y + 2 * (1 - t) * t * p1y + t**2 * p2y
                pts.append((px, py))

            # Draw stroke as overlapping circles (wet brush simulation)
            # Width tapers at both ends, max in middle
            for pi, (px, py) in enumerate(pts):
                t = pi / max(1, n_pts - 1)
                # Taper: sin curve peaks at center
                taper = math.sin(t * math.pi)
                radius = max(1, int(base_width * (0.3 + 0.7 * taper)))

                # Alpha: higher at center, lower at edges, with slight noise
                alpha_noise = bold_rng.uniform(0.85, 1.0)
                alpha = int(255 * base_opacity * (0.4 + 0.6 * taper) * alpha_noise)

                # Slight color variation along stroke (pigment isn't uniform)
                color_noise = bold_rng.uniform(0.9, 1.1)
                cr = max(0, min(255, int(stroke_r * color_noise)))
                cg = max(0, min(255, int(stroke_g * color_noise)))
                cb = max(0, min(255, int(stroke_b * color_noise)))

                ipx, ipy = int(px), int(py)
                if 0 <= ipx < w and 0 <= ipy < h:
                    g_draw.ellipse(
                        [ipx - radius, ipy - radius, ipx + radius, ipy + radius],
                        fill=(cr, cg, cb, alpha)
                    )

                    # Pigment pooling: darker edge ring at ~60% of points
                    if taper > 0.3 and bold_rng.random() < 0.6:
                        edge_r = max(1, radius + 1)
                        edge_alpha = max(10, int(alpha * 0.3))
                        edge_cr = max(0, cr - 30)
                        edge_cg = max(0, cg - 30)
                        edge_cb = max(0, cb - 30)
                        g_draw.ellipse(
                            [ipx - edge_r, ipy - edge_r, ipx + edge_r, ipy + edge_r],
                            outline=(edge_cr, edge_cg, edge_cb, edge_alpha),
                            width=1
                        )

            total_strokes += 1

    # Composite gouache layer onto result
    result = Image.alpha_composite(result.convert("RGBA"), gouache_layer).convert("RGB")
    log(output_dir, f"Added {total_strokes} gouache contour strokes across {len(ring_distances)} rings")

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

    # Step 6: Distance transform (for BG fade logging)
    log(output_dir, "Step 6: Generating cross-hatching with region-aware compositing...")
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
        bg_desat=args.bg_desat,
        output_dir=output_dir,
    )

    # Save result (with timestamp to avoid overwrites)
    ts = datetime.now(ISRAEL_TZ).strftime("%H%M%S")
    out_name = f"{src_name}_hatching_{args.style}_{ts}.jpg"
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
    comp_name = f"{src_name}_hatching_{args.style}_{ts}_comparison.jpg"
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
    parser.add_argument(
        "--bg-desat", type=float, default=0.6,
        help="Background desaturation before hatching overlay (0=none, 1=full, default: 0.6)"
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

    # Clamp values
    args.density = max(0.3, min(3.0, args.density))
    args.bg_desat = max(0.0, min(1.0, args.bg_desat))

    run_pipeline(args)


if __name__ == "__main__":
    main()
