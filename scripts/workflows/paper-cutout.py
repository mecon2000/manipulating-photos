#!/home/rong/openclaw-venv/bin/python3
"""
Paper Cutout — Transform portraits into layered paper cutout art.

Posterizes a photo into tonal layers, then renders each layer as a separate
sheet of textured craft paper with rough torn/cut edges and drop shadows,
stacked darkest-to-lightest to create genuine 3D depth perception.

Face region gets extra tonal detail (more layers) to preserve facial features.

Pipeline:
  1. Extract subject mask (BiRefNet)
  2. Detect face region (MediaPipe body-segment)
  3. Posterize into tonal layers (percentile-based thresholds)
  4. For each layer: rough edges + paper texture fill + drop shadow
  5. Stack layers back-to-front
  6. Composite with paper background
  7. Output + push notification

Pure PIL/numpy/scipy/cv2 — no API calls for the effect itself (only BiRefNet
for subject extraction).

Usage:
    python paper-cutout.py --source photo.jpg
    python paper-cutout.py --source photo.jpg --palette pastel --num-layers 5
    python paper-cutout.py --source photo.jpg --palette monochrome --shadow-strength 0.8
    python paper-cutout.py --list-palettes
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

import argparse
import importlib.util
import math
import random
import shutil
import threading
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from io import BytesIO

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import cv2
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from scipy.ndimage import gaussian_filter

# Use shared masking module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from masking import build_mask

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Palettes — each is a list of colors from darkest (back) to lightest (front)
# Plus a background paper color.
# ---------------------------------------------------------------------------
PALETTES = {
    "craft": {
        "bg": (200, 185, 160),
        "layers": [
            (70, 50, 30),       # deep brown
            (120, 90, 55),      # dark kraft
            (165, 135, 95),     # medium kraft
            (200, 175, 140),    # light kraft
            (235, 225, 205),    # cream
            (248, 242, 230),    # pale cream
        ],
        "description": "Kraft brown layers — deep brown to cream",
    },
    "pastel": {
        "bg": (250, 248, 245),
        "layers": [
            (160, 130, 155),    # dusty mauve
            (190, 160, 185),    # soft lavender
            (175, 200, 180),    # mint
            (240, 195, 190),    # pink
            (240, 230, 210),    # cream
            (250, 248, 245),    # near white
        ],
        "description": "Soft pastel colors — mauve, lavender, mint, pink, cream",
    },
    "monochrome": {
        "bg": (245, 245, 245),
        "layers": [
            (50, 50, 55),       # near black
            (100, 100, 105),    # dark grey
            (150, 150, 152),    # medium grey
            (195, 195, 197),    # light grey
            (225, 225, 227),    # pale grey
            (242, 242, 244),    # near white
        ],
        "description": "Shades of grey paper on white — bas-relief feel",
    },
    "warm": {
        "bg": (248, 240, 228),
        "layers": [
            (120, 65, 50),      # terracotta dark
            (165, 95, 70),      # terracotta
            (200, 130, 100),    # salmon
            (225, 180, 150),    # peach
            (240, 220, 200),    # cream
            (250, 242, 232),    # warm white
        ],
        "description": "Terracotta, salmon, peach, cream, white",
    },
    "cool": {
        "bg": (240, 245, 250),
        "layers": [
            (55, 65, 80),       # slate
            (85, 105, 130),     # steel blue
            (130, 155, 175),    # blue grey
            (175, 195, 210),    # powder blue
            (215, 228, 238),    # ice
            (238, 243, 248),    # near white
        ],
        "description": "Slate, steel blue, powder blue, ice, white",
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
            with open(log_path, "a") as f:
                f.write(formatted + "\n")


# ---------------------------------------------------------------------------
# Step 1: Extract subject mask
# ---------------------------------------------------------------------------
def extract_subject_mask(source, output_dir):
    """Extract subject mask via BiRefNet."""
    log(output_dir, "Extracting subject mask (BiRefNet)...")
    mask, mask_info = build_mask(source, affect="subject", exclude="", output_dir=output_dir)
    log(output_dir, f"Subject mask coverage: {mask_info['coverage_pct']:.1f}%")
    return mask


# ---------------------------------------------------------------------------
# Step 2: Get face mask via MediaPipe body-segment
# ---------------------------------------------------------------------------
def get_face_mask(img_array, output_dir):
    """Get face-skin mask using MediaPipe body segmentation."""
    log(output_dir, "Detecting face region (MediaPipe)...")
    try:
        spec = importlib.util.spec_from_file_location(
            "body_segment", os.path.join(script_dir, "body-segment.py")
        )
        bs_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs_mod)
        cat_mask = bs_mod.segment_body(img_array)
        # 0=bg, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
        face_mask = (cat_mask == 3).astype(np.uint8) * 255
        coverage = np.sum(face_mask > 0) / face_mask.size * 100
        log(output_dir, f"Face mask coverage: {coverage:.1f}%")
        return face_mask
    except Exception as e:
        log(output_dir, f"Face detection failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Step 3: Posterize into tonal layers
# ---------------------------------------------------------------------------
def posterize_to_layers(img, subject_mask_np, face_mask_np, num_layers, output_dir):
    """Convert image to grayscale, quantize to N tonal layers.

    Within the face region, use double the number of layers for finer detail.
    Returns list of binary masks (numpy bool arrays), one per layer,
    ordered from darkest (index 0) to lightest (index -1).
    """
    log(output_dir, f"Posterizing to {num_layers} layers...")
    gray = np.array(img.convert("L")).astype(np.float32)
    h, w = gray.shape

    # Only consider subject pixels for threshold calculation
    subject_bool = subject_mask_np > 127
    subject_pixels = gray[subject_bool]

    if len(subject_pixels) == 0:
        log(output_dir, "No subject pixels found, using full image", "WARN")
        subject_pixels = gray.flatten()

    # Percentile-based thresholds for even distribution
    percentiles = np.linspace(0, 100, num_layers + 1)[1:-1]
    thresholds = np.percentile(subject_pixels, percentiles)
    log(output_dir, f"Tonal thresholds: {[f'{t:.0f}' for t in thresholds]}")

    # Build body layer masks
    body_layers = []
    for i in range(num_layers):
        low = thresholds[i - 1] if i > 0 else 0
        high = thresholds[i] if i < len(thresholds) else 256
        layer = (gray >= low) & (gray < high) & subject_bool
        body_layers.append(layer)

    # Face: use more layers for finer detail
    face_layers = None
    if face_mask_np is not None:
        face_bool = face_mask_np > 127
        face_pixel_count = np.sum(face_bool)
        if face_pixel_count > 100:
            face_num = min(num_layers + 2, 6)  # extra layers for face
            face_percentiles = np.linspace(0, 100, face_num + 1)[1:-1]
            face_pixels = gray[face_bool]
            face_thresholds = np.percentile(face_pixels, face_percentiles)
            log(output_dir, f"Face detail: {face_num} layers, thresholds: {[f'{t:.0f}' for t in face_thresholds]}")

            face_layers = []
            for i in range(face_num):
                low = face_thresholds[i - 1] if i > 0 else 0
                high = face_thresholds[i] if i < len(face_thresholds) else 256
                layer = (gray >= low) & (gray < high) & face_bool
                face_layers.append(layer)

    # Merge: replace body layers within the face region with face layers
    if face_layers is not None:
        face_bool = face_mask_np > 127
        # Clear face region from body layers
        for bl in body_layers:
            bl[face_bool] = False

    return body_layers, face_layers, thresholds


# ---------------------------------------------------------------------------
# Paper texture generation
# ---------------------------------------------------------------------------
def generate_paper_texture(h, w, base_color, seed, short_edge):
    """Generate a paper fiber texture with a given base color.

    Creates directional noise that looks like paper fibers, modulating
    brightness by +/-10-15%.
    """
    rng = np.random.RandomState(seed)

    # Directional fiber noise: elongated in one direction
    # Generate at lower resolution and upscale for fiber-like appearance
    fiber_scale = max(4, int(short_edge * 0.005))
    small_h = max(1, h // fiber_scale)
    small_w = max(1, w // fiber_scale)

    # Horizontal fibers
    noise_h = rng.randn(small_h, small_w).astype(np.float32)
    # Stretch horizontally for fiber look
    noise_h = gaussian_filter(noise_h, sigma=[0.5, 2.0])

    # Vertical fibers (less prominent)
    noise_v = rng.randn(small_h, small_w).astype(np.float32)
    noise_v = gaussian_filter(noise_v, sigma=[2.0, 0.5])

    fiber = noise_h * 0.7 + noise_v * 0.3

    # Upscale to full resolution
    fiber = cv2.resize(fiber, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1]
    fmin, fmax = fiber.min(), fiber.max()
    if fmax - fmin > 1e-6:
        fiber = (fiber - fmin) / (fmax - fmin) * 2 - 1
    else:
        fiber = np.zeros((h, w), dtype=np.float32)

    # Fine grain noise for paper speckle
    speckle = rng.randn(h, w).astype(np.float32) * 0.3
    speckle = gaussian_filter(speckle, sigma=0.8)

    combined = fiber * 0.7 + speckle * 0.3

    # Modulate brightness by +/- 12%
    modulation = 1.0 + combined * 0.12

    # Apply to base color
    result = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        result[:, :, c] = base_color[c] * modulation

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Contour roughening — torn/cut paper edges
# ---------------------------------------------------------------------------
def roughen_contour_mask(mask_bool, roughness, short_edge, seed):
    """Displace contour edges with noise to simulate torn paper edges.

    Args:
        mask_bool: 2D boolean numpy array
        roughness: 0-1, controls displacement magnitude
        short_edge: image short edge in pixels
        seed: random seed

    Returns:
        New boolean mask with roughened edges.
    """
    if roughness <= 0:
        return mask_bool

    h, w = mask_bool.shape
    mask_uint8 = mask_bool.astype(np.uint8) * 255

    # Displacement magnitude: 0.5-1% of short edge, scaled by roughness
    max_disp = int(short_edge * 0.01 * roughness)
    max_disp = max(2, min(max_disp, int(short_edge * 0.015)))

    # Generate displacement noise field
    rng = np.random.RandomState(seed)
    # Low-frequency noise for organic feel (not pixel-level jitter)
    noise_scale = max(4, int(short_edge * 0.008))
    small_h = max(1, h // noise_scale)
    small_w = max(1, w // noise_scale)

    noise_x = rng.randn(small_h, small_w).astype(np.float32)
    noise_y = rng.randn(small_h, small_w).astype(np.float32)
    noise_x = gaussian_filter(noise_x, sigma=1.5)
    noise_y = gaussian_filter(noise_y, sigma=1.5)

    # Upscale
    noise_x = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_LINEAR)
    noise_y = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1]
    for arr in [noise_x, noise_y]:
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-6:
            arr[:] = (arr - mn) / (mx - mn) * 2 - 1

    # Scale to displacement pixels
    disp_x = (noise_x * max_disp).astype(np.float32)
    disp_y = (noise_y * max_disp).astype(np.float32)

    # Create coordinate maps for remapping
    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Only displace near edges — find edge band
    # Dilate and erode to find edge region
    kernel_size = max(3, max_disp * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    edge_band = ((dilated > 0) & (eroded == 0)).astype(np.float32)

    # Smooth the edge band for gradual transition
    edge_band = gaussian_filter(edge_band, sigma=max_disp * 0.5)

    # Apply displacement only in edge band
    new_x = map_x + disp_x * edge_band
    new_y = map_y + disp_y * edge_band

    # Remap
    roughened = cv2.remap(
        mask_uint8, new_x, new_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    return roughened > 127


# ---------------------------------------------------------------------------
# Drop shadow
# ---------------------------------------------------------------------------
def create_drop_shadow(mask_bool, shadow_offset, blur_radius, strength):
    """Create a drop shadow from a layer mask.

    Args:
        mask_bool: 2D boolean array (the layer casting the shadow)
        shadow_offset: (dx, dy) tuple in pixels
        blur_radius: gaussian blur sigma for shadow softness
        strength: 0-1, shadow opacity

    Returns:
        RGBA numpy array of the shadow (h, w, 4)
    """
    h, w = mask_bool.shape
    shadow = np.zeros((h, w), dtype=np.float32)

    # Shift the mask
    dx, dy = shadow_offset
    mask_uint8 = mask_bool.astype(np.float32)

    # Use affine transform for sub-pixel shifting
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(mask_uint8, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Blur
    shadow = gaussian_filter(shifted, sigma=blur_radius)

    # Normalize and apply strength
    if shadow.max() > 0:
        shadow = shadow / shadow.max() * strength

    return shadow


# ---------------------------------------------------------------------------
# Main compositing: build layered paper cutout
# ---------------------------------------------------------------------------
def build_paper_cutout(img, subject_mask_pil, body_layers, face_layers,
                       palette_name, num_layers, shadow_strength, edge_roughness,
                       seed, output_dir):
    """Composite all paper layers into final image."""
    palette = PALETTES[palette_name]
    bg_color = palette["bg"]
    layer_colors = palette["layers"]

    w, h = img.size
    short_edge = min(w, h)

    # Select colors for the requested number of layers
    # Evenly space through the palette color list
    num_body = len(body_layers)
    if num_body <= len(layer_colors):
        indices = np.linspace(0, len(layer_colors) - 1, num_body).astype(int)
        body_colors = [layer_colors[i] for i in indices]
    else:
        body_colors = layer_colors[:num_body]

    # Face layers get colors interpolated between the body layer colors
    face_colors = None
    if face_layers:
        num_face = len(face_layers)
        # Interpolate across the full color range for face
        face_colors = []
        for i in range(num_face):
            t = i / max(1, num_face - 1)
            idx_f = t * (len(layer_colors) - 1)
            lo = int(idx_f)
            hi = min(lo + 1, len(layer_colors) - 1)
            frac = idx_f - lo
            c = tuple(int(layer_colors[lo][j] * (1 - frac) + layer_colors[hi][j] * frac) for j in range(3))
            face_colors.append(c)

    # Shadow parameters — scaled to image size
    shadow_offset_px = max(2, int(short_edge * 0.004))
    shadow_offset = (shadow_offset_px, shadow_offset_px)
    shadow_blur = max(2, int(short_edge * 0.003))

    log(output_dir, f"Shadow: offset={shadow_offset_px}px, blur={shadow_blur}px, strength={shadow_strength}")
    log(output_dir, f"Edge roughness: {edge_roughness}, max displacement: ~{int(short_edge * 0.01 * edge_roughness)}px")

    # --- Generate background paper ---
    log(output_dir, "Generating background paper texture...")
    bg_paper = generate_paper_texture(h, w, bg_color, seed, short_edge)
    canvas = bg_paper.copy().astype(np.float32)

    # --- Build and stack body layers (darkest first = back) ---
    log(output_dir, f"Building {num_body} body layers...")
    for i, (layer_mask, color) in enumerate(zip(body_layers, body_colors)):
        layer_seed = seed + i + 1

        # Roughen edges
        rough_mask = roughen_contour_mask(layer_mask, edge_roughness, short_edge, layer_seed)

        # Drop shadow onto canvas (from this layer)
        if shadow_strength > 0:
            shadow = create_drop_shadow(rough_mask, shadow_offset, shadow_blur, shadow_strength)
            # Apply shadow: darken canvas
            for c in range(3):
                canvas[:, :, c] = canvas[:, :, c] * (1 - shadow * 0.7)

        # Generate paper texture for this layer
        paper = generate_paper_texture(h, w, color, layer_seed, short_edge)

        # Composite layer onto canvas
        mask_f = rough_mask.astype(np.float32)
        for c in range(3):
            canvas[:, :, c] = canvas[:, :, c] * (1 - mask_f) + paper[:, :, c].astype(np.float32) * mask_f

        log(output_dir, f"  Body layer {i+1}/{num_body}: color={color}, pixels={np.sum(rough_mask)}")

    # --- Build and stack face layers (extra detail) ---
    if face_layers and face_colors:
        log(output_dir, f"Building {len(face_layers)} face detail layers...")
        for i, (layer_mask, color) in enumerate(zip(face_layers, face_colors)):
            layer_seed = seed + num_body + i + 100

            rough_mask = roughen_contour_mask(layer_mask, edge_roughness * 0.7, short_edge, layer_seed)

            if shadow_strength > 0:
                # Smaller shadow for face detail layers
                face_shadow_offset = (max(1, shadow_offset_px // 2), max(1, shadow_offset_px // 2))
                shadow = create_drop_shadow(rough_mask, face_shadow_offset, max(1, shadow_blur // 2), shadow_strength * 0.6)
                for c in range(3):
                    canvas[:, :, c] = canvas[:, :, c] * (1 - shadow * 0.5)

            paper = generate_paper_texture(h, w, color, layer_seed, short_edge)

            mask_f = rough_mask.astype(np.float32)
            for c in range(3):
                canvas[:, :, c] = canvas[:, :, c] * (1 - mask_f) + paper[:, :, c].astype(np.float32) * mask_f

            log(output_dir, f"  Face layer {i+1}/{len(face_layers)}: color={color}")

    # Clip to valid range
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Paper Cutout — layered paper cutout art from portraits")
    parser.add_argument("--source", required=False, help="Input photo path")
    parser.add_argument("--palette", default="craft", choices=list(PALETTES.keys()),
                        help=f"Color palette (default: craft). Options: {', '.join(PALETTES.keys())}")
    parser.add_argument("--num-layers", type=int, default=4, choices=range(3, 7),
                        help="Number of tonal layers (3-6, default: 4)")
    parser.add_argument("--shadow-strength", type=float, default=0.6,
                        help="Drop shadow opacity (0-1, default: 0.6)")
    parser.add_argument("--edge-roughness", type=float, default=0.5,
                        help="Edge displacement roughness (0-1, default: 0.5)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None)
    parser.add_argument("--list-palettes", action="store_true")
    args = parser.parse_args()

    if args.list_palettes:
        print(f"\n{'Palette':<12} Description")
        print("=" * 60)
        for name, pal in PALETTES.items():
            print(f"  {name:<10} {pal['description']}")
        sys.exit(0)

    if not args.source:
        parser.error("--source is required (unless using --list-palettes)")

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Output directory
    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name}_{source_basename}_{timestamp}_cutout_{args.palette}_{seed % 100:02d}"
    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    timings = {}

    log(output_dir, "=" * 60)
    log(output_dir, f"Paper Cutout — {source_basename}")
    log(output_dir, f"Palette: {args.palette}, Layers: {args.num_layers}, Shadow: {args.shadow_strength}, Roughness: {args.edge_roughness}, Seed: {seed}")
    log(output_dir, "=" * 60)

    # ========================================================================
    # Step 1: Load image + extract subject mask
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 1/5: Extract subject mask ---")

    img = Image.open(source).convert("RGB")
    w, h = img.size
    short_edge = min(w, h)
    log(output_dir, f"Image size: {w}x{h}, short edge: {short_edge}px")

    subject_mask_pil = extract_subject_mask(source, output_dir)
    if subject_mask_pil.size != img.size:
        subject_mask_pil = subject_mask_pil.resize(img.size, Image.LANCZOS)
    subject_mask_np = np.array(subject_mask_pil)

    subject_mask_pil.save(os.path.join(output_dir, "1_subject_mask.png"))
    timings["mask"] = time.time() - t0
    log(output_dir, f"Step 1 done ({timings['mask']:.1f}s)")

    # ========================================================================
    # Step 2: Detect face region
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 2/5: Detect face region ---")

    img_array = np.array(img)
    face_mask_np = get_face_mask(img_array, output_dir)

    if face_mask_np is not None:
        Image.fromarray(face_mask_np).save(os.path.join(output_dir, "2_face_mask.png"))

    timings["face"] = time.time() - t0
    log(output_dir, f"Step 2 done ({timings['face']:.1f}s)")

    # ========================================================================
    # Step 3: Posterize into tonal layers
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 3/5: Posterize into tonal layers ---")

    body_layers, face_layers, thresholds = posterize_to_layers(
        img, subject_mask_np, face_mask_np, args.num_layers, output_dir
    )

    # Debug: save individual layer masks
    for i, layer in enumerate(body_layers):
        layer_img = Image.fromarray((layer * 255).astype(np.uint8))
        layer_img.save(os.path.join(output_dir, f"3_body_layer_{i}.png"))
    if face_layers:
        for i, layer in enumerate(face_layers):
            layer_img = Image.fromarray((layer * 255).astype(np.uint8))
            layer_img.save(os.path.join(output_dir, f"3_face_layer_{i}.png"))

    timings["posterize"] = time.time() - t0
    log(output_dir, f"Step 3 done ({timings['posterize']:.1f}s)")

    # ========================================================================
    # Step 4: Build paper cutout composite
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 4/5: Compositing paper layers ---")

    result = build_paper_cutout(
        img, subject_mask_pil, body_layers, face_layers,
        args.palette, args.num_layers, args.shadow_strength, args.edge_roughness,
        seed, output_dir
    )

    result_path = os.path.join(output_dir, "4_paper_cutout.jpg")
    result.save(result_path, "JPEG", quality=95)

    timings["composite"] = time.time() - t0
    log(output_dir, f"Step 4 done ({timings['composite']:.1f}s)")

    # ========================================================================
    # Step 5: Output + push
    # ========================================================================
    t0 = time.time()
    log(output_dir, "--- Step 5/5: Output ---")

    # Copy to finals
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared")
    finals_dir = os.path.join(local_out, "finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_name = os.path.basename(output_dir) + ".jpg"
    finals_dest = os.path.join(finals_dir, finals_name)
    with open(result_path, "rb") as f_in:
        with open(finals_dest, "wb") as f_out:
            f_out.write(f_in.read())
    log(output_dir, f"Final copied to: {finals_dest}")

    # Side-by-side comparison
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(img, (0, 0))
    comparison.paste(result, (w, 0))
    comp_path = os.path.join(finals_dir, f"{source_basename}_paper-cutout_{args.palette}_comparison.jpg")
    comparison.save(comp_path, quality=92)
    log(output_dir, f"Comparison: {comp_path}")

    # Push to phone
    try:
        from notify import push_image
        src_name = os.path.splitext(os.path.basename(args.source))[0]
        push_image(finals_dest, title=f"Paper Cutout — {src_name}",
                   body=f"{args.palette} palette, {args.num_layers} layers")
        log(output_dir, "Pushed to phone")
    except Exception as e:
        log(output_dir, f"Push notification failed: {e}", "WARN")

    # Copy script for reproducibility
    try:
        shutil.copy2(os.path.abspath(__file__),
                      os.path.join(output_dir, f"workflow_script_{os.path.basename(__file__)}"))
    except Exception:
        pass

    timings["output"] = time.time() - t0
    log(output_dir, f"Step 5 done ({timings['output']:.1f}s)")

    # --- Summary ---
    total = sum(timings.values())

    print(f"""
============================================================
  PAPER CUTOUT SUMMARY
============================================================
  Source:          {source}
  Palette:         {args.palette}
  Body layers:     {len(body_layers)}
  Face layers:     {len(face_layers) if face_layers else 0}
  Shadow strength: {args.shadow_strength}
  Edge roughness:  {args.edge_roughness}
  Seed:            {seed}

  Step Timings:
    1. Subject mask          {timings.get('mask', 0):>8.1f}s
    2. Face detection         {timings.get('face', 0):>8.1f}s
    3. Posterize              {timings.get('posterize', 0):>8.1f}s
    4. Composite              {timings.get('composite', 0):>8.1f}s
    5. Output                 {timings.get('output', 0):>8.1f}s
    TOTAL                    {total:>8.1f}s

  Output:
    Working dir:   {output_dir}
    Finals:        {finals_dest}
    Comparison:    {comp_path}
============================================================""")


if __name__ == "__main__":
    main()
