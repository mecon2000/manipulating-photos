#!/home/rong/openclaw-venv/bin/python3
"""
Ink Dissolution — Frequency-band replacement for photo-to-painting gradient.

Decomposes a portrait into frequency bands via Laplacian pyramid:
  - Low frequency = form, lighting, color masses (kept photographic)
  - High frequency = texture, detail (replaced with ink-wash / watercolor / canvas)

The replacement graduates by body region:
  - Face stays sharp (photographic)
  - Body dissolves into the chosen medium
  - Optionally depth-based: near=photo, far=painting

The result looks impossible: too real to be a painting, too painterly to be a photo.

Usage:
    python ink-dissolution.py --source photo.jpg
    python ink-dissolution.py --source photo.jpg --medium ink-wash --dissolve-strength 0.8
    python ink-dissolution.py --source photo.jpg --medium watercolor --face-preserve 0.9
    python ink-dissolution.py --list-media
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
import shutil
import threading
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageEnhance
from scipy.ndimage import gaussian_filter
import requests

# Use shared masking module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Media (texture replacement types)
# ---------------------------------------------------------------------------
MEDIA = {
    "ink-wash": {
        "description": "Japanese sumi-e ink wash — soft gradients, paper bleed",
        "texture_freq": 0.03,       # low-freq texture pattern
        "grain_amount": 0.15,       # paper grain overlay
        "bleed_radius": 0.02,       # edge bleed as fraction of image size
        "color_shift": (0, -5, 10), # slight warm-to-cool shift in replaced bands
        "paper_tone": (245, 238, 225),  # warm rice paper
    },
    "watercolor": {
        "description": "Wet watercolor — pigment bloom, soft edges, paper texture",
        "texture_freq": 0.05,
        "grain_amount": 0.20,
        "bleed_radius": 0.03,
        "color_shift": (5, 0, -5),
        "paper_tone": (250, 245, 235),
    },
    "canvas": {
        "description": "Oil on canvas — woven texture, thick impasto feel",
        "texture_freq": 0.08,
        "grain_amount": 0.25,
        "bleed_radius": 0.01,
        "color_shift": (3, 2, -3),
        "paper_tone": (235, 228, 215),
    },
    "charcoal": {
        "description": "Charcoal on paper — heavy grain, smudged transitions",
        "texture_freq": 0.04,
        "grain_amount": 0.35,
        "bleed_radius": 0.015,
        "color_shift": (-5, -5, -5),
        "paper_tone": (240, 235, 225),
    },
    "graphite": {
        "description": "Pencil graphite — fine grain, clean lines, subtle texture",
        "texture_freq": 0.06,
        "grain_amount": 0.12,
        "bleed_radius": 0.005,
        "color_shift": (-3, -3, -2),
        "paper_tone": (248, 245, 240),
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def log(level, msg):
    ts = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        print(f"[{ts}] [{level}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Laplacian Pyramid
# ---------------------------------------------------------------------------
def build_laplacian_pyramid(img_array, levels=5):
    """Build a Laplacian pyramid from an image array (float64, 0-255).

    Returns list of (detail_band, blurred) pairs from finest to coarsest,
    plus the final low-frequency residual.
    """
    pyramid = []
    current = img_array.astype(np.float64)

    for i in range(levels):
        # Blur radius increases with each level
        sigma = 2 ** (i + 1)  # 2, 4, 8, 16, 32
        blurred = np.zeros_like(current)
        for c in range(current.shape[2]):
            blurred[:, :, c] = gaussian_filter(current[:, :, c], sigma=sigma)

        # Detail band = current - blurred
        detail = current - blurred
        pyramid.append(detail)
        current = blurred

    # The residual is the lowest frequency component
    return pyramid, current


def reconstruct_from_pyramid(pyramid, residual):
    """Reconstruct image from Laplacian pyramid + low-freq residual."""
    result = residual.copy()
    for detail in reversed(pyramid):
        result = result + detail
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------
def generate_paper_grain(shape, grain_amount, seed=None):
    """Generate paper grain texture (multiplicative noise).

    Returns array of shape (H, W) with values centered around 1.0.
    grain_amount controls the deviation (0 = flat, 0.5 = heavy grain).
    """
    rng = np.random.RandomState(seed)

    # Multi-scale noise for realistic paper
    h, w = shape[:2]
    grain = np.ones((h, w), dtype=np.float64)

    # Fine grain
    fine = rng.randn(h, w) * grain_amount * 0.5
    fine = gaussian_filter(fine, sigma=1.0)
    grain += fine

    # Medium grain (fiber structure)
    med = rng.randn(h // 4 + 1, w // 4 + 1) * grain_amount * 0.3
    med = np.repeat(np.repeat(med, 4, axis=0), 4, axis=1)[:h, :w]
    med = gaussian_filter(med, sigma=3.0)
    grain += med

    # Coarse grain (paper irregularity)
    coarse = rng.randn(h // 16 + 1, w // 16 + 1) * grain_amount * 0.2
    coarse = np.repeat(np.repeat(coarse, 16, axis=0), 16, axis=1)[:h, :w]
    coarse = gaussian_filter(coarse, sigma=8.0)
    grain += coarse

    return grain


def generate_ink_texture(shape, medium_params, seed=None):
    """Generate a texture pattern that replaces the high-frequency detail.

    Returns an array of shape (H, W, 3) representing the replacement texture
    (values centered around 0, to be added in place of original detail).
    """
    rng = np.random.RandomState(seed)
    h, w = shape[:2]
    freq = medium_params["texture_freq"]

    # Base texture: directional noise (simulates brush/wash direction)
    # Horizontal bias for ink wash, more isotropic for canvas
    tex = rng.randn(h, w)
    sigma_x = max(1, int(w * freq))
    sigma_y = max(1, int(h * freq * 0.5))  # elongated horizontally
    tex = gaussian_filter(tex, sigma=[sigma_y, sigma_x])

    # Normalize to reasonable range
    tex = tex / (tex.std() + 1e-8) * 15  # subtle texture, ~15 units amplitude

    # Make 3-channel with slight color variation
    shift = medium_params["color_shift"]
    texture_3ch = np.stack([
        tex + shift[0],
        tex + shift[1],
        tex + shift[2],
    ], axis=-1)

    return texture_3ch


def generate_bleed_mask(mask_array, bleed_radius_frac, shape):
    """Create a soft bleed effect at mask boundaries.

    Simulates how ink/watercolor bleeds at edges. Returns a modified mask
    with softer, irregular boundaries.
    """
    if bleed_radius_frac <= 0:
        return mask_array

    h, w = shape[:2]
    bleed_px = max(1, int(min(h, w) * bleed_radius_frac))

    # Gaussian blur the mask edges
    bleeded = gaussian_filter(mask_array.astype(np.float64), sigma=bleed_px)

    # Add some noise to the bleed for irregularity
    noise = np.random.randn(h, w) * 0.1
    noise = gaussian_filter(noise, sigma=bleed_px * 0.5)
    bleeded = bleeded + noise

    return np.clip(bleeded, 0, 1)


# ---------------------------------------------------------------------------
# Dissolution map (controls how much each pixel dissolves)
# ---------------------------------------------------------------------------
def build_dissolution_map(img_pil, subject_mask, face_preserve=0.85,
                          body_dissolve=0.9, use_depth=False, fade_multiplier=3.5):
    """Build a per-pixel dissolution map (0 = keep photo, 1 = full dissolution).

    Radial gradient from face center:
    - Face: zero dissolution (sharp)
    - Near face (shoulders, hair): low dissolution
    - Far body/limbs: increasing dissolution
    - Background: full dissolution

    The gradient is distance-from-face-center, normalized so that the face
    radius = 0 dissolution and image corners = full dissolution.
    """
    w, h = img_pil.size
    short_edge = min(w, h)

    # Get body segments via MediaPipe
    try:
        import importlib.util
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bs_path = os.path.join(script_dir, "body-segment.py")
        spec = importlib.util.spec_from_file_location("body_segment", bs_path)
        bs_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs_mod)

        img_array_uint8 = np.array(img_pil)
        cat_mask = bs_mod.segment_body(img_array_uint8)

        face_mask = (cat_mask == 3).astype(np.float64)
        body_skin = (cat_mask == 2).astype(np.float64)
        hair_mask = (cat_mask == 1).astype(np.float64)
        clothes = (cat_mask == 4).astype(np.float64)
        others = (cat_mask == 5).astype(np.float64)

    except Exception as e:
        log("WARN", f"Body segmentation failed ({e}), using uniform gradient")
        face_mask = np.zeros((h, w), dtype=np.float64)
        body_skin = np.zeros((h, w), dtype=np.float64)
        hair_mask = np.zeros((h, w), dtype=np.float64)
        clothes = np.zeros((h, w), dtype=np.float64)
        others = np.zeros((h, w), dtype=np.float64)

    # Find face center
    face_pixels = np.where(face_mask > 0.5)
    if len(face_pixels[0]) > 0:
        face_cy = face_pixels[0].mean()
        face_cx = face_pixels[1].mean()
        # Face "radius" — approximate from bounding box
        face_h = face_pixels[0].max() - face_pixels[0].min()
        face_w = face_pixels[1].max() - face_pixels[1].min()
        face_radius = max(face_h, face_w) * 0.6  # slightly larger than face bbox
        log("INFO", f"Face center: ({face_cx:.0f}, {face_cy:.0f}), radius: {face_radius:.0f}px")
    else:
        # No face found — use image center, top third
        face_cy = h * 0.3
        face_cx = w * 0.5
        face_radius = short_edge * 0.08
        log("WARN", "No face detected, using top-center as dissolution origin")

    # Build radial distance map from face center
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - face_cx)**2 + (yy - face_cy)**2)

    # Normalize: 0 at face center, 1 at ~2x face radius from center
    # Beyond that, clamp to 1
    fade_distance = max(face_radius * fade_multiplier, short_edge * 0.4)  # how far until full dissolution
    radial = np.clip((dist - face_radius) / (fade_distance - face_radius), 0, 1)

    # Face mask gets zero dissolution regardless of distance
    radial[face_mask > 0.5] = 0

    # Background gets full dissolution
    subject_np = np.array(subject_mask.convert("L")).astype(np.float64) / 255.0
    subject_coverage = subject_np.mean()

    # Use MediaPipe segments as subject if BiRefNet is bad
    if subject_coverage < 0.10:
        log("WARN", f"BiRefNet coverage too low ({subject_coverage:.1%}), using MediaPipe segments")
        mp_subject = np.clip(face_mask + body_skin + hair_mask + clothes + others, 0, 1)
        is_bg = (mp_subject < 0.5)
    else:
        is_bg = (subject_np < 0.5)

    # BG: full dissolution
    radial[is_bg] = 1.0

    # Scale by overall strength
    dissolution = radial * body_dissolve

    # Smooth transitions
    dissolution = gaussian_filter(dissolution, sigma=max(2, short_edge * 0.015))

    log("INFO", f"Dissolution map: radial from face, bg=full, fade_dist={fade_distance:.0f}px")
    return np.clip(dissolution, 0, 1)


# ---------------------------------------------------------------------------
# Core: Ink Dissolution
# ---------------------------------------------------------------------------
def generate_medium_overlay(shape, medium_params, seed=None):
    """Generate a visible medium texture overlay (canvas weave, paper fiber, ink wash).

    Returns (H, W) float array, values roughly -1 to 1, representing the medium's
    physical texture. This gets composited on top of the smoothed image.
    """
    rng = np.random.RandomState(seed)
    h, w = shape[:2]
    short_edge = min(h, w)

    medium_desc = medium_params.get("description", "")

    if "canvas" in medium_desc.lower() or "oil" in medium_desc.lower():
        # Canvas weave: two directional patterns (warp + weft)
        # Horizontal threads
        freq_h = max(4, int(short_edge * 0.004))  # thread spacing ~0.4% of image
        horiz = np.sin(np.linspace(0, h / freq_h * 2 * np.pi, h))[:, np.newaxis]
        horiz = np.broadcast_to(horiz, (h, w)).copy()
        # Vertical threads
        freq_v = max(4, int(short_edge * 0.004))
        vert = np.sin(np.linspace(0, w / freq_v * 2 * np.pi, w))[np.newaxis, :]
        vert = np.broadcast_to(vert, (h, w)).copy()
        # Combine
        texture = (horiz * 0.5 + vert * 0.5)
        # Add irregularity
        noise = rng.randn(h // 8 + 1, w // 8 + 1) * 0.3
        noise = np.repeat(np.repeat(noise, 8, axis=0), 8, axis=1)[:h, :w]
        noise = gaussian_filter(noise, sigma=4)
        texture += noise

    elif "charcoal" in medium_desc.lower():
        # Charcoal: heavy directional grain (mostly vertical, like paper tooth)
        grain = rng.randn(h, w)
        # Strong vertical blur, weak horizontal = directional grain
        texture = gaussian_filter(grain, sigma=[1.0, 4.0])
        texture = texture / (texture.std() + 1e-8)
        # Coarse clumps
        coarse = rng.randn(h // 6 + 1, w // 6 + 1) * 0.4
        coarse = np.repeat(np.repeat(coarse, 6, axis=0), 6, axis=1)[:h, :w]
        coarse = gaussian_filter(coarse, sigma=3)
        texture += coarse

    elif "watercolor" in medium_desc.lower():
        # Watercolor: soft blooms, pigment pooling at edges
        # Large soft blobs
        blobs = rng.randn(h // 16 + 1, w // 16 + 1)
        blobs = np.repeat(np.repeat(blobs, 16, axis=0), 16, axis=1)[:h, :w]
        blobs = gaussian_filter(blobs, sigma=max(8, short_edge * 0.02))
        # Fine paper grain
        fine = rng.randn(h, w) * 0.3
        fine = gaussian_filter(fine, sigma=1.5)
        texture = blobs * 0.7 + fine * 0.3

    elif "graphite" in medium_desc.lower():
        # Graphite: fine, uniform grain with slight directionality
        grain = rng.randn(h, w)
        texture = gaussian_filter(grain, sigma=[1.5, 2.5])
        texture = texture / (texture.std() + 1e-8) * 0.7

    else:
        # Ink wash default: paper fiber + wash bleed
        # Paper fiber (directional)
        fiber = rng.randn(h, w)
        texture = gaussian_filter(fiber, sigma=[2.0, 0.8])
        # Wash blooms (large soft variations)
        bloom = rng.randn(h // 12 + 1, w // 12 + 1) * 0.5
        bloom = np.repeat(np.repeat(bloom, 12, axis=0), 12, axis=1)[:h, :w]
        bloom = gaussian_filter(bloom, sigma=max(6, short_edge * 0.015))
        texture = texture * 0.6 + bloom * 0.4

    # Normalize to [-1, 1] range
    texture = texture / (np.abs(texture).max() + 1e-8)
    return texture


def ink_dissolve(img_pil, subject_mask, medium="ink-wash",
                 dissolve_strength=0.85, face_preserve=0.85,
                 num_levels=5, seed=None, fade_multiplier=3.5,
                 stages=None):
    """Apply ink dissolution effect.

    The approach:
    1. Build dissolution map (face=preserve, body=dissolve, bg=none)
    2. SUPPRESS high-frequency detail bands (make dissolved areas smooth)
    3. OVERLAY medium texture (canvas/paper/ink grain) on smoothed areas
    4. Result: face is photographic, body is smooth + medium texture = painting

    Args:
        img_pil: PIL Image (RGB)
        subject_mask: PIL Image (L mode, white=subject)
        medium: one of MEDIA keys
        dissolve_strength: overall strength (0-1)
        face_preserve: how much to preserve face detail (0-1)
        num_levels: Laplacian pyramid levels
        seed: random seed for reproducibility

    Returns:
        PIL Image with dissolution effect applied
    """
    medium_params = MEDIA[medium]
    img_array = np.array(img_pil).astype(np.float64)
    h, w = img_array.shape[:2]
    short_edge = min(h, w)

    log("INFO", f"Building Laplacian pyramid ({num_levels} levels)...")
    pyramid, residual = build_laplacian_pyramid(img_array, levels=num_levels)

    # Build dissolution map (per-pixel, body-segment-aware)
    log("INFO", "Building dissolution map from body segments...")
    dissolution = build_dissolution_map(
        img_pil, subject_mask,
        face_preserve=face_preserve,
        body_dissolve=dissolve_strength,
        fade_multiplier=fade_multiplier,
    )
    dissolution_3ch = dissolution[:, :, np.newaxis]

    # --- STEP 1: Suppress detail bands (make dissolved areas smooth) ---
    log("INFO", "Suppressing detail in dissolved regions...")
    new_pyramid = []
    for i, detail_band in enumerate(pyramid):
        # Finest detail (i=0) gets suppressed MOST — it's skin pores, noise
        # Coarsest detail (i=num_levels-1) gets suppressed LEAST — it's form
        # suppression: 1.0 for finest → 0.5 for coarsest (aggressive)
        suppress = 1.0 - (i / max(1, num_levels - 1)) * 0.5

        # Where dissolution is high, kill the detail band
        attenuation = 1.0 - dissolution_3ch * suppress
        new_band = detail_band * attenuation
        new_pyramid.append(new_band)

    # Reconstruct the smoothed image
    log("INFO", "Reconstructing smoothed image...")
    smoothed = residual.copy()
    for detail in reversed(new_pyramid):
        smoothed = smoothed + detail

    # --- STEP 2: Slight color simplification in dissolved areas ---
    # Reduce color variation (shift toward fewer, flatter tones)
    # This mimics how paint has more uniform color than skin
    log("INFO", "Simplifying color in dissolved areas...")
    # Bilateral-like simplification: blur color, blend based on dissolution
    color_smooth = np.zeros_like(smoothed)
    blur_sigma = max(5, short_edge * 0.025)
    for c in range(3):
        color_smooth[:, :, c] = gaussian_filter(smoothed[:, :, c], sigma=blur_sigma)
    # Blend: more dissolved = more color-simplified
    color_blend = dissolution_3ch * 0.55  # noticeable — flatten color toward paint-like uniformity
    smoothed = smoothed * (1.0 - color_blend) + color_smooth * color_blend

    # --- STEP 3: Overlay medium texture ---
    log("INFO", f"Generating and overlaying {medium} texture...")
    medium_tex = generate_medium_overlay(
        img_array.shape, medium_params,
        seed=seed
    )

    # Texture intensity scales with dissolution AND luminance
    # Darker areas show more texture (like real paper/canvas)
    luminance = np.mean(smoothed, axis=2) / 255.0  # 0-1
    # Texture more visible in midtones, less in deep shadows and bright highlights
    lum_factor = 1.0 - np.abs(luminance - 0.5) * 1.2  # peaks at mid-gray
    lum_factor = np.clip(lum_factor, 0.3, 1.0)

    tex_intensity = medium_params["grain_amount"] * 250  # ~30-90 pixel value swing — visible texture
    tex_layer = medium_tex * tex_intensity * dissolution * lum_factor

    # Apply texture (additive, per-channel with slight color variation)
    color_shift = np.array(medium_params["color_shift"])
    for c in range(3):
        smoothed[:, :, c] += tex_layer + color_shift[c] * dissolution * 0.5

    # --- STEP 4: Optional edge darkening at dissolution boundary ---
    # Where dissolution gradient is steep, darken slightly (like ink bleeding)
    grad_y = np.gradient(dissolution, axis=0)
    grad_x = np.gradient(dissolution, axis=1)
    edge_strength = np.sqrt(grad_y**2 + grad_x**2)
    edge_strength = edge_strength / (edge_strength.max() + 1e-8)
    # Only darken where there's actual edge
    edge_darken = edge_strength * 0.15  # subtle
    smoothed = smoothed * (1.0 - edge_darken[:, :, np.newaxis])

    result = np.clip(smoothed, 0, 255).astype(np.uint8)
    if stages is not None:
        # 0-1 → 0-255 grayscale visualisations for the stack
        stages["dissolution_map"] = Image.fromarray(
            (np.clip(dissolution, 0, 1) * 255).astype(np.uint8)).convert("RGB")
        # Pre-overlay smoothed (just before texture) approximated by re-running
        # would double cost — skip; we have the final result + map + texture.
        tex_vis = (np.clip(medium_tex, -1, 1) + 1) * 127.5
        stages["medium_texture"] = Image.fromarray(tex_vis.astype(np.uint8)).convert("RGB")
    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------
def extract_subject_mask(img_pil):
    """Extract subject mask using BiRefNet."""
    try:
        mask, _ = build_mask(img_pil, affect="subject")
        return mask
    except Exception as e:
        log("ERROR", f"Mask extraction failed: {e}")
        # Return full-image mask
        return Image.new("L", img_pil.size, 255)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    """Run the ink dissolution pipeline."""

    # Load source
    log("INFO", f"Loading {args.source}")
    img = Image.open(args.source).convert("RGB")
    w, h = img.size
    log("INFO", f"Image size: {w}x{h}")

    # Extract subject mask
    log("INFO", "Extracting subject mask...")
    mask = extract_subject_mask(img)

    # Run dissolution
    log("INFO", f"Applying {args.medium} dissolution (strength={args.dissolve_strength}, face_preserve={args.face_preserve})...")
    stages = {} if getattr(args, "save_stack", False) else None
    result = ink_dissolve(
        img, mask,
        medium=args.medium,
        dissolve_strength=args.dissolve_strength,
        face_preserve=args.face_preserve,
        num_levels=args.levels,
        seed=args.seed,
        fade_multiplier=args.fade_distance,
        stages=stages,
    )

    # Output
    out_dir = args.local_output_dir
    os.makedirs(out_dir, exist_ok=True)

    src_name = os.path.splitext(os.path.basename(args.source))[0]
    out_name = f"{src_name}_ink-dissolution_{args.medium}.jpg"
    out_path = os.path.join(out_dir, out_name)

    result.save(out_path, quality=95)
    log("INFO", f"Saved: {out_path}")

    # Copy to finals
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_path = os.path.join(finals_dir, out_name)
    result.save(finals_path, quality=95)
    log("INFO", f"Finals: {finals_path}")

    # Also save side-by-side comparison
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(img, (0, 0))
    comparison.paste(result, (w, 0))
    comp_path = os.path.join(finals_dir, f"{src_name}_ink-dissolution_{args.medium}_comparison.jpg")
    comparison.save(comp_path, quality=92)
    log("INFO", f"Comparison: {comp_path}")

    # --save-stack: write multi-page TIFF
    if getattr(args, "save_stack", False):
        try:
            from _layered_tiff import save_stack
            layers = [
                ("00_original", img),
                ("01_subject_mask", mask.convert("RGB")),
            ]
            if stages:
                if "dissolution_map" in stages:
                    layers.append(("02_dissolution_map", stages["dissolution_map"]))
                if "medium_texture" in stages:
                    layers.append(("03_medium_texture", stages["medium_texture"]))
            layers.append(("99_final", result))
            stack_path = os.path.join(finals_dir, f"{src_name}_ink-dissolution_{args.medium}__stack.tif")
            save_stack(stack_path, layers)
            log("INFO", f"Stack: {stack_path} ({len(layers)} layers)")
        except Exception as e:
            log("WARN", f"save-stack failed: {e}")

    # Push to phone
    try:
        from notify import push_image
        push_image(finals_path, title=f"Ink Dissolution — {src_name}", body=f"{args.medium}")
        log("INFO", "Pushed to phone")
    except Exception as e:
        log("WARN", f"Push notification failed: {e}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ink Dissolution — frequency-band photo-to-painting")
    parser.add_argument("--source", required=True, help="Source photo path")
    parser.add_argument("--medium", default="ink-wash", choices=list(MEDIA.keys()),
                        help="Dissolution medium (default: ink-wash)")
    parser.add_argument("--dissolve-strength", type=float, default=0.85,
                        help="Overall dissolution strength 0-1 (default: 0.85)")
    parser.add_argument("--face-preserve", type=float, default=0.85,
                        help="How much to preserve face detail 0-1 (default: 0.85)")
    parser.add_argument("--fade-distance", type=float, default=3.5,
                        help="Fade distance multiplier — how many face-radii until full dissolution "
                             "(default: 3.5, higher = more body preserved)")
    parser.add_argument("--levels", type=int, default=5,
                        help="Laplacian pyramid levels (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-to", default="local", choices=["local", "gdrive", "both"])
    parser.add_argument("--local-output-dir",
                        default=os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"))
    parser.add_argument("--list-media", action="store_true",
                        help="List available dissolution media")
    parser.add_argument("--save-stack", action="store_true",
                        help="export pipeline stages as a multi-page TIFF")

    args = parser.parse_args()

    if args.list_media:
        for name, params in MEDIA.items():
            print(f"  {name:15s} — {params['description']}")
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
