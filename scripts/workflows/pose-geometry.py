#!/home/rong/openclaw-venv/bin/python3
"""
Pose Geometry — Geometric Art from Photo Silhouettes

Extracts the pose/silhouette from a photo and reconstructs it as geometric art,
then blends it back with the original for an "art gallery" look.

Presets:
  wireframe  — Clean edges as white/colored lines on dark background
  lowpoly    — Delaunay triangulation with sampled colors
  blocks     — Grid mosaic of rectangular blocks with average color
  contour    — Topographic-style contour lines at multiple brightness levels

Usage:
    ./pose-geometry.py --source photo.jpg --geometry lowpoly
    ./pose-geometry.py --source photo.jpg --geometry wireframe --line-color "#FF6633"
    ./pose-geometry.py --source photo.jpg --geometry blocks --block-size 25 --blend-mode alpha --blend-opacity 0.5
    ./pose-geometry.py --list-presets
"""

import os
import sys

# Ensure masking.py (sibling) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask, add_affect_args

import re
import json
import base64
import random
import shutil
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
import requests
from PIL import Image, ImageFilter, ImageStat, ImageDraw, ImageEnhance
from scipy.ndimage import sobel
from scipy.spatial import Delaunay

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEOMETRY_PRESETS = {
    "wireframe": "Clean edge lines (white or colored) on darkened background — architectural sketch feel",
    "lowpoly":   "Delaunay triangulation with sampled colors — faceted crystal/polygon portrait",
    "blocks":    "Rectangular grid mosaic — geometric pixelation with average-color blocks",
    "contour":   "Topographic contour lines at multiple brightness levels — elevation map portrait",
    "crystal":   "Edge-aware Delaunay — dense triangles at contours that 'shatter' along edges, sparse in flat areas",
}

BLEND_MODES = ["overlay", "multiply", "screen", "alpha"]

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
        with open(log_path, "a") as f:
            f.write(formatted + "\n")


# ---------------------------------------------------------------------------
# Quality Gate Utilities
# ---------------------------------------------------------------------------
def check_image_quality(img, label, output_dir):
    """Check if an image is degenerate (black, white, flat, zero-entropy)."""
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    entropy = gray.entropy()

    reasons = []
    if brightness < 10:
        reasons.append(f"nearly black (brightness={brightness:.1f})")
    elif brightness > 245:
        reasons.append(f"nearly white (brightness={brightness:.1f})")
    if contrast < 5:
        reasons.append(f"flat/uniform (contrast={contrast:.1f})")
    if entropy < 1.0:
        reasons.append(f"zero-entropy (entropy={entropy:.2f})")

    ok = len(reasons) == 0
    result = {
        "ok": ok,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "entropy": round(entropy, 2),
        "reason": "; ".join(reasons) if reasons else None,
    }
    if not ok:
        log(output_dir, f"QUALITY FAIL [{label}]: {result['reason']}", "WARN")
    else:
        log(output_dir, f"Quality OK [{label}]: brightness={brightness:.1f} contrast={contrast:.1f} entropy={entropy:.2f}")
    return result


# ---------------------------------------------------------------------------
# Gemini Aesthetic Evaluation
# ---------------------------------------------------------------------------
def _img_to_b64(img, max_size=1024):
    """Downscale and encode image to base64 JPEG."""
    img_resized = img.copy()
    img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_EVAL_PROMPT = """\
You are an art director evaluating a photograph that has been processed with geometric art effects.
If you see TWO images, the first is the ORIGINAL and the second is the PROCESSED result — compare them.

Evaluate the PROCESSED image on these criteria:
1. Overall aesthetic appeal — does it look like intentional art or a broken filter?
2. Subject integrity: is the person's silhouette/form still recognizable and visually appealing?
3. Geometric style coherence: is the geometric effect consistent and well-applied?
4. Background-subject balance: does the subject stand out appropriately?
5. Color harmony: do the geometric colors work with the overall palette?
6. Blend quality: is the blending between geometric art and original photo smooth and natural?

Respond ONLY with valid JSON (no markdown fences):
{
  "score": <int 1-10>,
  "critique": "<2-3 sentences>",
  "issues": [<zero or more from: "subject_lost", "too_dark", "too_bright", "blend_harsh", \
"colors_clash", "geometry_messy", "low_contrast", "artifacts", "subject_unrecognizable", \
"too_subtle", "too_aggressive">],
  "adjustments": {
    "blend_opacity": <null or suggested float 0.1-1.0>,
    "try_different_preset": <true/false>,
    "try_different_blend": <true/false>,
    "suggestion": "<one sentence about what to change>"
  }
}"""


def evaluate_with_gemini(img, output_dir, original_img=None):
    """Evaluate using Google Gemini Vision API."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping Gemini evaluation")
        return None

    try:
        img_b64 = _img_to_b64(img)

        parts = [
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        ]
        if original_img is not None:
            orig_b64 = _img_to_b64(original_img)
            parts.insert(0, {"text": "ORIGINAL (before processing):"})
            parts.insert(1, {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}})
            parts.append({"text": "PROCESSED (after geometric art):\n\n" + _EVAL_PROMPT})
        else:
            parts.append({"text": _EVAL_PROMPT})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            log(output_dir, f"Gemini API error ({response.status_code}): {response.text[:200]}", "WARN")
            return None

        resp_json = response.json()
        candidates = resp_json.get("candidates", [])
        if not candidates:
            reason = resp_json.get("promptFeedback", {}).get("blockReason", "unknown")
            log(output_dir, f"Gemini returned no candidates (reason: {reason})", "WARN")
            return None

        finish_reason = candidates[0].get("finishReason", "")
        content = candidates[0].get("content", {})
        parts_out = content.get("parts", [])
        if not parts_out:
            log(output_dir, f"Gemini candidate has no content parts (finishReason: {finish_reason})", "WARN")
            return None

        raw = parts_out[0].get("text", "").strip()
        log(output_dir, f"Gemini raw response ({len(raw)} chars, finishReason={finish_reason}): {raw[:500]}")

        # Strip markdown fences
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                log(output_dir, f"Gemini response contains no JSON object: {raw[:200]}", "WARN")
                return None
            try:
                result = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as e:
                log(output_dir, f"Gemini JSON parse failed: {e}. Raw: {raw[start:start+300]}", "WARN")
                return None

        score = result.get("score", "?")
        critique = result.get("critique", "")
        issues = result.get("issues", [])
        log(output_dir, f"Gemini score: {score}/10 — {critique}")
        if issues:
            log(output_dir, f"Gemini issues: {', '.join(issues)}")
        adjustments = result.get("adjustments", {})
        if adjustments.get("suggestion"):
            log(output_dir, f"Gemini suggests: {adjustments['suggestion']}")

        return result
    except Exception as e:
        log(output_dir, f"Gemini evaluation failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Edge Detection (local, no API)
# ---------------------------------------------------------------------------
def detect_edges(img_gray_np, threshold=30):
    """Canny-like edge detection using Sobel filters.

    Args:
        img_gray_np: 2D numpy array (grayscale, float64)
        threshold: edge strength threshold (0-255 scale)

    Returns:
        2D boolean array of edge pixels
    """
    sx = sobel(img_gray_np, axis=1)
    sy = sobel(img_gray_np, axis=0)
    magnitude = np.hypot(sx, sy)
    # Normalize to 0-255 range
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    return magnitude > threshold


def get_edge_points(edge_mask):
    """Extract (y, x) coordinates of edge pixels."""
    ys, xs = np.where(edge_mask)
    return np.column_stack((ys, xs))


def get_dominant_color(img_np, mask_np):
    """Get the dominant color from the subject area by finding the most common hue bucket."""
    if mask_np.sum() == 0:
        return (255, 255, 255)

    # Sample pixels inside the mask
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return (255, 255, 255)

    # Subsample for performance
    if len(ys) > 5000:
        indices = np.random.choice(len(ys), 5000, replace=False)
        ys, xs = ys[indices], xs[indices]

    pixels = img_np[ys, xs]  # shape (N, 3)

    # Find the color with the highest saturation (most "colorful")
    # Convert to a simple hue-saturation approach
    r, g, b = pixels[:, 0].astype(float), pixels[:, 1].astype(float), pixels[:, 2].astype(float)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = (max_c - min_c) / (max_c + 1e-6)

    # Pick pixels with high saturation
    sat_threshold = np.percentile(saturation, 75)
    saturated = saturation >= sat_threshold
    if saturated.sum() > 0:
        avg_r = int(np.mean(pixels[saturated, 0]))
        avg_g = int(np.mean(pixels[saturated, 1]))
        avg_b = int(np.mean(pixels[saturated, 2]))
        return (avg_r, avg_g, avg_b)

    # Fallback: just average all
    return (int(np.mean(pixels[:, 0])), int(np.mean(pixels[:, 1])), int(np.mean(pixels[:, 2])))


# ---------------------------------------------------------------------------
# Geometry Presets
# ---------------------------------------------------------------------------
def generate_wireframe(img_orig, mask, output_dir, line_color="auto", edge_threshold=30, seed=None):
    """Generate wireframe art: clean edge lines on black background within the mask."""
    log(output_dir, "Generating wireframe geometry...")

    img_np = np.array(img_orig)
    mask_np = np.array(mask)
    gray = np.array(img_orig.convert("L"), dtype=np.float64)

    # Apply mask to grayscale before edge detection
    gray_masked = gray * (mask_np / 255.0)
    edges = detect_edges(gray_masked, threshold=edge_threshold)

    # Resolve line color
    if line_color == "auto":
        color = get_dominant_color(img_np, mask_np)
    else:
        # Parse hex color
        line_color = line_color.lstrip("#")
        color = tuple(int(line_color[i:i+2], 16) for i in (0, 2, 4))

    # Create the wireframe image (black background)
    w, h = img_orig.size
    wireframe = Image.new("RGB", (w, h), (0, 0, 0))
    wire_np = np.array(wireframe)

    # Draw edges as colored pixels, but only within the mask
    edge_within_mask = edges & (mask_np > 127)
    wire_np[edge_within_mask] = color

    # Slight thickening via dilation (draw on PIL for anti-aliased look)
    wireframe = Image.fromarray(wire_np)
    # Dilate edges slightly for visibility
    wireframe_l = wireframe.convert("L")
    wireframe_l = wireframe_l.filter(ImageFilter.MaxFilter(3))

    # Re-apply color to dilated edges
    result = Image.new("RGB", (w, h), (0, 0, 0))
    result_np = np.array(result)
    dilated_mask = np.array(wireframe_l) > 10
    result_np[dilated_mask] = color
    result = Image.fromarray(result_np)

    log(output_dir, f"Wireframe: {np.sum(edge_within_mask)} edge pixels, color={color}")
    return result


def generate_lowpoly(img_orig, mask, output_dir, num_points=800, seed=None):
    """Generate low-poly art: Delaunay triangulation colored from original."""
    log(output_dir, f"Generating lowpoly geometry (target ~{num_points} points)...")

    if seed is not None:
        np.random.seed(seed)

    img_np = np.array(img_orig)
    mask_np = np.array(mask)
    gray = np.array(img_orig.convert("L"), dtype=np.float64)
    w, h = img_orig.size

    # Get edge points within the mask
    gray_masked = gray * (mask_np / 255.0)
    edges = detect_edges(gray_masked, threshold=25)
    edge_pts = get_edge_points(edges & (mask_np > 127))

    if len(edge_pts) < 10:
        log(output_dir, "Too few edge points for lowpoly — returning blank", "WARN")
        return Image.new("RGB", (w, h), (0, 0, 0))

    # Subsample edge points
    if len(edge_pts) > num_points:
        indices = np.random.choice(len(edge_pts), num_points, replace=False)
        points = edge_pts[indices]
    else:
        points = edge_pts

    # Add corner/boundary points of the mask bounding box for complete coverage
    ys_mask, xs_mask = np.where(mask_np > 127)
    if len(ys_mask) > 0:
        y_min, y_max = ys_mask.min(), ys_mask.max()
        x_min, x_max = xs_mask.min(), xs_mask.max()
        # Add boundary points along the mask edge
        boundary_pts = []
        n_boundary = min(100, num_points // 4)
        # Sample points along the mask boundary
        mask_boundary = np.array(mask.filter(ImageFilter.FIND_EDGES))
        boundary_ys, boundary_xs = np.where(mask_boundary > 127)
        if len(boundary_ys) > n_boundary:
            b_idx = np.random.choice(len(boundary_ys), n_boundary, replace=False)
            boundary_pts = np.column_stack((boundary_ys[b_idx], boundary_xs[b_idx]))
            points = np.vstack([points, boundary_pts])

    # Add some random interior points for better triangulation
    n_interior = num_points // 5
    if len(ys_mask) > n_interior:
        int_idx = np.random.choice(len(ys_mask), n_interior, replace=False)
        interior_pts = np.column_stack((ys_mask[int_idx], xs_mask[int_idx]))
        points = np.vstack([points, interior_pts])

    # Deduplicate points (Delaunay can fail on duplicates)
    points = np.unique(points, axis=0)

    if len(points) < 3:
        log(output_dir, "Too few unique points for triangulation", "WARN")
        return Image.new("RGB", (w, h), (0, 0, 0))

    log(output_dir, f"Lowpoly: triangulating {len(points)} points...")

    try:
        tri = Delaunay(points)
    except Exception as e:
        log(output_dir, f"Delaunay triangulation failed: {e}", "WARN")
        return Image.new("RGB", (w, h), (0, 0, 0))

    # Draw triangles
    result = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(result)

    for simplex in tri.simplices:
        # Points are (y, x) — convert to (x, y) for PIL
        triangle_pts = [(int(points[i][1]), int(points[i][0])) for i in simplex]

        # Check if triangle is within the mask
        cy = int(np.mean([points[i][0] for i in simplex]))
        cx = int(np.mean([points[i][1] for i in simplex]))
        if 0 <= cy < h and 0 <= cx < w and mask_np[cy, cx] < 127:
            continue  # Skip triangles whose center is outside the mask

        # Sample average color from original image at triangle center region
        # Use a small region around the centroid
        y_coords = [int(points[i][0]) for i in simplex]
        x_coords = [int(points[i][1]) for i in simplex]
        cy_clamp = max(0, min(h - 1, cy))
        cx_clamp = max(0, min(w - 1, cx))

        # Sample a small patch around centroid
        patch_r = 3
        y_lo = max(0, cy_clamp - patch_r)
        y_hi = min(h, cy_clamp + patch_r + 1)
        x_lo = max(0, cx_clamp - patch_r)
        x_hi = min(w, cx_clamp + patch_r + 1)
        patch = img_np[y_lo:y_hi, x_lo:x_hi]
        if patch.size > 0:
            avg_color = tuple(int(c) for c in patch.mean(axis=(0, 1)))
        else:
            avg_color = (128, 128, 128)

        draw.polygon(triangle_pts, fill=avg_color, outline=avg_color)

    log(output_dir, f"Lowpoly: drew {len(tri.simplices)} triangles")
    return result


def generate_crystal(img_orig, mask, output_dir, num_points=2000, saturation=1.3, seed=None):
    """Edge-aware Delaunay: dense triangles at contours, sparse in flat areas.

    Unlike lowpoly which scatters points uniformly, crystal places most vertices
    ON detected edges so triangles naturally break along contours — creating a
    shattered/crystalline effect where triangles 'jump out' of the silhouette.
    """
    log(output_dir, f"Generating crystal geometry (target ~{num_points} points, sat={saturation})...")

    if seed is not None:
        np.random.seed(seed)

    img_np = np.array(img_orig)
    mask_np = np.array(mask)
    gray = np.array(img_orig.convert("L"), dtype=np.float64)
    w, h = img_orig.size

    # --- Edge detection on the FULL image (not just mask) ---
    # Use multiple thresholds: strong edges get more points
    edges_strong = detect_edges(gray, threshold=40)
    edges_medium = detect_edges(gray, threshold=20)
    edges_weak = detect_edges(gray, threshold=10)

    # Restrict to mask area (with some bleed for the "jumping out" effect)
    mask_dilated = np.array(mask.filter(ImageFilter.MaxFilter(
        max(3, int(min(w, h) * 0.02)) | 1)))  # dilate 2% of short edge
    mask_zone = mask_dilated > 64  # generous zone

    strong_pts = get_edge_points(edges_strong & mask_zone)
    medium_pts = get_edge_points(edges_medium & mask_zone & ~edges_strong)
    weak_pts = get_edge_points(edges_weak & mask_zone & ~edges_medium)

    log(output_dir, f"Crystal edges: {len(strong_pts)} strong, {len(medium_pts)} medium, {len(weak_pts)} weak")

    # Allocate points: 50% strong edges, 25% medium, 10% weak, 15% random interior
    n_strong = int(num_points * 0.50)
    n_medium = int(num_points * 0.25)
    n_weak = int(num_points * 0.10)
    n_interior = int(num_points * 0.15)

    points = []

    # Sample from strong edges (densest)
    if len(strong_pts) > n_strong:
        idx = np.random.choice(len(strong_pts), n_strong, replace=False)
        points.append(strong_pts[idx])
    elif len(strong_pts) > 0:
        points.append(strong_pts)

    # Sample from medium edges
    if len(medium_pts) > n_medium:
        idx = np.random.choice(len(medium_pts), n_medium, replace=False)
        points.append(medium_pts[idx])
    elif len(medium_pts) > 0:
        points.append(medium_pts)

    # Sample from weak edges (sparser)
    if len(weak_pts) > n_weak:
        idx = np.random.choice(len(weak_pts), n_weak, replace=False)
        points.append(weak_pts[idx])
    elif len(weak_pts) > 0:
        points.append(weak_pts)

    # Random interior points for flat areas (prevents huge triangles)
    ys_mask, xs_mask = np.where(mask_np > 127)
    if len(ys_mask) > n_interior:
        idx = np.random.choice(len(ys_mask), n_interior, replace=False)
        points.append(np.column_stack((ys_mask[idx], xs_mask[idx])))

    # Add image corners and edge points for complete coverage
    corners = np.array([[0, 0], [0, w-1], [h-1, 0], [h-1, w-1]])
    # Add points along image borders
    border_n = 20
    border_pts = []
    for i in range(border_n):
        t = i / border_n
        border_pts.extend([
            [0, int(t * (w-1))], [h-1, int(t * (w-1))],
            [int(t * (h-1)), 0], [int(t * (h-1)), w-1],
        ])
    points.append(corners)
    points.append(np.array(border_pts))

    if not points:
        log(output_dir, "No points found for crystal — returning blank", "WARN")
        return Image.new("RGB", (w, h), (0, 0, 0))

    all_points = np.vstack(points)
    all_points = np.unique(all_points, axis=0)

    # Clamp to image bounds
    all_points[:, 0] = np.clip(all_points[:, 0], 0, h - 1)
    all_points[:, 1] = np.clip(all_points[:, 1], 0, w - 1)

    log(output_dir, f"Crystal: triangulating {len(all_points)} points...")

    try:
        tri = Delaunay(all_points)
    except Exception as e:
        log(output_dir, f"Delaunay triangulation failed: {e}", "WARN")
        return Image.new("RGB", (w, h), (0, 0, 0))

    # Draw ALL triangles (full image, no mask skip)
    result = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(result)

    for simplex in tri.simplices:
        triangle_pts = [(int(all_points[i][1]), int(all_points[i][0])) for i in simplex]

        # Centroid
        cy = int(np.mean([all_points[i][0] for i in simplex]))
        cx = int(np.mean([all_points[i][1] for i in simplex]))
        cy = max(0, min(h - 1, cy))
        cx = max(0, min(w - 1, cx))

        # Sample average color from a patch around centroid
        # Larger patch for bigger triangles
        verts_y = [all_points[i][0] for i in simplex]
        verts_x = [all_points[i][1] for i in simplex]
        tri_size = max(max(verts_y) - min(verts_y), max(verts_x) - min(verts_x))
        patch_r = max(2, int(tri_size * 0.15))

        y_lo = max(0, cy - patch_r)
        y_hi = min(h, cy + patch_r + 1)
        x_lo = max(0, cx - patch_r)
        x_hi = min(w, cx + patch_r + 1)
        patch = img_np[y_lo:y_hi, x_lo:x_hi]
        if patch.size > 0:
            avg_color = tuple(int(c) for c in patch.mean(axis=(0, 1)))
        else:
            avg_color = (128, 128, 128)

        # 3D facet brightness jitter: ±25% random brightness, 1.3x saturation boost
        r, g, b = avg_color
        # Convert to HSV-like: boost saturation by scaling distance from grey
        grey = (r + g + b) / 3.0
        r2 = int(np.clip(grey + (r - grey) * 1.3, 0, 255))
        g2 = int(np.clip(grey + (g - grey) * 1.3, 0, 255))
        b2 = int(np.clip(grey + (b - grey) * 1.3, 0, 255))
        # Apply ±25% brightness jitter
        jitter = np.random.uniform(0.75, 1.25)
        r3 = int(np.clip(r2 * jitter, 0, 255))
        g3 = int(np.clip(g2 * jitter, 0, 255))
        b3 = int(np.clip(b2 * jitter, 0, 255))
        facet_color = (r3, g3, b3)
        draw.polygon(triangle_pts, fill=facet_color, outline=facet_color)

    log(output_dir, f"Crystal: drew {len(tri.simplices)} triangles")

    # Boost saturation for bolder look
    if saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(saturation)
        result = ImageEnhance.Contrast(result).enhance(1.0 + (saturation - 1.0) * 0.3)

    return result


def generate_blocks(img_orig, mask, output_dir, block_size=30, seed=None):
    """Generate block mosaic: grid of rectangles with average color."""
    log(output_dir, f"Generating blocks geometry (block_size={block_size}px)...")

    img_np = np.array(img_orig)
    mask_np = np.array(mask)
    w, h = img_orig.size

    result = Image.new("RGB", (w, h), (0, 0, 0))
    result_np = np.array(result)

    blocks_drawn = 0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)

            # Check if this block overlaps with the mask
            mask_patch = mask_np[y:y_end, x:x_end]
            if mask_patch.mean() < 64:  # Less than ~25% of the block is subject
                continue

            # Get average color from original
            img_patch = img_np[y:y_end, x:x_end]
            # Only average pixels within the mask
            mask_bool = mask_patch > 127
            if mask_bool.sum() == 0:
                continue

            avg_color = tuple(int(c) for c in img_patch[mask_bool].mean(axis=0))
            result_np[y:y_end, x:x_end][mask_bool] = avg_color
            blocks_drawn += 1

    result = Image.fromarray(result_np)
    log(output_dir, f"Blocks: drew {blocks_drawn} blocks")
    return result


def generate_contour(img_orig, mask, output_dir, num_levels=8, seed=None):
    """Generate contour lines: topographic-style lines at multiple brightness levels."""
    log(output_dir, f"Generating contour geometry ({num_levels} levels)...")

    img_np = np.array(img_orig)
    mask_np = np.array(mask)
    gray = np.array(img_orig.convert("L"), dtype=np.float64)
    w, h = img_orig.size

    # Apply mask
    gray_masked = gray.copy()
    gray_masked[mask_np < 127] = 0

    # Generate contour lines at multiple threshold levels
    result = Image.new("RGB", (w, h), (0, 0, 0))
    result_np = np.array(result)

    # Threshold levels evenly spaced across the brightness range within the mask
    mask_pixels = gray[mask_np > 127]
    if len(mask_pixels) == 0:
        log(output_dir, "No mask pixels for contour generation", "WARN")
        return result

    lo = np.percentile(mask_pixels, 5)
    hi = np.percentile(mask_pixels, 95)
    levels = np.linspace(lo, hi, num_levels + 2)[1:-1]  # Skip extreme ends

    # Create a colormap — cycle through warm/cool tones
    base_colors = [
        (220, 80, 60),    # warm red
        (240, 160, 50),   # orange
        (230, 210, 70),   # yellow
        (80, 200, 120),   # green
        (60, 160, 220),   # blue
        (140, 100, 220),  # purple
        (220, 100, 180),  # pink
        (180, 200, 200),  # light teal
    ]

    total_contour_pixels = 0
    for i, level in enumerate(levels):
        # Find boundaries: pixels where the gray crosses this threshold
        above = gray_masked >= level
        # Erode and XOR to get boundary
        above_img = Image.fromarray((above * 255).astype(np.uint8))
        eroded = above_img.filter(ImageFilter.MinFilter(3))
        boundary = np.array(above_img).astype(bool) & ~np.array(eroded).astype(bool)

        # Apply mask
        boundary = boundary & (mask_np > 127)

        color = base_colors[i % len(base_colors)]
        result_np[boundary] = color
        total_contour_pixels += boundary.sum()

    result = Image.fromarray(result_np)
    log(output_dir, f"Contour: {total_contour_pixels} contour pixels across {num_levels} levels")
    return result


# ---------------------------------------------------------------------------
# Blend Modes
# ---------------------------------------------------------------------------
def blend_images(original, geometric, mask, mode="overlay", opacity=0.6, output_dir=None):
    """Blend geometric art with original photo.

    The geometric art is applied only within the mask area.
    Outside the mask, the original photo is slightly darkened for contrast.
    """
    w, h = original.size
    mask_np = np.array(mask)

    # Slightly darken the background (outside mask) for gallery contrast
    bg_darken = ImageEnhance.Brightness(original).enhance(0.7)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=8))

    # Start with darkened BG
    result = bg_darken.copy()
    # Restore original brightness inside mask area
    result.paste(original, mask=soft_mask)

    # Now blend the geometric art within the mask
    if mode == "alpha":
        # Simple alpha blend
        blended = Image.blend(original, geometric, opacity)
    elif mode == "overlay":
        blended = _blend_overlay(original, geometric, opacity)
    elif mode == "multiply":
        blended = _blend_multiply(original, geometric, opacity)
    elif mode == "screen":
        blended = _blend_screen(original, geometric, opacity)
    else:
        blended = Image.blend(original, geometric, opacity)

    # Paste blended result only within the subject mask
    result.paste(blended, mask=soft_mask)
    return result


def _blend_overlay(base, top, opacity):
    """Overlay blend mode: combines multiply and screen based on base brightness."""
    base_np = np.array(base, dtype=np.float64) / 255.0
    top_np = np.array(top, dtype=np.float64) / 255.0

    # Overlay formula: if base < 0.5: 2*base*top, else: 1 - 2*(1-base)*(1-top)
    low = 2 * base_np * top_np
    high = 1 - 2 * (1 - base_np) * (1 - top_np)
    overlay = np.where(base_np < 0.5, low, high)

    # Blend with original at opacity
    result = base_np * (1 - opacity) + overlay * opacity
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


def _blend_multiply(base, top, opacity):
    """Multiply blend mode: darken where geometric lines are."""
    base_np = np.array(base, dtype=np.float64) / 255.0
    top_np = np.array(top, dtype=np.float64) / 255.0

    multiplied = base_np * top_np
    result = base_np * (1 - opacity) + multiplied * opacity
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


def _blend_screen(base, top, opacity):
    """Screen blend mode: lighten where geometric lines are."""
    base_np = np.array(base, dtype=np.float64) / 255.0
    top_np = np.array(top, dtype=np.float64) / 255.0

    screened = 1 - (1 - base_np) * (1 - top_np)
    result = base_np * (1 - opacity) + screened * opacity
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Auto-Correction
# ---------------------------------------------------------------------------
def apply_adjustments(args, eval_result, output_dir):
    """Parse evaluation result and return correction strategy, or None if not needed."""
    if not eval_result:
        return None

    score = eval_result.get("score", 10)
    if score >= 7:
        return None

    adjustments = eval_result.get("adjustments", {})
    issues = eval_result.get("issues", [])
    changes = {}

    if "blend_harsh" in issues or "too_aggressive" in issues:
        changes["blend_opacity"] = max(0.2, args.blend_opacity - 0.2)

    if "too_subtle" in issues:
        changes["blend_opacity"] = min(1.0, args.blend_opacity + 0.2)

    if "subject_lost" in issues or "subject_unrecognizable" in issues:
        changes["blend_opacity"] = max(0.2, args.blend_opacity - 0.25)

    if "too_dark" in issues:
        changes["brighten"] = True

    if "colors_clash" in issues or "geometry_messy" in issues:
        changes["try_different_preset"] = True

    if adjustments.get("try_different_preset"):
        changes["try_different_preset"] = True

    if adjustments.get("try_different_blend"):
        changes["try_different_blend"] = True

    if adjustments.get("blend_opacity") is not None:
        changes["blend_opacity"] = adjustments["blend_opacity"]

    if not changes:
        return None

    log(output_dir, f"Auto-correction strategy: {changes}")
    return changes


def pick_alternative_preset(current):
    """Pick a random different geometry preset."""
    alternatives = [p for p in GEOMETRY_PRESETS if p != current]
    return random.choice(alternatives) if alternatives else current


def pick_alternative_blend(current):
    """Pick a random different blend mode."""
    alternatives = [b for b in BLEND_MODES if b != current]
    return random.choice(alternatives) if alternatives else current


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------
def upload_to_gdrive(local_dir, model_name, photo_name, timestamp, output_dir):
    import subprocess
    gdrive_path = f"gdrive:_photos from openclaw/daily_game/public/{model_name}_{photo_name}_{timestamp}"
    try:
        subprocess.run(["rclone", "copy", local_dir, gdrive_path], check=True, timeout=120)
        res = subprocess.run(["rclone", "link", gdrive_path], capture_output=True, text=True, timeout=30)
        link = res.stdout.strip()
        log(output_dir, f"GDrive upload OK: {link}")
        return link
    except Exception as e:
        log(output_dir, f"GDrive upload failed: {e}", "ERROR")
        return None


def copy_to_local(output_dir, local_dest):
    try:
        if os.path.exists(local_dest):
            for f in os.listdir(output_dir):
                src = os.path.join(output_dir, f)
                dst = os.path.join(local_dest, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            shutil.copytree(output_dir, local_dest)
        return local_dest
    except Exception as e:
        log(output_dir, f"Local copy failed: {e}", "ERROR")
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_workflow(args):
    # Resolve model/photo names from filename
    basename = os.path.basename(args.source)
    photo_name = os.path.splitext(basename)[0]
    model_name = args.model_name
    if not model_name:
        match = re.search(r"(Original_|BLD_|Rong_IMG_)(.*?)_(.*)\.(jpg|png|jpeg|JPG)", basename, re.IGNORECASE)
        if match:
            model_name = match.group(2).replace(" ", "_")
            photo_name = match.group(3).replace(" ", "_")
        else:
            source_abs = os.path.abspath(args.source)
            parts = source_abs.replace("\\", "/").split("/")
            try:
                photos_idx = parts.index("_photos")
                if photos_idx + 1 < len(parts):
                    model_name = parts[photos_idx + 1].replace(" ", "_")
            except ValueError:
                model_name = "Unknown"

    # Seed
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Output directory
    israel_dt = datetime.now(ISRAEL_TZ)
    timestamp = israel_dt.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name}_{photo_name}_{timestamp}_geo_{args.geometry}_{random.randint(10, 99)}"
    if args.local_output_dir:
        output_dir = os.path.join(args.local_output_dir, folder_name)
    else:
        output_dir = os.path.join("outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save a copy of this script for reproducibility
    try:
        with open(__file__, "r") as src, open(os.path.join(output_dir, f"pose_geometry_script_{timestamp}.py"), "w") as dst:
            dst.write(src.read())
    except OSError:
        log(output_dir, "Could not save script copy (permission issue, non-critical)", "WARN")

    # Log configuration
    log(output_dir, "=" * 60)
    log(output_dir, f"POSE GEOMETRY WORKFLOW START")
    log(output_dir, f"Source:         {args.source}")
    log(output_dir, f"Geometry:       {args.geometry}")
    log(output_dir, f"Blend mode:     {args.blend_mode}")
    log(output_dir, f"Blend opacity:  {args.blend_opacity}")
    log(output_dir, f"Line color:     {args.line_color}")
    log(output_dir, f"Block size:     {args.block_size}")
    log(output_dir, f"Seed:           {base_seed}")
    log(output_dir, f"Auto-correct:   {args.auto_correct}")
    log(output_dir, f"Output to:      {args.output_to}")
    log(output_dir, f"Output dir:     {output_dir}")
    log(output_dir, "=" * 60)

    img_orig = Image.open(args.source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), quality=95)

    # -----------------------------------------------------------------------
    # Step 1: Extract subject mask
    # -----------------------------------------------------------------------
    log(output_dir, "--- Step 1: Extract subject mask ---")
    affect = getattr(args, "affect", "subject")
    exclude = getattr(args, "exclude", "")
    mask, mask_info = build_mask(args.source, affect=affect, exclude=exclude,
                                 output_dir=output_dir)
    log(output_dir, f"Mask engine: {mask_info['engine']}, coverage: {mask_info['coverage_pct']}%")

    # Ensure mask matches image size
    if mask.size != img_orig.size:
        mask = mask.resize(img_orig.size, Image.LANCZOS)

    # Binarize and save
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    mask.save(os.path.join(output_dir, "1_mask.png"))

    mask_coverage = np.array(mask).mean() / 255.0
    log(output_dir, f"Mask coverage: {mask_coverage:.1%} of image")
    if mask_coverage < 0.03:
        log(output_dir, "Mask coverage < 3% — subject may be too small for geometric art", "WARN")

    # -----------------------------------------------------------------------
    # Step 2: Edge detection (local)
    # -----------------------------------------------------------------------
    log(output_dir, "--- Step 2: Edge detection ---")
    gray = np.array(img_orig.convert("L"), dtype=np.float64)
    mask_np = np.array(mask)
    gray_masked = gray * (mask_np / 255.0)
    edges = detect_edges(gray_masked, threshold=30)
    edge_count = edges.sum()
    log(output_dir, f"Detected {edge_count} edge pixels")

    # Save edge visualization
    edge_vis = Image.fromarray((edges * 255).astype(np.uint8))
    edge_vis.save(os.path.join(output_dir, "2_edges.png"))

    # -----------------------------------------------------------------------
    # Step 3: Generate geometric reconstruction
    # -----------------------------------------------------------------------
    log(output_dir, f"--- Step 3: Generate geometric art ({args.geometry}) ---")
    geo_img = _generate_geometry(img_orig, mask, args.geometry, output_dir,
                                  line_color=args.line_color, block_size=args.block_size,
                                  num_points=args.num_points, seed=base_seed)
    geo_img.save(os.path.join(output_dir, f"3_geometry_{args.geometry}.jpg"), quality=95)
    check_image_quality(geo_img, f"geometry-{args.geometry}", output_dir)

    # -----------------------------------------------------------------------
    # Step 4: Blend with original
    # -----------------------------------------------------------------------
    log(output_dir, f"--- Step 4: Blend ({args.blend_mode}, opacity={args.blend_opacity}) ---")
    final_img = blend_images(img_orig, geo_img, mask, mode=args.blend_mode,
                             opacity=args.blend_opacity, output_dir=output_dir)
    final_path = os.path.join(output_dir, "4_final.jpg")
    final_img.save(final_path, quality=95)
    qc = check_image_quality(final_img, "FINAL", output_dir)

    # -----------------------------------------------------------------------
    # Step 5: Gemini evaluation + auto-correct
    # -----------------------------------------------------------------------
    log(output_dir, "--- Step 5: Aesthetic evaluation ---")
    eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

    current_final = final_img
    current_path = final_path
    current_geometry = args.geometry
    current_blend = args.blend_mode
    current_opacity = args.blend_opacity

    if args.auto_correct and eval_result:
        for correction_round in range(1, args.max_corrections + 1):
            changes = apply_adjustments(args, eval_result, output_dir)
            if changes is None:
                log(output_dir, f"Score >= 7 or no corrections needed — keeping result")
                break

            log(output_dir, f"--- Auto-correction round {correction_round}/{args.max_corrections} ---")

            if changes.get("try_different_preset"):
                current_geometry = pick_alternative_preset(current_geometry)
                log(output_dir, f"Switching geometry preset to: {current_geometry}")

            if changes.get("try_different_blend"):
                current_blend = pick_alternative_blend(current_blend)
                log(output_dir, f"Switching blend mode to: {current_blend}")

            if "blend_opacity" in changes:
                current_opacity = changes["blend_opacity"]
                log(output_dir, f"Adjusting blend opacity to: {current_opacity}")

            # Re-generate geometry if preset changed
            if current_geometry != args.geometry or changes.get("try_different_preset"):
                new_seed = random.randint(0, 2**32 - 1)
                geo_img = _generate_geometry(img_orig, mask, current_geometry, output_dir,
                                              line_color=args.line_color, block_size=args.block_size,
                                              num_points=args.num_points, seed=new_seed)
                geo_img.save(os.path.join(output_dir, f"3_geometry_{current_geometry}_r{correction_round}.jpg"), quality=95)

            # Re-blend
            retry_final = blend_images(img_orig, geo_img, mask, mode=current_blend,
                                       opacity=current_opacity, output_dir=output_dir)

            if changes.get("brighten"):
                retry_final = ImageEnhance.Brightness(retry_final).enhance(1.25)

            retry_path = os.path.join(output_dir, f"4_final_r{correction_round}.jpg")
            retry_final.save(retry_path, quality=95)
            check_image_quality(retry_final, f"FINAL-R{correction_round}", output_dir)

            eval_result = evaluate_with_gemini(retry_final, output_dir, original_img=img_orig)

            if eval_result and eval_result.get("score", 0) >= 7:
                current_final = retry_final
                current_path = retry_path
                log(output_dir, f"Auto-correction round {correction_round} achieved score {eval_result.get('score')}")
                break

            if eval_result and eval_result.get("score", 0) > (eval_result or {}).get("score", 0):
                current_final = retry_final
                current_path = retry_path

    # -----------------------------------------------------------------------
    # Step 6: Output
    # -----------------------------------------------------------------------
    log(output_dir, "--- Step 6: Output ---")

    # Copy final to shared finals/ folder
    if args.local_output_dir:
        finals_dir = os.path.join(args.local_output_dir, "finals")
    else:
        finals_dir = os.path.join(output_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_name = os.path.basename(output_dir) + ".jpg"
    final_dest = os.path.join(finals_dir, finals_name)
    current_final.save(final_dest, quality=95)
    log(output_dir, f"Final copied to: {final_dest}")

    # Save metadata
    metadata = {
        "source": os.path.abspath(args.source),
        "model_name": model_name,
        "photo_name": photo_name,
        "geometry": current_geometry,
        "blend_mode": current_blend,
        "blend_opacity": current_opacity,
        "line_color": args.line_color,
        "block_size": args.block_size,
        "seed": base_seed,
        "timestamp": timestamp,
        "eval_score": eval_result.get("score") if eval_result else None,
        "eval_critique": eval_result.get("critique") if eval_result else None,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Upload/copy as requested
    if args.output_to in ("gdrive", "both"):
        upload_to_gdrive(finals_dir, model_name, photo_name, timestamp, output_dir)

    if args.output_to in ("local", "both"):
        if args.local_output_dir:
            local_dest = os.path.join(args.local_output_dir, folder_name)
            if local_dest != output_dir:
                copy_to_local(output_dir, local_dest)
                log(output_dir, f"Copied to local: {local_dest}")
            else:
                log(output_dir, f"Output already at local dir: {output_dir}")
        else:
            log(output_dir, f"Output at: {output_dir}")

    log(output_dir, "=" * 60)
    score_str = f"{eval_result.get('score')}/10" if eval_result else "N/A"
    log(output_dir, f"WORKFLOW COMPLETE — Score: {score_str}")
    log(output_dir, f"Output: {output_dir}")
    log(output_dir, "=" * 60)


def _generate_geometry(img_orig, mask, preset, output_dir, line_color="auto", block_size=30, num_points=None, seed=None):
    """Dispatch to the appropriate geometry generator."""
    if preset == "wireframe":
        return generate_wireframe(img_orig, mask, output_dir, line_color=line_color, seed=seed)
    elif preset == "lowpoly":
        return generate_lowpoly(img_orig, mask, output_dir, seed=seed)
    elif preset == "blocks":
        return generate_blocks(img_orig, mask, output_dir, block_size=block_size, seed=seed)
    elif preset == "contour":
        return generate_contour(img_orig, mask, output_dir, seed=seed)
    elif preset == "crystal":
        return generate_crystal(img_orig, mask, output_dir, num_points=num_points or 2000, seed=seed)
    else:
        raise ValueError(f"Unknown geometry preset: {preset}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pose Geometry — Geometric Art from Photo Silhouettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=None, help="Input image path")
    parser.add_argument("--model-name", default="", help="Model/subject name (auto-detected from filename if empty)")

    # Geometry preset
    parser.add_argument("--geometry", default="lowpoly", choices=list(GEOMETRY_PRESETS.keys()),
                        help="Geometry preset (default: lowpoly)")

    # Blend settings
    parser.add_argument("--blend-mode", default="overlay", choices=BLEND_MODES,
                        help="Blend mode for combining geometric art with original (default: overlay)")
    parser.add_argument("--blend-opacity", type=float, default=0.6,
                        help="Blend opacity 0.0-1.0 (default: 0.6)")

    # Geometry-specific options
    parser.add_argument("--line-color", default="auto",
                        help="Line color for wireframe preset: hex color or 'auto' to sample from image (default: auto)")
    parser.add_argument("--block-size", type=int, default=30,
                        help="Block size in pixels for blocks preset (default: 30)")
    parser.add_argument("--num-points", type=int, default=None,
                        help="Number of points for crystal triangulation (default: 2000). More = denser/smaller triangles")

    # Seed & corrections
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if not set)")
    parser.add_argument("--auto-correct", action="store_true", default=False,
                        help="If aesthetic score < 7, auto-adjust params and retry")
    parser.add_argument("--max-corrections", type=int, default=2,
                        help="Max auto-correction rounds (default: 2)")

    # Output
    parser.add_argument("--output-to", choices=["gdrive", "local", "both"], default="both",
                        help="Where to output results (default: both)")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")

    parser.add_argument("--list-presets", action="store_true", help="List all geometry presets and exit")

    # Mask targeting (--affect / --exclude) via shared masking module
    add_affect_args(parser)

    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{len(GEOMETRY_PRESETS)} geometry presets:\n")
        for name, desc in GEOMETRY_PRESETS.items():
            print(f"  {name:<12} {desc}")
        print()
        sys.exit(0)

    # Validate source is provided
    if not args.source:
        parser.error("--source is required (unless using --list-presets)")

    if not os.path.isfile(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        sys.exit(1)

    # Validate blend opacity
    args.blend_opacity = max(0.0, min(1.0, args.blend_opacity))

    # Validate line-color if not "auto"
    if args.line_color != "auto":
        hex_clean = args.line_color.lstrip("#")
        if len(hex_clean) != 6 or not all(c in "0123456789abcdefABCDEF" for c in hex_clean):
            print(f"ERROR: Invalid line-color '{args.line_color}'. Use hex like '#FF6633' or 'auto'.")
            sys.exit(1)

    run_workflow(args)


if __name__ == "__main__":
    main()
