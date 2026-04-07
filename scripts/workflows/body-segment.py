#!/usr/bin/env python3
"""
Body Segment — Fine-grained body part segmentation.

Uses MediaPipe multiclass selfie segmentation to separate body into:
  - face-skin: face, chin, jaw, forehead (excludes accessories like blindfolds/gags)
  - body-skin: neck, shoulders, chest, arms, torso skin
  - hair: all hair
  - clothes: clothing, accessories, blindfolds, gags, gloves
  - others: miscellaneous items
  - hands: detected via MediaPipe hand landmarker (can subtract someone else's hands)
  - ropes: detected via HSV color thresholding (shibari rope detection)

Combine masks with --include/--exclude flags to create exactly the mask you need.

Examples:
  # Face + neck + shoulders, no ropes, no hands, no hair
  body-segment.py --source photo.jpg --include face-skin,body-skin --exclude hands,ropes,hair

  # Just the model's skin (face + body), subtract another person's hands
  body-segment.py --source photo.jpg --include face-skin,body-skin --exclude hands

  # Everything except background
  body-segment.py --source photo.jpg --include all --exclude background

  # Just hair
  body-segment.py --source photo.jpg --include hair
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/openclaw-venv/mediapipe_models")
SELFIE_MODEL = os.path.join(MODELS_DIR, "selfie_multiclass.tflite")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker.task")

# MediaPipe multiclass categories
CATEGORIES = {
    0: "background",
    1: "hair",
    2: "body-skin",
    3: "face-skin",
    4: "clothes",
    5: "others",
}
CAT_BY_NAME = {v: k for k, v in CATEGORIES.items()}

# Rope color ranges in HSV (same as time-corruption.py)
ROPE_COLOR_RANGES = {
    "red": [
        {"h_min": 0, "h_max": 15, "s_min": 60, "s_max": 255, "v_min": 50, "v_max": 255},
        {"h_min": 160, "h_max": 180, "s_min": 60, "s_max": 255, "v_min": 50, "v_max": 255},
    ],
    "beige": [
        {"h_min": 15, "h_max": 35, "s_min": 30, "s_max": 150, "v_min": 120, "v_max": 255},
    ],
    "black": [
        {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 80, "v_min": 0, "v_max": 50},
    ],
    "white": [
        {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 30, "v_min": 200, "v_max": 255},
    ],
}

# Valid include/exclude names
VALID_PARTS = {"face-skin", "body-skin", "hair", "clothes", "others", "background",
               "hands", "ropes", "all", "skin"}


def rgb_to_hsv_array(rgb_array):
    """Convert RGB numpy array to HSV (H: 0-180, S: 0-255, V: 0-255) without OpenCV."""
    rgb = rgb_array.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    h = (h / 2).astype(np.uint8)  # Scale to 0-180

    # Saturation
    s = np.zeros_like(cmax)
    nonzero = cmax > 0
    s[nonzero] = (diff[nonzero] / cmax[nonzero]) * 255
    s = s.astype(np.uint8)

    # Value
    v = (cmax * 255).astype(np.uint8)

    return np.stack([h, s, v], axis=-1)


def detect_ropes(img_array, subject_mask=None, rope_color="auto"):
    """Detect ropes using HSV color thresholding within subject area."""
    hsv = rgb_to_hsv_array(img_array)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if rope_color == "auto":
        colors_to_try = ["red", "beige", "black"]
    else:
        colors_to_try = [rope_color]

    rope_mask = np.zeros(img_array.shape[:2], dtype=bool)

    for color in colors_to_try:
        if color not in ROPE_COLOR_RANGES:
            continue
        for rng in ROPE_COLOR_RANGES[color]:
            color_match = (
                (h >= rng["h_min"]) & (h <= rng["h_max"]) &
                (s >= rng["s_min"]) & (s <= rng["s_max"]) &
                (v >= rng["v_min"]) & (v <= rng["v_max"])
            )
            if subject_mask is not None:
                color_match = color_match & subject_mask
            rope_mask |= color_match

    # Morphological cleanup: thin the detected regions to rope-like structures
    # Ropes are thin — remove large blobs that are likely skin
    rope_uint8 = (rope_mask * 255).astype(np.uint8)
    rope_img = Image.fromarray(rope_uint8, "L")

    # Close small gaps in ropes
    short_edge = min(img_array.shape[:2])
    close_r = max(2, int(short_edge * 0.003))
    rope_img = rope_img.filter(ImageFilter.MaxFilter(close_r | 1))  # dilate (must be odd)
    rope_img = rope_img.filter(ImageFilter.MinFilter(close_r | 1))  # erode

    return np.array(rope_img) > 127


def detect_hands_mask(img_array):
    """Use MediaPipe hand landmarker to create hand masks."""
    import mediapipe as mp

    if not os.path.exists(HAND_MODEL):
        log.warning("Hand model not found at %s — skipping hand detection", HAND_MODEL)
        return np.zeros(img_array.shape[:2], dtype=bool)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    h_img, w_img = img_array.shape[:2]

    base = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL)
    opts = mp.tasks.vision.HandLandmarkerOptions(base_options=base, num_hands=4)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    result = detector.detect(mp_img)
    detector.close()

    hand_mask = np.zeros((h_img, w_img), dtype=bool)

    if not result.hand_landmarks:
        log.info("No hands detected")
        return hand_mask

    log.info("Detected %d hand(s)", len(result.hand_landmarks))

    for i, hand in enumerate(result.hand_landmarks):
        # Get bounding polygon from all 21 hand landmarks
        points = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in hand]

        # Create convex hull mask for this hand
        from PIL import ImageDraw
        hand_img = Image.new("L", (w_img, h_img), 0)
        draw = ImageDraw.Draw(hand_img)

        # Convex hull from landmarks
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Simple approach: expanded bounding box + convex hull polygon
        # Use scipy convex hull if available, otherwise just use the landmark polygon
        try:
            from scipy.spatial import ConvexHull
            pts_array = np.array(points)
            hull = ConvexHull(pts_array)
            hull_points = [tuple(pts_array[v]) for v in hull.vertices]
            draw.polygon(hull_points, fill=255)
        except ImportError:
            # Fallback: draw filled polygon from landmarks
            draw.polygon(points, fill=255)

        # Expand to cover full hand area (landmarks are at joints, not fingertip edges)
        expand_px = max(5, int(min(h_img, w_img) * 0.025))
        hand_img = hand_img.filter(ImageFilter.MaxFilter(expand_px | 1))

        hand_arr = np.array(hand_img) > 127
        hand_mask |= hand_arr
        wrist = hand[0]
        log.info("  Hand %d: wrist at (%d, %d), %d px",
                 i, int(wrist.x * w_img), int(wrist.y * h_img), np.sum(hand_arr))

    return hand_mask


def segment_body(img_array):
    """Run MediaPipe multiclass selfie segmentation. Returns category mask (0-5)."""
    import mediapipe as mp

    if not os.path.exists(SELFIE_MODEL):
        raise FileNotFoundError(f"Selfie model not found: {SELFIE_MODEL}")

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    base = mp.tasks.BaseOptions(model_asset_path=SELFIE_MODEL)
    opts = mp.tasks.vision.ImageSegmenterOptions(
        base_options=base,
        output_category_mask=True,
    )
    segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(opts)
    result = segmenter.segment(mp_img)
    segmenter.close()

    cat_mask = result.category_mask.numpy_view().squeeze()

    # Log category distribution
    h, w = img_array.shape[:2]
    total = h * w
    for cat_id, cat_name in CATEGORIES.items():
        count = np.sum(cat_mask == cat_id)
        pct = count / total * 100
        if pct > 0.1:
            log.info("  Segment %-12s: %6.1f%%", cat_name, pct)

    return cat_mask


def morphological_cleanup(mask_array, img_shape, operation="close", strength=1.0):
    """Clean up a boolean mask using morphological operations.

    Args:
        mask_array: boolean numpy array
        img_shape: (height, width) of the image
        operation: 'close' (fill holes), 'open' (remove noise), 'smooth' (both)
        strength: multiplier for kernel size (1.0 = default)
    """
    short_edge = min(img_shape[:2])
    base_r = max(3, int(short_edge * 0.004 * strength))
    kernel = base_r | 1  # ensure odd

    mask_uint8 = (mask_array * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8, "L")

    if operation in ("close", "smooth"):
        # Close: dilate then erode — fills small holes
        mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel))
        mask_img = mask_img.filter(ImageFilter.MinFilter(kernel))

    if operation in ("open", "smooth"):
        # Open: erode then dilate — removes small noise
        mask_img = mask_img.filter(ImageFilter.MinFilter(kernel))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel))

    return np.array(mask_img) > 127


def feather_mask(mask_array, img_shape, radius_pct=0.5):
    """Apply Gaussian blur to mask edges for soft transitions.

    Args:
        mask_array: boolean numpy array
        img_shape: (height, width)
        radius_pct: blur radius as % of short edge
    """
    short_edge = min(img_shape[:2])
    radius = max(1, int(short_edge * radius_pct / 100))

    mask_uint8 = (mask_array * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8, "L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))

    return np.array(mask_img).astype(np.float32) / 255.0


def build_mask(cat_mask, img_array, include_parts, exclude_parts,
               rope_color="auto", feather=0.5, cleanup="smooth"):
    """Build the final mask from segmentation categories and detections.

    Args:
        cat_mask: numpy array of category IDs (0-5) from MediaPipe
        img_array: original image as numpy array (for rope/hand detection)
        include_parts: set of part names to include
        exclude_parts: set of part names to exclude
        rope_color: rope color for detection ('auto', 'red', 'beige', etc.)
        feather: feather radius as % of short edge (0 = hard edges)
        cleanup: morphological cleanup mode ('close', 'open', 'smooth', 'none')

    Returns:
        tuple: (final_mask as float32 0-1, individual_masks dict)
    """
    h, w = img_array.shape[:2]

    # Resolve 'all' and 'skin' shortcuts
    if "all" in include_parts:
        include_parts = {"face-skin", "body-skin", "hair", "clothes", "others"}
    if "skin" in include_parts:
        include_parts = (include_parts - {"skin"}) | {"face-skin", "body-skin"}

    # Build category masks
    individual_masks = {}
    for cat_id, cat_name in CATEGORIES.items():
        individual_masks[cat_name] = (cat_mask == cat_id)

    # Detect hands if needed for exclusion
    if "hands" in exclude_parts:
        log.info("Detecting hands for exclusion...")
        individual_masks["hands"] = detect_hands_mask(img_array)
    else:
        individual_masks["hands"] = np.zeros((h, w), dtype=bool)

    # Detect ropes if needed for exclusion
    if "ropes" in exclude_parts:
        log.info("Detecting ropes for exclusion (color=%s)...", rope_color)
        # Use combined non-background as subject mask for rope detection
        subject = ~individual_masks["background"]
        individual_masks["ropes"] = detect_ropes(img_array, subject_mask=subject,
                                                  rope_color=rope_color)
        rope_pct = np.sum(individual_masks["ropes"]) / (h * w) * 100
        log.info("  Ropes detected: %.1f%% of image", rope_pct)
    else:
        individual_masks["ropes"] = np.zeros((h, w), dtype=bool)

    # Start with included parts
    final_mask = np.zeros((h, w), dtype=bool)
    for part in include_parts:
        if part in individual_masks:
            final_mask |= individual_masks[part]
            log.info("  + %-12s: %6.1f%%", part,
                     np.sum(individual_masks[part]) / (h * w) * 100)

    # Subtract excluded parts
    for part in exclude_parts:
        if part in individual_masks and np.any(individual_masks[part]):
            before = np.sum(final_mask)
            final_mask = final_mask & ~individual_masks[part]
            removed = before - np.sum(final_mask)
            if removed > 0:
                log.info("  - %-12s: removed %.1f%%", part, removed / (h * w) * 100)

    # Morphological cleanup
    if cleanup != "none":
        log.info("Morphological cleanup: %s", cleanup)
        final_mask = morphological_cleanup(final_mask, (h, w), operation=cleanup)

    final_pct = np.sum(final_mask) / (h * w) * 100
    log.info("Final mask: %.1f%% of image", final_pct)

    # Feather edges
    if feather > 0:
        log.info("Feathering edges: %.1f%% radius", feather)
        final_float = feather_mask(final_mask, (h, w), radius_pct=feather)
    else:
        final_float = final_mask.astype(np.float32)

    return final_float, individual_masks


def save_debug_masks(individual_masks, output_dir, img_size):
    """Save individual mask PNGs for inspection."""
    w, h = img_size
    for name, mask in individual_masks.items():
        if not np.any(mask):
            continue
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8, "L")
        if mask_img.size != (w, h):
            mask_img = mask_img.resize((w, h), Image.NEAREST)
        path = os.path.join(output_dir, f"mask_{name}.png")
        mask_img.save(path)
        log.info("  Saved: mask_%s.png", name)


def apply_mask_to_image(img, mask_float, bg_color=(0, 0, 0)):
    """Apply float mask to image, with optional background color."""
    img_array = np.array(img).astype(np.float32)
    mask_3d = mask_float[:, :, np.newaxis]
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)

    result = img_array * mask_3d + bg * (1 - mask_3d)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-grained body part segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parts: face-skin, body-skin, hair, clothes, others, background
Special: skin (= face-skin + body-skin), all (= everything except background)
Subtractable: hands, ropes, hair, clothes, others, background

Examples:
  %(prog)s --source photo.jpg --include face-skin,body-skin --exclude hands,ropes
  %(prog)s --source photo.jpg --include skin --exclude hands,ropes,hair
  %(prog)s --source photo.jpg --include all --exclude background
        """,
    )
    parser.add_argument("--source", required=True, help="Source image path")
    parser.add_argument("--include", default="skin",
                        help="Comma-separated parts to include (default: skin)")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated parts to exclude (default: none)")
    parser.add_argument("--rope-color", default="auto",
                        choices=["auto", "red", "beige", "black", "white"],
                        help="Rope color for detection (default: auto)")
    parser.add_argument("--feather", type=float, default=0.5,
                        help="Edge feather radius as %% of short edge (default: 0.5, 0=hard)")
    parser.add_argument("--cleanup", default="smooth",
                        choices=["close", "open", "smooth", "none"],
                        help="Morphological cleanup mode (default: smooth)")
    parser.add_argument("--debug", action="store_true",
                        help="Save individual mask PNGs for inspection")
    parser.add_argument("--output-to", default="local",
                        choices=["local"],
                        help="Output destination")
    parser.add_argument("--local-output-dir",
                        default=os.path.expanduser("~/.openclaw/workspace/shared"),
                        help="Local output directory")
    parser.add_argument("--bg-color", default="black",
                        choices=["black", "white", "transparent"],
                        help="Background color for masked result (default: black)")

    args = parser.parse_args()

    # Parse include/exclude
    include_parts = set(p.strip() for p in args.include.split(",") if p.strip())
    exclude_parts = set(p.strip() for p in args.exclude.split(",") if p.strip())

    # Validate part names
    for p in include_parts | exclude_parts:
        if p not in VALID_PARTS:
            parser.error(f"Unknown part '{p}'. Valid: {', '.join(sorted(VALID_PARTS))}")

    # Load image
    source_path = os.path.expanduser(args.source)
    if not os.path.exists(source_path):
        log.error("Source not found: %s", source_path)
        sys.exit(1)

    img = Image.open(source_path).convert("RGB")
    w, h = img.size
    log.info("Source: %s (%dx%d)", os.path.basename(source_path), w, h)

    # Create output directory
    photo_id = Path(source_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inc_str = "+".join(sorted(include_parts))
    exc_str = "-".join(sorted(exclude_parts)) if exclude_parts else "none"
    out_name = f"{photo_id}_{timestamp}_segment_{inc_str}_ex_{exc_str}"
    output_dir = os.path.join(os.path.expanduser(args.local_output_dir), out_name)
    os.makedirs(output_dir, exist_ok=True)
    log.info("Output: %s", output_dir)

    # Save original
    img.save(os.path.join(output_dir, "0_original.jpg"), quality=95)

    # Step 1: Multiclass segmentation
    t0 = time.time()
    log.info("--- Step 1: Multiclass body segmentation ---")
    img_array = np.array(img)
    cat_mask = segment_body(img_array)
    log.info("Step 1 done (%.1fs)", time.time() - t0)

    # Step 2: Build composite mask
    t1 = time.time()
    log.info("--- Step 2: Build mask (include=%s, exclude=%s) ---",
             ",".join(sorted(include_parts)), ",".join(sorted(exclude_parts)))
    final_mask, individual_masks = build_mask(
        cat_mask, img_array,
        include_parts=include_parts,
        exclude_parts=exclude_parts,
        rope_color=args.rope_color,
        feather=args.feather,
        cleanup=args.cleanup,
    )
    log.info("Step 2 done (%.1fs)", time.time() - t1)

    # Step 3: Save outputs
    t2 = time.time()
    log.info("--- Step 3: Save outputs ---")

    # Save final mask
    mask_uint8 = (final_mask * 255).clip(0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8, "L")
    mask_img.save(os.path.join(output_dir, "1_mask.png"))
    log.info("  Saved: 1_mask.png")

    # Save masked result
    if args.bg_color == "transparent":
        # RGBA with alpha channel
        result = img.copy().convert("RGBA")
        alpha = Image.fromarray(mask_uint8, "L")
        result.putalpha(alpha)
        result.save(os.path.join(output_dir, "2_masked.png"))
    else:
        bg = (0, 0, 0) if args.bg_color == "black" else (255, 255, 255)
        result = apply_mask_to_image(img, final_mask, bg_color=bg)
        result.save(os.path.join(output_dir, "2_masked.jpg"), quality=95)
    log.info("  Saved: 2_masked.%s", "png" if args.bg_color == "transparent" else "jpg")

    # Save overlay (original with mask tinted)
    overlay_array = img_array.copy().astype(np.float32)
    # Red tint on masked areas
    tint = np.zeros_like(overlay_array)
    tint[:, :, 0] = 255  # red channel
    mask_3d = final_mask[:, :, np.newaxis]
    overlay_array = overlay_array * (1 - mask_3d * 0.4) + tint * mask_3d * 0.4
    overlay = Image.fromarray(overlay_array.clip(0, 255).astype(np.uint8))
    overlay.save(os.path.join(output_dir, "3_overlay.jpg"), quality=95)
    log.info("  Saved: 3_overlay.jpg (red tint shows selected areas)")

    # Save debug masks if requested
    if args.debug:
        log.info("  Saving debug masks...")
        save_debug_masks(individual_masks, output_dir, (w, h))

    log.info("Step 3 done (%.1fs)", time.time() - t2)

    # Save manifest
    manifest = {
        "source": source_path,
        "include": sorted(include_parts),
        "exclude": sorted(exclude_parts),
        "rope_color": args.rope_color,
        "feather": args.feather,
        "cleanup": args.cleanup,
        "bg_color": args.bg_color,
        "mask_coverage_pct": float(np.mean(final_mask) * 100),
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t0
    log.info("=" * 60)
    log.info("DONE in %.1fs — mask covers %.1f%% of image",
             total_time, manifest["mask_coverage_pct"])
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    return output_dir


if __name__ == "__main__":
    main()
