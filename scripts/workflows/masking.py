"""
masking.py — Shared mask-building module for OpenClaw photo transformation tools.

Centralises all mask construction logic: BiRefNet (fal.ai API), MediaPipe body
segmentation (via body-segment.py), and mask post-processing (feathering,
morphological cleanup, resize).

Usage from any workflow script:

    from masking import add_affect_args, build_mask

    # In CLI setup:
    add_affect_args(parser)

    # At runtime:
    mask_pil, mask_info = build_mask(
        img_path_or_pil="photo.jpg",
        affect="skin",
        exclude="hands,ropes",
        output_dir="/tmp/run_001",
    )
    # mask_pil: L-mode PIL Image, 0=excluded, 255=included
    # mask_info: dict with engine, coverage_pct, parts, etc.
"""

import os
import sys
import base64
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
from PIL import Image, ImageFilter, ImageOps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Parts that require BiRefNet (fal.ai API, ~5s, excellent edges)
BIREFNET_PARTS = {"bg", "subject"}

# Parts that come from MediaPipe body-segment (local, ~0.5s)
BODYSEG_PARTS = {"face-skin", "body-skin", "hair", "clothes", "others"}

# Shortcut expansions
SHORTCUTS = {
    "skin": {"face-skin", "body-skin"},
    "all": set(),  # empty = full-white mask, no API call
}

# Valid exclusion targets
EXCLUDE_PARTS = {"hands", "ropes", "hair", "clothes", "others", "background"}

# All valid part names (for argument validation)
ALL_VALID_PARTS = BIREFNET_PARTS | BODYSEG_PARTS | set(SHORTCUTS.keys()) | EXCLUDE_PARTS

_log_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(output_dir, message, level="INFO"):
    """Thread-safe logging to stdout + workflow.log in output_dir."""
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
                pass  # output_dir may not exist yet during early init


# ---------------------------------------------------------------------------
# Env / API key helpers
# ---------------------------------------------------------------------------

def _ensure_env():
    """Load API keys from ~/sol/.env if not already set."""
    env_file = os.path.expanduser("~/sol/.env")
    if os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def _get_fal_key():
    _ensure_env()
    key = os.environ.get("FAL_API_KEY")
    if not key:
        raise EnvironmentError("FAL_API_KEY not set (checked ~/sol/.env and env)")
    return key


# ---------------------------------------------------------------------------
# argparse integration
# ---------------------------------------------------------------------------

def add_affect_args(parser):
    """Add --affect and --exclude arguments to an argparse parser.

    Call this from any tool's CLI setup to get standardised mask targeting.
    """
    parser.add_argument(
        "--affect", default="subject",
        help=(
            "Comma-separated body parts to target. "
            "Options: subject, bg, all, skin, face-skin, body-skin, hair, clothes, others. "
            "(default: subject)"
        ),
    )
    parser.add_argument(
        "--exclude", default="",
        help=(
            "Comma-separated parts to subtract from the mask. "
            "Options: hands, ropes, hair, clothes, others, background. "
            "(default: none)"
        ),
    )


# ---------------------------------------------------------------------------
# BiRefNet / rembg — fal.ai API
# ---------------------------------------------------------------------------

def _image_to_base64(img_or_path):
    """Convert a file path or PIL Image to a base64 JPEG string."""
    if isinstance(img_or_path, str):
        with open(img_or_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    else:
        buf = BytesIO()
        img_or_path.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_fal_birefnet(img_or_path, output_dir):
    """Extract foreground mask using BiRefNet (high quality edges).

    Args:
        img_or_path: file path (str) or PIL Image
        output_dir: for logging

    Returns:
        PIL L-mode mask or None on failure.
    """
    log(output_dir, "Extracting mask using BiRefNet...")
    headers = {
        "Authorization": f"Key {_get_fal_key()}",
        "Content-Type": "application/json",
    }
    img_b64 = _image_to_base64(img_or_path)

    try:
        response = requests.post(
            "https://fal.run/fal-ai/birefnet",
            headers=headers,
            json={"image_url": f"data:image/jpeg;base64,{img_b64}"},
            timeout=180,
        )
    except requests.RequestException as e:
        log(output_dir, f"BiRefNet request failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"BiRefNet failed ({response.status_code}): {response.text}", "ERROR")
        return None

    data = response.json()
    result_url = data["image"]["url"]
    result_img = Image.open(requests.get(result_url, stream=True, timeout=30).raw)
    if result_img.mode == "RGBA":
        return result_img.split()[3]
    else:
        return result_img.convert("L")


def run_fal_rembg(img_or_path, output_dir):
    """Fallback mask extraction using rembg via fal.ai.

    Args:
        img_or_path: file path (str) or PIL Image
        output_dir: for logging

    Returns:
        PIL L-mode mask or None on failure.
    """
    log(output_dir, "Extracting mask using rembg (fallback)...")
    headers = {
        "Authorization": f"Key {_get_fal_key()}",
        "Content-Type": "application/json",
    }
    img_b64 = _image_to_base64(img_or_path)

    try:
        response = requests.post(
            "https://fal.run/fal-ai/rembg",
            headers=headers,
            json={"image_url": f"data:image/jpeg;base64,{img_b64}"},
            timeout=180,
        )
    except requests.RequestException as e:
        log(output_dir, f"rembg request failed: {e}", "ERROR")
        return None

    if response.status_code != 200:
        log(output_dir, f"rembg failed ({response.status_code}): {response.text}", "ERROR")
        return None

    mask_url = response.json()["image"]["url"]
    mask_img = Image.open(requests.get(mask_url, stream=True, timeout=30).raw)
    if mask_img.mode == "RGBA":
        return mask_img.split()[3]
    return mask_img.convert("L")


def _extract_birefnet_mask(img_or_path, output_dir):
    """Extract foreground mask. Tries BiRefNet first, falls back to rembg."""
    mask = run_fal_birefnet(img_or_path, output_dir)
    if mask is not None:
        return mask
    log(output_dir, "BiRefNet failed, falling back to rembg", "WARN")
    return run_fal_rembg(img_or_path, output_dir)


# ---------------------------------------------------------------------------
# Body-segment integration (dynamic import of sibling script)
# ---------------------------------------------------------------------------

_body_segment_mod = None
_body_segment_lock = threading.Lock()


def _import_body_segment():
    """Import functions from body-segment.py (sibling script). Cached."""
    global _body_segment_mod
    if _body_segment_mod is not None:
        return _body_segment_mod
    with _body_segment_lock:
        if _body_segment_mod is not None:
            return _body_segment_mod
        import importlib.util
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bs_path = os.path.join(script_dir, "body-segment.py")
        if not os.path.exists(bs_path):
            raise FileNotFoundError(f"body-segment.py not found at {bs_path}")
        spec = importlib.util.spec_from_file_location("body_segment", bs_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _body_segment_mod = mod
        return mod


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------

def feather_edges(mask_pil, img_size, radius_pct=0.5):
    """Gaussian-blur mask edges for soft transitions.

    Args:
        mask_pil: L-mode PIL Image
        img_size: (width, height) of the full image
        radius_pct: blur radius as percentage of the short edge

    Returns:
        L-mode PIL Image with feathered edges.
    """
    w, h = img_size
    short_edge = min(w, h)
    radius = max(1, int(short_edge * radius_pct / 100))
    return mask_pil.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# build_mask — main entry point
# ---------------------------------------------------------------------------

def build_mask(img_path_or_pil, affect, exclude="", output_dir=None,
               rope_color="auto", feather=0.5, cleanup="smooth"):
    """Build a mask for the requested target parts.

    Args:
        img_path_or_pil: file path (str/os.PathLike) or PIL Image (RGB).
        affect: comma-separated string of parts to include, e.g.
                "skin", "bg", "subject", "face-skin,body-skin", "all".
        exclude: comma-separated string of parts to subtract, e.g.
                 "hands", "hands,ropes", or "" for none.
        output_dir: directory for logging and optional debug mask saves.
                    If None, logging goes to stdout only.
        rope_color: rope colour for HSV detection ("auto", "red", "beige",
                    "black", "white").
        feather: edge feather radius as % of short edge (0 = hard edges).
        cleanup: morphological cleanup mode ("close", "open", "smooth", "none").

    Returns:
        (mask_pil, mask_info) where:
          - mask_pil: L-mode PIL Image, 0=excluded, 255=included,
                      same size as input image.
          - mask_info: dict with keys: engine, coverage_pct, parts_detected,
                       affect_resolved, exclude_resolved.
    """
    # -- Resolve input image and path ----------------------------------------
    if isinstance(img_path_or_pil, (str, os.PathLike)):
        image_path = str(img_path_or_pil)
        orig_img = Image.open(image_path).convert("RGB")
    else:
        orig_img = img_path_or_pil.convert("RGB")
        image_path = None  # will use PIL Image directly for API calls

    w, h = orig_img.size

    # -- Parse affect / exclude strings --------------------------------------
    parts = {p.strip().lower() for p in affect.split(",") if p.strip()}
    exclude_set = {p.strip().lower() for p in exclude.split(",") if p.strip()}

    # Expand shortcuts in affect
    expanded = set()
    for p in parts:
        if p in SHORTCUTS:
            if SHORTCUTS[p]:  # non-empty expansion (e.g. "skin")
                expanded |= SHORTCUTS[p]
            else:
                expanded.add(p)  # "all" stays as sentinel
        else:
            expanded.add(p)
    parts = expanded

    mask_info = {
        "engine": None,
        "coverage_pct": 0.0,
        "parts_detected": [],
        "affect_resolved": sorted(parts),
        "exclude_resolved": sorted(exclude_set),
    }

    # Helper: the image source for API calls (path preferred, PIL fallback)
    api_src = image_path if image_path else orig_img

    # -- "all" shortcut: full white mask, no API call ------------------------
    if "all" in parts:
        log(output_dir, "Affect=all: full-image mask (no segmentation needed)")
        mask_pil = Image.new("L", (w, h), 255)
        mask_info["engine"] = "none"
        mask_info["coverage_pct"] = 100.0
        return mask_pil, mask_info

    # -- Determine which engine(s) we need -----------------------------------
    birefnet_parts = parts & BIREFNET_PARTS
    segment_parts = parts & (BODYSEG_PARTS | {"skin"})

    # Re-expand "skin" in segment_parts (in case it survived)
    if "skin" in segment_parts:
        segment_parts = (segment_parts - {"skin"}) | {"face-skin", "body-skin"}
        parts = (parts - {"skin"}) | {"face-skin", "body-skin"}

    # -- BiRefNet path (bg / subject) ----------------------------------------
    if birefnet_parts and not segment_parts:
        engine_name = "birefnet"
        log(output_dir, f"Affect={','.join(sorted(birefnet_parts))}: using BiRefNet")
        mask = _extract_birefnet_mask(api_src, output_dir)

        if mask is None:
            log(output_dir, "All mask extraction failed — returning full-image mask", "WARN")
            mask_pil = Image.new("L", (w, h), 255)
            mask_info["engine"] = "birefnet-failed"
            mask_info["coverage_pct"] = 100.0
            return mask_pil, mask_info

        mask = mask.resize((w, h), Image.LANCZOS)

        if "bg" in birefnet_parts:
            log(output_dir, "Inverting mask for background target")
            mask = ImageOps.invert(mask)

        # Feather if requested
        if feather > 0:
            mask = feather_edges(mask, (w, h), radius_pct=feather)

        coverage = np.mean(np.array(mask) > 127) * 100
        log(output_dir, f"BiRefNet mask coverage: {coverage:.1f}%")

        # Save debug mask
        if output_dir:
            try:
                mask.save(os.path.join(output_dir, "mask_affect.png"))
            except OSError:
                pass

        mask_info["engine"] = engine_name
        mask_info["coverage_pct"] = round(coverage, 1)
        mask_info["parts_detected"] = sorted(birefnet_parts)
        return mask, mask_info

    # -- MediaPipe body-segment path -----------------------------------------
    if segment_parts:
        log(output_dir, f"Affect={','.join(sorted(segment_parts))}: using MediaPipe body segmentation")
        try:
            bs = _import_body_segment()
        except Exception as e:
            log(output_dir, f"body-segment import failed: {e} — falling back to BiRefNet subject mask", "WARN")
            mask = _extract_birefnet_mask(api_src, output_dir)
            if mask is None:
                mask_pil = Image.new("L", (w, h), 255)
                mask_info["engine"] = "bodyseg-failed"
                mask_info["coverage_pct"] = 100.0
                return mask_pil, mask_info
            mask = mask.resize((w, h), Image.LANCZOS)
            if feather > 0:
                mask = feather_edges(mask, (w, h), radius_pct=feather)
            coverage = np.mean(np.array(mask) > 127) * 100
            mask_info["engine"] = "birefnet-fallback"
            mask_info["coverage_pct"] = round(coverage, 1)
            return mask, mask_info

        img_arr = np.array(orig_img)

        try:
            cat_mask = bs.segment_body(img_arr)
        except Exception as e:
            log(output_dir, f"MediaPipe segmentation failed: {e} — falling back to BiRefNet", "WARN")
            mask = _extract_birefnet_mask(api_src, output_dir)
            if mask is None:
                mask_pil = Image.new("L", (w, h), 255)
                mask_info["engine"] = "bodyseg-failed"
                mask_info["coverage_pct"] = 100.0
                return mask_pil, mask_info
            mask = mask.resize((w, h), Image.LANCZOS)
            if feather > 0:
                mask = feather_edges(mask, (w, h), radius_pct=feather)
            coverage = np.mean(np.array(mask) > 127) * 100
            mask_info["engine"] = "birefnet-fallback"
            mask_info["coverage_pct"] = round(coverage, 1)
            return mask, mask_info

        include_set = set(segment_parts)

        float_mask, indiv = bs.build_mask(
            cat_mask, img_arr, include_set, exclude_set,
            rope_color=rope_color, feather=feather, cleanup=cleanup,
        )

        # Convert float32 0-1 to uint8 0-255 L-mode PIL
        mask_uint8 = (float_mask * 255).clip(0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, "L")

        # Resize if body-segment returned 256x256 model output
        if mask_pil.size != (w, h):
            mask_pil = mask_pil.resize((w, h), Image.LANCZOS)

        coverage = np.mean(np.array(mask_pil) > 127) * 100
        log(output_dir, f"Body-segment mask coverage: {coverage:.1f}%")

        # Save debug masks
        if output_dir:
            try:
                mask_pil.save(os.path.join(output_dir, "mask_affect.png"))
                for name, m in indiv.items():
                    if np.any(m):
                        m_img = Image.fromarray((m * 255).astype(np.uint8), "L")
                        if m_img.size != (w, h):
                            m_img = m_img.resize((w, h), Image.NEAREST)
                        m_img.save(os.path.join(output_dir, f"mask_{name}.png"))
            except OSError:
                pass

        detected = [name for name, m in indiv.items() if np.any(m)]
        mask_info["engine"] = "mediapipe-bodyseg"
        mask_info["coverage_pct"] = round(coverage, 1)
        mask_info["parts_detected"] = sorted(detected)
        return mask_pil, mask_info

    # -- Fallback: unrecognised parts → BiRefNet subject ---------------------
    log(output_dir, f"Unknown affect parts {parts} — falling back to BiRefNet subject mask", "WARN")
    mask = _extract_birefnet_mask(api_src, output_dir)
    if mask is None:
        mask_pil = Image.new("L", (w, h), 255)
        mask_info["engine"] = "fallback-failed"
        mask_info["coverage_pct"] = 100.0
        return mask_pil, mask_info

    mask = mask.resize((w, h), Image.LANCZOS)
    if feather > 0:
        mask = feather_edges(mask, (w, h), radius_pct=feather)
    coverage = np.mean(np.array(mask) > 127) * 100
    mask_info["engine"] = "birefnet-fallback"
    mask_info["coverage_pct"] = round(coverage, 1)
    return mask, mask_info
