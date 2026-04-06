#!/home/rong/openclaw-venv/bin/python3
"""
Find candidate photos for stylization.

Scans _photos directory, picks random processed photos from different models,
copies them to a candidates folder with metadata. Designed to run without
an LLM — just pick candidates, review on Windows, tell the LLM which to use.

Usage:
    ./find-candidates.py                          # 5 random candidates
    ./find-candidates.py --count 10               # 10 candidates
    ./find-candidates.py --models "Anya,Jana"     # specific models only
    ./find-candidates.py --exclude "Michaela"     # skip specific models
    ./find-candidates.py --min-size 500000        # only photos > 500KB
    ./find-candidates.py --prefer-full-body       # prefer photos with more BG (larger files)
    ./find-candidates.py --output-dir /path/to    # custom output location
"""

import os
import sys
import json
import random
import shutil
import argparse
from datetime import datetime, timedelta, timezone
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

# Default paths
PHOTOS_DIR = os.path.expanduser("~/.openclaw/workspace/_photos")
SHARED_DIR = os.path.expanduser("~/.openclaw/workspace/shared")

SKIP_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}


def find_all_models(photos_dir):
    """Return list of (model_name, processed_dir) for all models with processed photos."""
    models = []
    try:
        entries = os.listdir(photos_dir)
    except OSError:
        return models

    for name in entries:
        model_dir = os.path.join(photos_dir, name)
        if not os.path.isdir(model_dir):
            continue
        if name.lower() in SKIP_FILES or name.endswith(".gdoc") or name.endswith(".bat"):
            continue

        # Prefer Processed, fall back to Unprocessed
        for sub in ["Processed", "processed", "Unprocessed", "unprocessed"]:
            sub_dir = os.path.join(model_dir, sub)
            if os.path.isdir(sub_dir):
                photos = [f for f in os.listdir(sub_dir)
                          if os.path.splitext(f)[1].lower() in PHOTO_EXTS
                          and f.lower() not in SKIP_FILES]
                if photos:
                    models.append((name, sub_dir, sub.lower().startswith("processed")))
                    break

    return models


def get_focal_length(photo_path):
    """Read focal length from EXIF data. Returns float (mm) or None."""
    try:
        img = Image.open(photo_path)
        exif = img._getexif()
        img.close()
        if exif:
            for tag_id, val in exif.items():
                if TAGS.get(tag_id) == "FocalLength":
                    return float(val)
    except Exception:
        pass
    return None


def get_photo_info(photo_path, model_name):
    """Get metadata for a photo."""
    stat = os.stat(photo_path)
    focal = get_focal_length(photo_path)
    return {
        "model": model_name,
        "filename": os.path.basename(photo_path),
        "path": photo_path,
        "size_kb": round(stat.st_size / 1024),
        "focal_mm": focal,
        "is_processed": "Processed" in photo_path or "processed" in photo_path,
    }


def pick_candidates(photos_dir, count=5, include_models=None, exclude_models=None,
                     min_size=0, prefer_full_body=False, per_model=1,
                     max_focal=None, min_focal=None):
    """Pick candidate photos from different models."""
    all_models = find_all_models(photos_dir)

    if not all_models:
        print(f"ERROR: No models found in {photos_dir}")
        return []

    # Filter models
    if include_models:
        include_set = {m.strip().lower() for m in include_models}
        all_models = [(n, d, p) for n, d, p in all_models if n.lower() in include_set]
    if exclude_models:
        exclude_set = {m.strip().lower() for m in exclude_models}
        all_models = [(n, d, p) for n, d, p in all_models if n.lower() not in exclude_set]

    if not all_models:
        print("ERROR: No models match the filter criteria")
        return []

    # Shuffle models to get variety
    random.shuffle(all_models)

    candidates = []
    models_used = set()

    for model_name, photo_dir, is_processed in all_models:
        if len(candidates) >= count:
            break
        if model_name in models_used and len(all_models) > count:
            continue  # Try to get different models first

        photos = [f for f in os.listdir(photo_dir)
                  if os.path.splitext(f)[1].lower() in PHOTO_EXTS
                  and f.lower() not in SKIP_FILES]

        # Filter by size
        if min_size > 0:
            photos = [f for f in photos
                      if os.path.getsize(os.path.join(photo_dir, f)) >= min_size]

        # Filter by focal length
        if max_focal is not None or min_focal is not None:
            filtered = []
            for f in photos:
                focal = get_focal_length(os.path.join(photo_dir, f))
                if focal is None:
                    continue  # Skip photos without EXIF focal length
                if max_focal is not None and focal > max_focal:
                    continue
                if min_focal is not None and focal < min_focal:
                    continue
                filtered.append(f)
            photos = filtered

        if not photos:
            continue

        # Sort by size descending if preferring full-body (larger files = more scene)
        if prefer_full_body:
            photos.sort(key=lambda f: os.path.getsize(os.path.join(photo_dir, f)), reverse=True)
            picks = photos[:per_model]
        else:
            picks = random.sample(photos, min(per_model, len(photos)))

        for pick in picks:
            if len(candidates) >= count:
                break
            photo_path = os.path.join(photo_dir, pick)
            info = get_photo_info(photo_path, model_name)
            candidates.append(info)
            models_used.add(model_name)

    return candidates


def copy_candidates(candidates, output_dir):
    """Copy candidate photos to output directory with clean names."""
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for i, c in enumerate(candidates, 1):
        # Clean name: ModelName_OriginalFilename
        clean_model = c["model"].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        ext = os.path.splitext(c["filename"])[1]
        dest_name = f"{clean_model}_{os.path.splitext(c['filename'])[0]}{ext}"
        dest_path = os.path.join(output_dir, dest_name)

        try:
            with open(c["path"], "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
        except OSError as e:
            print(f"  WARN: Failed to copy {c['filename']}: {e}")
            continue

        focal_str = f", {c['focal_mm']:.0f}mm" if c.get("focal_mm") else ""
        entry = {
            "index": i,
            "dest_name": dest_name,
            "model": c["model"],
            "original": c["filename"],
            "source_path": c["path"],
            "size_kb": c["size_kb"],
            "focal_mm": c.get("focal_mm"),
            "processed": c["is_processed"],
        }
        manifest.append(entry)
        print(f"  {i}. {dest_name} ({c['size_kb']}KB{focal_str}) — {c['model']}")

    # Write manifest
    manifest_path = os.path.join(output_dir, "candidates.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "created": datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(manifest),
            "candidates": manifest,
        }, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Find candidate photos for stylization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=5, help="Number of candidates to pick (default: 5)")
    parser.add_argument("--models", default=None, help="Comma-separated model names to include (default: all)")
    parser.add_argument("--exclude", default=None, help="Comma-separated model names to exclude")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum file size in bytes")
    parser.add_argument("--prefer-full-body", action="store_true", help="Prefer larger files (more background/full body)")
    parser.add_argument("--photos-dir", default=PHOTOS_DIR, help=f"Photos directory (default: {PHOTOS_DIR})")
    parser.add_argument("--output-dir", default=None, help=f"Output directory (default: {SHARED_DIR}/candidates)")
    parser.add_argument("--max-focal", type=float, default=None, help="Max focal length in mm (e.g. 50 for framing-friendly)")
    parser.add_argument("--min-focal", type=float, default=None, help="Min focal length in mm")
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit")

    args = parser.parse_args()

    if args.list_models:
        models = find_all_models(args.photos_dir)
        print(f"\n{len(models)} models with photos:\n")
        for name, photo_dir, is_processed in sorted(models):
            count = len([f for f in os.listdir(photo_dir)
                        if os.path.splitext(f)[1].lower() in PHOTO_EXTS
                        and f.lower() not in SKIP_FILES])
            tag = "Processed" if is_processed else "Unprocessed"
            print(f"  {name:<35} {count:>3} photos ({tag})")
        sys.exit(0)

    # Pick candidates
    include = args.models.split(",") if args.models else None
    exclude = args.exclude.split(",") if args.exclude else None

    print(f"\nPicking {args.count} candidates from {args.photos_dir}...\n")
    candidates = pick_candidates(
        args.photos_dir, args.count,
        include_models=include,
        exclude_models=exclude,
        min_size=args.min_size,
        prefer_full_body=args.prefer_full_body,
        max_focal=args.max_focal,
        min_focal=args.min_focal,
    )

    if not candidates:
        print("No candidates found!")
        sys.exit(1)

    # Copy to output
    output_dir = args.output_dir or os.path.join(SHARED_DIR, "candidates")
    print(f"Copying to {output_dir}:\n")
    manifest = copy_candidates(candidates, output_dir)

    print(f"\n{len(manifest)} candidates ready in: {output_dir}")
    print(f"Manifest: {os.path.join(output_dir, 'candidates.json')}")
    print("\nReview on Windows, then tell the LLM which to stylize (by number or name).")


if __name__ == "__main__":
    main()
