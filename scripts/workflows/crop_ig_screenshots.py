#!/usr/bin/env python3
"""Auto-crop Instagram screenshots to the post photo region.

Detects horizontal UI bands by row variance: UI rows (status bar, post header,
like/comment bar) tend to be near-uniform color (low row variance). Photo rows
have high variance. Find the longest run of high-variance rows → crop to that.

Usage:
  crop_ig_screenshots.py [--in DIR] [--out DIR]
    default --in  ~/.openclaw/workspace/shared/0010x0010
    default --out ~/.openclaw/workspace/shared/0010x0010/cropped
"""
import argparse, os, sys
import numpy as np
from PIL import Image

DEFAULT_IN = os.path.expanduser("~/.openclaw/workspace/shared/0010x0010")
DEFAULT_OUT = os.path.expanduser("~/.openclaw/workspace/shared/0010x0010/cropped")


def find_photo_bounds(arr, var_threshold_pct=20):
    """Return (y0, y1) of the longest contiguous high-variance vertical band."""
    h = arr.shape[0]
    row_var = arr.astype(np.float32).std(axis=(1, 2))
    threshold = np.percentile(row_var, var_threshold_pct)
    is_photo = row_var > threshold
    # find longest True run
    best_len = 0
    best_y0, best_y1 = 0, h
    cur_y0 = None
    for y in range(h):
        if is_photo[y]:
            if cur_y0 is None:
                cur_y0 = y
            cur_y1 = y + 1
        else:
            if cur_y0 is not None:
                if cur_y1 - cur_y0 > best_len:
                    best_len = cur_y1 - cur_y0
                    best_y0, best_y1 = cur_y0, cur_y1
                cur_y0 = None
    if cur_y0 is not None and cur_y1 - cur_y0 > best_len:
        best_y0, best_y1 = cur_y0, cur_y1
    return best_y0, best_y1


def crop_one(path, out_path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    y0, y1 = find_photo_bounds(arr)
    # also crop horizontal: same logic on columns within the band
    band = arr[y0:y1]
    col_var = band.astype(np.float32).std(axis=(0, 2))
    threshold = np.percentile(col_var, 20)
    is_photo = col_var > threshold
    nz = np.where(is_photo)[0]
    x0, x1 = (int(nz.min()), int(nz.max()) + 1) if len(nz) else (0, w)
    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(out_path, quality=95)
    return (x0, y0, x1, y1), cropped.size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default=DEFAULT_IN)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = [f for f in sorted(os.listdir(args.in_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))
             and not f.startswith("cropped")]
    if not files:
        print(f"No images in {args.in_dir}")
        sys.exit(1)
    for f in files:
        src = os.path.join(args.in_dir, f)
        dst = os.path.join(args.out, f)
        bbox, size = crop_one(src, dst)
        print(f"  {f}  bbox={bbox}  → {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
