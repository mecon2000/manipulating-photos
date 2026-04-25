#!/usr/bin/env python3
"""Remove persistent IG overlays (username text, status bar, like/comment bar)
from a stack of screenshots that share the same chrome.

Method:
  1. Resize all images to the same dimensions.
  2. Compute pixel-wise std across the stack — pixels that are identical
     across all screenshots are the persistent overlay.
  3. Threshold + dilate → mask. cv2.inpaint each image with that mask.

Usage:
  clean_ig_screenshots.py [--in DIR] [--out DIR] [--threshold N] [--dilate N]
"""
import argparse, os, sys
import numpy as np
import cv2
from PIL import Image

DEFAULT_IN = os.path.expanduser("~/.openclaw/workspace/shared/0010x0010")
DEFAULT_OUT = os.path.expanduser("~/.openclaw/workspace/shared/0010x0010/cleaned")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default=DEFAULT_IN)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--threshold", type=float, default=5.0,
                   help="pixels with cross-image std < this are 'persistent overlay'")
    p.add_argument("--dilate", type=int, default=3,
                   help="dilation iterations (catches anti-aliased text edges)")
    p.add_argument("--save-mask", action="store_true",
                   help="also save the detected overlay mask for inspection")
    p.add_argument("--crop-row-threshold", type=float, default=0.5,
                   help="rows where >X fraction is mask get cropped from top/bottom (default 0.5)")
    p.add_argument("--rect", action="append", default=[],
                   help='extra rect to add to mask: "x,y,w,h" in pixels (or use pct: "5%%,80%%,20%%,15%%"); '
                        'pass multiple times for multiple rects')
    args = p.parse_args()

    files = [f for f in sorted(os.listdir(args.in_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))
             and os.path.isfile(os.path.join(args.in_dir, f))]
    if len(files) < 2:
        print(f"need >= 2 images in {args.in_dir} (got {len(files)})", file=sys.stderr)
        sys.exit(1)

    # Use first image's dims as target
    first = Image.open(os.path.join(args.in_dir, files[0])).convert("RGB")
    W, H = first.size
    print(f"target size from first: {W}x{H}")

    arrs = []
    pil_imgs = []
    for f in files:
        img = Image.open(os.path.join(args.in_dir, f)).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.LANCZOS)
        pil_imgs.append(img)
        arrs.append(np.asarray(img))
    stack = np.stack(arrs, axis=0).astype(np.float32)
    std = stack.std(axis=0).mean(axis=2)   # H × W
    mask_bool = std < args.threshold
    mask = (mask_bool.astype(np.uint8)) * 255
    if args.dilate > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=args.dilate)

    def parse_dim(s, total):
        s = s.strip()
        return int(round(float(s.rstrip("%")) * total / 100)) if s.endswith("%") else int(s)
    for r in args.rect:
        try:
            xs, ys, ws, hs = [t.strip() for t in r.split(",")]
            x = parse_dim(xs, W); y = parse_dim(ys, H)
            ww = parse_dim(ws, W); hh = parse_dim(hs, H)
            mask[y:y+hh, x:x+ww] = 255
            print(f"  +rect ({x},{y},{ww},{hh})")
        except Exception as e:
            print(f"bad --rect '{r}': {e}", file=sys.stderr)

    # Crop off contiguous chrome bands at top/bottom (rows that are mostly mask).
    # Those bands have no useful photo content, so cropping is cleaner than inpaint.
    row_mask_pct = mask.mean(axis=1) / 255.0
    crop_row_threshold = args.crop_row_threshold
    top = 0
    while top < H and row_mask_pct[top] > crop_row_threshold:
        top += 1
    bottom = H
    while bottom > top + 10 and row_mask_pct[bottom - 1] > crop_row_threshold:
        bottom -= 1
    print(f"crop top→{top}, bottom→{bottom} (rows >{crop_row_threshold*100:.0f}% mask)")

    pct = 100.0 * mask_bool.mean()
    print(f"overlay mask: {pct:.1f}% of pixels (std<{args.threshold})")

    os.makedirs(args.out, exist_ok=True)
    mask_cropped = mask[top:bottom]
    if args.save_mask:
        Image.fromarray(mask_cropped).save(os.path.join(args.out, "_overlay_mask.png"))

    for f, img in zip(files, pil_imgs):
        # Crop chrome bands first, inpaint only the in-photo overlays remaining
        cropped = img.crop((0, top, W, bottom))
        bgr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, mask_cropped, inpaintRadius=5,
                                flags=cv2.INPAINT_TELEA)
        out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        Image.fromarray(out).save(os.path.join(args.out, f), quality=95)
        print(f"  ✔ {f}")


if __name__ == "__main__":
    main()
