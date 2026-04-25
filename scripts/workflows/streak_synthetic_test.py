#!/usr/bin/env python3
"""Synthetic streak test: white line on black, streaked to the left.

Lets us study the streak mechanism in isolation, without pose/mask pipeline.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.expanduser("~/.openclaw/workspace/shared/motion-streak-finals")
os.makedirs(OUT, exist_ok=True)


def make_line_image(w=1024, h=1024, angle_deg=100, thickness=3):
    """Black bg, white line through center at angle_deg from horizontal."""
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2
    L = max(w, h)
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), -np.sin(rad)  # screen y is down
    p1 = (cx - dx * L, cy - dy * L)
    p2 = (cx + dx * L, cy + dy * L)
    d.line([p1, p2], fill=(255, 255, 255), width=thickness)
    return img


def streak_cv2(img_arr, length_px, angle_deg, decay=2.0):
    """One-sided directional motion blur via cv2.filter2D.

    Builds an LxL kernel with a line of decaying weights starting from the
    kernel center, drawn at angle_deg. Convolution then averages each pixel
    with neighbors in that direction → trail extends in that direction.
    """
    import cv2
    L = max(3, int(length_px))
    K = L * 2 + 1   # odd, big enough so center can fit a half-line
    kernel = np.zeros((K, K), dtype=np.float32)
    cx = cy = K // 2
    rad = np.deg2rad(angle_deg)
    # tail extends *toward* angle_deg from center
    for t in range(L):
        x = int(round(cx + np.cos(rad) * t))
        y = int(round(cy - np.sin(rad) * t))   # screen y down
        if 0 <= x < K and 0 <= y < K:
            kernel[y, x] = np.exp(-decay * t / L)
    # NOT normalized — raw exp() weights so a white source pixel produces
    # a (near-)white trail pixel one step along, fading to dim at tail.
    out = cv2.filter2D(img_arr.astype(np.float32), -1, kernel,
                       borderType=cv2.BORDER_CONSTANT)
    return np.clip(out, 0, 255).astype(np.uint8)


def streak_left_convolve(img_arr, length_px, decay=2.0):
    """True directional motion blur via one-sided 1D convolution.

    Builds a horizontal kernel of length L. Sample weights decay along the
    trail (exponential), so the head is bright and the tail fades smoothly.
    Kernel is OFF-CENTER (anchored at the right edge) so the trail extends
    only to the LEFT of each bright source pixel.
    """
    from scipy.ndimage import convolve
    L = int(length_px)
    # weights[0] = brightest (head); weights[L-1] = faintest (tail)
    weights = np.exp(-decay * np.arange(L) / L).astype(np.float32)
    weights /= weights.sum()  # normalize so brightness is preserved on average
    kernel = weights[np.newaxis, :]   # 1×L kernel, horizontal
    # Anchor at right end: scipy.ndimage.convolve uses origin=0 (center).
    # Setting origin=-(L//2) shifts the kernel left, so output samples come
    # from positions to the right of the pixel — i.e. trail extends LEFT.
    origin_x = -(L // 2)
    out = np.empty_like(img_arr, dtype=np.float32)
    for c in range(img_arr.shape[2]):
        out[..., c] = convolve(img_arr[..., c].astype(np.float32),
                               kernel, mode="constant", cval=0,
                               origin=(0, origin_x))
    # Convolution averages — with normalized weights, a single bright pixel
    # produces a faint trail. Boost so the head reads as bright.
    out *= L * weights[0] * 1.0  # rescale: head ≈ original brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    line_img = make_line_image(angle_deg=100, thickness=3)
    line_path = os.path.join(OUT, "synth_line_100deg.jpg")
    line_img.save(line_path, quality=95)
    print(f"line  → {line_path}")

    arr = np.asarray(line_img)
    # angle 180 = straight left. cv2 path:
    for L, decay in [(200, 2.0), (400, 2.0), (400, 4.0), (400, 1.0)]:
        streaked = streak_cv2(arr, length_px=L, angle_deg=180, decay=decay)
        out_path = os.path.join(OUT,
            f"synth_line_100deg_cv2_L{L}_d{decay}.jpg")
        Image.fromarray(streaked).save(out_path, quality=95)
        print(f"cv2 L={L} decay={decay} → {out_path}")


if __name__ == "__main__":
    main()
