#!/home/rong/openclaw-venv/bin/python3
"""
Color matching experiments for baroque-surround compositing.
Generates one BG, then outputs many variants with different color matching approaches.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

env_file = os.path.expanduser("~/sol/.env")
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import numpy as np
import cv2
from PIL import Image, ImageFilter
from scipy import ndimage
import fal_client, requests
from io import BytesIO
from masking import build_mask
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")


def generate_bg_once(prompt, w, h, seed=42):
    """Generate a single BG."""
    print("Generating aurora BG...")
    handle = fal_client.submit("fal-ai/flux/dev", arguments={
        "prompt": prompt + ", NO person, NO figure, NO face",
        "image_size": {"width": w, "height": h},
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
        "seed": seed,
    })
    result = handle.get()
    url = result["images"][0]["url"]
    resp = requests.get(url, timeout=60)
    bg = Image.open(BytesIO(resp.content)).convert("RGB")
    print(f"  BG generated: {bg.size}")
    return bg


def basic_composite(orig_arr, bg_arr, mask_f):
    """Tight composite with 2px feather, no color matching."""
    m3 = mask_f[:, :, np.newaxis]
    return np.clip(orig_arr * m3 + bg_arr * (1 - m3), 0, 255).astype(np.uint8)


def lab_transfer(orig_arr, bg_arr, mask_f, strength=0.4):
    """LAB color space transfer at given strength."""
    orig_uint8 = np.clip(orig_arr, 0, 255).astype(np.uint8)
    bg_uint8 = np.clip(bg_arr, 0, 255).astype(np.uint8)
    orig_lab = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    subj = mask_f > 0.5
    bg_m = mask_f < 0.3
    if not (subj.any() and bg_m.any()):
        return orig_arr.copy()

    result_lab = orig_lab.copy()
    for ch in range(3):
        s_mean = orig_lab[:, :, ch][subj].mean()
        s_std = orig_lab[:, :, ch][subj].std() + 1e-8
        b_mean = bg_lab[:, :, ch][bg_m].mean()
        b_std = bg_lab[:, :, ch][bg_m].std() + 1e-8
        new_mean = s_mean + (b_mean - s_mean) * strength
        new_std = s_std + (b_std - s_std) * strength * 0.3
        shifted = (orig_lab[:, :, ch] - s_mean) * (new_std / s_std) + new_mean
        # Gradient: full at edges, partial inside
        edge_w = np.clip(1.0 - (mask_f - 0.2) / 0.6, strength * 0.4, 1.0)
        result_lab[:, :, ch] = orig_lab[:, :, ch] * (1 - edge_w) + shifted * edge_w

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def rgb_color_shift(orig_arr, bg_arr, mask_f, strength=0.3):
    """Simple RGB color shift: blend subject colors toward BG average."""
    subj = mask_f > 0.5
    bg_m = mask_f < 0.3
    if not (subj.any() and bg_m.any()):
        return orig_arr.copy()

    bg_mean = bg_arr[bg_m].reshape(-1, 3).mean(axis=0)
    subj_mean = orig_arr[subj].reshape(-1, 3).mean(axis=0)
    shift = (bg_mean - subj_mean) * strength

    result = orig_arr.copy()
    # Apply shift with gradient: stronger at edges
    edge_w = np.clip(1.0 - (mask_f - 0.1) / 0.7, strength * 0.3, 1.0)[:, :, np.newaxis]
    result = result + shift[np.newaxis, np.newaxis, :] * edge_w
    return np.clip(result, 0, 255)


def complementary_wash(orig_arr, bg_arr, mask_f, strength=0.3):
    """Complementary color wash from opposite sides — like neon gels.

    Samples BG color from left and right halves, creates opposing color washes
    that bleed into the subject from each side.
    """
    h, w = orig_arr.shape[:2]
    bg_m = mask_f < 0.3

    # Sample BG color from left and right halves
    left_half = np.zeros((h, w), dtype=bool)
    left_half[:, :w // 2] = True
    right_half = ~left_half

    left_bg = bg_m & left_half
    right_bg = bg_m & right_half

    if not (left_bg.any() and right_bg.any()):
        return orig_arr.copy()

    left_color = bg_arr[left_bg].reshape(-1, 3).mean(axis=0)
    right_color = bg_arr[right_bg].reshape(-1, 3).mean(axis=0)

    # Create horizontal gradient: left_color on left, right_color on right
    xx = np.linspace(0, 1, w)[np.newaxis, :]  # (1, w)
    xx = np.broadcast_to(xx, (h, w))

    wash = (1.0 - xx)[:, :, np.newaxis] * left_color + xx[:, :, np.newaxis] * right_color

    # Apply wash uniformly at given strength (mask_f controls where — full image if all-ones)
    wash_weight = np.clip(mask_f, 0, 1)[:, :, np.newaxis] * strength
    result = orig_arr * (1 - wash_weight) + wash * wash_weight
    return np.clip(result, 0, 255)


if __name__ == "__main__":
    src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
    img_orig = Image.open(src_path).convert("RGB")
    w, h = img_orig.size
    short_edge = min(w, h)

    # Extract mask (once)
    print("Extracting mask...")
    mask, info = build_mask(img_orig, affect="subject", exclude="", output_dir="/tmp", feather=0)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.LANCZOS)
    mask_binary = (np.array(mask) > 127).astype(np.uint8)
    # 2px feather
    mask_f = np.array(
        Image.fromarray((mask_binary * 255).astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(radius=2))
    ).astype(np.float32) / 255.0
    print(f"  Mask coverage: {info['coverage_pct']}%")

    # Generate aurora BG (once)
    bg_img = generate_bg_once(
        "sweeping northern lights aurora borealis forms, large flowing luminous curtains "
        "of green teal purple pink light against dark starry sky, organic undulating "
        "ribbons of light, atmospheric glow",
        w, h, seed=42)
    if bg_img.size != (w, h):
        bg_img = bg_img.resize((w, h), Image.LANCZOS)

    orig_arr = np.array(img_orig).astype(np.float32)
    bg_arr = np.array(bg_img).astype(np.float32)

    # Save BG for reference
    bg_img.save(os.path.join(FINALS, "color_exp_aurora_bg.jpg"), quality=95)

    experiments = []

    # --- 0. Baseline ---
    comp = basic_composite(orig_arr, bg_arr, mask_f)
    experiments.append(("00_baseline", comp, "No color match"))

    # --- 1-3. LAB on model: 70%, 85%, 100% ---
    for pct in [70, 85, 100]:
        shifted = lab_transfer(orig_arr, bg_arr, mask_f, strength=pct / 100)
        comp = basic_composite(shifted, bg_arr, mask_f)
        experiments.append((f"01_lab_{pct}", comp, f"LAB {pct}% (model)"))

    # --- 4-6. RGB on model only, strong: 50%, 70%, 90% ---
    for pct in [50, 70, 90]:
        shifted = rgb_color_shift(orig_arr, bg_arr, mask_f, strength=pct / 100)
        comp = basic_composite(shifted, bg_arr, mask_f)
        experiments.append((f"02_rgb_{pct}", comp, f"RGB {pct}% (model only)"))

    # --- 7-9. Complementary wash on ENTIRE composite: 15%, 30%, 50% ---
    base_comp = basic_composite(orig_arr, bg_arr, mask_f).astype(np.float32)
    for pct in [15, 30, 50]:
        # Apply wash to full composite (not just subject)
        dummy_mask = np.ones((h, w), dtype=np.float32)  # all pixels
        washed = complementary_wash(base_comp, bg_arr, dummy_mask, strength=pct / 100)
        experiments.append((f"03_wash_full_{pct}", washed.astype(np.uint8), f"Wash {pct}% (full image)"))

    # --- 10. LAB 70% + wash full 25% ---
    shifted = lab_transfer(orig_arr, bg_arr, mask_f, strength=0.70)
    comp = basic_composite(shifted, bg_arr, mask_f).astype(np.float32)
    dummy_mask = np.ones((h, w), dtype=np.float32)
    washed = complementary_wash(comp, bg_arr, dummy_mask, strength=0.25)
    experiments.append(("04_lab70_wash25", washed.astype(np.uint8), "LAB 70% + wash 25% full"))

    # --- 11. LAB 85% + RGB 50% + wash full 20% ---
    shifted = lab_transfer(orig_arr, bg_arr, mask_f, strength=0.85)
    shifted = rgb_color_shift(shifted, bg_arr, mask_f, strength=0.50)
    comp = basic_composite(shifted, bg_arr, mask_f).astype(np.float32)
    dummy_mask = np.ones((h, w), dtype=np.float32)
    washed = complementary_wash(comp, bg_arr, dummy_mask, strength=0.20)
    experiments.append(("04_lab85_rgb50_wash20", washed.astype(np.uint8), "LAB85+RGB50+wash20"))

    # --- 12. RGB 70% + wash full 30% ---
    shifted = rgb_color_shift(orig_arr, bg_arr, mask_f, strength=0.70)
    comp = basic_composite(shifted, bg_arr, mask_f).astype(np.float32)
    dummy_mask = np.ones((h, w), dtype=np.float32)
    washed = complementary_wash(comp, bg_arr, dummy_mask, strength=0.30)
    experiments.append(("04_rgb70_wash30", washed.astype(np.uint8), "RGB 70% + wash 30% full"))

    # Save and push all
    print(f"\n=== Saving {len(experiments)} experiments ===")
    for name, comp_arr, desc in experiments:
        out = os.path.join(FINALS, f"color_exp_{name}.jpg")
        Image.fromarray(comp_arr).save(out, quality=95)
        push_image(out, title=desc, body="Nastia + aurora BG")
        print(f"  {desc} — pushed")

    print("\n=== ALL DONE ===")
