#!/home/rong/openclaw-venv/bin/python3
"""Composite a relit subject onto a silk BG with LAB color matching."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageFilter
from masking import build_mask
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")

# Load images
relit = Image.open(os.path.join(FINALS, "Nastia Tsoy_BLD_5147_2026-04-16_11-13-07_Custom_40.jpg")).convert("RGB")
bg = Image.open(os.path.join(FINALS, "relight_silk_base.jpg")).convert("RGB")
orig = Image.open(os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")).convert("RGB")
w, h = orig.size

# Mask
mask, _ = build_mask(orig, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_b = (np.array(mask) > 127).astype(np.uint8)
mask_f = np.array(
    Image.fromarray((mask_b * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius=2))
).astype(np.float32) / 255.0

if relit.size != (w, h):
    relit = relit.resize((w, h), Image.LANCZOS)

relit_arr = np.array(relit).astype(np.float32)
bg_arr = np.array(bg).astype(np.float32)

# LAB 100% color transfer: relit subject → silk BG colors
relit_lab = cv2.cvtColor(np.clip(relit_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
bg_lab = cv2.cvtColor(np.clip(bg_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
subj = mask_f > 0.5
bg_m = mask_f < 0.3

for ch in range(3):
    s_mean = relit_lab[:, :, ch][subj].mean()
    s_std = relit_lab[:, :, ch][subj].std() + 1e-8
    b_mean = bg_lab[:, :, ch][bg_m].mean()
    b_std = bg_lab[:, :, ch][bg_m].std() + 1e-8
    new_mean = s_mean + (b_mean - s_mean) * 1.0
    new_std = s_std + (b_std - s_std) * 0.3
    shifted = (relit_lab[:, :, ch] - s_mean) * (new_std / s_std) + new_mean
    edge_w = np.clip(1.0 - (mask_f - 0.2) / 0.6, 0.4, 1.0)
    relit_lab[:, :, ch] = relit_lab[:, :, ch] * (1 - edge_w) + shifted * edge_w

relit_matched = cv2.cvtColor(np.clip(relit_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

# Composite
m3 = mask_f[:, :, np.newaxis]
comp = np.clip(relit_matched * m3 + bg_arr * (1 - m3), 0, 255).astype(np.uint8)

out = os.path.join(FINALS, "relit_then_silk_final.jpg")
Image.fromarray(comp).save(out, quality=95)
push_image(out, "Relit→Silk final", "Relight original, LAB 100%, composite onto silk BG")
print(f"Saved: {out}")
