#!/home/rong/openclaw-venv/bin/python3
"""Color wash variants on blend_5_combined — shift entire image toward BG palette."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")

comp = Image.open(os.path.join(FINALS, "blend_5_combined.jpg")).convert("RGB")
bg = Image.open(os.path.join(FINALS, "seam_blend_flux_bg.jpg")).convert("RGB")
if bg.size != comp.size:
    bg = bg.resize(comp.size, Image.LANCZOS)

comp_f = np.array(comp).astype(np.float32)
bg_f = np.array(bg).astype(np.float32)
w, h = comp.size

# BG mean color for RGB wash
bg_mean = bg_f.mean(axis=(0, 1))
print(f"BG mean color: R={bg_mean[0]:.0f} G={bg_mean[1]:.0f} B={bg_mean[2]:.0f}")

# --- 1. RGB wash at various strengths ---
for pct in [15, 25, 35]:
    s = pct / 100.0
    washed = comp_f * (1 - s) + bg_mean[np.newaxis, np.newaxis, :] * s
    out = os.path.join(FINALS, f"wash_rgb_{pct}.jpg")
    Image.fromarray(np.clip(washed, 0, 255).astype(np.uint8)).save(out, quality=95)
    push_image(out, f"RGB wash {pct}%", f"Shift toward BG mean")
    print(f"  RGB {pct}%: {out}")

# --- 2. LAB transfer on FULL image (not just edge) ---
comp_lab = cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2LAB).astype(np.float32)
bg_lab = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2LAB).astype(np.float32)

for strength in [0.3, 0.5, 0.7]:
    shifted = comp_lab.copy()
    for ch in range(3):
        c_mean = comp_lab[:, :, ch].mean()
        c_std = comp_lab[:, :, ch].std() + 1e-8
        b_mean = bg_lab[:, :, ch].mean()
        b_std = bg_lab[:, :, ch].std() + 1e-8
        new_mean = c_mean + (b_mean - c_mean) * strength
        new_std = c_std + (b_std - c_std) * strength * 0.3
        shifted[:, :, ch] = (shifted[:, :, ch] - c_mean) * (new_std / c_std) + new_mean
    result = cv2.cvtColor(np.clip(shifted, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    tag = int(strength * 100)
    out = os.path.join(FINALS, f"wash_lab_{tag}.jpg")
    Image.fromarray(result).save(out, quality=95)
    push_image(out, f"LAB wash {tag}%", f"Full-image LAB transfer")
    print(f"  LAB {tag}%: {out}")

# --- 3. Combined: LAB 50% + RGB 20% ---
shifted_50 = comp_lab.copy()
for ch in range(3):
    c_mean = comp_lab[:, :, ch].mean()
    c_std = comp_lab[:, :, ch].std() + 1e-8
    b_mean = bg_lab[:, :, ch].mean()
    b_std = bg_lab[:, :, ch].std() + 1e-8
    shifted_50[:, :, ch] = (shifted_50[:, :, ch] - c_mean) * ((c_std + (b_std - c_std) * 0.15) / c_std) + c_mean + (b_mean - c_mean) * 0.5
combo = cv2.cvtColor(np.clip(shifted_50, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
combo = combo * 0.80 + bg_mean[np.newaxis, np.newaxis, :] * 0.20
out = os.path.join(FINALS, "wash_combo_lab50_rgb20.jpg")
Image.fromarray(np.clip(combo, 0, 255).astype(np.uint8)).save(out, quality=95)
push_image(out, "LAB50+RGB20", "Combined wash")
print(f"  Combo: {out}")

print("\n=== DONE ===")
