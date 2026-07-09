#!/home/rong/openclaw-venv/bin/python3
"""
Test advanced blending techniques on baroque composite:
1. Poisson blending (cv2.seamlessClone)
2. Laplacian pyramid blending
3. Light wrap + LAB color transfer
4. Combined: Laplacian + light wrap + unified wash
All local — no API calls for blending.
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageFilter
from masking import build_mask
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")

# --- Load source + generated BG ---
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

# Use existing Flux-generated BG if available, otherwise use the aurora one
bg_path = os.path.join(FINALS, "seam_blend_flux_bg.jpg")
if not os.path.exists(bg_path):
    # Fall back to any existing baroque BG
    for candidate in ["baroque_bg_smoke.jpg", "baroque_bg_ethereal.jpg"]:
        p = os.path.join(FINALS, candidate)
        if os.path.exists(p):
            bg_path = p
            break
bg = Image.open(bg_path).convert("RGB")
if bg.size != (w, h):
    bg = bg.resize((w, h), Image.LANCZOS)

print(f"Source: {w}x{h}")
print(f"BG: {bg_path}")

# --- Mask ---
print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)
print(f"  Coverage: {info['coverage_pct']}%")

# Arrays
src_arr = np.array(img)  # uint8 BGR for cv2
bg_arr = np.array(bg)
src_f = src_arr.astype(np.float32)
bg_f = bg_arr.astype(np.float32)

# Soft masks at various feather levels
def soft_mask(binary, feather_px):
    m = Image.fromarray((binary * 255).astype(np.uint8), "L")
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return np.array(m).astype(np.float32) / 255.0


# ============================================================
# Technique 0: Baseline — simple feathered composite (for comparison)
# ============================================================
print("\n--- Baseline: 15px feather composite ---")
m_base = soft_mask(mask_binary, 15)[:, :, np.newaxis]
baseline = np.clip(src_f * m_base + bg_f * (1 - m_base), 0, 255).astype(np.uint8)
out = os.path.join(FINALS, "blend_0_baseline.jpg")
Image.fromarray(baseline).save(out, quality=95)
push_image(out, "Blend: baseline", "15px feather, no tricks")
print(f"  Saved: {out}")


# ============================================================
# Technique 1: Poisson Blending (cv2.seamlessClone)
# ============================================================
print("\n--- Technique 1: Poisson blending ---")
# cv2 works in BGR
src_bgr = cv2.cvtColor(src_arr, cv2.COLOR_RGB2BGR)
bg_bgr = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2BGR)
mask_255 = (mask_binary * 255).astype(np.uint8)

# Center of the subject (centroid of mask)
ys, xs = np.where(mask_binary > 0)
cx, cy = int(xs.mean()), int(ys.mean())

# Shrink mask by a few px to avoid edge-touching assertion
shrink_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_shrunk = cv2.erode(mask_255, shrink_k, iterations=2)
# Zero out border rows/cols
border = 5
mask_shrunk[:border, :] = 0
mask_shrunk[-border:, :] = 0
mask_shrunk[:, :border] = 0
mask_shrunk[:, -border:] = 0

for mode_name, mode in [("normal", cv2.NORMAL_CLONE), ("mixed", cv2.MIXED_CLONE)]:
    try:
        result = cv2.seamlessClone(src_bgr, bg_bgr, mask_shrunk, (cx, cy), mode)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        out = os.path.join(FINALS, f"blend_1_poisson_{mode_name}.jpg")
        Image.fromarray(result_rgb).save(out, quality=95)
        push_image(out, f"Poisson {mode_name}", "cv2.seamlessClone")
        print(f"  {mode_name}: {out}")
    except cv2.error as e:
        print(f"  {mode_name} FAILED: {e}")


# ============================================================
# Technique 2: Laplacian Pyramid Blending
# ============================================================
print("\n--- Technique 2: Laplacian pyramid blending ---")

def laplacian_pyramid(img_f, levels=6):
    """Build Laplacian pyramid."""
    pyr = []
    current = img_f.copy()
    for i in range(levels - 1):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = current - up
        pyr.append(lap)
        current = down
    pyr.append(current)  # coarsest level (Gaussian)
    return pyr

def gaussian_pyramid(mask_f, levels=6):
    """Build Gaussian pyramid for mask."""
    pyr = [mask_f.copy()]
    current = mask_f.copy()
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyr.append(current)
    return pyr

def reconstruct(pyr):
    """Reconstruct from Laplacian pyramid."""
    current = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        up = cv2.pyrUp(current, dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
        current = up + pyr[i]
    return current

levels = 6
# Build pyramids
src_pyr = laplacian_pyramid(src_f, levels)
bg_pyr = laplacian_pyramid(bg_f, levels)

# Mask pyramid — use progressively blurred mask at each level
mask_3ch = np.stack([mask_binary.astype(np.float32)] * 3, axis=-1)
mask_gpyr = gaussian_pyramid(mask_3ch, levels)

# Blend at each level
blended_pyr = []
for s, b, m in zip(src_pyr, bg_pyr, mask_gpyr):
    blended_pyr.append(s * m + b * (1 - m))

result = np.clip(reconstruct(blended_pyr), 0, 255).astype(np.uint8)
out = os.path.join(FINALS, "blend_2_laplacian.jpg")
Image.fromarray(result).save(out, quality=95)
push_image(out, "Laplacian pyramid", "6-level frequency blend")
print(f"  Saved: {out}")


# ============================================================
# Technique 3: Light Wrap
# ============================================================
print("\n--- Technique 3: Light wrap ---")

# Light wrap: BG bright areas spill onto subject edges
bg_bright = bg_f.copy()
# Heavy blur of BG
blur_r = max(30, int(short_edge * 0.08))
bg_blurred = cv2.GaussianBlur(bg_f, (0, 0), blur_r)

# Edge mask: only the outer edge of the subject (dilated - original)
kernel_size = max(5, int(short_edge * 0.025))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
edge_band = ((mask_dilated - mask_binary) > 0).astype(np.float32)
# Soften the edge band
edge_soft = cv2.GaussianBlur(edge_band, (0, 0), max(3, kernel_size // 2))
edge_3ch = edge_soft[:, :, np.newaxis]

# Start with baseline composite
m_tight = soft_mask(mask_binary, 3)[:, :, np.newaxis]
comp = src_f * m_tight + bg_f * (1 - m_tight)

# Add light wrap: blend blurred BG onto edges
wrap_strength = 0.25
comp_wrapped = comp * (1 - edge_3ch * wrap_strength) + bg_blurred * (edge_3ch * wrap_strength)

result = np.clip(comp_wrapped, 0, 255).astype(np.uint8)
out = os.path.join(FINALS, "blend_3_lightwrap.jpg")
Image.fromarray(result).save(out, quality=95)
push_image(out, "Light wrap", f"BG spill on edges, r={kernel_size}px")
print(f"  Saved: {out}")


# ============================================================
# Technique 4: LAB edge color match
# ============================================================
print("\n--- Technique 4: LAB edge color match ---")

# Start with tight composite
comp_lab = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
bg_lab = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

# Edge band for color matching (wider than light wrap)
edge_width = max(10, int(short_edge * 0.05))
kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
mask_eroded = cv2.erode(mask_binary, kernel_edge, iterations=1)
inner_edge = ((mask_binary - mask_eroded) > 0).astype(np.float32)
inner_soft = cv2.GaussianBlur(inner_edge, (0, 0), max(3, edge_width // 2))

# For each LAB channel, shift edge region toward local BG values
for ch in range(3):
    # Local BG stats near the edge
    bg_near = bg_lab[:, :, ch][edge_band > 0.3]
    if len(bg_near) == 0:
        continue
    bg_mean = bg_near.mean()
    # Subject edge stats
    subj_edge = comp_lab[:, :, ch][inner_soft > 0.3]
    if len(subj_edge) == 0:
        continue
    subj_mean = subj_edge.mean()
    # Shift: move edge pixels toward BG mean
    shift = (bg_mean - subj_mean) * 0.5  # 50% shift
    comp_lab[:, :, ch] += shift * inner_soft

comp_matched = cv2.cvtColor(np.clip(comp_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
result = np.clip(comp_matched, 0, 255).astype(np.uint8)
out = os.path.join(FINALS, "blend_4_lab_edge.jpg")
Image.fromarray(result).save(out, quality=95)
push_image(out, "LAB edge match", f"50% LAB shift on {edge_width}px inner edge")
print(f"  Saved: {out}")


# ============================================================
# Technique 5: COMBINED — Laplacian + light wrap + LAB edge + unified wash
# ============================================================
print("\n--- Technique 5: Combined (Laplacian + wrap + LAB + wash) ---")

# Start with Laplacian pyramid blend
lap_result = np.clip(reconstruct(blended_pyr), 0, 255).astype(np.float32)

# Add light wrap
lap_wrapped = lap_result * (1 - edge_3ch * wrap_strength) + bg_blurred * (edge_3ch * wrap_strength)

# LAB edge color match on the wrapped result
lap_lab = cv2.cvtColor(np.clip(lap_wrapped, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
for ch in range(3):
    bg_near = bg_lab[:, :, ch][edge_band > 0.3]
    if len(bg_near) == 0:
        continue
    subj_edge = lap_lab[:, :, ch][inner_soft > 0.3]
    if len(subj_edge) == 0:
        continue
    shift = (bg_near.mean() - subj_edge.mean()) * 0.4
    lap_lab[:, :, ch] += shift * inner_soft

combined = cv2.cvtColor(np.clip(lap_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

# Unified color wash: sample BG dominant color, apply very subtle wash to EVERYTHING
bg_mean_color = bg_f.mean(axis=(0, 1))
wash_strength = 0.08  # very subtle
combined = combined * (1 - wash_strength) + bg_mean_color[np.newaxis, np.newaxis, :] * wash_strength

result = np.clip(combined, 0, 255).astype(np.uint8)
out = os.path.join(FINALS, "blend_5_combined.jpg")
Image.fromarray(result).save(out, quality=95)
push_image(out, "Combined blend", "Laplacian + wrap + LAB + wash")
print(f"  Saved: {out}")


# ============================================================
# Technique 6: Poisson + light wrap (best of both?)
# ============================================================
print("\n--- Technique 6: Poisson + light wrap ---")
# Start with Poisson mixed clone
poisson_mixed = cv2.seamlessClone(src_bgr, bg_bgr, mask_255, (cx, cy), cv2.MIXED_CLONE)
poisson_rgb = cv2.cvtColor(poisson_mixed, cv2.COLOR_BGR2RGB).astype(np.float32)

# Light wrap on top
poisson_wrapped = poisson_rgb * (1 - edge_3ch * 0.20) + bg_blurred * (edge_3ch * 0.20)

# Subtle unified wash
poisson_wrapped = poisson_wrapped * (1 - 0.06) + bg_mean_color[np.newaxis, np.newaxis, :] * 0.06

result = np.clip(poisson_wrapped, 0, 255).astype(np.uint8)
out = os.path.join(FINALS, "blend_6_poisson_wrap.jpg")
Image.fromarray(result).save(out, quality=95)
push_image(out, "Poisson + wrap", "Mixed clone + light wrap + wash")
print(f"  Saved: {out}")

print("\n=== ALL DONE ===")
