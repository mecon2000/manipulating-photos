#!/home/rong/openclaw-venv/bin/python3
"""
Guide Flux BG generation AWAY from the subject area using a composition map.
Subject silhouette = dark void. Edges = colored blobs suggesting element placement.
"""
import os, sys, numpy as np, cv2, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client, requests
from PIL import Image, ImageFilter, ImageDraw
from masking import build_mask
from notify import push_image
from io import BytesIO

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")

# --- Load source + mask ---
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)
src_f = np.array(img).astype(np.float32)

# --- Build composition guide ---
print("Building composition guide...")

# Expand the mask a bit so elements stay well clear of the subject
expand_px = int(short_edge * 0.06)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px, expand_px))
mask_expanded = cv2.dilate(mask_binary, kernel, iterations=1)
mask_soft = cv2.GaussianBlur(mask_expanded.astype(np.float32), (0, 0),
                              max(5, int(short_edge * 0.03)))

# Base: dark smoky gradient (dark center, slightly lighter edges)
guide = np.zeros((h, w, 3), dtype=np.float32)

# Vertical gradient: darker at center, lighter at top/bottom
yy = np.abs(np.linspace(-1, 1, h))[:, np.newaxis]
xx = np.abs(np.linspace(-1, 1, w))[np.newaxis, :]
# Radial-ish gradient from center
edge_dist = np.sqrt(yy**2 + xx**2)
edge_dist = np.clip(edge_dist, 0, 1.4) / 1.4

# Base smoke color (dark charcoal with warm tint)
base_dark = np.array([25, 20, 18], dtype=np.float32)
base_light = np.array([80, 65, 50], dtype=np.float32)
for ch in range(3):
    guide[:, :, ch] = base_dark[ch] + (base_light[ch] - base_dark[ch]) * edge_dist

# Paint colored blobs at specific positions OUTSIDE the mask
# These suggest where Flux should place story elements
rng = np.random.RandomState(42)

def paint_blob(guide, cx, cy, radius, color, softness=0.7):
    """Paint a soft colored blob onto the guide."""
    yy = np.arange(h)[:, np.newaxis]
    xx = np.arange(w)[np.newaxis, :]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    weight = np.clip(1.0 - dist / radius, 0, 1) ** softness
    # Don't paint on the subject area
    weight = weight * (1.0 - mask_soft)
    w3 = weight[:, :, np.newaxis]
    color_f = np.array(color, dtype=np.float32)
    guide[:] = guide * (1 - w3 * 0.8) + color_f[np.newaxis, np.newaxis, :] * (w3 * 0.8)

# Find mask bounding box to know where subject is
ys, xs = np.where(mask_binary > 0)
subj_top, subj_bot = ys.min(), ys.max()
subj_left, subj_right = xs.min(), xs.max()
subj_cx, subj_cy = int(xs.mean()), int(ys.mean())

# Place blobs at corners and edges, away from subject
blob_r = int(short_edge * 0.25)
blob_positions = [
    (int(w * 0.1), int(h * 0.15)),   # top-left
    (int(w * 0.9), int(h * 0.15)),   # top-right
    (int(w * 0.05), int(h * 0.7)),   # bottom-left
    (int(w * 0.95), int(h * 0.7)),   # bottom-right
    (int(w * 0.5), int(h * 0.05)),   # top-center
    (int(w * 0.5), int(h * 0.95)),   # bottom-center
]

# Warm amber/crimson blobs (for roses theme)
blob_colors = [
    (120, 40, 30),   # dark red
    (150, 80, 30),   # amber
    (100, 30, 25),   # crimson
    (130, 60, 35),   # warm brown
    (160, 90, 40),   # golden
    (90, 35, 30),    # deep red
]

for (bx, by), color in zip(blob_positions, blob_colors):
    paint_blob(guide, bx, by, blob_r, color, softness=0.5)

# Make sure subject area is dark (void)
m3 = mask_soft[:, :, np.newaxis]
guide = guide * (1 - m3 * 0.9) + base_dark[np.newaxis, np.newaxis, :] * (m3 * 0.9)

guide_img = Image.fromarray(np.clip(guide, 0, 255).astype(np.uint8))
guide_img.save(os.path.join(FINALS, "baroque_guide_map.jpg"), quality=95)
push_image(os.path.join(FINALS, "baroque_guide_map.jpg"), "Guide map", "Dark center, colored edge blobs")
print(f"  Guide map saved")

# --- Upload guide and run Flux img2img ---
print("\nUploading guide...")
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    guide_img.save(tmp, format="JPEG", quality=92)
    tmp_path = tmp.name
guide_url = fal_client.upload_file(tmp_path)
os.unlink(tmp_path)

# Flux dimensions
flux_w = min(w, 1024)
flux_h = int(flux_w * h / w)
flux_h = (flux_h // 8) * 8
flux_w = (flux_w // 8) * 8

prompt_roses = (
    "surreal volumetric clouds and ethereal haze, hundreds of dark red rose "
    "petals floating and swirling through the air, wilting baroque roses with "
    "thorny stems emerging from dense smoke, scattered loose petals caught in "
    "wind, warm amber and deep crimson tones, romantic dark oil painting "
    "atmosphere, soft golden light, vanitas still life inspiration, "
    "elements concentrated around the edges, dark empty center"
)

# Test at different img2img strengths
for strength in [0.80, 0.90]:
    tag = f"guided_roses_s{int(strength*100)}"
    print(f"\nGenerating: {tag}...")

    handle = fal_client.submit("fal-ai/flux/dev/image-to-image", arguments={
        "image_url": guide_url,
        "prompt": prompt_roses,
        "strength": strength,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    })
    result = handle.get()
    bg_url = result["images"][0]["url"]
    resp = requests.get(bg_url, timeout=60)
    bg_img = Image.open(BytesIO(resp.content)).convert("RGB")
    if bg_img.size != (w, h):
        bg_img = bg_img.resize((w, h), Image.LANCZOS)

    # Save raw BG
    bg_img.save(os.path.join(FINALS, f"baroque_{tag}_bg.jpg"), quality=95)
    push_image(os.path.join(FINALS, f"baroque_{tag}_bg.jpg"), f"BG {tag}", "Guided generation")

    # Run full pipeline (steps 3-6)
    bg_f = np.array(bg_img).astype(np.float32)

    # Step 3: Laplacian pyramid blend
    def lap_pyr(img_f, levels=6):
        pyr, cur = [], img_f.copy()
        for _ in range(levels - 1):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cur - up)
            cur = down
        pyr.append(cur)
        return pyr

    def gauss_pyr(m, levels=6):
        pyr, cur = [m.copy()], m.copy()
        for _ in range(levels - 1):
            cur = cv2.pyrDown(cur)
            pyr.append(cur)
        return pyr

    def reconstruct(pyr):
        cur = pyr[-1]
        for i in range(len(pyr) - 2, -1, -1):
            cur = cv2.pyrUp(cur, dstsize=(pyr[i].shape[1], pyr[i].shape[0])) + pyr[i]
        return cur

    levels = 6
    s_pyr = lap_pyr(src_f, levels)
    b_pyr = lap_pyr(bg_f, levels)
    m3_pyr = np.stack([mask_binary.astype(np.float32)] * 3, axis=-1)
    m_pyr = gauss_pyr(m3_pyr, levels)
    blended = [s * m + b * (1 - m) for s, b, m in zip(s_pyr, b_pyr, m_pyr)]
    result_f = np.clip(reconstruct(blended), 0, 255).astype(np.float32)

    # Step 4: Light wrap
    blur_r = max(30, int(short_edge * 0.08))
    bg_blur = cv2.GaussianBlur(bg_f, (0, 0), blur_r)
    ks = max(5, int(short_edge * 0.025))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(mask_binary, kern, iterations=1)
    edge = ((dilated - mask_binary) > 0).astype(np.float32)
    edge_s = cv2.GaussianBlur(edge, (0, 0), max(3, ks // 2))[:, :, np.newaxis]
    result_f = result_f * (1 - edge_s * 0.25) + bg_blur * (edge_s * 0.25)

    # Step 5: LAB edge color match
    bg_lab = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    res_lab = cv2.cvtColor(np.clip(result_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ew = max(10, int(short_edge * 0.05))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ew, ew))
    eroded = cv2.erode(mask_binary, ke, iterations=1)
    inner = ((mask_binary - eroded) > 0).astype(np.float32)
    inner_s = cv2.GaussianBlur(inner, (0, 0), max(3, ew // 2))
    for ch in range(3):
        bg_near = bg_lab[:, :, ch][edge > 0.3]
        subj_edge = res_lab[:, :, ch][inner_s > 0.3]
        if len(bg_near) == 0 or len(subj_edge) == 0:
            continue
        res_lab[:, :, ch] += (bg_near.mean() - subj_edge.mean()) * 0.4 * inner_s
    result_f = cv2.cvtColor(np.clip(res_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Step 6: Full-image LAB 60% wash
    comp_lab = cv2.cvtColor(np.clip(result_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in range(3):
        c_mean = comp_lab[:, :, ch].mean()
        c_std = comp_lab[:, :, ch].std() + 1e-8
        b_mean = bg_lab[:, :, ch].mean()
        b_std = bg_lab[:, :, ch].std() + 1e-8
        new_mean = c_mean + (b_mean - c_mean) * 0.6
        new_std = c_std + (b_std - c_std) * 0.18
        comp_lab[:, :, ch] = (comp_lab[:, :, ch] - c_mean) * (new_std / c_std) + new_mean
    final = cv2.cvtColor(np.clip(comp_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    out = os.path.join(FINALS, f"baroque_{tag}.jpg")
    Image.fromarray(final).save(out, quality=95)
    push_image(out, f"Guided: {tag}", "Guide map → Flux img2img → full pipeline")
    print(f"  Final: {out}")

# Also do one with pure text-to-image (no guide) for comparison with same prompt
print(f"\nGenerating: unguided roses (comparison)...")
handle = fal_client.submit("fal-ai/flux/dev", arguments={
    "prompt": prompt_roses,
    "image_size": {"width": flux_w, "height": flux_h},
    "num_images": 1,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "output_format": "jpeg",
    "enable_safety_checker": False,
})
result = handle.get()
bg_url = result["images"][0]["url"]
resp = requests.get(bg_url, timeout=60)
bg_unguided = Image.open(BytesIO(resp.content)).convert("RGB")
if bg_unguided.size != (w, h):
    bg_unguided = bg_unguided.resize((w, h), Image.LANCZOS)
bg_unguided.save(os.path.join(FINALS, "baroque_unguided_roses_bg.jpg"), quality=95)
push_image(os.path.join(FINALS, "baroque_unguided_roses_bg.jpg"), "BG unguided", "No composition guide")

print("\n=== DONE ===")
