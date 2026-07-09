#!/home/rong/openclaw-venv/bin/python3
"""
Gradient-mask inpainting: binary subject protection + distance-based
gradient in BG. Close to subject = gentle regen, far = full regen.
Like PS generative fill with a radial gradient mask.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client, requests, tempfile
from PIL import Image, ImageFilter
from masking import build_mask
from scipy.ndimage import distance_transform_edt
from io import BytesIO
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")

img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

# --- Mask ---
print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)
print(f"  Coverage: {info['coverage_pct']}%")

# --- Build gradient inpainting mask ---
# Subject = 0 (protected). BG = gradient based on distance from subject edge.
# Close to subject edge: low value (gentle regen). Far: 255 (full regen).

# Distance transform: each BG pixel gets its distance from nearest subject pixel
bg_mask = (mask_binary == 0).astype(np.float64)  # 1 where BG
dist = distance_transform_edt(bg_mask)  # distance from subject edge in BG

# Normalize: 0 at subject edge, 1 at max_distance
# Use a "fade distance" — how many pixels until full regeneration
fade_px = int(short_edge * 0.25)  # 25% of short edge
gradient = np.clip(dist / fade_px, 0, 1)

# Map to mask values: 0 at subject, ramps to 255 over fade_px
# Start at ~80 right at the edge (not zero — we want SOME regen at the boundary)
inpaint_values = np.where(mask_binary > 0, 0, (80 + gradient * 175)).astype(np.uint8)

# Save gradient mask for inspection
gradient_mask = Image.fromarray(inpaint_values, "L")
gradient_mask.save(os.path.join(FINALS, "gradient_inpaint_mask.png"))
print(f"  Gradient mask: fade={fade_px}px, edge=80, far=255")

# Also save a version with more aggressive edge (starts at 180)
inpaint_aggressive = np.where(mask_binary > 0, 0, (180 + gradient * 75)).astype(np.uint8)
aggressive_mask = Image.fromarray(inpaint_aggressive, "L")
aggressive_mask.save(os.path.join(FINALS, "gradient_inpaint_mask_aggressive.png"))

# --- Upload ---
print("Uploading...")
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    img.save(tmp, format="JPEG", quality=95)
    img_path = tmp.name
img_url = fal_client.upload_file(img_path)
os.unlink(img_path)

prompt = (
    "surreal volumetric clouds and soft ethereal haze surrounding the figure, "
    "warm golden cream and soft blue-grey tones, dreamy atmospheric smoke, "
    "baroque oil painting atmosphere, soft directional light from right, "
    "organic flowing cloud forms, no sharp edges"
)
negative = "text, watermark, cartoon, flat, modern, digital, black background, solid black, dark void"

# --- ALSO prepare a version with BG painted solid cream ---
print("Painting BG solid cream...")
cream = np.array([210, 190, 165], dtype=np.float32)  # warm cream
orig_arr = np.array(img).astype(np.float32)
m3 = mask_binary[:, :, np.newaxis].astype(np.float32)
cream_bg = orig_arr * m3 + cream[np.newaxis, np.newaxis, :] * (1 - m3)
cream_img = Image.fromarray(np.clip(cream_bg, 0, 255).astype(np.uint8))
cream_img.save(os.path.join(FINALS, "gradient_inpaint_cream_input.jpg"), quality=95)

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    cream_img.save(tmp, format="JPEG", quality=95)
    cream_path = tmp.name
cream_url = fal_client.upload_file(cream_path)
os.unlink(cream_path)

# --- Try both masks × both inputs (original dark + cream-painted) ---
inputs = [
    ("dark", img_url),
    ("cream", cream_url),
]
masks = [("gradient", gradient_mask)]

for input_name, input_url in inputs:
    for mask_name, mask_pil in masks:
        tag = f"{input_name}_{mask_name}"
        print(f"\nInpainting: {tag}...")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            mask_pil.save(tmp, format="PNG")
            mask_path = tmp.name
        mask_url = fal_client.upload_file(mask_path)
        os.unlink(mask_path)

        handle = fal_client.submit("fal-ai/flux-general/inpainting", arguments={
            "image_url": input_url,
            "mask_url": mask_url,
            "prompt": prompt,
            "negative_prompt": negative,
            "strength": 0.95,
            "num_images": 1,
            "output_format": "jpeg",
            "enable_safety_checker": False,
        })
        result = handle.get()
        images = result.get("images", [])
        if images:
            url = images[0].get("url", "")
            resp = requests.get(url, timeout=60)
            out_img = Image.open(BytesIO(resp.content)).convert("RGB")
            if out_img.size != (w, h):
                out_img = out_img.resize((w, h), Image.LANCZOS)
            out = os.path.join(FINALS, f"gradient_inpaint_{tag}.jpg")
            out_img.save(out, quality=95)
            push_image(out, f"Grad inpaint: {tag}", "Clouds prompt, gradient BG mask")
            print(f"  Saved: {out}")
        else:
            print(f"  No images returned for {tag}")

print("\n=== DONE ===")
