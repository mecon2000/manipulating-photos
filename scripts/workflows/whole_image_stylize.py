#!/home/rong/openclaw-venv/bin/python3
"""
Whole-image img2img approach: blur BG, stylize EVERYTHING in one pass,
then selectively blend original realism back with a body-region gradient.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client, requests
from PIL import Image, ImageFilter
from masking import build_mask
from scipy import ndimage
from io import BytesIO
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")

img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

# --- Step 1: Mask (for knowing where subject is, not for cutting) ---
print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_arr = np.array(mask).astype(np.float32) / 255.0
mask_binary = (mask_arr > 0.5).astype(np.uint8)
print(f"  Coverage: {info['coverage_pct']}%")

# --- Step 2: Pre-process — blur BG, soften subject slightly ---
print("Pre-processing...")
orig_arr = np.array(img).astype(np.float32)

# Heavy BG blur to destroy structure
bg_blur_r = max(50, int(short_edge * 0.18))
full_blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=bg_blur_r))).astype(np.float32)

# Light subject blur (just soften edges, not destroy detail)
subj_blur_r = max(3, int(short_edge * 0.015))
light_blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=subj_blur_r))).astype(np.float32)

# Soft mask for blending (wide feather so transition is gradual)
feather = max(10, int(short_edge * 0.06))
mask_soft = np.array(
    Image.fromarray((mask_binary * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=feather))
).astype(np.float32) / 255.0
m3 = mask_soft[:, :, np.newaxis]

# Lift dark BG to warm mid-tones so Flux has content to paint on
bg_mean = full_blur[mask_soft < 0.3].mean() if (mask_soft < 0.3).any() else 128
if bg_mean < 80:
    lift_target = 110.0
    lift = np.clip(full_blur * (lift_target / max(bg_mean, 1)), 0, 255)
    # Warm shift
    lift[:, :, 0] = np.clip(lift[:, :, 0] * 1.10, 0, 255)  # red boost
    lift[:, :, 2] = np.clip(lift[:, :, 2] * 0.85, 0, 255)  # blue reduce
    full_blur = lift
    print(f"  Dark BG lifted: {bg_mean:.0f} → ~{lift_target:.0f} (warm)")

# Add gradient noise to BG: 100% at bottom → 0% at top
# This encourages Flux to be more aggressive/creative in the lower BG
rng = np.random.RandomState(42)
noise_base = rng.randint(0, 120, (h, w, 3)).astype(np.float32)
# Vertical gradient: 0 at top, 1 at bottom
yy_noise = np.linspace(0, 1, h)[:, np.newaxis, np.newaxis]
yy_noise = np.broadcast_to(yy_noise, (h, w, 3)).copy()
gradient_noise = noise_base * yy_noise  # strong at bottom, zero at top
# Only on BG (invert mask)
bg_weight = (1.0 - m3)
full_blur = np.clip(full_blur + gradient_noise * bg_weight - 40 * bg_weight * yy_noise, 0, 255)
print(f"  Gradient noise added to BG (bottom=strong, top=none)")

# Subject = lightly blurred, BG = heavily blurred + lifted + noisy
prepped = light_blur * m3 + full_blur * (1 - m3)
prepped = np.clip(prepped, 0, 255).astype(np.uint8)
prepped_img = Image.fromarray(prepped)
prepped_img.save(os.path.join(FINALS, "whole_prepped.jpg"), quality=95)
print(f"  BG blur={bg_blur_r}px, subject blur={subj_blur_r}px")

# --- Step 3: img2img the WHOLE thing at different strengths ---
print("Uploading prepped image...")
import tempfile
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    prepped_img.save(tmp, format="JPEG", quality=92)
    tmp_path = tmp.name
img_url = fal_client.upload_file(tmp_path)
os.unlink(tmp_path)

prompt = (
    "dramatic dark oil painting, woman surrounded by swirling volumetric smoke "
    "and haze, huge dark crow wings emerging from behind her spreading outward, "
    "dark feathers dissolving into smoke at the tips, moody chiaroscuro lighting, "
    "soft warm light from the right, deep shadows, the smoke and wings engulf "
    "and wrap around her body, baroque atmosphere, Caravaggio inspired, "
    "rich amber gold and deep charcoal tones, painterly brushwork"
)
negative = (
    "text, watermark, cartoon, anime, bright, flat lighting, sharp edges, "
    "modern, digital, clean, multiple people"
)

strengths = [0.75, 0.85, 0.93]

for strength in strengths:
    tag = f"{int(strength * 100)}"
    print(f"\nimg2img strength={strength}...")

    handle = fal_client.submit("fal-ai/flux/dev/image-to-image", arguments={
        "image_url": img_url,
        "prompt": prompt,
        "strength": strength,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    })
    result = handle.get()
    styled_url = result["images"][0]["url"]
    resp = requests.get(styled_url, timeout=60)
    styled = Image.open(BytesIO(resp.content)).convert("RGB")
    if styled.size != (w, h):
        styled = styled.resize((w, h), Image.LANCZOS)

    styled_arr = np.array(styled).astype(np.float32)

    # --- Step 4: Save the raw styled result (no blending) ---
    raw_path = os.path.join(FINALS, f"whole_raw_{tag}.jpg")
    styled.save(raw_path, quality=95)
    push_image(raw_path, f"Whole raw s={tag}%", f"No blend-back, pure img2img at {tag}%")
    print(f"  Raw saved: {raw_path}")

    # --- Step 5: Selective blend-back with vertical gradient ---
    # How much of the original to bring back at each body region:
    # Face (top 25%): configurable — 0% for anon, 70% for identity
    # Upper body (25-50%): 50% original
    # Lower body (50-75%): 20% original
    # Feet/BG (75-100%): 0% original

    yy = np.linspace(0, 1, h)[:, np.newaxis]  # (h, 1)
    yy_full = np.broadcast_to(yy, (h, w))

    for face_mode, face_keep in [("anon", 0.0), ("face", 0.65)]:
        # Build vertical gradient for how much original to keep
        # Smooth piecewise: face_keep at top, ramps down
        if face_keep > 0:
            # Face preserved: high at top, drops through body
            keep_gradient = np.where(
                yy_full < 0.25, face_keep,
                np.where(yy_full < 0.45, face_keep * (1 - (yy_full - 0.25) / 0.20) + 0.40 * ((yy_full - 0.25) / 0.20),
                np.where(yy_full < 0.65, 0.40 * (1 - (yy_full - 0.45) / 0.20) + 0.10 * ((yy_full - 0.45) / 0.20),
                np.where(yy_full < 0.85, 0.10 * (1 - (yy_full - 0.65) / 0.20),
                0.0))))
        else:
            # Anonymous: no face preservation, mild body preservation
            keep_gradient = np.where(
                yy_full < 0.30, 0.0,
                np.where(yy_full < 0.50, 0.30 * ((yy_full - 0.30) / 0.20),
                np.where(yy_full < 0.65, 0.30 * (1 - (yy_full - 0.50) / 0.15) + 0.10 * ((yy_full - 0.50) / 0.15),
                np.where(yy_full < 0.85, 0.10 * (1 - (yy_full - 0.65) / 0.20),
                0.0))))

        # Only blend back within the subject area (BG stays fully styled)
        keep_weight = keep_gradient * mask_soft
        kw3 = keep_weight[:, :, np.newaxis]

        blended = orig_arr * kw3 + styled_arr * (1 - kw3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out = os.path.join(FINALS, f"whole_{tag}_{face_mode}.jpg")
        Image.fromarray(blended).save(out, quality=95)
        desc = f"s={tag}% {'face kept' if face_keep > 0 else 'anon'}"
        push_image(out, f"Whole {desc}", f"Smoke+wings, vertical blend")
        print(f"  {desc}: {out}")

print("\n=== ALL DONE ===")
