#!/home/rong/openclaw-venv/bin/python3
"""Generate silk composite, then relight the whole scene with different prompts."""
import os, sys, numpy as np, cv2
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

# --- Mask ---
print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_b = (np.array(mask) > 127).astype(np.uint8)
mask_f = np.array(
    Image.fromarray((mask_b * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius=2))
).astype(np.float32) / 255.0

# --- Generate silk BG ---
print("Generating silk BG...")
handle = fal_client.submit("fal-ai/flux/dev", arguments={
    "prompt": "large flowing luxurious silk fabric forms billowing in wind, organic draping "
              "shapes in rich burgundy gold ivory, volumetric folds catching dramatic light, "
              "Renaissance drapery study, sensual flowing textile, NO person NO figure NO face",
    "image_size": {"width": w, "height": h},
    "num_inference_steps": 28, "guidance_scale": 3.5, "num_images": 1,
    "output_format": "jpeg", "enable_safety_checker": False, "seed": 77,
})
bg_url = handle.get()["images"][0]["url"]
bg = Image.open(BytesIO(requests.get(bg_url, timeout=60).content)).convert("RGB")
if bg.size != (w, h):
    bg = bg.resize((w, h), Image.LANCZOS)
print(f"  BG: {bg.size}")

# --- LAB 100% ---
orig_arr = np.array(img).astype(np.float32)
bg_arr = np.array(bg).astype(np.float32)
orig_lab = cv2.cvtColor(np.clip(orig_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
bg_lab = cv2.cvtColor(np.clip(bg_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
subj = mask_f > 0.5
bg_m = mask_f < 0.3
for ch in range(3):
    s_mean, s_std = orig_lab[:,:,ch][subj].mean(), orig_lab[:,:,ch][subj].std() + 1e-8
    b_mean, b_std = bg_lab[:,:,ch][bg_m].mean(), bg_lab[:,:,ch][bg_m].std() + 1e-8
    shifted = (orig_lab[:,:,ch] - s_mean) * ((s_std + (b_std - s_std) * 0.3) / s_std) + (s_mean + (b_mean - s_mean))
    edge_w = np.clip(1.0 - (mask_f - 0.2) / 0.6, 0.4, 1.0)
    orig_lab[:,:,ch] = orig_lab[:,:,ch] * (1 - edge_w) + shifted * edge_w
shifted_arr = cv2.cvtColor(np.clip(orig_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

# --- Composite ---
m3 = mask_f[:, :, np.newaxis]
comp = np.clip(shifted_arr * m3 + bg_arr * (1 - m3), 0, 255).astype(np.uint8)
comp_path = os.path.join(FINALS, "relight_silk_base.jpg")
Image.fromarray(comp).save(comp_path, quality=95)
push_image(comp_path, "Silk composite (no relight)", "LAB 100%, base for relighting")
print(f"Saved base: {comp_path}")

# --- Relight with different prompts ---
relight_prompts = [
    ("side_nowindow",
     "dramatic directional light from the right side, strong rim light on right edge, "
     "deep shadows on left, warm golden light, no visible light source, no window, "
     "no lamp, no furniture, just pure directional light illuminating the scene",
     "window, lamp, furniture, room, interior, halo, corona, glow, lens flare"),

    ("warm_wrap",
     "warm enveloping light wrapping around the subject from behind and right, "
     "volumetric warm amber light rays, atmospheric haze, no visible source, "
     "light seems to emanate from the fabric itself, intimate warm glow",
     "window, lamp, furniture, cold, blue, halo, corona, lens flare"),

    ("chiaroscuro",
     "extreme Caravaggio chiaroscuro lighting, single hard directional light from upper left, "
     "pitch black shadows, dramatic contrast, Renaissance painting lighting, no visible source, "
     "theatrical spot illumination from far outside the upper left edge of frame",
     "window, lamp, even lighting, flat, halo, corona, glow, lens flare, fill light"),
]

for tag, prompt, negative in relight_prompts:
    print(f"\nRelighting: {tag}...")
    # Upload composite
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.fromarray(comp).save(tmp, format="JPEG", quality=95)
        tmp_path = tmp.name
    comp_url = fal_client.upload_file(tmp_path)
    os.unlink(tmp_path)

    # Run IC-Light on the FULL composite via direct API call
    try:
        import base64
        comp_pil = Image.fromarray(comp)
        buf = BytesIO()
        comp_pil.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        headers = {
            "Authorization": f"Key {os.environ.get('FAL_KEY', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image_url": f"data:image/jpeg;base64,{img_b64}",
            "negative_prompt": negative,
            "lowres_denoise": 0.85,
            "highres_denoise": 0.40,
            "guidance_scale": 2.5,
            "num_inference_steps": 28,
            "enable_hr_fix": True,
            "output_format": "jpeg",
            "num_images": 1,
        }
        resp = requests.post("https://fal.run/fal-ai/iclight-v2",
                            headers=headers, json=payload, timeout=600)
        if resp.status_code != 200:
            print(f"  IC-Light error {resp.status_code}: {resp.text[:200]}")
            continue
        result = resp.json()
        images = result.get("images", [])
        if images:
            url = images[0].get("url", "")
            r = requests.get(url, timeout=60)
            relit = Image.open(BytesIO(r.content)).convert("RGB")
            if relit.size != (w, h):
                relit = relit.resize((w, h), Image.LANCZOS)

            # Blend: 60% relit + 40% original composite (keep some of the original BG)
            relit_arr = np.array(relit).astype(np.float32)
            comp_f = comp.astype(np.float32)
            blended = np.clip(relit_arr * 0.6 + comp_f * 0.4, 0, 255).astype(np.uint8)

            out = os.path.join(FINALS, f"relight_silk_{tag}.jpg")
            Image.fromarray(blended).save(out, quality=95)
            push_image(out, f"Silk relit: {tag}", f"60/40 blend, denoise=0.40")
            print(f"  Saved: {out}")

            # Also save pure relit (no blend)
            out2 = os.path.join(FINALS, f"relight_silk_{tag}_pure.jpg")
            relit.save(out2, quality=95)
            push_image(out2, f"Silk relit PURE: {tag}", "100% IC-Light, denoise=0.40")
            print(f"  Saved pure: {out2}")
        else:
            print(f"  No images returned for {tag}")
    except Exception as e:
        print(f"  Failed: {e}")

print("\n=== ALL DONE ===")
