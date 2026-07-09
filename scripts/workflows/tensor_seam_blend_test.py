#!/home/rong/openclaw-venv/bin/python3
"""
Two-step baroque: generate BG via Flux, rough-composite, then use Tensor Art
IMAGE_TO_INPAINT to harmonize the seam zone only.
"""
import os, sys, uuid, time, json, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import requests, fal_client
from PIL import Image, ImageFilter, ImageStat
from masking import build_mask
from scipy.ndimage import distance_transform_edt
from notify import push_image
from io import BytesIO

TENSOR_BASE = "https://ap-east-1.tensorart.cloud/v1"
MODEL_ID = "965126062386242266"
FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")


def tensor_headers():
    return {"Authorization": f"Bearer {os.environ['TENSOR_API_KEY']}", "Content-Type": "application/json"}


def upload_to_tensor(image_pil):
    w, h = image_pil.size
    MAX_PX = 2073600
    if w * h > MAX_PX:
        scale = (MAX_PX / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
        image_pil = image_pil.resize((w, h), Image.LANCZOS)
    w, h = (w // 8) * 8, (h // 8) * 8
    image_pil = image_pil.resize((w, h), Image.LANCZOS)
    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    res = requests.post(f"{TENSOR_BASE}/resource/image", json={}, headers=tensor_headers(), timeout=30)
    if res.status_code != 200:
        print(f"  Upload init failed ({res.status_code}): {res.text[:200]}")
        return None, w, h
    data = res.json()
    put_resp = requests.put(data["putUrl"], data=buf.getvalue(), headers=data["headers"], timeout=120)
    if put_resp.status_code not in (200, 201):
        print(f"  Upload PUT failed ({put_resp.status_code})")
        return None, w, h
    return data["resourceId"], w, h


def run_tensor_job(payload):
    res = requests.post(f"{TENSOR_BASE}/jobs", headers=tensor_headers(), json=payload, timeout=30)
    if res.status_code != 200:
        print(f"  Job creation failed ({res.status_code}): {res.text[:500]}")
        return None
    job_id = res.json().get("job", {}).get("id")
    if not job_id:
        return None
    print(f"  Job {job_id} polling...")
    for attempt in range(60):
        time.sleep(5)
        try:
            r = requests.get(f"{TENSOR_BASE}/jobs/{job_id}", headers=tensor_headers(), timeout=15).json()
        except:
            continue
        status = r.get("job", {}).get("status")
        if status == "SUCCESS":
            images = r["job"].get("successInfo", {}).get("images", [])
            return images[0]["url"] if images else None
        elif status == "FAILED":
            print(f"  FAILED: {json.dumps(r.get('job', {}), indent=2)[:500]}")
            return None
        if attempt % 6 == 5:
            print(f"  Still waiting... ({attempt})")
    return None


# --- Step 1: Load source + mask ---
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

print("Step 1: Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)

# --- Step 2: Generate dramatic BG via Flux ---
print("\nStep 2: Generating BG via Flux...")
bg_prompt = (
    "surreal volumetric clouds and soft ethereal haze, warm golden cream and "
    "soft blue-grey tones, dreamy atmospheric smoke, huge dark crow wings "
    "spreading outward, baroque oil painting atmosphere, soft directional light, "
    "dark moody background with swirling smoke and feathers"
)

# Match original image dimensions for Flux
# Flux needs aspect ratio close to original
flux_w = min(w, 1024)
flux_h = int(flux_w * h / w)
flux_h = (flux_h // 8) * 8
flux_w = (flux_w // 8) * 8

handle = fal_client.submit("fal-ai/flux/dev", arguments={
    "prompt": bg_prompt,
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
bg_flux = Image.open(BytesIO(resp.content)).convert("RGB")
if bg_flux.size != (w, h):
    bg_flux = bg_flux.resize((w, h), Image.LANCZOS)

bg_flux.save(os.path.join(FINALS, "seam_blend_flux_bg.jpg"), quality=95)
push_image(os.path.join(FINALS, "seam_blend_flux_bg.jpg"), "Flux BG", "Generated background")
print(f"  BG generated: {bg_flux.size}")

# --- Step 3: Rough composite ---
print("\nStep 3: Rough composite...")
orig_arr = np.array(img).astype(np.float32)
bg_arr = np.array(bg_flux).astype(np.float32)

# Tight 2px feather for sharp composite (intentionally cutout-looking)
feather_r = 2
mask_soft = np.array(
    Image.fromarray((mask_binary * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=feather_r))
).astype(np.float32) / 255.0
m3 = mask_soft[:, :, np.newaxis]

composite = np.clip(orig_arr * m3 + bg_arr * (1 - m3), 0, 255).astype(np.uint8)
composite_img = Image.fromarray(composite)
composite_img.save(os.path.join(FINALS, "seam_blend_rough_composite.jpg"), quality=95)
push_image(os.path.join(FINALS, "seam_blend_rough_composite.jpg"), "Rough composite", "Before seam blend")
print(f"  Rough composite saved (cutout look expected)")

# --- Step 4: Build transition-zone mask ---
# Subject interior = 0 (keep), far BG = 0 (keep), transition zone = 255 (inpaint)
print("\nStep 4: Building seam masks...")

# Distance from subject edge INTO subject
subj_dist = distance_transform_edt(mask_binary)
# Distance from subject edge INTO background
bg_dist = distance_transform_edt(1 - mask_binary)

# Several seam widths to test
for seam_name, inner_px, outer_px in [
    ("narrow", int(short_edge * 0.02), int(short_edge * 0.04)),   # ~25px in, ~50px out
    ("medium", int(short_edge * 0.04), int(short_edge * 0.08)),   # ~50px in, ~100px out
    ("wide",   int(short_edge * 0.06), int(short_edge * 0.15)),   # ~75px in, ~190px out
]:
    # Seam mask: 255 in the transition zone, 0 elsewhere
    # Inner side (into subject): gentle falloff
    inner_weight = np.clip(1.0 - subj_dist / max(inner_px, 1), 0, 1)
    # Outer side (into BG): gentle falloff
    outer_weight = np.clip(1.0 - bg_dist / max(outer_px, 1), 0, 1)
    # Combine: only the overlap zone near the edge
    seam = np.where(mask_binary > 0, inner_weight, outer_weight)
    seam_mask = (seam * 255).astype(np.uint8)
    seam_mask_img = Image.fromarray(seam_mask, "L")
    seam_mask_img.save(os.path.join(FINALS, f"seam_blend_mask_{seam_name}.png"))
    print(f"  {seam_name}: inner={inner_px}px, outer={outer_px}px, "
          f"coverage={100*np.mean(seam_mask > 10):.1f}%")

# --- Step 5: Upload composite + seam masks to Tensor Art ---
print("\nStep 5: Uploading to Tensor Art...")
comp_rid, uw, uh = upload_to_tensor(composite_img)
print(f"  Composite: {comp_rid} ({uw}x{uh})")

seam_rids = {}
for seam_name in ["narrow", "medium", "wide"]:
    smask = Image.open(os.path.join(FINALS, f"seam_blend_mask_{seam_name}.png")).convert("L")
    smask_resized = smask.resize((uw, uh), Image.LANCZOS).convert("RGB")
    rid, _, _ = upload_to_tensor(smask_resized)
    seam_rids[seam_name] = rid
    print(f"  {seam_name} mask: {rid}")

# --- Step 6: Inpaint the seam zone ---
prompt_blend = (
    "surreal volumetric clouds and ethereal haze engulfing a woman, "
    "warm golden tones, dreamy atmospheric smoke, dark crow wings and feathers, "
    "baroque oil painting atmosphere, soft directional light, "
    "seamless blend between photo and painted elements"
)
negative = "text, watermark, cartoon, flat, sharp edge, cutout, pasted, collage"

for seam_name, seam_rid in seam_rids.items():
    for strength in [0.65, 0.85]:
        tag = f"{seam_name}_s{int(strength*100)}"
        print(f"\n{'='*50}")
        print(f"Seam blend: {tag}")
        print(f"{'='*50}")

        payload = {
            "requestId": str(uuid.uuid4()),
            "stages": [
                {
                    "type": "INPUT_INITIALIZE",
                    "inputInitialize": {
                        "image_resource_id": comp_rid,
                        "count": 1,
                        "seed": 42,
                    },
                },
                {
                    "type": "IMAGE_TO_INPAINT",
                    "imageToInpaint": {
                        "maskImageResourceId": seam_rid,
                        "maskBlur": 4,
                        "resizeMode": "JUST_RESIZE",
                        "inpaintingFill": "ORIGINAL",
                        "inpaintFullRes": False,
                        "inpaintFullResPadding": 32,
                        "diffusion": {
                            "width": uw,
                            "height": uh,
                            "prompts": [{"text": prompt_blend, "weight": 1.0}],
                            "negativePrompts": [{"text": negative, "weight": 1.0}],
                            "sdModel": MODEL_ID,
                            "steps": 30,
                            "cfgScale": 7,
                            "denoisingStrength": strength,
                            "sampler": "Euler a",
                        },
                    },
                },
            ],
        }

        url = run_tensor_job(payload)
        if url:
            resp = requests.get(url, timeout=60)
            out_img = Image.open(BytesIO(resp.content)).convert("RGB")
            out = os.path.join(FINALS, f"seam_blend_{tag}.jpg")
            out_img.save(out, quality=95)
            push_image(out, f"Seam: {tag}", f"Flux BG + Tensor seam blend")
            print(f"  Saved: {out}")
        else:
            print(f"  FAILED")

print("\n=== DONE ===")
