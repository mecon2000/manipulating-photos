#!/home/rong/openclaw-venv/bin/python3
"""Test Tensor Art IMAGE_TO_INPAINT stage (proper inpainting, not img2img with mask field)."""
import os, sys, uuid, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import requests
from PIL import Image, ImageFilter, ImageStat
from masking import build_mask
from scipy.ndimage import distance_transform_edt
from notify import push_image
from io import BytesIO

TENSOR_BASE = "https://ap-east-1.tensorart.cloud/v1"
MODEL_ID = "965126062386242266"  # Z-Image-Uncensored
FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")


def tensor_headers():
    return {"Authorization": f"Bearer {os.environ['TENSOR_API_KEY']}", "Content-Type": "application/json"}


def upload_to_tensor(image_pil):
    """Upload PIL image to Tensor Art, return (resource_id, w, h)."""
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
    """Submit job and poll until done."""
    print(f"  Payload stages: {[s['type'] for s in payload['stages']]}")
    res = requests.post(f"{TENSOR_BASE}/jobs", headers=tensor_headers(), json=payload, timeout=30)
    if res.status_code != 200:
        print(f"  Job creation failed ({res.status_code}): {res.text[:500]}")
        return None
    job_data = res.json()
    job_id = job_data.get("job", {}).get("id")
    if not job_id:
        print(f"  No job ID: {json.dumps(job_data, indent=2)[:500]}")
        return None
    print(f"  Job {job_id} submitted, polling...")

    for attempt in range(60):
        time.sleep(5)
        try:
            r = requests.get(f"{TENSOR_BASE}/jobs/{job_id}", headers=tensor_headers(), timeout=15).json()
        except Exception as e:
            print(f"  Poll error: {e}")
            continue
        status = r.get("job", {}).get("status")
        if status == "SUCCESS":
            images = r["job"].get("successInfo", {}).get("images", [])
            if images:
                return images[0]["url"]
            print("  Job succeeded but no images")
            return None
        elif status == "FAILED":
            print(f"  Job FAILED: {json.dumps(r.get('job', {}), indent=2)[:800]}")
            return None
        if attempt % 6 == 5:
            print(f"  Still waiting... (attempt {attempt})")

    print("  Job timed out")
    return None


# --- Load and mask ---
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)

# Binary inpaint mask: white = inpaint area (BG), black = keep (subject)
inpaint_mask = Image.fromarray(((1 - mask_binary) * 255).astype(np.uint8), "L").convert("RGB")

# Gradient inpaint mask: subject=black, BG=distance-based gradient
bg_area = (mask_binary == 0).astype(np.float64)
dist = distance_transform_edt(bg_area)
fade_px = int(short_edge * 0.25)
gradient = np.clip(dist / fade_px, 0, 1)
inpaint_values = np.where(mask_binary > 0, 0, (80 + gradient * 175)).astype(np.uint8)
gradient_mask = Image.fromarray(inpaint_values, "L").convert("RGB")

# --- Upload all three ---
print("Uploading image...")
img_rid, uw, uh = upload_to_tensor(img)
print(f"  Image: {img_rid} ({uw}x{uh})")

# Resize masks to match
binary_resized = inpaint_mask.resize((uw, uh), Image.LANCZOS)
gradient_resized = gradient_mask.resize((uw, uh), Image.LANCZOS)

print("Uploading binary mask...")
bin_rid, _, _ = upload_to_tensor(binary_resized)
print(f"  Binary mask: {bin_rid}")

print("Uploading gradient mask...")
grad_rid, _, _ = upload_to_tensor(gradient_resized)
print(f"  Gradient mask: {grad_rid}")

prompt = ("surreal volumetric clouds and soft ethereal haze, warm golden cream and "
          "soft blue-grey tones, dreamy atmospheric smoke, huge dark crow wings "
          "spreading outward, baroque oil painting atmosphere, soft directional light")
negative = "text, watermark, cartoon, flat, modern, digital, solid black"

# === Test 1: IMAGE_TO_INPAINT stage (proper inpainting) ===
# Try multiple payload variants since the API docs are unclear

payloads = []

# Variant A: IMAGE_TO_INPAINT with nested diffusion
payloads.append(("inpaint_A_binary", {
    "requestId": str(uuid.uuid4()),
    "stages": [
        {
            "type": "INPUT_INITIALIZE",
            "inputInitialize": {
                "image_resource_id": img_rid,
                "count": 1,
                "seed": 42,
            },
        },
        {
            "type": "IMAGE_TO_INPAINT",
            "imageToInpaint": {
                "maskImageResourceId": bin_rid,
                "maskBlur": 4,
                "resizeMode": "JUST_RESIZE",
                "inpaintingFill": "ORIGINAL",
                "inpaintFullRes": True,
                "inpaintFullResPadding": 32,
                "diffusion": {
                    "width": uw,
                    "height": uh,
                    "prompts": [{"text": prompt, "weight": 1.0}],
                    "negativePrompts": [{"text": negative, "weight": 1.0}],
                    "sdModel": MODEL_ID,
                    "steps": 30,
                    "cfgScale": 7,
                    "denoisingStrength": 0.75,
                    "sampler": "Euler a",
                },
            },
        },
    ],
}))

# Variant B: IMAGE_TO_INPAINT with gradient mask
payloads.append(("inpaint_B_gradient", {
    "requestId": str(uuid.uuid4()),
    "stages": [
        {
            "type": "INPUT_INITIALIZE",
            "inputInitialize": {
                "image_resource_id": img_rid,
                "count": 1,
                "seed": 42,
            },
        },
        {
            "type": "IMAGE_TO_INPAINT",
            "imageToInpaint": {
                "maskImageResourceId": grad_rid,
                "maskBlur": 4,
                "resizeMode": "JUST_RESIZE",
                "inpaintingFill": "ORIGINAL",
                "inpaintFullRes": True,
                "inpaintFullResPadding": 32,
                "diffusion": {
                    "width": uw,
                    "height": uh,
                    "prompts": [{"text": prompt, "weight": 1.0}],
                    "negativePrompts": [{"text": negative, "weight": 1.0}],
                    "sdModel": MODEL_ID,
                    "steps": 30,
                    "cfgScale": 7,
                    "denoisingStrength": 0.75,
                    "sampler": "Euler a",
                },
            },
        },
    ],
}))

# Variant C: IMAGE_TO_INPAINT without INPUT_INITIALIZE (image in inpaint stage)
payloads.append(("inpaint_C_no_init", {
    "requestId": str(uuid.uuid4()),
    "stages": [
        {
            "type": "IMAGE_TO_INPAINT",
            "imageToInpaint": {
                "image_resource_id": img_rid,
                "maskImageResourceId": bin_rid,
                "maskBlur": 4,
                "resizeMode": "JUST_RESIZE",
                "inpaintingFill": "ORIGINAL",
                "inpaintFullRes": True,
                "inpaintFullResPadding": 32,
                "diffusion": {
                    "width": uw,
                    "height": uh,
                    "prompts": [{"text": prompt, "weight": 1.0}],
                    "negativePrompts": [{"text": negative, "weight": 1.0}],
                    "sdModel": MODEL_ID,
                    "steps": 30,
                    "cfgScale": 7,
                    "denoisingStrength": 0.75,
                    "sampler": "Euler a",
                    "count": 1,
                    "seed": 42,
                },
            },
        },
    ],
}))

# Variant D: DIFFUSION stage with inpaintMaskResourceId (camelCase variant)
payloads.append(("inpaint_D_diffusion_camel", {
    "requestId": str(uuid.uuid4()),
    "stages": [
        {
            "type": "INPUT_INITIALIZE",
            "inputInitialize": {
                "image_resource_id": img_rid,
                "count": 1,
                "seed": 42,
            },
        },
        {
            "type": "DIFFUSION",
            "diffusion": {
                "width": uw,
                "height": uh,
                "prompts": [{"text": prompt, "weight": 1.0}],
                "negativePrompts": [{"text": negative, "weight": 1.0}],
                "sdModel": MODEL_ID,
                "steps": 30,
                "cfgScale": 7,
                "denoisingStrength": 0.75,
                "sampler": "Euler a",
                "inpaintMaskResourceId": bin_rid,
                "maskBlur": 4,
                "inpaintingFill": 1,
                "inpaintFullRes": 1,
                "inpaintFullResPadding": 32,
            },
        },
    ],
}))

# Run all variants
for tag, payload in payloads:
    print(f"\n{'='*60}")
    print(f"Testing: {tag}")
    print(f"{'='*60}")

    img_url = run_tensor_job(payload)
    if img_url:
        resp = requests.get(img_url, timeout=60)
        out_img = Image.open(BytesIO(resp.content)).convert("RGB")
        stat = ImageStat.Stat(out_img)
        brightness = stat.mean[0]
        out = os.path.join(FINALS, f"tensor_v2_{tag}.jpg")
        out_img.save(out, quality=95)
        push_image(out, f"Tensor v2: {tag}", f"brightness={brightness:.0f}")
        print(f"  Saved: {out} (brightness={brightness:.0f})")
    else:
        print(f"  FAILED: {tag}")

print("\n=== DONE ===")
