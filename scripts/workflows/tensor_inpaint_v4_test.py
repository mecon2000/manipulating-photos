#!/home/rong/openclaw-venv/bin/python3
"""Tensor Art IMAGE_TO_INPAINT — pre-paint BG warm, test fill modes and CFG."""
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
        print(f"  No job ID")
        return None
    print(f"  Job {job_id} polling...")
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
            return images[0]["url"] if images else None
        elif status == "FAILED":
            print(f"  FAILED: {json.dumps(r.get('job', {}), indent=2)[:500]}")
            return None
        if attempt % 6 == 5:
            print(f"  Still waiting... ({attempt})")
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

# --- Pre-paint BG warm cream ---
print("Pre-painting BG warm...")
orig_arr = np.array(img).astype(np.float32)
cream = np.array([195, 170, 150], dtype=np.float32)

# Soft mask for blending
feather_r = max(5, int(short_edge * 0.02))
mask_soft = np.array(
    Image.fromarray((mask_binary * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=feather_r))
).astype(np.float32) / 255.0
m3 = mask_soft[:, :, np.newaxis]

warm_bg = orig_arr * m3 + cream[np.newaxis, np.newaxis, :] * (1 - m3)
warm_img = Image.fromarray(np.clip(warm_bg, 0, 255).astype(np.uint8))
warm_img.save(os.path.join(FINALS, "tensor_v4_warm_input.jpg"), quality=95)
print(f"  Warm input saved")

# Binary mask
binary_mask_arr = ((1 - mask_binary) * 255).astype(np.uint8)
binary_mask = Image.fromarray(binary_mask_arr, "L").convert("RGB")

# Gradient mask
bg_area = (mask_binary == 0).astype(np.float64)
dist = distance_transform_edt(bg_area)
fade_px = int(short_edge * 0.30)  # 30% of short edge for wider fade
gradient = np.clip(dist / fade_px, 0, 1)
inpaint_values = np.where(mask_binary > 0, 0, (100 + gradient * 155)).astype(np.uint8)
gradient_mask = Image.fromarray(inpaint_values, "L").convert("RGB")

# --- Upload ---
print("Uploading warm image...")
warm_rid, uw, uh = upload_to_tensor(warm_img)
print(f"  Warm image: {warm_rid} ({uw}x{uh})")

print("Uploading original dark image...")
dark_rid, _, _ = upload_to_tensor(img)
print(f"  Dark image: {dark_rid}")

bin_rid, _, _ = upload_to_tensor(binary_mask.resize((uw, uh), Image.LANCZOS))
grad_rid, _, _ = upload_to_tensor(gradient_mask.resize((uw, uh), Image.LANCZOS))
print(f"  Masks uploaded")

prompt = ("surreal volumetric clouds and soft ethereal haze, warm golden cream and "
          "soft blue-grey tones, dreamy atmospheric smoke, huge dark crow wings "
          "spreading outward, baroque oil painting atmosphere, soft directional light")
negative = "text, watermark, cartoon, flat, modern, digital, solid black, dark background"


def make_payload(img_rid, mask_rid, strength, fill_mode, cfg=7):
    return {
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
                    "maskImageResourceId": mask_rid,
                    "maskBlur": 4,
                    "resizeMode": "JUST_RESIZE",
                    "inpaintingFill": fill_mode,
                    "inpaintFullRes": False,
                    "inpaintFullResPadding": 32,
                    "diffusion": {
                        "width": uw,
                        "height": uh,
                        "prompts": [{"text": prompt, "weight": 1.0}],
                        "negativePrompts": [{"text": negative, "weight": 1.0}],
                        "sdModel": MODEL_ID,
                        "steps": 30,
                        "cfgScale": cfg,
                        "denoisingStrength": strength,
                        "sampler": "Euler a",
                    },
                },
            },
        ],
    }


tests = [
    # (tag, image_rid, mask_rid, strength, fill_mode, cfg)
    # Warm input + ORIGINAL fill
    ("warm_bin_orig_s95", warm_rid, bin_rid, 0.95, "ORIGINAL", 7),
    ("warm_grad_orig_s95", warm_rid, grad_rid, 0.95, "ORIGINAL", 7),
    # Warm input + LATENT_NOISE fill (generate from noise, ignore original)
    ("warm_bin_noise_s95", warm_rid, bin_rid, 0.95, "LATENT_NOISE", 7),
    # Dark input + LATENT_NOISE (noise should override the dark pixels)
    ("dark_bin_noise_s95", dark_rid, bin_rid, 0.95, "LATENT_NOISE", 7),
    # Warm input + higher CFG for more prompt adherence
    ("warm_bin_orig_cfg12", warm_rid, bin_rid, 0.95, "ORIGINAL", 12),
    # Dark input + FILL mode
    ("dark_bin_fill_s95", dark_rid, bin_rid, 0.95, "FILL", 7),
]

for tag, img_r, mask_r, strength, fill, cfg in tests:
    print(f"\n{'='*50}")
    print(f"Testing: {tag} (s={strength}, fill={fill}, cfg={cfg})")
    print(f"{'='*50}")
    payload = make_payload(img_r, mask_r, strength, fill, cfg)
    url = run_tensor_job(payload)
    if url:
        resp = requests.get(url, timeout=60)
        out_img = Image.open(BytesIO(resp.content)).convert("RGB")
        stat = ImageStat.Stat(out_img)
        brightness = stat.mean[0]
        out = os.path.join(FINALS, f"tensor_v4_{tag}.jpg")
        out_img.save(out, quality=95)
        push_image(out, f"v4 {tag}", f"fill={fill}, cfg={cfg}, bright={brightness:.0f}")
        print(f"  Saved: {out} (brightness={brightness:.0f})")
    else:
        print(f"  FAILED")

print("\n=== DONE ===")
