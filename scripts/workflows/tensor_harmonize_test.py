#!/home/rong/openclaw-venv/bin/python3
"""
Composite-then-rediffuse: take best local composite, run through Tensor Art
IMAGE_TO_INPAINT at low strength to harmonize lighting/color in the transition zone.
Face fully protected, BG mostly protected, only the seam zone gets harmonized.
"""
import os, sys, uuid, time, json, numpy as np, cv2
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


# --- Load source + BG ---
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

bg_path = os.path.join(FINALS, "seam_blend_flux_bg.jpg")
bg = Image.open(bg_path).convert("RGB")
if bg.size != (w, h):
    bg = bg.resize((w, h), Image.LANCZOS)

# --- Mask ---
print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)

# --- Build the best local composite (Laplacian pyramid + light wrap + LAB edge + wash) ---
print("Building composite...")
src_f = np.array(img).astype(np.float32)
bg_f = np.array(bg).astype(np.float32)

# Laplacian pyramid blend
def laplacian_pyramid(img_f, levels=6):
    pyr = []
    current = img_f.copy()
    for i in range(levels - 1):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        pyr.append(current - up)
        current = down
    pyr.append(current)
    return pyr

def gaussian_pyramid(mask_f, levels=6):
    pyr = [mask_f.copy()]
    current = mask_f.copy()
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyr.append(current)
    return pyr

def reconstruct(pyr):
    current = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        up = cv2.pyrUp(current, dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
        current = up + pyr[i]
    return current

levels = 6
src_pyr = laplacian_pyramid(src_f, levels)
bg_pyr = laplacian_pyramid(bg_f, levels)
mask_3ch = np.stack([mask_binary.astype(np.float32)] * 3, axis=-1)
mask_gpyr = gaussian_pyramid(mask_3ch, levels)
blended_pyr = [s * m + b * (1 - m) for s, b, m in zip(src_pyr, bg_pyr, mask_gpyr)]
lap_result = np.clip(reconstruct(blended_pyr), 0, 255).astype(np.float32)

# Light wrap
blur_r = max(30, int(short_edge * 0.08))
bg_blurred = cv2.GaussianBlur(bg_f, (0, 0), blur_r)
kernel_size = max(5, int(short_edge * 0.025))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
edge_band = ((mask_dilated - mask_binary) > 0).astype(np.float32)
edge_soft = cv2.GaussianBlur(edge_band, (0, 0), max(3, kernel_size // 2))
edge_3ch = edge_soft[:, :, np.newaxis]
lap_wrapped = lap_result * (1 - edge_3ch * 0.25) + bg_blurred * (edge_3ch * 0.25)

# LAB edge color match
bg_lab = cv2.cvtColor(np.array(bg).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
lap_lab = cv2.cvtColor(np.clip(lap_wrapped, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
edge_width = max(10, int(short_edge * 0.05))
kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
mask_eroded = cv2.erode(mask_binary, kernel_edge, iterations=1)
inner_edge = ((mask_binary - mask_eroded) > 0).astype(np.float32)
inner_soft = cv2.GaussianBlur(inner_edge, (0, 0), max(3, edge_width // 2))
for ch in range(3):
    bg_near = bg_lab[:, :, ch][edge_band > 0.3]
    subj_edge = lap_lab[:, :, ch][inner_soft > 0.3]
    if len(bg_near) == 0 or len(subj_edge) == 0:
        continue
    shift = (bg_near.mean() - subj_edge.mean()) * 0.4
    lap_lab[:, :, ch] += shift * inner_soft

combined = cv2.cvtColor(np.clip(lap_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

# Unified color wash
bg_mean_color = bg_f.mean(axis=(0, 1))
combined = combined * 0.92 + bg_mean_color[np.newaxis, np.newaxis, :] * 0.08

composite = np.clip(combined, 0, 255).astype(np.uint8)
composite_img = Image.fromarray(composite)
composite_img.save(os.path.join(FINALS, "harmonize_input_composite.jpg"), quality=95)
print(f"  Composite ready")

# --- Build harmonization masks ---
# Key idea: we want to rediffuse ONLY the transition zone
# Face = 0 (fully protected)
# Subject interior = low (barely touched)
# Subject edge + near BG = higher (harmonize zone)
# Far BG = 0 (already fine, don't waste diffusion on it)

print("Building harmonization masks...")

# Distance from subject edge
subj_dist = distance_transform_edt(mask_binary)     # into subject
bg_dist = distance_transform_edt(1 - mask_binary)   # into BG

# Face detection via MediaPipe for face protection
try:
    from body_segment import _segment_face_skin
    face_mask = _segment_face_skin(img)
    if face_mask is not None:
        face_binary = (np.array(face_mask.resize((w, h), Image.LANCZOS)) > 127).astype(np.float32)
        # Expand face region for safety
        face_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        face_expanded = cv2.dilate(face_binary.astype(np.uint8), face_kernel).astype(np.float32)
        face_soft = cv2.GaussianBlur(face_expanded, (0, 0), 25)
        print(f"  Face detected, coverage: {100*face_binary.mean():.1f}%")
    else:
        face_soft = np.zeros((h, w), dtype=np.float32)
        print("  No face detected, using top-quarter protection")
        face_soft[:h//4, :] = 1.0
        face_soft = cv2.GaussianBlur(face_soft, (0, 0), 50)
except Exception as e:
    print(f"  Face detection failed ({e}), using top-quarter")
    face_soft = np.zeros((h, w), dtype=np.float32)
    face_soft[:h//4, :] = 1.0
    face_soft = cv2.GaussianBlur(face_soft, (0, 0), 50)

# Build several mask variants
masks = {}

# Mask A: Seam-only — narrow band around subject edge
seam_inner = int(short_edge * 0.04)   # ~50px into subject
seam_outer = int(short_edge * 0.10)   # ~125px into BG
inner_w = np.clip(1.0 - subj_dist / max(seam_inner, 1), 0, 1)
outer_w = np.clip(1.0 - bg_dist / max(seam_outer, 1), 0, 1)
seam_zone = np.where(mask_binary > 0, inner_w, outer_w)
# Remove face from seam
seam_zone = seam_zone * (1.0 - face_soft * 0.9)
masks["seam"] = (seam_zone * 255).astype(np.uint8)

# Mask B: Full subject at low value + seam at higher value
# Subject body = 40 (very gentle touch), seam = 150, face = 0, far BG = 0
body_base = np.where(mask_binary > 0, 40, 0).astype(np.float32)
seam_boost = seam_zone * 150
mask_b = np.clip(body_base + seam_boost, 0, 255)
mask_b = mask_b * (1.0 - face_soft * 0.95)  # protect face
masks["body_seam"] = mask_b.astype(np.uint8)

# Mask C: Wider harmonization — entire subject at moderate + wide BG transition
wide_outer = int(short_edge * 0.20)   # ~250px into BG
outer_wide = np.clip(1.0 - bg_dist / max(wide_outer, 1), 0, 1)
mask_c = np.where(mask_binary > 0,
                   np.clip(80 + inner_w * 80, 0, 160),  # subject: 80-160
                   outer_wide * 120)                       # BG transition: 0-120
mask_c = mask_c * (1.0 - face_soft * 0.95)
masks["wide"] = np.clip(mask_c, 0, 255).astype(np.uint8)

# Save masks for inspection
for name, m in masks.items():
    Image.fromarray(m, "L").save(os.path.join(FINALS, f"harmonize_mask_{name}.png"))
    coverage = 100 * np.mean(m > 10)
    print(f"  {name}: coverage={coverage:.1f}%")

# --- Upload ---
print("\nUploading composite...")
comp_rid, uw, uh = upload_to_tensor(composite_img)
print(f"  Composite: {comp_rid} ({uw}x{uh})")

mask_rids = {}
for name, m in masks.items():
    m_rgb = Image.fromarray(m, "L").resize((uw, uh), Image.LANCZOS).convert("RGB")
    rid, _, _ = upload_to_tensor(m_rgb)
    mask_rids[name] = rid
    print(f"  {name} mask: {rid}")

prompt = (
    "dramatic dark oil painting atmosphere, volumetric smoke and clouds engulfing "
    "a woman, warm golden and charcoal tones, baroque chiaroscuro lighting, "
    "seamless blend between photo and painted elements, unified lighting"
)
negative = "text, watermark, cartoon, flat, cutout, pasted, collage, sharp edge between subject and background"

# --- Run harmonization passes ---
tests = [
    # (tag, mask_name, strength)
    ("seam_s30", "seam", 0.30),
    ("seam_s50", "seam", 0.50),
    ("seam_s70", "seam", 0.70),
    ("body_seam_s30", "body_seam", 0.30),
    ("body_seam_s50", "body_seam", 0.50),
    ("wide_s30", "wide", 0.30),
    ("wide_s50", "wide", 0.50),
]

for tag, mask_name, strength in tests:
    print(f"\n{'='*50}")
    print(f"Harmonize: {tag}")
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
                    "maskImageResourceId": mask_rids[mask_name],
                    "maskBlur": 4,
                    "resizeMode": "JUST_RESIZE",
                    "inpaintingFill": "ORIGINAL",
                    "inpaintFullRes": False,
                    "inpaintFullResPadding": 32,
                    "diffusion": {
                        "width": uw,
                        "height": uh,
                        "prompts": [{"text": prompt, "weight": 1.0}],
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
        out = os.path.join(FINALS, f"harmonize_{tag}.jpg")
        out_img.save(out, quality=95)
        push_image(out, f"Harmonize: {tag}", f"Rediffuse {mask_name} at {strength}")
        print(f"  Saved: {out}")
    else:
        print(f"  FAILED")

print("\n=== DONE ===")
