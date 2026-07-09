#!/home/rong/openclaw-venv/bin/python3
"""Test: does Flux inpainting work on SFW photos? If yes, NSFW blocking is the problem."""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client, requests
from PIL import Image, ImageStat
from masking import build_mask
from io import BytesIO
from notify import push_image

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")
src = os.path.expanduser("~/.openclaw/workspace/_photos/Daniella/Processed/BLD_4084.jpg")
img = Image.open(src).convert("RGB")
w, h = img.size

mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)
inpaint_mask = Image.fromarray(((1 - mask_binary) * 255).astype(np.uint8), "L")

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    img.save(tmp, format="JPEG", quality=95)
    img_url = fal_client.upload_file(tmp.name)
    os.unlink(tmp.name)
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    inpaint_mask.save(tmp, format="PNG")
    mask_url = fal_client.upload_file(tmp.name)
    os.unlink(tmp.name)

print("Inpainting SFW photo (Daniella in red dress)...")
handle = fal_client.submit("fal-ai/flux-general/inpainting", arguments={
    "image_url": img_url,
    "mask_url": mask_url,
    "prompt": "surreal volumetric clouds and ethereal haze, warm golden cream tones, baroque oil painting atmosphere",
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
    stat = ImageStat.Stat(out_img)
    brightness = stat.mean[0]
    print(f"Result brightness: {brightness:.1f} (black < 10, normal > 50)")
    if brightness < 10:
        print(">>> BLACK — NSFW theory WRONG. Flux inpainting is broken for all photos.")
    else:
        print(">>> VISIBLE — NSFW theory CONFIRMED. Flux blocks Nastia but works on Daniella.")
    out = os.path.join(FINALS, "sfw_inpaint_test_daniella.jpg")
    out_img.save(out, quality=95)
    push_image(out, "SFW inpaint test", f"Daniella brightness={brightness:.0f}")
    print(f"Saved: {out}")
else:
    print("No images returned")
