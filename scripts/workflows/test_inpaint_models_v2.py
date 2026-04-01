import os
import sys
import requests
import json
import base64
import uuid
import time
from PIL import Image, ImageFilter
from io import BytesIO
import numpy as np

# API Keys
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def upload_to_tensor(image_pil):
    # Ensure Multiple of 8
    w, h = image_pil.size
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    if (new_w, new_h) != (w, h):
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
    
    # Tensor Art Limit check
    MAX_PIXELS = 1048576 # 1024x1024 for SDXL stability
    if new_w * new_h > MAX_PIXELS:
        scale = (MAX_PIXELS / (new_w * new_h)) ** 0.5
        new_w, new_h = int(new_w * scale), int(new_h * scale)
        new_w, new_h = (new_w // 8) * 8, (new_h // 8) * 8
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)

    buffer = BytesIO()
    image_pil.save(buffer, format='PNG')
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers).json()
    requests.put(res["putUrl"], data=buffer.getvalue(), headers=res["headers"])
    return res["resourceId"], new_w, new_h

def run_tensor_inpaint(image_pil, mask_pil, mid):
    print(f"--- Testing Model: {mid} ---")
    mask = mask_pil.convert("L").point(lambda p: 255 if p > 127 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(25)).point(lambda p: 255 if p > 127 else 0)
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0: return None
    margin = int(max(image_pil.size) * 0.1)
    x_min, x_max = max(0, xs.min() - margin), min(image_pil.width, xs.max() + margin)
    y_min, y_max = max(0, ys.min() - margin), min(image_pil.height, ys.max() + margin)
    image_crop = image_pil.crop((x_min, y_min, x_max, y_max))
    mask_crop = mask.crop((x_min, y_min, x_max, y_max))
    
    img_res_id, cw, ch = upload_to_tensor(image_crop)
    mask_res_id, mw, mh = upload_to_tensor(mask_crop)

    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            { "type": "INPUT_INITIALIZE", "inputInitialize": { "image_resource_id": img_res_id, "count": 1, "seed": 42 } },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": cw, "height": ch,
                    "prompts": [{ "text": "empty background, no person, natural scene", "weight": 1 }],
                    "sdModel": mid,
                    "steps": 30, "cfgScale": 7, "denoisingStrength": 1.0, "sampler": "Euler a",
                    "mask_resource_id": mask_res_id
                }
            }
        ]
    }
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(f"{TENSOR_BASE_URL}/jobs", headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    job_id = response.json()["job"]["id"]
    for _ in range(60):
        time.sleep(5)
        res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers).json()
        status = res["job"]["status"]
        if status == "SUCCESS":
            img_url = res["job"]["successInfo"]["images"][0]["url"]
            result_crop = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            result_crop = result_crop.resize((x_max - x_min, y_max - y_min), Image.LANCZOS)
            final = image_pil.copy()
            final.paste(result_crop, (x_min, y_min))
            return final
        elif status == "FAILED":
            print(f"Failed: {res}")
            break
    return None

def main():
    source = "inputs/Rong_IMG_9214.jpg"
    img_orig = Image.open(source).convert("RGB")
    url = "https://fal.run/fal-ai/rembg"
    headers = {"Authorization": f"Key {FAL_API_KEY}"}
    with open(source, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    res = requests.post(url, headers=headers, json={"image_url": f"data:image/jpeg;base64,{img_b64}"}).json()
    mask = Image.open(requests.get(res["image"]["url"], stream=True).raw).split()[3]
    
    models = ["677354709792734564"] # JuggerXL
    for mid in models:
        result = run_tensor_inpaint(img_orig, mask, mid)
        if result:
            filename = f"outputs/test_inpaint_{mid}.jpg"
            result.save(filename)
            print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
