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
    buffer = BytesIO()
    image_pil.save(buffer, format='PNG')
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers).json()
    requests.put(res["putUrl"], data=buffer.getvalue(), headers=res["headers"])
    return res["resourceId"]

def run_tensor_inpaint(image_pil, mask_pil, output_dir, model_id):
    print(f"--- Testing Model: {model_id} ---")
    
    # ROI Logic
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
    
    try:
        img_res_id = upload_to_tensor(image_crop)
        mask_res_id = upload_to_tensor(mask_crop)
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            { "type": "INPUT_INITIALIZE", "inputInitialize": { "image_resource_id": img_res_id, "count": 1, "seed": 42 } },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": image_crop.width, "height": image_crop.height,
                    "prompts": [{ "text": "empty background, no person, natural scene", "weight": 1 }],
                    "sdModel": model_id,
                    "steps": 28, "cfgScale": 6.5, "denoisingStrength": 0.75, "sampler": "Euler a",
                    "mask_resource_id": mask_res_id
                }
            }
        ]
    }

    url = f"{TENSOR_BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    job_id = response.json()["job"]["id"]
    print(f"Job ID: {job_id}")
    
    for _ in range(60):
        time.sleep(5)
        res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers).json()
        status = res["job"]["status"]
        print(f"Status: {status}")
        if status == "SUCCESS":
            img_url = res["job"]["successInfo"]["images"][0]["url"]
            result_crop = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
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
    
    # Get mask using rembg (fal.ai)
    url = "https://fal.run/fal-ai/rembg"
    headers = {"Authorization": f"Key {FAL_API_KEY}"}
    with open(source, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    res = requests.post(url, headers=headers, json={"image_url": f"data:image/jpeg;base64,{img_b64}"}).json()
    mask = Image.open(requests.get(res["image"]["url"], stream=True).raw).split()[3]
    
    models_to_test = [
        "sdxl-inpainting",
        "sdxl_1.0_inpainting",
        "854427802932352804", # ANNO Inpaint
        "677354709792734564"  # JuggerXL Inpaint
    ]
    
    for mid in models_to_test:
        result = run_tensor_inpaint(img_orig, mask, "outputs/", mid)
        if result:
            filename = f"outputs/test_inpaint_{mid.replace('.', '_')}.jpg"
            result.save(filename)
            print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
