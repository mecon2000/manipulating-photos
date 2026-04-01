import os
import sys
import requests
import json
import base64
import uuid
import time
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageDraw
import numpy as np
from io import BytesIO

# API Keys
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

# Model IDs
MODEL_NSFW = "965126062386242266" # Z-Image-Uncensored-fp16-v3
MODEL_TURBO = "979103055941916325" # Z-Image-Turbo
MODEL_FLUX = "1046927429141014138" # FLUX.1-dev

def run_fal_rembg(image_path):
    print(f"--- Step 1: Extracting Mask using Fal.ai (rembg) ---")
    url = "https://fal.run/fal-ai/rembg"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {"image_url": f"data:image/jpeg;base64,{img_b64}"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        res = response.json()
        mask_url = res["image"]["url"]
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).split()[3]
        return mask_img
    else:
        print(f"Fal.ai Error {response.status_code}: {response.text}")
        return None

def pinching_fill(img, mask):
    """
    Fills the mask area by 'pinching' colors from the edges of the mask inwards.
    A simple implementation using a strong radial blur/smear.
    """
    print("--- BG Prep: Applying 'pinching' fill to subject hole ---")
    img_np = np.array(img).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32) / 255.0
    
    # Simple Content-Aware-ish fill: blur the background heavily under the mask
    # and then feather it in.
    bg_only = img.copy()
    # Paste a blurred version over the mask area
    blurred_bg = img.filter(ImageFilter.GaussianBlur(radius=20))
    bg_only.paste(blurred_bg, mask=mask)
    
    # For a 'pinched' look, we'd ideally smear boundary pixels. 
    # Here we'll just do a multi-pass large blur to fill the void.
    for _ in range(3):
        bg_only = bg_only.filter(ImageFilter.GaussianBlur(radius=10))
        # Re-paste original background (where mask is 0)
        bg_only.paste(img, mask=ImageOps.invert(mask))
    
    return bg_only

def run_tensor_job(payload):
    url = f"{TENSOR_BASE_URL}/jobs"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Tensor Art Job Creation Error {response.status_code}: {response.text}")
        return None
    
    result = response.json()
    job_id = result.get("job", {}).get("id")
    if not job_id: return None

    # Polling
    for _ in range(60):
        time.sleep(5)
        res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers)
        job_data = res.json()
        status = job_data.get("job", {}).get("status")
        if status == "SUCCESS":
            return job_data["job"]["successInfo"]["images"][0]["url"]
        elif status == "FAILED":
            print(f"Tensor Art Job {job_id} failed: {json.dumps(job_data, indent=2)}")
            return None
    return None

def upload_to_tensor(image_pil):
    buffer = BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer_size = buffer.tell()
    buffer.seek(0)
    
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers)
    put_info = res.json()
    
    requests.put(put_info["putUrl"], data=buffer, headers=put_info["headers"])
    return put_info["resourceId"]

def tensor_stylize(image_pil, prompt, model_id, strength):
    # Determine resource ID
    resource_id = upload_to_tensor(image_pil)
    
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": { "seed": -1, "count": 1 }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": image_pil.width,
                    "height": image_pil.height,
                    "prompts": [{ "text": prompt, "weight": 1.0 }],
                    "sdModel": model_id,
                    "steps": 30 if "FLUX" not in model_id else 20,
                    "cfgScale": 7,
                    "denoisingStrength": strength,
                    "image": resource_id
                }
            }
        ]
    }
    
    img_url = run_tensor_job(payload)
    if img_url:
        return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return None

def check_not_black(img, name):
    stat = ImageStat.Stat(img)
    if sum(stat.extrema[0]) == 0 and sum(stat.extrema[1]) == 0 and sum(stat.extrema[2]) == 0:
        raise Exception(f"CRITICAL ERROR: {name} is solid black!")
    print(f"--- Check passed: {name} is not black ---")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--style", default="Oil_Paint_Impasto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_orig = Image.open(args.source).convert("RGB")
    img_orig.save(os.path.join(args.output_dir, "0_original.jpg"))
    
    # Step 1: Mask
    mask = run_fal_rembg(args.source)
    if not mask: return
    mask.save(os.path.join(args.output_dir, "1_mask.png"))
    
    # Step 2: BG Prep (Pinching Fill)
    bg_prepped = pinching_fill(img_orig, mask)
    bg_prepped.save(os.path.join(args.output_dir, "2_bg_prep_pinched.jpg"))
    
    # Step 2b: AI Fill (Strength 0.2)
    print("--- Step 2b: AI Filling Background (Strength 0.2) ---")
    bg_ai_fill = tensor_stylize(bg_prepped, "remove, clean background, abstract", MODEL_TURBO, 0.2)
    if not bg_ai_fill: return
    check_not_black(bg_ai_fill, "BG AI Fill")
    bg_ai_fill.save(os.path.join(args.output_dir, "3_bg_ai_fill.jpg"))
    
    # Step 4: Stylize BG (Strength 0.8)
    print("--- Step 4: Stylizing Background (Strength 0.8) ---")
    bg_stylized = tensor_stylize(bg_ai_fill, f"An abstract fine art {args.style} background, moody, cinematic", MODEL_TURBO, 0.8)
    if not bg_stylized: return
    check_not_black(bg_stylized, "Stylized BG")
    bg_stylized.save(os.path.join(args.output_dir, "4_bg_stylized.jpg"))
    
    # Step 6: Stylize Model (Strength 0.4)
    print("--- Step 6: Stylizing Model (Strength 0.4) ---")
    # Isolate model first
    model_only = Image.new("RGB", img_orig.size, (0,0,0))
    model_only.paste(img_orig, mask=mask)
    model_only.save(os.path.join(args.output_dir, "5_model_only.jpg"))
    
    model_stylized = tensor_stylize(model_only, f"A fine art portrait, {args.style} style, high detail, realistic skin texture", MODEL_NSFW, 0.4)
    if not model_stylized: return
    check_not_black(model_stylized, "Stylized Model")
    model_stylized.save(os.path.join(args.output_dir, "6_model_stylized.jpg"))
    
    # Step 8: Composite
    print("--- Step 8: Final Compositing ---")
    # Soften mask slightly for better blend
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    final = Image.composite(model_stylized, bg_stylized, soft_mask)
    final.save(os.path.join(args.output_dir, "7_final_result.jpg"), "JPEG", quality=95)
    
    print(f"Workflow Complete! Results in {args.output_dir}")

if __name__ == '__main__':
    main()
