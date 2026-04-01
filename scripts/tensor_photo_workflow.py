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
# Ronnie likes this one: Z-Image-Uncensored-fp16-v3
MODEL_DEFAULT = "965126062386242266" 

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
    """
    print("--- BG Prep: Applying 'pinching' fill to subject hole ---")
    
    # Fill mask area with average color first to avoid black artifacts
    avg_color = ImageStat.Stat(img).median
    bg_only = img.copy()
    fill = Image.new("RGB", img.size, tuple(avg_color))
    bg_only.paste(fill, mask=mask)
    
    # Iterative large BoxBlur to smear surrounding pixels into the hole
    # This creates a 'streaky' effect that pulls colors from the boundary.
    for r in [4, 8, 16, 32, 64]:
        smeared = bg_only.filter(ImageFilter.BoxBlur(radius=r))
        bg_only.paste(smeared, mask=mask)
        
    # Re-paste original background (outside mask)
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
    # Ensure dimensions are multiples of 8
    w, h = image_pil.size
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    if (new_w, new_h) != (w, h):
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)

    buffer = BytesIO()
    image_pil.save(buffer, format='JPEG', quality=95)
    buffer_size = buffer.tell()
    buffer.seek(0)
    
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers)
    put_info = res.json()
    
    requests.put(put_info["putUrl"], data=buffer, headers=put_info["headers"])
    return put_info["resourceId"], new_w, new_h

def tensor_stylize(image_pil, prompt, strength):
    # Determine resource ID and normalized dimensions
    resource_id, w, h = upload_to_tensor(image_pil)
    
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
                    "width": w,
                    "height": h,
                    "prompts": [{ "text": prompt, "weight": 1.0 }],
                    "sdModel": MODEL_DEFAULT,
                    "steps": 30,
                    "cfgScale": 7,
                    "denoisingStrength": strength,
                    "image": resource_id,
                    "sampler": "Euler a"
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
    
    # Step 3: AI Fill Background (Strength 0.2)
    print("--- Step 3: AI Filling Background (Strength 0.2) ---")
    bg_ai_fill = tensor_stylize(bg_prepped, "simple background, clean art studio", 0.2)
    if not bg_ai_fill: return
    check_not_black(bg_ai_fill, "BG AI Fill")
    bg_ai_fill.save(os.path.join(args.output_dir, "3_bg_ai_fill.jpg"))
    
    # Step 4: Stylize BG (Strength 0.8)
    print("--- Step 4: Stylizing Background (Strength 0.8) ---")
    bg_stylized = tensor_stylize(bg_ai_fill, f"An abstract fine art {args.style} background, moody, cinematic, dark artistic atmosphere", 0.8)
    if not bg_stylized: return
    check_not_black(bg_stylized, "Stylized BG")
    bg_stylized.save(os.path.join(args.output_dir, "4_bg_stylized.jpg"))
    
    # Step 6: Stylize Model (Strength 0.4)
    print("--- Step 6: Stylizing Model (Strength 0.4) ---")
    model_only = Image.new("RGB", img_orig.size, (0,0,0))
    model_only.paste(img_orig, mask=mask)
    model_only.save(os.path.join(args.output_dir, "5_model_only.jpg"))
    
    model_stylized = tensor_stylize(model_only, f"A fine art portrait, {args.style} style, high detail, realistic skin texture, masterwork art", 0.4)
    if not model_stylized: return
    check_not_black(model_stylized, "Stylized Model")
    model_stylized.save(os.path.join(args.output_dir, "6_model_stylized.jpg"))
    
    # Step 8: Composite
    print("--- Step 8: Final Compositing ---")
    soft_mask = mask.resize(model_stylized.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=2))
    final = Image.composite(model_stylized, bg_stylized, soft_mask)
    final.save(os.path.join(args.output_dir, "7_final_result.jpg"), "JPEG", quality=95)
    
    print(f"Workflow Complete! Results in {args.output_dir}")

if __name__ == '__main__':
    main()
