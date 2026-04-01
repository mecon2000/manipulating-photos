import os
import sys
import requests
import json
import base64
import uuid
import time
import shutil
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageDraw
import numpy as np
from io import BytesIO

# Configure line buffering for stdout to ensure logs are visible in real-time
sys.stdout.reconfigure(line_buffering=True)

# API Keys
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

# Model IDs
MODEL_DEFAULT = "965126062386242266" # Z-Image-Uncensored-fp16-v3

def log_to_file(output_dir, message):
    log_path = os.path.join(output_dir, "workflow.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_path, "a") as f:
        f.write(formatted_message + "\n")

def run_fal_rembg(image_path, output_dir):
    log_to_file(output_dir, "--- Step 1: Extracting Mask using Fal.ai (rembg) ---")
    url = "https://fal.run/fal-ai/rembg"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {"image_url": f"data:image/jpeg;base64,{img_b64}"}
    
    log_to_file(output_dir, f"Calling Fal.ai rembg for {image_path}")
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        res = response.json()
        mask_url = res["image"]["url"]
        log_to_file(output_dir, f"Mask URL: {mask_url}")
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).split()[3]
        return mask_img
    else:
        log_to_file(output_dir, f"Fal.ai Error {response.status_code}: {response.text}")
        return None

def run_fal_faceswap(source_path, target_path, output_dir):
    log_to_file(output_dir, "--- Step 9: Running Face Swap using Fal.ai ---")
    url = "https://fal.run/fal-ai/face-swap"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(source_path, "rb") as f:
        source_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open(target_path, "rb") as f:
        target_b64 = base64.b64encode(f.read()).decode('utf-8')
    data = {
        "base_image_url": f"data:image/jpeg;base64,{target_b64}",
        "swap_image_url": f"data:image/jpeg;base64,{source_b64}"
    }
    
    log_to_file(output_dir, "Calling Fal.ai Face Swap")
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        res = response.json()
        img_url = res.get("image", {}).get("url")
        if img_url:
            log_to_file(output_dir, f"Face Swap Success: {img_url}")
            return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    log_to_file(output_dir, f"Face Swap Failed: {response.text}")
    return None

def pinching_fill(img, mask, output_dir):
    log_to_file(output_dir, "--- BG Prep: Applying 'pinching' fill to subject hole ---")
    avg_color = ImageStat.Stat(img).median
    bg_only = img.copy()
    fill = Image.new("RGB", img.size, tuple(avg_color))
    bg_only.paste(fill, mask=mask)
    for r in [4, 8, 16, 32, 64]:
        smeared = bg_only.filter(ImageFilter.BoxBlur(radius=r))
        bg_only.paste(smeared, mask=mask)
    bg_only.paste(img, mask=ImageOps.invert(mask))
    return bg_only

def run_tensor_job(payload, output_dir):
    url = f"{TENSOR_BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    
    log_to_file(output_dir, f"Creating Tensor Art Job with payload: {json.dumps(payload, indent=2)}")
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        log_to_file(output_dir, f"Tensor Art Job Creation Error {response.status_code}: {response.text}")
        return None
    
    job_data = response.json()
    job_id = job_data.get("job", {}).get("id")
    log_to_file(output_dir, f"Job Created. ID: {job_id}")
    
    if not job_id: return None
    for i in range(60):
        time.sleep(5)
        res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers).json()
        status = res.get("job", {}).get("status")
        log_to_file(output_dir, f"Polling Job {job_id}: {status}")
        if status == "SUCCESS":
            img_url = res["job"]["successInfo"]["images"][0]["url"]
            log_to_file(output_dir, f"Job Success: {img_url}")
            return img_url
        elif status == "FAILED":
            log_to_file(output_dir, f"Job Failed: {json.dumps(res, indent=2)}")
            return None
    return None

def upload_to_tensor(image_pil, output_dir):
    w, h = image_pil.size
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    if (new_w, new_h) != (w, h):
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
    
    buffer = BytesIO()
    image_pil.save(buffer, format='JPEG', quality=95)
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    
    log_to_file(output_dir, "Uploading resource to Tensor Art...")
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers).json()
    put_url = res["putUrl"]
    resource_id = res["resourceId"]
    
    requests.put(put_url, data=buffer.getvalue(), headers=res["headers"])
    log_to_file(output_dir, f"Resource Uploaded. ID: {resource_id}")
    return resource_id, new_w, new_h

def tensor_stylize(image_pil, prompt, strength, output_dir):
    resource_id, w, h = upload_to_tensor(image_pil, output_dir)
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE", 
                "inputInitialize": { 
                    "image_resource_id": resource_id, 
                    "count": 1,
                    "seed": -1
                }
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
                    "sampler": "Euler a"
                }
            }
        ]
    }
    img_url = run_tensor_job(payload, output_dir)
    if img_url: return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--style", default="Oil_Paint_Impasto")
    parser.add_argument("--faceswap", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize log file
    open(os.path.join(args.output_dir, "workflow.log"), "w").close()
    
    # Copy script to output dir for auditability
    shutil.copy(__file__, os.path.join(args.output_dir, "tensor_photo_workflow.py"))
    
    log_to_file(args.output_dir, f"Starting Workflow Pro v4 for {args.source}")
    log_to_file(args.output_dir, f"Output directory: {args.output_dir}")
    log_to_file(args.output_dir, f"Style: {args.style}")

    img_orig = Image.open(args.source).convert("RGB")
    img_orig.save(os.path.join(args.output_dir, "0_original.jpg"))
    
    mask = run_fal_rembg(args.source, args.output_dir)
    if not mask: return
    mask.save(os.path.join(args.output_dir, "1_mask.png"))
    
    bg_prepped = pinching_fill(img_orig, mask, args.output_dir)
    bg_prepped.save(os.path.join(args.output_dir, "2_bg_prep_pinched.jpg"))
    
    log_to_file(args.output_dir, "--- Step 3: AI Filling Background (Strength 0.2) ---")
    bg_ai_fill = tensor_stylize(bg_prepped, "simple background, clean art studio", 0.2, args.output_dir)
    if not bg_ai_fill: return
    bg_ai_fill.save(os.path.join(args.output_dir, "3_bg_ai_fill.jpg"))
    
    log_to_file(args.output_dir, "--- Step 4: Stylizing Background (Strength 0.8) ---")
    bg_stylized = tensor_stylize(bg_ai_fill, f"An abstract fine art {args.style} background, moody, cinematic", 0.8, args.output_dir)
    if not bg_stylized: return
    bg_stylized.save(os.path.join(args.output_dir, "4_bg_stylized.jpg"))
    
    log_to_file(args.output_dir, "--- Step 6: Stylizing Model (Strength 0.4) ---")
    model_only = Image.new("RGB", img_orig.size, (0,0,0))
    model_only.paste(img_orig, mask=mask)
    model_only.save(os.path.join(args.output_dir, "5_model_only.jpg"))
    
    model_stylized = tensor_stylize(model_only, f"A fine art portrait, {args.style} style, high detail, realistic skin texture", 0.4, args.output_dir)
    if not model_stylized: return
    model_stylized.save(os.path.join(args.output_dir, "6_model_stylized.jpg"))
    
    log_to_file(args.output_dir, "--- Step 8: Final Compositing ---")
    soft_mask = mask.resize(model_stylized.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=2))
    final = Image.composite(model_stylized, bg_stylized, soft_mask)
    final_path = os.path.join(args.output_dir, "7_final_result.jpg")
    final.save(final_path, "JPEG", quality=95)
    
    if args.faceswap:
        final_swapped = run_fal_faceswap(args.source, final_path, args.output_dir)
        if final_swapped:
            final_swapped.save(os.path.join(args.output_dir, "8_final_swapped.jpg"), "JPEG", quality=95)
    
    log_to_file(args.output_dir, f"Workflow Complete! Results in {args.output_dir}")

if __name__ == '__main__':
    main()
