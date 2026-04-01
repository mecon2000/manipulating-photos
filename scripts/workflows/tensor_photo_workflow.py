import os
import sys
import requests
import json
import base64
import uuid
import time
import shutil
import subprocess
import re
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageDraw
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta

# Configure line buffering for stdout
sys.stdout.reconfigure(line_buffering=True)

# API Keys
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

# Model IDs
MODEL_DEFAULT = "965126062386242266" # Z-Image-Uncensored-fp16-v3

def log_to_file(output_dir, message):
    log_path = os.path.join(output_dir, "workflow.log")
    # Israel Time (UTC+3 for April)
    israel_time = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{israel_time}] {message}"
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
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        res = response.json()
        mask_url = res["image"]["url"]
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).split()[3]
        return mask_img
    else:
        log_to_file(output_dir, f"Fal.ai Error {response.status_code}: {response.text}")
        return None

def run_tensor_job(payload, output_dir):
    url = f"{TENSOR_BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        log_to_file(output_dir, f"Tensor Art Job Creation Error {response.status_code}: {response.text}")
        return None
    
    job_data = response.json()
    job_id = job_data.get("job", {}).get("id")
    if not job_id: return None
    for _ in range(60):
        time.sleep(5)
        res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers).json()
        status = res.get("job", {}).get("status")
        if status == "SUCCESS":
            return res["job"]["successInfo"]["images"][0]["url"]
        elif status == "FAILED":
            log_to_file(output_dir, f"Job Failed: {json.dumps(res, indent=2)}")
            return None
    return None

def upload_to_tensor(image_pil, output_dir):
    w, h = image_pil.size
    MAX_PIXELS = 2073600
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
        w, h = image_pil.size
    
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    if (new_w, new_h) != (w, h):
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
    
    buffer = BytesIO()
    image_pil.save(buffer, format='PNG') # Use PNG for internal quality
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers).json()
    requests.put(res["putUrl"], data=buffer.getvalue(), headers=res["headers"])
    return res["resourceId"], new_w, new_h

def run_tensor_inpaint(image_pil, mask_pil, output_dir, dilation_px=25, margin_px=96):
    log_to_file(output_dir, f"--- Step 2: Cleaning Background using Tensor (Inpaint, Dilation: {dilation_px}px) ---")
    
    # 1. Harden and Dilate Mask
    mask = mask_pil.convert("L")
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    k = dilation_px if dilation_px % 2 != 0 else dilation_px + 1
    mask = mask.filter(ImageFilter.MaxFilter(k))
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    mask.save(os.path.join(output_dir, "1b_dilated_mask_final.png"))

    # 2. Compute ROI
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        log_to_file(output_dir, "Empty mask, skipping inpaint")
        return image_pil

    x_min, x_max = max(0, xs.min() - margin_px), min(image_pil.width, xs.max() + margin_px)
    y_min, y_max = max(0, ys.min() - margin_px), min(image_pil.height, ys.max() + margin_px)
    
    image_crop = image_pil.crop((x_min, y_min, x_max, y_max))
    mask_crop = mask.crop((x_min, y_min, x_max, y_max))
    
    # 3. Upload to Tensor
    img_res_id, cw, ch = upload_to_tensor(image_crop, output_dir)
    mask_res_id, mw, mh = upload_to_tensor(mask_crop, output_dir)
    
    # 4. Job Payload (Using Juggernaut for realistic inpainting)
    MODEL_INPAINT = "677354709792734564"
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE", 
                "inputInitialize": { "image_resource_id": img_res_id, "count": 1, "seed": 42 }
            },
            {
                "type": "DIFFUSION", 
                "diffusion": {
                    "width": cw, "height": ch, 
                    "prompts": [{ "text": "clean background, remove person completely, reconstruct missing area naturally, sharp edges, realistic scene, no blur", "weight": 1.0 }],
                    "negative_prompts": [{ "text": "person, human, body, blur, artifacts, distortion", "weight": 1.0 }],
                    "sdModel": MODEL_INPAINT, "steps": 30, "cfgScale": 7.5, 
                    "denoisingStrength": 1.0, "sampler": "Euler a",
                    "mask_resource_id": mask_res_id
                }
            }
        ]
    }
    
    img_url = run_tensor_job(payload, output_dir)
    if img_url:
        result_crop = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        result_crop = result_crop.resize((x_max - x_min, y_max - y_min), Image.LANCZOS)
        final = image_pil.copy()
        final.paste(result_crop, (x_min, y_min))
        return final
    
    log_to_file(output_dir, "Tensor Inpaint Failed")
    return None

def tensor_stylize(image_pil, prompt, strength, cfg_scale, output_dir):
    resource_id, w, h = upload_to_tensor(image_pil, output_dir)
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE", 
                "inputInitialize": { "image_resource_id": resource_id, "count": 1, "seed": 42 }
            },
            {
                "type": "DIFFUSION", 
                "diffusion": {
                    "width": w, "height": h, 
                    "prompts": [{ "text": prompt, "weight": 1.0 }],
                    "sdModel": MODEL_DEFAULT, "steps": 30, "cfgScale": cfg_scale, 
                    "denoisingStrength": strength, "sampler": "Euler a"
                }
            }
        ]
    }
    img_url = run_tensor_job(payload, output_dir)
    if img_url: return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
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
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        res = response.json()
        img_url = res.get("image", {}).get("url")
        if img_url:
            return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    log_to_file(output_dir, f"Face Swap Failed: {response.text}")
    return None

def upload_to_gdrive(local_dir, model_name, photo_name, timestamp):
    gdrive_path = f"gdrive:_photos from openclaw/daily_game/public/{model_name}_{photo_name}_{timestamp}"
    try:
        subprocess.run(["rclone", "copy", local_dir, gdrive_path], check=True)
        res = subprocess.run(["rclone", "link", gdrive_path], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"GDrive Upload Failed: {e}"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--style", default="Baroque Chiaroscuro")
    parser.add_argument("--prompt-add", default="")
    parser.add_argument("--bg-strength", type=float, default=0.6)
    parser.add_argument("--cfg-scale", type=float, default=6.5)
    parser.add_argument("--dilation", type=int, default=25)
    parser.add_argument("--faceswap", action="store_true", default=True)
    args = parser.parse_args()

    basename = os.path.basename(args.source)
    photo_name = os.path.splitext(basename)[0]
    
    model_name = args.model_name
    if not model_name:
        match = re.search(r"(Original_|BLD_|Rong_IMG_)(.*?)_(.*)\.(jpg|png|jpeg|JPG)", basename, re.IGNORECASE)
        if match:
            model_name = match.group(2).replace(" ", "_")
            photo_name = match.group(3).replace(" ", "_")
        else:
            model_name = "Unknown"

    prompt_add = args.prompt_add
    if not prompt_add:
        if "Baroque" in args.style:
            prompt_add = "dramatic Caravaggio-style lighting, intense contrast, theatrical, moody"
        elif "Oil Impasto" in args.style:
            prompt_add = "thick paint, heavy texture, visible brushstrokes, rich colors, artistic"
        elif "Indigo" in args.style:
            prompt_add = "deep indigo wash, watercolor stains, bleeding ink, ethereal, monochromatic blue"
        elif "Smoke" in args.style:
            prompt_add = "ethereal smoke swirls, wispy textures, atmospheric, moody, dreamlike"
        else:
            prompt_add = "fine art, detailed, artistic"

    # Israel Time (UTC+3 for April)
    israel_dt = datetime.utcnow() + timedelta(hours=3)
    timestamp = israel_dt.strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/workflow_pro_v14_{photo_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    shutil.copy(__file__, os.path.join(output_dir, f"tensor_photo_workflow_v14_{timestamp}.py"))
    log_to_file(output_dir, f"Starting Workflow Pro v14 (Model: {model_name} | Style: {args.style})")
    log_to_file(output_dir, f"BG Strength: {args.bg_strength} | Dilation: {args.dilation}px")

    img_orig = Image.open(args.source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"))
    
    mask = run_fal_rembg(args.source, output_dir)
    if not mask: return
    mask.save(os.path.join(output_dir, "1_mask.png"))
    
    # Step 2: Generative Inpainting with Tensor
    bg_clean = run_tensor_inpaint(img_orig, mask, output_dir, args.dilation)
    if not bg_clean:
        log_to_file(output_dir, "Inpainting failed, falling back to original")
        bg_clean = img_orig.copy()
    bg_clean.save(os.path.join(output_dir, "2_bg_clean_tensor.jpg"))
    
    log_to_file(output_dir, f"--- Step 4: Stylizing Background (Strength {args.bg_strength}) ---")
    bg_prompt = f"An abstract fine art {args.style} background, {prompt_add}, moody, cinematic, painterly textures"
    bg_stylized = tensor_stylize(bg_clean, bg_prompt, args.bg_strength, args.cfg_scale, output_dir)
    if not bg_stylized: return
    bg_stylized.save(os.path.join(output_dir, "4_bg_stylized.jpg"))
    
    log_to_file(output_dir, "--- Step 6: Stylizing Model (Strength 0.4) ---")
    model_only = Image.new("RGB", img_orig.size, (0,0,0))
    model_only.paste(img_orig, mask=mask)
    model_only.save(os.path.join(output_dir, "5_model_only.jpg"))
    
    model_prompt = f"A fine art portrait, {args.style} style, {prompt_add}, high detail, realistic skin texture"
    model_stylized = tensor_stylize(model_only, model_prompt, 0.4, args.cfg_scale, output_dir)
    if not model_stylized: return
    model_stylized.save(os.path.join(output_dir, "6_model_stylized.jpg"))
    
    log_to_file(output_dir, "--- Step 8: Final Compositing ---")
    soft_mask = mask.resize(model_stylized.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=3))
    final = Image.composite(model_stylized, bg_stylized, soft_mask)
    final_path = os.path.join(output_dir, "7_final_result.jpg")
    final.save(final_path, "JPEG", quality=95)
    
    if args.faceswap:
        final_swapped = run_fal_faceswap(args.source, final_path, output_dir)
        if final_swapped:
            final_swapped.save(os.path.join(output_dir, "8_final_swapped.jpg"), "JPEG", quality=95)
    
    log_to_file(output_dir, "--- Final Step: Uploading to GDrive ---")
    public_link = upload_to_gdrive(output_dir, model_name, photo_name, timestamp)
    log_to_file(output_dir, f"Workflow Complete! Public Link: {public_link}")

if __name__ == '__main__':
    main()
