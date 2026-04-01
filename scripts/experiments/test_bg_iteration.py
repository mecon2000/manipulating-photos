import os
import sys
import requests
import json
import uuid
import time
import subprocess
from PIL import Image
from io import BytesIO

# API Keys
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

# Target Source: 3_bg_ai_fill from the latest run (V6)
SOURCE_IMG = "outputs/shahar_zach_workflow_pro_20260401_123448/3_bg_ai_fill.jpg"
OUTPUT_DIR = "outputs/bg_style_test_20260401_1304"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_UNCENSORED = "965126062386242266"

# Styles from L marked photos
STYLES = [
    {
        "name": "Indigo_Dye_Aesthetic",
        "prompt": "An abstract fine art Indigo Dye Aesthetic background, deep blue monochromatic look, shibori fabric patterns, moody, cinematic"
    },
    {
        "name": "Smoke_&_Mirrors",
        "prompt": "An abstract fine art Smoke & Mirrors background, heavy swirling smoke, multiple mirror reflections, moody, cinematic"
    },
    {
        "name": "Oil_Paint_Impasto",
        "prompt": "An abstract fine art Oil Paint Impasto background, thick brushstrokes, heavy texture, visible paint dabs, rich color palette, moody, cinematic"
    }
]

STRENGTHS = [0.3, 0.4, 0.5]

def run_tensor_job(payload):
    url = f"{TENSOR_BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    job_id = response.json().get("job", {}).get("id")
    for _ in range(60):
        time.sleep(5)
        res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers).json()
        status = res.get("job", {}).get("status")
        if status == "SUCCESS":
            return res["job"]["successInfo"]["images"][0]["url"]
        elif status == "FAILED":
            print(f"Job Failed: {json.dumps(res, indent=2)}")
            return None
    return None

def upload_to_tensor(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    res = requests.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers).json()
    requests.put(res["putUrl"], data=buffer.getvalue(), headers=res["headers"])
    return res["resourceId"], new_w, new_h

def main():
    resource_id, w, h = upload_to_tensor(SOURCE_IMG)
    
    for style in STYLES:
        for s in STRENGTHS:
            print(f"--- Testing {style['name']} at Strength {s} ---")
            payload = {
                "requestId": str(uuid.uuid4()),
                "stages": [
                    {"type": "INPUT_INITIALIZE", "inputInitialize": {"image_resource_id": resource_id, "count": 1, "seed": 42}},
                    {"type": "DIFFUSION", "diffusion": {
                        "width": w, "height": h, "prompts": [{"text": style['prompt']}],
                        "sdModel": MODEL_UNCENSORED, "steps": 30, "cfgScale": 5.5, 
                        "denoisingStrength": s, "sampler": "Euler a"
                    }}
                ]
            }
            url = run_tensor_job(payload)
            if url:
                filename = f"{style['name']}_s{s}.jpg"
                local_path = os.path.join(OUTPUT_DIR, filename)
                with open(local_path, "wb") as f:
                    f.write(requests.get(url).content)
                print(f"  Saved: {filename}")
            else:
                print(f"  Failed: {style['name']} at {s}")

    # Upload results to GDrive
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gdrive_path = f"gdrive:_photos from openclaw/daily_game/public/BG_STYLE_STRENGTH_MATRIX_{timestamp}"
    subprocess.run(["rclone", "copy", OUTPUT_DIR, gdrive_path, "-vv"], check=True)
    res = subprocess.run(["rclone", "link", gdrive_path], capture_output=True, text=True)
    print(f"\nMatrix Complete! Public Link: {res.stdout.strip()}")

if __name__ == "__main__":
    main()
