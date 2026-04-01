import os
import json
import requests
import uuid
import time
from PIL import Image
from io import BytesIO
from datetime import datetime

# Configuration
PHOTO_PATH = "catalog/preview/Elly__BLD_5440E.jpg"
STYLE_NAME = "Smoke & Mirrors"
OUTPUT_ROOT = "outputs/daily_game/retry_elly_identity/"
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
STRENGTH = 0.6 
MODEL_ID = "965126062386242266"

def get_style_prompt():
    with open("state/style_bank.json", "r") as f:
        styles = json.load(f)
        for s in styles:
            if s["name"] == STYLE_NAME:
                return s["prompt_add"]
    return "smoke and mirrors style"

def get_resource_id(local_path):
    img = Image.open(local_path).convert('RGB')
    w, h = img.size
    if w > h:
        new_w, new_h = 1024, int((h / w) * 1024)
    else:
        new_h, new_w = 1024, int((w / h) * 1024)
    new_w, new_h = (new_w // 8) * 8, (new_h // 8) * 8
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='PNG')
    url = f"{BASE_URL}/resource/image"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={}).json()
    requests.put(response["putUrl"], data=buf.getvalue(), headers=response["headers"])
    return response["resourceId"], new_w, new_h

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    res_id, w, h = get_resource_id(PHOTO_PATH)
    style_prompt = get_style_prompt()
    
    # User's specific identity prompt
    identity_prompt = "subject: use the provided photo as reference. Preserve exact facial features, proportions, skin tone, eyes, nose, chin, hair and expression — identity must be as close as possible to the reference photo."
    
    full_prompt = f"artistic figure photography, high quality, masterpiece. {identity_prompt} Style: {style_prompt}."
    
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    
    links = []
    for i in range(2):
        payload = {
            "requestId": str(uuid.uuid4()),
            "stages": [
                {"type": "INPUT_INITIALIZE", "inputInitialize": {"image_resource_id": res_id, "count": 1}},
                {"type": "DIFFUSION", "diffusion": {
                    "width": w, "height": h,
                    "prompts": [{"text": full_prompt}],
                    "sdModel": MODEL_ID, "steps": 30, "cfgScale": 7, "denoisingStrength": STRENGTH
                }}
            ]
        }
        job_id = requests.post(f"{BASE_URL}/jobs", headers=headers, json=payload).json()["job"]["id"]
        
        while True:
            result = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=headers).json()
            status = result["job"]["status"]
            if status == "SUCCESS":
                img_url = result["job"]["successInfo"]["images"][0]["url"]
                img_data = requests.get(img_url).content
                filename = f"Elly_Smoke_Mirrors_Strict_Identity_{i+1}.png"
                local_path = os.path.join(OUTPUT_ROOT, filename)
                with open(local_path, "wb") as f:
                    f.write(img_data)
                
                gdrive_path = f"gdrive:_photos from openclaw/outputs/daily_game/public/retries/"
                os.system(f"rclone copy '{local_path}' '{gdrive_path}'")
                link_proc = os.popen(f"rclone link '{gdrive_path}{filename}'")
                public_link = link_proc.read().strip()
                links.append(public_link)
                print(f"Success {i+1}: {public_link}")
                break
            elif status == "FAILED":
                print(f"Job {i+1} failed")
                break
            time.sleep(5)
    
    print(f"\nFinal Prompt: {full_prompt}")

if __name__ == "__main__":
    main()
