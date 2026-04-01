import os
import random
import json
import requests
import uuid
import time
import subprocess
from PIL import Image
from io import BytesIO
from datetime import datetime

# Configuration
OUTPUT_ROOT = "outputs/daily_game/"
DATABASE_FILE = "state/style_game_database.json"
STYLE_BANK_FILE = "state/style_bank.json"
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
STRENGTH = 0.55 
MODEL_ID = "965126062386242266" # Uncensored SD 1.5

def load_json(file_path, default=None):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default or []

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_resource_id(local_path):
    try:
        img = Image.open(local_path).convert('RGB')
        w, h = img.size
        if w > h:
            new_w, new_h = 1024, int((h / w) * 1024)
        else:
            new_h, new_w = 1024, int((w / h) * 1024)
        new_w, new_h = (new_w // 8) * 8, (new_h // 8) * 8
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=95)
        img_data = buf.getvalue()

        url = f"{BASE_URL}/resource/image"
        headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={})
        if response.status_code != 200: return None, None, None
        
        put_info = response.json()
        requests.put(put_info["putUrl"], data=img_data, headers=put_info["headers"])
        return put_info["resourceId"], new_w, new_h
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")
        return None, None, None

def create_job(resource_id, style_prompt, width, height):
    url = f"{BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    identity_prompt = "subject: use the provided photo as reference. Preserve exact facial features, skin tone, hair and expression. Realistic skin texture, no over-smoothing."
    content_desc = f"artistic figure photography, high quality, masterpiece. {identity_prompt}"
    
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {"type": "INPUT_INITIALIZE", "inputInitialize": {"image_resource_id": resource_id, "count": 1}},
            {"type": "DIFFUSION", "diffusion": {
                "width": width, "height": height,
                "prompts": [{"text": f"{content_desc} Style: {style_prompt}."}],
                "sdModel": MODEL_ID, "steps": 35, "cfgScale": 8, "denoisingStrength": STRENGTH
            }}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200: 
        print(f"Error creating job: {response.text}")
        return None
    return response.json().get("job", {}).get("id")

def wait_for_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    for _ in range(60):
        try:
            response = requests.get(url, headers=headers)
            result = response.json()
            status = result.get("job", {}).get("status")
            if status == "SUCCESS":
                return result["job"]["successInfo"]["images"][0]["url"]
            if status == "FAILED": return None
        except: pass
        time.sleep(5)
    return None

def main():
    today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    daily_output_dir = os.path.join(OUTPUT_ROOT, today_str)
    os.makedirs(daily_output_dir, exist_ok=True)
    styles = load_json(STYLE_BANK_FILE)
    database = load_json(DATABASE_FILE, default={"history": [], "stats": {}})
    
    batch = [
        {"path": "batch_2_photo_1.jpg", "name": "Maya_BD", "orig_rel": "Maya B.D/05 With Noam/Lightly processed/BLD_8253 - UNPROCESSED.jpg"},
        {"path": "batch_2_photo_2.jpg", "name": "Shahar_Zach", "orig_rel": "Shahar Zach/Onlyfans/Free/ULed already/WhatsApp Image 2021-01-16 at 21.23.17 (2).jpeg"}
    ]
    
    results_summary = []
    
    for item in batch:
        local_source = item["path"]
        model_name = item["name"]
        rel_path = item["orig_rel"]
        
        photo_base = os.path.splitext(os.path.basename(rel_path))[0][:30].strip()
        selected_styles = random.sample(styles, 4)
        resource_id, width, height = get_resource_id(local_source)
        if not resource_id: continue
        
        # Original copy
        with open(local_source, "rb") as f_src:
            with open(os.path.join(daily_output_dir, f"Original_{model_name}_{photo_base}.jpg"), "wb") as f_dst:
                f_dst.write(f_src.read())

        for style in selected_styles:
            job_id = create_job(resource_id, style['prompt_add'], width, height)
            if job_id:
                image_url = wait_for_job(job_id)
                if image_url:
                    clean_name = model_name.replace(" ", "_")
                    clean_style = style['name'].replace(' ', '_')
                    filename = f"{clean_name}_{photo_base}_{clean_style}.jpg"
                    local_file = os.path.join(daily_output_dir, filename)
                    
                    img_resp = requests.get(image_url)
                    img = Image.open(BytesIO(img_resp.content))
                    img.save(local_file, "JPEG", quality=95)
                    
                    gdrive_path = f"gdrive:_photos from openclaw/outputs/daily_game/public/{today_str}/"
                    os.system(f"rclone copy '{local_file}' '{gdrive_path}'")
                    link_proc = os.popen(f"rclone link '{gdrive_path}{filename}'")
                    public_link = link_proc.read().strip()
                    
                    variant_info = {
                        "date": today_str,
                        "photo": rel_path,
                        "style_id": style['id'],
                        "style_name": style['name'],
                        "file": local_file,
                        "public_link": public_link
                    }
                    database["history"].append(variant_info)
                    results_summary.append(variant_info)
                    print(f"Generated: {filename}")
                    
    save_json(DATABASE_FILE, database)
    print(f"DONE: {today_str}")

if __name__ == "__main__":
    main()
