import os
import random
import json
import requests
import uuid
import time
import re
import subprocess
from PIL import Image
from io import BytesIO
from datetime import datetime

# Configuration
GDRIVE_INDEX = "state/gdrive_photos_index.json"
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

def get_photos_from_index():
    if not os.path.exists(GDRIVE_INDEX):
        return []
    with open(GDRIVE_INDEX, 'r') as f:
        return json.load(f)

def parse_model_name(rel_path):
    # gdrive:_Photos/MODEL_NAME/...
    parts = rel_path.split('/')
    if len(parts) > 0:
        clean = re.sub(r'^(Adi_Levi__|\d+_)', '', parts[0])
        clean = re.split(r'__| |,|\(', clean)[0] # Split at __ or space or , or (
        return clean.capitalize()
    return "Unknown"

def fetch_from_gdrive(rel_path):
    local_filename = os.path.basename(rel_path)
    local_path = os.path.join("/tmp", local_filename)
    gdrive_path = f"gdrive:_Photos/{rel_path}"
    print(f"Fetching: {gdrive_path}")
    subprocess.run(['rclone', 'copy', gdrive_path, '/tmp/'])
    return local_path

def get_resource_id(local_path):
    try:
        img = Image.open(local_path).convert('RGB')
        w, h = img.size
        # Scaled for SD 1.5 performance but maintaining aspect ratio
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
    identity_prompt = "subject: use the provided photo as reference. Preserve exact facial features, proportions, skin tone, eyes, nose, chin, hair and expression — identity must be as close as possible to the reference photo."
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
    all_photos = get_photos_from_index()
    if not all_photos or not styles: return
    
    # 2 source photos for 8 total variations
    selected_rel_paths = random.sample(all_photos, min(2, len(all_photos)))
    results_summary = []
    
    for rel_path in selected_rel_paths:
        model_name = parse_model_name(rel_path)
        local_source = fetch_from_gdrive(rel_path)
        if not os.path.exists(local_source): continue
        
        photo_base = os.path.splitext(os.path.basename(rel_path))[0][:30].strip()
        selected_styles = random.sample(styles, 4)
        resource_id, width, height = get_resource_id(local_source)
        if not resource_id: continue
        
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
        
        # Cleanup /tmp
        if os.path.exists(local_source): os.remove(local_source)
                    
    save_json(DATABASE_FILE, database)
    print(f"DONE: {today_str}")

if __name__ == "__main__":
    main()
