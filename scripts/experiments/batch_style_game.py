import os
import random
import json
import requests
import uuid
import time
from PIL import Image
from io import BytesIO
from datetime import datetime

# Configuration
PHOTO_ROOT = "catalog/preview/"
OUTPUT_ROOT = "outputs/daily_game/"
DATABASE_FILE = "state/style_game_database.json"
STYLE_BANK_FILE = "state/style_bank.json"
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
STRENGTH = 0.6 
MODEL_ID = "965126062386242266" # Uncensored

def load_json(file_path, default=None):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default or []

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_all_photos():
    photos = []
    if not os.path.exists(PHOTO_ROOT): return []
    for root, dirs, files in os.walk(PHOTO_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                photos.append(os.path.join(root, file))
    return photos

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
        img.save(buf, format='PNG')
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
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "image_resource_id": resource_id,
                    "count": 1
                }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": width,
                    "height": height,
                    "prompts": [{"text": f"{content_desc} Style: {style_prompt}."}],
                    "sdModel": MODEL_ID,
                    "steps": 30,
                    "cfgScale": 7,
                    "denoisingStrength": STRENGTH
                }
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200: return None
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
    all_photos = get_all_photos()
    if not all_photos or not styles: return
    
    # 4 photos as requested
    selected_photos = random.sample(all_photos, min(4, len(all_photos)))
    results_summary = []
    
    for photo_path in selected_photos:
        photo_name = os.path.basename(photo_path)
        # 4 styles per photo to get 16 images total
        selected_styles = random.sample(styles, 4)
        resource_id, width, height = get_resource_id(photo_path)
        if not resource_id: continue
        
        for style in selected_styles:
            job_id = create_job(resource_id, style['prompt_add'], width, height)
            if job_id:
                image_url = wait_for_job(job_id)
                if image_url:
                    filename = f"{os.path.splitext(photo_name)[0]}_{style['name'].replace(' ', '_')}.png"
                    local_file = os.path.join(daily_output_dir, filename)
                    img_resp = requests.get(image_url)
                    with open(local_file, 'wb') as f:
                        f.write(img_resp.content)
                    
                    gdrive_path = f"gdrive:_photos from openclaw/outputs/daily_game/public/{today_str}/"
                    os.system(f"rclone copy '{local_file}' '{gdrive_path}'")
                    link_proc = os.popen(f"rclone link '{gdrive_path}{filename}'")
                    public_link = link_proc.read().strip()
                    
                    variant_info = {
                        "date": today_str,
                        "photo": photo_path,
                        "style_id": style['id'],
                        "style_name": style['name'],
                        "file": local_file,
                        "public_link": public_link
                    }
                    database["history"].append(variant_info)
                    results_summary.append(variant_info)
                    print(f"Generated: {photo_name} - {style['name']}")
                    
    save_json(DATABASE_FILE, database)
    # Output the final JSON for the assistant to process
    print("FINAL_RESULTS_JSON:")
    print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    main()
