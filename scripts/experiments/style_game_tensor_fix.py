import os
import requests
import uuid
import time
from PIL import Image
from io import BytesIO

# Config
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
STRENGTH = 0.55 
MODEL_ID = "965126062386242266" # Uncensored SD 1.5
PHOTO_PATH = "game_candidate_3.jpg"
OUTPUT_DIR = "outputs/style_game_tensor_fix/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        print(f"Error: {e}")
        return None, None, None

def create_job(resource_id, style_prompt, width, height):
    url = f"{BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    identity_prompt = "subject: use the provided photo as reference. Preserve exact facial features, skin tone, and hair. Keep realistic skin textures."
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
    if response.status_code != 200: return None
    return response.json().get("job", {}).get("id")

def wait_for_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    for _ in range(60):
        try:
            res = requests.get(url, headers=headers).json()
            status = res.get("job", {}).get("status")
            if status == "SUCCESS": return res["job"]["successInfo"]["images"][0]["url"]
            if status == "FAILED": return None
        except: pass
        time.sleep(5)
    return None

def main():
    styles = [
        "nebula and stars, deep space nebula, cinematic lighting",
        "golden and crimson nebula, cosmic dust, cinematic lighting",
        "blue and violet galactic nebula, stardust, glowing orbits"
    ]
    resource_id, w, h = get_resource_id(PHOTO_PATH)
    if not resource_id: return
    
    # Save original
    with open(PHOTO_PATH, "rb") as f_src:
        with open(os.path.join(OUTPUT_DIR, "Original_BLD_9490E.jpg"), "wb") as f_dst:
            f_dst.write(f_src.read())

    for i, style in enumerate(styles, 1):
        print(f"Variation {i}...")
        job_id = create_job(resource_id, style, w, h)
        if job_id:
            url = wait_for_job(job_id)
            if url:
                img_data = requests.get(url).content
                with open(os.path.join(OUTPUT_DIR, f"Tensor_Nebula_V{i}.jpg"), "wb") as f:
                    f.write(img_data)
                print(f"  Saved V{i}")

if __name__ == "__main__":
    main()
