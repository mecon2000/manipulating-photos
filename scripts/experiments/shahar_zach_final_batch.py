import os
import requests
import uuid
import time
import base64
import random
from PIL import Image
from io import BytesIO

# Config
FAL_API_KEY = os.getenv("FAL_API_KEY")
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
SD_MODEL_ID = "965126062386242266"

PHOTO_PATH = "shahar_zach_true_original.jpg"
STYLE_PROMPT = "black ink wash effect, black ink spreading through water, merging with the silhouette, masterpiece, high quality, fine art"
IDENTITY_PROMPT = "subject: preserve exact facial features, skin tone, hair and expression. Realistic skin texture, no over-smoothing."
OUTPUT_DIR = "outputs/shahar_zach_true_batch/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path, i):
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(photo_path)
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": f"Professional photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
            "strength": 0.8,
            "seed": random.randint(1, 1000000),
            "num_inference_steps": 30,
            "enable_safety_checker": False
        }
        res = requests.post(url, headers=headers, json=data).json()
        img_url = res.get("images", [{}])[0].get("url") or res.get("image", {}).get("url")
        if img_url:
            content = requests.get(img_url).content
            filename = f"Shahar_Zach_True_InstantID_V{i}.png"
            with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                f.write(content)
            return True
    except: pass
    return False

def get_tensor_resource(local_path):
    try:
        img = Image.open(local_path).convert('RGB')
        w, h = img.size
        new_w, new_h = (1024, int((h/w)*1024)) if w > h else (int((w/h)*1024), 1024)
        new_w, new_h = (new_w // 8) * 8, (new_h // 8) * 8
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=95)
        headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
        res = requests.post(f"{TENSOR_BASE_URL}/resource/image", headers=headers, json={}).json()
        requests.put(res["putUrl"], data=buf.getvalue(), headers=res["headers"])
        return res["resourceId"], new_w, new_h
    except: return None, None, None

def run_tensor_img2img(resource_id, w, h, i):
    try:
        url = f"{TENSOR_BASE_URL}/jobs"
        headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "requestId": str(uuid.uuid4()),
            "stages": [
                {"type": "INPUT_INITIALIZE", "inputInitialize": {"image_resource_id": resource_id, "count": 1}},
                {"type": "DIFFUSION", "diffusion": {
                    "width": w, "height": h,
                    "prompts": [{"text": f"artistic figure photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}"}],
                    "sdModel": SD_MODEL_ID, "steps": 35, "cfgScale": 8, "denoisingStrength": 0.4
                }}
            ]
        }
        job_id = requests.post(url, headers=headers, json=payload).json()["job"]["id"]
        for _ in range(60):
            status_res = requests.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers={"Authorization": f"Bearer {TENSOR_API_KEY}"}).json()
            if status_res["job"]["status"] == "SUCCESS":
                img_url = status_res["job"]["successInfo"]["images"][0]["url"]
                content = requests.get(img_url).content
                filename = f"Shahar_Zach_True_Tensor_V{i}.jpg"
                with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                    f.write(content)
                return True
            time.sleep(5)
    except: pass
    return False

def main():
    # Save original
    with open(PHOTO_PATH, "rb") as f_src:
        with open(os.path.join(OUTPUT_DIR, "Original_Shahar_Zach_True.jpg"), "wb") as f_dst:
            f_dst.write(f_src.read())
            
    print("Generating True InstantID variants...")
    run_fal_instantid(PHOTO_PATH, 1)
    run_fal_instantid(PHOTO_PATH, 2)
    
    print("Generating True Tensor variants...")
    res_id, w, h = get_tensor_resource(PHOTO_PATH)
    if res_id:
        run_tensor_img2img(res_id, w, h, 1)
        run_tensor_img2img(res_id, w, h, 2)

if __name__ == "__main__":
    main()
