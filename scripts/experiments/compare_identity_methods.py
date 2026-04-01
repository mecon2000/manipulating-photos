import os
import requests
import json
import uuid
import time
import base64
from PIL import Image
from io import BytesIO

# Config
TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/Elly__BLD_5440E.jpg"
STYLE_PROMPT = "heavy swirling smoke, partially obscured subject, multiple mirror reflections"
IDENTITY_PROMPT = "subject: use the provided photo as reference. Preserve exact facial features, proportions, skin tone, eyes, nose, chin, hair and expression — identity must be as close as possible to the reference photo."
FULL_PROMPT = f"artistic figure photography, high quality, masterpiece. {IDENTITY_PROMPT} Style: {STYLE_PROMPT}."
STRENGTH = 0.55
OUTPUT_DIR = "outputs/identity_comparison/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def upload_tensor(path):
    url = "https://ap-east-1.tensorart.cloud/v1/resource/image"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    img = Image.open(path).convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    response = requests.post(url, headers=headers, json={}).json()
    requests.put(response["putUrl"], data=buf.getvalue(), headers=response["headers"])
    return response["resourceId"]

def run_tensor_prompt():
    res_id = upload_tensor(PHOTO_PATH)
    url = "https://ap-east-1.tensorart.cloud/v1/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {"type": "INPUT_INITIALIZE", "inputInitialize": {"image_resource_id": res_id, "count": 1}},
            {"type": "DIFFUSION", "diffusion": {
                "width": 1024, "height": 1024,
                "prompts": [{"text": FULL_PROMPT}],
                "sdModel": "965126062386242266", "steps": 30, "cfgScale": 7, "denoisingStrength": STRENGTH
            }}
        ]
    }
    job_id = requests.post(url, headers=headers, json=payload).json()["job"]["id"]
    while True:
        res = requests.get(f"{url}/{job_id}", headers=headers).json()
        if res["job"]["status"] == "SUCCESS":
            return res["job"]["successInfo"]["images"][0]["url"]
        elif res["job"]["status"] == "FAILED": return None
        time.sleep(5)

def run_fal_flux_img2img():
    url = "https://fal.run/fal-ai/flux/dev/image-to-image"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    img_b64 = encode_image(PHOTO_PATH)
    data = {
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "prompt": f"Professional photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
        "strength": STRENGTH,
        "num_inference_steps": 28
    }
    res = requests.post(url, headers=headers, json=data).json()
    return res["images"][0]["url"]

def run_fal_instantid():
    url = "https://fal.run/fal-ai/instantid"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    img_b64 = encode_image(PHOTO_PATH)
    data = {
        "face_image_url": f"data:image/jpeg;base64,{img_b64}",
        "prompt": f"Professional photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
        "strength": 0.8,
        "num_inference_steps": 30
    }
    # Handling potential API structure differences for InstantID
    try:
        res = requests.post(url, headers=headers, json=data).json()
        return res["images"][0]["url"]
    except:
        return None

def run_fal_face_swap(base_image_url):
    url = "https://fal.run/fal-ai/face-swap"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    img_b64 = encode_image(PHOTO_PATH)
    data = {
        "base_image_url": base_image_url,
        "swap_image_url": f"data:image/jpeg;base64,{img_b64}"
    }
    res = requests.post(url, headers=headers, json=data).json()
    return res["image"]["url"]

def main():
    methods = {
        "Method_1_Tensor_Prompt": run_tensor_prompt,
        "Method_2_Fal_Flux_Img2Img": run_fal_flux_img2img,
        "Method_3_Fal_InstantID": run_fal_instantid
    }
    
    results = []
    for name, func in methods.items():
        for i in range(2):
            print(f"Running {name} #{i+1}...")
            url = func()
            if url:
                img_data = requests.get(url).content
                filename = f"{name}_{i+1}.jpg"
                local_path = os.path.join(OUTPUT_DIR, filename)
                with open(local_path, "wb") as f:
                    f.write(img_data)
                
                # For Method 2, also do a Face Swap version (Method 4)
                if "Fal_Flux" in name:
                    print(f"  Adding Face Swap to {name}...")
                    swap_url = run_fal_face_swap(url)
                    if swap_url:
                        swap_data = requests.get(swap_url).content
                        swap_filename = f"Method_4_Fal_FaceSwap_on_Flux_{i+1}.jpg"
                        with open(os.path.join(OUTPUT_DIR, swap_filename), "wb") as f:
                            f.write(swap_data)
                
                print(f"  Done: {filename}")
                
    # Upload everything to GDrive and get folder link
    gdrive_path = f"gdrive:_photos from openclaw/outputs/daily_game/public/identity_test_2026-03-28/"
    os.system(f"rclone copy '{OUTPUT_DIR}' '{gdrive_path}'")
    link_proc = os.popen(f"rclone link '{gdrive_path}'")
    folder_link = link_proc.read().strip()
    print(f"\nFinal Folder Link: {folder_link}")

if __name__ == "__main__":
    main()
