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
PHOTO_PATH = "Raaia_portrait.jpg"
STYLE_PROMPT = "Artistic figure photography, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background."
IDENTITY_PROMPT = "subject: use the provided photo as reference. Preserve exact facial features, proportions, skin tone, eyes, nose, chin, hair and expression — identity must be as close as possible to the reference photo."
FULL_PROMPT = f"{STYLE_PROMPT} {IDENTITY_PROMPT}"
STRENGTH = 0.55
OUTPUT_DIR = "outputs/raaia_portrait_fix/"
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
    try:
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
        resp = requests.post(url, headers=headers, json=payload).json()
        job_id = resp["job"]["id"]
        while True:
            res = requests.get(f"{url}/{job_id}", headers=headers).json()
            if res["job"]["status"] == "SUCCESS":
                return res["job"]["successInfo"]["images"][0]["url"]
            elif res["job"]["status"] == "FAILED": return None
            time.sleep(5)
    except Exception as e:
        print(f"Tensor error: {e}")
        return None

def run_fal_flux_img2img():
    try:
        url = "https://fal.run/fal-ai/flux/dev/image-to-image"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(PHOTO_PATH)
        data = {
            "image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": f"Professional photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
            "strength": STRENGTH,
            "num_inference_steps": 28,
            "enable_safety_checker": False  # To avoid black images due to false NSFW
        }
        res = requests.post(url, headers=headers, json=data).json()
        if "images" in res:
            return res["images"][0]["url"]
        else:
            print(f"Flux Img2Img unexpected response: {res}")
            return None
    except Exception as e:
        print(f"Flux Img2Img error: {e}")
        return None

def run_fal_instantid():
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(PHOTO_PATH)
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": f"Professional photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
            "strength": 0.8,
            "num_inference_steps": 30,
            "enable_safety_checker": False
        }
        res = requests.post(url, headers=headers, json=data).json()
        if "images" in res:
            return res["images"][0]["url"]
        else:
            print(f"InstantID unexpected response: {res}")
            return None
    except Exception as e:
        print(f"InstantID error: {e}")
        return None

def run_fal_face_swap(base_image_url):
    try:
        url = "https://fal.run/fal-ai/face-swap"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(PHOTO_PATH)
        data = {
            "base_image_url": base_image_url,
            "swap_image_url": f"data:image/jpeg;base64,{img_b64}"
        }
        res = requests.post(url, headers=headers, json=data).json()
        if "image" in res:
            return res["image"]["url"]
        elif "images" in res:
            return res["images"][0]["url"]
        else:
            print(f"Face Swap unexpected response: {res}")
            return None
    except Exception as e:
        print(f"Face Swap error: {e}")
        return None

def main():
    methods = {
        "Method_1_Tensor_Prompt": run_tensor_prompt,
        "Method_2_Fal_Flux_Img2Img": run_fal_flux_img2img,
        "Method_3_Fal_InstantID": run_fal_instantid
    }
    
    for name, func in methods.items():
        print(f"Running {name}...")
        url = func()
        if url:
            img_data = requests.get(url).content
            filename = f"Raaia_portrait_{name}.jpg"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {filename}")
            
            if "Fal_Flux" in name:
                print(f"  Adding Face Swap to {name}...")
                swap_url = run_fal_face_swap(url)
                if swap_url:
                    swap_data = requests.get(swap_url).content
                    swap_filename = f"Raaia_portrait_Method_4_Fal_FaceSwap_on_Flux.jpg"
                    with open(os.path.join(OUTPUT_DIR, swap_filename), "wb") as f:
                        f.write(swap_data)
                    print(f"  Saved: {swap_filename}")

if __name__ == "__main__":
    main()
