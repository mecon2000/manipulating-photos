import os
import requests
import json
import base64
import time

FAL_API_KEY = os.getenv("FAL_API_KEY")
STYLE_PROMPT = "Artistic figure photography of a Caucasian woman, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition."
IDENTITY_PROMPT = "subject: Caucasian woman, preserve exact facial features, skin tone, eyes, nose, chin, and hair. Realistic skin texture, do not over-smooth."
OUTPUT_DIR = "outputs/style_game_batch_v2_debug/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path, strength):
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(photo_path)
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": f"Professional photography, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
            "strength": strength,
            "num_inference_steps": 30,
            "enable_safety_checker": False
        }
        res = requests.post(url, headers=headers, json=data)
        if res.status_code != 200:
            print(f"API Error: {res.status_code} - {res.text}")
            return None
        res_json = res.json()
        if "images" in res_json:
            return res_json["images"][0]["url"]
        elif "image" in res_json:
            return res_json["image"]["url"]
        print(f"Unexpected response: {res_json}")
        return None
    except Exception as e:
        print(f"Script Error: {e}")
        return None

if __name__ == "__main__":
    photo = "game_candidate_3.jpg"
    strength = 0.85
    url = run_fal_instantid(photo, strength)
    if url:
        print(f"Success: {url}")
    else:
        print("Failed")
