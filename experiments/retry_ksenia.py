import os
import requests
import json
import base64

FAL_API_KEY = os.getenv("FAL_API_KEY")
STYLE_PROMPT = "Artistic figure photography, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition."
IDENTITY_PROMPT = "subject: woman, preserve exact facial features, skin tone, eyes, nose, chin, and hair. Realistic skin textures."
OUTPUT_DIR = "outputs/omry_ksenia_actual_instantid/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path):
    url = "https://fal.run/fal-ai/instantid"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    img_b64 = encode_image(photo_path)
    data = {
        "face_image_url": f"data:image/jpeg;base64,{img_b64}",
        "prompt": f"Professional photography of a woman, {STYLE_PROMPT}. {IDENTITY_PROMPT}",
        "strength": 0.75,
        "num_inference_steps": 30,
        "enable_safety_checker": False
    }
    res = requests.post(url, headers=headers, json=data)
    print(f"Status: {res.status_code}")
    print(f"Body: {res.text}")
    res_json = res.json()
    if "images" in res_json:
        return res_json["images"][0]["url"]
    return None

if __name__ == "__main__":
    url = run_fal_instantid("ref_ksenia.jpg")
    if url:
        img_data = requests.get(url).content
        with open(os.path.join(OUTPUT_DIR, "InstantID_Ksenia_Real.png"), "wb") as f:
            f.write(img_data)
        print("Success")
