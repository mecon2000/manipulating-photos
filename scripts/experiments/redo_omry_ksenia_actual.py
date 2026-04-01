import os
import requests
import json
import base64
import time

FAL_API_KEY = os.getenv("FAL_API_KEY")
STYLE_PROMPT = "Artistic figure photography, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition."
IDENTITY_PROMPT = "subject: preserve exact facial features, skin tone, eyes, nose, chin, and hair. Avoid over-smoothing skin, keep realistic textures."
OUTPUT_DIR = "outputs/omry_ksenia_actual_instantid/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path, gender_prompt):
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(photo_path)
        full_prompt = f"Professional photography of a {gender_prompt}, {STYLE_PROMPT}. {IDENTITY_PROMPT}"
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": full_prompt,
            "strength": 0.7,
            "num_inference_steps": 35,
            "enable_safety_checker": False
        }
        res = requests.post(url, headers=headers, json=data).json()
        if "images" in res:
            return res["images"][0]["url"]
        elif "image" in res:
            return res["image"]["url"]
        return None
    except Exception as e:
        return None

def main():
    targets = [
        {"name": "Omry", "file": "ref_omry.jpg", "gender": "man"},
        {"name": "Ksenia", "file": "ref_ksenia.jpg", "gender": "woman"}
    ]
    for target in targets:
        print(f"Running InstantID for {target['name']}...")
        url = run_fal_instantid(target['file'], target['gender'])
        if url:
            img_data = requests.get(url).content
            filename = f"InstantID_{target['name']}_Real.png"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {filename}")
        else:
            print(f"  Failed: {target['name']}")

if __name__ == "__main__":
    main()
