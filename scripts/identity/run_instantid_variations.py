import os
import requests
import json
import base64
import time

FAL_API_KEY = os.getenv("FAL_API_KEY")
IDENTITY_PROMPT = "subject: preserve exact facial features, skin tone, eyes, nose, chin, and hair."
OUTPUT_DIR = "outputs/sfw_instantid_variations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path, style):
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(photo_path)
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": f"Professional photography, {style}. {IDENTITY_PROMPT}",
            "strength": 0.8,
            "num_inference_steps": 30,
            "enable_safety_checker": False
        }
        res = requests.post(url, headers=headers, json=data).json()
        if "images" in res:
            return res["images"][0]["url"]
        elif "image" in res:
            return res["image"]["url"]
        else:
            return None
    except Exception as e:
        return None

def main():
    photos = ["sfw_ref_1.jpg", "sfw_ref_2.jpg", "sfw_ref_3.jpg"]
    styles = [
        "Artistic figure photography, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition.",
        "Artistic figure photography, nebula and stars, deep space nebula, soft ethereal glow, cinematic lighting, dark background, detailed composition.",
        "Artistic figure photography, deep space nebula, glowing stars, cinematic lighting, dark background, detailed composition."
    ]
    for i, photo in enumerate(photos, 1):
        for j, style in enumerate(styles, 1):
            print(f"Running style {j} for {photo}...")
            url = run_fal_instantid(photo, style)
            if url:
                img_data = requests.get(url).content
                filename = f"InstantID_Result_{i}_Style_{j}.png"
                local_path = os.path.join(OUTPUT_DIR, filename)
                with open(local_path, "wb") as f:
                    f.write(img_data)
                print(f"  Saved: {filename}")
            else:
                print(f"  Failed: {photo} Style {j}")

if __name__ == "__main__":
    main()
