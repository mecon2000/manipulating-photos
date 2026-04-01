import os
import requests
import json
import base64
import time

FAL_API_KEY = os.getenv("FAL_API_KEY")
# Added "Caucasian woman" to prompt to prevent ethnicity hallucination
STYLE_PROMPT = "Artistic figure photography of a Caucasian woman, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition."
IDENTITY_PROMPT = "subject: Caucasian woman, preserve exact facial features, skin tone, eyes, nose, chin, and hair. Realistic skin texture, do not over-smooth."
OUTPUT_DIR = "outputs/style_game_batch_v2/"
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
        res = requests.post(url, headers=headers, json=data).json()
        if "images" in res:
            return res["images"][0]["url"]
        elif "image" in res:
            return res["image"]["url"]
        return None
    except Exception as e:
        return None

def main():
    photo = "game_candidate_2.jpg"
    strengths = [0.8, 0.85, 0.9] # Higher strengths to ensure identity
    
    # Save original
    with open(photo, "rb") as f_src:
        with open(os.path.join(OUTPUT_DIR, "Omry_Ksenia_BLD_0047_Original.jpg"), "wb") as f_dst:
            f_dst.write(f_src.read())

    for i, strength in enumerate(strengths, 1):
        print(f"Running variation {i} with strength {strength}...")
        url = run_fal_instantid(photo, strength)
        if url:
            img_data = requests.get(url).content
            filename = f"Omry_Ksenia_BLD_0047_Strength_{strength}.png"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {filename}")
        else:
            print(f"  Failed: Variation {i}")

if __name__ == "__main__":
    main()
