import os
import requests
import json
import base64
import time

FAL_API_KEY = os.getenv("FAL_API_KEY")
IDENTITY_PROMPT = "subject: preserve exact facial features, skin tone, eyes, nose, chin, and hair. Avoid over-smoothing skin, keep realistic textures."
OUTPUT_DIR = "outputs/style_game_batch/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path, style_prompt):
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(photo_path)
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": f"Professional photography, {style_prompt}. {IDENTITY_PROMPT}",
            "strength": 0.7,
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
    photo = "game_ref_original.jpg"
    styles = [
        "Artistic figure photography, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition.",
        "Artistic figure photography, golden and crimson nebula, cosmic dust, cinematic lighting, masterpiece, ethereal atmosphere, dark background.",
        "Artistic figure photography, blue and violet galactic nebula, stardust, glowing orbits, cinematic lighting, high quality, ethereal glow."
    ]
    
    # Save original
    with open(photo, "rb") as f_src:
        with open(os.path.join(OUTPUT_DIR, "Omry_Ksenia_BLD_0106_Original.jpg"), "wb") as f_dst:
            f_dst.write(f_src.read())

    for i, style in enumerate(styles, 1):
        print(f"Running variation {i}...")
        url = run_fal_instantid(photo, style)
        if url:
            img_data = requests.get(url).content
            filename = f"Omry_Ksenia_BLD_0106_Nebula_V{i}.png"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {filename}")
        else:
            print(f"  Failed: Variation {i}")

if __name__ == "__main__":
    main()
