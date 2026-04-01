import os
import requests
import json
import base64
import time

FAL_API_KEY = os.getenv("FAL_API_KEY")
STYLE_PROMPT = "Artistic figure photography, nebula and stars, deep space nebula, cinematic lighting, masterpiece, high quality, ethereal glow, dark background, detailed composition."
IDENTITY_PROMPT = "subject: preserve exact facial features, skin tone, eyes, nose, chin, and hair. Avoid over-smoothing skin, keep realistic textures."
OUTPUT_DIR = "outputs/omry_ksenia_instantid/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IDs for Omry and Ksenia from previous sessions
# Omry: 1JSjd4WgSlMguVjJKX9xrqqMhOqwUP7ub (was directory ID, need specific file IDs)
# Actually, I'll use the local copies if I have them or download again.
# Based on previous turns, Omry was BLD_9729, Ksenia/Raaia was BLD_6662.

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_instantid(photo_path, gender_prompt):
    try:
        url = "https://fal.run/fal-ai/instantid"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        img_b64 = encode_image(photo_path)
        # prompt refinement: explicitly state gender
        full_prompt = f"Professional photography of a {gender_prompt}, {STYLE_PROMPT}. {IDENTITY_PROMPT}"
        data = {
            "face_image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": full_prompt,
            "strength": 0.7, # Lowered strength from 0.8 to reduce "photoshopped" look
            "num_inference_steps": 35, # Increased steps for better texture
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
    # Ensuring we have the source files. 
    # From previous successful runs, these were the filenames used.
    targets = [
        {"name": "Omry", "file": "sfw_ref_1.jpg", "gender": "man"}, # Assuming Omry is man based on previous output "2 males"
        {"name": "Ksenia", "file": "sfw_ref_3.jpg", "gender": "woman"} # Assuming Ksenia/Raaia is woman
    ]
    
    for target in targets:
        print(f"Running InstantID for {target['name']}...")
        url = run_fal_instantid(target['file'], target['gender'])
        if url:
            img_data = requests.get(url).content
            filename = f"InstantID_{target['name']}_Realistic.png"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {filename}")
        else:
            print(f"  Failed: {target['name']}")

if __name__ == "__main__":
    main()
