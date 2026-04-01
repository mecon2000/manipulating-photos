import os
import requests
import json
import base64

FAL_API_KEY = os.getenv("FAL_API_KEY")
STYLE_PROMPT = "Artistic figure photography, black ink wash effect, black ink spreading through water, merging with the silhouette, masterpiece, high quality, fine art."
IDENTITY_PROMPT = "subject: preserve exact facial features, skin tone, hair and expression. Realistic skin texture, no over-smoothing."
OUTPUT_DIR = "outputs/shahar_zach_redo/"
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
    photo = "batch_2_photo_2.jpg"
    strength = 0.8
    
    # Save original
    with open(photo, "rb") as f_src:
        with open(os.path.join(OUTPUT_DIR, "Original_Shahar_Zach.jpg"), "wb") as f_dst:
            f_dst.write(f_src.read())

    print(f"Running InstantID for Shahar Zach...")
    url = run_fal_instantid(photo, strength)
    if url:
        img_data = requests.get(url).content
        filename = "Shahar_Zach_Black_Ink_InstantID.png"
        local_path = os.path.join(OUTPUT_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(img_data)
        print(f"  Saved: {filename}")
    else:
        print(f"  Failed")

if __name__ == "__main__":
    main()
