import os
import requests
import json

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "catalog/thumbs/Michaela__Processed/BLD_1654E.jpg"
PROMPT = "A cinematic fine art portrait of a woman in red lingerie, back view, standing in a moody, atmospheric studio with ethereal lighting, oil paint impasto texture, dark and artistic aesthetic."
OUTPUT_DIR = "outputs/michaela_dual_strength/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_fal_img2img(image_path, prompt, strength):
    try:
        url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
            "Content-Type": "application/json"
        }
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "prompt": prompt,
            "image_url": f"data:image/jpeg;base64,{img_b64}",
            "strength": strength,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "enable_safety_checker": False
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        if "images" in res:
            return res["images"][0]["url"]
        return None
    except Exception as e:
        print(f"Error at strength {strength}: {e}")
        return None

def main():
    if not FAL_API_KEY:
        print("Missing FAL_API_KEY")
        return

    for strength in [0.85, 0.45]:
        print(f"Processing strength: {strength}")
        result_url = run_fal_img2img(SOURCE_IMAGE, PROMPT, strength)
        if result_url:
            img_data = requests.get(result_url).content
            filename = f"Michaela_Strength_{strength}.jpg"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {local_path}")
        else:
            print(f"  Failed for strength {strength}")

if __name__ == '__main__':
    main()
