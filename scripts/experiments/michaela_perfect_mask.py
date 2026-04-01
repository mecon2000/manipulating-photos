import os
import requests
import json
from PIL import Image, ImageOps

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Processed/BLD_1654E.jpg"
OUTPUT_DIR = "outputs/michaela_perfect_workflow/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_fal_rembg(image_path):
    url = "https://fal.run/fal-ai/rembg"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    import base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {"image_url": f"data:image/jpeg;base64,{img_b64}"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["image"]["url"]
    return None

def main():
    print("Running Perfect Background Removal...")
    rembg_url = run_fal_rembg(SOURCE_IMAGE)
    if not rembg_url:
        print("  Failed")
        return
        
    img_no_bg_data = requests.get(rembg_url).content
    no_bg_path = os.path.join(OUTPUT_DIR, "michaela_no_bg.png")
    with open(no_bg_path, "wb") as f:
        f.write(img_no_bg_data)
    
    # Create mask from Alpha channel
    img_no_bg = Image.open(no_bg_path)
    alpha = img_no_bg.split()[3]
    mask_path = os.path.join(OUTPUT_DIR, "michaela_perfect_mask.png")
    alpha.save(mask_path)
    
    print(f"  Saved perfect mask to: {mask_path}")

if __name__ == '__main__':
    main()
