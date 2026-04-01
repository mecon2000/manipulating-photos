import os
import requests
import json
import base64
import argparse
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageEnhance

FAL_API_KEY = os.environ.get("FAL_API_KEY")

def run_fal_api(endpoint, payload):
    url = f"https://fal.run/{endpoint}"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200: return response.json()
    print(f"Error {response.status_code}: {response.text}")
    return None

def color_grade_pro(img, style):
    if "Indigo" in style:
        # Complex Indigo Dyeing Effect
        r, g, b = img.split()
        r = r.point(lambda i: i * 0.7) # Mute reds
        g = g.point(lambda i: i * 0.9)
        b = b.point(lambda i: min(255, i * 1.3)) # Boost blues
        img = Image.merge("RGB", (r, g, b))
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Color(img).enhance(0.6) # Desaturate skin for art look
    elif "Oil_Paint" in style:
        # Oil Paint look: high contrast, warmer tones
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Sharpness(img).enhance(0.5) # Softer edges
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--style", default="Indigo_Dye_Aesthetic")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_orig = Image.open(args.source).convert("RGB")
    orig_size = img_orig.size
    
    # 1. Get Mask
    print(f"--- Step 1: Getting Mask for {args.source} ---")
    with open(args.source, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    res_rembg = run_fal_api("fal-ai/rembg", {"image_url": f"data:image/jpeg;base64,{img_b64}"})
    if not res_rembg: return
    mask_url = res_rembg["image"]["url"]
    mask_img = Image.open(requests.get(mask_url, stream=True).raw).split()[3]
    mask_path = os.path.join(args.output_dir, "pro_mask.png")
    mask_img.save(mask_path)

    # 2. Inpaint Background
    print("--- Step 2: Inpainting Background (Perfect Blend) ---")
    with open(mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    res_inpaint = run_fal_api("fal-ai/fast-sdxl/inpainting", {
        "prompt": f"An abstract fine art {args.style} background, masterfully painted, oil paint impasto, moody atmosphere, NO PEOPLE.",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "mask_url": f"data:image/png;base64,{mask_b64}",
        "strength": 0.85, # Keep some context of the room
        "num_inference_steps": 40,
        "enable_safety_checker": False
    })
    
    if res_inpaint and "images" in res_inpaint:
        bg_path = os.path.join(args.output_dir, "gen_bg_inpaint.jpg")
        with open(bg_path, "wb") as f: f.write(requests.get(res_inpaint["images"][0]["url"]).content)
        img_bg = Image.open(bg_path).convert("RGB").resize(orig_size, Image.LANCZOS)
    else: return

    # 3. Process Model (Pro Color Grade)
    print("--- Step 3: Pro Color Grading for Model ---")
    img_model = color_grade_pro(img_orig, args.style)
    
    # 4. Final Composite
    print("--- Step 4: Final Compositing (Clean Edges) ---")
    soft_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=1)) # Very sharp edges
    final = Image.composite(img_model, img_bg, soft_mask)
    final_path = os.path.join(args.output_dir, f"Final_Art_Result_{args.style}.jpg")
    final.save(final_path, "JPEG", quality=95)
    print(f"Success! Saved: {final_path}")

if __name__ == '__main__':
    main()
