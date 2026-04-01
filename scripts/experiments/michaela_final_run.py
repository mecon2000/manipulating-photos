import os
import requests
import json
import base64
from PIL import Image, ImageFilter, ImageOps

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Processed/BLD_1654E.jpg"
MASK_IMAGE = "outputs/michaela_perfect_workflow/michaela_perfect_mask.png"
OUTPUT_DIR = "outputs/michaela_final_workflow/"
PROMPT = "A cinematic fine art portrait in a moody, atmospheric studio with ethereal lighting, oil paint impasto texture, dark and artistic aesthetic."

def run_fal_img2img(image_path, prompt, strength):
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "prompt": prompt,
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "strength": strength,
        "num_inference_steps": 30,
        "enable_safety_checker": False
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["images"][0]["url"]
    return None

def main():
    img = Image.open(SOURCE_IMAGE).convert("RGB")
    mask = Image.open(MASK_IMAGE).convert("L")
    
    # 1. Prep BG Only (Blur model area)
    img_bg_prep = img.copy()
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=80))
    img_bg_prep.paste(img_blurred, mask=mask)
    bg_prep_path = os.path.join(OUTPUT_DIR, "input_bg_prep.jpg")
    img_bg_prep.save(bg_prep_path, "JPEG", quality=95)
    
    # 2. Prep Model Only (Black BG)
    img_model_prep = Image.new("RGB", img.size, (0, 0, 0))
    img_model_prep.paste(img, mask=mask)
    model_prep_path = os.path.join(OUTPUT_DIR, "input_model_prep.jpg")
    img_model_prep.save(model_prep_path, "JPEG", quality=95)
    
    # 3. Run Tensor (Fal) for BG
    print("Generating Background (High Strength)...")
    bg_url = run_fal_img2img(bg_prep_path, PROMPT, 0.95)
    if bg_url:
        bg_data = requests.get(bg_url).content
        bg_gen_path = os.path.join(OUTPUT_DIR, "generated_bg.jpg")
        with open(bg_gen_path, "wb") as f:
            f.write(bg_data)
    else:
        print("  BG Generation Failed")
        return

    # 4. Run Tensor (Fal) for Model
    print("Generating Model (Low Strength)...")
    model_url = run_fal_img2img(model_prep_path, PROMPT, 0.40)
    if model_url:
        model_data = requests.get(model_url).content
        model_gen_path = os.path.join(OUTPUT_DIR, "generated_model.jpg")
        with open(model_gen_path, "wb") as f:
            f.write(model_data)
    else:
        print("  Model Generation Failed")
        return

    # 5. Composite
    print("Compositing...")
    gen_bg = Image.open(bg_gen_path).convert("RGB")
    gen_model = Image.open(model_gen_path).convert("RGB")
    # Soften mask slightly for the final blend
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    final = Image.composite(gen_model, gen_bg, soft_mask)
    final_path = os.path.join(OUTPUT_DIR, "Michaela_Final_Composite.jpg")
    final.save(final_path, "JPEG", quality=95)
    print(f"  Done: {final_path}")

if __name__ == '__main__':
    main()
