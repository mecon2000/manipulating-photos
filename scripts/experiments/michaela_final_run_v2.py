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
    img_orig = Image.open(SOURCE_IMAGE).convert("RGB")
    orig_size = img_orig.size
    mask_orig = Image.open(MASK_IMAGE).convert("L")
    
    bg_gen_path = os.path.join(OUTPUT_DIR, "generated_bg.jpg")
    model_gen_path = os.path.join(OUTPUT_DIR, "generated_model.jpg")

    # 1. Resize and Composite
    print("Compositing with resizing...")
    gen_bg = Image.open(bg_gen_path).convert("RGB").resize(orig_size, Image.LANCZOS)
    gen_model = Image.open(model_gen_path).convert("RGB").resize(orig_size, Image.LANCZOS)
    
    # Soften mask slightly for the final blend
    soft_mask = mask_orig.filter(ImageFilter.GaussianBlur(radius=5))
    final = Image.composite(gen_model, gen_bg, soft_mask)
    final_path = os.path.join(OUTPUT_DIR, "Michaela_Final_Composite_V2.jpg")
    final.save(final_path, "JPEG", quality=95)
    print(f"  Done: {final_path}")

if __name__ == '__main__':
    main()
