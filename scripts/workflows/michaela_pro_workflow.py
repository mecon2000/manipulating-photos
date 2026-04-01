import os
import requests
import json
import base64
from PIL import Image, ImageFilter, ImageOps, ImageChops

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Processed/BLD_1654E.jpg"
MASK_IMAGE = "outputs/michaela_perfect_workflow/michaela_perfect_mask.png"
OUTPUT_DIR = "outputs/michaela_pro_workflow/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG_PROMPT = "An abstract fine art oil painting impasto background, moody atmospheric studio, ethereal lighting, dark aesthetic, NO PEOPLE, NO FIGURES, NO HUMANS."
MODEL_PROMPT = "A fine art portrait of a woman in red lingerie, back view, oil paint impasto texture, high detail, realistic skin."

def run_fal_img2img(image_path, prompt, strength, model_endpoint="fal-ai/fast-sdxl/image-to-image"):
    url = f"https://fal.run/{model_endpoint}"
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
        "num_inference_steps": 35,
        "enable_safety_checker": False
    }
    # Flux has a different schema often, but let's try standard first
    if "flux" in model_endpoint:
        url = "https://fal.run/fal-ai/flux/dev/image-to-image"
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["images"][0]["url"]
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def main():
    img_orig = Image.open(SOURCE_IMAGE).convert("RGB")
    orig_size = img_orig.size
    mask = Image.open(MASK_IMAGE).convert("L")
    
    # 1. Prepare BG Only (Content-aware fill simulation: replace model with dark texture)
    print("Preparing BG-only input...")
    img_bg_prep = img_orig.copy()
    # Fill mask area with a blurred patch from the side to remove figure hints
    side_patch = img_orig.crop((0, 0, 100, orig_size[1])).resize(orig_size).filter(ImageFilter.GaussianBlur(radius=50))
    img_bg_prep.paste(side_patch, mask=mask)
    bg_prep_path = os.path.join(OUTPUT_DIR, "intermediate_input_bg.jpg")
    img_bg_prep.save(bg_prep_path, "JPEG", quality=95)
    
    # 2. Prepare Model Only (Black BG)
    print("Preparing Model-only input...")
    img_model_prep = Image.new("RGB", orig_size, (0, 0, 0))
    img_model_prep.paste(img_orig, mask=mask)
    model_prep_path = os.path.join(OUTPUT_DIR, "intermediate_input_model.jpg")
    img_model_prep.save(model_prep_path, "JPEG", quality=95)
    
    # 3. Generate Backgrounds
    bg_models = {
        "Z-Image-Turbo": "fal-ai/fast-sdxl/image-to-image", # Using fast-sdxl as turbo proxy if specific ID not found
        "Flux-Dev": "fal-ai/flux/dev/image-to-image"
    }
    
    generated_bgs = {}
    for name, endpoint in bg_models.items():
        print(f"Generating Background with {name}...")
        url = run_fal_img2img(bg_prep_path, BG_PROMPT, 0.98, endpoint)
        if url:
            data = requests.get(url).content
            path = os.path.join(OUTPUT_DIR, f"intermediate_gen_bg_{name}.jpg")
            with open(path, "wb") as f: f.write(data)
            generated_bgs[name] = path
            
    # 4. Generate Model (one pass is enough, we composite it on both BGs)
    print("Generating Model (Low Strength)...")
    model_url = run_fal_img2img(model_prep_path, MODEL_PROMPT, 0.40, "fal-ai/fast-sdxl/image-to-image")
    if model_url:
        data = requests.get(model_url).content
        model_gen_path = os.path.join(OUTPUT_DIR, "intermediate_gen_model.jpg")
        with open(model_gen_path, "wb") as f: f.write(data)
    else:
        return

    # 5. Final Composites
    gen_model_img = Image.open(model_gen_path).convert("RGB").resize(orig_size, Image.LANCZOS)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=7))
    
    for name, bg_path in generated_bgs.items():
        print(f"Compositing {name}...")
        gen_bg_img = Image.open(bg_path).convert("RGB").resize(orig_size, Image.LANCZOS)
        final = Image.composite(gen_model_img, gen_bg_img, soft_mask)
        final_path = os.path.join(OUTPUT_DIR, f"Michaela_Final_{name}_Impasto.jpg")
        final.save(final_path, "JPEG", quality=95)
        print(f"  Saved: {final_path}")

if __name__ == '__main__':
    main()
