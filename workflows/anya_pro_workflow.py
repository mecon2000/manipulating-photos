import os
import requests
import json
import base64
from PIL import Image, ImageFilter, ImageOps

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Unprocessed/0762-UNPROCESSED.jpg"
OUTPUT_DIR = "outputs/anya_pro_workflow/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG_PROMPT = "An abstract fine art oil painting impasto background, indigo dye aesthetic, moody atmospheric studio, ethereal lighting, dark aesthetic, NO PEOPLE, NO FIGURES, NO HUMANS."
MODEL_PROMPT = "A fine art portrait of a woman, indigo dye aesthetic, oil paint impasto texture, high detail, realistic skin."

def run_fal_api(endpoint, payload):
    url = f"https://fal.run/{endpoint}"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def main():
    # 1. Get Mask via rembg
    print("Getting Perfect Mask (rembg)...")
    with open(SOURCE_IMAGE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    res_rembg = run_fal_api("fal-ai/rembg", {"image_url": f"data:image/jpeg;base64,{img_b64}"})
    if not res_rembg: return
    mask_url = res_rembg["image"]["url"]
    mask_img = Image.open(requests.get(mask_url, stream=True).raw).split()[3]
    mask_path = os.path.join(OUTPUT_DIR, "anya_mask.png")
    mask_img.save(mask_path)

    img_orig = Image.open(SOURCE_IMAGE).convert("RGB")
    orig_size = img_orig.size
    
    # 2. Prepare BG Only input (muffle the model area)
    print("Preparing BG input...")
    img_bg_prep = img_orig.copy()
    side_patch = img_orig.crop((0, 0, 100, orig_size[1])).resize(orig_size).filter(ImageFilter.GaussianBlur(radius=50))
    img_bg_prep.paste(side_patch, mask=mask_img)
    bg_prep_path = os.path.join(OUTPUT_DIR, "input_bg.jpg")
    img_bg_prep.save(bg_prep_path, "JPEG", quality=95)

    # 3. Prepare Model Only input
    print("Preparing Model input...")
    img_model_prep = Image.new("RGB", orig_size, (0, 0, 0))
    img_model_prep.paste(img_orig, mask=mask_img)
    model_prep_path = os.path.join(OUTPUT_DIR, "input_model.jpg")
    img_model_prep.save(model_prep_path, "JPEG", quality=95)

    # 4. Run Tensor for BG (Turbo and Flux)
    bg_models = {
        "Z-Image-Turbo": "fal-ai/fast-sdxl/image-to-image",
        "Flux-Dev": "fal-ai/flux/dev/image-to-image"
    }
    
    gen_bgs = {}
    for name, endpoint in bg_models.items():
        print(f"Generating BG with {name}...")
        payload = {
            "prompt": BG_PROMPT,
            "image_url": f"data:image/jpeg;base64,{img_b64}", # Use full img for context but strength 0.98
            "strength": 0.98,
            "num_inference_steps": 35,
            "enable_safety_checker": False
        }
        # For BG run, let's use the prepped BG to avoid person hints
        with open(bg_prep_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode('utf-8')
        payload["image_url"] = f"data:image/jpeg;base64,{bg_b64}"
        
        res = run_fal_api(endpoint, payload)
        if res:
            url = res["images"][0]["url"]
            path = os.path.join(OUTPUT_DIR, f"gen_bg_{name}.jpg")
            with open(path, "wb") as f: f.write(requests.get(url).content)
            gen_bgs[name] = path

    # 5. Run Tensor for Model
    print("Generating Model (Low Strength)...")
    with open(model_prep_path, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "prompt": MODEL_PROMPT,
        "image_url": f"data:image/jpeg;base64,{model_b64}",
        "strength": 0.40,
        "num_inference_steps": 30,
        "enable_safety_checker": False
    }
    res_model = run_fal_api("fal-ai/fast-sdxl/image-to-image", payload)
    if res_model:
        url = res_model["images"][0]["url"]
        model_gen_path = os.path.join(OUTPUT_DIR, "gen_model.jpg")
        with open(model_gen_path, "wb") as f: f.write(requests.get(url).content)
    else: return

    # 6. Composite (Reduced Feathering)
    print("Compositing (Reduced Feathering radius=2)...")
    gen_model_img = Image.open(model_gen_path).convert("RGB").resize(orig_size, Image.LANCZOS)
    soft_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=2)) # Reduced from 7 to 2
    
    for name, bg_path in gen_bgs.items():
        gen_bg_img = Image.open(bg_path).convert("RGB").resize(orig_size, Image.LANCZOS)
        final = Image.composite(gen_model_img, gen_bg_img, soft_mask)
        final_path = os.path.join(OUTPUT_DIR, f"Anya_Final_{name}_Impasto.jpg")
        final.save(final_path, "JPEG", quality=95)
        print(f"  Saved: {final_path}")

if __name__ == '__main__':
    main()
