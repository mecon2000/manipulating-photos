import os
import sys
import requests
import json
import base64
import argparse
from PIL import Image, ImageFilter, ImageOps, ImageStat, ImageEnhance

FAL_API_KEY = os.environ.get("FAL_API_KEY")

def is_black(img_path):
    img = Image.open(img_path).convert("RGB")
    stat = ImageStat.Stat(img)
    return sum(stat.extrema[0]) == 0 and sum(stat.extrema[1]) == 0 and sum(stat.extrema[2]) == 0

def run_fal_api(endpoint, payload):
    url = f"https://fal.run/{endpoint}"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200: return response.json()
    print(f"Error {response.status_code}: {response.text}")
    return None

def color_grade_to_indigo(img):
    # Match the model to indigo dye aesthetic: slight blue tint, lower saturation
    r, g, b = img.split()
    r = r.point(lambda i: i * 0.8)
    b = b.point(lambda i: min(255, i * 1.2))
    img = Image.merge("RGB", (r, g, b))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.7)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--style", default="Oil_Paint_Impasto")
    parser.add_argument("--bg-strength", type=float, default=0.98)
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
    mask_path = os.path.join(args.output_dir, "intermediate_mask.png")
    mask_img.save(mask_path)

    # 2. Prepare BG Input (Full Wipe)
    img_bg_prep = img_orig.copy()
    avg_color = ImageStat.Stat(img_orig).median
    fill = Image.new("RGB", orig_size, tuple(avg_color))
    img_bg_prep.paste(fill, mask=mask_img)
    bg_prep_path = os.path.join(args.output_dir, "intermediate_input_bg.jpg")
    img_bg_prep.save(bg_prep_path, "JPEG", quality=95)

    # 3. Generate BGs
    bg_prompts = {
        "Flux-Dev": f"An abstract fine art {args.style} background, NO PEOPLE, NO FIGURES, NO HUMANS, dark cinematic aesthetic.",
        "Turbo-SDXL": f"An abstract fine art {args.style} background, NO PEOPLE, NO FIGURES, NO HUMANS, dark cinematic aesthetic."
    }
    endpoints = {
        "Flux-Dev": "fal-ai/flux/dev/image-to-image",
        "Turbo-SDXL": "fal-ai/fast-sdxl/image-to-image"
    }
    
    gen_bgs = {}
    with open(bg_prep_path, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode('utf-8')

    for name, prompt in bg_prompts.items():
        print(f"--- Step 2: Generating Background ({name}) ---")
        res = run_fal_api(endpoints[name], {
            "prompt": prompt,
            "image_url": f"data:image/jpeg;base64,{bg_b64}",
            "strength": args.bg_strength,
            "num_inference_steps": 35,
            "enable_safety_checker": False
        })
        if res:
            bg_path = os.path.join(args.output_dir, f"intermediate_gen_bg_{name}.jpg")
            with open(bg_path, "wb") as f: f.write(requests.get(res["images"][0]["url"]).content)
            if not is_black(bg_path): gen_bgs[name] = bg_path

    # 4. Prepare Model (Processed Original)
    print("--- Step 3: Preparing Model (Color Graded) ---")
    if "Indigo" in args.style:
        img_model_final = color_grade_to_indigo(img_orig)
    else:
        img_model_final = img_orig # Fallback
    
    # 5. Composite
    print("--- Step 4: Final Compositing ---")
    soft_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    
    for name, bg_path in gen_bgs.items():
        gen_bg_img = Image.open(bg_path).convert("RGB").resize(orig_size, Image.LANCZOS)
        final = Image.composite(img_model_final, gen_bg_img, soft_mask)
        final_path = os.path.join(args.output_dir, f"Final_Result_{name}_{args.style}.jpg")
        final.save(final_path, "JPEG", quality=95)
        print(f"Success! Saved: {final_path}")

if __name__ == '__main__':
    main()
