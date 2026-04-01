import os
import requests
import base64
import json
from datetime import datetime

# configuration
FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"
OUTPUT_ROOT = "outputs/daily_game/"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_fal_flux_img2img(prompt, image_b64):
    """Calls Fal.ai Flux Dev Image-to-Image for high-fidelity stylization."""
    url = "https://fal.run/fal-ai/flux/dev/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "strength": 0.45, # 0.45 is the "sweet spot" for stylization while keeping identity
        "guidance_scale": 7.5,
        "num_inference_steps": 28,
        "enable_safety_checker": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        return response.json()['images'][0]['url']
    except Exception as e:
        return f"ERROR: {e} - Response: {getattr(response, 'text', 'No response body') if 'response' in locals() else 'N/A'}"

def main():
    today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    daily_output_dir = os.path.join(OUTPUT_ROOT, today_str)
    os.makedirs(daily_output_dir, exist_ok=True)
    
    if not os.path.exists(PHOTO_PATH):
        print(f"Photo not found: {PHOTO_PATH}")
        return

    print(f"Fal.ai Flux-Img2Img: {PHOTO_PATH}")
    image_b64 = encode_image(PHOTO_PATH)
    
    # Prompt focusing on the Earthy Ochre & Clay style
    style_prompt = (
        "Artistic transformation of the original scene. "
        "A high-fidelity photographic study using natural ochre and clay pigments, "
        "textured earthy backgrounds, warm organic tones, and fine art lighting. "
        "Maintain the exact anatomical structure, pose, and identity of the subject."
    )
    
    result_url = call_fal_flux_img2img(style_prompt, image_b64)
    
    if result_url.startswith("http"):
        img_data = requests.get(result_url).content
        filename = "jenia_7266_FAL_Flux_Img2Img_Strength_45.png"
        save_path = os.path.join(daily_output_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(img_data)
        print(f"SUCCESS: {save_path}")
    else:
        print(result_url)

if __name__ == "__main__":
    main()
