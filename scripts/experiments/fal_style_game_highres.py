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
    """Calls Fal.ai Flux Dev Image-to-Image with high-res settings."""
    url = "https://fal.run/fal-ai/flux/dev/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    # Let's force a high resolution
    data = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "strength": 0.45,
        "guidance_scale": 7.5,
        "num_inference_steps": 28,
        "image_size": { "width": 1024, "height": 680 },
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
    
    image_b64 = encode_image(PHOTO_PATH)
    
    style_prompt = (
        "Professional fine art photography. Earthy Ochre and Clay style transformation. "
        "High-fidelity texture, natural pigments, warm lighting. Exact same person and pose."
    )
    
    print("Generating High-Res Flux-Img2Img...")
    result_url = call_fal_flux_img2img(style_prompt, image_b64)
    
    if result_url.startswith("http"):
        img_data = requests.get(result_url).content
        filename = "jenia_7266_HighRes_Flux.png"
        save_path = os.path.join(daily_output_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(img_data)
        print(f"SUCCESS: {save_path}")
        print(f"File size: {len(img_data)} bytes")
    else:
        print(result_url)

if __name__ == "__main__":
    main()
