import os
import requests
import base64
from datetime import datetime

# configuration
FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"
OUTPUT_ROOT = "outputs/daily_game/"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_fal_flux_v1_1(prompt, image_b64):
    """Calls Fal.ai Flux Pro v1.1 with Image-to-Image logic."""
    # Based on docs: https://fal.ai/models/fal-ai/flux-pro/v1.1/api
    # Note: If v1.1 doesn't support direct image_url in the prompt like img2img, 
    # we might need to use a dedicated img2img model if available for v1.1
    url = "https://fal.run/fal-ai/flux-pro/v1.1"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "prompt": f"Apply Earthy Ochre & Clay style to this image. Maintain exact anatomical structure, pose, and identity. Style details: natural ochre and clay pigments, textured earthy background, warm organic tones, high-quality photographic finish. Original image: {image_data_uri}",
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"ERROR: {e} - Response: {getattr(response, 'text', 'No response body') if 'response' in locals() else 'N/A'}"

def main():
    today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    daily_output_dir = os.path.join(OUTPUT_ROOT, today_str)
    os.makedirs(daily_output_dir, exist_ok=True)
    
    print(f"Fal.ai Flux-Pro v1.1: {PHOTO_PATH}")
    image_b64 = encode_image(PHOTO_PATH)
    
    style_prompt = (
        "Professional fine art photography. Earthy Ochre and Clay style transformation. "
        "Keep the exact same subject, pose, and details. Photo-realistic."
    )
    
    result = call_fal_flux_v1_1(style_prompt, image_b64)
    print(result)

if __name__ == "__main__":
    main()
