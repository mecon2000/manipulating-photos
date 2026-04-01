import os
import json
import requests
import base64
from datetime import datetime

# configuration
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"
OUTPUT_ROOT = "outputs/daily_game/"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_imagen_img2img(prompt, image_b64):
    """Calls Gemini's Imagen 4.0 model with a reference image (Image-to-Image)."""
    if not GEMINI_API_KEY:
        return "ERROR: GOOGLE_API_KEY not set."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Structure for Image-to-Image / Reference Image
    data = {
        "instances": [
            {
                "prompt": prompt,
                "image": {
                    "bytesBase64Encoded": image_b64
                }
            }
        ],
        "parameters": {
            "sampleCount": 1,
            # Increasing structural consistency if supported by the model version
            "imageKind": "image", 
            "editConfig": {
                "editMode": "IMG2IMG" # Common parameter for Image-to-Image
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        predictions = response.json().get('predictions', [])
        if not predictions:
            return f"ERROR: No predictions. Response: {response.text}"
        return base64.b64decode(predictions[0].get('bytesBase64Encoded'))
    except Exception as e:
        return f"ERROR: {e} - Response: {getattr(response, 'text', 'No response body') if 'response' in locals() else 'N/A'}"

def main():
    today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    daily_output_dir = os.path.join(OUTPUT_ROOT, today_str)
    os.makedirs(daily_output_dir, exist_ok=True)
    
    if not os.path.exists(PHOTO_PATH):
        print(f"Photo not found: {PHOTO_PATH}")
        return

    print(f"Retrying V3 (Image-to-Image): {PHOTO_PATH}")
    image_b64 = encode_image(PHOTO_PATH)
    
    # Focus on structural preservation in the prompt
    style_prompt = (
        "Apply an artistic transformation using natural ochre and clay pigments, "
        "textured earthy background, and warm organic tones. "
        "CRITICAL: Maintain the exact body shape, pose, and structure of the person in the source image. "
        "Do not change the woman's curves, the angle of her body, or the composition of the bathtub. "
        "The person must remain recognizable as the same individual. Only change the artistic style."
    )
    
    result = call_imagen_img2img(style_prompt, image_b64)
    
    if isinstance(result, bytes):
        filename = "jenia_7266_RETRY_v3_ImageToImage.png"
        save_path = os.path.join(daily_output_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(result)
        print(f"SUCCESS: {save_path}")
    else:
        print(result)

if __name__ == "__main__":
    main()
