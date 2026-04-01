import os
import requests
import base64
import json

FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def debug_fal_sdxl():
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_b64 = encode_image(PHOTO_PATH)
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "image_url": image_data_uri,
        "prompt": "Professional fine art photography. Earthy Ochre and Clay style transformation.",
        "strength": 0.5,
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=120)
    print(json.dumps(response.json(), indent=2))

debug_fal_sdxl()
