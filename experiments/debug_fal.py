import os
import requests
import base64
import json

FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def debug_fal():
    url = "https://fal.run/fal-ai/flux/dev/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_b64 = encode_image(PHOTO_PATH)
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "image_url": image_data_uri,
        "prompt": "Earthy style transformation.",
        "strength": 0.45,
        "image_size": { "width": 1024, "height": 680 },
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=180)
    print("STATUS CODE:", response.status_code)
    try:
        print("JSON RESPONSE:", json.dumps(response.json(), indent=2))
    except:
        print("TEXT RESPONSE:", response.text)

debug_fal()
