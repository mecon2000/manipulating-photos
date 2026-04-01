import os
import requests
import base64
import json

FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_fal_safe():
    url = "https://fal.run/fal-ai/flux/dev/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_b64 = encode_image(PHOTO_PATH)
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    # A very safe, boring prompt
    data = {
        "image_url": image_data_uri,
        "prompt": "A beautiful sunflower in a field of green grass, bright sunny day, 8k resolution.",
        "strength": 0.8, # Change it a lot to make it safe
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=180)
    res_json = response.json()
    print(json.dumps(res_json, indent=2))
    
    if "images" in res_json:
        url = res_json['images'][0]['url']
        os.system(f"curl -s -o outputs/test_safe_result.jpg {url}")
        os.system("ls -lh outputs/test_safe_result.jpg")

test_fal_safe()
