import os
import requests
import base64
import json

FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_fal_sdxl():
    # Trying SDXL (older but much more permissive)
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_b64 = encode_image(PHOTO_PATH)
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "image_url": image_data_uri,
        "prompt": "Professional fine art photography. Earthy Ochre and Clay style transformation. Warm organic tones.",
        "strength": 0.5,
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=120)
    res_json = response.json()
    print("NSFW flag:", res_json.get("has_nsfw_concepts"))
    if "images" in res_json:
        img_url = res_json['images'][0]['url']
        img_data = requests.get(img_url).content
        size = len(img_data)
        print("Image size:", size)
        if size > 5000:
            with open("outputs/test_sdxl_result.jpg", "wb") as f:
                f.write(img_data)
            print("Verdict: REAL IMAGE (SDXL works!)")
        else:
            print("Verdict: BLACK BOX")

test_fal_sdxl()
