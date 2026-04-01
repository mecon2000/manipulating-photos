import os
import requests
import base64

FAL_API_KEY = os.getenv("FAL_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_fal_nsfw_src():
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_b64 = encode_image(PHOTO_PATH)
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "image_url": image_data_uri,
        "prompt": "A bowl of fresh fruit on a wooden table, hyperrealistic, high resolution.",
        "strength": 0.9, # High strength to change it completely
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=120)
    res_json = response.json()
    print("NSFW flag:", res_json.get("has_nsfw_concepts"))
    if "images" in res_json:
        img_url = res_json['images'][0]['url']
        img_data = requests.get(img_url).content
        print("Image size:", len(img_data))
        if len(img_data) < 5000:
            print("Verdict: BLACK BOX (Silent censoring)")
        else:
            print("Verdict: REAL IMAGE")

test_fal_nsfw_src()
