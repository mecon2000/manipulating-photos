import os
import requests
import base64
from PIL import Image

FAL_API_KEY = os.getenv("FAL_API_KEY")

# Create a simple SFW image: a bright red square
img = Image.new('RGB', (512, 512), color = (255, 0, 0))
img.save('outputs/sfw_test.jpg')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_fal_sfw():
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_b64 = encode_image('outputs/sfw_test.jpg')
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"
    
    data = {
        "image_url": image_data_uri,
        "prompt": "A beautiful landscape with a mountain and a river, oil painting style.",
        "strength": 0.5,
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=120)
    res_json = response.json()
    print("NSFW flag:", res_json.get("has_nsfw_concepts"))
    if "images" in res_json:
        img_url = res_json['images'][0]['url']
        img_data = requests.get(img_url).content
        print("Image size:", len(img_data))
        with open("outputs/sfw_test_result.jpg", "wb") as f:
            f.write(img_data)
        print("SFW Result saved to outputs/sfw_test_result.jpg")

test_fal_sfw()
