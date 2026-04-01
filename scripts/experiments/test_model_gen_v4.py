import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE = "work/sources/Unprocessed/0762-UNPROCESSED.jpg"

def test_gen(endpoint):
    url = f"https://fal.run/{endpoint}"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(SOURCE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "prompt": "A landscape.",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "strength": 0.50,
        "num_inference_steps": 25,
        "enable_safety_checker": False
    }
    res = requests.post(url, headers=headers, json=payload).json()
    print(f"Endpoint: {endpoint} | URL: {res['images'][0]['url'] if 'images' in res else res}")

if __name__ == '__main__':
    test_gen("fal-ai/sdxl/image-to-image")
    test_gen("fal-ai/flux/dev/image-to-image")
