import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE = "work/sources/Unprocessed/0762-UNPROCESSED.jpg"
MASK = "outputs/anya_indigo_final_v2/intermediate_mask.png"

def test_inpainting():
    url = "https://fal.run/fal-ai/fast-sdxl/inpainting"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(SOURCE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open(MASK, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    payload = {
        "prompt": "An abstract fine art Indigo Dye Aesthetic background, oil paint impasto, moody atmosphere, NO PEOPLE.",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "mask_url": f"data:image/png;base64,{mask_b64}",
        "strength": 0.95,
        "num_inference_steps": 30,
        "enable_safety_checker": False
    }
    res = requests.post(url, headers=headers, json=payload).json()
    if "images" in res:
        print(f"Inpainting Result: {res['images'][0]['url']}")
    else:
        print(f"Inpainting FAILED: {res}")

if __name__ == '__main__':
    test_inpainting()
