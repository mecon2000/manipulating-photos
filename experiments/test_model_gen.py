import os
import requests
import json
import base64
from PIL import Image

FAL_API_KEY = os.environ.get("FAL_API_KEY")
MODEL_INPUT = "outputs/anya_indigo_final/intermediate_input_model.jpg"

def test_gen(strength):
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(MODEL_INPUT, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "prompt": "A fine art portrait, Indigo_Dye_Aesthetic style, high detail, realistic skin texture.",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "strength": strength,
        "num_inference_steps": 30,
        "enable_safety_checker": False
    }
    res = requests.post(url, headers=headers, json=payload).json()
    print(f"Strength {strength}: {res}")

if __name__ == '__main__':
    test_gen(0.40)
    test_gen(0.60)
