import os
import requests
import base64
import json

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_imagen_3_img2img(prompt, image_b64):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "instances": [
            {
                "prompt": prompt,
                "image": {"bytesBase64Encoded": image_b64}
            }
        ],
        "parameters": {"sampleCount": 1}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        return response.text
    except Exception as e:
        return str(e)

image_b64 = encode_image(PHOTO_PATH)
prompt = "Maintain the structure and pose. Apply Earthy Ochre & Clay style."
print(call_imagen_3_img2img(prompt, image_b64))
