import os
import requests
import base64
import json

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PHOTO_PATH = "catalog/preview/jenia/BLD_7266.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gemini_vision_to_image(prompt, image_b64):
    # Trying gemini-2.0-flash-exp if available, otherwise 1.5
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        return response.text
    except Exception as e:
        return str(e)

image_b64 = encode_image(PHOTO_PATH)
prompt = "Recreate this exact photo but in the style of Earthy Ochre & Clay pigments. Maintain the exact pose and identity. Output the result as an image."
print(call_gemini_vision_to_image(prompt, image_b64))
