import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Processed/BLD_1654E.jpg"

def run_rembg():
    url = "https://fal.run/fal-ai/rembg"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    with open(SOURCE_IMAGE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "image_url": f"data:image/jpeg;base64,{img_b64}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.status_code, response.text

if __name__ == '__main__':
    status, text = run_rembg()
    print(f"Status: {status}\nText: {text}")
