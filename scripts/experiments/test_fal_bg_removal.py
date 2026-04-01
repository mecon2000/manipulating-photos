import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Processed/BLD_1654E.jpg"

def run_bg_removal():
    # Attempting to find a generic background removal endpoint on Fal AI
    # Many providers have this as fal-ai/modnet or fal-ai/bg-remover
    url = "https://fal.run/fal-ai/modnet"
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
    status, text = run_bg_removal()
    print(f"Status: {status}\nText: {text}")
