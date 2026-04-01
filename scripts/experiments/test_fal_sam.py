import os
import requests
import json
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "work/sources/Processed/BLD_1654E.jpg"

def run_fal_sam():
    try:
        # Check if fal-ai/sam or similar exists
        # Actually, let's look for background removal
        url = "https://fal.run/fal-ai/sam/image-to-mask"
        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
            "Content-Type": "application/json"
        }
        with open(SOURCE_IMAGE, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # We need a prompt or a point for SAM
        # Let's try a generic prompt "person" or just get segments
        payload = {
            "image_url": f"data:image/jpeg;base64,{img_b64}",
            "prompt": "person"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        return response.status_code, response.text
    except Exception as e:
        return 500, str(e)

if __name__ == '__main__':
    status, text = run_fal_sam()
    print(f"Status: {status}\nText: {text}")
