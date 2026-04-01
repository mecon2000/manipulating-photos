import os
import requests
import base64

FAL_API_KEY = os.getenv("FAL_API_KEY")

def test_lama():
    url = "https://fal.run/fal-ai/lama"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    
    with open("outputs/workflow_pro_v9_Rong_IMG_9214_20260401_170441/0_original.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open("outputs/workflow_pro_v9_Rong_IMG_9214_20260401_170441/1_mask.png", "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    payload = {
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "mask_url": f"data:image/png;base64,{mask_b64}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    print(response.json())

if __name__ == "__main__":
    test_lama()
