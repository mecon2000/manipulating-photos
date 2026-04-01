import os
import requests
import json
import uuid
import time
from PIL import Image, ImageDraw
from io import BytesIO

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def test_inpaint():
    # 1. Create a simple test image (red background with a blue square)
    img = Image.new("RGB", (512, 512), (255, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 200, 300, 300], fill=(0, 0, 255))
    
    # 2. Create a mask (white over the blue square)
    mask = Image.new("L", (512, 512), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle([200, 200, 300, 300], fill=255)
    
    # 3. Upload to Tensor
    def upload(image_pil):
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        res = requests.post(f"{BASE_URL}/resource/image", json={}, headers={"Authorization": f"Bearer {TENSOR_API_KEY}"}).json()
        requests.put(res["putUrl"], data=buf.getvalue(), headers=res["headers"])
        return res["resourceId"]

    img_res = upload(img)
    mask_res = upload(mask)
    
    # 4. Job
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": { "image_resource_id": img_res, "count": 1 }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": 512, "height": 512,
                    "prompts": [{ "text": "solid red background", "weight": 1.0 }],
                    "sdModel": "615016259364993183", # Standard SD 1.5
                    "steps": 20, "cfgScale": 7, "denoisingStrength": 1.0,
                    "mask_resource_id": mask_res
                }
            }
        ]
    }
    
    res = requests.post(f"{BASE_URL}/jobs", json=payload, headers={"Authorization": f"Bearer {TENSOR_API_KEY}"}).json()
    print(res); job_id = res.get("job", {}).get("id")
    print(f"Job ID: {job_id}")
    
    for _ in range(20):
        time.sleep(5)
        res = requests.get(f"{BASE_URL}/jobs/{job_id}", headers={"Authorization": f"Bearer {TENSOR_API_KEY}"}).json()
        status = res["job"]["status"]
        print(f"Status: {status}")
        if status == "SUCCESS":
            print(f"URL: {res['job']['successInfo']['images'][0]['url']}")
            break

test_inpaint()
