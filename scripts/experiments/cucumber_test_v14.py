import os
import requests
import json
import uuid
from PIL import Image
from io import BytesIO

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def get_resource_id():
    # Create a tiny 64x64 PNG
    img = Image.new('RGB', (64, 64), color='red')
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_data = buf.getvalue()

    url = f"{BASE_URL}/resource/image"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={})
    put_info = response.json()
    requests.put(put_info["putUrl"], data=img_data, headers=put_info["headers"])
    return put_info["resourceId"]

def create_job(resource_id):
    url = f"{BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "image_resource_id": resource_id,
                    "count": 1
                }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": 512,
                    "height": 512,
                    "prompts": [{"text": "red square"}],
                    "sdModel": "600423083519508503",
                    "steps": 10,
                    "cfgScale": 7
                }
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(response.text)

resource_id = get_resource_id()
create_job(resource_id)
