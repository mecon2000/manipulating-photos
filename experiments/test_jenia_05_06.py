import os
import requests
import json
import uuid
import time
from PIL import Image
from io import BytesIO

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def get_resource_id(local_path):
    img = Image.open(local_path).convert('RGB')
    img = img.resize((1024, 1024))
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_data = buf.getvalue()

    url = f"{BASE_URL}/resource/image"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={})
    put_info = response.json()
    requests.put(put_info["putUrl"], data=img_data, headers=put_info["headers"])
    return put_info["resourceId"]

def create_job(resource_id, strength):
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
                    "width": 1024,
                    "height": 1024,
                    "prompts": [{"text": "Professional fine art photography of a woman in a bathtub, Earthy Ochre and Clay style, artistic figure photography, high quality, masterpiece."}],
                    "sdModel": "965126062386242266",
                    "steps": 30,
                    "cfgScale": 7,
                    "denoisingStrength": strength
                }
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return None
    return response.json().get("job", {}).get("id")

def wait_for_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    for _ in range(40):
        response = requests.get(url, headers=headers)
        result = response.json()
        status = result.get("job", {}).get("status")
        if status == "SUCCESS":
            return result["job"]["successInfo"]["images"][0]["url"]
        time.sleep(5)
    return None

resource_id = get_resource_id("catalog/preview/jenia/BLD_7266.jpg")
if resource_id:
    for s in [0.5, 0.6]:
        job_id = create_job(resource_id, s)
        if job_id:
            url = wait_for_job(job_id)
            print(f"RESULT_{s}: {url}")
