import os
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def create_img2img_job(source_image_url, strength):
    url = f"{BASE_URL}/jobs"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": -1,
                    "count": 1
                }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": 1024,
                    "height": 1024,
                    "prompts": [
                        {
                            "text": "Professional fine art photography of a woman in a bathtub, artistic figure photography. Earthy Ochre and Clay style transformation, natural pigments, muted organic tones, high quality, fine art aesthetic.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266",
                    "steps": 30,
                    "cfgScale": 7,
                    "sampler": "Euler a",
                    "img2img": {
                        "image": source_image_url,
                        "strength": strength
                    }
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
    for _ in range(30):
        response = requests.get(url, headers=headers)
        result = response.json()
        status = result.get("job", {}).get("status")
        if status == "SUCCESS":
            return result["job"]["successInfo"]["images"][0]["url"]
        elif status == "FAILED":
            return None
        time.sleep(5)
    return None

source_url = "https://drive.google.com/uc?export=download&id=11BWGEdtegNT7s7-ImhfI48YpjFf7_Fqw"
# Testing 0.02 and 0.05
for s in [0.02, 0.05]:
    job_id = create_img2img_job(source_url, s)
    if job_id:
        url = wait_for_job(job_id)
        print(f"STRENGTH_{s}: {url}")
