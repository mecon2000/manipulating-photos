import os
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def create_job(resource_id, strength):
    url = f"{BASE_URL}/jobs"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Trying the structure from the TAMS img2img tutorial
    payload = {
        "request_id": str(uuid.uuid4()),
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
                    "prompts": [
                        {
                            "text": "a cucumber on a table",
                            "weight": 1.0
                        }
                    ],
                    "sd_model": "965126062386242266",
                    "steps": 25,
                    "cfg_scale": 7,
                    "sampler": "Euler a",
                    "denoising_strength": strength
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
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
            print(f"Failed: {json.dumps(result, indent=2)}")
            return None
        time.sleep(5)
    return None

resource_id = "ada705a9-6e27-4b4a-9914-81cabf021ee1" # From previous run
job_id = create_job(resource_id, 0.0001)
if job_id:
    print(f"Job ID: {job_id}")
    url = wait_for_job(job_id)
    print(f"CUCUMBER_RESULT_V3: {url}")
