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
                    "width": 512,
                    "height": 512,
                    "prompts": [
                        {
                            "text": "Professional fine art photography of a cucumber, high quality.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266",
                    "steps": 25,
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
        print(f"Error {response.status_code}: {response.text}")
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

source_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Cucumber_sliced.jpg/800px-Cucumber_sliced.jpg"
job_id = create_img2img_job(source_url, 0.0001)
if job_id:
    url = wait_for_job(job_id)
    print(f"CUCUMBER_RESULT: {url}")
