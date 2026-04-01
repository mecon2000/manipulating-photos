import os
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def create_img2img_job(source_image_url, strength=0.45):
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
                            "text": "Professional fine art photography. Earthy Ochre and Clay style transformation, artistic lighting, deep textures, muted tones, masterpiece, highly detailed, photorealistic.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266", # Uncensored model
                    "steps": 35,
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
    result = response.json()
    if response.status_code == 200:
        return result.get("job", {}).get("id")
    else:
        print(f"Error creating job: {json.dumps(result, indent=2)}")
        return None

def wait_for_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    for _ in range(60):
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

source_url = "https://drive.google.com/uc?export=download&id=11BWGEdtegNT7s7-ImhfI48YpjFf7_Fqw"
print(f"Starting job with source: {source_url}")
job_id = create_img2img_job(source_url)
if job_id:
    print(f"Job ID: {job_id}")
    final_url = wait_for_job(job_id)
    print(f"Final Image URL: {final_url}")
