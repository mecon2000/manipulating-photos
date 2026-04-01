import os
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def create_job():
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
                            "text": "A stylized oil painting of a sunflower, vibrant colors.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266",
                    "steps": 25,
                    "cfgScale": 7,
                    "sampler": "Euler a"
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    print(f"Status: {response.status_code}")
    print(json.dumps(result, indent=2))
    
    if response.status_code == 200:
        job_id = result.get("job", {}).get("id")
        print(f"Job created! ID: {job_id}")
        return job_id
    return None

def wait_for_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}"
    }
    
    print(f"Waiting for job {job_id}...")
    for _ in range(30):
        response = requests.get(url, headers=headers)
        result = response.json()
        status = result.get("job", {}).get("status")
        print(f"Status: {status}")
        
        if status == "SUCCESS":
            image_url = result["job"]["successInfo"]["images"][0]["url"]
            print(f"Success! Image URL: {image_url}")
            return image_url
        elif status == "FAILED":
            print(f"Failed: {json.dumps(result, indent=2)}")
            return None
        
        time.sleep(5)
    print("Timed out.")
    return None

job_id = create_job()
if job_id:
    wait_for_job(job_id)
