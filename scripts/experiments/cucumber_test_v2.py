import os
import requests
import json
import uuid
import time
from io import BytesIO

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def get_resource_id(image_url):
    # 1. Download image
    resp = requests.get(image_url)
    img_data = resp.content
    
    # 2. Get upload address
    url = f"{BASE_URL}/resource/image"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    # Empty payload as seen in GitHub code
    response = requests.post(url, headers=headers, json={})
    if response.status_code != 200:
        print(f"Error getting upload address: {response.text}")
        return None
    
    put_info = response.json()
    put_url = put_info["putUrl"]
    resource_id = put_info["resourceId"]
    upload_headers = put_info["headers"]
    
    # 3. Upload image
    resp_put = requests.put(put_url, data=img_data, headers=upload_headers)
    if resp_put.status_code != 200:
        print(f"Error uploading image: {resp_put.text}")
        return None
    
    return resource_id

def create_job(resource_id, strength):
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
                            "text": "Professional fine art photography of a cucumber.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266",
                    "steps": 25,
                    "cfgScale": 7,
                    "sampler": "Euler a",
                    "img2img": {
                        "strength": strength
                    }
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error creating job: {response.text}")
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

resource_id = get_resource_id("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Cucumber_sliced.jpg/800px-Cucumber_sliced.jpg")
if resource_id:
    print(f"Resource ID: {resource_id}")
    job_id = create_job(resource_id, 0.0001)
    if job_id:
        url = wait_for_job(job_id)
        print(f"CUCUMBER_RESULT_V2: {url}")
