import os
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def get_resource_id(image_url):
    resp = requests.get(image_url)
    img_data = resp.content
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
                    "width": 512,
                    "height": 512,
                    "prompts": [{"text": "a cucumber on a table"}],
                    "sdModel": "600423083519508503", # Standard model for test
                    "steps": 25,
                    "cfgScale": 7,
                    "sampler": "Euler a",
                    "denoisingStrength": strength
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
        time.sleep(5)
    return None

resource_id = get_resource_id("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Cucumber_sliced.jpg/800px-Cucumber_sliced.jpg")
if resource_id:
    print(f"Resource ID: {resource_id}")
    job_id = create_job(resource_id, 0.0001)
    if job_id:
        print(f"Job ID: {job_id}")
        url = wait_for_job(job_id)
        print(f"CUCUMBER_FINAL_RESULT: {url}")
