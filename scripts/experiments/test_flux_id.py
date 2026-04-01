import os
import sys
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def run_tensor_job(payload):
    url = f"{TENSOR_BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    return response.json()

payload = {
    "requestId": str(uuid.uuid4()),
    "stages": [
        {"type": "INPUT_INITIALIZE", "inputInitialize": {"seed": 42, "count": 1}},
        {"type": "DIFFUSION", "diffusion": {
            "width": 512, "height": 512, "prompts": [{"text": "a cute cat on a table"}],
            "sdModel": "1046927429141014138", "steps": 25, "cfgScale": 3.5, "sampler": "Euler a"
        }}
    ]
}
print(json.dumps(run_tensor_job(payload), indent=2))
