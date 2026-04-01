import os
import requests
import json
import uuid

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def get_upload_address():
    url = f"{BASE_URL}/resource/upload-address"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "requestId": str(uuid.uuid4()),
        "filename": "test.png",
        "type": "IMAGE"
    }
    response = requests.post(url, headers=headers, json=payload)
    print(f"POST {url}: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

get_upload_address()
