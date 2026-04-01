import os
import requests
import json

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def get_upload_address():
    url = f"{BASE_URL}/resource/upload-address"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    # Some APIs require some parameters for upload address (e.g. filename)
    payload = {"filename": "test.jpg"}
    response = requests.post(url, headers=headers, json=payload)
    print(f"POST {url}: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

get_upload_address()
