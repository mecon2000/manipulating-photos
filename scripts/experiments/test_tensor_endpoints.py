import os
import requests
import json

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def test_endpoint(path):
    url = f"{BASE_URL}/{path}"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    response = requests.get(url, headers=headers)
    print(f"GET {path}: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2)[:500])
    
    response = requests.post(url, headers=headers)
    print(f"POST {path}: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2)[:500])

# Try likely resource endpoints
test_endpoint("resource/upload-address")
test_endpoint("resource/upload")
test_endpoint("resources")
test_endpoint("resource")
