import os
import requests
import json

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def test_resource():
    url = f"{BASE_URL}/resource/models?pageSize=1"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

test_resource()
