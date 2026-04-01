import os
import requests
import json

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def find_models(query):
    print(f"Searching for: {query}")
    url = f"{BASE_URL}/models?search={query}&pageSize=20"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return
        res = response.json()
        for m in res.get("models", []):
            print(f"Name: {m['name']} | ID: {m['id']}")
    except Exception as e:
        print(f"Error: {e}")

find_models("sdxl inpainting")
find_models("sdxl inpaint")
