import os
import requests
import json

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def find_inpaint():
    url = f"{BASE_URL}/models?search=inpainting&pageSize=10"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    res = requests.get(url, headers=headers).json()
    for m in res.get("models", []):
        print(f"Name: {m['name']} | ID: {m['id']}")

find_inpaint()
