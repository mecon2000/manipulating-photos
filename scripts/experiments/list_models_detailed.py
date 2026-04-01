import os
import requests
import json

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def list_models():
    # Attempting to fetch a list of models with 'inpaint' in name
    url = f"{BASE_URL}/models?pageSize=50"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    try:
        res = requests.get(url, headers=headers).json()
        for m in res.get("models", []):
            if "inpaint" in m.get("name", "").lower():
                print(f"Name: {m['name']} | ID: {m['id']}")
    except:
        pass

list_models()
