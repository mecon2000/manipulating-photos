import os
import requests
import json
import uuid

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def create_job():
    url = f"{BASE_URL}/jobs"
    headers = {
        "Authorization": f"Bearer {TENSOR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Trying the structure from the search snippet, but using requestId (camelCase)
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": -1,
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
                            "text": "a cute kitten",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266", # Uncensored model (if available for txt2img)
                    "steps": 20,
                    "cfgScale": 7,
                    "sampler": "Euler a"
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    print(f"POST {url}: {response.status_code}")
    print(response.text)

create_job()
