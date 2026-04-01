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
                            "text": "sunflower",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "stable-diffusion-v1-5"
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    print(json.dumps(response.json(), indent=2))

create_job()
