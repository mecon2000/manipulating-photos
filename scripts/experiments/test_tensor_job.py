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
    
    # requestId must be unique
    request_id = str(uuid.uuid4())
    
    payload = {
        "requestId": request_id,
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
                            "text": "A beautiful sunflower in a sunny field, oil painting style.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "603223030312674256" # Attempting a common SDXL ID
                }
            }
        ]
    }
    
    print(f"Creating job with requestId: {request_id}")
    response = requests.post(url, headers=headers, json=payload)
    
    # Log the full result for the user
    with open("outputs/tensor_last_response.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

create_job()
