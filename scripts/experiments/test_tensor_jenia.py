import os
import requests
import json
import uuid
import time

TENSOR_API_KEY = os.getenv("TENSOR_API_KEY")
BASE_URL = "https://ap-east-1.tensorart.cloud/v1"

def create_img2img_job(source_image_url, strength=0.45):
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
                    "width": 1024, # Let's use 1024 for higher quality if the model supports it
                    "height": 1024,
                    "prompts": [
                        {
                            "text": "Professional fine art photography. Earthy Ochre and Clay style transformation, artistic lighting, deep textures, muted tones, masterpiece.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266",
                    "steps": 30,
                    "cfgScale": 7,
                    "sampler": "Euler a",
                    "img2img": {
                        "image": source_image_url,
                        "strength": strength
                    }
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    if response.status_code == 200:
        return result.get("job", {}).get("id")
    else:
        print(f"Error creating job: {json.dumps(result, indent=2)}")
        return None

def wait_for_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {TENSOR_API_KEY}"}
    for _ in range(60): # 5 mins max
        response = requests.get(url, headers=headers)
        result = response.json()
        status = result.get("job", {}).get("status")
        if status == "SUCCESS":
            return result["job"]["successInfo"]["images"][0]["url"]
        elif status == "FAILED":
            print(f"Failed: {json.dumps(result, indent=2)}")
            return None
        time.sleep(5)
    return None

# The user will provide the direct download URL for BLD_7266.jpg in Drive
# Since I cannot get it automatically due to GOG password, I'll ask him for it or assume he uploaded it.
# Wait, I'll try to guess the direct download URL if the file ID is known.
# Actually, I'll try to find the ID of the file he just uploaded (if he did).
# If he just created the folder, I'll use a placeholder and explain.
