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
                    "width": 512,
                    "height": 512,
                    "prompts": [
                        {
                            "text": "Professional fine art photography. Earthy Ochre and Clay style transformation, artistic lighting.",
                            "weight": 1.0
                        }
                    ],
                    "sdModel": "965126062386242266",
                    "steps": 25,
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
    for _ in range(30):
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

# Use the SFW model photo link directly (from workspace or public URL if needed)
# Since the tool needs a URL, we'll assume we can use a temporary link or Ronnie's GDrive link if shared.
# For now, let's use the provided SFW photo filename if it was a URL.
# Wait, I don't have a direct URL for 'BLD_7266_Earthy_Ochre_&_Clay.png' unless it's in Drive.
# Let's try to get a Drive URL if possible, or use a placeholder to test the structure.
source_url = "https://raw.githubusercontent.com/openclaw/openclaw/main/docs/static/img/logo.png" # Placeholder for test
job_id = create_img2img_job(source_url)
if job_id:
    image_url = wait_for_job(job_id)
    print(f"Final Image URL: {image_url}")
