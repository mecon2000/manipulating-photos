import os
import requests

FAL_API_KEY = os.environ.get("FAL_API_KEY")

def test_gen():
    url = "https://fal.run/fal-ai/fast-sdxl"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": "A beautiful woman in a red bikini.",
        "num_inference_steps": 25,
        "enable_safety_checker": False
    }
    res = requests.post(url, headers=headers, json=payload).json()
    print(f"Txt2Img NSFW: {res}")

if __name__ == '__main__':
    test_gen()
