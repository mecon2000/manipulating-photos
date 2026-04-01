import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE = "work/sources/Unprocessed/0762-UNPROCESSED.jpg"

def test_model(endpoint, name):
    url = f"https://fal.run/{endpoint}"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(SOURCE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "prompt": "A fine art portrait, oil painting style.",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "strength": 0.45,
        "num_inference_steps": 25,
        "enable_safety_checker": False
    }
    # Some models use different param names
    if "flux" in endpoint:
        payload["enable_safety_checker"] = False
        
    res = requests.post(url, headers=headers, json=payload).json()
    if "images" in res:
        print(f"{name}: SUCCESS - {res['images'][0]['url']}")
    else:
        print(f"{name}: FAILED - {res}")

if __name__ == '__main__':
    test_model("fal-ai/flux/dev/image-to-image", "Flux-Dev")
    test_model("fal-ai/fast-sdxl/image-to-image", "Fast-SDXL")
    test_model("fal-ai/fooocus/image-to-image", "Fooocus")
