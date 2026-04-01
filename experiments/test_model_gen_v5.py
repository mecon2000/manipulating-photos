import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE = "work/sources/Unprocessed/0762-UNPROCESSED.jpg"

def test_gen(prompt):
    url = "https://fal.run/fal-ai/fast-sdxl/image-to-image"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    from PIL import Image
    img = Image.open(SOURCE).resize((1024, 1024))
    img.save("temp_resize.jpg", quality=95)
    
    with open("temp_resize.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "prompt": prompt,
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "strength": 0.50,
        "num_inference_steps": 25,
        "enable_safety_checker": False
    }
    res = requests.post(url, headers=headers, json=payload).json()
    print(f"Resized Prompt: {prompt} | URL: {res['images'][0]['url']}")

if __name__ == '__main__':
    test_gen("A simple portrait.")
