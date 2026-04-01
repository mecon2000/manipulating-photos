import os
import requests
import json
import base64

FAL_API_KEY = os.getenv("FAL_API_KEY")
SOURCE_IMAGE = "shahar_zach_true_original.jpg"
TARGET_IMAGE = "outputs/shahar_zach_true_batch/Shahar_Zach_True_Tensor_V1.jpg"
OUTPUT_DIR = "outputs/shahar_zach_faceswap/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_fal_faceswap(source_path, target_path):
    try:
        url = "https://fal.run/fal-ai/face-swap"
        headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
        source_b64 = encode_image(source_path)
        target_b64 = encode_image(target_path)
        data = {
            "base_image_url": f"data:image/jpeg;base64,{target_b64}",
            "swap_image_url": f"data:image/jpeg;base64,{source_b64}"
        }
        res = requests.post(url, headers=headers, json=data).json()
        if "image" in res:
            return res["image"]["url"]
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("Running Face Swap...")
    url = run_fal_faceswap(SOURCE_IMAGE, TARGET_IMAGE)
    if url:
        img_data = requests.get(url).content
        filename = "Shahar_Zach_FaceSwap_V1.png"
        local_path = os.path.join(OUTPUT_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(img_data)
        print(f"  Saved: {filename}")
    else:
        print("  Failed")

if __name__ == "__main__":
    main()
