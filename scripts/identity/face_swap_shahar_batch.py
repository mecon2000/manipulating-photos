import os
import requests
import base64
import json

FAL_API_KEY = os.environ.get("FAL_API_KEY")
SOURCE_IMAGE = "shahar_zach_true_original.jpg"
TARGETS = [
    "outputs/shahar_zach_true_batch/Shahar_Zach_True_Tensor_V1.jpg",
    "outputs/shahar_zach_true_batch/Shahar_Zach_True_Tensor_V2.jpg"
]
OUTPUT_DIR = "outputs/shahar_zach_faceswap_batch/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_fal_faceswap(source_path, target_path):
    try:
        url = "https://fal.run/fal-ai/face-swap"
        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        with open(source_path, "rb") as f_s:
            source_b64 = base64.b64encode(f_s.read()).decode('utf-8')
        with open(target_path, "rb") as f_t:
            target_b64 = base64.b64encode(f_t.read()).decode('utf-8')

        payload = {
            "base_image_url": f"data:image/jpeg;base64,{target_b64}",
            "swap_image_url": f"data:image/jpeg;base64,{source_b64}"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "image" in res:
            return res["image"]["url"]
        return None
    except Exception as e:
        print(f"Error swapping {target_path}: {e}")
        return None

def main():
    if not FAL_API_KEY:
        print("Missing FAL_API_KEY")
        return

    for i, target in enumerate(TARGETS, 1):
        print(f"Processing target {i}: {target}")
        result_url = run_fal_faceswap(SOURCE_IMAGE, target)
        if result_url:
            img_data = requests.get(result_url).content
            filename = f"Shahar_Zach_FaceSwap_Tensor_V{i}.png"
            local_path = os.path.join(OUTPUT_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(img_data)
            print(f"  Saved: {local_path}")
        else:
            print(f"  Failed for {target}")

if __name__ == '__main__':
    main()
