import os
import requests
import base64
import json

FAL_API_KEY = os.environ.get("FAL_API_KEY")
TASKS = [
    {
        "target": "work/targets/Michaela_BLD_1654E_Oil_Paint_Impasto L.jpg",
        "source": "work/sources/Processed/BLD_1654E.jpg",
        "output": "outputs/favorites_faceswap/Michaela_BLD_1654E_Oil_Paint_Impasto L-face-swap.png"
    },
    {
        "target": "work/targets/Elly__BLD_5440E_Smoke_&_Mirrors L.png",
        "source": "work/sources/01 hotel/Processed/BLD_5440E.jpg",
        "output": "outputs/favorites_faceswap/Elly__BLD_5440E_Smoke_&_Mirrors L-face-swap.png"
    },
    {
        "target": "work/targets/BLD_2888_Lace_Pattern_Shadows L.png",
        "source": "work/sources/Unprocessed/BLD_2888 - UNPROCESSED.jpg",
        "output": "outputs/favorites_faceswap/BLD_2888_Lace_Pattern_Shadows L-face-swap.png"
    },
    {
        "target": "work/targets/Anya_0762-UNPROCESSED_Indigo_Dye_Aesthetic L.jpg",
        "source": "work/sources/Unprocessed/0762-UNPROCESSED.jpg",
        "output": "outputs/favorites_faceswap/Anya_0762-UNPROCESSED_Indigo_Dye_Aesthetic L-face-swap.png"
    }
]
os.makedirs("outputs/favorites_faceswap", exist_ok=True)

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
    for task in TASKS:
        print(f"Processing: {task['target']}")
        result_url = run_fal_faceswap(task["source"], task["target"])
        if result_url:
            img_data = requests.get(result_url).content
            with open(task["output"], "wb") as f:
                f.write(img_data)
            print(f"  Saved: {task['output']}")
        else:
            print(f"  Failed for {task['target']}")

if __name__ == '__main__':
    main()
