import os
import requests
import base64

FAL_API_KEY = os.environ.get("FAL_API_KEY")
MODEL_INPUT = "outputs/anya_indigo_final/intermediate_input_model.jpg"

def test_gen(strength, model_endpoint):
    url = f"https://fal.run/{model_endpoint}"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    with open(MODEL_INPUT, "rb") as f:
        img_b64 = base64.get_encoder().encode(f.read()).decode('utf-8') if hasattr(base64, 'get_encoder') else base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "prompt": "A fine art portrait of a woman, Indigo_Dye_Aesthetic style, high detail.",
        "image_url": f"data:image/jpeg;base64,{img_b64}",
        "strength": strength,
        "num_inference_steps": 30,
        "enable_safety_checker": False
    }
    
    response = requests.post(url, headers=headers, json=payload)
    print(f"Model: {model_endpoint} | Strength: {strength} | Status: {response.status_code}")
    res = response.json()
    if "images" in res:
        print(f"  URL: {res['images'][0]['url']}")
    else:
        print(f"  Error: {res}")

if __name__ == '__main__':
    # Try different models for the model pass
    test_gen(0.40, "fal-ai/fast-sdxl/image-to-image")
    test_gen(0.40, "fal-ai/stable-diffusion-v1-5/image-to-image")
