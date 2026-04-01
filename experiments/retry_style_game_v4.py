import os
import random
import json
import requests
import base64
from datetime import datetime

# configuration
OUTPUT_ROOT = "outputs/daily_game/"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def call_imagen(prompt):
    """Calls Gemini's Imagen 4.0 model to generate an image."""
    if not GEMINI_API_KEY:
        return "ERROR: GOOGLE_API_KEY not set."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        predictions = response.json().get('predictions', [])
        if not predictions:
            return f"ERROR: No predictions. Response: {response.text}"
        return base64.b64decode(predictions[0].get('bytesBase64Encoded'))
    except Exception as e:
        return f"ERROR: {e}"

def main():
    today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    daily_output_dir = os.path.join(OUTPUT_ROOT, today_str)
    os.makedirs(daily_output_dir, exist_ok=True)
    
    photo_name = "jenia_7266"
    
    # Coordinate-based blueprint for structural preservation
    blueprint = (
        "A photographic study of a woman (Jenia) in a white bathtub filled with water. "
        "The composition follows a strict diagonal arc from top right to bottom left. "
        "Her chin and tilted head are at the top right (coordinates 90,10). "
        "Her center chest is at (65,25) with her left breast at (55,15) catching a bright highlight. "
        "Her right hand is placed firmly on her right breast at (76,26) with fingers spread: "
        "index at (75,24), middle at (80,24), ring at (83,28), pinky at (82,35). "
        "Her left hand is submerged at (38,41). Her body curves down to the submerged navel at (60,80). "
        "A right thigh enters from the lower-left corner (15,80) angling down. "
        "The scene features high-contrast chiaroscuro lighting from the top left, "
        "creating specular highlights on wet skin and deep shadows on the right. "
        "The skin is covered in glistening water droplets."
    )
    
    style_prompt = (
        "Apply an artistic 'Earthy Ochre & Clay' style: use natural ochre and clay pigments, "
        "a textured earthy background, and warm organic tones. "
        "The result must be a photographic transformation that keeps the EXACT structure, pose, "
        "and identity described above. Maintain the woman's specific curves and the intimate bathtub setting."
    )
    
    full_prompt = f"Professional fine art photography. Blueprint: {blueprint}. Style: {style_prompt}."
    
    print(f"Retrying V4 (Blueprint): {photo_name}")
    result = call_imagen(full_prompt)
    
    if isinstance(result, bytes):
        filename = f"{photo_name}_RETRY_v4_Blueprint.png"
        save_path = os.path.join(daily_output_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(result)
        print(f"SUCCESS: {save_path}")
    else:
        print(result)

if __name__ == "__main__":
    main()
