import sys

new_func = """
def run_tensor_inpaint(
    image_pil,
    mask_pil,
    output_dir,
    dilation_px=9,
    margin_ratio=0.1,
    max_retries=3,
    timeout=60,
):
    def log(msg):
        log_to_file(output_dir, msg)

    # 1. HARDEN MASK
    mask = mask_pil.convert("L")
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    k = dilation_px if dilation_px % 2 != 0 else dilation_px + 1
    mask = mask.filter(ImageFilter.MaxFilter(k))
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    mask.save(os.path.join(output_dir, "mask_final.png"))

    # 2. ROI
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return image_pil

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    margin = int(max(image_pil.size) * margin_ratio)

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image_pil.width, x_max + margin)
    y_max = min(image_pil.height, y_max + margin)

    image_crop = image_pil.crop((x_min, y_min, x_max, y_max))
    mask_crop = mask.crop((x_min, y_min, x_max, y_max))

    # 3. UPLOAD RESOURCES (Proper way for Tensor Cloud API)
    log("Uploading ROI and Mask to Tensor Art...")
    img_res_id, cw, ch = upload_to_tensor(image_crop, output_dir)
    mask_res_id, mw, mh = upload_to_tensor(mask_crop, output_dir)

    # 4. REQUEST
    MODEL_INPAINT = "845475299014578498" # https://tensor.art/models/845475299014578498
    payload = {
        "requestId": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": { "image_resource_id": img_res_id, "count": 1, "seed": 42 }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": cw, "height": ch,
                    "prompts": [{
                        "text": "empty background, no person, natural scene, preserve original environment, consistent perspective, realistic textures",
                        "weight": 1
                    }],
                    "negative_prompts": [{
                        "text": "person, human, body, skin, face, clothes, blur, artifacts",
                        "weight": 1
                    }],
                    "sdModel": MODEL_INPAINT,
                    "steps": 28, "cfgScale": 6.5, "denoisingStrength": 0.75, "sampler": "Euler a",
                    "mask_resource_id": mask_res_id
                }
            }
        ]
    }

    # 5. EXECUTION
    for attempt in range(max_retries):
        log(f"--- Inpaint Attempt {attempt+1} ---")
        img_url = run_tensor_job(payload, output_dir)
        if img_url:
            result_crop = Image.open(requests.get(img_url, stream=True, timeout=timeout).raw).convert("RGB")
            # Ensure resizing back to ROI size in case Tensor adjusted it
            result_crop = result_crop.resize((x_max - x_min, y_max - y_min), Image.LANCZOS)
            final = image_pil.copy()
            final.paste(result_crop, (x_min, y_min))
            return final

    return None
"""

with open("scripts/workflows/tensor_photo_workflow.py", "r") as f:
    content = f.read()

import re
pattern = r"def run_tensor_inpaint\(.*?\n    return None"
content = re.sub(pattern, new_func, content, flags=re.DOTALL)

with open("scripts/workflows/tensor_photo_workflow.py", "w") as f:
    f.write(content)
