#!/home/rong/openclaw-venv/bin/python3
"""Quick baroque experiment: scene composition + generated BG approaches."""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

env_file = os.path.expanduser("~/sol/.env")
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client, requests, numpy as np
from PIL import Image, ImageFilter
from masking import build_mask
from scipy import ndimage
from io import BytesIO

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")
os.makedirs(FINALS, exist_ok=True)

def push(path, title, body=""):
    try:
        from notify import push_image
        push_image(path, title=title, body=body)
        print(f"  Pushed: {title}")
    except Exception as e:
        print(f"  Push failed: {e}")


def experiment_scene_composition(src_path, prompt, name_tag):
    """Use fal scene-composition to place subject into generated scene."""
    print(f"\n=== Scene Composition: {name_tag} ===")
    url = fal_client.upload_file(src_path)
    handle = fal_client.submit("fal-ai/image-editing/scene-composition", arguments={
        "image_url": url,
        "prompt": prompt,
        "guidance_scale": 3.5,
        "num_inference_steps": 30,
        "aspect_ratio": "9:16",
        "safety_tolerance": "6",
        "output_format": "jpeg",
    })
    result = handle.get()
    images = result.get("images", [])
    if not images:
        print("  No images returned!")
        return None
    img_url = images[0].get("url", "")
    resp = requests.get(img_url, timeout=60)
    img = Image.open(BytesIO(resp.content))
    out = os.path.join(FINALS, f"baroque_scene_{name_tag}.jpg")
    img.save(out, quality=95)
    print(f"  Saved: {out}")
    push(out, f"Scene Comp — {name_tag}", prompt[:60])
    return img


def experiment_generated_bg(src_path, bg_prompt, name_tag):
    """Generate BG from scratch, then composite subject onto it."""
    print(f"\n=== Generated BG: {name_tag} ===")
    img_orig = Image.open(src_path).convert("RGB")
    w, h = img_orig.size
    short_edge = min(w, h)

    # Extract mask
    print("  Extracting mask...")
    mask, info = build_mask(img_orig, affect="subject", exclude="", output_dir="/tmp", feather=0)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.LANCZOS)
    print(f"  Mask coverage: {info['coverage_pct']}%")

    # Generate BG
    print("  Generating BG...")
    handle = fal_client.submit("fal-ai/flux/dev", arguments={
        "prompt": bg_prompt,
        "image_size": {"width": w, "height": h},
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    })
    result = handle.get()
    bg_url = result["images"][0]["url"]
    resp = requests.get(bg_url, timeout=60)
    bg_img = Image.open(BytesIO(resp.content)).convert("RGB")
    if bg_img.size != (w, h):
        bg_img = bg_img.resize((w, h), Image.LANCZOS)

    # Composite with bleed + light wrap
    print("  Compositing...")
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    struct = ndimage.generate_binary_structure(2, 1)

    # Feather
    feather_px = max(3, int(short_edge * 0.04))
    mask_feathered = Image.fromarray((mask_arr * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=feather_px))
    mask_f = np.array(mask_feathered).astype(np.float32) / 255.0

    # Bleed zone: bottom-weighted, noisy
    bleed_depth = max(3, int(short_edge * 0.03))
    mask_binary = (mask_arr > 0.5).astype(np.uint8)
    mask_eroded = ndimage.binary_erosion(mask_binary, structure=struct, iterations=bleed_depth)
    bleed_zone = np.clip(mask_binary.astype(np.float32) - mask_eroded.astype(np.float32), 0, 1)
    yy = np.linspace(0, 1, h)[:, np.newaxis]
    bleed_zone *= np.clip(yy * 1.5 - 0.3, 0, 1)
    rng = np.random.RandomState(42)
    noise = ndimage.gaussian_filter(rng.randn(h, w), sigma=max(10, int(short_edge * 0.04)))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    bleed_zone *= noise
    bleed_zone = ndimage.gaussian_filter(bleed_zone, sigma=max(2, int(short_edge * 0.01)))
    composite_mask = mask_f * (1.0 - np.clip(bleed_zone, 0, 1) * 0.6)

    # Color harmonization
    orig_arr = np.array(img_orig).astype(np.float32)
    bg_arr = np.array(bg_img).astype(np.float32)
    bg_zone = composite_mask < 0.4
    subj_zone = composite_mask > 0.6
    if bg_zone.any() and subj_zone.any():
        bg_mean = bg_arr[bg_zone].mean(axis=0)
        subj_mean = orig_arr[subj_zone].mean(axis=0)
        color_shift = (bg_mean - subj_mean) * 0.15
        edge_w = np.clip((0.7 - composite_mask) / 0.4, 0, 1)[:, :, np.newaxis]
        orig_arr = np.clip(orig_arr + color_shift * edge_w, 0, 255)

    # Light wrap
    wrap_radius = max(5, int(short_edge * 0.04))
    bg_blurred = bg_img.filter(ImageFilter.GaussianBlur(radius=wrap_radius))
    bg_bl_arr = np.array(bg_blurred).astype(np.float32)
    wrap_w = np.clip((0.85 - composite_mask) / 0.35, 0, 1) * np.clip(composite_mask / 0.5, 0, 1)
    wrap_w3 = (wrap_w * 0.25)[:, :, np.newaxis]
    orig_arr = np.clip(orig_arr * (1 - wrap_w3) + bg_bl_arr * wrap_w3, 0, 255)

    # Final composite
    mask_3ch = composite_mask[:, :, np.newaxis]
    comp = np.clip(orig_arr * mask_3ch + bg_arr * (1 - mask_3ch), 0, 255).astype(np.uint8)
    result_img = Image.fromarray(comp)

    out = os.path.join(FINALS, f"baroque_genbg_{name_tag}.jpg")
    result_img.save(out, quality=95)
    print(f"  Saved: {out}")
    push(out, f"Gen BG — {name_tag}", bg_prompt[:60])
    return result_img


def experiment_tensor_blur_stylize(src_path, style_prompt, strength, name_tag):
    """Blur BG heavily, run whole image through Tensor Art img2img, composite subject back."""
    print(f"\n=== Tensor Blur+Stylize: {name_tag} ===")
    img_orig = Image.open(src_path).convert("RGB")
    w, h = img_orig.size
    short_edge = min(w, h)

    # Extract mask
    print("  Extracting mask...")
    mask, info = build_mask(img_orig, affect="subject", exclude="", output_dir="/tmp", feather=0)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.LANCZOS)

    # Heavy blur on BG to destroy structure
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    orig_arr = np.array(img_orig).astype(np.float32)
    blur_r = max(40, int(short_edge * 0.15))
    blurred_full = img_orig.filter(ImageFilter.GaussianBlur(radius=blur_r))
    blurred_arr = np.array(blurred_full).astype(np.float32)

    # Blend: subject stays mostly sharp, BG is blurred
    # Use soft transition (10% of short edge)
    feather = max(5, int(short_edge * 0.10))
    mask_soft = Image.fromarray((mask_arr * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=feather))
    mask_soft_arr = np.array(mask_soft).astype(np.float32) / 255.0
    mask_soft_3ch = mask_soft_arr[:, :, np.newaxis]
    # Subject at ~30% blur (slight softening), BG at full blur
    subject_blur = 0.3
    blend = orig_arr * (1 - subject_blur) * mask_soft_3ch + blurred_arr * (1 - mask_soft_3ch * (1 - subject_blur))
    preblurred = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))
    preblurred.save(os.path.join(FINALS, f"baroque_tensor_preblur_{name_tag}.jpg"), quality=95)
    print(f"  Pre-blurred (radius={blur_r}px, subject softened {subject_blur:.0%})")

    # Upload to Tensor Art for img2img
    print("  Running Tensor Art img2img...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        preblurred.save(tmp, format="JPEG", quality=95)
        tmp_path = tmp.name

    tensor_url = fal_client.upload_file(tmp_path)
    os.unlink(tmp_path)

    handle = fal_client.submit("fal-ai/flux/dev/image-to-image", arguments={
        "image_url": tensor_url,
        "prompt": style_prompt,
        "strength": strength,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    })
    result = handle.get()
    styled_url = result["images"][0]["url"]
    resp = requests.get(styled_url, timeout=60)
    styled_img = Image.open(BytesIO(resp.content)).convert("RGB")
    if styled_img.size != (w, h):
        styled_img = styled_img.resize((w, h), Image.LANCZOS)
    print(f"  Styled: {styled_img.size}")

    # Composite: face stays original, body transitions to styled, BG fully styled
    styled_arr = np.array(styled_img).astype(np.float32)
    orig_arr = np.array(img_orig).astype(np.float32)

    # Face-focused gradient mask: face=1.0 (keep original), lower body=0.3, BG=0.0
    struct = ndimage.generate_binary_structure(2, 1)
    mask_binary = (mask_arr > 0.5).astype(np.uint8)

    # Feather for compositing
    comp_feather = max(3, int(short_edge * 0.04))
    comp_mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=comp_feather))
    comp_mask_arr = np.array(comp_mask).astype(np.float32) / 255.0

    # Vertical gradient: top of subject (face) → full protection, bottom → less
    yy = np.linspace(0, 1, h)[:, np.newaxis]
    face_protection = np.clip(1.0 - yy * 0.7, 0.3, 1.0)  # 1.0 at top, 0.3 at bottom
    # Only apply gradient within the subject
    final_mask = comp_mask_arr * face_protection.squeeze()[:, np.newaxis] * np.ones((1, w))
    # Wait, need to fix shape
    final_mask = comp_mask_arr * np.clip(1.0 - yy * 0.7, 0.3, 1.0)

    # Bleed zone (same as generated_bg approach)
    bleed_depth = max(3, int(short_edge * 0.025))
    mask_eroded = ndimage.binary_erosion(mask_binary, structure=struct, iterations=bleed_depth)
    bleed_zone = np.clip(mask_binary.astype(np.float32) - mask_eroded.astype(np.float32), 0, 1)
    bleed_zone *= np.clip(yy * 1.5 - 0.3, 0, 1)
    rng = np.random.RandomState(42)
    noise = ndimage.gaussian_filter(rng.randn(h, w), sigma=max(10, int(short_edge * 0.04)))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    bleed_zone *= noise
    bleed_zone = ndimage.gaussian_filter(bleed_zone, sigma=max(2, int(short_edge * 0.01)))
    final_mask = final_mask * (1.0 - np.clip(bleed_zone, 0, 1) * 0.5)

    # Light wrap from styled BG
    wrap_r = max(5, int(short_edge * 0.04))
    styled_blur = styled_img.filter(ImageFilter.GaussianBlur(radius=wrap_r))
    styled_bl_arr = np.array(styled_blur).astype(np.float32)
    wrap_w = np.clip((0.85 - final_mask) / 0.35, 0, 1) * np.clip(final_mask / 0.5, 0, 1)
    wrap_w3 = (wrap_w * 0.2)[:, :, np.newaxis]
    orig_arr = np.clip(orig_arr * (1 - wrap_w3) + styled_bl_arr * wrap_w3, 0, 255)

    # Composite
    fm3 = final_mask[:, :, np.newaxis]
    comp = np.clip(orig_arr * fm3 + styled_arr * (1 - fm3), 0, 255).astype(np.uint8)
    result_img = Image.fromarray(comp)

    out = os.path.join(FINALS, f"baroque_tensor_{name_tag}.jpg")
    result_img.save(out, quality=95)
    print(f"  Saved: {out}")
    push(out, f"Tensor Blur — {name_tag}", f"strength={strength}")
    return result_img


if __name__ == "__main__":
    src = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")

    # Generated BG experiments (skip scene composition — safety blocked)
    experiment_generated_bg(src,
        "large flowing amorphous organic shapes, swirling smoke plumes and billowing "
        "drapery forms, baroque oil painting, dramatic chiaroscuro, warm amber and cool "
        "grey and cream tones, Caravaggio inspired, abstract undulating masses, "
        "NO person NO figure, just abstract painterly forms",
        "Nastia_smoke")

    experiment_generated_bg(src,
        "ethereal luminous flowing abstract forms, soft pearl and ivory and pale gold "
        "organic shapes, dreamy angelic oil painting, billowing cloud-like masses, "
        "divine atmospheric radiance, NO person NO figure, just abstract light forms",
        "Nastia_ethereal")

    # Tensor Art blur+stylize experiments
    experiment_tensor_blur_stylize(src,
        "baroque oil painting, large flowing amorphous organic drapery forms, "
        "dramatic chiaroscuro, swirling smoke and fabric, warm amber cream and grey, "
        "Caravaggio Bouguereau masterpiece, smooth blended brushwork",
        0.65, "Nastia_baroque_065")

    experiment_tensor_blur_stylize(src,
        "baroque oil painting, large flowing amorphous organic drapery forms, "
        "dramatic chiaroscuro, swirling smoke and fabric, warm amber cream and grey, "
        "Caravaggio Bouguereau masterpiece, smooth blended brushwork",
        0.80, "Nastia_baroque_080")

    experiment_tensor_blur_stylize(src,
        "ethereal luminous flowing forms, dreamy angelic oil painting, soft pearl "
        "ivory gold mist, divine radiance, smooth oil glazing technique",
        0.70, "Nastia_ethereal_070")

    print("\n=== All experiments complete ===")
