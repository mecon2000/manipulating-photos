import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SOURCE_PATH = "work/sources/Processed/BLD_1654E.jpg"
OUTPUT_DIR = "outputs/michaela_workflow/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_mask_and_layers():
    img = Image.open(SOURCE_PATH).convert("RGB")
    width, height = img.size
    
    # We don't have rembg, so let's use a simpler approach:
    # Thresholding the red channel since she's in red lingerie.
    r, g, b = img.split()
    
    # In red lingerie, the red channel should be dominant.
    # We want (Red - Green) > threshold or (Red - Blue) > threshold
    diff_rg = ImageOps.invert(ImageOps.equalize(r)).point(lambda x: 255 if x < 100 else 0)
    # Actually, a simple threshold on the Red channel might work better if the BG is dark.
    mask = r.point(lambda x: 255 if x > 60 else 0)
    
    # Clean up the mask: Median filter and small blur
    mask = mask.filter(ImageFilter.MedianFilter(size=11))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, "michaela_mask.png")
    mask.save(mask_path)
    
    # Layer 1: Background (model removed/blurred)
    # We'll just blur the model area heavily to "remove" her for the BG-only pass.
    img_bg = img.copy()
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=50))
    img_bg.paste(img_blurred, mask=mask)
    bg_path = os.path.join(OUTPUT_DIR, "bg_only.jpg")
    img_bg.save(bg_path, "JPEG", quality=95)
    
    # Layer 2: Model only (on black)
    img_model = Image.new("RGB", (width, height), (0, 0, 0))
    img_model.paste(img, mask=mask)
    model_path = os.path.join(OUTPUT_DIR, "model_only.jpg")
    img_model.save(model_path, "JPEG", quality=95)
    
    print(f"Created mask and layers in: {OUTPUT_DIR}")

if __name__ == '__main__':
    create_mask_and_layers()
