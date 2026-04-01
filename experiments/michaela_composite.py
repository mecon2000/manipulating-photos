from PIL import Image, ImageDraw, ImageFilter

path_085 = "outputs/michaela_dual_strength/Michaela_Strength_0.85.jpg"
path_045 = "outputs/michaela_dual_strength/Michaela_Strength_0.45.jpg"
output_path = "outputs/michaela_dual_strength/Michaela_Composite_OptionA.jpg"

def composite():
    img_bg = Image.open(path_085).convert("RGB")
    img_model = Image.open(path_045).convert("RGB")
    width, height = img_bg.size
    
    # Create mask: White in center (keep model), Black at edges (keep BG)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw an oval for the model (centered)
    # Michaela is back shot, let's assume she covers the middle 50%
    left = int(width * 0.25)
    top = int(height * 0.05)
    right = int(width * 0.75)
    bottom = int(height * 0.95)
    draw.ellipse([left, top, right, bottom], fill=255)
    
    # Blur the mask to get a soft transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=50))
    
    # Composite
    # img_bg (0.85 strength) is the background.
    # img_model (0.45 strength) is the model.
    # composite: img_bg * (1-mask) + img_model * mask
    final = Image.composite(img_model, img_bg, mask)
    final.save(output_path, "JPEG", quality=95)
    print(f"Saved composite to: {output_path}")

if __name__ == '__main__':
    composite()
