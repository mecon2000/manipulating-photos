from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'

with Image.open(img_path) as img:
    # Target height 1037px (approx 32% of 3240, from 68% to 100%)
    target_h = 1037
    target_w = int(target_h * 9 / 16) # 583.3 -> 583
    
    # Left edge around 40.2%
    left = int(img.width * 0.402)
    right = left + target_w
    top = int(img.height * 0.68)
    bottom = top + target_h
    
    # Clamp to image boundaries
    left = max(0, min(left, img.width - target_w))
    right = left + target_w
    top = max(0, min(top, img.height - target_h))
    bottom = top + target_h
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    cropped.save(out_path, quality=95)
