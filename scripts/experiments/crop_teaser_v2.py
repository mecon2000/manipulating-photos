from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'

with Image.open(img_path) as img:
    w, h = img.size
    
    # Target height 1300px (approx 40% of 3240)
    target_h = 1300
    target_w = int(target_h * 9 / 16) # 731.25 -> 731
    
    # Left edge around 34.5% to capture both feet
    left = int(w * 0.345)
    right = left + target_w
    top = h - target_h
    bottom = h
    
    # Clamp to image boundaries
    left = max(0, min(left, w - target_w))
    right = left + target_w
    top = max(0, min(top, h - target_h))
    bottom = top + target_h
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    # Using high quality for moody teaser
    cropped.save(out_path, quality=95)
