from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'

with Image.open(img_path) as img:
    w, h = img.size
    
    # Target height 1685px (approx 52% of 3240, from 48% to 100%)
    target_h = 1685
    target_w = int(target_h * 9 / 16) # 947.8 -> 948
    
    # Left edge around 32.7%
    left = int(w * 0.327)
    right = left + target_w
    top = int(h * 0.48)
    bottom = top + target_h
    
    # Clamp to image boundaries
    left = max(0, min(left, w - target_w))
    right = left + target_w
    top = max(0, min(top, h - target_h))
    bottom = top + target_h
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    cropped.save(out_path, quality=95)
