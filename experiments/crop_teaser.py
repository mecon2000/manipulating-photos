from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'

with Image.open(img_path) as img:
    w, h = img.size
    
    # Target height (bottom 30%)
    target_h = int(h * 0.35) # from 0.65 to 1.0
    target_w = int(target_h * 9 / 16)
    
    # Center X around 55%
    center_x = int(w * 0.55)
    
    left = center_x - (target_w // 2)
    right = left + target_w
    top = int(h * 0.65)
    bottom = h
    
    # Clamp to image boundaries
    left = max(0, min(left, w - target_w))
    right = left + target_w
    top = max(0, min(top, h - target_h))
    bottom = top + target_h
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    cropped.save(out_path)
