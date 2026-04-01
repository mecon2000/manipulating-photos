from PIL import Image, ImageEnhance

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'
check_path = 'agents/lux/working/story_teaser_check.jpg'

with Image.open(img_path) as img:
    # Target height 1100px (approx 34% of 3240, from 2140 to 3240)
    target_h = 1100
    target_w = int(target_h * 9 / 16) # 618.75 -> 619
    
    # Left edge around 41.5%
    left = int(img.width * 0.415)
    right = left + target_w
    top = 3240 - target_h
    bottom = 3240
    
    # Clamp to image boundaries
    left = max(0, min(left, img.width - target_w))
    right = left + target_w
    top = max(0, min(top, img.height - target_h))
    bottom = top + target_h
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    
    # Save the moody version
    cropped.save(out_path, quality=95)
    
    # Save a brightened version for the AI to check visibility
    enhancer = ImageEnhance.Brightness(cropped)
    brightened = enhancer.enhance(3.0) # Boost brightness by 3x
    brightened.save(check_path, quality=95)
