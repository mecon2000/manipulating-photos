from PIL import Image, ImageEnhance

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'
check_path = 'agents/lux/working/story_teaser_check.jpg'

with Image.open(img_path) as img:
    # Based on x ~ 1300-1728, y ~ 2174-3240
    # Let's try X from 1200 to 1800 (width 600)
    # Height = 1066
    
    left = 1200
    right = 1800
    bottom = 3240
    top = bottom - 1066
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    
    # Save the moody version
    cropped.save(out_path, quality=95)
    
    # Save a brightened version for the AI to check visibility
    enhancer = ImageEnhance.Brightness(cropped)
    brightened = enhancer.enhance(3.0) # Boost brightness by 3x
    brightened.save(check_path, quality=95)
