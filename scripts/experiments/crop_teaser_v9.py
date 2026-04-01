from PIL import Image, ImageEnhance

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'
check_path = 'agents/lux/working/story_teaser_check.jpg'

with Image.open(img_path) as img:
    # Based on (x1, y1, x2, y2) = (580, 2580, 1150, 3140) in 1822x3240 crop
    # which has Left = 385 offset.
    
    # Target width 570px, Height 1013px for 9:16
    left = 385 + 580
    right = 385 + 1150
    bottom = 3140
    top = bottom - 1013
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    
    # Save the moody version
    cropped.save(out_path, quality=95)
    
    # Save a brightened version for the AI to check visibility
    enhancer = ImageEnhance.Brightness(cropped)
    brightened = enhancer.enhance(3.0) # Boost brightness by 3x
    brightened.save(check_path, quality=95)
