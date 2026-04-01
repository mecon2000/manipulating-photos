from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'

with Image.open(img_path) as img:
    # Recommended crop: (780, 1747, 1620, 3240)
    left = 780
    top = 1747
    right = 1620
    bottom = 3240
    
    print(f"Cropping to: ({left}, {top}, {right}, {bottom})")
    cropped = img.crop((left, top, right, bottom))
    
    # Optional: slight contrast/brightness adjustment if too dark?
    # But user wants "moody", so let's stick to the original look first.
    
    cropped.save(out_path, quality=95)
