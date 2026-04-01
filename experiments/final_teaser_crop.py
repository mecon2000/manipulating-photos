from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
out_path = 'agents/lux/working/story_teaser_mood.jpg'

with Image.open(img_path) as img:
    # Wider Tight Crop:
    # X: 700 to 1700 (width 1000)
    # Y: 1463 to 3240 (height 1777)
    left = 700
    top = 1463
    right = 1700
    bottom = 3240
    
    cropped = img.crop((left, top, right, bottom))
    cropped.save(out_path, quality=95)
