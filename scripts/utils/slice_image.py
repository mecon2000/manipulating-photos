from PIL import Image

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'

with Image.open(img_path) as img:
    w, h = img.size
    # Slicing the bottom part into 3 strips
    bottom_part = img.crop((0, int(h * 0.5), w, h))
    strip_w = w // 3
    
    strip1 = bottom_part.crop((0, 0, strip_w, bottom_part.height))
    strip2 = bottom_part.crop((strip_w, 0, strip_w * 2, bottom_part.height))
    strip3 = bottom_part.crop((strip_w * 2, 0, w, bottom_part.height))
    
    strip1.save('strip1.jpg', quality=95)
    strip2.save('strip2.jpg', quality=95)
    strip3.save('strip3.jpg', quality=95)
