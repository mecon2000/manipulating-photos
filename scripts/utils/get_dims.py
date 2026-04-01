from PIL import Image
import os

img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
with Image.open(img_path) as img:
    width, height = img.size
    print(f"Dimensions: {width}x{height}")
