from PIL import Image, ImageOps

img_path = './agents/lux/BLD_4183.jpg'
out_path = './agents/lux/working/Daniella_BLD_4183_ig.jpg'

img = Image.open(img_path)
print(f"Original size: {img.size}")

# We want 4:5 aspect ratio.
# Since it's portrait 2160x3240, its aspect ratio is 2160/3240 = 0.666 (2:3)
# 4:5 is 0.8. So we need to pad the width.
target_width = int(img.height * 0.8)
target_height = img.height

# Let's pad it with white or black? Since it's a silhouette, black is probably best, or white.
# Let's use white for Instagram. 
new_img = Image.new("RGB", (target_width, target_height), "white")

# Calculate offset
offset_x = (target_width - img.width) // 2
offset_y = (target_height - img.height) // 2

new_img.paste(img, (offset_x, offset_y))

new_img.save(out_path, quality=95)
print(f"Saved to {out_path} with size {new_img.size}")
