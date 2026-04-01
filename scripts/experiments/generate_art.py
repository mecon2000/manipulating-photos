from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import numpy as np

def create_paper_tear_ruby(top_path, bottom_path, output_path):
    top = Image.open(top_path).convert('RGBA')
    bottom = Image.open(bottom_path).convert('RGBA')
    bottom_bw = ImageOps.grayscale(bottom).convert('RGBA')
    width, height = bottom_bw.size
    bottom_bw = bottom_bw.transform(bottom_bw.size, Image.AFFINE, (1, 0, 0, 0, 1, 8))
    draw_bottom = ImageDraw.Draw(bottom_bw)
    draw_bottom.rectangle([0, 0, width // 4, height // 4], fill=(0, 0, 0, 255))
    mask = Image.new('L', top.size, 255)
    draw_mask = ImageDraw.Draw(mask)
    eye_y_center = height * 0.38
    eye_x_start = width * 0.12
    eye_x_end = width * 0.88
    num_steps = 100
    points_top = []
    points_bottom = []
    current_width = width * 0.13
    for i in range(num_steps + 1):
        x = eye_x_start + (eye_x_end - eye_x_start) * (i / num_steps)
        progress = i / num_steps
        w = current_width * (1 - progress * 0.5)
        y_top = eye_y_center - w/2 + random.uniform(-10, 10)
        y_bottom = eye_y_center + w/2 + random.uniform(-10, 10)
        points_top.append((x, y_top))
        points_bottom.append((x, y_bottom))
    tear_poly = points_top + points_bottom[::-1]
    draw_mask.polygon(tear_poly, fill=0)
    mask_blur = mask.filter(ImageFilter.GaussianBlur(radius=0.5))
    final = Image.composite(top, bottom_bw, mask_blur).convert('RGB')
    final = final.crop((0, 0, width, int(height * 0.65)))
    final.save(output_path, 'JPEG', quality=95)
    print(f'Saved to {output_path}')

create_paper_tear_ruby('working_ruby/ref1.jpg', 'working_ruby/ref2.jpg', 'working_ruby/ruby_tear_v7.jpg')