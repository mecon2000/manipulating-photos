import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import PIL.ImageFilter
import PIL.ImageChops
import random
import os
import numpy as np

def create_paper_tear_v7(top_path, bottom_path, output_path):
    # Load images
    top = PIL.Image.open(top_path).convert("RGBA")
    bottom = PIL.Image.open(bottom_path).convert("RGBA")
    
    # Resize bottom to match top
    bottom = bottom.resize(top.size, PIL.Image.Resampling.LANCZOS)
    
    width, height = top.size
    print(f"Processing {width}x{height}")
    
    # 1. Generate ORGANIC Tear Path (Flowing macro curves)
    eye_y = height * 0.4
    tear_height = height * 0.16
    num_points = 800
    
    def generate_flowing_edge(y_center, amplitude, freq, micro_j):
        edge = []
        for i in range(num_points + 1):
            x = (width * i) / num_points
            y_macro = y_center + amplitude * (
                0.6 * np.sin(2 * np.pi * x / width) + 
                0.3 * np.sin(4 * np.pi * x / width + 1.5) +
                0.1 * np.sin(10 * np.pi * x / width)
            )
            y_micro = random.uniform(-micro_j, micro_j)
            edge.append((x, y_macro + y_micro))
        return edge

    # Scale amplitudes for high-res
    amp_scale = height / 1000.0
    top_edge = generate_flowing_edge(eye_y - tear_height/3, 40 * amp_scale, 1.0, 3 * amp_scale)
    
    bottom_edge = []
    for i in range(num_points, -1, -1):
        x = (width * i) / num_points
        narrow_factor = (x / width)
        y_center = eye_y + (tear_height * 0.8) * narrow_factor
        y_macro = y_center + 30 * amp_scale * (
            0.5 * np.sin(2 * np.pi * x / width + 0.5) +
            0.4 * np.cos(3 * np.pi * x / width)
        )
        y_micro = random.uniform(-2.5 * amp_scale, 2.5 * amp_scale)
        bottom_edge.append((x, y_macro + y_micro))
        
    tear_points = top_edge + bottom_edge
    
    # 2. Masks
    mask_hole = PIL.Image.new("L", (width, height), 0)
    draw_hole = PIL.ImageDraw.Draw(mask_hole)
    draw_hole.polygon(tear_points, fill=255)
    
    mask_top = PIL.ImageOps.invert(mask_hole)
    
    # 3. DIRECTIONAL SHADOW
    shadow_mask = PIL.Image.new("L", (width, height), 0)
    s_draw = PIL.ImageDraw.Draw(shadow_mask)
    s_draw.polygon(tear_points, fill=255)
    
    offset_y = int(15 * amp_scale)
    shadow_mask_offset = PIL.Image.new("L", (width, height), 0)
    shadow_mask_offset.paste(shadow_mask, (0, offset_y))
    shadow_mask_blur = shadow_mask_offset.filter(PIL.ImageFilter.GaussianBlur(radius=15 * amp_scale))
    shadow_final_mask = PIL.ImageChops.multiply(shadow_mask_blur, mask_hole)
    
    shadow_layer = PIL.Image.new("RGBA", (width, height), (0, 0, 0, 200))
    
    # 4. FIBROUS WHITE EDGE
    white_fiber_layer = PIL.Image.new("RGBA", (width, height), (0,0,0,0))
    wf_draw = PIL.ImageDraw.Draw(white_fiber_layer)
    
    for offset in range(-int(2*amp_scale), int(6*amp_scale)):
        f_pts = []
        for p in top_edge: f_pts.append((p[0], p[1] + offset))
        for p in bottom_edge: f_pts.append((p[0], p[1] - offset))
        wf_draw.polygon(f_pts, fill=(255, 255, 255, 255))
        
    wf_draw.polygon(tear_points, fill=(0,0,0,0))
    
    for _ in range(int(12000 * amp_scale)):
        p = random.choice(tear_points)
        rx, ry = p[0] + random.uniform(-3*amp_scale, 3*amp_scale), p[1] + random.uniform(-8*amp_scale, 8*amp_scale)
        if 0 <= rx < width and 0 <= ry < height:
            wf_draw.point((rx, ry), fill=(255, 255, 255, 180))
            
    white_fiber_layer = white_fiber_layer.filter(PIL.ImageFilter.GaussianBlur(radius=1 * amp_scale))
    
    # 5. Composite
    result = bottom.copy()
    result = PIL.Image.alpha_composite(result, PIL.Image.composite(shadow_layer, PIL.Image.new("RGBA", (width, height), (0,0,0,0)), shadow_final_mask))
    result = PIL.Image.alpha_composite(result, white_fiber_layer)
    result = PIL.Image.composite(top, result, mask_top)
    
    # 6. Global Paper Texture
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    res_np = np.array(result.convert("RGB")).astype(np.int16)
    res_np = np.clip(res_np + noise, 0, 255).astype(np.uint8)
    result = PIL.Image.fromarray(res_np)
    
    # Resize for CICA: Longest dimension 2500px (to keep file size under control but quality high)
    # 2500px at 10" is 250dpi, which is good.
    max_dim = 2500
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        result = result.resize((int(width * scale), int(height * scale)), PIL.Image.Resampling.LANCZOS)

    # Save as JPG with quality that fits 100-500 KB
    quality = 85
    result.save(output_path, "JPEG", quality=quality, optimize=True)
    while os.path.getsize(output_path) > 500 * 1024 and quality > 30:
        quality -= 5
        result.save(output_path, "JPEG", quality=quality, optimize=True)
    
    print(f"Saved {output_path}, size: {os.path.getsize(output_path)/1024:.1f} KB, quality: {quality}")

tasks = [
    {"top": "working_art/originals/Danielle_1.jpg", "output": "working_art/cica_submission/Danielle_01_Form.jpg"},
    {"top": "working_art/originals/Danielle_2.jpg", "output": "working_art/cica_submission/Danielle_02_Form.jpg"},
    {"top": "working_art/originals/Jenia.jpg", "output": "working_art/cica_submission/Jenia_01_Form.jpg"}
]

for task in tasks:
    img = PIL.Image.open(task["top"])
    bw = PIL.ImageOps.grayscale(img).convert("RGBA")
    temp_bw = task["output"] + ".bw.png"
    bw.save(temp_bw)
    create_paper_tear_v7(task["top"], temp_bw, task["output"])
    os.remove(temp_bw)
