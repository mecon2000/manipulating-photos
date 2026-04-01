from PIL import Image
import os

def create_crops():
    input_path = 'working/BLD_5552.jpg'
    if not os.path.exists(input_path):
        os.system('rclone copy "gdrive:_Photos/Elly (Eleanora)/01 hotel/Processed/BLD_5552.jpg" working/')
    
    img = Image.open(input_path)
    w, h = img.size
    
    crops = [
        # Option 1: Broad Portrait (Classic)
        {'name': '1_broad', 'w_p': 0.6, 'top_p': 0.1, 'left_p': 0.2},
        # Option 2: Moody Medium (Shoulders/Neck)
        {'name': '2_medium', 'w_p': 0.35, 'top_p': 0.15, 'left_p': 0.35},
        # Option 3: Extreme Macro (Abstract light/skin)
        {'name': '3_macro', 'w_p': 0.18, 'top_p': 0.2, 'left_p': 0.45}
    ]
    
    for c in crops:
        cw = w * c['w_p']
        ch = cw * (16/9)
        left = w * c['left_p']
        top = h * c['top_p']
        img_crop = img.crop((left, top, left + cw, top + ch))
        out_path = f"working/story_elly_option_{c['name']}.jpg"
        img_crop.save(out_path, quality=90)
        os.system(f'rclone copy "{out_path}" "gdrive:_photos from openclaw/Story Assets/Options/"')

if __name__ == "__main__":
    create_crops()
