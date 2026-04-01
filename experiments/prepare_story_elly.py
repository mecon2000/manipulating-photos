from PIL import Image
import os

def process():
    input_path = 'working/BLD_5552.jpg'
    # Download file first
    os.system('rclone copy "gdrive:_Photos/Elly (Eleanora)/01 hotel/Processed/BLD_5552.jpg" working/')
    
    img = Image.open(input_path)
    w, h = img.size
    
    # 1. Portrait Crop (9:16)
    target_ratio = 9/16
    current_ratio = w/h
    
    # Center crop for 9:16
    new_w = h * target_ratio
    left = (w - new_w) / 2
    right = (w + new_w) / 2
    img_portrait = img.crop((left, 0, right, h))
    img_portrait.save('working/story_elly_portrait.jpg', quality=85)
    
    # 2. Teaser/Abstract Crop (Extreme isolation)
    # Focus on the shoulder/light area if possible, or just a tight crop
    # Based on general boudoir window light, usually top-third is interesting
    teaser_w = w * 0.4
    teaser_h = teaser_w * (16/9)
    # Just take a tight vertical slice from the center-top
    t_left = (w - teaser_w) / 2
    t_top = h * 0.1
    t_right = t_left + teaser_w
    t_bottom = t_top + teaser_h
    img_teaser = img.crop((t_left, t_top, t_right, t_bottom))
    img_teaser.save('working/story_elly_teaser.jpg', quality=85)

    # Upload to GDrive
    os.system('rclone copy working/story_elly_portrait.jpg "gdrive:_photos from openclaw/Story Assets/"')
    os.system('rclone copy working/story_elly_teaser.jpg "gdrive:_photos from openclaw/Story Assets/"')

if __name__ == "__main__":
    if not os.path.exists('working'): os.makedirs('working')
    process()
