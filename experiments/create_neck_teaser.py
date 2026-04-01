from PIL import Image
import os

def create_teaser():
    input_path = 'working/BLD_5552.jpg'
    if not os.path.exists(input_path):
        os.system('rclone copy "gdrive:_Photos/Elly (Eleanora)/01 hotel/Processed/BLD_5552.jpg" working/')
    
    img = Image.open(input_path)
    w, h = img.size
    
    # Ultra Macro crop: Neck area
    # Based on general boudoir portraits, neck/shoulder is usually center-top.
    teaser_w = w * 0.15 # Very tight, 15%
    teaser_h = teaser_w * (16/9)
    
    left = w * 0.45
    top = h * 0.15
    right = left + teaser_w
    bottom = top + teaser_h
    
    img_macro = img.crop((left, top, right, bottom))
    img_macro.save('working/story_elly_ultra_macro_neck.jpg', quality=95)
    
    os.system('rclone copy working/story_elly_ultra_macro_neck.jpg "gdrive:_photos from openclaw/Story Assets/"')
    print("Ultra macro neck teaser uploaded.")

if __name__ == "__main__":
    create_teaser()
