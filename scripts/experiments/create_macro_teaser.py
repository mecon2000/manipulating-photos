from PIL import Image
import os

def create_teaser():
    # Use the main image which is already in working/
    input_path = 'working/BLD_5552.jpg'
    if not os.path.exists(input_path):
        os.system('rclone copy "gdrive:_Photos/Elly (Eleanora)/01 hotel/Processed/BLD_5552.jpg" working/')
    
    img = Image.open(input_path)
    w, h = img.size
    
    # Macro crop: Focus on a shoulder/neck area with light
    # Based on the typical orientation, let's take a 9:16 slice of a small area
    teaser_w = w * 0.25 # Only 25% of the width
    teaser_h = teaser_w * (16/9)
    
    # Try to hit the "sweet spot" of light/shadow (usually upper middle)
    left = w * 0.4
    top = h * 0.2
    right = left + teaser_w
    bottom = top + teaser_h
    
    img_macro = img.crop((left, top, right, bottom))
    img_macro.save('working/story_elly_macro_shoulder.jpg', quality=90)
    
    # Upload
    os.system('rclone copy working/story_elly_macro_shoulder.jpg "gdrive:_photos from openclaw/Story Assets/"')
    print("Macro teaser uploaded.")

if __name__ == "__main__":
    create_teaser()
