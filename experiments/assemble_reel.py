from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip
import os
from PIL import Image
# Fix for MoviePy compatibility with PIL 10+
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

def create_reel():
    input_dir = 'agents/lux/working/reel_crops_final/'
    output_path = 'agents/lux/working/Shadow_and_Soul_Reel_Preview.mp4'
    
    # 1. Sequence order
    # I want to start and end strong.
    # Start with Liel (Body Paint) or Miki (Window) - strong portraits.
    # End with Daniella silhouette (the "Face" of the account today) or Jenia lights.
    
    # Actually, Ronnie chose 7 shots. Let's sequence them for mood:
    files = [
        "Liel_BodyPaint_reel.jpg",       # Strong artistic opener
        "Valeria_Bath_reel.jpg",        # High cinematic impact
        "Noga_Candles_2_reel.jpg",      # Intimate light
        "Ruby_Rimlight_reel.jpg",       # Minimalist shadow
        "Miki_Boudoir_rotated_reel.jpg",# Abstract/Texture
        "Miki_Window_reel.jpg",         # Classic portraiture
        "Jenia_Lights_2_reel.jpg"       # Warm, soulful closer
    ]
    
    clips = []
    
    # Duration per shot (approx 2.1s for a ~15s reel total)
    duration = 2.1
    
    for f in files:
        path = os.path.join(input_dir, f)
        if os.path.exists(path):
            img_clip = ImageClip(path).set_duration(duration)
            # FORCE RESIZE to 1080x1920 to ensure no black bars
            img_clip = img_clip.resize(newsize=(1080, 1920))
            clips.append(img_clip)
        else:
            print(f"Warning: {path} not found")

    if not clips:
        print("No clips found to assemble.")
        return

    # 2. Assemble
    video = concatenate_videoclips(clips, method="compose")
    
    # 3. Export
    # Use libx264 for high compatibility
    # No audio (Ronnie will add on IG for algorithm boost)
    video.write_videofile(output_path, fps=30, codec="libx264")
    print(f"Reel assembled at {output_path}")

if __name__ == "__main__":
    create_reel()
