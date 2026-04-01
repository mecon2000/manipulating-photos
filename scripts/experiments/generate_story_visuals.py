from PIL import Image, ImageDraw, ImageFont
import os

def create_crops():
    img_path = 'agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg'
    img = Image.open(img_path)
    w, h = img.size

    # Abstract/Macro Crop - Tighter on hand/glass/light
    # Aiming for (530, 430, 680, 580) normalized
    left1 = 0.530 * w
    top1 = 0.430 * h
    right1 = 0.680 * w
    bottom1 = 0.580 * h
    crop1 = img.crop((left1, top1, right1, bottom1))
    crop1.save('agents/lux/working/story_poll_abstract.jpg')
    print(f"Saved refined poll abstract crop: {crop1.size}")

    # Teaser Crop - Reframed to vertical 9:16
    # Normalized: (390, 700, 600, 1000) for 9:16 feel centered on heel
    left2 = 0.390 * w
    top2 = 0.700 * h
    right2 = 0.600 * w
    bottom2 = 1.000 * h
    crop2 = img.crop((left2, top2, right2, bottom2))
    crop2.save('agents/lux/working/story_teaser_mood.jpg')
    print(f"Saved teaser mood crop (vertical): {crop2.size}")

def create_lighting_diagram():
    # 1080x1920
    w, h = 1080, 1920
    bg_color = (15, 15, 15) 
    img = Image.new('RGB', (w, h), bg_color)
    draw = ImageDraw.Draw(img, 'RGBA') # Support transparency for light

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 65)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 45)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    draw.text((w/2, 120), "The Silhouette Setup", fill=(255, 220, 100), font=font_title, anchor="mm")

    # Hallway / Walls
    draw.rectangle([150, 300, 200, 1600], fill=(50, 50, 50)) # Left wall
    draw.rectangle([880, 300, 930, 1600], fill=(50, 50, 50)) # Right wall
    
    # Window with Blinds
    window_rect = [350, 300, 730, 450]
    draw.rectangle(window_rect, fill=(255, 255, 245))
    # Draw horizontal blinds
    for y in range(310, 450, 20):
        draw.line([(350, y), (730, y)], fill=(60, 60, 60), width=4)
    
    draw.text((w/2, 260), "WINDOW (BACKLIGHT)", fill=(255, 255, 245), font=font_label, anchor="mm")

    # Light Cone
    light_poly = [(350, 450), (730, 450), (1080, 1400), (0, 1400)]
    draw.polygon(light_poly, fill=(255, 255, 200, 60)) # Increased opacity from 40 to 60
    
    # Light Rays (Striped/Shadows from blinds) - Brighter
    for offset in range(-200, 200, 60):
        draw.line([(350 + offset, 450), (w/2 + offset*4, 1500)], fill=(255, 255, 220, 40), width=18)

    # Model (Centered in path)
    model_pos = (w/2, 850)
    # Simple model icon (circle + shoulders)
    draw.ellipse([model_pos[0]-60, model_pos[1]-60, model_pos[0]+60, model_pos[1]+60], fill=(200, 0, 0, 200), outline=(255, 255, 255))
    draw.text((model_pos[0] + 100, model_pos[1]), "MODEL", fill=(255, 255, 255), font=font_label, anchor="lm")

    # Camera
    camera_pos = (w/2, 1600)
    draw.rectangle([camera_pos[0]-60, camera_pos[1]-40, camera_pos[0]+60, camera_pos[1]+40], fill=(80, 80, 80), outline=(255, 255, 255))
    draw.rectangle([camera_pos[0]-20, camera_pos[1]-70, camera_pos[0]+20, camera_pos[1]-40], fill=(80, 80, 80), outline=(255, 255, 255)) # Lens
    draw.text((camera_pos[0], camera_pos[1] + 100), "CAMERA", fill=(255, 255, 255), font=font_label, anchor="mm")

    # Settings & Tips (Filling dead space)
    draw.text((w/2, 1780), "TIPS: Expose for highlights to kill detail in shadows.", fill=(200, 200, 200), font=font_small, anchor="mm")
    # draw.text((w/2, 1830), "SETTINGS: f/2.8 | 1/200s | ISO 400", fill=(200, 200, 200), font=font_small, anchor="mm")

    img.save('agents/lux/working/story_lighting_diagram.jpg')
    print("Saved refined lighting diagram")

if __name__ == "__main__":
    create_crops()
    create_lighting_diagram()
