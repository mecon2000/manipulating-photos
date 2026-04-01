from PIL import Image
import sys

def create_teaser(input_path, output_path):
    with Image.open(input_path) as img:
        w, h = img.size
        # Bottom 35% crop for "mood" teaser
        target_h = int(h * 0.35)
        target_w = int(target_h * 9 / 16)
        center_x = int(w * 0.55)
        
        left = max(0, min(center_x - (target_w // 2), w - target_w))
        top = h - target_h
        right = left + target_w
        bottom = h
        
        img.crop((left, top, right, bottom)).save(output_path, "JPEG", quality=80)
        print(f"Teaser saved to {output_path}")

if __name__ == "__main__":
    create_teaser(sys.argv[1], sys.argv[2])
