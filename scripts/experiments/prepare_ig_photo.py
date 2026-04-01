from PIL import Image, ImageFilter
import sys
import os

def create_padded_image(input_path, output_path, target_ratio=(4, 5)):
    with Image.open(input_path) as img:
        w, h = img.size
        target_w_ratio, target_h_ratio = target_ratio
        
        # Determine target dimensions
        if w / h > target_w_ratio / target_h_ratio:
            # Image is wider than target ratio
            new_h = int(w * target_h_ratio / target_w_ratio)
            new_w = w
        else:
            # Image is taller than target ratio
            new_w = int(h * target_w_ratio / target_h_ratio)
            new_h = h
            
        # Create background (blurred side strips)
        background = Image.new("RGB", (new_w, new_h))
        
        if new_w > w:
            # Padding horizontally
            pad_width = (new_w - w) // 2
            
            # Left strip
            left_strip = img.crop((0, 0, min(pad_width, w), h))
            left_strip = left_strip.resize((pad_width, new_h), Image.Resampling.LANCZOS)
            left_strip = left_strip.filter(ImageFilter.GaussianBlur(radius=50))
            background.paste(left_strip, (0, 0))
            
            # Right strip
            right_strip = img.crop((max(0, w - pad_width), 0, w, h))
            right_strip = right_strip.resize((pad_width, new_h), Image.Resampling.LANCZOS)
            right_strip = right_strip.filter(ImageFilter.GaussianBlur(radius=50))
            background.paste(right_strip, (new_w - pad_width, 0))
            
            # Paste original in center
            background.paste(img, (pad_width, 0))
        else:
            # Padding vertically (less common for IG portrait)
            pad_height = (new_h - h) // 2
            background.paste(img, (0, pad_height))

        # Save with quality 80 as requested in LUX state
        background.save(output_path, "JPEG", quality=80)
        print(f"Saved padded image to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 prepare_ig_photo.py <input> <output>")
    else:
        create_padded_image(sys.argv[1], sys.argv[2])
