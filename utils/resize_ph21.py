import os
from PIL import Image

def resize_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        
        filepath = os.path.join(input_dir, filename)
        try:
            with Image.open(filepath) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                
                if height > width: # Portrait
                    target_height = 775
                    ratio = target_height / float(height)
                    target_width = int(width * ratio)
                else: # Landscape or Square
                    target_width = 1280 if width > 1280 else width
                    ratio = target_width / float(width)
                    target_height = int(height * ratio)
                    
                    if target_height > 1280:
                        target_height = 1280
                        ratio = target_height / float(height)
                        target_width = int(width * ratio)

                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                output_path = os.path.join(output_dir, filename)
                img.save(output_path, "JPEG", quality=90, dpi=(72, 72))
                print(f"Resized {filename} to {target_width}x{target_height}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    resize_images("ph21_submission_files", "ph21_submission_resized")
