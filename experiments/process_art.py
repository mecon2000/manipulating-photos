import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter

def cleanup_background(image):
    # Mask out the right side where the shelf/socket are
    w, h = image.size
    img_array = np.array(image).astype(np.float32)
    
    # Create a gradient mask for the right side (starting from 70% width)
    mask = np.ones((h, w), dtype=np.float32)
    start_x = int(w * 0.7)
    for x in range(start_x, w):
        factor = 1.0 - (x - start_x) / (w - start_x)
        mask[:, x] *= factor
    
    img_array *= mask[..., np.newaxis]
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def remove_watermark(image):
    w, h = image.size
    return image.crop((0, 0, w, h - 80))

def apply_glow(image, blur_radius=20, intensity=0.4):
    gray = image.convert('L')
    mask = gray.point(lambda x: 255 if x > 210 else 0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.blend(image, ImageChops.screen(image, mask.convert('RGB')), intensity)

def split_tone(image, shadow_rgb=(10, 0, 30), highlight_rgb=(30, 20, 0)):
    img_array = np.array(image).astype(np.float32)
    gray = (np.mean(img_array, axis=2) / 255.0)[..., np.newaxis]
    shadow_mask = np.clip(1.0 - gray * 2.0, 0, 1)
    highlight_mask = np.clip((gray - 0.4) * 2.0, 0, 1)
    img_array += shadow_mask * np.array(shadow_rgb)
    img_array += highlight_mask * np.array(highlight_rgb)
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def apply_vignette(image, intensity=1.5):
    w, h = image.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1.0 - (radius * intensity), 0, 1)
    vignette = Image.fromarray((vignette * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=w//15))
    return ImageChops.multiply(image, vignette.convert('RGB'))

def piece_1_shroud(path, output_path):
    img = Image.open(path).convert('RGB')
    img = remove_watermark(img)
    img = cleanup_background(img)
    img = split_tone(img, shadow_rgb=(30, 0, 50), highlight_rgb=(50, 40, 10))
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = apply_vignette(img, intensity=1.2)
    img = apply_glow(img, blur_radius=15, intensity=0.5)
    img.save(output_path, quality=95)

def piece_2_metamorphosis(path1, path2, output_path):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    img1 = remove_watermark(img1); img2 = remove_watermark(img2)
    blended = ImageChops.screen(img1, img2)
    blended = cleanup_background(blended)
    blended = split_tone(blended, shadow_rgb=(0, 40, 60), highlight_rgb=(40, 40, 0))
    blended = ImageEnhance.Contrast(blended).enhance(2.0)
    blended = apply_vignette(blended, intensity=1.3)
    blended.save(output_path, quality=95)

def piece_3_emergence(path, output_path):
    img = Image.open(path).convert('RGB')
    img = remove_watermark(img)
    img = cleanup_background(img)
    # RGB Split (Subtle)
    r, g, b = img.split()
    r = ImageChops.offset(r, 10, 0); b = ImageChops.offset(b, -10, 0)
    img = Image.merge('RGB', (r, g, b))
    img = split_tone(img, shadow_rgb=(40, 10, 0), highlight_rgb=(60, 50, 20))
    img = ImageEnhance.Contrast(img).enhance(2.8)
    img = apply_vignette(img, intensity=1.4)
    img.save(output_path, quality=95)

def piece_4_core(path, output_path):
    img = Image.open(path).convert('RGB')
    img = remove_watermark(img)
    img = cleanup_background(img)
    img = split_tone(img, shadow_rgb=(0, 30, 60), highlight_rgb=(60, 100, 150))
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = apply_vignette(img, intensity=1.5)
    img = apply_glow(img, blur_radius=30, intensity=0.8)
    w, h = img.size
    img = img.crop((w//5, h//5, 4*w//5, 4*h//5))
    img.save(output_path, quality=95)

if __name__ == "__main__":
    piece_1_shroud("working_art/7116-UNPROCESSED.jpg", "working_art/Celestial_Shroud.jpg")
    piece_2_metamorphosis("working_art/7130-UNPROCESSED.jpg", "working_art/7185-UNPROCESSED.jpg", "working_art/Metamorphosis.jpg")
    piece_3_emergence("working_art/7144-UNPROCESSED.jpg", "working_art/Emergence.jpg")
    piece_4_core("working_art/7160-UNPROCESSED.jpg", "working_art/Digital_Core.jpg")
