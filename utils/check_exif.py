from PIL import Image
from PIL.ExifTags import TAGS

def get_exif(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            print("No EXIF data found.")
            return

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag in ["FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal", "Model", "FocalLength"]:
                print(f"{tag}: {value}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_exif("agents/lux/working/Daniella_BLD_4183_blurred_ig.jpg")
