import PIL.Image
import os

files = [
    "working_art/paper_tear_expanded/Paper_Tear_Danielle_1_v7.png",
    "working_art/paper_tear_expanded/Paper_Tear_Danielle_2_v7.png",
    "working_art/paper_tear_expanded/Paper_Tear_Jenia_v7.png"
]

output_dir = "working_art/cica_submission"
os.makedirs(output_dir, exist_ok=True)

for f in files:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
    img = PIL.Image.open(f)
    name = os.path.basename(f).replace("_v7.png", "_CICA.jpg")
    out_path = os.path.join(output_dir, name)
    # Target 100-500 KB. PNGs are small, so JPGs will be smaller.
    # We might need to UPSCALE if they are too small, but CICA says 100-500 KB.
    # Current PNGs are ~90KB.
    img.convert("RGB").save(out_path, "JPEG", quality=95)
    size = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path}, size: {size:.1f} KB")
