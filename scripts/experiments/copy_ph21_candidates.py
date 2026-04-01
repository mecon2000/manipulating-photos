import os

candidates = [
    "Rong_IMG_1613-Final.jpg",
    "BLD_8838+8847_M2E2.jpg",
    "BLD_8920E.jpg",
    "BLD_1144E Censored.jpg",
    "BLD_5103.jpg",
    "BLD_0Y9A2338E.jpg",
    "BLD_1540EE.jpg",
    "BLD_9025E.jpg",
    "Rong_IMG_9096EE.jpg"
]

src_base = "gdrive:_My galleries/16 This will be in exhebitions!/"
dest_base = "gdrive:_photos from openclaw/PH21 Candidates/"

for i, filename in enumerate(candidates, 1):
    new_name = f"{i}_{filename}"
    cmd = f'rclone copyto "{src_base}{filename}" "{dest_base}{new_name}"'
    print(f"Executing: {cmd}")
    os.system(cmd)

print("Copy completed.")
