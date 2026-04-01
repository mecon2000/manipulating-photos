import subprocess
import os

with open('batch_2_candidates.txt', 'r') as f:
    paths = [line.strip() for line in f if line.strip()]

for i, p in enumerate(paths, 1):
    dest = f"batch_2_photo_{i}.jpg"
    print(f"Downloading {p} to {dest}")
    # Use list format to avoid shell issues with spaces
    subprocess.run(['rclone', 'copy', f"gdrive:_Photos/{p}", './'])
    base = os.path.basename(p)
    if os.path.exists(base):
        os.rename(base, dest)
