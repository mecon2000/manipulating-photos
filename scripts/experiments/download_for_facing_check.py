import subprocess
import os

with open('candidate_paths.txt', 'r') as f:
    paths = [line.strip() for line in f if line.strip()]

for i, p in enumerate(paths, 1):
    dest = f"facing_check_{i}.jpg"
    print(f"Downloading {p} to {dest}")
    subprocess.run(['rclone', 'copy', f"gdrive:_Photos/{p}", './'])
    base = os.path.basename(p)
    if os.path.exists(base):
        os.rename(base, dest)
