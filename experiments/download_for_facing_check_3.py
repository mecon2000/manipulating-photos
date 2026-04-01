import subprocess
import os

with open('candidate_paths_3.txt', 'r') as f:
    paths = [line.strip().strip('", ') for line in f if line.strip()]

for i, p in enumerate(paths, 1):
    dest = f"facing_check_v3_{i}.jpg"
    print(f"Downloading {p} to {dest}")
    subprocess.run(['rclone', 'copy', f"gdrive:_Photos/{p}", './'])
    base = os.path.basename(p)
    if os.path.exists(base):
        os.rename(base, dest)
