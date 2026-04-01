import os
import subprocess
import requests
import base64

def get_candidates():
    with open('candidates.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def download_file(rel_path):
    print(f"Downloading {rel_path}")
    subprocess.run(['rclone', 'copy', f"gdrive:_Photos/{rel_path}", './'])
    return os.path.basename(rel_path)

def main():
    candidates = get_candidates()
    for rel_path in candidates:
        local_name = download_file(rel_path)
        # Check orientation with vision? No, I'll do a batch check after I have a few local ones.
        pass

if __name__ == "__main__":
    main()
