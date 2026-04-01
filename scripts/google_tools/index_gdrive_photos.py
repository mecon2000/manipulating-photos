import os
import json
import subprocess

def list_gdrive_recursive(path):
    print(f"Indexing {path}...")
    try:
        # Using rclone ls to get all files
        # This might take a minute for 20k files
        proc = subprocess.run(['rclone', 'ls', path, '--include', '*.jpg', '--include', '*.png', '--include', '*.jpeg'], capture_output=True, text=True)
        lines = proc.stdout.splitlines()
        
        photos = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                size, rel_path = parts
                photos.append(rel_path)
        return photos
    except Exception as e:
        print(f"Error indexing: {e}")
        return []

def main():
    photos = list_gdrive_recursive("gdrive:_Photos")
    print(f"Found {len(photos)} photos.")
    with open("state/gdrive_photos_index.json", "w") as f:
        json.dump(photos, f, indent=2)

if __name__ == "__main__":
    main()
