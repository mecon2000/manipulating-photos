import os
import subprocess

def download_file_by_path(rel_path, dest):
    print(f"Rcloning: {rel_path}")
    # Copy specifically the file to the current directory
    cmd = ['rclone', 'copy', f"gdrive:_Photos/{rel_path}", './']
    subprocess.run(cmd)
    base = os.path.basename(rel_path)
    if os.path.exists(base):
        # Only rename if it's not already the name
        if base != dest:
            os.rename(base, dest)
        return True
    return False

if __name__ == "__main__":
    download_file_by_path("Mickey/1 just Mickey,her house/Unprocessed photos/BLD_4471.jpg", "batch_photo_1.jpg")
    download_file_by_path("Zohar & Asher/Unprocessed/2 Outdoors Unprocessed/BLD_3481 - UNPROCESSED.jpg", "batch_photo_2.jpg")
