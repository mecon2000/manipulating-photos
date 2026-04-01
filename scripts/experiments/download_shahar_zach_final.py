import subprocess
import os

path = "Shahar Zach/Onlyfans/Free/ULed already/WhatsApp Image 2021-01-16 at 21.23.17 (2).jpeg"
dest = "shahar_zach_true_original.jpg"
print(f"Downloading {path}")
subprocess.run(['rclone', 'copy', f"gdrive:_Photos/{path}", './'])
base = os.path.basename(path)
if os.path.exists(base):
    os.rename(base, dest)
