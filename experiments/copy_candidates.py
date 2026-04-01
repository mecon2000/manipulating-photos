import os
import subprocess

def rclone_copy_limited(src, dest, filenames):
    for fn in filenames:
        cmd = ["rclone", "copyto", f"{src}{fn}", f"{dest}{fn}"]
        subprocess.run(cmd)

maayan_files = [
    "BLD_4078.jpg", "BLD_4107.jpg", "BLD_4140-2.jpg", "BLD_4140.jpg", "BLD_4178.jpg",
    "BLD_4225.jpg", "BLD_4264.jpg", "BLD_4391.jpg", "BLD_4392.jpg", "BLD_4393.jpg",
    "BLD_4426.jpg", "BLD_4484.jpg", "BLD_4487.jpg", "BLD_4555.jpg", "BLD_4576.jpg"
]

dina_files = [
    "BLD_7930E.jpg", "BLD_7935E.jpg", "BLD_7952-2.jpg", "BLD_8002E.jpg", "BLD_8040.jpg",
    "BLD_8072E.jpg", "BLD_8085E.jpg", "BLD_8102E.jpg"
] # Only found 8 in thumbs head -n 15, let's grab more or just use these.

anya_files = [
    "BLD_0480E.jpg", "BLD_0610E.jpg", "BLD_0652E.jpg", "BLD_0673EE.jpg", "BLD_0734.jpg",
    "BLD_0756E.jpg", "BLD_0780E.jpg", "BLD_0825EE.jpg", "BLD_0828E.jpg", "BLD_0967-EditE.jpg",
    "BLD_0973EE.jpg"
]

rclone_copy_limited('gdrive:_Photos/Mayanu_/3 Her house/Processed/', 'gdrive:_photos from openclaw/Reel Candidates/Theme A - Still Light (Maayan)/', maayan_files)
rclone_copy_limited('gdrive:_Photos/Adi Levi/42 Dina, Japanese garden/Processed/', 'gdrive:_photos from openclaw/Reel Candidates/Theme B - Nature (Dina)/', dina_files)
rclone_copy_limited('gdrive:_Photos/Anya/Processed/', 'gdrive:_photos from openclaw/Reel Candidates/Theme C - Studio Soul (Anya)/', anya_files)
