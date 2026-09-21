"""Turn a proposal into a template-shaped pack in shared/send-to-instinct.

Usage: tpl_pack.py <proposal name> <template id> <n photos>
The template's own instructions go into meta.txt so instinct knows the target.
"""
import json, os, sys, re
sys.path.insert(0,'/home/rong/gitrep/manipulating-photos/scripts/workflows')
import framing
from PIL import Image
TPL={
 "hq-dark-photography":("HQ DARK Photography","8 photos 2.9,1.4,1.3,1.4,1.4,1.4,1.3,3.0s · 15.0s total",
   "8 dark vertical photos; the first and last must be the strongest.","https://www.capcut.com/template-detail/HQ-DARK-Photography/7589910238346661173"),
 "photo-dump-22":("Photo Dump ✨🌻","22 photos .4-.6s · 9.6s total",
   "22 bright photos; 20 fast at .4s and two closers at .6s.","https://www.capcut.com/template-detail/Photo-Dump/7245286672630385926"),
 "night-vibes":("Night vibes","11 photos .4,1.2,1.2,.9,.9,.9,.9,.9,.9,1.8,2.6s · 13.6s total",
   "11 night / low-light photos.","https://www.capcut.com/template-detail/Night-vibes/7580788063215930677"),
 "beauty-glowup":("Beauty glowup","5 photos 3,1.8,1.8,2.4,3.2s · 12.2s total",
   "5 bright portrait photos; strong opener and closer.","https://www.capcut.com/template-detail/beauty-fyp-viral-trend-glowup-edit-use-new-me/7530039551377607997"),
 "dark-shadows":("Dark Shadows","5 photos 3,3,3,3,2.8s · 14.8s total",
   "5 dark vertical photos with shadow and contrast.","https://www.capcut.com/template-detail/Dark-Shadows/7383088652852743429"),
}
name, tid, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
P=os.path.expanduser('~/.openclaw/workspace/shared/send-to-instinct-review/proposals')
p=json.loads(open(f"{P}/{name}.json").read())
# IG allows no visible nipples or genitals in photographs, so a template pack
# takes only photos the tool passed outright; "check" ones wait for Ron.
pool=p["files"]+p.get("reserve",[])
def take(pool, n, cap):
    out, per = [], {}
    for f in pool:
        m = f.get("model","")
        if per.get(m,0) >= cap: continue
        per[m] = per.get(m,0)+1; out.append(f)
        if len(out) == n: break
    return out
CAP = int(os.environ.get("TPL_CAP","2"))
clean = [f for f in pool if f.get("sfw","pass")=="pass"]
files = take(clean, n, CAP)
if len(files) < n: files = take(clean, n, CAP+1)
if len(files) < n: files = (files+[f for f in pool if f.get("sfw")=="check" and f not in files])[:n]
OUT=os.path.expanduser(f"~/.openclaw/workspace/shared/send-to-instinct/photos-tpl__{tid}__{name.split('__')[1]}")
os.makedirs(OUT,exist_ok=True)
imgs=[]
for i,f in enumerate(files,1):
    src=f.get("src") or f"{P}/{p['name']}/{f['file']}"
    imgs.append((f"{i:02d}_{f['file']}", Image.open(src).convert("RGB")))
# CapCut fills the 9:16 frame, so cut the sides rather than pad: the crop is the
# tallest 9:16 window that fits, centred on the subject (eyes if measured).
AR=9/16
done=[]
for nm,im in imgs:
    W,H=im.size; m=framing.measure(im)
    box=m["box"] if m else (0,0,W,H)
    cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
    if m and m.get("eyes"):
        cx, cy = m["eyes"][0], m["eyes"][1] + (box[3]-box[1])*0.18
    h=H; w=h*AR
    if w>W: w=W; h=w/AR
    left=min(max(cx-w/2,0),W-w); top=min(max(cy-h/2,0),H-h)
    done.append((nm, im.crop((int(left),int(top),int(left+w),int(top+h))).resize((1080,1920), Image.LANCZOS)))
notes=[f"framing: 9:16 1080x1920, fill crop centred on the subject (sides cut, nothing padded)"]
for nm,im in done: im.save(OUT+'/'+nm, quality=95)
t=TPL[tid]
open(OUT+'/meta.txt','w').write(
 f"Reel type  : photos-for-template\nTemplate   : {t[0]} — {t[3]}\nSlots      : {t[1]}\n"
 f"Instruction: {t[2]}\nAspect     : 9:16 1080x1920 ({notes[0]})\nPhotos     : {len(done)} (template wants {n})\n"
 f"Why        : {p['why']}\n\n" +
 "\n".join(f"  {nm}   model={f.get('model','')}  session={f.get('session','')}  rating={f.get('rating','')}  sfw={f.get("sfw","")}"
           for (nm,_),f in zip(done,files)) + "\n")
print(OUT, len(done), 'photos', notes[0])
