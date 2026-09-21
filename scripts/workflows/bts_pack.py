"""Pack of BTS clips + result photos for a CapCut BTS template.

No time sync needed: the clips only have to be from the same shoot.
Usage: bts_pack.py <session dir under _archive> <photo glob root> <template id> <n clips> <n photos> <pack name>
"""
import os, sys, glob, subprocess, random, re
sys.path.insert(0,'/home/rong/gitrep/manipulating-photos/scripts/workflows')
os.environ.setdefault('MPLBACKEND','Agg')
import sfw, framing
from PIL import Image
FF=os.path.expanduser('~/openclaw-venv/bin/ffmpeg'); FP=os.path.expanduser('~/openclaw-venv/bin/ffprobe')
A=os.path.expanduser('~/.openclaw/workspace/_archive/'); OUTROOT=os.path.expanduser('~/.openclaw/workspace/shared/send-to-instinct/')
ORDER={  # how clips (v) and photos (p) alternate through the template's slots
 "bts-11": list("vpvpvpvpvpv"),
 "behind-the-scene-19": list("vppvppvppvppvppvppv"),
 "bts-photography": ["v"] + ["v","p","v","p","v","p","p","v","p","p","p","p","p","p"],
}
TPL={
 "bts-11":("Bts · Photoshoot","11 slots .9-2.1s · 16.3s","5-6 BTS clips + 5-6 photos. Durations: 1.4,2.1,.9,1.8,1.6,1.7,1.7,1.7,1.7,1.2,2.1s",
   "https://www.capcut.com/template-detail/behindthescene-photoshoot-viral-trend-model/7584424218624838917"),
 "behind-the-scene-19":("BEHIND THE SCENE","19 slots .2-2.8s · 12.2s","7-9 BTS clips + 10-12 photos. Durations: 2.8,.4,.8,.5,1.2,.9,.6,.4,.2,1.3,.8,2.6,.8,1.3,.8,.6,.4,.2,1.0s",
   "https://www.capcut.com/template-detail/behindthescene-viral-trend-photoshoot-fyp/7582633962619653381"),
 "bts-photography":("BTS photography","2 slots 1.2, 17.3s · 18.1s","slot 1: a BTS clip. slot 2: a finished montage; ideally 4-6 BTS clips + 8-12 result photos",
   "https://www.capcut.com/template-detail/BTS-photography/7433841777594993925"),
}
sess, photoroot, tid, nclip, nphoto, packname = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
OUT=OUTROOT+packname; os.makedirs(OUT,exist_ok=True)
vids=[v for v in glob.glob(A+sess+'/**/*.mp4',recursive=True)+glob.glob(A+sess+'/**/*.MOV',recursive=True)]
random.seed(7); random.shuffle(vids)
def dur(v):
    try: return float(subprocess.run([FP,'-v','error','-show_entries','format=duration','-of','csv=p=0',v],capture_output=True,text=True).stdout)
    except Exception: return 0
clips=[]
for v in vids:
    if len(clips)>=nclip: break
    d=dur(v)
    if d<4: continue
    start=max(0.5, d*0.35)
    t=f"/tmp/frame_{len(clips)}.jpg"
    subprocess.run([FF,'-y','-v','error','-ss',str(start+1),'-i',v,'-frames:v','1',t])
    if not os.path.exists(t) or sfw.check(t)['verdict']!='pass':
        print('skip clip (sfw/frame):', os.path.basename(v)); continue
    out=f"{OUT}/.clip{len(clips)+1:02d}_{re.sub(r'[^A-Za-z0-9]+','_',os.path.basename(v))[:40]}.mp4"
    subprocess.run([FF,'-y','-v','error','-ss',str(start),'-i',v,'-t','3',
                    '-vf','scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920','-c:v','libx264','-crf','20','-an',out])
    clips.append((out, os.path.basename(v), round(start,1), round(d,1)))
# photos MUST come from this same session: resolve them through the catalog,
# which verifies each export against its RAW instead of trusting folder names
import instinct_pack as P
R=P.Resolver(); fav=P.favored()
folder=os.path.basename(sess.rstrip('/'))
rows=P.db().execute("""select p.filename,p.lr_rating,p.lr_pick from photos p
    join sessions s on s.id=p.session_id where s.folder_path like ?""",('%/'+folder,)).fetchall()
rows.sort(key=lambda r:-((r[1] or 0)*2+(r[2]==1)+(2 if r[0].rsplit('.',1)[0] in fav else 0)))
keep=[]; unedited=0
for fn,rt,pk in rows:
    if len(keep)>=nphoto: break
    st=fn.rsplit('.',1)[0]
    e=R.export(folder, st, 'edited') or R.export(folder, st, 'unedited')
    if not e: continue
    if sfw.check(e[0])['verdict']!='pass': continue
    if 'unprocessed' in e[0].lower(): unedited+=1
    keep.append(e[0])
print(f"photos from session {folder}: {len(keep)} ({unedited} unprocessed)")
AR=9/16; names=[]
for i,e in enumerate(keep,1):
    im=Image.open(e).convert('RGB'); W,H=im.size; m=framing.measure(im)
    box=m['box'] if m else (0,0,W,H); cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    if m and m.get('eyes'): cx,cy=m['eyes'][0], m['eyes'][1]+(box[3]-box[1])*0.18
    h=H; w=h*AR
    if w>W: w=W; h=w/AR
    l=min(max(cx-w/2,0),W-w); t2=min(max(cy-h/2,0),H-h)
    nm=f".photo{i:02d}_{re.search(r'([A-Z]{3}_\d{4})',os.path.basename(e)).group(1) if re.search(r'([A-Z]{3}_\d{4})',os.path.basename(e)) else i}.jpg"
    im.crop((int(l),int(t2),int(l+w),int(t2+h))).resize((1080,1920),Image.LANCZOS).save(OUT+'/'+nm,quality=95); names.append((nm,e))
# lay the files out in the template's slot order: 01_, 02_, ... is the order to add them
order=ORDER.get(tid) or (["v","p"]*40)
ci=pi=0; seq=[]
for slot,kind in enumerate(order,1):
    if kind=="v" and ci<len(clips):
        src=clips[ci][0]; ci+=1
    elif pi<len(names):
        src=OUT+'/'+names[pi][0]; pi+=1
    elif ci<len(clips):
        src=clips[ci][0]; ci+=1
    else: continue
    base=os.path.basename(src).lstrip('.')
    base=base.split('_',1)[1] if '_' in base else base
    dst=f"{OUT}/{slot:02d}_{'clip' if src.endswith('.mp4') else 'photo'}_{base}"
    os.rename(src,dst); seq.append((slot,os.path.basename(dst)))
for extra in [c[0] for c in clips[ci:]]+[OUT+'/'+n[0] for n in names[pi:]]:
    os.rename(extra, OUT+'/zz_spare_'+os.path.basename(extra).lstrip('.'))
t=TPL[tid]
open(OUT+'/meta.txt','w').write(
 f"Reel type  : bts-for-template\nTemplate   : {t[0]} — {t[3]}\nSlots      : {t[1]}\nInstruction: {t[2]}\n"
 f"Aspect     : 9:16 1080x1920 (clips cropped to fill, 3s each)\nSession    : {sess}\n"
 f"Clips      : {len(clips)} (template wants {nclip})   Photos: {len(names)} (wants {nphoto})\n"
 "Sync       : not needed for this template — clips are from the same shoot, not matched to a photo\n\n"
 + "Order      : add the files to CapCut by their number prefix, 01 first\n\n"
 + "\n".join(f"  {n:02d}  {f}" for n,f in seq) + "\n")
print(packname, len(clips),'clips', len(names),'photos')
