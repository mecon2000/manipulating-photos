import sqlite3, datetime as D, os, glob, re, json, subprocess
c=sqlite3.connect('file:/home/rong/gitrep/photo-catalogging/data/photo-catalog.db?mode=ro',uri=True)
A=os.path.expanduser('~/.openclaw/workspace/_archive/'); PH=os.path.expanduser('~/.openclaw/workspace/_photos/')
T=lambda s: D.datetime.fromisoformat(s[:19]) + D.timedelta(seconds=float('0.'+s[20:]) if len(s)>20 else 0)
sess=[ # (label, folder like, video dir, model photo folders, anchor stem, video, pos s)
 ("Nitsan","2025/2025-02-13 Nitsan Perry%","video",["Nitzan Perry","Nitsan Perry"],"BLD_5273","VID_20250213_130216.mp4",134),
 ("Sharon","2020/2020-09-03 Sharon Cohens%","BTS from my phone",["Sharon Cohens"],"BLD_5790","VID_20200903_122153.mp4",229),
 ("Shay","2024/2024-08-23 Shay & Ariel%","Video",["Shay & Ariel"],"BLD_7180","VID20240822234501.mp4",25)]
def vstart(n):
    m=re.search(r'(\d{8})_?(\d{6})',n); return D.datetime.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S')
out={}
for lab,like,vd,mf,ast,av,apos in sess:
    rows=c.execute("select p.filename,p.taken_at,p.lr_rating,p.lr_pick,s.folder_path from photos p join sessions s on s.id=p.session_id where s.folder_path like ?",('I:/Photos/'+like,)).fetchall()
    folder=A+rows[0][4][len('I:/Photos/'):]
    at=[r[1] for r in rows if r[0].startswith(ast)][0]
    off=((vstart(av)+D.timedelta(seconds=apos))-T(at)).total_seconds()
    vids=[]
    for v in sorted(glob.glob(folder+'/'+vd+'/*.mp4')):
        dur=float(subprocess.run([os.path.expanduser('~/openclaw-venv/bin/ffprobe'),'-v','error','-show_entries','format=duration','-of','csv=p=0',v],capture_output=True,text=True).stdout)
        vids.append((v,vstart(os.path.basename(v)),dur))
    exports={}
    for m in mf:
        for f in glob.glob(PH+m+'/**/*.jpg',recursive=True):
            k=re.search(r'([A-Z]{3}_\d{4})',os.path.basename(f))
            if k and 'unprocessed' not in f.lower(): exports.setdefault(k.group(1),[]).append(f)
    pairs=[]
    seen=set()
    for fn,ta,r,pk,_ in sorted(rows,key=lambda r:r[1]):
        st=fn[:8]
        if st in seen: continue
        seen.add(st)
        w=T(ta)+D.timedelta(seconds=off)
        for v,vs,dur in vids:
            pos=(w-vs).total_seconds()
            if 5<=pos<=dur-0.5:
                pairs.append(dict(stem=st,taken=ta,rating=r or 0,pick=pk or 0,video=v,pos=round(pos,2),edited=sorted(exports.get(st,[])),raw=folder+'/'+fn))
    out[lab]=dict(offset=off,folder=folder,pairs=pairs)
    print(lab,'offset',round(off,1),'photos in video:',len(pairs),'edited:',sum(1 for p in pairs if p['edited']))
json.dump(out,open(os.path.dirname(__file__)+'/bts.json','w'),indent=1)
