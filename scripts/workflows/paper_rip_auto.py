#!/usr/bin/env python3
"""Paper-rip reel from a stylized image, zero manual geometry.

For each --source (a stylized final, e.g. surreal-with-face/*__final.jpg or a favorite):
  1. find the original it was made from (sibling __src.jpg, sidecar "relit", or the
     favorites.json command's --relit), resized to the stylized frame
  2. detect the face (insightface) -> square crop + face ellipse
  3. build a tear layout (blocks | diagonal | sweep-h | sweep-v), a long jagged sliver
     over the face aimed at the nearest free edge, tears on the music's beats
  4. render with paper_rip_reel.py -> <out>/<date>_<Model>_paperrip_<tag>/
       <name>.mp4 (with music, for review)   <name>_silent.mp4 (to upload)
       <name>_music.json (what to pick in IG) <name>.config.json (exact reproduction)

  paper_rip_auto.py --source STYLIZED.jpg [...] [--music storm] [--layout diagonal]
                    [--text-top "ART LIES."] [--text-drop "SHE DOESN'T."] [--seed 3]
                    [--local-output-dir DIR]
"""
import argparse, glob, json, os, re, shlex, subprocess, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.expanduser('~/.openclaw/workspace/shared')
ASSETS = os.path.join(SHARED, 'ig-reels/_assets')
FAVS = os.path.join(SHARED, 'favorites/favorites.json')
S = 1280
LAYOUTS = ['blocks', 'diagonal', 'sweep-h', 'sweep-v']
TEXTS = [                                   # (before the drop, on the drop)
    ("ART LIES.", "SHE DOESN'T."),
    ("THEY PAINTED A STORM.", "SHE WAS THE CALM."),
    ("THE PAINT WAS A MASK.", "THIS IS HER."),
    ("THEY WANTED A PAINTING.", "SHE WAS ALREADY ONE."),
    ("EVERY LAYER A LIE.", "EXCEPT HER."),
    ("STRIP THE ART.", "KEEP THE SOUL."),
    ("FILTERS FADE.", "SHE STAYS."),
]


# ---------- finding the original ----------
def _fav_entry(path):
    try:
        favs = json.load(open(FAVS)).get('favorites', [])
    except (OSError, ValueError):
        return {}
    name = os.path.basename(path)
    return next((e for e in favs if e.get('file') == name), {})


def find_original(start):
    """-> (original path, model name or None)."""
    stem = os.path.splitext(start)[0]
    stem = re.sub(r'_4x$', '', stem)
    prefix = re.sub(r'__(final|layers|surreal)$', '', stem)
    if os.path.isfile(prefix + '__src.jpg'):
        src = prefix + '__src.jpg'
    else:
        src = None
    side = {}
    for sc in (stem + '.json', prefix + '__final.json'):
        if os.path.isfile(sc):
            side = json.load(open(sc)); break
    if not src:
        for k in ('relit', 'original_path', 'source'):
            if side.get(k) and os.path.isfile(os.path.expanduser(side[k])):
                src = os.path.expanduser(side[k]); break
    fav = _fav_entry(start)
    if not src and fav:
        m = re.search(r"--relit\s+('[^']+'|\"[^\"]+\"|\S+)", fav.get('command') or '')
        if m and os.path.isfile(os.path.expanduser(shlex.split(m.group(1))[0])):
            src = os.path.expanduser(shlex.split(m.group(1))[0])
        elif fav.get('source'):
            hits = glob.glob(os.path.expanduser(fav['source']), recursive=True)
            src = next((h for h in hits if '/Processed/' in h), hits[0] if hits else None)
    model = fav.get('model') or side.get('model')
    if not model:
        m = re.search(r'/_photos/([^/]+)/', (side.get('relit') or src or ''))
        model = m.group(1) if m else None
    return src, model


# ---------- face ----------
_fa = None


def detect_face(img):
    global _fa
    if _fa is None:
        from insightface.app import FaceAnalysis
        _fa = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'],
                           providers=['CPUExecutionProvider'])
        _fa.prepare(ctx_id=-1, det_size=(640, 640))
    faces = _fa.get(img)
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return [float(v) for v in f.bbox]


# ---------- layout ----------
def band_rules(face, target, label0=0):
    """4 bands from the far `target` point toward the face, then the face's
    neighbourhood split in two (labels 4 and 5)."""
    fx, fy = face[:2]
    dx, dy = target[0] - fx, target[1] - fy
    n = max(np.hypot(dx, dy), 1)
    far = max(((cx - fx) * dx + (cy - fy) * dy) / n for cx in (0, S) for cy in (0, S))
    near = 170
    cuts = np.linspace(near, far, 5)
    rules = []
    # side split of the near zone: perpendicular to the band direction
    px, py = -dy, dx
    rules.append({"label": 5, "band": [fx, fy, dx, dy, -9999, near], "side": [fx, fy, px, py]})
    for k in range(4):                         # outermost band tears first
        rules.append({"label": 3 - k, "band": [fx, fy, dx, dy, float(cuts[k]),
                                                9999 if k == 3 else float(cuts[k + 1])]})
    return {"default": 4, "rules": rules}


def blocks_rules(face):
    fx, fy = face[:2]
    L, R, T = fx - 268, fx + 282, fy + 150
    return {"default": 4, "rules": [
        {"label": 3, "rect": [R, -9999, 9999, fy + 390]},
        {"label": 2, "rect": [-9999, -9999, L, 9999]},
        {"label": 1, "rect": [R, fy + 390, 9999, 9999]},
        {"label": 0, "rect": [L, fy + 570, R, 9999]},
        {"label": 5, "rect": [L, -9999, R, T]},
    ]}


def layout(kind, face, rng):
    fx, fy = face[:2]
    if kind == 'blocks':
        return blocks_rules(face), {"from": [S / 2, S], "to": [S / 2, 0]}
    if kind == 'diagonal':
        tx, ty = max([(0, 0), (S, 0), (0, S), (S, S)], key=lambda c: np.hypot(c[0] - fx, c[1] - fy))
    elif kind == 'sweep-h':
        tx, ty = (S if fx < S / 2 else 0), fy
    else:
        tx, ty = fx, (S if fy < S / 2 else 0)
    return band_rules(face, (tx, ty)), {"from": [tx, ty], "to": [2 * fx - tx, 2 * fy - ty]}


def sliver(face, rng):
    """Long jagged dagger: an irregular hexagon round the face + a narrow tip on an
    edge; convex hull, so the face is always fully under it and it never reads as a circle."""
    fx, fy, rx, ry = face
    best = None
    for a in rng.permutation(np.radians([-60, -120, -30, -150, 0, 180, -90])):
        dx, dy = np.cos(a), np.sin(a)
        ts = [t for t in ((0 - fx) / dx if dx < 0 else (S - fx) / dx if dx > 0 else np.inf,
                          (0 - fy) / dy if dy < 0 else (S - fy) / dy if dy > 0 else np.inf)]
        d = min(ts)
        if 2.2 * max(rx, ry) < d < 1000:
            best = (a, d); break
    if best is None:
        best = (np.radians(-90), fy)
    a, d = best
    tip = np.array([fx + d * np.cos(a), fy + d * np.sin(a)])
    perp = np.array([-np.sin(a), np.cos(a)]) * rng.uniform(30, 55)
    off = rng.uniform(0, np.pi / 3)
    hexa = [(fx + 1.24 * rng.uniform(1, 1.1) * rx * np.cos(off + k * np.pi / 3),
             fy + 1.24 * rng.uniform(1, 1.1) * ry * np.sin(off + k * np.pi / 3)) for k in range(6)]
    pts = np.float32(hexa + [tuple(tip + perp), tuple(tip - perp)])
    hull = cv2.convexHull(pts).reshape(-1, 2)
    return [[round(float(x)), round(float(y))] for x, y in hull], [round(float(tip[0])), round(float(tip[1]))]


def beat_tears(mus, seg_path):
    import librosa
    y, sr = librosa.load(seg_path, sr=22050, mono=True, offset=mus['t0'], duration=mus['drop'] - mus['t0'])
    _, b = librosa.beat.beat_track(y=y, sr=sr, units='time')
    b = [t + mus['t0'] for t in b if mus['t0'] + .45 <= t + mus['t0'] <= mus['break'] + .02]
    if len(b) >= 8:
        L = len(b) - 1
        return [b[L - 7], b[L - 5], b[L - 3], b[L - 1], b[L]]
    return list(np.linspace(mus['t0'] + .55, mus['break'], 5))


def accent(img):
    """Most saturated colour in the stylized image, for the drop line."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) * (hsv[..., 2] > 80)
    if s.max() < 90:
        return [235, 30, 25]
    sel = hsv[s >= np.percentile(s, 98)]
    h = int(np.median(sel[:, 0]))
    bgr = cv2.cvtColor(np.uint8([[[h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


# ---------- main ----------
def build(start, args, idx):
    rng = np.random.default_rng(args.seed + idx)
    end, model = find_original(start)
    if args.end:
        end = args.end
    if not end:
        raise SystemExit(f'no original found for {start} (pass --end)')
    A, B = cv2.imread(start), cv2.imread(end)
    h, w = A.shape[:2]
    if B.shape[:2] != (h, w):
        ar_a, ar_b = w / h, B.shape[1] / B.shape[0]
        if abs(ar_a - ar_b) / ar_a > .02:
            print(f'WARNING: aspect differs ({ar_a:.3f} vs {ar_b:.3f}); stretching original')
        B = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)
    bb = detect_face(B) or detect_face(A)
    if bb is None:
        raise SystemExit(f'no face found in {end}')
    cs = min(w, h)
    fcx, fcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    x0 = int(np.clip(fcx - cs / 2, 0, w - cs))
    y0 = int(np.clip(fcy - cs * .42, 0, h - cs))
    k = S / cs
    face = [round((fcx - x0) * k), round((fcy - y0) * k),
            round((bb[2] - bb[0]) * .62 * k), round((bb[3] - bb[1]) * .72 * k)]

    kind = args.layout if args.layout != 'auto' else LAYOUTS[(args.seed + idx) % len(LAYOUTS)]
    regions, cam = layout(kind, face, rng)
    sl, pivot = sliver(face, rng)

    presets = json.load(open(os.path.join(ASSETS, 'music_presets.json')))
    mname = args.music if args.music != 'auto' else sorted(presets)[(args.seed + idx) % len(presets)]
    mus = dict(presets[mname])
    mus['t1'] = round(mus['t0'] + 7.0, 2)
    tears = [[t, None, 1 if i % 2 else -1, 1.0] for i, t in enumerate(beat_tears(mus, os.path.join(ASSETS, mus['file'])))]
    tears.append([round(mus['break'] + .25, 2), None, -1, .45])
    tears.append([round(mus['drop'] - .02, 2), float(np.degrees(np.arctan2(pivot[1] - face[1], pivot[0] - face[0])) + 180), 1, 3.2])

    top, drop = TEXTS[(args.seed + idx) % len(TEXTS)]
    top, drop = args.text_top or top, args.text_drop or drop
    # the renderer works on the (resized) original saved next to the reel when sizes differ
    cfg = {
        "start": start, "end": end, "crop": [x0, y0, cs], "face": face,
        "regions": regions, "sliver": sl, "sliver_pivot": pivot,
        "sliver_lift": round(min(1.35, mus['drop'] - mus['break'] - .37), 2),
        "tears": tears, "music": mus,
        "camera": {"zoom": 1.3, **cam},
        "text": [
            {"text": top, "t": mus['t0'] + .1, "size": 128, "color": [255, 255, 255], "row": 0, "pop": .25, "dur": .18},
            {"text": drop, "t": mus['drop'], "size": 150, "color": accent(A), "row": 1, "pop": .5, "dur": .12},
        ],
        "_meta": {"layout": kind, "music_preset": mname, "model": model, "seed": args.seed + idx},
    }
    return cfg, B if cv2.imread(end).shape[:2] != (h, w) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', nargs='+', required=True)
    ap.add_argument('--end', help='override the original (single source only)')
    ap.add_argument('--music', default='auto')
    ap.add_argument('--layout', default='auto', choices=['auto'] + LAYOUTS)
    ap.add_argument('--text-top', default='')
    ap.add_argument('--text-drop', default='')
    ap.add_argument('--seed', type=int, default=int(time.time()) % 1000)
    ap.add_argument('--local-output-dir', default=os.path.join(SHARED, 'ig-reels'))
    ap.add_argument('--dry-run', action='store_true', help='config + contact sheet only, no video')
    args = ap.parse_args()

    for i, start in enumerate(args.source):
        start = os.path.abspath(os.path.expanduser(start))
        cfg, resized = build(start, args, i)
        tag = re.sub(r'__(final|layers)(_4x)?$', '', os.path.splitext(os.path.basename(start))[0])[:70]
        model = re.sub(r'\W', '', (cfg['_meta']['model'] or 'Unknown').title())
        name = f"{time.strftime('%Y-%m-%d')}_{model}_paperrip_{tag}"
        out = os.path.join(args.local_output_dir, name)
        os.makedirs(out, exist_ok=True)
        if resized is not None:                      # keep the resized original with the reel
            cfg['end'] = os.path.join(out, name + '_original.jpg')
            cv2.imwrite(cfg['end'], resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cpath = os.path.join(out, name + '.config.json')
        json.dump(cfg, open(cpath, 'w'), indent=2)
        mus = cfg['music']
        json.dump({"artist": mus['artist'], "title": mus['title'], "ig_search": mus['ig_search'],
                   "use_from": f"{int(mus['t0'] // 60)}:{mus['t0'] % 60:04.1f}",
                   "use_to": f"{int(mus['t1'] // 60)}:{mus['t1'] % 60:04.1f}",
                   "drop_at": f"{int(mus['drop'] // 60)}:{mus['drop'] % 60:04.1f}",
                   "note": f"In IG: add audio -> search '{mus['ig_search']}' -> start at "
                           f"{int(mus['t0'] // 60)}:{int(mus['t0'] % 60):02d}. Upload the _silent.mp4.",
                   "youtube_ref": mus['youtube']},
                  open(os.path.join(out, name + '_music.json'), 'w'), indent=2)
        print(f"[{name}] layout={cfg['_meta']['layout']} music={cfg['_meta']['music_preset']} "
              f"text={cfg['text'][0]['text']!r}/{cfg['text'][1]['text']!r}", flush=True)
        mp4 = os.path.join(out, name + '.mp4')
        if args.dry_run:
            subprocess.run([sys.executable, os.path.join(HERE, 'paper_rip_reel.py'), cpath, mp4,
                            '--preview', os.path.join(out, name + '_sheet.jpg')], check=True)
            continue
        subprocess.run([sys.executable, os.path.join(HERE, 'paper_rip_reel.py'), cpath, mp4], check=True)
        import static_ffmpeg; static_ffmpeg.add_paths()
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp4, '-an', '-c:v', 'copy',
                        os.path.join(out, name + '_silent.mp4')], check=True)
        print('done', out, flush=True)


if __name__ == '__main__':
    main()
