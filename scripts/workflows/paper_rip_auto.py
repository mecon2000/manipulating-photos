#!/usr/bin/env python3
"""Paper-rip reel from a stylized image, zero manual geometry.

For each --source (a stylized final, e.g. surreal-with-face/*__final.jpg or a favorite):
  1. find the original it was made from (sibling __src.jpg, sidecar "relit", or the
     favorites.json command's --relit), resized to the stylized frame
  2. detect face (insightface) + body (mediapipe pose) and choose the frame per image:
     the tallest of 9:16 / 4:5 / 1:1 that still holds the whole figure, hands included
  3. build a tear layout (shards | strips | wedges | blocks | diagonal | sweep-h | sweep-v),
     a long jagged sliver over the face aimed at the nearest free edge, tears on the
     music's beats, and a camera (cuts | whip | drift) that jumps to the legs, hands,
     hips... as each piece's crack starts
  4. render with paper_rip_reel.py -> <out>/<date>_<Model>_paperrip_<tag>/
       <name>.mp4 (with music, for review)   <name>_silent.mp4 (to upload)
       <name>_music.json (what to pick in IG) <name>.config.json (exact reproduction)

  paper_rip_auto.py --source STYLIZED.jpg [...] [--music storm] [--layout shards]
                    [--aspect 9:16] [--camera whip]
                    [--text-top "ART LIES."] [--text-drop "SHE DOESN'T."] [--seed 3]
                    [--local-output-dir DIR]
"""
import argparse, glob, json, os, re, shlex, subprocess, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.expanduser('~/.openclaw/workspace/shared')
ASSETS = os.path.join(SHARED, 'ig-reels/_assets')
FAVS = os.path.join(SHARED, 'favorites/favorites.json')
LAYOUTS = ['shards', 'diagonal', 'strips', 'blocks', 'wedges', 'sweep-h', 'sweep-v']
CAMERAS = ['cuts', 'whip', 'drift']
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


def _sat(im):
    return float(cv2.cvtColor(cv2.resize(im, (256, 256)), cv2.COLOR_BGR2HSV)[..., 1].mean())


_REG = {}                                               # colour master -> (affine to edit frame, size)


def _register(master, edit):
    """Affine (scale+shift) placing `master` onto `edit`'s frame, or None if they don't match.
    B&W edits are often re-cropped, so equal size/aspect proves nothing."""
    g_e = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
    g_m = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(4000)
    ke, de = sift.detectAndCompute(g_e, None)
    km, dm = sift.detectAndCompute(g_m, None)
    if de is None or dm is None:
        return None
    good = [x for x, y in cv2.BFMatcher().knnMatch(de, dm, k=2) if x.distance < .75 * y.distance]
    if len(good) < 20:
        return None
    M, _ = cv2.estimateAffinePartial2D(np.float32([km[g.trainIdx].pt for g in good]),
                                       np.float32([ke[g.queryIdx].pt for g in good]),
                                       ransacReprojThreshold=4)
    if M is None:
        return None
    w = cv2.warpAffine(g_m, M, (g_e.shape[1], g_e.shape[0]))
    corr = np.corrcoef(g_e.ravel().astype(np.float32), w.ravel().astype(np.float32))[0, 1]
    return M if corr > .9 else None


def colour_sibling(src):
    """A B&W edit (BLD_1539E-2) usually has a colour master next to it (BLD_1539, often
    re-cropped); the reveal lands much harder in colour, so register and use that."""
    im = cv2.imread(src)
    if im is None or _sat(im) > 45:                     # toned B&W still reads ~35
        return src
    stem = re.match(r'(.*?\d{3,})', os.path.splitext(os.path.basename(src))[0])
    if not stem:
        return src
    for cand in sorted(glob.glob(os.path.join(os.path.dirname(src), glob.escape(stem.group(1)) + '*.jpg'))):
        c = cv2.imread(cand)
        if cand == src or c is None or _sat(c) <= max(60, 2.5 * _sat(im)):
            continue
        M = _register(c, im)
        if M is not None:
            print(f'original is B&W; using colour master {os.path.basename(cand)} (registered)')
            _REG[cand] = (M, (im.shape[1], im.shape[0]))
            return cand
    return src


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
    if src:
        src = colour_sibling(src)
    model = fav.get('model') or side.get('model')
    if not model:
        m = re.search(r'/_photos/([^/]+)/', (side.get('relit') or src or ''))
        model = m.group(1) if m else None
    return src, model


# ---------- face ----------
_fa = None


def detect_face(img):
    """Largest face bbox [x0, y0, x1, y1]. Tilted heads (and faces under projected
    patterns) slip past the detector upright, so retry on rotated copies."""
    global _fa
    if _fa is None:
        from insightface.app import FaceAnalysis
        _fa = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'],
                           providers=['CPUExecutionProvider'])
        _fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=.3)
    h, w = img.shape[:2]
    best = None                                        # (score, angle, bbox upright)
    for ang in (0, 35, -35, 60, -60, 90, -90):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
        im = img if ang == 0 else cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        faces = _fa.get(im)
        if not faces:
            continue
        f = max(faces, key=lambda f: f.det_score)
        # box from the 5 keypoints (eyes, nose, mouth corners): the detector's own box
        # drifts onto hair on tilted / low-confidence faces
        kp = cv2.transform(np.float32(f.kps)[None], cv2.invertAffineTransform(M))[0]
        eyes, mouth = kp[:2].mean(0), kp[3:5].mean(0)
        cx, cy = (eyes + mouth) / 2
        half = 1.35 * np.hypot(*(mouth - eyes))
        box = [float(cx - half), float(cy - half * 1.1), float(cx + half), float(cy + half * .9)]
        if best is None or f.det_score > best[0]:
            best = (float(f.det_score), ang, box)
        if f.det_score >= .7:                          # confident: stop looking
            break
    if best is None:
        return None
    if best[1]:
        print(f'face found at {best[1]} deg rotation (score {best[0]:.2f})')
    return best[2]


def face_from_pose(img):
    """Last resort: eyes/ears/nose landmarks -> face box."""
    import mediapipe as mp
    detect_body(img)
    r = _pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks:
        return None
    h, w = img.shape[:2]
    pts = np.float32([(l.x * w, l.y * h) for l in r.pose_landmarks[0][:11] if l.visibility > .3])
    if len(pts) < 3:
        return None
    cx, cy = pts.mean(0)
    half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * .75
    return [float(cx - half), float(cy - half), float(cx + half), float(cy + half)]


# ---------- body ----------
_pose = None


def detect_body(img):
    """Pose landmarks -> (visible points [(x, y)], named parts {name: (x, y)}) in image px."""
    global _pose
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if _pose is None:
        _pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(ASSETS, 'pose_landmarker_full.task')),
            num_poses=1))
    r = _pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks:
        return [], {}
    h, w = img.shape[:2]
    lm = [(l.x * w, l.y * h, l.visibility) for l in r.pose_landmarks[0]]
    pts = [(x, y) for x, y, v in lm if v > .5 and -.05 * w < x < 1.05 * w and -.05 * h < y < 1.05 * h]
    named = {'left hand': 15, 'right hand': 16, 'left elbow': 13, 'right elbow': 14,
             'hips': (23, 24), 'left knee': 25, 'right knee': 26, 'left foot': 27, 'right foot': 28}
    parts = {}
    for name, idx in named.items():
        ids = idx if isinstance(idx, tuple) else (idx,)
        if all(lm[i][2] > .5 for i in ids):
            parts[name] = (float(np.mean([lm[i][0] for i in ids])), float(np.mean([lm[i][1] for i in ids])))
    return pts, parts


ASPECTS = {'9:16': 9 / 16, '4:5': 4 / 5, '1:1': 1.0}


def choose_crop(w, h, face_bb, body_pts, want='auto'):
    """Tallest frame that still holds the whole figure (hands included): 9:16 fills the
    phone, 4:5 and 1:1 give up screen to keep wide poses whole. -> (x0, y0, cw, ch, name, why)."""
    fx0, fy0, fx1, fy1 = face_bb
    fw = fx1 - fx0
    xs = [p[0] for p in body_pts] + [fx0 - .6 * fw, fx1 + .6 * fw]
    ys = [p[1] for p in body_pts] + [fy0 - .5 * fw, fy1]
    bx0, bx1 = max(min(xs), 0), min(max(xs), w)
    need_w = (bx1 - bx0) * 1.12
    options = [want] if want != 'auto' else ['9:16', '4:5', '1:1']
    for name in options:
        a = ASPECTS[name]
        ch = h if w / h > a else int(w / a)
        cw = int(ch * a)
        if cw >= need_w or name == options[-1]:
            why = (f'figure spans {int(bx1 - bx0)}px of a {cw}px-wide {name} crop' if want == 'auto'
                   else 'forced')
            break
    x0 = int(np.clip((bx0 + bx1) / 2 - cw / 2, 0, w - cw))
    top = min(min(ys), fy0 - .5 * fw)                    # keep headroom above the face
    y0 = int(np.clip(min(top - .04 * ch, (fy0 + fy1) / 2 - .3 * ch), 0, h - ch))
    return x0, y0, cw, ch, name, why


# ---------- layout ----------
def band_rules(face, target, SW, SH):
    """4 bands from the far `target` point toward the face, then the face's
    neighbourhood split in two (labels 4 and 5)."""
    fx, fy = face[:2]
    dx, dy = target[0] - fx, target[1] - fy
    n = max(np.hypot(dx, dy), 1)
    far = max(((cx - fx) * dx + (cy - fy) * dy) / n for cx in (0, SW) for cy in (0, SH))
    near = 170 * max(SW, SH) / 1280
    cuts = np.linspace(near, far, 5)
    rules = [{"label": 5, "band": [fx, fy, dx, dy, -9999, near], "side": [fx, fy, -dy, dx]}]
    for k in range(4):                         # outermost band tears first
        rules.append({"label": 3 - k, "band": [fx, fy, dx, dy, float(cuts[k]),
                                                9999 if k == 3 else float(cuts[k + 1])]})
    return {"kind": "rules", "default": 4, "rules": rules}


def blocks_rules(face, SW, SH):
    fx, fy = face[:2]
    kx, ky = SW / 1280, SH / 1280
    L, R, T = fx - 268 * kx, fx + 282 * kx, fy + 150 * ky
    return {"kind": "rules", "default": 4, "rules": [
        {"label": 3, "rect": [R, -9999, 9999, fy + 390 * ky]},
        {"label": 2, "rect": [-9999, -9999, L, 9999]},
        {"label": 1, "rect": [R, fy + 390 * ky, 9999, 9999]},
        {"label": 0, "rect": [L, fy + 570 * ky, R, 9999]},
        {"label": 5, "rect": [L, -9999, R, T]},
    ]}


def layout(kind, face, rng, SW, SH):
    fx, fy = face[:2]
    pan = {"from": [SW / 2, SH], "to": [SW / 2, 0]}
    if kind == 'blocks':
        return blocks_rules(face, SW, SH), pan
    if kind == 'shards':
        return {"kind": "shards", "n": int(rng.integers(13, 19)), "seed": int(rng.integers(1e6))}, pan
    if kind == 'strips':
        return {"kind": "strips", "angle": float(rng.choice([rng.uniform(60, 120), rng.uniform(15, 40),
                                                             rng.uniform(140, 165)])),
                "width": max(SW, SH) / rng.uniform(8, 12), "seed": int(rng.integers(1e6))}, pan
    if kind == 'wedges':
        return {"kind": "wedges", "n": 10, "seed": int(rng.integers(1e6))}, pan
    if kind == 'diagonal':
        tx, ty = max([(0, 0), (SW, 0), (0, SH), (SW, SH)], key=lambda c: np.hypot(c[0] - fx, c[1] - fy))
    elif kind == 'sweep-h':
        tx, ty = (SW if fx < SW / 2 else 0), fy
    else:
        tx, ty = fx, (SH if fy < SH / 2 else 0)
    return band_rules(face, (tx, ty), SW, SH), {"from": [tx, ty], "to": [2 * fx - tx, 2 * fy - ty]}


def sliver(face, rng, SW, SH):
    """Long jagged dagger: an irregular hexagon round the face + a narrow tip on an
    edge; convex hull, so the face is always fully under it and it never reads as a circle."""
    fx, fy, rx, ry = face
    best = None
    for a in rng.permutation(np.radians([-60, -120, -30, -150, 0, 180, -90])):
        dx, dy = np.cos(a), np.sin(a)
        d = min((0 - fx) / dx if dx < 0 else (SW - fx) / dx if dx > 0 else np.inf,
                (0 - fy) / dy if dy < 0 else (SH - fy) / dy if dy > 0 else np.inf)
        if 2.2 * max(rx, ry) < d < .8 * max(SW, SH):
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


def pick(opt, choices, seed):
    return opt if opt != 'auto' else choices[seed % len(choices)]


# ---------- main ----------
def build(start, args, idx):
    seed = args.seed + idx
    rng = np.random.default_rng(seed)
    end, model = find_original(start)
    if args.end:
        end = args.end
    if not end:
        raise SystemExit(f'no original found for {start} (pass --end)')
    A, B = cv2.imread(start), cv2.imread(end)
    h, w = A.shape[:2]
    derived = False                                    # END differs from the file on disk
    if end in _REG:                                    # colour master -> the edit's frame
        M, size = _REG[end]
        B, derived = cv2.warpAffine(B, M, size, flags=cv2.INTER_CUBIC), True
    if B.shape[:2] != (h, w):
        derived = True
        ar_a, ar_b = w / h, B.shape[1] / B.shape[0]
        if abs(ar_a - ar_b) / ar_a > .02:
            print(f'WARNING: aspect differs ({ar_a:.3f} vs {ar_b:.3f}); stretching original')
        B = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)
    bb = detect_face(B) or detect_face(A) or face_from_pose(B)
    if bb is None:
        raise SystemExit(f'no face found in {end}')
    body, parts = detect_body(B)
    x0, y0, cw, ch, aspect, why = choose_crop(w, h, bb, body, args.aspect)
    SW, SH = work_size(cw / ch)
    k = SW / cw
    fcx, fcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    face = [round((fcx - x0) * k), round((fcy - y0) * k),
            round((bb[2] - bb[0]) * .62 * k), round((bb[3] - bb[1]) * .72 * k)]
    wparts = {n: [round((x - x0) * k), round((y - y0) * k)] for n, (x, y) in parts.items()
              if 0 <= (x - x0) * k < SW and 0 <= (y - y0) * k < SH}

    kind = pick(args.layout, LAYOUTS, seed)
    regions, pan = layout(kind, face, rng, SW, SH)
    sl, pivot = sliver(face, rng, SW, SH)
    cmode = pick(args.camera, CAMERAS, seed // 2 + idx)

    presets = json.load(open(os.path.join(ASSETS, 'music_presets.json')))
    mname = pick(args.music, sorted(presets), seed)
    mus = dict(presets[mname])
    mus['t1'] = round(mus['t0'] + 7.0, 2)
    tears = [[t, None, 1 if i % 2 else -1, 1.0] for i, t in enumerate(beat_tears(mus, os.path.join(ASSETS, mus['file'])))]
    tears.append([round(mus['break'] + .25, 2), None, -1, .45])
    tears.append([round(mus['drop'] - .02, 2), float(np.degrees(np.arctan2(pivot[1] - face[1], pivot[0] - face[0])) + 180), 1, 3.2])

    top, drop = TEXTS[seed % len(TEXTS)]
    top, drop = args.text_top or top, args.text_drop or drop
    cfg = {
        "start": start, "end": end, "crop": [x0, y0, cw, ch], "face": face,
        "parts": list(wparts.values()),
        "regions": regions, "sliver": sl, "sliver_pivot": pivot,
        "sliver_lift": round(min(1.35, mus['drop'] - mus['break'] - .37), 2),
        "tears": tears, "music": mus,
        "camera": {"mode": cmode, "zoom": 1.3, "transition": "mixed", "seed": seed, **pan},
        "text": [
            {"text": top, "t": mus['t0'] + .1, "size": 128, "color": [255, 255, 255], "row": 0, "pop": .25, "dur": .18},
            {"text": drop, "t": mus['drop'], "size": 150, "color": accent(A), "row": 1, "pop": .5, "dur": .12},
        ],
        "_meta": {"aspect": aspect, "aspect_why": why, "layout": kind, "camera": cmode,
                  "music_preset": mname, "model": model, "seed": seed, "parts": sorted(wparts)},
    }
    return cfg, B if derived else None


def work_size(aspect):
    """Must match paper_rip_reel.work_size."""
    if abs(aspect - 1) < 1e-3:
        return 1280, 1280
    sh = 1920 if aspect < .7 else 1600
    return int(round(sh * aspect)), sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', nargs='+', required=True)
    ap.add_argument('--end', help='override the original (single source only)')
    ap.add_argument('--music', default='auto')
    ap.add_argument('--layout', default='auto', choices=['auto'] + LAYOUTS)
    ap.add_argument('--aspect', default='auto', choices=['auto'] + list(ASPECTS))
    ap.add_argument('--camera', default='auto', choices=['auto', 'cuts', 'whip', 'drift', 'pan'])
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
        m = cfg['_meta']
        base = f"{time.strftime('%Y-%m-%d')}_{model}_paperrip_{tag[:48]}_{m['layout']}-{m['camera']}"
        name, n = base, 2
        while os.path.exists(os.path.join(args.local_output_dir, name)):   # never overwrite a reel
            name, n = f'{base}_{n}', n + 1
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
        print(f"[{name}] aspect={m['aspect']} ({m['aspect_why']}) layout={m['layout']} camera={m['camera']} "
              f"shots on {m['parts']} music={m['music_preset']} "
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
