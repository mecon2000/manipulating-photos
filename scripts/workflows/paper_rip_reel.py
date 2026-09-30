"""Paper-rip reel: the stylized image is a printed sheet lying over the original photo.
On each beat a crack runs around a piece, the piece peels off, falls and casts a shadow;
the last piece (a long jagged sliver that happens to cover the face) lifts through the
musical break and rips away on the drop. 1080x1920 IG reel, 1:1 image over a blurred fill.

Everything photo/track-specific lives in a JSON config (see paper_rip_configs/):
  start/end        stylized image / original photo (same framing)
  crop             [x0, y0, size] square crop in source pixels; geometry below is in
                   1280x1280 work units of that crop
  face             [cx, cy, rx, ry] face ellipse (used for camera + coverage check)
  regions          {"default": label, "rules": [{"label", "rect"?: [x0,y0,x1,y1],
                   "band"?: [ox,oy,dx,dy,pmin,pmax], "side"?: [ox,oy,nx,ny]}, ...]} applied in order on
                   noise-warped coords; labels 0..n-2 tear in order, n-1 is the sliver
  sliver, sliver_pivot   polygon of the last piece, and the point it peels up from
  tears            [[time, pull_deg, spin, speed], ...] one per label, song time
  music            {file, t0, t1, drop, break}  (break = where the push-in starts)
  camera           {zoom, from: [cx,cy], to: [cx,cy]} pan during the beats
  text             [{text, t, size, color[r,g,b], row, pop, dur}]

  python paper_rip_reel.py CONFIG.json OUT.mp4 [--silent] [--preview SHEET.jpg]
"""
import json, sys, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ASSETS = '/home/rong/.openclaw/workspace/shared/ig-reels/_assets/'
cfg = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
SILENT = '--silent' in sys.argv
PREVIEW = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None

S, W, H, FPS, SQ_Y = 1280, 1080, 1920, 30, 220
cx0, cy0, cs = cfg['crop']


def load(p):
    im = cv2.imread(p)[cy0:cy0 + cs, cx0:cx0 + cs]
    return cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) if cs != S \
        else im.astype(np.float32)


A, B = load(cfg['start']), load(cfg['end'])
FACE = cfg['face']
mus = cfg['music']
T0, T1, DROP, BRK = mus['t0'], mus['t1'], mus['drop'], mus['break']
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)


def noise(scale, seed, size=S):
    r = np.random.default_rng(seed).random((size // scale + 2, size // scale + 2)).astype(np.float32)
    return cv2.resize(r, (size, size), interpolation=cv2.INTER_CUBIC)


# printed-paper grain on the START sheet
A = A * (0.94 + 0.08 * noise(2, 99)[..., None]) * (0.97 + 0.05 * noise(60, 98)[..., None])

# ---- torn-region label map on multi-octave warped coords ----
wx = xx + 70 * (noise(45, 1) - .5) + 12 * (noise(22, 3) - .5) + 4 * (noise(5, 5) - .5)
wy = yy + 70 * (noise(45, 2) - .5) + 12 * (noise(22, 4) - .5) + 4 * (noise(5, 6) - .5)
face_d = ((xx - FACE[0]) / FACE[2]) ** 2 + ((yy - FACE[1]) / FACE[3]) ** 2
lab = np.full((S, S), cfg['regions']['default'], np.uint8)
for r in cfg['regions']['rules']:
    m = np.ones((S, S), bool)
    if 'rect' in r:
        x0, y0, x1, y1 = r['rect']
        m &= (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
    if 'band' in r:
        ox, oy, dx, dy, lo, hi = r['band']
        n = np.hypot(dx, dy); p = ((wx - ox) * dx + (wy - oy) * dy) / n
        m &= (p >= lo) & (p < hi)
    if 'side' in r:                                      # half-plane: which side of a line
        ox, oy, px, py = r['side']
        m &= ((wx - ox) * px + (wy - oy) * py) >= 0
    lab[m] = r['label']
SLV = len(cfg['tears']) - 1
sl = np.zeros((S, S), np.float32); cv2.fillPoly(sl, [np.int32(cfg['sliver'])], 1)
sx = xx + 10 * (noise(22, 60) - .5) + 4 * (noise(5, 61) - .5)   # fine tear jitter only:
sy = yy + 10 * (noise(22, 62) - .5) + 4 * (noise(5, 63) - .5)   # keeps the long straight-ish rip
lab[cv2.remap(sl, sx, sy, cv2.INTER_LINEAR) > .5] = SLV
print('face under the sliver: %.0f%%' % (100 * (lab[face_d < 1] == SLV).mean()))

TEARS = cfg['tears']
LIFE = .75                     # seconds a piece stays visible after tearing

# split each region into 2 fragments along a rough line -> messier, more paper-like
frags = []
for k, (t, ang, spin, spd) in enumerate(TEARS):
    m = lab == k
    if not m.any(): continue
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    if k == SLV:
        parts = [m]
    else:
        a = np.radians(np.random.default_rng(20 + k).uniform(0, 180))
        side = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a) + 40 * (noise(15, 30 + k) - .5)
        parts = [m & (side < 0), m & (side >= 0)]
    for j, p in enumerate(parts):
        if p.sum() < 500: continue
        ys, xs = np.nonzero(p)
        if ang is None:                                  # auto: pull outward, away from the face
            ang = np.degrees(np.arctan2(ys.mean() - FACE[1], xs.mean() - FACE[0]))
        frags.append(dict(mask=p, t0=t + .06 * j, dir=np.radians(ang + 25 * (j - .5)),
                          spin=spin * (1 if j == 0 else -.6), spd=spd,
                          c=(xs.mean(), ys.mean())))

# tear propagation: before a piece comes off, a crack runs along its outline
for k, f in enumerate(frags):
    m8 = f['mask'].astype(np.uint8)
    f['sd'] = cv2.distanceTransform(m8, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
    cx, cy = f['c']
    a0 = np.random.default_rng(80 + k).uniform(-np.pi, np.pi)
    f['ord'] = ((np.arctan2(yy - cy, xx - cx) - a0) / (2 * np.pi)) % 1.0
    f['tp'] = cfg.get('sliver_lift', 1.35) if k == len(frags) - 1 else .42
RIMW = 2 + 7 * np.clip(noise(9, 40), 0, 1) ** 1.5              # torn-fibre width map (px)
FIBER = 222 + 30 * noise(2, 41)                                # off-white paper core with fibres


def torn_rgba(mask, img):
    """Colour + alpha for a sheet: image inside, white torn fibre rim along the tear edges."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    rim = np.clip((RIMW - dist) / 2, 0, 1)[..., None]
    return img * (1 - rim) + FIBER[..., None] * rim, np.clip(dist / 1.2, 0, 1)


def shadow(alpha, dx, dy, blur):
    sh = cv2.GaussianBlur(alpha, (0, 0), max(blur, .5))
    return cv2.warpAffine(sh, np.float32([[1, 0, dx], [0, 1, dy]]), (S, S))


_static, _frag = {}, {}


def static_layer(torn_ids):
    key = tuple(torn_ids)
    if key not in _static:
        m = np.ones((S, S), bool)
        for i in torn_ids: m &= ~frags[i]['mask']
        col, al = torn_rgba(m, A)
        _static.clear(); _static[key] = (col, al, shadow(al, 5, 9, 5))
    return _static[key]


def frag_rgba(i):
    if i not in _frag: _frag[i] = torn_rgba(frags[i]['mask'], A)
    return _frag[i]


def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def compose(ts):
    last = len(frags) - 1                                # the sliver lifts off during the break
    torn = [i for i, f in enumerate(frags) if ts >= f['t0'] - (f['tp'] if i == last else 0)]
    col, al, sh = static_layer(torn)
    img = B * (1 - .55 * sh[..., None])                  # sheet's contact shadow on the photo
    img = img * (1 - al[..., None]) + col * al[..., None]

    for i, f in enumerate(frags):                        # cracks growing before the rip
        if i == last: continue
        u = (ts - (f['t0'] - f['tp'])) / f['tp']
        if not 0 < u < 1: continue
        grow = np.clip((u ** 1.4 * 1.02 - f['ord']) / .06, 0, 1)   # accelerates into the beat
        ad = np.abs(f['sd'])
        fib = np.clip((RIMW * grow - ad) / 1.5, 0, 1)[..., None]
        gap = np.clip((1.6 * grow - ad) / 1., 0, 1)[..., None]
        img = img * (1 - fib) + FIBER[..., None] * fib
        img = img * (1 - gap) + B * .55 * gap            # thin split showing the photo underneath

    for i in torn:                                       # lifting / flying pieces
        f = frags[i]; tau = ts - f['t0']
        if tau > LIFE: continue
        pc, pa = frag_rgba(i)
        if tau < 0:                                      # sliver peeling up from its tail, trembling
            u = 1 + tau / f['tp']; lift = .5 * u ** 2
            M = cv2.getRotationMatrix2D(tuple(cfg['sliver_pivot']),
                                        f['spin'] * (3 * u ** 2 + .8 * u * np.sin(ts * 45)), 1 + .05 * u ** 2)
            wc = cv2.warpAffine(pc, M, (S, S)); wa = cv2.warpAffine(pa, M, (S, S))
            s = shadow(wa, 5 + 40 * lift, 9 + 60 * lift, 5 + 18 * lift) * .6
            img = img * (1 - s[..., None])
            img = img * (1 - wa[..., None]) + wc * wa[..., None]
            continue
        g, v0 = 2200 * f['spd'], 160 * f['spd']
        dx = v0 * tau * np.cos(f['dir'])
        dy = v0 * tau * np.sin(f['dir']) + .5 * g * tau * tau
        lift = smooth(tau / .4) * (1.0 if i < last else 2.5)
        if i == last: lift = max(lift, .5)
        sc = 1 + .22 * lift + .15 * tau
        rot = f['spin'] * (55 * tau + 10 * smooth(tau / .15))
        fade = 1 - smooth((tau - LIFE * .55) / (LIFE * .45))
        if i == last: fade = 1 - smooth((tau - .08) / .14)
        M = cv2.getRotationMatrix2D(f['c'], rot, sc); M[:, 2] += (dx, dy)
        wc = cv2.warpAffine(pc, M, (S, S)); wa = cv2.warpAffine(pa, M, (S, S)) * fade
        # the lifted piece casts a softer, farther shadow the higher it gets
        s = shadow(wa, 10 + 45 * lift, 16 + 70 * lift, 6 + 22 * lift) * (.6 - .25 * lift / 2.5)
        img = img * (1 - s[..., None])
        img = img * (1 - wa[..., None]) + wc * (1 - .12 * lift / 2.5) * wa[..., None]
    return img


CAM = cfg['camera']


def clamp(cx, cy, z):
    half = S / (2 * z)
    return np.clip(cx, half, S - half), np.clip(cy, half, S - half)


def camera(ts):
    """Pan with the tears, push in on the face through the break, pull out to full frame after the drop."""
    z0 = CAM['zoom']
    fx, fy = FACE[0], FACE[1] + 40
    ex, ey = clamp(*CAM['to'], z0)
    if ts < BRK:
        u = smooth((ts - T0) / (BRK - T0)); z = z0
        bx, by = clamp(*CAM['from'], z0)
        cx, cy = bx + (ex - bx) * u, by + (ey - by) * u
    elif ts < DROP:
        u = smooth((ts - BRK) / (DROP - BRK)); z = z0 + .45 * u
        cx, cy = ex + (fx - ex) * u, ey + (fy - ey) * u
    else:
        u = 1 - (1 - min((ts - DROP) / 1.3, 1)) ** 3
        z = z0 + .45 + (1.0 - z0 - .45) * u
        cx, cy = fx + (S / 2 - fx) * u, fy + (S / 2 - fy) * u
    for f in TEARS[:-2]:
        if ts >= f[0]: z += .03 * np.exp(-(ts - f[0]) / .12)
    if ts >= DROP: z += .08 * np.exp(-(ts - DROP) / .1)
    cx, cy = clamp(cx, cy, z)
    if DROP <= ts < DROP + .25:
        k = np.exp(-(ts - DROP) / .07); cx += 12 * k * np.sin(ts * 170); cy += 9 * k * np.cos(ts * 130)
    return z, cx, cy


TEXT = []
for tx in cfg['text']:
    font = ImageFont.truetype(ASSETS + 'Anton.ttf', tx['size'])
    im = Image.new('RGBA', (W, 260), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    tw = dr.textlength(tx['text'], font=font)
    if tw > W - 160:                                     # auto-fit long lines (IG-safe margins)
        font = ImageFont.truetype(ASSETS + 'Anton.ttf', int(tx["size"] * (W - 160) / tw))
        tw = dr.textlength(tx['text'], font=font)
    dr.text(((W - tw) / 2, 40), tx['text'], font=font, fill=tuple(tx['color']) + (255,),
            stroke_width=4, stroke_fill=(0, 0, 0, 255))
    TEXT.append((tx, np.array(im).astype(np.float32)))


def frame(ts):
    img = compose(ts)
    z, cx, cy = camera(ts)
    k = W / S * z
    M = np.float32([[k, 0, W / 2 - k * cx], [0, k, W / 2 - k * cy]])
    sq = cv2.warpAffine(img, M, (W, W), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    bg = cv2.resize(sq, (W // 12, W // 12), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 3), (H, H), interpolation=cv2.INTER_CUBIC)
    canvas = bg[:, (H - W) // 2:(H - W) // 2 + W] * .45
    canvas[SQ_Y:SQ_Y + W] = sq
    if DROP <= ts < DROP + .3:
        canvas += 255 * .7 * np.exp(-(ts - DROP) / .06)
    for tx, layer in TEXT:
        if ts < tx['t']: continue
        u = 1 - (1 - min((ts - tx['t']) / tx['dur'], 1)) ** 3
        a = cv2.warpAffine(layer, cv2.getRotationMatrix2D((W / 2, 130), 0, 1 + tx['pop'] * (1 - u)), (W, 260))
        rgb, al = a[..., 2::-1], a[..., 3:4] * u / 255.
        y = SQ_Y + W + (20 if tx['row'] == 0 else 150)
        canvas[y:y + 260] = canvas[y:y + 260] * (1 - al) + rgb * al
    return np.clip(canvas, 0, 255).astype(np.uint8)


if PREVIEW:
    ts_list = np.linspace(T0 + .1, T1 - .1, 10)
    ims = [cv2.resize(frame(t), (270, 480)) for t in ts_list]
    cv2.imwrite(PREVIEW, np.vstack([np.hstack(ims[:5]), np.hstack(ims[5:])]))
    sys.exit()

import static_ffmpeg; static_ffmpeg.add_paths()
audio = [] if SILENT else ['-ss', str(T0), '-t', str(T1 - T0), '-i', ASSETS + mus['file'],
                           '-af', f'afade=t=out:st={T1 - T0 - .35:.2f}:d=0.35',
                           '-c:a', 'aac', '-b:a', '256k', '-shortest']
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
                      '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      *audio[:6], '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
                      '-profile:v', 'high', '-movflags', '+faststart', *audio[6:], OUT], stdin=subprocess.PIPE)
for i in range(int(round((T1 - T0) * FPS))):
    p.stdin.write(frame(T0 + i / FPS).tobytes())
p.stdin.close(); p.wait()
print('wrote', OUT)
