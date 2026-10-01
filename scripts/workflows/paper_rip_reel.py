"""Paper-rip reel: the stylized image is a printed sheet lying over the original photo.
On each beat a crack runs round a piece, the piece peels off, falls and casts a shadow;
the last piece (a long jagged sliver that happens to cover the face) lifts through the
musical break and rips away on the drop. 1080x1920 IG reel.

Everything photo/track-specific lives in a JSON config (see paper_rip_configs/,
paper_rip_auto.py builds them):
  start/end        stylized image / original photo (same framing)
  crop             [x0, y0, size] (square) or [x0, y0, w, h]; the crop's aspect is the
                   on-screen frame: 9:16 fills the screen, 4:5 / 1:1 sit over a blurred fill.
                   Geometry below is in work units of that crop (1280 square; else
                   1920/1600 tall, see work_size)
  face             [cx, cy, rx, ry] face ellipse
  parts            optional [[x, y], ...] body points (hands, knees...) for camera shots
  regions          tear layout, see paper_rip_shapes.py (kind: rules|shards|strips|wedges)
  sliver, sliver_pivot, sliver_lift   last piece polygon, its peel-up point, lift seconds
  tears            [[time, pull_deg|null, spin, speed], ...] per label 0..6 (null = outward)
  music            {file, t0, t1, drop, break}
  camera           see paper_rip_camera.py (mode: pan|cuts|whip|drift)
  text             [{text, t, size, color[r,g,b], row, pop, dur}]

  python paper_rip_reel.py CONFIG.json OUT.mp4 [--silent] [--preview SHEET.jpg]
"""
import json, os, sys, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rip_camera as camera_mod
import paper_rip_shapes as shapes

ASSETS = '/home/rong/.openclaw/workspace/shared/ig-reels/_assets/'
cfg = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
SILENT = '--silent' in sys.argv
PREVIEW = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None

W, H, FPS = 1080, 1920, 30
crop = cfg['crop']
cx0, cy0, cw = crop[:3]
ch = crop[3] if len(crop) > 3 else cw


def work_size(aspect):
    if abs(aspect - 1) < 1e-3:
        return 1280, 1280
    sh = 1920 if aspect < .7 else 1600
    return int(round(sh * aspect)), sh


SW, SH = work_size(cw / ch)
OH = min(H, int(round(W * ch / cw)))                   # on-screen frame
BOX_Y = 0 if OH >= H - 20 else (220 if OH <= W else 120)
OH = H if BOX_Y == 0 else OH


def load(p):
    im = cv2.imread(p)[cy0:cy0 + ch, cx0:cx0 + cw]
    return cv2.resize(im, (SW, SH), interpolation=cv2.INTER_AREA).astype(np.float32) \
        if (cw, ch) != (SW, SH) else im.astype(np.float32)


A, B = load(cfg['start']), load(cfg['end'])
FACE = cfg['face']
mus = cfg['music']
T0, T1, DROP, BRK = mus['t0'], mus['t1'], mus['drop'], mus['break']
yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
M_ = max(SW, SH)


def noise(scale, seed):
    r = np.random.default_rng(seed).random((M_ // scale + 2, M_ // scale + 2)).astype(np.float32)
    return cv2.resize(r, (M_, M_), interpolation=cv2.INTER_CUBIC)[:SH, :SW]


# printed-paper grain on the START sheet
A = A * (0.94 + 0.08 * noise(2, 99)[..., None]) * (0.97 + 0.05 * noise(60, 98)[..., None])

wx = xx + 70 * (noise(45, 1) - .5) + 12 * (noise(22, 3) - .5) + 4 * (noise(5, 5) - .5)
wy = yy + 70 * (noise(45, 2) - .5) + 12 * (noise(22, 4) - .5) + 4 * (noise(5, 6) - .5)
face_d = ((xx - FACE[0]) / FACE[2]) ** 2 + ((yy - FACE[1]) / FACE[3]) ** 2
sl = np.zeros((SH, SW), np.float32); cv2.fillPoly(sl, [np.int32(cfg['sliver'])], 1)
sx = xx + 10 * (noise(22, 60) - .5) + 4 * (noise(5, 61) - .5)   # fine tear jitter only:
sy = yy + 10 * (noise(22, 62) - .5) + 4 * (noise(5, 63) - .5)   # keeps the long straight-ish rip
g = dict(SW=SW, SH=SH, xx=xx, yy=yy, wx=wx, wy=wy, noise=noise, face=FACE,
         sliver=cv2.remap(sl, sx, sy, cv2.INTER_LINEAR) > .5)
lab, piece = shapes.build(cfg, g)
SLV = 6
print('face under the sliver: %.0f%%' % (100 * (lab[face_d < 1] == SLV).mean()))

TEARS = cfg['tears']
LIFE = .75                     # seconds a piece stays visible after tearing
PAD = 24

frags = []
for k, (t, ang, spin, spd) in enumerate(TEARS):
    ids = np.unique(piece[lab == k])
    for j, pid in enumerate(ids):
        m = (lab == k) & (piece == pid)
        if m.sum() < 500: continue
        ys, xs = np.nonzero(m)
        a = ang
        if a is None:                                    # auto: pull outward, away from the face
            a = np.degrees(np.arctan2(ys.mean() - FACE[1], xs.mean() - FACE[0]))
        x0, y0 = max(xs.min() - PAD, 0), max(ys.min() - PAD, 0)
        x1, y1 = min(xs.max() + PAD + 1, SW), min(ys.max() + PAD + 1, SH)
        frags.append(dict(label=k, mask=m, t0=t + .06 * j, dir=np.radians(a + 25 * (j - .5)),
                          spin=spin * (1 if j % 2 == 0 else -.6), spd=spd,
                          c=(xs.mean(), ys.mean()), bbox=(x0, y0, x1, y1)))

# tear propagation: before a piece comes off, a crack runs along its outline
for k, f in enumerate(frags):
    x0, y0, x1, y1 = f['bbox']
    m8 = f['mask'][y0:y1, x0:x1].astype(np.uint8)
    f['sd'] = cv2.distanceTransform(m8, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
    cx, cy = f['c']
    a0 = np.random.default_rng(80 + k).uniform(-np.pi, np.pi)
    f['ord'] = ((np.arctan2(yy[y0:y1, x0:x1] - cy, xx[y0:y1, x0:x1] - cx) - a0) / (2 * np.pi)) % 1.0
    f['tp'] = cfg.get('sliver_lift', 1.35) if f['label'] == SLV else .42
RIMW = 2 + 7 * np.clip(noise(9, 40), 0, 1) ** 1.5              # torn-fibre width map (px)
FIBER = 222 + 30 * noise(2, 41)                                # off-white paper core with fibres


def torn_rgba(mask, img, rimw, fiber):
    """Colour + alpha for a sheet: image inside, white torn fibre rim along the tear edges."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    rim = np.clip((rimw - dist) / 2, 0, 1)[..., None]
    return img * (1 - rim) + fiber[..., None] * rim, np.clip(dist / 1.2, 0, 1)


def shadow(alpha, dx, dy, blur):
    sh = cv2.GaussianBlur(alpha, (0, 0), max(blur, .5))
    return cv2.warpAffine(sh, np.float32([[1, 0, dx], [0, 1, dy]]), (SW, SH))


_static, _frag = {}, {}


def static_layer(torn_ids):
    key = tuple(torn_ids)
    if key not in _static:
        m = np.ones((SH, SW), bool)
        for i in torn_ids: m &= ~frags[i]['mask']
        col, al = torn_rgba(m, A, RIMW, FIBER)
        _static.clear(); _static[key] = (col, al, shadow(al, 5, 9, 5))
    return _static[key]


def frag_rgba(i):
    """Piece colour/alpha cropped to its bbox (full-frame buffers per piece would be GBs)."""
    if i not in _frag:
        x0, y0, x1, y1 = frags[i]['bbox']
        _frag[i] = torn_rgba(frags[i]['mask'][y0:y1, x0:x1], A[y0:y1, x0:x1],
                             RIMW[y0:y1, x0:x1], FIBER[y0:y1, x0:x1])
    return _frag[i]


def warp_piece(i, M):
    x0, y0 = frags[i]['bbox'][:2]
    M = M.copy(); M[:, 2] += M[:, :2] @ np.float64([x0, y0])   # bbox-local -> frame
    pc, pa = frag_rgba(i)
    return cv2.warpAffine(pc, M, (SW, SH)), cv2.warpAffine(pa, M, (SW, SH))


def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def compose(ts):
    torn = [i for i, f in enumerate(frags) if ts >= f['t0'] - (f['tp'] if f['label'] == SLV else 0)]
    col, al, sh = static_layer(torn)
    img = B * (1 - .55 * sh[..., None])                  # sheet's contact shadow on the photo
    img = img * (1 - al[..., None]) + col * al[..., None]

    for f in frags:                                      # cracks growing before the rip
        if f['label'] == SLV: continue
        u = (ts - (f['t0'] - f['tp'])) / f['tp']
        if not 0 < u < 1: continue
        x0, y0, x1, y1 = f['bbox']
        grow = np.clip((u ** 1.4 * 1.02 - f['ord']) / .06, 0, 1)   # accelerates into the beat
        ad = np.abs(f['sd'])
        fib = np.clip((RIMW[y0:y1, x0:x1] * grow - ad) / 1.5, 0, 1)[..., None]
        gap = np.clip((1.6 * grow - ad) / 1., 0, 1)[..., None]
        reg = img[y0:y1, x0:x1]
        reg = reg * (1 - fib) + FIBER[y0:y1, x0:x1, None] * fib
        img[y0:y1, x0:x1] = reg * (1 - gap) + B[y0:y1, x0:x1] * .55 * gap

    for i in torn:                                       # lifting / flying pieces
        f = frags[i]; tau = ts - f['t0']
        if tau > LIFE: continue
        if tau < 0:                                      # sliver peeling up from its tail, trembling
            u = 1 + tau / f['tp']; lift = .5 * u ** 2
            M = cv2.getRotationMatrix2D(tuple(cfg['sliver_pivot']),
                                        f['spin'] * (3 * u ** 2 + .8 * u * np.sin(ts * 45)), 1 + .05 * u ** 2)
            wc, wa = warp_piece(i, M)
            s = shadow(wa, 5 + 40 * lift, 9 + 60 * lift, 5 + 18 * lift) * .6
            img = img * (1 - s[..., None])
            img = img * (1 - wa[..., None]) + wc * wa[..., None]
            continue
        g_, v0 = 2200 * f['spd'], 160 * f['spd']
        dx = v0 * tau * np.cos(f['dir'])
        dy = v0 * tau * np.sin(f['dir']) + .5 * g_ * tau * tau
        is_last = f['label'] == SLV
        lift = smooth(tau / .4) * (2.5 if is_last else 1.0)
        if is_last: lift = max(lift, .5)
        sc = 1 + .22 * lift + .15 * tau
        rot = f['spin'] * (55 * tau + 10 * smooth(tau / .15))
        fade = 1 - smooth((tau - .08) / .14) if is_last else 1 - smooth((tau - LIFE * .55) / (LIFE * .45))
        M = cv2.getRotationMatrix2D(f['c'], rot, sc); M[:, 2] += (dx, dy)
        wc, wa = warp_piece(i, M); wa = wa * fade
        # the lifted piece casts a softer, farther shadow the higher it gets
        s = shadow(wa, 10 + 45 * lift, 16 + 70 * lift, 6 + 22 * lift) * (.6 - .25 * lift / 2.5)
        img = img * (1 - s[..., None])
        img = img * (1 - wa[..., None]) + wc * (1 - .12 * lift / 2.5) * wa[..., None]
    return img


shots = camera_mod.plan_shots(frags, TEARS, cfg.get('parts', []), SW, SH, T0)
cam = camera_mod.make(cfg, dict(SW=SW, SH=SH, FACE=FACE, T0=T0, DROP=DROP, BRK=BRK, TEARS=TEARS, shots=shots))

TEXT = []
for tx in cfg['text']:
    font = ImageFont.truetype(ASSETS + 'Anton.ttf', tx['size'])
    im = Image.new('RGBA', (W, 260), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    tw = dr.textlength(tx['text'], font=font)
    if tw > W - 160:                                     # auto-fit long lines (IG-safe margins)
        font = ImageFont.truetype(ASSETS + 'Anton.ttf', int(tx['size'] * (W - 160) / tw))
        tw = dr.textlength(tx['text'], font=font)
    dr.text(((W - tw) / 2, 40), tx['text'], font=font, fill=tuple(tx['color']) + (255,),
            stroke_width=4, stroke_fill=(0, 0, 0, 255))
    TEXT.append((tx, np.array(im).astype(np.float32)))
TEXT_Y = (1320, 1450)
# where the frame runs under the text, darken it so the words stay readable
GRAD = (np.clip((np.arange(H) - 1150) / 250, 0, 1) * .55 * (np.arange(H) < BOX_Y + OH))[:, None, None]


def post_fx(sq, fx, k):
    if fx.get('blur'):
        bx, by = fx['blur'][0] * k, fx['blur'][1] * k
        n = int(min(max(abs(bx), abs(by)), 90))
        if n > 2:
            ker = np.zeros((2 * n + 1, 2 * n + 1), np.float32)
            cv2.line(ker, (int(n - bx / 2), int(n - by / 2)), (int(n + bx / 2), int(n + by / 2)), 1, 1)
            sq = cv2.filter2D(sq, -1, ker / ker.sum())
    if fx.get('glitch'):
        d = int(fx['glitch'])
        sq = sq.copy(); sq[:, :, 2] = np.roll(sq[:, :, 2], d, 1); sq[:, :, 0] = np.roll(sq[:, :, 0], -d, 1)
    return sq


def frame(ts):
    img = compose(ts)
    z, cx, cy, fx = cam(ts)
    k = W / SW * z
    M = np.float32([[k, 0, W / 2 - k * cx], [0, k, OH / 2 - k * cy]])
    sq = cv2.warpAffine(img, M, (W, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sq = post_fx(sq, fx, k)
    if OH < H:
        bg = cv2.resize(sq, (W // 12, OH // 12), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 3), (H, H), interpolation=cv2.INTER_CUBIC)
        canvas = bg[:, (H - W) // 2:(H - W) // 2 + W] * .45
        canvas[BOX_Y:BOX_Y + OH] = sq
    else:
        canvas = sq
    if BOX_Y + OH > 1300 and ts >= cfg['text'][0]['t']:
        canvas = canvas * (1 - GRAD * min((ts - cfg['text'][0]['t']) / .2, 1))
    if DROP <= ts < DROP + .3:
        canvas = canvas + 255 * .7 * np.exp(-(ts - DROP) / .06)
    if fx.get('flash'):
        canvas = canvas + 255 * fx['flash']
    for tx, layer in TEXT:
        if ts < tx['t']: continue
        u = 1 - (1 - min((ts - tx['t']) / tx['dur'], 1)) ** 3
        a = cv2.warpAffine(layer, cv2.getRotationMatrix2D((W / 2, 130), 0, 1 + tx['pop'] * (1 - u)), (W, 260))
        rgb, al = a[..., 2::-1], a[..., 3:4] * u / 255.
        y = TEXT_Y[0 if tx['row'] == 0 else 1]
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
