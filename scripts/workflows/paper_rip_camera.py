"""Camera for paper_rip_reel: where the view is at each moment, plus cut effects.

make(cfg, ctx) -> cam(ts) -> (zoom, cx, cy, fx)
  fx: {"blur": (dx, dy) px of directional motion blur, "glitch": px RGB split,
       "flash": 0..1 white, "shake": px}
ctx: SW, SH, FACE, T0, DROP, BRK, TEARS, shots (list of (t_start, cx, cy, zoom) per
     build-up tear, aimed at the piece about to rip, snapped to a body part when one is inside)

modes (cfg["camera"]["mode"])
  pan    the original: one slow pan from `from` to `to`
  cuts   hard jump cuts on the beat: a tight shot on each piece as its crack starts
         (legs -> hands -> head...), a slow creep inside each shot
  whip   same shots, joined by fast whip-pans with motion blur
  drift  one continuous move that swings and zooms from piece to piece
transition (cuts only): punch | glitch | flash | shake | mixed
Every mode then pushes in on the face through the break and snaps out on the drop.
"""
import numpy as np


def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def ease_out(x):
    return 1 - (1 - np.clip(x, 0, 1)) ** 3


def make(cfg, ctx):
    CAM = cfg['camera']
    mode = CAM.get('mode', 'pan')
    SW, SH, FACE = ctx['SW'], ctx['SH'], ctx['FACE']
    T0, DROP, BRK, TEARS = ctx['T0'], ctx['DROP'], ctx['BRK'], ctx['TEARS']
    z0 = CAM.get('zoom', 1.3)
    rng = np.random.default_rng(CAM.get('seed', 3))
    trans = CAM.get('transition', 'mixed')
    shots = ctx['shots']
    kinds = ['punch', 'glitch', 'flash', 'shake']
    cut_fx = [trans if trans != 'mixed' else kinds[i % 4] for i in rng.permutation(len(shots) + 2)]

    def clamp(cx, cy, z):
        hw, hh = SW / (2 * z), SH / (2 * z)
        return np.clip(cx, hw, SW - hw), np.clip(cy, hh, SH - hh)

    fx_, fy_ = FACE[0], FACE[1] + 40 * SH / 1280          # a touch below the eyes

    def build_phase(ts):
        """(z, cx, cy, fx) before the break."""
        fx = {}
        if mode == 'pan':
            u = smooth((ts - T0) / (BRK - T0))
            bx, by = clamp(*CAM['from'], z0)
            ex, ey = clamp(*CAM['to'], z0)
            return z0, bx + (ex - bx) * u, by + (ey - by) * u, fx
        # establishing shot until the first cut: the whole sheet, slowly closing in
        seq = [(T0, SW / 2, SH / 2, 1.08)] + shots
        i = max(k for k, s in enumerate(seq) if ts >= s[0])
        t, cx, cy, z = seq[i]
        nxt = seq[i + 1][0] if i + 1 < len(seq) else BRK
        u = (ts - t) / max(nxt - t, 1e-3)
        if mode == 'drift':                                  # continuous swing between shots
            if i + 1 < len(seq):
                _, ncx, ncy, nz = seq[i + 1]
                w = smooth(u)
                cx, cy, z = cx + (ncx - cx) * w, cy + (ncy - cy) * w, z + (nz - z) * w
            return z * (1 + .04 * np.sin(np.pi * u)), cx, cy, fx
        z = z * (1 + .07 * u)                                # slow creep inside the shot
        cx, cy = cx + (fx_ - cx) * .06 * u, cy + (fy_ - cy) * .06 * u
        dt = ts - t
        if mode == 'whip' and i + 1 < len(seq) and nxt - ts < .1:
            _, ncx, ncy, nz = seq[i + 1]                     # whip toward the next shot
            w = smooth(1 - (nxt - ts) / .1)
            fx['blur'] = ((ncx - cx) * w * .15, (ncy - cy) * w * .15)    # work px
            cx, cy, z = cx + (ncx - cx) * w, cy + (ncy - cy) * w, z + (nz - z) * w
        elif mode == 'whip' and i > 0 and dt < .08:
            _, pcx, pcy, _ = seq[i - 1]
            k = 1 - dt / .08
            fx['blur'] = ((cx - pcx) * k * .15, (cy - pcy) * k * .15)
        elif mode == 'cuts' and i > 0 and dt < .2:
            kind = cut_fx[i]
            k = np.exp(-dt / .06)
            if kind == 'punch':
                z *= 1 + .12 * k
            elif kind == 'glitch' and dt < .1:
                fx['glitch'] = 14 * k
            elif kind == 'flash' and dt < .07:
                fx['flash'] = .55 * k
            elif kind == 'shake':
                fx['shake'] = 16 * k
        return z, cx, cy, fx

    def cam(ts):
        fx = {}
        if ts < BRK:
            z, cx, cy, fx = build_phase(ts)
        elif ts < DROP:                                      # push in on the face through the break
            zb, bx, by, _ = build_phase(BRK - 1e-3)
            u = smooth((ts - BRK) / (DROP - BRK))
            if mode != 'pan':
                zb, bx, by = 1.25, SW / 2, SH / 2
                if ts - BRK < .15:                           # cut into the break
                    fx['flash'] = .35 * (1 - (ts - BRK) / .15)
            bx, by = clamp(bx, by, zb)
            z = zb + .45 * u
            cx, cy = bx + (fx_ - bx) * u, by + (fy_ - by) * u
        else:
            zb = (z0 if mode == 'pan' else 1.25) + .45
            u = ease_out((ts - DROP) / 1.3)
            z = zb + (1.0 - zb) * u
            cx, cy = fx_ + (SW / 2 - fx_) * u, fy_ + (SH / 2 - fy_) * u
        if mode == 'pan':
            for f in TEARS[:-2]:
                if ts >= f[0]: z += .03 * np.exp(-(ts - f[0]) / .12)
        if ts >= DROP: z += .08 * np.exp(-(ts - DROP) / .1)
        cx, cy = clamp(cx, cy, z)
        if DROP <= ts < DROP + .25:
            k = np.exp(-(ts - DROP) / .07); cx += 12 * k * np.sin(ts * 170); cy += 9 * k * np.cos(ts * 130)
        if fx.get('shake'):
            cx += fx['shake'] * np.sin(ts * 190); cy += fx['shake'] * np.cos(ts * 150)
        return z, cx, cy, fx

    return cam


def plan_shots(frags, tears, parts, SW, SH, t_open, tp=.42):
    """One tight shot per build-up tear: on the piece whose crack is about to run,
    centred on a body part (legs, hands, hips...) if one lies inside it."""
    shots = []
    for k, (t, *_rest) in enumerate(tears[:-2]):
        fs = [f for f in frags if f['label'] == k]
        if not fs:
            continue
        x0 = min(f['bbox'][0] for f in fs); y0 = min(f['bbox'][1] for f in fs)
        x1 = max(f['bbox'][2] for f in fs); y1 = max(f['bbox'][3] for f in fs)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = [p for p in parts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        if inside:
            p = min(inside, key=lambda p: np.hypot(p[0] - cx, p[1] - cy))
            cx, cy = (cx + 2 * p[0]) / 3, (cy + 2 * p[1]) / 3
        z = float(np.clip(.8 * min(SW / max(x1 - x0, 1), SH / max(y1 - y0, 1)), 1.35, 2.3))
        shots.append((t - tp, cx, cy, z))
    if shots:                                  # the opening frame is the hook: hold it clean
        shots[0] = (max(shots[0][0], t_open + .35),) + shots[0][1:]
    return shots
