"""Tear layouts for paper_rip_reel: which pixels come off together, and when.

build(cfg, g) -> (lab, piece)
  lab   uint8 per pixel: 0..4 tear on the five build-up beats, 5 floats off in the
        break, 6 is the face sliver (drop)
  piece int32 per pixel: fragment id; every id flies as its own piece of paper
g carries the shared grids: SW, SH, xx, yy, wx, wy (warped), noise(scale, seed), face, sliver.

kinds
  rules   explicit rect/band/side rules (the original hand-made layouts; each label is
          split in two along a rough line)
  shards  Voronoi cells with straight-ish torn edges, far from the face first
  strips  long parallel shredded strips, outermost first
  wedges  petals radiating from the face, torn round the clock
"""
import numpy as np


def _split_two(lab, g, k):
    m = lab == k
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    a = np.radians(np.random.default_rng(20 + k).uniform(0, 180))
    side = (g['xx'] - cx) * np.cos(a) + (g['yy'] - cy) * np.sin(a) + 40 * (g['noise'](15, 30 + k) - .5)
    return m & (side < 0), m & (side >= 0)


def _rules(cfg, g):
    wx, wy = g['wx'], g['wy']
    R = cfg['regions']
    lab = np.full((g['SH'], g['SW']), R['default'], np.uint8)
    for r in R['rules']:
        m = np.ones_like(lab, bool)
        if 'rect' in r:
            x0, y0, x1, y1 = r['rect']
            m &= (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
        if 'band' in r:
            ox, oy, dx, dy, lo, hi = r['band']
            p = ((wx - ox) * dx + (wy - oy) * dy) / np.hypot(dx, dy)
            m &= (p >= lo) & (p < hi)
        if 'side' in r:
            ox, oy, px, py = r['side']
            m &= ((wx - ox) * px + (wy - oy) * py) >= 0
        lab[m] = r['label']
    return lab


def _fine(g):
    """Torn-but-straight edges: only the fine jitter octaves (as for the sliver)."""
    n = g['noise']
    return (g['xx'] + 10 * (n(22, 70) - .5) + 4 * (n(5, 71) - .5),
            g['yy'] + 10 * (n(22, 72) - .5) + 4 * (n(5, 73) - .5))


def _group(cells_far_to_near, near_ids, n=5):
    """Map cell id -> label: far cells tear first, grouped across the five beats."""
    order = [c for c in cells_far_to_near if c not in near_ids]
    out = {c: 5 for c in near_ids}
    for k, grp in enumerate(np.array_split(np.array(order, int), n)):
        for c in grp:
            out[int(c)] = k
    return out


def _labels_from_cells(cell, g, near_r):
    fx, fy = g['face'][:2]
    ids = np.unique(cell)
    dist = {}
    for c in ids:
        ys, xs = np.nonzero(cell == c)
        dist[int(c)] = np.hypot(xs.mean() - fx, ys.mean() - fy)
    far = sorted(dist, key=lambda c: -dist[c])
    near = {c for c in dist if dist[c] < near_r} or {far[-1]}
    m = _group(far, near)
    lut = np.zeros(ids.max() + 1, np.uint8)
    for c, l in m.items():
        lut[c] = l
    return lut[cell]


def _shards(cfg, g, rng):
    SW, SH = g['SW'], g['SH']
    n = cfg['regions'].get('n', 16)
    fx, fy, rx, ry = g['face']
    pts = []                                   # dart-throwing: roughly even cells
    while len(pts) < n:
        p = rng.uniform((0, 0), (SW, SH))
        if all(np.hypot(*(p - q)) > .55 * np.sqrt(SW * SH / n) for q in pts):
            pts.append(p)
    x, y = _fine(g)
    best = np.full((SH, SW), np.inf, np.float32)
    cell = np.zeros((SH, SW), np.int32)
    for i, (px, py) in enumerate(pts):
        d = (x - px) ** 2 + (y - py) ** 2
        cell[d < best] = i
        best = np.minimum(best, d)
    return _labels_from_cells(cell, g, 2.4 * max(rx, ry)), cell


def _strips(cfg, g, rng):
    fx, fy, rx, ry = g['face']
    ang = np.radians(cfg['regions'].get('angle', rng.uniform(0, 180)))
    w = cfg['regions'].get('width', max(g['SW'], g['SH']) / 10)
    x, y = _fine(g)
    wave = 18 * (g['noise'](60, 74) - .5)                       # strips aren't ruler-straight
    perp = (x - fx) * -np.sin(ang) + (y - fy) * np.cos(ang) + wave
    cell = np.floor(perp / w + .5).astype(np.int32)
    cell -= cell.min()
    return _labels_from_cells(cell, g, 1.1 * w), cell


def _wedges(cfg, g, rng):
    fx, fy, rx, ry = g['face']
    n = cfg['regions'].get('n', 10)
    a0 = rng.uniform(0, 2 * np.pi)
    x, y = g['wx'], g['wy']
    phi = (np.arctan2(y - fy, x - fx) - a0) % (2 * np.pi)
    cell = np.floor(phi / (2 * np.pi / n)).astype(np.int32)
    r = np.hypot((g['xx'] - fx) / rx, (g['yy'] - fy) / ry) + .35 * (g['noise'](14, 75) - .5)
    lab = np.zeros(cell.shape, np.uint8)
    for k in range(n):                                          # round the clock, two per beat
        lab[cell == k] = min(k * 5 // n, 4)
    ring = r < 2.3
    lab[ring] = 5
    cell[ring] = n
    return lab, cell


def build(cfg, g):
    kind = cfg['regions'].get('kind', 'rules')
    rng = np.random.default_rng(cfg['regions'].get('seed', 7))
    if kind == 'rules':
        lab = _rules(cfg, g)
        piece = np.zeros(lab.shape, np.int32)
        for k in range(6):
            if not (lab == k).any():
                continue
            a, b = _split_two(lab, g, k)
            piece[a], piece[b] = 2 * k, 2 * k + 1
    else:
        lab, piece = {'shards': _shards, 'strips': _strips, 'wedges': _wedges}[kind](cfg, g, rng)
    sl = g['sliver']
    lab[sl] = 6
    piece[sl] = piece.max() + 1
    return lab, piece
