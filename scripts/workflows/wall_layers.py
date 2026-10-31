"""Layer generators for layer_compose.py (see plans/deconstruct.md).

Every generator: gen(L, ctx) -> Layer(rgb HxWx3 float 0-1, alpha HxW float 0-1,
blend 'normal'|'multiply', clip 'subject'|None). Procedural ones are free and
seeded; 'asset' ones generate an element via Flux on white (cached on disk) and
key it to alpha. All sizes scale with ctx.s (short edge) — never fixed pixels.
"""
import hashlib, json, math, os, re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, distance_transform_edt

ASSETS = Path("~/.openclaw/workspace/shared/deconstruct/assets").expanduser()
FONTS = {
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "cyrillic": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "latin": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "digits": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "ascii": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "katakana": "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
}
CHARSETS = {
    "katakana": "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン",
    "cyrillic": "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    "latin": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "digits": "0123456789",
    "ascii": "0123456789ABCDEF<>/\\|=+*#@$%&",
}


@dataclass
class Layer:
    rgb: np.ndarray
    alpha: np.ndarray
    blend: str = "normal"
    clip: str | None = None


@dataclass
class Ctx:
    W: int
    H: int
    rng: np.random.Generator
    sbox: tuple          # subject bbox x0,y0,x1,y1 (px)
    head: tuple          # head centre (x, y) px
    mask: np.ndarray     # subject alpha HxW 0-1
    lum: np.ndarray      # B&W subject luminance HxW 0-1
    palette: list
    focus: np.ndarray | None = None   # 1 near/behind the figure -> 0 in the quiet margins
    face: dict | None = None          # face_align landmarks in canvas px (eye_l, eye_r, nose, mouth_l...)

    @property
    def s(self):
        return min(self.W, self.H)


# ---------------------------------------------------------------- helpers
def hex_rgb(h, default="#c4001f"):
    h = (h or default).lstrip("#")
    if len(h) != 6:
        h = default.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def solid(ctx, color, alpha, **kw):
    rgb = np.empty((ctx.H, ctx.W, 3), np.float32)
    rgb[:] = hex_rgb(color)
    return Layer(rgb, np.clip(alpha, 0, 1).astype(np.float32), **kw)


def region_box(region, ctx):
    W, H = ctx.W, ctx.H
    if str(region).startswith("at:"):        # pinned: "at:0.2,0.88" (normalized centre)
        fx, fy = (float(v) for v in region[3:].split(","))
        r = 0.01 * min(W, H)
        return (int(fx * W - r), int(fy * H - r), int(fx * W + r), int(fy * H + r))
    x0, y0, x1, y1 = ctx.sbox
    sw, sh = x1 - x0, y1 - y0
    hx, hy = ctx.head
    boxes = {
        "behind-head": (hx - 0.9 * sw, max(0, hy - 0.45 * sh), hx + 0.9 * sw, hy + 0.65 * sh),
        "around": (x0 - 0.3 * sw, y0 - 0.1 * sh, x1 + 0.3 * sw, y1 + 0.1 * sh),
        "left": (0.02 * W, 0.08 * H, max(0.25 * W, x0 + 0.2 * sw), 0.95 * H),
        "right": (min(0.75 * W, x1 - 0.2 * sw), 0.08 * H, 0.98 * W, 0.95 * H),
        "top": (0.05 * W, 0.0, 0.95 * W, max(0.22 * H, y0 + 0.15 * sh)),
        "bottom": (0.05 * W, 0.72 * H, 0.95 * W, H),
        "edges": (0, 0, W, H),
        "anywhere": (0, 0, W, H),
    }
    bx = boxes.get(region, boxes["anywhere"])
    return (int(max(0, bx[0])), int(max(0, bx[1])), int(min(W, bx[2])), int(min(H, bx[3])))


def region_pt(region, ctx, n=1):
    x0, y0, x1, y1 = region_box(region, ctx)
    rng = ctx.rng
    if str(region).startswith("at:"):
        return np.array([((x0 + x1) / 2, (y0 + y1) / 2)] * n)
    if region == "edges":
        pts = []
        for _ in range(n):
            side = rng.integers(4)
            t = rng.uniform(0, 1)
            m = rng.uniform(0, 0.15)
            pts.append({0: (t * ctx.W, m * ctx.H), 1: (t * ctx.W, (1 - m) * ctx.H),
                        2: (m * ctx.W, t * ctx.H), 3: ((1 - m) * ctx.W, t * ctx.H)}[side])
        return np.array(pts)
    cand = np.stack([rng.uniform(x0, max(x0 + 1, x1), n * 8),
                     rng.uniform(y0, max(y0 + 1, y1), n * 8)], 1)
    if ctx.focus is None:
        return cand[:n]
    w = ctx.focus[np.clip(cand[:, 1].astype(int), 0, ctx.H - 1),
                  np.clip(cand[:, 0].astype(int), 0, ctx.W - 1)] ** 2 + 0.02
    return cand[rng.choice(len(cand), n, replace=False, p=w / w.sum())]


def stamp(dst, src, cx, cy, mode="max"):
    """Paste 2-D src centred at (cx, cy) into dst with max/add, clipped to bounds."""
    h, w = src.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    s = src[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    d = dst[Y0:Y1, X0:X1]
    if mode == "max":
        np.maximum(d, s, out=d)
    else:
        d += s


def rotate(a, deg):
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


def fbm(shape, rng, scale, octaves=4):
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        sig = max(0.5, scale / (2 ** o))
        n = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sig)
        n /= n.std() + 1e-6
        out += amp * n
        tot += amp
        amp *= 0.5
    return out / tot


def drips(alpha, ctx, amount=0.5, color_alpha=1.0):
    """Gravity drips hanging from the bottom edges of painted marks."""
    s, rng = ctx.s, ctx.rng
    a = (alpha > 0.5).astype(np.uint8)
    edge = a & (np.roll(a, -1, 0) == 0)
    ys, xs = np.nonzero(edge)
    if len(xs) == 0:
        return alpha
    n = int(len(xs) * 0.004 * amount) + int(8 * amount)
    out = alpha.copy()
    img = np.zeros_like(alpha)
    for i in rng.choice(len(xs), min(n, len(xs)), replace=False):
        x, y = int(xs[i]), int(ys[i])
        L = int(s * min(0.35, rng.exponential(0.05) + 0.01))
        w = max(1, int(s * rng.uniform(0.0015, 0.005)))
        wob = 0
        pts = []
        for k in range(0, L, max(2, L // 12)):
            wob += rng.normal(0, 0.3)
            pts.append((x + int(wob), y + k))
        for (p, q), j in zip(zip(pts, pts[1:]), range(len(pts))):
            ww = max(1, int(w * (1 - 0.5 * j / max(1, len(pts)))))
            cv2.line(img, p, q, 1.0, ww, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], max(1, int(w * 1.3)), 1.0, -1, cv2.LINE_AA)
    return np.maximum(out, img * color_alpha)


def dry_texture(shape, ctx, along_deg=90, strength=0.5):
    """Dry-brush streaks: 1 = paint, 0 = gap."""
    h, w = shape
    n = ctx.rng.standard_normal((h, w)).astype(np.float32)
    if along_deg % 180 == 90:
        n = gaussian_filter(n, (ctx.s * 0.02, 0.8))
    else:
        n = gaussian_filter(n, (0.8, ctx.s * 0.02))
    n /= n.std() + 1e-6
    return np.clip((n + 1.6 - strength * 2.2), 0, 1)


# ---------------------------------------------------------------- runes
def _nib_line(img, p, q, t, ang=math.radians(40)):
    """Chisel-nib stroke: a parallelogram — thick on the diagonal, thin across."""
    n = np.array([math.cos(ang), -math.sin(ang)]) * t
    p, q = np.array(p, float), np.array(q, float)
    poly = np.array([p + n, q + n, q - n, p - n], np.int32)
    cv2.fillConvexPoly(img, poly, 1.0, cv2.LINE_AA)


def rune(gw, gh, rng, t, style="block"):
    """One invented glyph in a gw x gh cell (alpha 0-1)."""
    img = np.zeros((gh, gw), np.float32)
    m = t * 1.2
    cols = [m, gw / 2, gw - m]
    rows = [m, gh * 0.33, gh * 0.66, gh - m]
    stems = rng.choice([1, 2, 2, 3])
    xs = sorted(rng.choice(cols, stems, replace=False))
    for x in xs:
        y0 = rows[0] if rng.random() < 0.8 else rows[1]
        y1 = rows[3] if rng.random() < 0.8 else rows[2]
        _nib_line(img, (x, y0), (x + rng.normal(0, t * 0.3), y1), t)
    for _ in range(rng.integers(1, 4)):
        kind = rng.integers(4)
        a, b = rng.choice(cols, 2, replace=False)
        ya, yb = rng.choice(rows, 2)
        if kind == 0:      # bar
            _nib_line(img, (a, ya), (b, ya), t)
        elif kind == 1:    # diagonal
            _nib_line(img, (a, ya), (b, yb), t)
        elif kind == 2:    # hook
            _nib_line(img, (a, ya), (a + (b - a) * 0.5, ya - t * 2), t)
            _nib_line(img, (a + (b - a) * 0.5, ya - t * 2), (b, ya + t), t)
        else:              # diamond dot
            c = (rng.uniform(m, gw - m), rng.uniform(m, gh - m))
            cv2.circle(img, (int(c[0]), int(c[1])), int(t * 1.1), 1.0, -1, cv2.LINE_AA)
    if style == "brush" and rng.random() < 0.45:   # long flourish tail
        x = xs[-1]
        _nib_line(img, (x, gh - m), (x + rng.normal(0, gw * 0.3), gh + gh * 0.0), t * 0.6)
    return img


GLYPH_FONTS = {
    "block": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "brush": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
}
# Latin + Greek shapes only: Cyrillic look-alikes read as broken Russian to a Russian reader
RUNE_CHARS = "AEFHIKLMNRTVWXYZΔΣΨΩΞΛΓΠ"


def font_glyph(ch, size, style, rng, condense=0.55, shear=0.0):
    """A real letter made unreadable: mirrored/flipped, condensed, sheared, chipped."""
    f = ImageFont.truetype(GLYPH_FONTS[style], int(size))
    im = Image.new("L", (int(size * 1.4), int(size * 1.5)), 0)
    ImageDraw.Draw(im).text((size * 0.15, size * 0.05), ch, fill=255, font=f)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.2:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    bb = im.getbbox()
    if not bb:
        return np.zeros((2, 2), np.float32)
    im = im.crop(bb)
    w, h = im.size
    im = im.resize((max(2, int(w * condense)), max(2, int(h * 1.15))), Image.LANCZOS)
    if shear:
        w, h = im.size
        im = im.transform((w + int(abs(shear) * h), h), Image.AFFINE,
                          (1, shear, -shear * h if shear > 0 else 0, 0, 1, 0), Image.BICUBIC)
    a = np.asarray(im, np.float32) / 255
    if rng.random() < 0.5:   # occasional added bar / spike so it isn't just a letter
        h, w = a.shape
        y = int(rng.uniform(0.2, 0.8) * h)
        a[max(0, y - h // 18):y + h // 18, :] = np.maximum(a[max(0, y - h // 18):y + h // 18, :],
                                                       (rng.random() < 0.5))
    return a


def rune_block(w, h, ctx, glyph, t_frac=0.16, style="block", vertical=True, gap=0.12):
    """Packed invented glyphs. vertical: stacked columns (graffiti panels);
    else: lines of wet-brush calligraphy."""
    rng = ctx.rng
    out = np.zeros((h, w), np.float32)
    if vertical:
        x = 0
        while x < w:
            y, colw = 0, 0
            while y < h:
                g = font_glyph(rng.choice(list(RUNE_CHARS)), glyph, style, rng, condense=rng.uniform(0.45, 0.8))
                gh, gw = g.shape
                stamp(out, g, x + gw / 2, y + gh / 2)
                y += int(gh * (1 + gap * 0.5))
                colw = max(colw, gw)
            x += int(colw * (1 + gap))
    else:
        y = 0
        while y < h:
            x = 0
            while x < w:
                g = font_glyph(rng.choice(list(RUNE_CHARS)), glyph * rng.uniform(0.8, 1.25), style,
                               rng, condense=rng.uniform(0.5, 0.75), shear=rng.uniform(0.15, 0.35))
                gh, gw = g.shape
                stamp(out, g, x + gw / 2, y + h * 0.02 + gh / 2 + rng.normal(0, glyph * 0.06))
                x += int(gw * (0.85 + gap))
            y += int(glyph * 1.6)
    return out


def gen_runes(L, ctx, mode):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    cov = float(p.get("coverage", 0.35))
    acc = np.zeros((ctx.H, ctx.W), np.float32)
    x0, y0, x1, y1 = region_box(L.get("region", "behind-head"), ctx)
    if mode == "block":                      # stacked vertical panels (graffiti glyphs)
        n = max(2, int(2 + cov * 8))
        for _ in range(n):
            pw = int(s * rng.uniform(0.05, 0.12))
            ph = int(s * rng.uniform(0.3, 0.75))
            g = s * rng.uniform(0.045, 0.07)
            blk = rune_block(pw, ph, ctx, g, 0.17, "block", vertical=True, gap=0.08)
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            stamp(acc, blk, cx, cy)
    else:                                    # big wet-brush calligraphy lines
        n = max(1, int(1 + cov * 5))
        vertical = p.get("orientation") == "vertical"
        for _ in range(n):
            g = s * rng.uniform(0.07, 0.13)
            cnt = rng.integers(4, 9)
            bw_, bh_ = (int(g * 1.2), int(g * 1.15 * cnt)) if vertical else (int(g * 1.15 * cnt), int(g * 1.3))
            blk = rune_block(bw_, bh_, ctx, g, 0.2, "brush", vertical=False, gap=0.0)
            blk = rotate(blk, rng.normal(0, 6))
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            stamp(acc, blk, cx, cy)
    # light dry-brush breakup + chipped edges (marks must still read as glyphs)
    chip = np.clip(1.4 + 0.9 * fbm(acc.shape, rng, s * 0.003, 2), 0, 1)
    acc = acc * chip * (0.75 + 0.25 * dry_texture(acc.shape, ctx, 90, 0.3))
    acc = np.clip(acc * 1.2, 0, 1)
    if p.get("drips", mode == "block"):
        acc = drips(acc, ctx, amount=1.0 + cov)
    # spray speckle around the marks
    near = gaussian_filter(acc, s * 0.01) > 0.05
    sp = (rng.random(acc.shape) < 0.012) & near & (acc < 0.1)
    acc = np.maximum(acc, gaussian_filter(sp.astype(np.float32), 0.6) * 2.5)
    return solid(ctx, p.get("color"), acc * float(p.get("opacity", 0.95)))


# ---------------------------------------------------------------- marks
BRUSH_PROMPT = ("one single bold black ink dry-brush stroke on white paper, sumi-e, fast gesture, "
                "ragged bristle streaks at the end, small ink spatter, no text")


def gen_ink_smear(L, ctx):
    """Black ink masses: real brush strokes generated once (Flux, cached — a small
    library of 4 strokes), keyed to alpha, scattered near the figure."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    if p.get("style") == "procedural":
        return _ink_smear_procedural(L, ctx)
    cov = float(p.get("coverage", 0.3))
    acc = np.zeros((ctx.H, ctx.W), np.float32)
    strokes = []
    b0 = int(p.get("brush_seed", 500))      # 500 = the original shared library (old pieces)
    for k in range(4):
        a = np.asarray(asset_image(BRUSH_PROMPT, b0 + k), np.float32) / 255
        lum = a @ np.array([0.3, 0.59, 0.11], np.float32)
        al = np.clip((0.85 - lum) / 0.55, 0, 1)
        ys, xs = np.nonzero(al > 0.2)
        if len(xs):
            al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        strokes.append(al)
    base_dir = float(re.sub(r"[^\d.-]", "", str(p.get("direction", "-30"))) or -30)
    for cx, cy in region_pt(L.get("region", "behind-head"), ctx, int(2 + cov * 8)):
        st = strokes[rng.integers(len(strokes))]
        ln = s * rng.uniform(0.3, 0.7)
        sc = ln / max(st.shape)
        st = cv2.resize(st, (max(4, int(st.shape[1] * sc)), max(4, int(st.shape[0] * sc))),
                        interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            st = st[:, ::-1]
        stamp(acc, rotate(np.ascontiguousarray(st), base_dir + rng.normal(0, 35)), cx, cy)
    return solid(ctx, p.get("color", "#101015"), acc * float(p.get("opacity", 0.9)))


def _ink_smear_procedural(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    cov = float(p.get("coverage", 0.3))
    base_dir = float(re.sub(r"[^\d.-]", "", str(p.get("direction", "-30"))) or -30)
    acc = np.zeros((ctx.H, ctx.W), np.float32)
    pts = region_pt(L.get("region", "behind-head"), ctx, int(2 + cov * 7))
    for cx, cy in pts:
        ln, wd = int(s * rng.uniform(0.25, 0.6)), int(s * rng.uniform(0.08, 0.22))
        n = gaussian_filter(rng.standard_normal((wd, ln)).astype(np.float32), (0.7, ln * 0.08))
        n /= n.std() + 1e-6
        u = np.linspace(0, 1, ln)[None, :]
        load = 1.1 - 1.4 * u ** 1.5 + 0.15 * rng.standard_normal()     # paint runs out
        v = np.abs(np.linspace(-1, 1, wd))[:, None]
        edge = 1 - (v + 0.25 * fbm((wd, ln), rng, wd * 0.15, 2)) ** 3
        st = np.clip((n * 0.35 + load) * edge, 0, 1)
        st = np.clip((st - 0.15) * 4.0, 0, 1)            # solid core, dry streaky tail
        st = rotate(st, base_dir + rng.normal(0, 25))
        stamp(acc, st, cx, cy)
    return solid(ctx, p.get("color", "#101015"), acc * float(p.get("opacity", 0.9)))


def gen_splatter(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    cov = float(p.get("coverage", 0.12))
    big = p.get("size") == "large"
    img = np.zeros((ctx.H, ctx.W), np.float32)
    for cx, cy in region_pt(L.get("region", "around"), ctx, int(3 + cov * 20)):
        spread = s * rng.uniform(0.03, 0.12)
        for _ in range(int(30 + cov * 300)):
            r = s * min(0.03 if big else 0.012, 0.0008 * (rng.pareto(2.2) + 1))
            x, y = rng.normal(cx, spread), rng.normal(cy, spread)
            if rng.random() < 0.15:        # elongated fleck
                ang = rng.uniform(0, 180)
                cv2.ellipse(img, (int(x), int(y)), (int(r * 3) + 1, max(1, int(r * 0.6))),
                            ang, 0, 360, 1.0, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), max(1, int(r)), 1.0, -1, cv2.LINE_AA)
    return solid(ctx, p.get("color", "#0d0d12"), img * float(p.get("opacity", 0.9)))


BRAILLE = {c: d for c, d in zip("abcdefghijklmnopqrstuvwxyz", [
    "1", "12", "14", "145", "15", "124", "1245", "125", "24", "245", "13", "123", "134",
    "1345", "135", "1234", "12345", "1235", "234", "2345", "136", "1236", "2456", "1346",
    "13456", "1356"])}
BRAILLE.update({"'": "3", ",": "2", ".": "256", "-": "36", "?": "236", "!": "235"})
DOT_POS = {"1": (0, 0), "2": (0, 1), "3": (0, 2), "4": (1, 0), "5": (1, 1), "6": (1, 2)}


def braille_cells(text):
    """Grade-1 braille: one cell per letter, None for a space."""
    return [None if ch == " " else BRAILLE.get(ch) for ch in text.lower()
            if ch == " " or ch in BRAILLE]


def gen_dot_rows(L, ctx):
    """Rows of braille. With params.text the dots spell it (Grade-1, one row per line,
    lines split on '/'); without, random cells."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    d = s * float(p.get("dot_pitch", 0.009))     # real proportions: dot pitch d,
    r = d * 0.3                                  # dot Ø 0.6d, cell pitch 2.4d, line pitch 4d
    cell, line = d * 2.4, d * 4.0
    SS = 4                                       # supersample → crisp round dots
    big = np.zeros((ctx.H * SS // 2, ctx.W * SS // 2), np.uint8)
    k = SS // 2
    img = np.zeros((ctx.H, ctx.W), np.float32)
    x0, y0, x1, y1 = region_box(L.get("region", "anywhere"), ctx)
    lines = [t.strip() for t in str(p.get("text", "")).split("/") if t.strip()]
    anchor = p.get("anchor")
    if lines and anchor and ctx.face and "mouth_l" in ctx.face:
        # 'cheek-left': the text block ENDS on her cheek (image-left side of the face)
        # and runs out to the left of her face, rows stacked downward
        f = ctx.face
        eye = min((f["eye_l"], f["eye_r"]), key=lambda q: q[0])
        mouth = min((f["mouth_l"], f["mouth_r"]), key=lambda q: q[0])
        # cheek apple: under the eye, a bit toward the nose (eye_l is the OUTER corner)
        cx = eye[0] + (f["nose"][0] - eye[0]) * 0.5 if anchor == "cheek-left" else f["nose"][0]
        cy = eye[1] + (mouth[1] - eye[1]) * 0.4
        for i, t in enumerate(lines):
            cells = braille_cells(t)
            x = cx - len(cells) * cell
            y = cy + i * line
            for c in cells:
                for kk in (c or ""):
                    dx, dy = DOT_POS[kk]
                    cv2.circle(big, (int((x + dx * d) * k), int((y + dy * d) * k)),
                               max(1, int(r * k)), 255, -1, cv2.LINE_AA)
                x += cell
        img = cv2.resize(big, (ctx.W, ctx.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return solid(ctx, p.get("color", "#d4001e"), img * float(p.get("opacity", 1.0)))
    for i in range(len(lines) or int(p.get("rows", 3))):
        y = rng.uniform(y0 + 0.1 * (y1 - y0), y1 - 0.1 * (y1 - y0))
        if lines:
            cells = braille_cells(lines[i])
            x = rng.uniform(x0, max(x0 + 1, x1 - len(cells) * d * 3))
            for c in cells:
                for k in (c or ""):
                    dx, dy = DOT_POS[k]
                    cv2.circle(img, (int(x + dx * d), int(y + dy * d)), max(1, int(round(r))), 1.0, -1, cv2.LINE_AA)
                x += d * 3
            continue
        xa = rng.uniform(x0, x0 + 0.4 * (x1 - x0))
        xb = rng.uniform(xa + 0.25 * (x1 - x0), x1)
        x = xa
        while x < xb:
            if rng.random() < 0.15:          # gap between words
                x += d * 3
                continue
            for cx in range(2):
                for cy in range(3):
                    if rng.random() < 0.5:
                        cv2.circle(img, (int(x + cx * d), int(y + cy * d)), max(1, int(round(r))), 1.0, -1, cv2.LINE_AA)
            x += d * 3
    return solid(ctx, p.get("color", "#d4001e"), img * float(p.get("opacity", 1.0)))


def _smoke_path(ctx, cx, cy):
    rng, s = ctx.rng, ctx.s
    pts, ang = [], -math.pi / 2
    x, y = cx, cy
    for i in range(int(rng.integers(60, 140))):
        ang += rng.normal(0, 0.18) + 0.12 * math.sin(i * 0.15 + rng.uniform(0, 6))
        ang = -math.pi / 2 + np.clip(ang + math.pi / 2, -1.3, 1.3)
        step = s * 0.006
        x += math.cos(ang) * step + math.sin(i * 0.3) * s * 0.002
        y += math.sin(ang) * step
        pts.append((int(x), int(y)))
    return np.array(pts, np.int32)


def gen_pen_loops(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    smoke = p.get("style") == "smoke" or "smoke" in L.get("id", "")
    img = np.zeros((ctx.H, ctx.W), np.float32)
    lw = max(1, int(s * (0.0012 if p.get("weight", "hairline") == "hairline" else 0.0025)))
    for cx, cy in region_pt(L.get("region", "anywhere"), ctx, int(p.get("count", 8))):
        if smoke:      # rising curling wisps, several parallel strands
            base = _smoke_path(ctx, cx, cy)
            for k in range(int(rng.integers(2, 5))):
                off = (np.cumsum(rng.normal(0, 0.6, (len(base), 2)), 0) * s * 0.0006).astype(np.int32)
                cv2.polylines(img, [base + off + int(k * s * 0.004)], False, 1.0, lw, cv2.LINE_AA)
        else:
            ax = (int(s * rng.uniform(0.12, 0.5)), int(s * rng.uniform(0.06, 0.3)))
            a0 = rng.uniform(0, 360)
            cv2.ellipse(img, (int(cx), int(cy)), ax, rng.uniform(0, 180), a0,
                        a0 + rng.uniform(120, 360), 1.0, lw, cv2.LINE_AA)
    return solid(ctx, p.get("color", "#d8d8d8"), img * float(p.get("opacity", 0.7)))


ABSTRACT = re.compile(r"loop|zigzag|tangl|gestur|signature|chaotic|scrawl")
OBJECTS = re.compile(r"sketch|eye|figure|braid|hoop|buckle|smoke|rope|face|hand|flower|bird|"
                     r"skull|animal|ring|strand")


def gen_scribbles(L, ctx):
    p = L.get("params", {})
    subj = str(p.get("subject", ""))
    if ABSTRACT.search(subj) and not OBJECTS.search(subj):
        return _marker_scribble(L, ctx)
    med = p.get("medium", "ballpoint")
    prompt = (f"loose hand-drawn {med} doodle sketch: {subj}. single colour line art, "
              f"black ink on plain white paper, no shading, no background, flat scan")
    return gen_asset({**L, "params": {**p, "prompt": prompt, "tint": True}}, ctx)


def _marker_scribble(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    img = np.zeros((ctx.H, ctx.W), np.float32)
    lw = max(1, int(s * 0.0028))
    for cx, cy in region_pt(L.get("region", "around"), ctx, int(1 + float(p.get("coverage", 0.2)) * 8)):
        n = int(rng.integers(80, 260))
        v = rng.normal(0, 1, 2)
        pts = []
        x, y = cx, cy
        for _ in range(n):
            v = 0.85 * v + rng.normal(0, 0.9, 2)
            x += v[0] * s * 0.004
            y += v[1] * s * 0.004
            pts.append((x, y))
        pts = np.array(pts)
        pts[:, 0] = gaussian_filter(pts[:, 0], 1.5)
        pts[:, 1] = gaussian_filter(pts[:, 1], 1.5)
        cv2.polylines(img, [pts.astype(np.int32)], False, 1.0, lw, cv2.LINE_AA)
    return solid(ctx, p.get("color", "#cc1126"), img * float(p.get("opacity", 0.85)))


def gen_micro_text(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    fs = max(7, int(s * 0.0095))
    font = ImageFont.truetype(FONTS["mono"], fs)
    im = Image.new("L", (ctx.W, ctx.H), 0)
    d = ImageDraw.Draw(im)
    words = ["FIG.", "SPEC.", "No", "LOT", "REF", "SER.", "PLATE", "ARCH.", "DOC", "INV."]
    for cx, cy in region_pt(L.get("region", "edges"), ctx, int(rng.integers(2, 6))):
        vocab = ("silver gelatin print · specimen · plate · archive · exposure · negative · "
                 "contact sheet · ration · issued · received · copy · subject · study · "
                 "window light · morning · unknown · catalogue · ink on paper").split(" · ")
        lines = p.get("lines") or [f"{rng.choice(words)} {rng.integers(10, 9999)} — "
                 + ", ".join(rng.choice(vocab, rng.integers(1, 4), replace=False))
                 for _ in range(rng.integers(2, 6))]
        d.multiline_text((cx, cy), "\n".join(lines), fill=255, font=font, spacing=1)
    return solid(ctx, p.get("color", "#555555"),
                 np.asarray(im, np.float32) / 255 * float(p.get("opacity", 0.8)))


def gen_glyph_rain(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    cs = p.get("charset", "katakana")
    chars = CHARSETS.get(cs, CHARSETS["ascii"])
    fs = max(8, int(s * 0.018))
    font = ImageFont.truetype(FONTS.get(cs, FONTS["mono"]), fs)
    im = Image.new("L", (ctx.W, ctx.H), 0)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = region_box(L.get("region", "right"), ctx)
    cov = float(p.get("coverage", 0.2))
    for x in np.arange(x0, x1, fs * 1.2):
        if rng.random() > 0.35 + cov:
            continue
        n = int(rng.integers(6, max(7, int((y1 - y0) / fs))))
        ys = rng.uniform(y0, max(y0 + 1, y1 - n * fs))
        for k in range(n):
            a = int(255 * (0.25 + 0.75 * k / n))          # head of the column is brightest
            d.text((x, ys + k * fs * 1.05), str(rng.choice(list(chars))), fill=a, font=font)
    return solid(ctx, p.get("color", "#8a0f1e"),
                 np.asarray(im, np.float32) / 255 * float(p.get("opacity", 0.75)))


def gen_stain(L, ctx):
    """Fingerprint stamps (default) or coffee/water rings."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    img = np.zeros((ctx.H, ctx.W), np.float32)
    cov = float(p.get("coverage", 0.2))
    rings = p.get("type") in ("coffee",) and "finger" not in L.get("id", "")
    for cx, cy in region_pt(L.get("region", "behind-head"), ctx, int(2 + cov * 10)):
        R = int(s * rng.uniform(0.04, 0.08))
        yy, xx = np.mgrid[-R:R, -int(R * 0.75):int(R * 0.75)].astype(np.float32)
        warp = fbm(yy.shape, rng, R * 0.3, 3) * R * 0.08
        if rings:
            r = np.sqrt(xx ** 2 + yy ** 2) + warp
            a = np.exp(-((r - R * 0.85) / (R * 0.04)) ** 2) * 0.9 + (r < R * 0.85) * 0.12
        else:
            r = np.sqrt((xx / 0.75) ** 2 + (yy + R * 0.2) ** 2) + warp
            ridges = 0.5 + 0.5 * np.sin(r / (R * 0.045) * 2 * np.pi * 0.5)
            inside = np.clip((R - np.sqrt((xx / 0.75) ** 2 + yy ** 2)) / (R * 0.15), 0, 1)
            a = (ridges > 0.45) * inside * np.clip(0.6 + 0.6 * fbm(yy.shape, rng, R * 0.2, 2), 0, 1)
        stamp(img, rotate(a.astype(np.float32), rng.uniform(-40, 40)), cx, cy)
    return solid(ctx, p.get("color", "#222222"), img * float(p.get("opacity", 0.6)))


# ---------------------------------------------------------------- paper + assets
def _fal_flux(prompt, seed, w=1024, h=1024):
    import fal_client, requests
    os.environ.setdefault("FAL_KEY", os.environ.get("FAL_API_KEY", ""))
    r = fal_client.submit("fal-ai/flux/dev", arguments={
        "prompt": prompt, "image_size": {"width": w, "height": h}, "num_inference_steps": 28,
        "guidance_scale": 3.5, "num_images": 1, "output_format": "png",
        "enable_safety_checker": False, "seed": int(seed)}).get()
    return Image.open(BytesIO(requests.get(r["images"][0]["url"], timeout=60).content)).convert("RGB")


def asset_image(prompt, seed):
    """Generated element on white, cached by (prompt, seed)."""
    key = hashlib.sha1(f"{prompt}|{seed}".encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-")
    path = ASSETS / f"{slug}__{key}.png"
    if not path.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        _fal_flux(prompt + ", isolated on plain white background, flat even lighting, "
                           "no shadow, top-down scan", seed).save(path)
        (ASSETS / "index.jsonl").open("a").write(
            json.dumps({"file": path.name, "prompt": prompt, "seed": int(seed)}) + "\n")
    return Image.open(path).convert("RGB")


PAPERISH = re.compile(r"coupon|ticket|voucher|talon|newspaper|label|card|receipt|stamp sheet|"
                      r"page|letter|banknote|poster|form|ephemera")


def _torn(h, w, rng, rough):
    """Torn-paper alpha: fractal edges on all four sides."""
    m = np.ones((h, w), np.float32)
    for side in range(4):
        n = w if side < 2 else h
        e = np.cumsum(rng.normal(0, 1, n))
        e = gaussian_filter(e - np.linspace(e[0], e[-1], n), 1) * rough
        e = np.clip(np.abs(e) + rough * 0.6, 0, None).astype(int)
        for i in range(n):
            k = min(e[i], (h if side < 2 else w) // 3)
            if side == 0: m[:k, i] = 0
            elif side == 1: m[h - k:, i] = 0
            elif side == 2: m[i, :k] = 0
            else: m[i, w - k:] = 0
    return gaussian_filter(m, 0.6)


def _scraps(L, ctx, src, gray=True):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    rgb = np.zeros((ctx.H, ctx.W, 3), np.float32)
    al = np.zeros((ctx.H, ctx.W), np.float32)
    srcn = np.asarray(src, np.float32) / 255
    if gray:
        g = srcn @ np.array([0.3, 0.59, 0.11], np.float32)
        srcn = np.repeat(g[..., None], 3, 2)
    count = int(p.get("count", max(1, round(float(p.get("coverage", 0.2)) * 12))))
    for cx, cy in region_pt(L.get("region", "behind-head"), ctx, count):
        sw_, sh_ = int(s * rng.uniform(0.1, 0.26)), int(s * rng.uniform(0.08, 0.3))
        sh_, sw_ = min(sh_, srcn.shape[0] - 2), min(sw_, srcn.shape[1] - 2)
        oy, ox = rng.integers(0, srcn.shape[0] - sh_), rng.integers(0, srcn.shape[1] - sw_)
        piece = srcn[oy:oy + sh_, ox:ox + sw_]
        m = _torn(sh_, sw_, rng, s * 0.004)
        ang = rng.normal(0, 8)
        pr = np.dstack([rotate(piece[..., c], ang) for c in range(3)])
        pm = rotate(m, ang)
        h, w = pm.shape
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(ctx.W, x0 + w), min(ctx.H, y0 + h)
        if X1 <= X0 or Y1 <= Y0:
            continue
        mm = pm[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
        rgb[Y0:Y1, X0:X1] = rgb[Y0:Y1, X0:X1] * (1 - mm) + pr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * mm
        al[Y0:Y1, X0:X1] = np.maximum(al[Y0:Y1, X0:X1], mm[..., 0])
    return Layer(rgb, al * float(p.get("opacity", 0.9)))


def paper_object(src):
    """Cut a generated paper/object out of its white background -> (rgb float, alpha)."""
    a = np.asarray(src, np.float32) / 255
    white = np.median(np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)]), 0)
    dist = np.abs(a - white).max(2)
    m = (dist > 0.06).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    x, y, w, h = cv2.boundingRect(m)
    return a[y:y + h, x:x + w], gaussian_filter(m[y:y + h, x:x + w].astype(np.float32), 0.7)


def place_object(L, ctx, rgb_obj, a_obj, size_frac, gray=False):
    p, rng = L.get("params", {}), ctx.rng
    rgb = np.zeros((ctx.H, ctx.W, 3), np.float32)
    al = np.zeros((ctx.H, ctx.W), np.float32)
    if gray:
        g = rgb_obj @ np.array([0.3, 0.59, 0.11], np.float32)
        rgb_obj = np.repeat(g[..., None], 3, 2)
    for cx, cy in region_pt(L.get("region", "behind-head"), ctx, int(p.get("count", 1))):
        sc = ctx.s * size_frac * rng.uniform(0.8, 1.2) / max(a_obj.shape)
        w, h = max(4, int(a_obj.shape[1] * sc)), max(4, int(a_obj.shape[0] * sc))
        ang = rng.normal(0, 10)
        A = rotate(cv2.resize(a_obj, (w, h), interpolation=cv2.INTER_AREA), ang)
        C = np.dstack([rotate(cv2.resize(rgb_obj[..., c], (w, h), interpolation=cv2.INTER_AREA), ang)
                       for c in range(3)])
        hh, ww = A.shape
        X, Y = int(cx - ww / 2), int(cy - hh / 2)
        X0, Y0, X1, Y1 = max(0, X), max(0, Y), min(ctx.W, X + ww), min(ctx.H, Y + hh)
        if X1 <= X0 or Y1 <= Y0:
            continue
        mm = A[Y0 - Y:Y1 - Y, X0 - X:X1 - X, None]
        rgb[Y0:Y1, X0:X1] = rgb[Y0:Y1, X0:X1] * (1 - mm) + C[Y0 - Y:Y1 - Y, X0 - X:X1 - X] * mm
        al[Y0:Y1, X0:X1] = np.maximum(al[Y0:Y1, X0:X1], mm[..., 0])
    return Layer(rgb, al * float(p.get("opacity", 0.9)))


def gen_paper_scraps(L, ctx):
    p = L.get("params", {})
    content = p.get("content", "vintage newspaper page, dense columns of small print, headlines")
    # Flux can't spell: never ask it for specific words (quoted words came out as
    # 'Sefeworu' for 'safeword'). Strip them and keep all print tiny and unreadable.
    content = re.sub(r"[\'\"‘’“”][^\'\"‘’“”]{1,40}[\'\"‘’“”]", "", content)
    content = re.sub(r"\s+,", ",", content).strip(" ,")
    src = asset_image(f"{content}, black and white print, all text tiny and unreadable, "
                      "no headlines, no large words, flat scan of the full page",
                      L.get("seed", 1))
    rgb_o, a_o = paper_object(src)
    ys, xs = np.nonzero(a_o > 0.99)
    if len(xs):   # interior only — never crop the white backdrop
        y0, y1 = np.percentile(ys, [8, 92]).astype(int)
        x0, x1 = np.percentile(xs, [8, 92]).astype(int)
        src = Image.fromarray((rgb_o[y0:y1, x0:x1] * 255).astype(np.uint8))
    return _scraps(L, ctx, src, gray=not p.get("keep_color"))


def gen_asset(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    prompt = p.get("prompt") or L.get("note") or L["id"].replace("_", " ")
    src = asset_image(prompt, L.get("seed", 1))
    if PAPERISH.search(prompt) and not p.get("tint"):
        rgb_o, a_o = paper_object(src)
        size = 0.18 + 0.6 * float(p.get("coverage", 0.15))
        return place_object(L, ctx, rgb_o, a_o, min(0.45, size))
    a = np.asarray(src, np.float32) / 255
    lum = a @ np.array([0.3, 0.59, 0.11], np.float32)
    sat = a.max(2) - a.min(2)
    alpha = np.clip(np.maximum((0.92 - lum) / 0.6, sat * 1.5), 0, 1)
    cov = float(p.get("coverage", 0.15))
    rgb = np.zeros((ctx.H, ctx.W, 3), np.float32)
    al = np.zeros((ctx.H, ctx.W), np.float32)
    x0, y0, x1, y1 = region_box(L.get("region", "anywhere"), ctx)
    target = int(math.sqrt(max(1, cov * ctx.W * ctx.H)) * 1.3)
    sc = target / max(src.size)
    w, h = max(8, int(src.size[0] * sc)), max(8, int(src.size[1] * sc))
    A = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    C = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if p.get("tint"):
        C[:] = hex_rgb(p.get("color", "#111111"))
    elif p.get("gray", True) not in (False, "false", 0, "0"):   # keep the palette: no brown moths
        C[:] = C.mean(2, keepdims=True)
    cx, cy = rng.uniform(x0, max(x0 + 1, x1)), rng.uniform(y0, max(y0 + 1, y1))
    X, Y = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(0, X), max(0, Y), min(ctx.W, X + w), min(ctx.H, Y + h)
    if X1 > X0 and Y1 > Y0:
        al[Y0:Y1, X0:X1] = A[Y0 - Y:Y1 - Y, X0 - X:X1 - X]
        rgb[Y0:Y1, X0:X1] = C[Y0 - Y:Y1 - Y, X0 - X:X1 - X]
    return Layer(rgb, al * float(p.get("opacity", 0.9)))


SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PAPER_PROMPT = "a single blank sheet of aged cream paper, slightly stained, torn rough edges, no text, no print"

# real Soviet-era wording (checked): tobacco/sugar/soap ration coupons, 1990
PRINTS = {
    "tobacco_talon": {"head": ["ТАЛОН", "НА ТАБАЧНЫЕ ИЗДЕЛИЯ", "на март 1990 г."],
                      "cells": ["1 пачка"] * 6, "foot": "Исполком горсовета", "stamp": "ПОГАШЕНО"},
    "sugar_talon": {"head": ["ТАЛОН", "НА САХАР", "на апрель 1990 г."],
                    "cells": ["1 кг", "1 кг", "0,5 кг", "0,5 кг"], "foot": "Исполком горсовета",
                    "stamp": None},
    "soap_talon": {"head": ["ТАЛОН", "НА МЫЛО", "на май 1990 г."],
                   "cells": ["1 кусок"] * 4, "foot": "Исполком горсовета", "stamp": None},
    "papirosy": {"head": ["ПАПИРОСЫ", "БЕЛОМОРКАНАЛ"], "cells": [], "foot": "Цена 22 коп.",
                 "stamp": None},
}


def _txt(d, xy, text, size, font=SANS, fill=(0, 0, 0), anchor="mm"):
    d.text(xy, text, font=ImageFont.truetype(font, max(6, int(size))), fill=fill, anchor=anchor)


def render_print(variant, w, rng, ink=(178, 34, 40)):
    """A printed coupon/label with REAL Russian text on a generated paper sheet."""
    spec = PRINTS.get(variant, PRINTS["tobacco_talon"])
    h = int(w * (1.35 if spec["cells"] else 0.7))
    paper_rgb, paper_a = paper_object(asset_image(PAPER_PROMPT, 31 + rng.integers(3)))
    paper = Image.fromarray((paper_rgb * 255).astype(np.uint8)).resize((w, h))
    alpha = Image.fromarray((paper_a * 255).astype(np.uint8)).resize((w, h))
    d = ImageDraw.Draw(paper)
    m = int(w * 0.07)
    d.rectangle([m, m, w - m, h - m], outline=ink, width=max(1, w // 160))
    d.rectangle([m + w // 70, m + w // 70, w - m - w // 70, h - m - w // 70], outline=ink, width=1)
    y = m + w * 0.1
    for i, t in enumerate(spec["head"]):
        sz = w * (0.11 if i == 0 else 0.065 if i == 1 else 0.05)
        _txt(d, (w / 2, y), t, sz, SERIF if i < 2 else SANS, ink)
        y += sz * 1.35
    cells = spec["cells"]
    if cells:
        cols = 2
        rows = (len(cells) + 1) // 2
        top, bot = y + w * 0.03, h - m - w * 0.16
        cw, ch = (w - 2 * m) / cols, (bot - top) / rows
        for j, t in enumerate(cells):
            cx0, cy0 = m + (j % cols) * cw, top + (j // cols) * ch
            for xx in range(int(cx0), int(cx0 + cw), max(3, w // 60)):     # perforation
                d.line([xx, cy0, xx + w // 120, cy0], fill=ink, width=1)
            if j % cols:
                for yy in range(int(cy0), int(cy0 + ch), max(3, w // 60)):
                    d.line([cx0, yy, cx0, yy + w // 120], fill=ink, width=1)
            _txt(d, (cx0 + cw / 2, cy0 + ch * 0.42), t, w * 0.06, SANS_B, ink)
            _txt(d, (cx0 + cw / 2, cy0 + ch * 0.72), f"№ {rng.integers(1, 12):02d}", w * 0.035,
                 SANS, ink)
    _txt(d, (w / 2, h - m - w * 0.07), spec["foot"], w * 0.045, SANS, (40, 40, 40))
    a = np.asarray(paper, np.float32) / 255
    # ink wear: printed areas lose ink in patches
    a = a + (1 - a) * np.clip(0.35 * fbm(a.shape[:2], rng, w * 0.02, 3), 0, 0.6)[..., None]
    if spec.get("stamp"):
        st = render_stamp(int(w * 0.55), rng, ["ИСПОЛКОМ", "1990"], spec["stamp"], (60, 60, 140))
        sa = rotate(st[..., 3], rng.uniform(-30, 30))
        sr = np.dstack([rotate(st[..., c], 0 if False else 0) for c in range(3)])
        sh, sw = sa.shape
        ox, oy = int(w * 0.4), int(h * 0.55)
        reg = a[oy:oy + sh, ox:ox + sw]
        sa = sa[:reg.shape[0], :reg.shape[1], None] * 0.8
        reg[:] = reg * (1 - sa) + reg * np.array([0.25, 0.25, 0.55]) * sa
    return np.clip(a, 0, 1), np.asarray(alpha, np.float32) / 255


def draw_triskelion(d, cx, cy, R, fill=255):
    """The BDSM emblem: a ring split by three spokes that sweep from the centre to the
    rim (each turning ~70°), with a hole in each of the three sections."""
    lw = max(2, int(R * 0.14))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=fill, width=lw)
    for k in range(3):
        a0 = math.radians(-90 + k * 120)
        pts = []
        for i in range(25):
            t = i / 24
            r, th = R * t, a0 + math.radians(70) * t * t
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        d.line(pts, fill=fill, width=lw, joint="curve")
        th = a0 + math.radians(70 + 52)              # the hole sits mid-section, near the rim
        hr, hx, hy = R * 0.12, cx + R * 0.62 * math.cos(th), cy + R * 0.62 * math.sin(th)
        d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=fill)


def render_stamp(size, rng, ring, center, color=(178, 34, 40)):
    """Round rubber stamp: text around the ring, word in the middle, worn ink. RGBA float."""
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    lw = max(2, size // 40)
    d.ellipse([lw, lw, size - lw, size - lw], outline=255, width=lw)
    d.ellipse([size * 0.2, size * 0.2, size * 0.8, size * 0.8], outline=255, width=max(1, lw // 2))
    text = " • ".join(ring) + " • "
    f = ImageFont.truetype(SANS_B, max(6, int(size * 0.085)))
    n = len(text)
    for i, ch in enumerate(text):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        r = size * 0.39
        g = Image.new("L", (int(size * 0.12),) * 2, 0)
        ImageDraw.Draw(g).text((g.width / 2, g.height / 2), ch, font=f, fill=255, anchor="mm")
        g = g.rotate(-math.degrees(ang) - 90, resample=Image.BICUBIC)
        im.paste(255, (int(size / 2 + r * math.cos(ang) - g.width / 2),
                       int(size / 2 + r * math.sin(ang) - g.height / 2)), g)
    if center == "@triskelion":
        draw_triskelion(d, size / 2, size / 2, size * 0.25)
    else:
        _txt(d, (size / 2, size / 2), center, size * (0.11 if len(center) > 6 else 0.14), SANS_B, 255)
    a = np.asarray(im, np.float32) / 255
    a *= np.clip(0.75 + 0.6 * fbm(a.shape, rng, size * 0.03, 3), 0, 1)        # worn ink
    rgb = np.ones((size, size, 3), np.float32) * np.array(color, np.float32) / 255
    return np.dstack([rgb, a])


MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def render_typed(w, rng, title, lines, strike=(), sign="Подпись ____________",
                 ink=(30, 30, 34), red=(178, 34, 40)):
    """A typewritten sheet with REAL text (Flux can't spell): title, numbered lines,
    red strike-throughs on the `strike` indices, a signature line, uneven ribbon ink."""
    m = int(w * 0.09)
    f0 = ImageFont.truetype(MONO, 100)
    longest = max([f0.getlength(t) for t in lines + [sign or ""]] + [1])
    fs = max(6, int(min(w * 0.042, (w - 2 * m) * 100 / longest)))   # every line fits the sheet
    f = ImageFont.truetype(MONO, fs)
    h = int(2 * m + fs * (3 if title else 0) + fs * 1.7 * len(lines) + (fs * 2.5 if sign else 0))
    paper_rgb, paper_a = paper_object(asset_image(PAPER_PROMPT, 31 + rng.integers(3)))
    paper_rgb = np.repeat(paper_rgb.mean(2, keepdims=True), 3, 2)   # grey paper, red ink stays red
    paper = Image.fromarray((paper_rgb * 255).astype(np.uint8)).resize((w, h))
    alpha = Image.fromarray((paper_a * 255).astype(np.uint8)).resize((w, h))
    d = ImageDraw.Draw(paper)
    y = m
    if title:
        _txt(d, (w / 2, y + fs), title, fs * 1.35, MONO, ink)
        y += fs * 3
    for i, t in enumerate(lines):
        x = m + rng.normal(0, w * 0.002)
        d.text((x, y), t, font=f, fill=ink)
        if i in strike:
            tw = d.textlength(t, font=f)
            yy = y + fs * 0.55 + rng.normal(0, fs * 0.08)
            d.line([x - fs * 0.3, yy, x + tw + fs * 0.3, yy + rng.normal(0, fs * 0.15)],
                   fill=red, width=max(1, fs // 6))
        y += fs * 1.7
    if sign:
        d.text((m, y + fs), sign, font=f, fill=ink)
    a = np.asarray(paper, np.float32) / 255
    a = a + (1 - a) * np.clip(0.45 * fbm(a.shape[:2], rng, w * 0.015, 3), 0, 0.7)[..., None]
    return np.clip(a, 0, 1), np.asarray(alpha, np.float32) / 255


def _as_list(x):
    """List params arrive as lists, JSON strings ('["a","b"]') or 'a/b' strings."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except ValueError:
            x = [t.strip() for t in x.split("/") if t.strip()]
    return x if isinstance(x, list) else [x]


def gen_print(L, ctx):
    """Printed ephemera with real text: params.variant in PRINTS (tobacco_talon,
    sugar_talon, soap_talon, papirosy), variant 'stamp' with params.ring/center, or
    variant 'typed' with params.title / lines / strike (indices) / sign."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    v = p.get("variant", "tobacco_talon")
    size = int(s * (0.14 + 0.5 * float(p.get("coverage", 0.15))))
    if v == "stamp":
        rgba = render_stamp(size, rng, [str(t) for t in _as_list(p.get("ring", ["ИСПОЛКОМ", "ГОРСОВЕТ", "1990"]))],
                            p.get("center", "ВЫДАНО"), tuple(int(c * 255) for c in hex_rgb(p.get("color"))))
        return place_object(L, ctx, rgba[..., :3], rgba[..., 3], size / s)
    if v == "typed":
        rgb_o, a_o = render_typed(size, rng, p.get("title", ""), [str(t) for t in _as_list(p.get("lines") or [])],
                                  tuple(int(i) for i in _as_list(p.get("strike", []))),
                                  p.get("sign", "Подпись ____________"))
        return place_object(L, ctx, rgb_o, a_o, size / s)
    rgb_o, a_o = render_print(v, size, rng)
    return place_object(L, ctx, rgb_o, a_o, size / s)


# ---------------------------------------------------------------- on the subject
def gen_skin_overlay(L, ctx):
    """Dense rune script tattooed onto the skin: multiply, clipped to the subject,
    strongest in the midtones, patchy so it reads as ink on skin not a pattern fill."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    g = s * 0.022
    tex = rune_block(ctx.W, ctx.H, ctx, g, 0.14, "brush", vertical=False, gap=0.02)
    # thin crack/vein lines
    for _ in range(int(20 * float(p.get("coverage", 0.5)))):
        x, y = region_pt("around", ctx)[0]
        pts = [(x, y)]
        ang = rng.uniform(0, 2 * math.pi)
        for _ in range(int(rng.integers(10, 40))):
            ang += rng.normal(0, 0.5)
            x += math.cos(ang) * s * 0.006
            y += math.sin(ang) * s * 0.006
            pts.append((x, y))
        cv2.polylines(tex, [np.array(pts, np.int32)], False, 1.0, max(1, int(s * 0.0012)), cv2.LINE_AA)
    mid = np.clip(1 - np.abs(ctx.lum - 0.5) * 1.6, 0, 1)
    patch = np.clip(0.5 + 0.8 * fbm((ctx.H, ctx.W), rng, s * 0.08, 3), 0, 1)
    cov = float(p.get("coverage", 0.55))
    a = tex * mid * patch * ctx.mask * float(p.get("opacity", 0.6)) * min(1.0, cov * 1.6)
    return solid(ctx, p.get("color", "#1a1a22"), a, blend="multiply", clip="subject")


GENERATORS = {
    "paint_pattern": lambda L, c: gen_runes(L, c, "block"),
    "calligraphy": lambda L, c: gen_runes(L, c, "brush"),
    "ink_smear": gen_ink_smear,
    "splatter": gen_splatter,
    "dot_rows": gen_dot_rows,
    "pen_loops": gen_pen_loops,
    "scribbles": gen_scribbles,
    "micro_text": gen_micro_text,
    "glyph_rain": gen_glyph_rain,
    "stain": gen_stain,
    "paper_scraps": gen_paper_scraps,
    "asset": gen_asset,
    "print": gen_print,
    "skin_overlay": gen_skin_overlay,
}
# handled by the compositor itself (they change the subject / the whole print):
# subject_cut, grain, scratches
