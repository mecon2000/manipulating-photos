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
    photo: np.ndarray | None = None   # toned photo HxWx3 (base.look=photo): ghosts/smears are made of it
    hair: np.ndarray | None = None    # MediaPipe hair mask HxW 0-1 (only built when a layer needs it)
    mesh: np.ndarray | None = None    # 468-pt face mesh (N,2) canvas px (pen sketches)
    hands: np.ndarray | None = None   # MediaPipe hands mask HxW 0-1 (selective drags)

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


CALLIG_FONT = str(Path(__file__).resolve().parent / "fonts" / "RuslanDisplay-Regular.ttf")


def text_block(text, g, font=CALLIG_FONT):
    """REAL words (Cyrillic ok) as brush lettering, glyph height ≈ g; '/' starts a new
    line (centred). Alpha 0-1."""
    f = ImageFont.truetype(font, max(8, int(g)))
    lines = [t.strip() for t in text.split("/") if t.strip()] or [text]
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbs = [d0.textbbox((0, 0), t, font=f) for t in lines]
    lh = max(b[3] - b[1] for b in bbs)
    W = max(b[2] - b[0] for b in bbs) + int(g * 0.4)
    H = int(lh * len(lines) + g * 0.25 * (len(lines) - 1) + g * 0.4)
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for i, (t, b) in enumerate(zip(lines, bbs)):
        x = (W - (b[2] - b[0])) / 2 - b[0]
        y = g * 0.2 + i * (lh + g * 0.25) - b[1]
        dr.text((x, y), t, font=f, fill=255)
    return np.asarray(im, np.float32) / 255


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
        # params.text: real words instead of runes. They take the FIRST line's spot, size and
        # rotation; the other lines are dropped (rng still drawn, so nothing else moves).
        text = str(p.get("text") or p.get("words") or "").strip()
        for i in range(n):
            g = s * rng.uniform(0.07, 0.13)
            cnt = rng.integers(4, 9)
            bw_, bh_ = (int(g * 1.2), int(g * 1.15 * cnt)) if vertical else (int(g * 1.15 * cnt), int(g * 1.3))
            blk = rune_block(bw_, bh_, ctx, g, 0.2, "brush", vertical=False, gap=0.0)
            ang = rng.normal(0, 6)
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if text:
                if i:
                    continue
                gt = g * float(p.get("text_scale", 0.8))
                la = [float(v) for v in str(p.get("line_angles") or "").replace(";", ",").split(",") if v.strip()]
                if la:                                # each '/' line gets its own tilt, stacked
                    parts = [t.strip() for t in text.split("/") if t.strip()]
                    rows = [rotate(text_block(t, gt), la[min(k, len(la) - 1)]) for k, t in enumerate(parts)]
                    W_ = max(r.shape[1] for r in rows)
                    H_ = sum(int(r.shape[0] * 0.8) for r in rows) + rows[-1].shape[0]
                    blk = np.zeros((H_, W_), np.float32)
                    y_ = 0
                    for k, r in enumerate(rows):
                        x_ = int(W_ * 0.12 * k) if k else 0      # later lines step right a little
                        x_ = min(x_, W_ - r.shape[1])
                        sub = blk[y_:y_ + r.shape[0], x_:x_ + r.shape[1]]
                        np.maximum(sub, r[:sub.shape[0], :sub.shape[1]], out=sub)
                        y_ += int(r.shape[0] * 0.8)
                else:
                    blk = text_block(text, gt)
                    if vertical:
                        blk = np.rot90(blk, -1)
                    blk = rotate(blk, float(p["angle"]) if p.get("angle") not in (None, "") else ang)
                if p.get("at"):                       # pin the words to a spot (normalized centre)
                    ax, ay = [float(v) for v in str(p["at"]).split(",")[:2]]
                    cx, cy = ax * ctx.W, ay * ctx.H
                m = ctx.s * 0.03                      # words must be whole: keep inside the frame
                if blk.shape[1] > ctx.W - 2 * m:
                    blk = cv2.resize(blk, (int(ctx.W - 2 * m), int(blk.shape[0] * (ctx.W - 2 * m) / blk.shape[1])))
                cx = min(max(cx, m + blk.shape[1] / 2), ctx.W - m - blk.shape[1] / 2)
                cy = min(max(cy, m + blk.shape[0] / 2), ctx.H - m - blk.shape[0] / 2)
                stamp(acc, blk, cx, cy)
                continue
            stamp(acc, rotate(blk, ang), cx, cy)
    # light dry-brush breakup + chipped edges (marks must still read as glyphs)
    chip = np.clip((1.7 if (p.get("text") or p.get("words")) else 1.4)
                   + 0.9 * fbm(acc.shape, rng, s * 0.003, 2), 0, 1)
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
        d = base_dir + rng.normal(0, 35)
        if p.get("burst") not in (None, False, "false", 0, "0"):   # strokes radiate from her head
            hx, hy = ctx.head
            d = -math.degrees(math.atan2(cy - hy, cx - hx)) + rng.normal(0, 8)
        stamp(acc, rotate(np.ascontiguousarray(st), d), cx, cy)
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
    k = float(p.get("scale", 1.0))          # bigger drops + wider spread (zoomed-in subject)
    for cx, cy in region_pt(L.get("region", "around"), ctx, int(3 + cov * 20)):
        spread = s * rng.uniform(0.03, 0.12) * k ** 0.5
        for _ in range(int(30 + cov * 300)):
            r = s * min(0.03 if big else 0.012, 0.0008 * (rng.pareto(2.2) + 1)) * k
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
    if lines:
        # a readable block: lines stacked top→bottom in the order given, left-aligned.
        # params.at = "x,y" (normalized top-left of the block), else random inside the region
        widest = max(len(braille_cells(t)) for t in lines) * cell
        at = p.get("at")
        if at:
            ax, ay = [float(v) for v in str(at).split(",")[:2]]
            bx, by = ax * ctx.W, ay * ctx.H
        else:
            bx = rng.uniform(x0, max(x0 + 1, x1 - widest))
            by = rng.uniform(y0 + 0.1 * (y1 - y0), max(y0 + 1, y1 - len(lines) * line * 1.5))
        for i, t in enumerate(lines):
            x, y = bx, by + i * line * 1.5
            for c in braille_cells(t):
                for kk in (c or ""):
                    dx, dy = DOT_POS[kk]
                    cv2.circle(big, (int((x + dx * d) * k), int((y + dy * d) * k)),
                               max(1, int(r * k)), 255, -1, cv2.LINE_AA)
                x += cell
        img = cv2.resize(big, (ctx.W, ctx.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return solid(ctx, p.get("color", "#d4001e"), img * float(p.get("opacity", 1.0)))
    for i in range(int(p.get("rows", 3))):
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
    n = int(p.get("count") or (2 + cov * 10))
    for cx, cy in region_pt(L.get("region", "behind-head"), ctx, n):
        R = int(s * rng.uniform(0.04, 0.08) * float(p.get("scale", 1.0)))   # scale 4+: one giant partial print
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
        if rng.random() < float(p.get("shadow", 0.4)):     # paper lifts off the wall a little
            off = max(1, int(rng.uniform(0.003, 0.007) * s))
            shd = gaussian_filter(np.pad(mm[..., 0], ((off, 0), (off, 0)))[:Y1 - Y0, :X1 - X0], 0.004 * s) * 0.4
            rgb[Y0:Y1, X0:X1] *= (1 - shd[..., None])
            al[Y0:Y1, X0:X1] = np.maximum(al[Y0:Y1, X0:X1], shd)
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
        if p.get("angle") not in (None, ""):
            ang = float(p["angle"])
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
    # params.unreadable (set on pieces made after this fix) — older pieces keep their
    # original prompt so a re-render reproduces the same scraps.
    stripped = re.sub(r"[\'\"‘’“”][^\'\"‘’“”]{1,40}[\'\"‘’“”]", "", content)
    if stripped != content or p.get("unreadable"):
        content = re.sub(r"\s+,", ",", stripped).strip(" ,")
        prompt = (f"{content}, black and white print, all text tiny and unreadable, "
                  "no headlines, no large words, flat scan of the full page")
    else:
        prompt = f"{content}, black and white print, flat scan of the full page"
    src = asset_image(prompt,
                      L.get("seed", 1))
    rgb_o, a_o = paper_object(src)
    ys, xs = np.nonzero(a_o > 0.99)
    if len(xs):   # interior only — never crop the white backdrop
        y0, y1 = np.percentile(ys, [8, 92]).astype(int)
        x0, x1 = np.percentile(xs, [8, 92]).astype(int)
        src = Image.fromarray((rgb_o[y0:y1, x0:x1] * 255).astype(np.uint8))
    return _scraps(L, ctx, src, gray=not p.get("keep_color"))


LIBRARY = ASSETS / "library"


def library_asset(L, ctx):
    """A pre-made cut-out from assets/library/ (RGBA, real alpha from BiRefNet), placed in
    the layer's region with an optional drop shadow. params: library (id = <cat>/<slug>),
    coverage, opacity, gray (default false: library items keep their colour), shadow
    (0-1 chance), angle (deg, default random +-15)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    im = Image.open(LIBRARY / f"{p['library']}.png").convert("RGBA")
    a = np.asarray(im, np.float32) / 255
    C, A = a[..., :3], a[..., 3]
    ys, xs = np.nonzero(A > 0.05)
    if len(xs):
        C, A = C[ys.min():ys.max() + 1, xs.min():xs.max() + 1], A[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cov = min(float(p.get("coverage", 0.06)), 0.08)       # accents, never the main event
    target = int(math.sqrt(max(1, cov * ctx.W * ctx.H)) * 1.3)
    sc = target / max(C.shape[:2])
    w, h = max(8, int(C.shape[1] * sc)), max(8, int(C.shape[0] * sc))
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    C = cv2.resize(C, (w, h), interpolation=cv2.INTER_AREA)
    if p.get("gray") in (True, "true", 1, "1"):
        C[:] = C.mean(2, keepdims=True)
    ang = float(p.get("angle", rng.uniform(-15, 15)))
    C = np.dstack([rotate(C[..., c], ang) for c in range(3)])
    A = rotate(A, ang)
    h, w = A.shape
    x0, y0, x1, y1 = region_box(L.get("region", "anywhere"), ctx)
    fz = _face_soft(ctx, 2.0) > 0.05 if ctx.face else None
    for _try in range(12):                                # keep clear of her face
        cx, cy = rng.uniform(x0, max(x0 + 1, x1)), rng.uniform(y0, max(y0 + 1, y1))
        if fz is None:
            break
        bx0, by0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
        if not fz[by0:int(cy + h / 2), bx0:int(cx + w / 2)].any():
            break
    rgb = np.zeros((ctx.H, ctx.W, 3), np.float32)
    al = np.zeros((ctx.H, ctx.W), np.float32)
    X, Y = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(0, X), max(0, Y), min(ctx.W, X + w), min(ctx.H, Y + h)
    if X1 > X0 and Y1 > Y0:
        A_ = A[Y0 - Y:Y1 - Y, X0 - X:X1 - X]
        if rng.random() < float(p.get("shadow", 0.6)):
            off = max(1, int(0.006 * s))
            shd = gaussian_filter(np.pad(A_, ((off, 0), (off, 0)))[:Y1 - Y0, :X1 - X0], 0.005 * s) * 0.45
            al[Y0:Y1, X0:X1] = shd
        rgb[Y0:Y1, X0:X1] = C[Y0 - Y:Y1 - Y, X0 - X:X1 - X] * A_[..., None]
        al[Y0:Y1, X0:X1] = A_ + al[Y0:Y1, X0:X1] * (1 - A_)
        rgb[Y0:Y1, X0:X1] /= np.maximum(al[Y0:Y1, X0:X1], 1e-3)[..., None]
    return Layer(np.clip(rgb, 0, 1), al * float(p.get("opacity", 0.95)))


def gen_asset(L, ctx):
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    if p.get("library") and (LIBRARY / f"{p['library']}.png").exists():
        return library_asset(L, ctx)
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



# ---------------------------------------------------------------- photographic (smears look)
def _photo(ctx):
    return ctx.photo if ctx.photo is not None else np.repeat(ctx.lum[..., None], 3, 2)


def _shift(a, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _deg(p, key, default):
    try:
        return float(re.sub(r"[^\d.-]", "", str(p.get(key, default))) or default)
    except ValueError:
        return float(default)


def gen_haze(L, ctx):
    """Defocused smoky glow behind the figure: a soft light mass hugging the silhouette
    plus slow smoke texture. Screen blend — lifts the black, never flattens it."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    m = gaussian_filter(ctx.mask, s * float(p.get("spread", 0.12)))
    m = m / max(1e-3, m.max())
    smoke = np.clip(0.55 + 0.9 * fbm((ctx.H, ctx.W), rng, s * 0.18, 4), 0, 1.2)
    a = np.clip(m * smoke, 0, 1) * float(p.get("opacity", 0.45)) * min(1.0, 0.4 + float(p.get("coverage", 0.6)))
    return solid(ctx, p.get("color", "#8a8d98"), a, blend="screen")


def gen_ghost(L, ctx):
    """Out-of-focus duplicates of HER, drifted along one direction (double exposure in
    a shaking hand). Made of the photo itself, so they read as the same body."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    ph, m = _photo(ctx), ctx.mask
    ang = np.deg2rad(_deg(p, "direction", rng.uniform(-180, 180)))
    st = float(p.get("strength", 1.0))       # one dial: more copies, further apart, more opaque
    n = int(round(int(p.get("count", 2)) * max(1.0, st ** 0.5)))
    dist = s * float(p.get("offset", 0.06)) * st ** 0.5
    blur = s * float(p.get("blur", 0.012))
    acc_rgb = np.zeros_like(ph)
    acc_a = np.zeros_like(m)
    for k in range(1, n + 1):
        dx, dy = np.cos(ang) * dist * k, np.sin(ang) * dist * k
        dx += rng.normal(0, dist * 0.25)
        dy += rng.normal(0, dist * 0.25)
        a = gaussian_filter(_shift(m, dx, dy), blur * k)
        c = gaussian_filter(_shift(ph * m[..., None], dx, dy), (blur * k, blur * k, 0))
        w = 1.0 / k
        acc_rgb += c * w
        acc_a += a * w
    rgb = acc_rgb / np.maximum(acc_a[..., None], 1e-3)
    alpha = np.clip(acc_a * st ** 0.5, 0, 1) * min(1.0, float(p.get("opacity", 0.5)) * st ** 0.5)
    return Layer(np.clip(rgb, 0, 1).astype(np.float32), alpha.astype(np.float32))


def gen_motion_smear(L, ctx):
    """Directional motion smear of the body that SPREADS past the silhouette (blur in
    place is invisible), streaked into fine ribs along the motion like a slit-scan."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    ph, m = _photo(ctx), ctx.mask
    ang = np.deg2rad(_deg(p, "direction", rng.uniform(-60, 60)))
    st = float(p.get("strength", 1.0))       # one dial: longer, denser, more of the body
    length = s * float(p.get("length", 0.12)) * st ** 0.5
    steps = int(28 * max(1.0, st ** 0.5))
    acc_rgb = np.zeros_like(ph)
    acc_a = np.zeros_like(m)
    src = ph * m[..., None]
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
    # arc mode: the smear swings around a pivot (params.pivot = x,y normalized) by
    # params.sweep degrees (sign = direction), so it follows a curved body instead of a line
    arc = p.get("mode") == "arc"
    if arc:
        px, py = [float(v) for v in str(p.get("pivot", "0.5,0.9")).split(",")[:2]]
        px, py = px * ctx.W, py * ctx.H
        sweep = float(p.get("sweep", 25)) * st ** 0.5
    for k in range(steps):
        f = k / (steps - 1)
        w = 1 - k / steps
        if arc:
            M = cv2.getRotationMatrix2D((px, py), sweep * f, 1.0)
            warp = lambda a_, M=M: cv2.warpAffine(a_, M, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            acc_rgb += warp(src) * w
            acc_a += warp(m) * w
        else:
            t = f * length
            acc_rgb += _shift(src, np.cos(ang) * t, np.sin(ang) * t) * w
            acc_a += _shift(m, np.cos(ang) * t, np.sin(ang) * t) * w
    rgb = acc_rgb / np.maximum(acc_a[..., None], 1e-3)
    a = acc_a / max(1e-3, acc_a.max())
    if arc:
        perp = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)     # ribs = rings around the pivot
    else:
        perp = -xx * np.sin(ang) + yy * np.cos(ang)      # across the motion
    period = s * float(p.get("rib", 0.006))
    jitter = fbm((ctx.H, ctx.W), rng, s * 0.05, 2) * period * 2
    ribs = 0.5 + 0.5 * np.sin((perp + jitter) / period * 2 * np.pi)
    rib_amt = float(p.get("ribs", 0.6))
    a = a * (1 - rib_amt + rib_amt * ribs)
    # mostly on the trailing side: keep her own body readable underneath
    a = a * (1 - float(p.get("keep_body", 0.6)) * m)
    # only part of the body smears (an arm, a shoulder) — a soft blob picks where
    cov = float(p.get("coverage", 0.5))
    pick = np.clip(0.5 + 1.6 * fbm((ctx.H, ctx.W), rng, s * 0.25, 2) + (cov - 0.5) + 0.25 * (st - 1), 0, 1)
    a = np.clip(a * st, 0, 1) * pick * float(p.get("opacity", 0.85))
    return Layer(np.clip(rgb, 0, 1).astype(np.float32), np.clip(a, 0, 1).astype(np.float32))


def gen_embers(L, ctx):
    """Sparks blown about in wind around her: a general wind (params.direction deg, or
    mode=arc + pivot + sweep for a curved wind) that each spark only loosely follows —
    a turbulent flow field bends every path, and params.chaos (0-1, default 0.45) sets how
    far sparks stray, a fraction tumbling off in their own direction. Nearly monochrome by
    default: params.saturation (0-1, default 0.12) mixes params.color toward grey.
    params.count, size, trail (0-1), spread (fraction of short edge, default 0.3)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    n = int(p.get("count", 160 * float(p.get("coverage", 0.5)) * 2))
    outside = distance_transform_edt(ctx.mask < 0.5)
    spread = s * float(p.get("spread", 0.3))
    depth = max(1.0, s * float(p.get("inside", 0.025)))
    inner = distance_transform_edt(ctx.mask > 0.5)
    w = np.exp(-outside / (spread * 0.6)) * (outside > 0)
    # sparks are also born inside her outline, thinning to none at params.inside deep
    w = w + 3 * (inner > 0) * np.clip(1 - inner / depth, 0, 1) ** 1.5
    bg = outside > 0
    w = w / max(1e-9, w.sum()) * 0.8 + bg / max(1, bg.sum()) * 0.2
    flat = w.ravel() / max(1e-9, w.sum())
    idx = rng.choice(flat.size, size=max(1, n), p=flat)
    ys, xs = np.divmod(idx, W)
    arc = p.get("mode") == "arc"
    if arc:
        px, py = [float(v) for v in str(p.get("pivot", "0.5,0.9")).split(",")[:2]]
        px, py = px * W, py * H
        spin = -np.sign(float(p.get("sweep", -30)) or -1)   # cv2: +sweep = counter-clockwise
    else:
        ang0 = np.deg2rad(_deg(p, "direction", 200))
    chaos = float(p.get("chaos", 0.45))
    # turbulent wind: a smooth random angle field bends each path as it travels
    ds = 4
    turb = fbm((H // ds + 1, W // ds + 1), rng, s * 0.08 / ds, 3) * np.pi * 0.9 * chaos

    def heading(x, y):
        if arc:
            th = np.arctan2(y - py, x - px) + spin * np.pi / 2
        else:
            th = ang0
        yi, xi = int(np.clip(y, 0, H - 1)) // ds, int(np.clip(x, 0, W - 1)) // ds
        return th + turb[yi, xi]

    core = np.zeros((H, W), np.float32)
    trail = float(p.get("trail", 0.5))
    size = float(p.get("size", 1.0))
    steps = 22
    for x, y in zip(xs.astype(np.float32), ys.astype(np.float32)):
        r0 = s * rng.uniform(0.001, 0.0035) * size * (1.7 if rng.random() < 0.06 else 1.0)
        b = rng.uniform(0.25, 1.0) ** 1.5             # most faint, a few bright
        own = rng.normal(0, 0.35 * chaos)             # each spark's own drift
        if rng.random() < 0.25 * chaos:
            own = rng.uniform(-np.pi, np.pi)          # strays, tumbling their own way
        ln = s * rng.uniform(0.015, 0.11) * trail
        step = ln / steps
        wob = rng.uniform(0.5, 2.5)                   # how much it flutters
        tx, ty = [x], [y]
        for k in range(1, steps):                     # walk back along where it came from
            th = heading(tx[-1], ty[-1]) + own + rng.normal(0, 0.12 * wob)
            tx.append(tx[-1] - np.cos(th) * step)
            ty.append(ty[-1] - np.sin(th) * step)
        cv2.circle(core, (int(tx[0]), int(ty[0])), max(1, int(r0)), float(b), -1, cv2.LINE_AA)
        for k in range(1, steps):                    # continuous tapering streak
            f = 1 - k / steps
            cv2.line(core, (int(tx[k - 1]), int(ty[k - 1])), (int(tx[k]), int(ty[k])),
                     float(b * f ** 1.8), max(1, int(2 * r0 * (0.2 + 0.8 * f))), cv2.LINE_AA)
    core = np.clip(core, 0, 1)
    glow = gaussian_filter(core, s * 0.005) * 1.2 + gaussian_filter(core, s * 0.018) * 0.6
    a = np.clip(core + glow, 0, 1) * float(p.get("opacity", 0.55))
    # kept off her: they may push into her outline by params.inside (fraction of the short
    # edge, default 0.025 ≈ 50-100px), feathered, never into her midst
    t_in = np.clip(inner / depth, 0, 1)
    a = a * (1 - t_in * t_in * (3 - 2 * t_in))
    col = hex_rgb(p.get("color", "#ff6a1a"))
    sat = float(p.get("saturation", 0.12))
    col = col.mean() + (col - col.mean()) * sat
    hot = np.array([1.0, 1.0, 0.97], np.float32)            # cores burn white
    rgb = col + (hot - col) * np.clip(core, 0, 1)[..., None]
    return Layer(rgb.astype(np.float32), a.astype(np.float32), blend="screen")


def _spectrum(t):
    """t 0..1 -> rainbow RGB (prism order: red → violet), soft."""
    h = np.clip(t, 0, 1) * 0.78
    hsv = np.stack([h * 179, np.full_like(h, 200), np.full_like(h, 255)], -1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255


def gen_light_leak(L, ctx):
    """Prism / lens-flare rainbow bands raking across body and background (screen)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    n = int(p.get("count", 2))
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
    rgb = np.zeros((ctx.H, ctx.W, 3), np.float32)
    alpha = np.zeros((ctx.H, ctx.W), np.float32)
    base_ang = _deg(p, "direction", rng.uniform(-70, -30))
    for cx, cy in region_pt(L.get("region", "anywhere"), ctx, n):
        ang = np.deg2rad(base_ang + rng.normal(0, 8))
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)     # across the band
        v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)    # along it
        wdt = s * rng.uniform(0.04, 0.09) * float(p.get("width", 1.0))
        ln = s * rng.uniform(0.35, 0.7)
        prof = np.exp(-(u / wdt) ** 2) * np.exp(-(v / ln) ** 2)
        warp = fbm((ctx.H, ctx.W), rng, s * 0.06, 2) * 0.25
        col = _spectrum((u / wdt + 1.2) / 2.4 + warp)
        rgb = rgb * (1 - prof[..., None]) + col * prof[..., None]
        alpha = np.maximum(alpha, prof)
    alpha = gaussian_filter(alpha, s * 0.004) * float(p.get("opacity", 0.5))
    return Layer(rgb, np.clip(alpha, 0, 1), blend="screen")


def gen_vignette(L, ctx):
    """Edges fall to near-black (low-key)."""
    p, s = L.get("params", {}), ctx.s
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
    hx, hy = ctx.head
    cx = ctx.W / 2 * 0.6 + hx * 0.4
    cy = ctx.H / 2 * 0.6 + hy * 0.4
    r = np.sqrt(((xx - cx) / (ctx.W * 0.62)) ** 2 + ((yy - cy) / (ctx.H * 0.62)) ** 2)
    a = np.clip((r - float(p.get("inner", 0.45))) / 0.6, 0, 1) ** 1.6
    return solid(ctx, p.get("color", "#05050a"), a * float(p.get("opacity", 0.85)))

def gen_rim_light(L, ctx):
    """Coloured rim light on the subject's edges that face the light (shifted-mask rim):
    pixels inside her where one step toward the light leaves the silhouette. One-sided,
    follows the form, broken by dry-brush streaks like a painted rim — not a glow.
    params: color, opacity, direction (deg the light comes FROM, screen: 180=left, 0=right,
    90=top), direction2 (optional second light), width (fraction of short edge),
    breakup (0-1 dry-brush gaps), lit_only (0-1: favour lit skin over shadows)."""
    from scipy.ndimage import shift as _shift
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    inside = (gaussian_filter(ctx.mask, 0.004 * s) > 0.5).astype(np.float32)
    w = float(p.get("width", 0.02)) * s
    rim = np.zeros_like(inside)
    dirs = [p.get("direction", 180)] + ([p["direction2"]] if p.get("direction2") not in (None, "") else [])
    for d in dirs:
        a = math.radians(float(re.sub(r"[^\d.-]", "", str(d)) or 180))
        dx, dy = math.cos(a), -math.sin(a)
        sh = _shift(inside, (-dy * w, -dx * w), order=0, mode="nearest")
        rim = np.maximum(rim, inside * (1 - sh))
    rim = gaussian_filter(rim, 0.003 * s) * inside
    br = float(p.get("breakup", 0.4))
    if br > 0:
        n = gaussian_filter(rng.random((ctx.H, ctx.W)).astype(np.float32), (0.03 * s, 0.003 * s))
        n = (n - n.min()) / (np.ptp(n) + 1e-6)
        rim *= np.clip((n - 0.15 * br) * (1 + 3 * br), 0, 1)
    lo = float(p.get("lit_only", 0.3))
    rim *= np.clip((1 - lo) + lo * 1.5 * ctx.lum, 0, 1)
    return solid(ctx, p.get("color", "#d3202f"), np.clip(rim * 1.5, 0, 1) * float(p.get("opacity", 0.9)))


def gen_backdrop(L, ctx):
    """Full-frame painted background (Flux, generated once per prompt+bg_seed and cached),
    cover-fit to the canvas. bg_seed is a RECIPE-level param on purpose: every piece of
    the recipe gets the same backdrop, so a carousel reads as one series.
    params: prompt, bg_seed (default 1), opacity, brightness."""
    p = L.get("params", {})
    if p.get("image"):              # a ready background (e.g. bg_from_stylized.py's AI-version BG)
        im = Image.open(os.path.expanduser(str(p["image"]))).convert("RGB")
        sc = max(ctx.W / im.width, ctx.H / im.height)
        im = im.resize((max(ctx.W, round(im.width * sc)), max(ctx.H, round(im.height * sc))), Image.LANCZOS)
        x0, y0 = (im.width - ctx.W) // 2, (im.height - ctx.H) // 2
        rgb = np.asarray(im.crop((x0, y0, x0 + ctx.W, y0 + ctx.H)), np.float32) / 255
        rgb = np.clip(rgb * float(p.get("brightness", 1.0)), 0, 1)
        return Layer(rgb, np.full((ctx.H, ctx.W), float(p.get("opacity", 1.0)), np.float32))
    prompt = p.get("prompt") or L.get("note") or "abstract dark painted background"
    prompt += ", abstract background texture only, no people, no figure, no face, no text"
    seed = int(p.get("bg_seed", 1))
    land = ctx.W >= ctx.H
    w, h = (1344, 896) if land else (896, 1344)
    key = hashlib.sha1(f"backdrop|{prompt}|{seed}|{w}x{h}".encode()).hexdigest()[:12]
    path = ASSETS / f"backdrop__{key}.png"
    if not path.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        _fal_flux(prompt, seed, w, h).save(path)
        (ASSETS / "index.jsonl").open("a").write(
            json.dumps({"file": path.name, "prompt": prompt, "seed": seed, "kind": "backdrop"}) + "\n")
    im = Image.open(path).convert("RGB")
    sc = max(ctx.W / im.width, ctx.H / im.height)               # cover-fit
    im = im.resize((max(ctx.W, round(im.width * sc)), max(ctx.H, round(im.height * sc))), Image.LANCZOS)
    x0, y0 = (im.width - ctx.W) // 2, (im.height - ctx.H) // 2
    rgb = np.asarray(im.crop((x0, y0, x0 + ctx.W, y0 + ctx.H)), np.float32) / 255
    rgb = np.clip(rgb * float(p.get("brightness", 1.0)), 0, 1)
    return Layer(rgb, np.full((ctx.H, ctx.W), float(p.get("opacity", 1.0)), np.float32))


def gen_dry_brush(L, ctx):
    """Painted shadows on her body (subject-clipped), the way the ref does it: shadows
    break into solid black whose boundary is a ragged dry-brush edge (bristle streaks
    along one direction), and the silhouette edge gets eaten by strokes. Her shape and
    clothes stay; only the shading turns into brushwork.
    params: color, opacity, direction (deg), coverage (0-1: how far into the midtones
    the black reaches), roughness (0-1 streak strength), edge (0-1 strokes eating the
    outline from inside), length (stroke length, fraction of short edge)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    cov = float(p.get("coverage", 0.5))
    rough = float(p.get("roughness", 0.6))
    ang = float(re.sub(r"[^\d.-]", "", str(p.get("direction", "-60"))) or -60)
    ln = float(p.get("length", 0.08)) * s

    def _streaks(sig_along, sig_across):
        n = gaussian_filter(rng.standard_normal((H + W, H + W)).astype(np.float32),
                            (sig_across, sig_along))
        r = rotate(n, ang)
        y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
        r = r[y0:y0 + H, x0:x0 + W]
        return r / (r.std() + 1e-6)

    bristle = _streaks(ln, 0.0015 * s)          # thin hairs of the brush
    clump = _streaks(ln * 1.5, 0.012 * s)       # loaded vs dry passes
    dark = 1 - gaussian_filter(ctx.lum, 0.002 * s)
    field = dark + rough * (0.12 * bristle + 0.18 * clump)
    inside = ctx.mask > 0.5
    e = float(p.get("edge", 0.5))
    if e > 0:
        dist = distance_transform_edt(inside)
        band = np.clip(1 - dist / (0.06 * s), 0, 1)
        field += e * 0.45 * band * np.clip(0.6 + 0.5 * clump, 0, 1.5)
    thr = 1.0 - 0.55 * cov
    a = np.clip((field - thr) * 7, 0, 1)
    a *= gaussian_filter(ctx.mask, 0.002 * s)
    return solid(ctx, p.get("color", "#050609"), a * float(p.get("opacity", 0.95)))


def _face_soft(ctx, grow=1.6):
    """Soft 0-1 mask over the face (from landmarks), 0 elsewhere."""
    m = np.zeros((ctx.H, ctx.W), np.float32)
    f = ctx.face or {}
    if not f or "nose" not in f:
        return m
    pts = np.array([v for v in f.values()], np.float32)
    cx, cy = pts.mean(0)
    r = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.5 * grow + 1
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    return np.clip(1.6 - d, 0, 1).astype(np.float32)


def gen_shutter_drag(L, ctx):
    """Slow-shutter smear of HER OWN photo (not added copies): a directional or rotational
    blur of the toned frame laid over her, kept off the face so it stays readable — the
    'dragged shutter + flash' look. params: mode (linear|rotate|zoom), direction (deg,
    linear), length (fraction of short edge), angle (deg, rotate), strength (0-1),
    face_keep (0-1: how sharp the face stays), spread (0-1: how far the smear leaves her)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    src = ctx.photo if ctx.photo is not None else np.repeat(ctx.lum[..., None], 3, 2)
    mode = p.get("mode", "linear")
    n = 40                          # enough steps (jittered below) that the smear never shows copies
    m3 = ctx.mask[..., None].astype(np.float32)
    src = np.concatenate([src * m3, m3], 2)          # premultiplied: smear her, not her old BG
    acc = np.zeros_like(src)
    H, W = ctx.H, ctx.W
    if mode == "rotate":
        hx, hy = ctx.head
        cx, cy = (hx, hy + 0.25 * (ctx.sbox[3] - ctx.sbox[1]))
        ang = float(p.get("angle", 8))
        for k in range(n):
            M = cv2.getRotationMatrix2D((cx, cy), ang * ((k + rng.uniform(-.5, .5)) / (n - 1) - 0.5), 1.0)
            acc += cv2.warpAffine(src, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    elif mode == "zoom":
        hx, hy = ctx.head
        z = float(p.get("length", 0.08))
        for k in range(n):
            sc = 1 + z * k / (n - 1)
            M = cv2.getRotationMatrix2D((hx, hy), 0, sc)
            acc += cv2.warpAffine(src, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    else:
        a = math.radians(float(re.sub(r"[^\d.-]", "", str(p.get("direction", "0"))) or 0))
        ln = float(p.get("length", 0.12)) * s
        for k in range(n):
            t = ((k + rng.uniform(-.5, .5)) / (n - 1) - 0.5) * ln
            M = np.float32([[1, 0, math.cos(a) * t], [0, 1, -math.sin(a) * t]])
            acc += cv2.warpAffine(src, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    acc /= n
    a_sm = acc[..., 3]
    blur = acc[..., :3] / np.maximum(a_sm, 1e-3)[..., None]
    alpha = np.clip(a_sm * (1 + float(p.get("spread", 0.6))), 0, 1) * float(p.get("strength", 0.8))
    # selective: where the drag lands — hair / body / outer edges / hands, each 0-1. Name
    # any of them and the rest default to 0 (so hands=1 alone drags only her hands).
    # Without any, the whole of her is dragged.
    if any(k in p for k in ("hair", "body", "edges", "hands")):
        hair = ctx.hair if ctx.hair is not None else np.zeros_like(ctx.mask)
        hair = gaussian_filter(hair, 0.01 * s)
        hands = gaussian_filter(ctx.hands, 0.01 * s) if ctx.hands is not None else np.zeros_like(ctx.mask)
        edge = np.clip(1 - distance_transform_edt(ctx.mask > 0.5) / (0.06 * s), 0, 1)
        sel = np.maximum.reduce([float(p.get("hair", 0)) * hair,
                                 float(p.get("body", 0)) * ctx.mask * (1 - hair),
                                 float(p.get("edges", 0)) * edge,
                                 float(p.get("hands", 0)) * np.clip(hands * 1.5, 0, 1)])
        # sharp zone: a vertical band across her that stays crisp (keep_side -1 left …
        # 0 middle … +1 right of her bbox; keep_width 0 = none, 1 = her full width)
        kw = float(p.get("keep_width", 0))
        if kw > 0:
            x0, y0, x1, y1 = ctx.sbox
            u = (np.arange(ctx.W, dtype=np.float32) - (x0 + x1) / 2) / max(1, (x1 - x0) / 2)
            d = np.abs(u - float(p.get("keep_side", 0))) / max(kw, 1e-3)
            band = np.clip(1.5 - d, 0, 1) ** 1.5                      # soft-edged
            sel = sel * (1 - band[None, :])
        out_reach = gaussian_filter(sel, 0.03 * s) * (1 - ctx.mask)     # trails leave her too
        alpha *= np.clip(np.maximum(sel, out_reach * 1.5), 0, 1)
    alpha *= 1 - float(p.get("face_keep", 0.8)) * _face_soft(ctx)
    return Layer(blur.astype(np.float32), alpha.astype(np.float32))


def _chaikin(pts, it=2):
    for _ in range(it):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([pts[:1], np.stack([q, r], 1).reshape(-1, 2), pts[-1:]])
    return pts


# (mesh indices, drawn whole?) — upper lids firm, lower lids/jaw only partly, like a quick sketch
SKETCH_FEATURES = [
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], True),          # upper lid L
    ([263, 466, 388, 387, 386, 385, 384, 398, 362], True),         # upper lid R
    ([33, 7, 163, 144, 145, 153, 154, 155, 133], False),           # lower lid L
    ([263, 249, 390, 373, 374, 380, 381, 382, 362], False),        # lower lid R
    ([70, 63, 105, 66, 107], True),                                # brow L
    ([300, 293, 334, 296, 336], True),                             # brow R
    ([98, 97, 2, 326, 327], True),                                 # nostrils
    ([193, 168, 6, 197, 195, 5], False),                           # bridge (one side)
    ([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], True),     # upper lip
    ([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308], True),    # mouth line
    ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], False),  # lower lip
    ([234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
      397, 288, 361, 323, 454], False),                            # jaw
]


def gen_pen_outline(L, ctx):
    """A human sketch of her in a thin ballpoint: vector strokes along her silhouette plus a
    few long inner lines (face features, folds) — minimal, a little off, some lines gone
    over twice. Each stroke wobbles, tapers, sometimes overshoots; gaps where the hand
    lifted. Skips the hair. Drawn at 2x and scaled down so the line stays fine.
    params: color (default blue ballpoint), opacity, weight (1 = fine ballpoint),
    detail (0-1: how many inner lines), repeat (0-1: share of strokes redrawn),
    wobble (0-1), gaps (0-1), hair (true to draw over hair too)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    K = 2                                             # supersample
    hair = None
    if ctx.hair is not None and p.get("hair") not in (True, "true", 1, "1"):
        hair = gaussian_filter(ctx.hair, 0.004 * s) > 0.25
    det, rep = float(p.get("detail", 0.5)), float(p.get("repeat", 0.5))
    wob, gaps = float(p.get("wobble", 0.5)), float(p.get("gaps", 0.35))
    wt = float(p.get("weight", 1.0))

    paths = []
    m8 = (gaussian_filter(ctx.mask, 0.003 * s) > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        if len(c) > 0.05 * s:
            paths.append(c[:, 0, :].astype(np.float32))
    if det > 0:                                        # inner lines: long Canny edges on her
        lu = (gaussian_filter(ctx.lum, 0.002 * s) * 255).astype(np.uint8)
        ed = cv2.Canny(lu, 40, 110) * (cv2.erode(m8, np.ones((7, 7), np.uint8)) > 0)
        ci, _ = cv2.findContours(ed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ci = sorted(ci, key=len, reverse=True)
        keep = int(4 + 26 * det)
        for c in ci[:keep * 3]:
            pts = c[:, 0, :].astype(np.float32)
            pts = pts[: max(2, len(pts) // 2 + 1)]     # open contour: one side of the edge loop
            if len(pts) > 0.08 * s:
                paths.append(pts)
            if len(paths) > keep + len(cs):
                break
    feats = []
    if ctx.mesh is not None and len(ctx.mesh) >= 468:     # face drawn the way a sketcher does
        M = ctx.mesh
        for idx, full in SKETCH_FEATURES:
            pts = M[idx].copy()
            if not full:                                  # partial: a random stretch of it
                n = len(pts)
                a0 = int(rng.uniform(0, 0.35) * n)
                pts = pts[a0:a0 + max(3, int(rng.uniform(0.5, 0.9) * n))]
            feats.append(pts)
    elif ctx.face:                                     # face: eyes, brows, nose, lips — short lines ok
            face = _face_soft(ctx, 1.0) > 0.25
            luf = (gaussian_filter(ctx.lum, 0.0012 * s) * 255).astype(np.uint8)
            ef = cv2.Canny(luf, 30, 90) * face
            cf, _ = cv2.findContours(ef.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cf = sorted(cf, key=len, reverse=True)[:int(6 + 10 * det)]
            for c in cf:
                pts = c[:, 0, :].astype(np.float32)
                pts = pts[: max(2, len(pts) // 2 + 1)]
                if len(pts) > 0.012 * s:
                    paths.append(pts)

    canvas = np.zeros((H * K, W * K), np.float32)
    nf = len(paths)
    paths = paths + feats
    for pi, pts in enumerate(paths):
        feat = pi >= nf
        if feat:
            pts = np.vstack([pts[0] + (pts[0] - pts[1]) * 0.3, pts]) if len(pts) > 1 else pts
            pts = _chaikin(pts, 2)
        step = max(1, int(0.004 * s)) if not feat and len(pts) > 0.06 * s else 1
        pts = pts[::step]
        if len(pts) < 3:
            continue
        # break into strokes of random length (the hand lifts), drop some (gaps)
        i = 0
        while i < len(pts) - 2:
            ln = int(rng.uniform(0.06, 0.25) * s / step)
            seg = pts[i:i + ln + 1]
            i += ln + int(rng.uniform(0, 0.02) * s / step)
            if feat:                                    # features: one whole stroke each
                seg, i = pts, len(pts)
            if len(seg) < 3 or rng.random() < gaps * (0.12 if feat else 0.6):
                continue
            if hair is not None and not feat:
                inh = hair[np.clip(seg[:, 1].astype(int), 0, H - 1), np.clip(seg[:, 0].astype(int), 0, W - 1)]
                if inh.mean() > 0.3:
                    continue
            passes = 1 + (rng.random() < rep) + (rng.random() < rep * 0.35)
            for k in range(passes):
                q = seg.copy()
                n = len(q)
                t = np.linspace(0, 1, n)
                # low-frequency wobble perpendicular-ish + a small offset per redraw
                amp = (0.002 + 0.004 * wob) * s
                if feat:                                # small features: wobble scales with them
                    amp = min(amp, 0.04 * float(np.ptp(seg, 0).max()) + 0.3)
                ph = rng.uniform(0, 6.28, 2)
                q[:, 0] += amp * np.sin(t * rng.uniform(1, 3) * 3.14 + ph[0]) + (k and rng.normal(0, 0.003 * s))
                q[:, 1] += amp * np.sin(t * rng.uniform(1, 3) * 3.14 + ph[1]) + (k and rng.normal(0, 0.003 * s))
                if rng.random() < 0.4:                 # overshoot past the end
                    d = q[-1] - q[-2]
                    q = np.vstack([q, q[-1] + d * rng.uniform(2, 6)])
                if k:                                  # redraws are shorter, start late / end early
                    a0, a1 = int(rng.uniform(0, 0.3) * n), int(rng.uniform(0.7, 1.0) * n)
                    q = q[a0:max(a0 + 3, a1)]
                q = _chaikin(q, 2) * K
                m = len(q)
                taper = np.sin(np.linspace(0, np.pi, m)) ** 0.5
                ink = rng.uniform(0.55, 0.95) * (0.75 if k else 1.0)
                for j in range(0, m - 1, 2):           # pressure: thickness + ink follow the taper
                    th = max(1, int(round(K * wt * (0.6 + 0.8 * taper[j]))))
                    cv2.line(canvas, tuple(np.int32(q[j])), tuple(np.int32(q[min(j + 2, m - 1)])),
                             float(ink * (0.5 + 0.5 * taper[j])), th, cv2.LINE_AA)
    # ballpoint: ink builds up where lines cross, faint skips along the line
    canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    skip = gaussian_filter(rng.random((H, W)).astype(np.float32), 0.8)
    a = np.clip(canvas * 1.6, 0, 1) * np.clip(0.6 + 1.2 * skip, 0, 1)
    return solid(ctx, p.get("color", "#1f3a9e"), np.clip(a, 0, 1) * float(p.get("opacity", 0.9)))


def gen_line_screen(L, ctx):
    """Diagonal line-screen (engraving/halftone hatch). Line thickness follows tone so it
    reads as printed. params: color, opacity, angle, period (fraction of short edge),
    where (highlights|shadows|subject|background|all), threshold."""
    p, s = L.get("params", {}), ctx.s
    a = math.radians(float(p.get("angle", 45)))
    per = max(2.0, float(p.get("period", 0.006)) * s)
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
    wave = 0.5 + 0.5 * np.sin((xx * math.cos(a) + yy * math.sin(a)) / per * 2 * math.pi)
    where = p.get("where", "highlights")
    lum = ctx.lum
    if where == "highlights":
        tone, reg = lum, np.clip((lum - float(p.get("threshold", 0.6))) * 4, 0, 1) * ctx.mask
    elif where == "shadows":
        tone, reg = 1 - lum, np.clip((float(p.get("threshold", 0.4)) - lum) * 4, 0, 1) * ctx.mask
    elif where == "hair":
        hm = ctx.hair if ctx.hair is not None else np.zeros_like(lum)
        tone, reg = np.full_like(lum, 0.45), np.clip(gaussian_filter(hm, 0.002 * s) * 1.4, 0, 1)
    elif where == "background":
        tone, reg = np.full_like(lum, 0.5), 1 - ctx.mask
    elif where == "subject":
        tone, reg = 1 - lum, ctx.mask
    else:
        tone, reg = np.full_like(lum, 0.5), np.ones_like(lum)
    ink = (wave < 0.25 + 0.5 * tone).astype(np.float32)
    ink = gaussian_filter(ink, 0.5)
    return solid(ctx, p.get("color", "#111111"), ink * reg * float(p.get("opacity", 0.6)))


def _torn_poly(pts, rng, s, rough):
    """Polygon edges made imperfect: midpoint-displaced (torn, jagged), the odd small rip
    (a notch cut into an edge)."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = np.array(pts[i], float), np.array(pts[(i + 1) % n], float)
        seg = [a, b]
        amp = np.linalg.norm(b - a) * 0.035 * rough
        for _ in range(5):                                   # fractal midpoint displacement
            nxt = [seg[0]]
            for u, v in zip(seg[:-1], seg[1:]):
                d = v - u
                nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
                nxt += [(u + v) / 2 + nrm * rng.normal(0, amp), v]
            seg, amp = nxt, amp * 0.55
        if rng.random() < 0.35 * rough:                     # a rip: small V notch inward
            k = rng.integers(2, len(seg) - 2)
            u, v = seg[k - 1], seg[k + 1]
            d = v - u
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
            seg[k] = seg[k] - nrm * rng.uniform(0.004, 0.012) * s * rng.choice([-1, 1])
        out += seg[:-1]
    return np.int32(out)


def _brush_tex(h, w, rng, ang):
    """Thick-brush paint texture: long streaks + clumps, as a -1..1 relief field."""
    n = rng.standard_normal((h, w)).astype(np.float32)
    t = gaussian_filter(n, (max(1, h * 0.002), max(1, w * 0.06)))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    t = cv2.warpAffine(t, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    t += 0.5 * gaussian_filter(n, max(1, w * 0.01))
    return t / (np.abs(t).max() + 1e-6)


def gen_shards(L, ctx):
    """Torn collage shards: almost-triangles cut from painted paper. Imperfect on purpose —
    torn jagged edges, small rips, a tip sometimes cut off, the odd quad. Some are thick
    brush-painted paper (streaky relief + a lit/shaded torn edge), some hatched, many cast
    a soft drop shadow. Fills the background; on 'over' layers keep coverage low.
    params: colors (comma list), coverage (0-1, ~count), size (fraction of short edge),
    opacity, hatch (share hatched), paint (share brush-textured), shadow (share with a drop
    shadow), rough (edge tear 0-1), avoid_subject (0-1: keep off her)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    cols = [c.strip() for c in str(p.get("colors", "#ff2fd0,#6a3cff,#ffffff")).split(",") if c.strip()]
    cov = float(p.get("coverage", 0.5))
    size = float(p.get("size", 0.25)) * s
    rough = float(p.get("rough", 0.7))
    avoid = float(p.get("avoid_subject", 0.0))
    H, W = ctx.H, ctx.W
    rgb = np.zeros((H, W, 3), np.float32)
    al = np.zeros((H, W), np.float32)
    n_sh = int(6 + cov * 70)
    face_zone = _face_soft(ctx, 2.2) > 0.05 if (ctx.face and L.get("z") == "over") else None
    for k in range(n_sh):
        for _try in range(6):                                  # placement (optionally off her)
            cx, cy = rng.uniform(-0.05, 1.05) * W, rng.uniform(-0.05, 1.05) * H
            mi = ctx.mask[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]
            if rng.random() > avoid * mi:
                break
        ang = rng.uniform(0, 2 * math.pi)
        ln = size * rng.uniform(0.35, 1.5)
        wd = ln * rng.uniform(0.15, 0.6)
        tip = np.array([cx + math.cos(ang) * ln / 2, cy + math.sin(ang) * ln / 2])
        bse = np.array([cx - math.cos(ang) * ln / 2, cy - math.sin(ang) * ln / 2])
        nrm = np.array([-math.sin(ang), math.cos(ang)])
        b1 = bse + nrm * wd * rng.uniform(0.3, 1.0)
        b2 = bse - nrm * wd * rng.uniform(0.0, 1.0)
        pts = [tip, b1, b2]
        if rng.random() < 0.4:                                 # tip cut off
            c = rng.uniform(0.08, 0.3)
            pts = [tip + (b1 - tip) * c, b1, b2, tip + (b2 - tip) * c * rng.uniform(0.5, 1.5)]
        elif rng.random() < 0.2:                               # an odd quad
            pts.insert(2, bse + nrm * wd * rng.uniform(-0.3, 0.3) - (tip - bse) * rng.uniform(0.05, 0.2))
        poly = _torn_poly(pts, rng, s, rough)
        x0, y0 = np.maximum(poly.min(0) - int(0.03 * s), 0)
        x1, y1 = np.minimum(poly.max(0) + int(0.03 * s), [W, H])
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        h, w = y1 - y0, x1 - x0
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly - [x0, y0]], 1)
        m = m.astype(np.float32)
        if m.sum() < 10:
            continue
        if face_zone is not None and (m * face_zone[y0:y1, x0:x1]).any():
            continue                                           # over her: never on the face
        col = hex_rgb(cols[rng.integers(len(cols))])
        c3 = np.broadcast_to(col, (h, w, 3)).copy()
        if rng.random() < float(p.get("paint", 0.45)):          # thick brush-painted paper
            t = _brush_tex(h, w, rng, rng.uniform(0, 180))
            gy, gx = np.gradient(gaussian_filter(t, 1.0))
            shade = np.clip(1 + 0.12 * t - 2.5 * (gx + gy), 0.7, 1.25)
            c3 = np.clip(c3 * shade[..., None], 0, 1)
            # thickness: a lit lip on one torn edge, a dark one on the other
            er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(0.002 * s)))
            lip = m - er
            sh = np.roll(np.roll(m, 2, 0), 2, 1)
            c3 = c3 * (1 - lip[..., None] * 0.35 * (sh[..., None] < 0.5)) + \
                lip[..., None] * 0.25 * (sh[..., None] >= 0.5)
            c3 = np.clip(c3, 0, 1)
        elif rng.random() < float(p.get("hatch", 0.3)):        # printed hatch
            yy, xx = np.mgrid[0:h, 0:w]
            m = m * (np.sin((xx + yy) / (0.005 * s)) > -0.2).astype(np.float32)
        if rng.random() < 0.5 and col.mean() < 0.9:            # white torn paper core on an edge
            core = m - cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(0.0025 * s)))
            core *= gaussian_filter(rng.random((h, w)).astype(np.float32), 2) > 0.5
            c3 = c3 * (1 - core[..., None]) + 0.96 * core[..., None]
        a = gaussian_filter(m, 0.6) * rng.uniform(0.95, 1.0)     # paper is opaque
        sl = (slice(y0, y1), slice(x0, x1))
        if rng.random() < float(p.get("shadow", 0.5)):         # drop shadow under the shard
            off = int(rng.uniform(0.003, 0.008) * s)
            shd = gaussian_filter(np.pad(m, ((off, 0), (off, 0)))[:h, :w], 0.004 * s) * 0.45
            rgb[sl] *= (1 - shd[..., None])
            al[sl] = np.maximum(al[sl], shd)
        rgb[sl] = rgb[sl] * (1 - a[..., None]) + c3 * a[..., None]
        al[sl] = np.maximum(al[sl], a)
    return Layer(rgb, al * float(p.get("opacity", 0.95)))


def gen_cracks(L, ctx):
    """Fractal cracks running across her like split stone: branching random-walk lines,
    dark core with a pale chipped lip. The compositor keeps them off the eyes.
    params: color, opacity, count, length (fraction of short edge), width, branch (0-1)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    lines = np.zeros((ctx.H, ctx.W), np.float32)
    ys, xs = np.nonzero(ctx.mask > 0.6)
    if not len(xs):
        return solid(ctx, "#000000", lines)
    w = max(1, int(float(p.get("width", 1.0)) * 0.0018 * s))

    def walk(x, y, ang, ln, depth):
        step = 0.006 * s
        pts = [(x, y)]
        for _ in range(int(ln / step)):
            ang += rng.normal(0, 0.35)
            x, y = x + math.cos(ang) * step, y + math.sin(ang) * step
            pts.append((x, y))
            if depth < 3 and rng.random() < 0.04 * float(p.get("branch", 0.5)):
                walk(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 1.1), ln * 0.45, depth + 1)
        cv2.polylines(lines, [np.int32(pts)], False, 1.0, max(1, w - depth), cv2.LINE_AA)

    for _ in range(int(p.get("count", 4))):
        i = rng.integers(len(xs))
        walk(float(xs[i]), float(ys[i]), rng.uniform(0, 2 * math.pi),
             float(p.get("length", 0.35)) * s, 0)
    lines *= ctx.mask
    lip = np.clip(cv2.dilate(lines, np.ones((3, 3), np.uint8)) - lines, 0, 1) * 0.5
    rgb = np.empty((ctx.H, ctx.W, 3), np.float32)
    rgb[:] = hex_rgb(p.get("color", "#0a0a0a"))
    rgb = rgb * (1 - lip[..., None]) + np.array([0.92, 0.9, 0.86], np.float32) * lip[..., None]
    return Layer(rgb, np.clip(lines + lip, 0, 1) * float(p.get("opacity", 0.9)))


def gen_grunge_frame(L, ctx):
    """Rough black border eating in from the edges: noisy distance-to-edge threshold with
    dry vertical streaks. params: color, opacity, width (fraction of short edge), roughness."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
    d = np.minimum.reduce([xx, yy, ctx.W - 1 - xx, ctx.H - 1 - yy]) / s
    w = float(p.get("width", 0.04))
    n = fbm((ctx.H, ctx.W), rng, 0.01 * s, 4) * float(p.get("roughness", 0.6)) * w
    streak = gaussian_filter(rng.standard_normal((ctx.H, ctx.W)).astype(np.float32), (0.05 * s, 0.002 * s))
    streak /= streak.std() + 1e-6
    a = np.clip((w + n + 0.008 * streak - d) / (0.004 + 1e-6), 0, 1)
    return solid(ctx, p.get("color", "#0a0a0a"), a * float(p.get("opacity", 1.0)))


def gen_spike_burst(L, ctx):
    """Poster 'explosion' behind her head: sharp jagged spikes radiating from a point,
    dry-brush streaks running along each ray (so edges break up like paint dragged
    outward), a rough solid core and droplets flung along the rays.
    params: color, opacity, count (spikes), length (fraction of short edge), width (base
    width, fraction of length), angle (deg, centre of the fan; 90 = up), spread (deg fan
    width, 360 = all round), core (0-1 core size), dry (0-1 streak breakup),
    drops (0-1), center (head|subject|x,y fractions)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    c = str(p.get("center", "head"))
    if "," in c:
        fx, fy = map(float, c.split(","))
        cx, cy = fx * W, fy * H
    elif c == "subject":
        x0, y0, x1, y1 = ctx.sbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        cx, cy = ctx.head
        cy += 0.04 * s
    count = int(p.get("count", 18))
    ln = float(p.get("length", 0.55)) * s
    wd = float(p.get("width", 0.12))
    ang0 = math.radians(float(p.get("angle", 90)))
    spread = math.radians(float(p.get("spread", 220)))
    m = np.zeros((H, W), np.float32)
    for _ in range(count):
        a = ang0 + rng.uniform(-0.5, 0.5) * spread
        L1 = ln * rng.uniform(0.35, 1.15) * (1.3 if rng.random() < 0.15 else 1.0)
        r0 = ln * rng.uniform(0.0, 0.2)
        bw = L1 * wd * rng.uniform(0.4, 1.4)
        dx, dy = math.cos(a), -math.sin(a)
        nx, ny = -dy, dx
        bx, by = cx + dx * r0, cy + dy * r0
        tip = (bx + dx * L1 + nx * rng.normal(0, bw * 0.3), by + dy * L1 + ny * rng.normal(0, bw * 0.3))
        kink = rng.uniform(0.3, 0.7)                     # a bent edge: spikes aren't clean triangles
        mid = (bx + dx * L1 * kink + nx * bw * rng.uniform(0.2, 0.6), by + dy * L1 * kink + ny * bw * rng.uniform(0.2, 0.6))
        poly = np.int32([(bx + nx * bw / 2, by + ny * bw / 2), mid, tip, (bx - nx * bw / 2, by - ny * bw / 2)])
        cv2.fillPoly(m, [poly], 1.0)
    core = float(p.get("core", 0.5))
    if core > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot(xx - cx, yy - cy) / (ln * 0.35 * core + 1)
        m = np.maximum(m, (d + 0.5 * fbm((H, W), rng, 0.03 * s, 3) < 1).astype(np.float32))
    # dry-brush streaks along the rays: noise in polar coordinates (fine in angle, long in radius)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)       # 0..1
    rr = np.hypot(xx - cx, yy - cy) / s
    N = gaussian_filter(rng.standard_normal((64, 1400)).astype(np.float32), (1.2, 1.0))
    N /= N.std() + 1e-6
    streak = cv2.remap(N, (th * 1399).astype(np.float32), (rr * 30).astype(np.float32) % 63,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    dist = distance_transform_edt(m > 0.5) / (0.012 * s)               # 0 at edge -> 1 inside
    dry = float(p.get("dry", 0.6))
    keep = np.clip((streak + 2.2 * np.clip(dist, 0, 1.5) - 0.6 * dry + 0.4 - rr * dry) * 2.5, 0, 1)
    a = m * keep
    # flecks beyond the edges, along the rays
    if float(p.get("drops", 0.5)) > 0:
        fl = (streak > 2.0 - 0.6 * float(p.get("drops", 0.5))) & (rr * s < ln * 1.25)
        fl &= gaussian_filter(m, 0.03 * s) > 0.05
        a = np.maximum(a, fl.astype(np.float32) * 0.9)
    a = gaussian_filter(a, 0.5)
    return solid(ctx, p.get("color", "#E0103C"), a * float(p.get("opacity", 0.95)))


def gen_scrape_streaks(L, ctx):
    """Squeegee / scraped dry paint: a band of streaks dragged in one direction, dense at
    the start, running dry toward the end, ragged ends. params: color, opacity, region,
    direction (deg: 90 = vertical drag), width (band width, fraction of short edge),
    length (fraction), density (0-1), count (bands)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    acc = np.zeros((H, W), np.float32)
    bw = int(float(p.get("width", 0.12)) * s)
    bl = int(float(p.get("length", 0.6)) * s)
    dens = float(p.get("density", 0.6))
    for _ in range(int(p.get("count", 2))):
        n = rng.standard_normal((bl, bw)).astype(np.float32)
        st = gaussian_filter(n, (bl * 0.08, max(0.6, 0.0012 * s)))     # long along the drag
        st /= st.std() + 1e-6
        t = np.linspace(0, 1, bl)[:, None]
        ink = np.clip((st + 1.6 * dens - 2.2 * t ** 1.5) * 2, 0, 1)      # runs dry toward the end
        edge = np.clip(np.minimum(np.arange(bw), bw - 1 - np.arange(bw)) / (0.15 * bw + 1), 0, 1)[None]
        ink *= np.clip(edge + 0.3 * gaussian_filter(rng.standard_normal((bl, bw)).astype(np.float32), 3), 0, 1)
        tile = rotate(ink, float(p.get("direction", 90)) - 90)
        cx, cy = region_pt(L.get("region", "left"), ctx, 1)[0]
        stamp(acc, tile, cx, cy)
    return solid(ctx, p.get("color", "#111111"), np.clip(acc, 0, 1) * float(p.get("opacity", 0.9)))


def gen_depth_trail(L, ctx):
    """Dragged exposure with depth: copies of her travel along a curved 3D path that ends
    BEHIND her — each copy shrinks with depth (perspective), defocuses and fades into the
    backdrop, farthest drawn first; darks (hair, clothes) leave the strongest trail. Lives
    under the subject, so her sharp self stays in front.
    params: end (x,y: where the trail ends, fractions of short edge from her centre),
    bend (x,y: control point of the curve, same units), depth (how far behind: 0.5 = half
    size at the end), steps, opacity, defocus (0-1), haze (0-1 fade toward the backdrop),
    darks (0-1: how much darks dominate), backdrop (hex the trail fades into)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    src = ctx.photo if ctx.photo is not None else np.repeat(ctx.lum[..., None], 3, 2)
    m = ctx.mask.astype(np.float32)
    lum = src.mean(2)
    wdark = float(p.get("darks", 0.7))
    a0 = m * ((1 - wdark) + wdark * np.clip((0.85 - lum) / 0.7, 0, 1))
    rgba = np.dstack([src * a0[..., None], a0]).astype(np.float32)
    x0, y0, x1, y1 = ctx.sbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    def pt(v, d):
        try:
            a, b = map(float, str(v).split(","))
        except ValueError:
            a, b = d
        return np.array([a * s, b * s])
    P2 = pt(p.get("end", "0.35,-0.12"), (0.35, -0.12))
    P1 = pt(p.get("bend", "0.25,0.1"), (0.25, 0.1))
    depth = float(p.get("depth", 0.6))
    n = max(int(p.get("steps", 36)), 56)
    defocus = float(p.get("defocus", 0.6))
    haze = float(p.get("haze", 0.6))
    bd = hex_rgb(p.get("backdrop", "#ECEBE8"))
    out = np.zeros((H, W, 3), np.float32)
    al = np.zeros((H, W), np.float32)
    for k in range(n, 0, -1):                                # far -> near
        t = (k - rng.uniform(0, 1)) / n                      # jittered: no stair-stepped copies
        off = 2 * (1 - t) * t * P1 + t * t * P2             # quadratic bezier from her (t=0)
        sc = 1 / (1 + (1 / max(1e-3, 1 - min(depth, 0.95)) - 1) * t)
        M = np.float32([[sc, 0, (1 - sc) * cx + off[0]], [0, sc, (1 - sc) * cy + off[1]]])
        w = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        sig = defocus * t * 0.012 * s
        if sig > 0.6:
            w = cv2.GaussianBlur(w, (0, 0), sig)
        a = w[..., 3] * (1 - t) ** 0.6 * 0.5
        c = w[..., :3] / np.maximum(w[..., 3:], 1e-3)
        c = c * (1 - haze * t) + bd * (haze * t)             # recedes into the backdrop
        out = out * (1 - a[..., None]) + c * a[..., None]
        al = al + a * (1 - al)
    rgb = out / np.maximum(al, 1e-3)[..., None]
    return Layer(np.clip(rgb, 0, 1).astype(np.float32), np.clip(al, 0, 1) * float(p.get("opacity", 1.0)))


def gen_studio_arcs(L, ctx):
    """Soft arcs/rings in the backdrop — studio gear (ring light, reflector rim) smeared by a
    long exposure. Shaded like tubes (darker rim, lighter core) so they have volume.
    params: count, color (tube tone), opacity, size (radius, fraction of short edge),
    thickness, softness, center (head|x,y fractions), sweep (deg of each arc)."""
    p, s, rng = L.get("params", {}), ctx.s, ctx.rng
    H, W = ctx.H, ctx.W
    c = str(p.get("center", "head"))
    if "," in c:
        fx, fy = map(float, c.split(","))
        cx, cy = fx * W, fy * H
    else:
        cx, cy = ctx.head
        cy += 0.12 * s
    K = 2
    rim = np.zeros((H // K, W // K), np.float32)
    core = np.zeros_like(rim)
    for _ in range(int(p.get("count", 3))):
        r = float(p.get("size", 0.45)) * s * rng.uniform(0.7, 1.3)
        th = max(2, int(float(p.get("thickness", 0.035)) * s * rng.uniform(0.6, 1.4)))
        ccx, ccy = cx + rng.normal(0, 0.08 * s), cy + rng.normal(0, 0.08 * s)
        sweep = float(p.get("sweep", 200)) * rng.uniform(0.6, 1.2)
        st = rng.uniform(150, 260) - sweep / 2
        ax = (int(r / K), int(r / K * rng.uniform(0.75, 1.1)))
        cv2.ellipse(rim, (int(ccx / K), int(ccy / K)), ax, rng.uniform(-20, 20), st, st + sweep, 1.0, max(1, th // K))
        cv2.ellipse(core, (int(ccx / K), int(ccy / K)), ax, rng.uniform(-3, 3), st + 5, st + sweep - 5, 1.0, max(1, th // (3 * K)))
    rim = cv2.resize(rim, (W, H))
    core = cv2.resize(core, (W, H))
    soft = float(p.get("softness", 0.012)) * s
    rim = gaussian_filter(rim, soft)
    core = gaussian_filter(core, soft * 0.7)
    a = np.clip(rim * 1.4, 0, 1)
    tone = hex_rgb(p.get("color", "#9e9c98"))
    rgb = tone[None, None] * (1 - 0.5 * np.clip(core * 1.5, 0, 1))[..., None] + \
        np.array([1, 1, 1], np.float32) * 0.5 * np.clip(core * 1.5, 0, 1)[..., None]
    return Layer(np.clip(rgb, 0, 1).astype(np.float32), a * float(p.get("opacity", 0.5)))


def gen_hair_tone(L, ctx):
    """Recolour her hair (MediaPipe hair mask) to one flat tone — the pop-art white hair —
    keeping the strands as darker lines (high-pass of her own hair) so it still reads as hair.
    params: color, opacity, strands (0-1 how much strand detail shows), soft (edge feather)."""
    p, s = L.get("params", {}), ctx.s
    if ctx.hair is None:
        return None
    lum = ctx.lum.astype(np.float32)
    detail = lum - gaussian_filter(lum, 0.004 * s)
    strands = float(p.get("strands", 0.5))
    shade = 1 - strands * 0.6 * np.clip(-detail * 8, 0, 1)
    col = hex_rgb(p.get("color", "#fbf8ff"))
    rgb = (col[None, None] * shade[..., None]).astype(np.float32)
    a = np.clip(gaussian_filter(ctx.hair, float(p.get("soft", 0.002)) * s) * 1.4, 0, 1)
    return Layer(rgb, a * float(p.get("opacity", 1.0)))


def gen_drop_shadow(L, ctx):
    """Her cast shadow on the wall behind her (z=under): the silhouette shifted, blurred and
    filled flat. params: color, opacity, angle (deg, where the shadow falls; 135 = down-right),
    distance (fraction of short edge), blur (fraction), spread (grow the silhouette, fraction)."""
    p, s = L.get("params", {}), ctx.s
    m = ctx.mask.astype(np.float32)
    grow = float(p.get("spread", 0.0))
    if grow > 0:
        m = np.clip(gaussian_filter(m, grow * s) * 2.2, 0, 1)
    ang = np.deg2rad(float(p.get("angle", 135)))
    dist = float(p.get("distance", 0.03)) * s
    dx, dy = dist * np.cos(ang - np.pi / 2), dist * np.sin(ang - np.pi / 2)
    from scipy.ndimage import shift as _shift
    m = _shift(m, (-dy if False else dy, dx), order=1, mode="constant")
    m = gaussian_filter(m, float(p.get("blur", 0.012)) * s)
    rgb = np.broadcast_to(hex_rgb(p.get("color", "#1a1530"))[None, None], (ctx.H, ctx.W, 3)).astype(np.float32)
    return Layer(rgb.copy(), np.clip(m, 0, 1) * float(p.get("opacity", 0.55)), blend=p.get("blend", "multiply"))


def gen_subject_fill(L, ctx):
    """Her silhouette filled with one flat colour (z=over) — dial opacity to whiten / tint her.
    params: color (default white), opacity, soft (edge feather, fraction of short edge)."""
    p, s = L.get("params", {}), ctx.s
    m = ctx.mask.astype(np.float32)
    soft = float(p.get("soft", 0.0015))
    if soft > 0:
        m = gaussian_filter(m, soft * s)
    rgb = np.broadcast_to(hex_rgb(p.get("color", "#ffffff"))[None, None], (ctx.H, ctx.W, 3)).astype(np.float32)
    return Layer(rgb.copy(), np.clip(m, 0, 1) * float(p.get("opacity", 0.3)), blend=p.get("blend", "normal"))


GENERATORS = {
    "drop_shadow": gen_drop_shadow,
    "subject_fill": gen_subject_fill,
    "hair_tone": gen_hair_tone,
    "depth_trail": gen_depth_trail,
    "studio_arcs": gen_studio_arcs,
    "spike_burst": gen_spike_burst,
    "scrape_streaks": gen_scrape_streaks,
    "shutter_drag": gen_shutter_drag,
    "pen_outline": gen_pen_outline,
    "line_screen": gen_line_screen,
    "shards": gen_shards,
    "cracks": gen_cracks,
    "grunge_frame": gen_grunge_frame,
    "backdrop": gen_backdrop,
    "dry_brush": gen_dry_brush,
    "rim_light": gen_rim_light,
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
    "haze": gen_haze,
    "ghost": gen_ghost,
    "motion_smear": gen_motion_smear,
    "light_leak": gen_light_leak,
    "vignette": gen_vignette,
    "embers": gen_embers,
}
# handled by the compositor itself (they change the subject / the whole print):
# subject_cut, grain, scratches
