#!/usr/bin/env python3
"""Frame a sequence of photos of one person so they cut together.

One solver replaces three competing crop modes that each broke something:

  * NO DISTORTION. Every image keeps its own aspect. RAWs are decoded at their
    native shape — resizing a RAW to a Lightroom export's pixel size stretched
    every frame whose export had been cropped in Lightroom.
  * WHOLE SUBJECT. The person's box comes from pose landmarks PLUS headroom
    (pose stops at the eyes, which cut heads off), and the crop is sized to hold
    every frame's box. Where the crop runs past the photo, it is padded with a
    blurred, darkened copy of that photo — never by cutting the subject.
  * EYE LOCK. Each frame is scaled so the inter-ocular distance matches the
    run's median (outliers rejected) and translated so the eyes land on the
    same pixel. This is what made the Nastia burst work.

The aspect is chosen per sequence, tallest first (9:16, 4:5, 1:1, 5:4), taking
the first that needs little padding. The aspect is never changed afterwards —
the final resize is uniform.
"""
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

FACE = os.path.expanduser("~/openclaw-venv/mediapipe_models/face_landmarker.task")
POSE = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")
ASPECTS = [("9:16", 9 / 16, (1080, 1920)), ("4:5", 4 / 5, (1080, 1350)),
           ("1:1", 1.0, (1080, 1080)), ("5:4", 5 / 4, (1350, 1080))]
MAX_PAD = 0.06           # accept an aspect if <=6% of the frame area is padding
_face = _pose = None


def _detectors():
    global _face, _pose
    if _face is None:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        _face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=FACE), num_faces=2))
        _pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=POSE), num_poses=2))
    return _face, _pose


def load_raw(path, max_side=2400):
    """Decode a CR2/CR3 at its NATIVE aspect ratio."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
    im = Image.fromarray(rgb)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def measure(im):
    """Eyes, inter-ocular distance and subject box (pixels) for one image.
    Returns None if no person is found. `people` counts detected faces."""
    import mediapipe as mp
    face, pose = _detectors()
    W, H = im.size
    small = im if max(W, H) <= 1600 else im.resize(
        (round(W * 1600 / max(W, H)), round(H * 1600 / max(W, H))))
    k = W / small.width
    arr = np.asarray(small.convert("RGB"), dtype=np.uint8)
    mpim = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    fr, pr = face.detect(mpim), pose.detect(mpim)
    sw, sh = small.size

    eyes = iod = None
    top = None
    if fr.face_landmarks:
        L = fr.face_landmarks[0]
        lx, ly = (L[33].x + L[133].x) / 2 * sw, (L[33].y + L[133].y) / 2 * sh
        rx, ry = (L[362].x + L[263].x) / 2 * sw, (L[362].y + L[263].y) / 2 * sh
        eyes = ((lx + rx) / 2 * k, (ly + ry) / 2 * k)
        iod = float(np.hypot(lx - rx, ly - ry)) * k
        fh = (L[152].y - L[10].y) * sh * k            # forehead to chin
        top = L[10].y * sh * k - 0.45 * fh           # room for hair
    xs, ys = [], []
    if pr.pose_landmarks:
        P = pr.pose_landmarks[0]
        pts = [(p.x * sw * k, p.y * sh * k) for p in P if getattr(p, "visibility", 1) > 0.4]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if eyes is None and len(P) > 5:
            ex = (P[2].x + P[5].x) / 2 * sw * k
            ey = (P[2].y + P[5].y) / 2 * sh * k
            eyes = (ex, ey)
            iod = float(np.hypot((P[2].x - P[5].x) * sw, (P[2].y - P[5].y) * sh)) * k or None
    if not xs and eyes is None:
        return None
    if eyes and iod:
        top = min(top if top is not None else 1e9, eyes[1] - 2.6 * iod)
        xs += [eyes[0] - 2.2 * iod, eyes[0] + 2.2 * iod]
        ys += [top, eyes[1] + 3.0 * iod]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 0.04 * max(x1 - x0, y1 - y0)
    box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
    return dict(eyes=eyes, iod=iod, box=box, size=(W, H),
                people=max(len(fr.face_landmarks), len(pr.pose_landmarks)))


def same_photo(a, b, ma=None, mb=None):
    """Correlation between two renderings of (supposedly) one photograph — e.g.
    a RAW and its Lightroom export. Compares a window around the eyes after
    matching scale, so a Lightroom crop or a grade does not sink the score the
    way a whole-frame comparison did. Frame numbers recycle across sessions;
    this is what stops a stranger's RAW being attached to a hero."""
    ma = ma or measure(a)
    mb = mb or measure(b)

    def window(im, m):
        if m and m["eyes"] and m["iod"]:
            ex, ey = m["eyes"]; r = 3.2 * m["iod"]
            im = im.crop((ex - r, ey - r, ex + r, ey + r))
        else:
            s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                          (im.width + s) // 2, (im.height + s) // 2))
        v = np.asarray(im.convert("L").resize((96, 96)), dtype=float)
        return (v - v.mean()) / (v.std() + 1e-6)

    return float((window(a, ma) * window(b, mb)).mean())


def solve(measures, iod_tol=0.30):
    """Common crop geometry for a run. Returns a plan dict, or None."""
    ok = [m for m in measures if m and m["eyes"] and m["iod"]]
    if not ok:
        return None
    iods = sorted(m["iod"] for m in ok)
    med = iods[len(iods) // 2]
    ref = med
    scales = []
    for m in measures:
        if not (m and m["eyes"] and m["iod"]):
            scales.append(None)
            continue
        iod = m["iod"] if abs(m["iod"] - med) / med <= iod_tol else med
        scales.append(ref / iod)
    L = T = R = B = 0.0
    for m, s in zip(measures, scales):
        if s is None:
            continue
        ex, ey = m["eyes"]; x0, y0, x1, y1 = m["box"]
        L = max(L, (ex - x0) * s); R = max(R, (x1 - ex) * s)
        T = max(T, (ey - y0) * s); B = max(B, (y1 - ey) * s)
    need_w, need_h = L + R, T + B

    best = None
    for name, ar, out in ASPECTS:
        if need_w / need_h > ar:
            w, h = need_w, need_w / ar
        else:
            w, h = need_h * ar, need_h
        tx = L + (w - need_w) / 2
        ty = T + (h - need_h) / 2
        pads = []
        for m, s in zip(measures, scales):
            if s is None:
                continue
            ex, ey = m["eyes"]; W, H = m["size"]
            cx0, cy0 = ex * s - tx, ey * s - ty
            ix0, iy0 = max(cx0, 0), max(cy0, 0)
            ix1, iy1 = min(cx0 + w, W * s), min(cy0 + h, H * s)
            inside = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            pads.append(1 - inside / (w * h))
        pad = max(pads) if pads else 1
        cand = dict(aspect=name, ar=ar, out=out, w=w, h=h, tx=tx, ty=ty,
                    scales=scales, pad=pad, median_iod=med)
        if pad <= MAX_PAD:
            return cand
        if best is None or pad < best["pad"]:
            best = cand
    return best


def render(im, m, plan, i):
    """Crop one image per the plan, padding past its edges. Uniform resize only."""
    ow, oh = plan["out"]
    s = plan["scales"][i]
    W, H = im.size
    if s is None:                       # no eyes: centre the subject box instead
        x0, y0, x1, y1 = m["box"] if m else (0, 0, W, H)
        s = min(plan["w"] / max(1, x1 - x0), plan["h"] / max(1, y1 - y0))
        ex, ey = (x0 + x1) / 2, (y0 + y1) / 2
        tx, ty = plan["w"] / 2, plan["h"] / 2
    else:
        ex, ey = m["eyes"]
        tx, ty = plan["tx"], plan["ty"]
    k = ow / plan["w"]                  # plan space -> output pixels
    sc = s * k
    # Zoom in rather than pad: if at the eye-locked scale the photo is smaller
    # than the frame, enlarge it just enough to cover — unless that would push
    # the subject out, in which case padding is the lesser harm.
    cover = max(ow / W, oh / H)
    if sc < cover and m:
        bw = max(1.0, m["box"][2] - m["box"][0])
        bh = max(1.0, m["box"][3] - m["box"][1])
        sc = min(cover, max(sc, min(ow / bw, oh / bh)))
    scaled = im.resize((max(1, round(W * sc)), max(1, round(H * sc))), Image.LANCZOS)
    ox = ex * sc - tx * k
    oy = ey * sc - ty * k
    # Soft lock: slide the crop back inside the photo when the exact eye lock
    # would pad, but only as far as the whole subject stays in frame. A few
    # pixels of eye drift reads fine; a blurred band across the frame does not.
    if m:
        SW, SH = W * sc, H * sc
        bx0, by0, bx1, by1 = (v * sc for v in m["box"])
        for axis in (0, 1):
            o, size, full = (ox, ow, SW) if axis == 0 else (oy, oh, SH)
            lo_s, hi_s = ((bx1 - size, bx0) if axis == 0 else (by1 - size, by0))
            lo_p, hi_p = 0.0, full - size
            lo, hi = max(lo_s, lo_p), min(hi_s, hi_p)
            if lo <= hi:
                o = min(max(o, lo), hi)
            elif lo_s <= hi_s:
                o = min(max(o, lo_s), hi_s)
            if axis == 0:
                ox = o
            else:
                oy = o
    ox, oy = round(ox), round(oy)
    # padding: the same photo, cover-filled, blurred and darkened
    bg_s = max(ow / W, oh / H)
    bg = im.resize((round(W * bg_s), round(H * bg_s)), Image.BILINEAR)
    bg = bg.crop(((bg.width - ow) // 2, (bg.height - oh) // 2,
                  (bg.width - ow) // 2 + ow, (bg.height - oh) // 2 + oh))
    bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(40))).enhance(0.45)
    canvas = bg.convert("RGB")
    canvas.paste(scaled.convert("RGB"), (-ox, -oy))
    ix = max(0, min(ow, scaled.width - ox) - max(0, -ox))
    iy = max(0, min(oh, scaled.height - oy) - max(0, -oy))
    render.last_pad = 1 - (ix * iy) / (ow * oh)     # residual padding, honest
    return canvas


def frame_independent(items, aspects=ASPECTS[:2], force=None):
    """Crop unrelated photos (a theme, a set's highlights) to ONE shared aspect.

    No eye lock — they are different people, scales and scenes — but the same
    rules as a burst: the whole subject stays in, nothing is stretched, and a
    photo too wide for the frame is padded rather than cut. The aspect is the
    tallest one whose worst photo needs the least padding.
    Returns [(name, PIL)], aspect name, notes."""
    ms = [measure(im) for _, im in items]

    def crop_for(im, m, ar):
        W, H = im.size
        x0, y0, x1, y1 = m["box"] if m else (0, 0, W, H)
        bw, bh = x1 - x0, y1 - y0
        h = min(H, max(bh, bw / ar))          # tall enough for the subject...
        w = h * ar
        if w > W:                             # ...and never wider than the photo
            w, h = W, W / ar
        h = max(h, bh)                        # subject taller than that: pad
        w = max(w, bw)                        # subject wider than that: pad
        if w / h > ar:
            h = w / ar
        else:
            w = h * ar
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left = min(max(cx - w / 2, 0), max(0, W - w)) if w <= W else cx - w / 2
        top = min(max(cy - h / 2, 0), max(0, H - h)) if h <= H else cy - h / 2
        inside = max(0, min(W, left + w) - max(0, left)) * max(0, min(H, top + h) - max(0, top))
        return (left, top, w, h), 1 - inside / (w * h)

    best = None
    if force:                                  # the aspect Ron already approved
        aspects = [a for a in ASPECTS if a[0] == force] or aspects
    for name, ar, out in aspects:
        pads = [crop_for(im, m, ar)[1] for (_, im), m in zip(items, ms)]
        worst = max(pads)
        if best is None or worst < best[1] - 0.02:
            best = ((name, ar, out), worst)
    (name, ar, (ow, oh)), worst = best
    result = []
    for (fn, im), m in zip(items, ms):
        (left, top, w, h), _ = crop_for(im, m, ar)
        k = ow / w
        W, H = im.size
        scaled = im.resize((max(1, round(W * k)), max(1, round(H * k))), Image.LANCZOS)
        bg_s = max(ow / W, oh / H)
        bg = im.resize((round(W * bg_s), round(H * bg_s)), Image.BILINEAR)
        bg = bg.crop(((bg.width - ow) // 2, (bg.height - oh) // 2,
                      (bg.width - ow) // 2 + ow, (bg.height - oh) // 2 + oh))
        canvas = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(40))).enhance(0.45).convert("RGB")
        canvas.paste(scaled.convert("RGB"), (-round(left * k), -round(top * k)))
        result.append((fn, canvas))
    notes = [f"framing: {name} {ow}x{oh}, subject-fit per photo, worst padding {worst * 100:.1f}%"]
    return result, name, notes


def frame_sequence(items, outdir):
    """items: [(filename, PIL image)] in playback order. Writes every frame and
    returns (plan, notes). Refuses to guess when nobody can be measured."""
    ms = [measure(im) for _, im in items]
    plan = solve(ms)
    if plan is None:
        raise SystemExit("no eyes found in any frame; cannot lock the framing")
    notes, pads = [], []
    for i, ((name, im), m) in enumerate(zip(items, ms)):
        render(im, m, plan, i).save(os.path.join(outdir, name), quality=94)
        pads.append(render.last_pad)
        if m is None or plan["scales"][i] is None:
            notes.append(f"  {name}: no eyes found — centred on the subject instead")
        elif m.get("people", 1) > 1:
            notes.append(f"  {name}: WARNING {m['people']} people detected")
    notes.insert(0, f"framing     : {plan['aspect']} {plan['out'][0]}x{plan['out'][1]}, "
                    f"eye-locked, iod median {plan['median_iod']:.0f}px, "
                    f"padding worst {max(pads) * 100:.1f}%")
    return plan, notes
