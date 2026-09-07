#!/usr/bin/env python3
"""Seamless-looping transform reel from one stylized output.

Structure (default):
    burst-in   N unprocessed frames from the same setup, hard cuts, tightly cropped
    build      unprocessed -> processed -> stylized   (the payoff)
    burst-out  the same beats in reverse, faster

The last frame is the first frame, so IG replays it invisibly and watch time
compounds.  Nothing is re-rendered; every beat already exists on disk.

Transitions come from ffmpeg's built-in xfade (58 of them, no extra install):
  fade wipeleft wiperight wipeup wipedown slide* circlecrop rectcrop distance
  fadeblack fadewhite radial smooth* circleopen circleclose vertopen vertclose
  horzopen horzclose dissolve pixelize diag* hlslice hrslice vuslice vdslice
  hblur fadegrays wipetl wipetr wipebl wipebr squeezeh squeezev zoomin fadefast
  fadeslow hlwind hrwind vuwind vdwind cover* reveal*
"""
import argparse, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PHOTOS = Path(os.path.expanduser("~/.openclaw/workspace/_photos"))
ARCHIVE = Path(os.path.expanduser("~/.openclaw/workspace/_archive"))   # I:\Photos, the RAWs
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
POSE = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")
W, H, FPS = 1080, 1920, 30
CUT = 0.001          # xfade duration that reads as a hard cut


def ffmpeg() -> str:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found (expected via static_ffmpeg in ~/openclaw-venv)")
    return exe


def locate_crop(full_path: Path, crop_path: Path):
    """Where does `crop` sit inside `full`? -> (x, y, w, h, score)"""
    full = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if full is None or crop is None:
        return None
    best = None
    for s in np.arange(0.30, 1.02, 0.01):
        t = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
        if t.shape[0] > full.shape[0] or t.shape[1] > full.shape[1]:
            continue
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(full, t, cv2.TM_CCOEFF_NORMED))
        if best is None or mx > best[-1]:
            best = (loc[0], loc[1], t.shape[1], t.shape[0], mx)
    return best


def tighten(rect, frac):
    """Shrink a rect toward its top edge, keeping 9:16. Burst frames get a
    closer crop than the hero: punchier at flash speed, and it keeps the
    frames SFW at the top of the body where the hero crop may not be."""
    x, y, w, h = rect
    bh = int(h * frac)
    bw = int(bh * w / h)
    return (x + (w - bw) // 2, y, bw, bh)


def raw_path(session: str, stem: str):
    for ext in (".CR3", ".CR2", ".cr3", ".cr2"):
        for d in ARCHIVE.glob(f"*/{session}*"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def load_raw(path: Path, export_size):
    """Decode a CR3/CR2 and resize to the export's pixel dimensions, so the crop
    rect measured on the export applies unchanged. Lightroom exports these
    uncropped, so this is a pure resize -- verified by aspect ratio."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
    return Image.fromarray(rgb).resize(export_size, Image.LANCZOS)


def chest_lines(images):
    """Where does the nipple line sit, as a fraction of frame height, across all
    the given frames? Estimated from pose landmarks: roughly a third of the way
    from the shoulder line down to the hips. Returns (lo, hi) over the set, so a
    single band can be placed once and cover every frame -- a band that jumps
    per frame reads as redaction; one that holds still reads as a title."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=POSE), num_poses=1))
    ys = []
    for im in images:
        ys.append(None)
        ys.pop()
        a = np.array(im.convert("RGB"))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
        if not res.pose_landmarks:
            continue
        p = res.pose_landmarks[0]
        sh = (p[11].y + p[12].y) / 2
        hp = (p[23].y + p[24].y) / 2
        ys.append(sh + (hp - sh) * 0.33)
    del det
    return ys


def lace_hem(im):
    """Lowest row of the dark garment across the torso, as a fraction of height.
    Measured, not extrapolated -- pose landmarks put the chest line several
    percent too high to censor against."""
    a = np.asarray(im.convert("L"), dtype=float) / 255
    H, W = a.shape
    c = a[:, int(W * 0.20):int(W * 0.80)]
    rows = np.where((c < 0.28).mean(axis=1) > 0.12)[0]
    rows = rows[(rows > H * 0.25) & (rows < H * 0.85)]
    return rows.max() / H if len(rows) else None


def censor(im, top, bot, px=26):
    """Pixelate a horizontal strip. Quieter than a black bar, and it can be made
    generously tall without shouting -- which is what makes it reliable when the
    thing being covered moves between frames."""
    H = im.height
    y0, y1 = max(0, int(H * top)), min(H, int(H * bot))
    if y1 <= y0:
        return im
    strip = im.crop((0, y0, im.width, y1))
    small = strip.resize((max(1, im.width // px), max(1, (y1 - y0) // px)),
                         Image.BILINEAR)
    im.paste(small.resize(strip.size, Image.NEAREST), (0, y0))
    return im


def title_band(im, text, y_mid, h_frac, bg, fg, cap=0.42):
    """Draw a full-width band with centred text. Doubles as a censor bar: it is
    drawn at the same place on every burst frame, so it reads as a title
    treatment rather than as three frames that got redacted."""
    d = ImageDraw.Draw(im)
    bh = int(im.height * h_frac)
    y = max(0, min(im.height - bh, int(im.height * y_mid - bh / 2)))
    d.rectangle([0, y, im.width, y + bh], fill=bg)

    def fit(txt, path, cap):
        size = int(bh * cap)
        while size > 8:
            f = ImageFont.truetype(path, size)
            l, _, r, _ = d.textbbox((0, 0), txt, font=f)
            if r - l <= im.width * 0.88:
                return f
            size -= 2
        return ImageFont.truetype(path, 8)

    spaced = " ".join(text)          # letterspaced: reads as a label, not a shout
    d.text((im.width / 2, y + bh / 2), spaced, font=fit(spaced, FONT_R, cap),
           fill=fg, anchor="mm")
    return im


def variant_dir(model: str, stem: str, variant: str):
    for d in PHOTOS.glob(f"{model}*/{variant}"):
        for ext in (".jpg", ".JPG", ".jpeg"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def prep(src, rect, out: Path, band=None, cen=None):
    im = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
    im = im.convert("RGB")
    if rect:
        x, y, w, h = rect
        im = im.crop((x, y, x + w, y + h))
    sc = max(W / im.width, H / im.height)
    im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((l, t, l + W, t + H))
    if cen:
        im = censor(im, *cen)
    if band:
        im = title_band(im, *band)
    im.save(out, quality=95)
    return out


def build(segments, outfile, zoom):
    """segments: [(image_path, hold_seconds, xfade_into_next, transition_name)].
    The clip cross-fades from the last segment back into the first."""
    exe = ffmpeg()
    args = [exe, "-y"]
    for path, hold, xf, _ in segments:
        args += ["-loop", "1", "-t", f"{hold + xf + 0.05:.3f}", "-i", str(path)]

    fc = [f"[{i}:v]scale={W}:{H},setsar=1,fps={FPS}[v{i}]" for i in range(len(segments))]
    prev, offset = "v0", segments[0][1]
    for i in range(1, len(segments)):
        xf, tr = segments[i - 1][2], segments[i - 1][3]
        fc.append(f"[{prev}][v{i}]xfade=transition={tr}:duration={max(xf, CUT):.3f}:"
                  f"offset={offset:.3f}[x{i}]")
        prev, offset = f"x{i}", offset + segments[i][1]
    if zoom:
        fc.append(f"[{prev}]zoompan=z='min(1+0.0004*in,1.10)':d=1:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}[out]")
        prev = "out"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{prev}]",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-preset", "slow", "-movflags", "+faststart", str(outfile)]
    r = subprocess.run(args, capture_output=True)
    if r.returncode:
        sys.exit(r.stderr.decode()[-2000:])
    return offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="a *__final.jpg from an output_ dir")
    ap.add_argument("--model", required=True, help="folder name under _photos/")
    ap.add_argument("--input-crop", help="the 9:16 crop fed to the stylizer "
                                         "(default: the sibling __src.jpg)")
    ap.add_argument("--burst", default="",
                    help="comma-separated frame stems flashed before the hero, in "
                         "order, e.g. BLD_5178,BLD_5180,BLD_5187,BLD_5202. They are "
                         "replayed in reverse on the way out.")
    ap.add_argument("--burst-range",
                    help="inclusive frame range, e.g. 5199-5205. Uses the RAWs, so "
                         "frames never exported from Lightroom are still available.")
    ap.add_argument("--session", default="",
                    help="session folder under _archive/ (substring match); "
                         "required with --burst-range")
    ap.add_argument("--burst-hold", type=float, default=0.13)
    ap.add_argument("--title", help="text for the slim band drawn across the burst "
                                    "frames; it also censors")
    ap.add_argument("--title-y", type=float,
                    help="band centre, 0-1. Default: tracks each frame's chest line, "
                         "so the band can stay slim.")
    ap.add_argument("--title-h", type=float, default=0.075, help="band height, 0-1")
    ap.add_argument("--title-cap", type=float, default=0.40,
                    help="text height as a fraction of band height")
    ap.add_argument("--censor", action="store_true",
                    help="pixelate the chest strip on burst frames, tracked per "
                         "frame off the garment hem. Use when the burst run is not "
                         "SFW at the hero crop; the title band is then free to sit "
                         "wherever it looks best.")
    ap.add_argument("--censor-span", default="0.20,0.01",
                    help="strip runs from hem-A to hem-B (fractions of height)")
    ap.add_argument("--title-bg", default="#000000")
    ap.add_argument("--title-fg", default="#ffffff")
    ap.add_argument("--holds", default="0.45,0.9,2.6",
                    help="hero holds: unprocessed,processed,stylized")
    ap.add_argument("--xf", type=float, default=0.35, help="hero transition seconds")
    ap.add_argument("--transition", default="fade")
    ap.add_argument("--out-transition", default="fadefast",
                    help="transition used on the way back out")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    final = Path(a.final)
    crop = Path(a.input_crop) if a.input_crop else final.with_name(
        re.sub(r"__final\.jpg$", "__src.jpg", final.name))
    m = re.search(r"(BLD_\d+|IMG_\d+|[A-Z0-9]{2,4}_\d+)", final.name)
    if not m:
        sys.exit(f"cannot read a frame number out of {final.name}")
    stem = m.group(1)

    processed = variant_dir(a.model, stem, "Processed")
    unproc = variant_dir(a.model, stem, "Unprocessed")
    if not processed or not unproc:
        sys.exit(f"need both Processed/ and Unprocessed/ {stem}.jpg under "
                 f"{PHOTOS}/{a.model}*")

    rect = None
    if crop.exists():
        loc = locate_crop(processed, crop)
        if loc and loc[-1] > 0.90:
            rect = loc[:4]
            print(f"crop rect {rect} (match {loc[-1]:.3f})")
        else:
            print("WARN: crop not located — centre cover-fit")
    else:
        print(f"WARN: no reference crop at {crop.name} — centre cover-fit")

    export_size = Image.open(processed).size
    prefix = re.match(r"([A-Za-z_]+)", stem).group(1)

    if a.burst_range:
        if not a.session:
            sys.exit("--burst-range needs --session (folder under _archive/)")
        lo, hi = (int(v) for v in a.burst_range.split("-"))
        burst, missing = [], []
        for n in range(lo, hi + 1):
            st = f"{prefix}{n}"
            (burst if raw_path(a.session, st) else missing).append(st)
        if missing:
            sys.exit(f"no RAW for: {', '.join(missing)}")
        loader = lambda st: load_raw(raw_path(a.session, st), export_size)
    else:
        burst = [x.strip() for x in a.burst.split(",") if x.strip()]
        missing = [x for x in burst if not variant_dir(a.model, x, "Unprocessed")]
        if missing:
            sys.exit(f"burst frames not found in Unprocessed/: {', '.join(missing)}")
        loader = lambda st: variant_dir(a.model, st, "Unprocessed")
    print(f"burst: {len(burst)} frames {burst[0]}..{burst[-1]}"
          + (" (from RAW)" if a.burst_range else ""))

    hu, hp, hs = (float(x) for x in a.holds.split(","))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        segs = []

        # render burst frames once, unbanded, so the band can be measured on them
        plain = {st: prep(loader(st), rect, td / f"b{i}.jpg")
                 for i, st in enumerate(burst)}

        stems = list(plain)
        cens = {}
        if a.censor:
            ca, cb = (float(v) for v in a.censor_span.split(","))
            for st in stems:
                hem = lace_hem(Image.open(plain[st]))
                if hem is None:
                    sys.exit(f"no garment hem found in {st}; cannot censor safely")
                cens[st] = (hem - ca, hem - cb)
            lo = min(v[0] for v in cens.values()); hi = max(v[1] for v in cens.values())
            print(f"censor strip tracks the hem, spanning {lo:.3f}..{hi:.3f} "
                  f"across the burst")

        if a.title:
            if a.title_y is not None:
                ys = [a.title_y] * len(stems)
            else:
                ys = chest_lines([Image.open(plain[st]) for st in stems])
                if len(ys) != len(stems):
                    sys.exit("pose not found in every burst frame; pass --title-y")
                print(f"chest line {min(ys):.3f}..{max(ys):.3f} "
                      f"(band tracks it; drifts {100*(max(ys)-min(ys)):.1f}% "
                      f"of frame height across the burst)")
            cache = {st: prep(Image.open(plain[st]), None, td / f"t{i}.jpg",
                              (a.title, ys[i], a.title_h, a.title_bg, a.title_fg,
                               a.title_cap), cens.get(st))
                     for i, st in enumerate(stems)}
        elif cens:
            cache = {st: prep(Image.open(plain[st]), None, td / f"t{i}.jpg",
                              None, cens[st]) for i, st in enumerate(stems)}
        else:
            cache = plain
        for st in burst:                                   # burst in, hard cuts
            segs.append((cache[st], a.burst_hold, CUT, "fade"))
        pu = prep(unproc, rect, td / "u.jpg")
        pp = prep(processed, rect, td / "p.jpg")
        pf = prep(final, None, td / "f.jpg")
        segs.append((pu, hu, a.xf, a.transition))
        segs.append((pp, hp, a.xf, a.transition))
        segs.append((pf, hs, a.xf, a.out_transition))      # payoff, then back out
        segs.append((pp, hp * 0.35, a.xf * 0.7, a.out_transition))
        segs.append((pu, hu * 0.6, CUT, "fade"))
        for st in reversed(burst):                         # burst out, faster
            segs.append((cache[st], a.burst_hold * 0.75, CUT, "fade"))
        # last segment cross-fades back into segs[0]; give it a real duration
        segs[-1] = (segs[-1][0], segs[-1][1], CUT, "fade")
        dur = build(segs, a.out, not a.no_zoom)

    print(f"{a.out}  {len(segs)} segments  ~{dur:.1f}s  loop-closed")


if __name__ == "__main__":
    main()
