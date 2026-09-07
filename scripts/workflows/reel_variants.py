#!/usr/bin/env python3
"""Generate a spanning set of reel variants from one prepared group folder.

Deliberately NOT ten guesses at a good reel: ten points across a stated design,
so watching them locates which axis matters. Silent by design -- Instagram's
Edits app detects beats in the audio you choose and snaps clips to them, and
trending audio has to be added in-app anyway, so baking a track in only removes
options. Cuts sit on a documented BPM grid so any track near that tempo lands.
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_transform_reel import ffmpeg, W, H, FPS
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
import numpy as np

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CUT = 0.001


def load_group(d):
    d = Path(d)
    burst = sorted(d.glob("0*_burst_*.jpg"))
    g = dict(burst=burst,
             unedited=d / "10_unedited.jpg",
             edited=d / "20_edited.jpg",
             stylized=d / "30_stylized.jpg")
    if not g["edited"].exists():
        g["edited"] = None
    return g


# ---------- treatments -------------------------------------------------------

def grade(im, kind):
    if kind == "bw":
        return ImageOps.grayscale(im).convert("RGB")
    if kind == "contrast":
        return ImageEnhance.Contrast(im).enhance(1.45)
    if kind == "cool":
        r, g, b = im.split()
        return Image.merge("RGB", (r.point(lambda v: v * 0.92),
                                   g, b.point(lambda v: min(255, v * 1.12))))
    return im


def eye_band(im, frac=0.17, anchor=0.31):
    """A letterboxed strip through the eyes: the rest of the frame goes black.
    Cuts the composition down to the one thing that is aligned across every
    frame, which is what makes the flicker read as one face rather than many."""
    out = Image.new("RGB", im.size, "black")
    h = int(im.height * frac)
    y = int(im.height * anchor) - h // 2          # anchor is the eye line, centred
    out.paste(im.crop((0, y, im.width, y + h)), (0, y))
    return out


def sticker(im, text, y=0.055, size=0.030, bg="#ffffff", fg="#000000"):
    d = ImageDraw.Draw(im)
    fsz = max(12, int(im.height * size))
    f = ImageFont.truetype(FONT, fsz)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    pad = int(fsz * 0.42)
    x0 = (im.width - (r - l)) // 2 - pad
    y0 = int(im.height * y)
    d.rounded_rectangle([x0, y0, x0 + (r - l) + 2 * pad, y0 + (b - t) + 2 * pad],
                        radius=int(fsz * 0.35), fill=bg)
    d.text((im.width // 2, y0 + pad + (b - t) // 2), text, font=f, fill=fg,
           anchor="mm")
    return im


def prep(path, size, treat=None, label=None):
    im = Image.open(path).convert("RGB")
    sc = max(size[0] / im.width, size[1] / im.height)
    im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    l, t = (im.width - size[0]) // 2, (im.height - size[1]) // 2
    im = im.crop((l, t, l + size[0], t + size[1]))
    if treat == "bw" or treat == "contrast" or treat == "cool":
        im = grade(im, treat)
    elif treat == "eyes":
        im = eye_band(im)
    if label:
        im = sticker(im, label)
    return im


# ---------- timing -----------------------------------------------------------

def ramp(n, fast, slow):
    """Accelerating holds: a constant cut rate reads as a slideshow."""
    if n == 1:
        return [slow]
    return [slow + (fast - slow) * i / (n - 1) for i in range(n)]


def render(frames, holds, out, zoom_last=None):
    exe = ffmpeg()
    args = [exe, "-y"]
    for p, h in zip(frames, holds):
        args += ["-loop", "1", "-t", f"{h + 0.06:.3f}", "-i", str(p)]
    fc = []
    for i in range(len(frames)):
        if zoom_last is not None and i == zoom_last[0]:
            z, n = zoom_last[1], max(1, round(holds[i] * FPS))
            fc.append(f"[{i}:v]scale={W*2}:{H*2},setsar=1,fps={FPS},"
                      f"zoompan=z='max({z}-({z}-1)*on/{n},1)':d=1:"
                      f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}[v{i}]")
        else:
            fc.append(f"[{i}:v]scale={W}:{H},setsar=1,fps={FPS}[v{i}]")
    prev, off = "v0", holds[0]
    for i in range(1, len(frames)):
        fc.append(f"[{prev}][v{i}]xfade=transition=fade:duration={CUT}:"
                  f"offset={off:.4f}[x{i}]")
        prev, off = f"x{i}", off + holds[i]
    args += ["-filter_complex", ";".join(fc), "-map", f"[{prev}]",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
             "-preset", "medium", "-movflags", "+faststart", str(out)]
    r = subprocess.run(args, capture_output=True)
    if r.returncode:
        sys.exit(r.stderr.decode()[-1500:])
    return off


# ---------- the design -------------------------------------------------------
# Each entry varies ONE cluster of things from its neighbours so the comparison
# is informative. "beats" are eighth notes at the stated BPM.
# Round 2. Everything sits on the base that round 1 settled -- payoff-first,
# chronological burst -- and each pair varies ONE thing to an extreme, because
# round 1's subtle variations were unjudgeable watched cold.
ROUND2 = [
    dict(id="A1_fast",     order="payoff_first", density="sixteenth",
         treat=None,   payoff=30, label=None, note="burst at sixteenths - as fast as it goes"),
    dict(id="A2_slow",     order="payoff_first", density="quarter",
         treat=None,   payoff=14, label=None, note="burst at quarters - deliberately slow"),
    dict(id="B1_mono",     order="payoff_first", density="eighth",
         treat="bw",   payoff=14, label=None, note="burst in mono, payoff is the only colour"),
    dict(id="B2_eyes",     order="payoff_first", density="eighth",
         treat="eyes", payoff=14, label=None, note="burst as an eye-strip only"),
    dict(id="C1_hook",     order="payoff_first", density="eighth",
         treat=None,   payoff=14, label="how i made this",
         note="text hook on the opening frame"),
    dict(id="C2_nohook",   order="payoff_first", density="eighth",
         treat=None,   payoff=14, label=None, note="identical, no text"),
]

PAIRS = [("A1_fast", "A2_slow", "speed"),
         ("B1_mono", "B2_eyes", "treatment"),
         ("C1_hook", "C2_nohook", "text hook")]

VARIANTS = [
    dict(id="01_baseline",        order="burst_first",  density="eighth",
         treat=None,     payoff=16, label=None,
         note="what we have now: burst -> unedited -> edited -> stylized"),
    dict(id="02_payoff_first",    order="payoff_first", density="eighth",
         treat=None,     payoff=14, label=None,
         note="opens on the stylized frame, then collapses backwards"),
    dict(id="03_ramped",          order="burst_first",  density="ramp",
         treat=None,     payoff=16, label=None,
         note="cuts accelerate into the payoff instead of ticking"),
    dict(id="04_flicker",         order="flicker",      density="sixteenth",
         treat=None,     payoff=12, label=None,
         note="eyes locked, very fast cuts: one face morphing, not a slideshow"),
    dict(id="05_eyes_only",       order="burst_first",  density="eighth",
         treat="eyes",   payoff=16, label=None,
         note="burst reduced to a letterboxed strip through the eyes"),
    dict(id="06_bw_burst",        order="burst_first",  density="eighth",
         treat="bw",     payoff=16, label=None,
         note="burst in mono so the stylized frame is the only colour"),
    dict(id="07_no_burst",        order="no_burst",     density="quarter",
         treat=None,     payoff=16, label=None,
         note="control: does the burst earn its screen time at all?"),
    dict(id="08_sticker",         order="payoff_first", density="eighth",
         treat=None,     payoff=14, label="how i made this",
         note="payoff-first plus a text hook, read with sound off"),
    dict(id="09_shuffled",        order="shuffled",     density="ramp",
         treat="contrast", payoff=14, label=None,
         note="burst order scrambled: is chronology doing any work?"),
    dict(id="10_long_hold",       order="burst_first",  density="quarter",
         treat="cool",   payoff=32, label=None,
         note="slow, longer payoff: tests whether fast is actually better"),
]


def build_sequence(g, v, eighth):
    """-> (list of (path_or_image, hold_seconds), index_to_zoom)"""
    burst = list(g["burst"])
    mult = {"eighth": 1, "sixteenth": 0.5, "quarter": 2, "ramp": 1}[v["density"]]
    seq, zoom_at = [], None

    def unit(n=1):
        return n * eighth * mult

    chain = [p for p in (g["unedited"], g["edited"], g["stylized"]) if p]
    if v["order"] == "shuffled":
        rng = np.random.default_rng(7)
        burst = list(rng.permutation(burst))

    if v["order"] == "payoff_first":
        seq.append((g["stylized"], unit(4), "hero"))
        for p in reversed(chain[:-1]):
            seq.append((p, unit(2), "hero"))
        for p in reversed(burst):
            seq.append((p, unit(1), "burst"))
        seq.append((g["stylized"], unit(v["payoff"]), "payoff"))
    elif v["order"] == "no_burst":
        for p in chain[:-1]:
            seq.append((p, unit(4), "hero"))
        seq.append((g["stylized"], unit(v["payoff"]), "payoff"))
        for p in reversed(chain[:-1]):
            seq.append((p, unit(2), "hero"))
    elif v["order"] == "flicker":
        # two passes: one is under Instagram's 3s floor, and the morph effect
        # needs a beat or two before the eye actually reads it as one face
        for _ in range(2):
            for p in burst + chain[:-1] + list(reversed(burst)):
                seq.append((p, unit(1), "burst"))
        seq.append((g["stylized"], unit(v["payoff"]), "payoff"))
    else:                                              # burst_first
        holds = (ramp(len(burst), eighth * 0.75, eighth * 1.9)
                 if v["density"] == "ramp" else [unit(1)] * len(burst))
        for p, h in zip(burst, holds):
            seq.append((p, h, "burst"))
        for p in chain[:-1]:
            seq.append((p, unit(3), "hero"))
        seq.append((g["stylized"], unit(v["payoff"]), "payoff"))
        for p in reversed(chain[:-1]):
            seq.append((p, unit(1), "hero"))
        for p in reversed(burst):
            seq.append((p, unit(0.75), "burst"))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--bpm", type=float, default=161.5)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--only", help="comma-separated variant ids")
    ap.add_argument("--round2", action="store_true",
                    help="build the round-2 set and the side-by-side pairs")
    a = ap.parse_args()

    g = load_group(a.group)
    if not g["burst"]:
        sys.exit(f"no burst frames in {a.group}")
    eighth = 60.0 / a.bpm / 2
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_frames"; tmp.mkdir(exist_ok=True)
    only = set(a.only.split(",")) if a.only else None

    rows = []
    design = ROUND2 if a.round2 else VARIANTS
    for v in design:
        if only and v["id"] not in only:
            continue
        seq = build_sequence(g, v, eighth)
        frames, holds, zoom_at = [], [], None
        for i, (p, h, role) in enumerate(seq):
            treat = v["treat"] if role == "burst" else None
            label = v["label"] if i == 0 else None
            im = prep(p, (W, H), treat, label)
            fp = tmp / f"{v['id']}_{i:02d}.jpg"
            im.save(fp, quality=93)
            frames.append(fp); holds.append(h)
            if role == "payoff" and zoom_at is None:
                zoom_at = (i, 1.28)
        dur = render(frames, holds, out / f"{v['id']}.mp4", zoom_at)
        rows.append(dict(id=v["id"], cuts=len(frames), seconds=round(dur, 2),
                         order=v["order"], density=v["density"],
                         treat=v["treat"] or "-", note=v["note"]))
        print(f"{v['id']:16s} {len(frames):3d} cuts  {dur:5.2f}s  {v['note']}")

    if a.round2:
        exe = ffmpeg()
        for l, r, axis in PAIRS:
            lp, rp = out / f"{l}.mp4", out / f"{r}.mp4"
            if not (lp.exists() and rp.exists()):
                continue
            dst = out / f"PAIR_{axis.replace(' ', '_')}.mp4"
            # side by side on one vertical canvas, each labelled, both looping
            fc = (f"[0:v]scale=540:960,setsar=1[a];[1:v]scale=540:960,setsar=1[b];"
                  f"[a][b]hstack=inputs=2[s];"
                  f"color=black:1080x1920:d=1[bg];"
                  f"[bg][s]overlay=0:(H-h)/2:shortest=1,"
                  f"drawtext=text='{l}':x=270-tw/2:y=520:fontsize=34:fontcolor=white,"
                  f"drawtext=text='{r}':x=810-tw/2:y=520:fontsize=34:fontcolor=white[o]")
            rr = subprocess.run([exe, "-y", "-stream_loop", "6", "-i", str(lp),
                                 "-stream_loop", "6", "-i", str(rp),
                                 "-filter_complex", fc, "-map", "[o]", "-t", "16",
                                 "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-crf", "20", "-preset", "medium", str(dst)],
                                capture_output=True)
            print(f"pair [{axis}] {'ok' if rr.returncode==0 else rr.stderr.decode()[-300:]} -> {dst.name}")

    (out / "design.json").write_text(json.dumps(
        dict(bpm=a.bpm, eighth_note=round(eighth, 4), group=str(a.group),
             variants=rows), indent=1))
    print(f"\n{len(rows)} variants -> {out}")


if __name__ == "__main__":
    main()
