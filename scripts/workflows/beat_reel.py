#!/usr/bin/env python3
"""PoC: the transform reel, cut to music instead of to arbitrary hold values.

Everything visual is reused from make_transform_reel.py. What is new here is
that no duration is chosen by hand: the structure is written in eighth notes,
and the track's own tempo turns those into seconds. Every cut lands on the
grid, and the payoff lands on a downbeat.

Motion on the payoff beat is the zoom-reveal idea from
reactvideoeditor/remotion-templates (MIT) -- scale 2 -> 1 over the beat.
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_transform_reel import (W, H, FPS, ffmpeg, locate_crop, load_raw, raw_path,
                                 variant_dir, prep, censor, lace_hem, title_band)
from PIL import Image

# structure in eighth notes. one bar = 8.
PLAN = dict(burst_in=1, unproc=4, proc=4, styled=16, proc_out=2, unproc_out=2,
            burst_out=1)


def analyse(track, start, dur):
    import librosa, numpy as np
    y, sr = librosa.load(track, sr=22050, offset=start, duration=dur + 10)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    return tempo, float(onset.std() / max(onset.mean(), 1e-6))


def best_start(track, want, hop=2.0):
    """Pick the loudest sustained window of the track, so the reel sits on the
    part with the most going on rather than on an intro."""
    import librosa, numpy as np
    y, sr = librosa.load(track, sr=22050)
    o = librosa.onset.onset_strength(y=y, sr=sr)
    t = librosa.frames_to_time(np.arange(len(o)), sr=sr)
    n = int(want / (t[1] - t[0]))
    if n >= len(o):
        return 0.0
    sums = np.convolve(o, np.ones(n), mode="valid")
    return float(t[int(np.argmax(sums))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--burst-range", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--music-start", type=float,
                    help="seconds into the track (default: its busiest window)")
    ap.add_argument("--title", default="THE TRANSITION")
    ap.add_argument("--title-y", type=float, default=0.90)
    ap.add_argument("--censor", action="store_true")
    ap.add_argument("--zoom-reveal", type=float, default=1.35,
                    help="payoff beat zooms from this scale down to 1.0")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    final = Path(a.final)
    crop = final.with_name(final.name.replace("__final.jpg", "__src.jpg"))
    import re
    stem = re.search(r"(BLD_\d+|IMG_\d+)", final.name).group(1)
    prefix = re.match(r"([A-Za-z_]+)", stem).group(1)

    processed = variant_dir(a.model, stem, "Processed")
    unproc = variant_dir(a.model, stem, "Unprocessed")
    rect = None
    if crop.exists():
        loc = locate_crop(processed, crop)
        if loc and loc[-1] > 0.90:
            rect = loc[:4]

    lo, hi = (int(v) for v in a.burst_range.split("-"))
    burst = [f"{prefix}{n}" for n in range(lo, hi + 1)]
    n_burst = len(burst)

    plan = dict(PLAN)
    eighths = (plan["burst_in"] * n_burst + plan["unproc"] + plan["proc"]
               + plan["styled"] + plan["proc_out"] + plan["unproc_out"]
               + plan["burst_out"] * n_burst)
    # pad the payoff so the whole reel is a whole number of bars -- otherwise the
    # loop restarts mid-bar and fights the track every time round
    pad = (-eighths) % 8
    plan["styled"] += pad
    eighths += pad

    tempo, punch = analyse(a.music, 0, 60)
    eighth = 60.0 / tempo / 2
    total = eighths * eighth
    start = a.music_start if a.music_start is not None else best_start(a.music, total)
    print(f"{Path(a.music).name}: {tempo:.1f} BPM, onset variance {punch:.2f}")
    print(f"  eighth note = {eighth:.3f}s, bar = {eighth*8:.3f}s")
    print(f"  structure = {eighths} eighths = {eighths//8} bars exactly = {total:.2f}s"
          f"  (payoff padded by {pad} eighths to close the bar)")
    print(f"  music from {start:.1f}s (busiest window)")

    export_size = Image.open(processed).size
    exe = ffmpeg()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        imgs, cens = {}, {}
        for i, st in enumerate(burst):
            src = load_raw(raw_path(a.session, st), export_size)
            p = prep(src, rect, td / f"b{i}.jpg")
            if a.censor:
                hem = lace_hem(Image.open(p))
                cens[st] = (hem - 0.20, hem - 0.01)
            imgs[st] = p
        for st in burst:                                   # censor + title band
            im = Image.open(imgs[st]).convert("RGB")
            if st in cens:
                im = censor(im, *cens[st])
            im = title_band(im, a.title, a.title_y, 0.075, "#000000", "#ffffff", 0.40)
            im.save(imgs[st], quality=95)

        pu = prep(unproc, rect, td / "u.jpg")
        pp = prep(processed, rect, td / "p.jpg")
        pf = prep(final, None, td / "f.jpg")

        # segments: (image, eighth-note count, zoom_from)
        segs = [(imgs[s], plan["burst_in"], 1.0) for s in burst]
        segs += [(pu, plan["unproc"], 1.0), (pp, plan["proc"], 1.0),
                 (pf, plan["styled"], a.zoom_reveal),
                 (pp, plan["proc_out"], 1.0), (pu, plan["unproc_out"], 1.0)]
        segs += [(imgs[s], plan["burst_out"], 1.0) for s in reversed(burst)]

        args = [exe, "-y"]
        for path, n, _ in segs:
            args += ["-loop", "1", "-t", f"{n * eighth + 0.05:.3f}", "-i", str(path)]
        args += ["-ss", f"{start:.3f}", "-t", f"{total:.3f}", "-i", str(a.music)]

        fc = []
        for i, (_, n, z) in enumerate(segs):
            frames = max(1, round(n * eighth * FPS))
            if z > 1.0:      # zoom-reveal, from remotion-templates/image-zoom-reveal
                fc.append(f"[{i}:v]scale={int(W*2)}:{int(H*2)},setsar=1,fps={FPS},"
                          f"zoompan=z='max({z}-({z}-1)*on/{frames},1)':d=1:"
                          f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                          f"s={W}x{H}:fps={FPS}[v{i}]")
            else:
                fc.append(f"[{i}:v]scale={W}:{H},setsar=1,fps={FPS}[v{i}]")
        prev, off = "v0", segs[0][1] * eighth
        for i in range(1, len(segs)):
            fc.append(f"[{prev}][v{i}]xfade=transition=fade:duration=0.001:"
                      f"offset={off:.4f}[x{i}]")
            prev, off = f"x{i}", off + segs[i][1] * eighth
        fc.append(f"[{len(segs)}:a]afade=t=in:d=0.15,"
                  f"afade=t=out:st={max(0,total-0.4):.3f}:d=0.4[a]")

        args += ["-filter_complex", ";".join(fc), "-map", f"[{prev}]", "-map", "[a]",
                 "-t", f"{total:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "18", "-preset", "slow", "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", str(a.out)]
        r = subprocess.run(args, capture_output=True)
        if r.returncode:
            sys.exit(r.stderr.decode()[-2500:])
    print(f"{a.out}  {len(segs)} cuts, all on the eighth-note grid")


if __name__ == "__main__":
    main()
