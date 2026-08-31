#!/usr/bin/env python3
"""Format D: behind-the-scenes clip cut to the finished photo.

Which seconds to use is derived, not asked for. The catalog knows exactly when the
final frame was shot, and the BTS clips carry their start time in the filename
(VID20250609113155) even when the container metadata is missing — so the interesting
footage is the couple of minutes leading up to that shutter press. Within that window
segments are scored on motion and sharpness, and 2-3 of them are stitched to 5-8s.

The photographer must not appear. Faces are counted per sampled frame; anything with
more than one face is dropped, and the whole candidate is flagged for confirmation.

  ig_bts.py --session "2025-06-09 Elly*" --final BLD_7071E.tif
"""
import argparse, re, shutil, subprocess, sys, sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ig_reel as R                                   # helpers, paths, render_video
import ig_meta

LOOKBACK_S = 180          # how far before the shutter press to consider
SEG_S = 2.5               # length of one stitched segment
WANT_S = 7.0              # total BTS length to aim for


def clip_start(path):
    """Start time from the filename, falling back to the container's creation time."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})", path.name)
    if m:
        return datetime(*[int(x) for x in m.groups()])
    try:
        out = subprocess.run([R.ffprobe(), "-v", "error",
                              "-show_entries", "format_tags=creation_time",
                              "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True).stdout.strip()
        return datetime.fromisoformat(out.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def duration(path):
    out = subprocess.run([R.ffprobe(), "-v", "error",
                          "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def shot_time(session_dir, stem):
    con = sqlite3.connect(str(R.CATALOG))
    row = con.execute(
        """SELECT p.taken_at FROM photos p JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.filename LIKE ?""",
        (f"%{session_dir.name}%", f"{stem}.%")).fetchone()
    con.close()
    if not row or not row[0]:
        return None
    return datetime.fromisoformat(row[0]).replace(tzinfo=None)


def video_dir(session_dir):
    for d in session_dir.iterdir():
        if d.is_dir() and re.search(r"video|bts|from my phone", d.name, re.I):
            return d
    return None


def score_segments(clip, t0, t1, step=1.0):
    """Rate each candidate start time on motion and sharpness.

    Still, blurry footage makes a dull cut; this prefers moments where something is
    actually happening and the frame is in focus.
    """
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out, prev = [], None
    t = t0
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, 180))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(grey, cv2.CV_64F).var()
        motion = float(np.abs(grey.astype(np.int16) - prev).mean()) if prev is not None else 0.0
        prev = grey.astype(np.int16)
        out.append({"t": t, "sharp": sharp, "motion": motion})
        t += step
    cap.release()
    if not out:
        return []
    ms = max(s["sharp"] for s in out) or 1
    mm = max(s["motion"] for s in out) or 1
    for s in out:
        s["score"] = 0.6 * (s["sharp"] / ms) + 0.4 * (s["motion"] / mm)
    return out


def faces_in_segment(clip, start, dur, samples=3):
    """Highest face count seen across the segment. >1 means someone else is in shot."""
    cap = cv2.VideoCapture(str(clip))
    worst = 0
    tmp = HERE / "_bts_probe.jpg"
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + dur * i / max(samples - 1, 1)) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(tmp), frame)
        try:
            worst = max(worst, ig_meta.face_count(tmp))
        except Exception:
            pass
    cap.release()
    tmp.unlink(missing_ok=True)
    return worst


def pick_segments(clip, shot_at, clip_started, want=WANT_S, seg=SEG_S):
    """2-3 well-spaced, in-focus, single-person segments from before the shutter press."""
    dur = duration(clip)
    offset = (shot_at - clip_started).total_seconds()
    hi = min(offset, dur - seg)
    lo = max(0.0, hi - LOOKBACK_S)
    if hi <= lo:
        return []
    scored = sorted(score_segments(clip, lo, hi), key=lambda s: -s["score"])
    picked, flagged = [], False
    for s in scored:
        if sum(seg for _ in picked) >= want:
            break
        if any(abs(s["t"] - p) < seg * 1.5 for p in picked):
            continue                                   # keep the cuts apart
        n = faces_in_segment(clip, s["t"], seg)
        if n > 1:
            flagged = True
            continue                                   # someone else — likely me
        picked.append(s["t"])
    return sorted(picked)[:3], flagged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--final", required=True)
    p.add_argument("--want", type=float, default=WANT_S)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-root", default=str(R.OUT_ROOT))
    args = p.parse_args()

    hits = sorted(R.ARCHIVE.glob(f"*/{args.session}"))
    if not hits:
        sys.exit(f"no session matching {args.session!r}")
    sd = hits[0]
    model = R.model_from_session(sd.name)
    ok, verdict = R.consent_ok(model)
    print(f"session : {sd.name}\nmodel   : {model}\nconsent : {verdict}")
    if not ok:
        sys.exit("consent gate: stopping")

    stem = re.match(r"(BLD_\d+)", Path(args.final).stem).group(1)
    shot_at = shot_time(sd, stem)
    vd = video_dir(sd)
    if not shot_at or not vd:
        sys.exit(f"need both a shot time ({shot_at}) and a video folder ({vd})")

    clips = sorted(p for p in vd.iterdir() if p.suffix.lower() in (".mp4", ".mov"))
    best = None
    for c in clips:
        st = clip_start(c)
        if not st:
            continue
        gap = (shot_at - st).total_seconds()
        if 0 <= gap <= LOOKBACK_S + duration(c):
            if best is None or gap < best[1]:
                best = (c, gap, st)
    if not best:
        sys.exit(f"no clip covers the minutes before {stem} was shot ({shot_at})")
    clip, gap, started = best
    print(f"photo   : {stem} at {shot_at}\nclip    : {clip.name} started {started} "
          f"({gap:.0f}s before the shutter)")

    segs, flagged = pick_segments(clip, shot_at, started, args.want)
    print(f"segments: {[round(s,1) for s in segs]} "
          f"({len(segs)} x {SEG_S}s){' — some rejected for extra faces' if flagged else ''}")
    if args.dry_run or not segs:
        return

    stamp = datetime.now().strftime("%Y-%m-%d")
    tag = f"{stamp}_{re.sub(r'[^A-Za-z0-9]+','',model)}_bts2shot_{stem}"
    out_dir = Path(args.out_root) / tag
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    exe = R.ffmpeg()

    parts = []
    for i, t in enumerate(segs):
        seg_path = out_dir / "frames" / f"seg{i}.mp4"
        subprocess.run([exe, "-y", "-ss", f"{t}", "-t", f"{SEG_S}", "-i", str(clip),
                        "-vf", f"scale={R.W}:{R.H}:force_original_aspect_ratio=increase,"
                               f"crop={R.W}:{R.H},fps={R.FPS}",
                        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(seg_path)], capture_output=True)
        parts.append(seg_path)

    finals = R.find_finals(sd)
    final_path = finals.get(stem, sd / args.final)
    from PIL import Image
    still = R.to_reel(np.asarray(Image.open(final_path).convert("RGB")))
    still.save(out_dir / "frames" / f"{stem}_final.png")
    still.save(out_dir / "frames" / f"{stem}_final.jpg", quality=95)
    R.text_layer(["this is what it was for"], 220, 64).save(out_dir / "frames" / "t_final.png")

    concat = out_dir / "frames" / "segs.txt"
    concat.write_text("".join(f"file '{p}'\n" for p in parts))
    bts = out_dir / "frames" / "bts.mp4"
    subprocess.run([exe, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", str(bts)], capture_output=True)

    mp4 = out_dir / f"{tag}.mp4"
    stitched = out_dir / "frames" / "final_seg.mp4"
    subprocess.run([exe, "-y", "-loop", "1", "-framerate", str(R.FPS), "-t", "4",
                    "-i", str(out_dir / "frames" / f"{stem}_final.png"),
                    "-loop", "1", "-framerate", str(R.FPS), "-t", "4",
                    "-i", str(out_dir / "frames" / "t_final.png"),
                    "-filter_complex",
                    f"[0:v]fps={R.FPS},scale={R.W}:{R.H}:flags=lanczos[b];"
                    f"[1:v]format=rgba,scale={R.W}:{R.H}:flags=lanczos,"
                    f"fade=t=in:st=0.6:d=0.5:alpha=1[t];[b][t]overlay=format=auto[v]",
                    "-map", "[v]", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(stitched)], capture_output=True)
    allcat = out_dir / "frames" / "all.txt"
    allcat.write_text(f"file '{bts}'\nfile '{stitched}'\n")
    r = subprocess.run([exe, "-y", "-f", "concat", "-safe", "0", "-i", str(allcat),
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr.strip()[-500:]); sys.exit("ffmpeg failed")

    ratio, sfw_note = ig_meta.sfw_flags(out_dir / "frames" / f"{stem}_final.jpg")
    meta = ig_meta.caption_and_hooks(model, sd.name, "D", R.session_count(sd))
    flags = [sfw_note, "BTS: confirm the photographer is not in shot"]
    if flagged:
        flags.append("some segments rejected for extra faces")
    R.write_txt(out_dir / f"{tag}.txt", model, sd, "D",
                [str(clip), str(final_path)], "see DB",
                "this is what it was for", meta["hooks"], meta["caption"],
                meta["keywords"], flags)
    ig_meta.write_sidecar(mp4, {"model": model, "session_date": sd.name[:10], "format": "D",
                                "hooks": meta["hooks"], "caption": meta["caption"],
                                "skin_ratio": round(ratio, 3), "sfw": sfw_note,
                                "clip": clip.name, "segments": [round(s, 1) for s in segs],
                                "status": "candidate"})
    ig_meta.write_log({"date": stamp, "model": model, "session": sd.name, "format": "D",
                       "set": stem, "files": mp4.name, "skin_ratio": round(ratio, 3),
                       "faces": "", "status": "candidate"})
    print(f"  → {mp4}")


if __name__ == "__main__":
    main()
