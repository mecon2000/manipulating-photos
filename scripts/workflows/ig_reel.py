#!/usr/bin/env python3
"""Render Instagram reel candidates from the RAW archive. Reels v2.

Two formats, both silent 1080x1920 H.264 (music goes on in-app):

  A raw2final    the untouched RAW, then the edit you actually published
  B contactsheet a run of near-identical frames from one set, then the keeper

Nothing is posted. Every render also writes a .txt (caption, keywords, sources)
and a frames/ folder holding the exact stills used, so a candidate can be
rebuilt by hand in CapCut with nicer transitions.

  ig_reel.py --session "2025-06-09 Elly*" --format A --final BLD_7071E.tif
"""
import argparse, glob, json, os, re, subprocess, sys, shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARCHIVE = Path("~/.openclaw/workspace/_archive").expanduser()
OUT_ROOT = Path("~/.openclaw/workspace/shared/ig-reels").expanduser()
CATALOG = Path("~/gitrep/photo-catalogging/data/photo-catalog.db").expanduser()
CONSENT = Path("~/.claude/skills/candidates/consent.py").expanduser()
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
W, H, FPS = 1080, 1920, 30
SS = 2                     # stills are rendered at 2x; zoompan crops down into them,
                           # so the slow push keeps real detail instead of upscaling

# Folder names carrying a publish decision. Negative wins over positive, and only
# excludes its own contents — a session is not condemned by one bad folder.
NEG = re.compile(r"don'?t\s*publish|not\s*approved|do\s*not\s*publish|onlyfans|only ?fans", re.I)
POS = re.compile(r"approved|ok to publish|can publish", re.I)


def ffmpeg():
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def ffprobe():
    """Never derive this from the ffmpeg path by string replacement — the binary lives
    under .../static_ffmpeg/bin/, so replacing "ffmpeg" rewrites the directory too."""
    ffmpeg()
    return shutil.which("ffprobe")


def model_from_session(name):
    """'2025-06-09 Elly (Eleanora) at Yogev's house' -> 'Elly (Eleanora)'."""
    s = re.sub(r"^\d{4}-\d{2}-\d{2}\s*", "", name)
    return re.split(r"\s+at\s+|\s+,|\s+with\s+", s)[0].strip(" ,-")


def consent_ok(model):
    r = subprocess.run([sys.executable, str(CONSENT), "--check", model],
                       capture_output=True, text=True, cwd=str(CONSENT.parent))
    line = (r.stdout or "").strip().splitlines()
    verdict = line[-1] if line else ""
    return verdict.startswith("OK"), verdict


def develop(raw_path, mode="flat", half=False):
    """RAW -> RGB. 'camera' approximates the JPEG the camera would have made;
    'flat' is linear and unlifted, which is the point of the format."""
    import rawpy
    with rawpy.imread(str(raw_path)) as raw:
        if mode == "camera":
            return raw.postprocess(use_camera_wb=True, half_size=half)
        return raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                               gamma=(1, 1), output_bps=8, half_size=half)


def lift(img, gamma=1.0):
    if gamma == 1.0:
        return img
    x = np.clip(img.astype(np.float32) / 255.0, 0, 1) ** (1.0 / gamma)
    return (x * 255).astype(np.uint8)


def face_center(img):
    """Where to keep the crop window. Falls back to the frame centre."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import face_align as FA
        pts = FA.landmarks(np.asarray(img))
        if pts is not None:
            c = (pts["forehead"] + pts["chin"]) / 2
            return float(c[0]) / img.shape[1], float(c[1]) / img.shape[0]
    except Exception:
        pass
    return 0.5, 0.42


def to_reel(arr, gamma=1.0, ss=SS):
    """Fit any frame to (ss x) 1080x1920, biasing the crop toward the subject."""
    arr = lift(arr, gamma)
    fx, fy = face_center(arr)
    h, w = arr.shape[:2]
    tw, th = W * ss, H * ss
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
    x = int(np.clip(fx * nw - tw / 2, 0, max(nw - tw, 0)))
    y = int(np.clip(fy * nh - th / 2, 0, max(nh - th, 0)))
    return im.crop((x, y, x + tw, y + th))


def text_layer(lines, y, size, bold=True, ss=SS):
    """Transparent overlay so the text can fade independently of the picture."""
    layer = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size * ss)
    cy = y * ss
    for line in lines:
        tw = d.textbbox((0, 0), line, font=font)[2]
        x = (W * ss - tw) // 2
        d.text((x + 3 * ss, cy + 3 * ss), line, font=font, fill=(0, 0, 0, 170))
        d.text((x, cy), line, font=font, fill=(255, 255, 255, 255))
        cy += int(size * ss * 1.25)
    return layer


def find_finals(session_dir, blue=None):
    """Finals are E-siblings, files under a processed/ folder, or Blue-labelled.

    Most edits never leave Lightroom, so an E suffix is the exception; treating it as
    the only signal would miss the majority of published frames. But a POSITIVE signal
    is required: without it the fallback accepted any jpg in the session, including
    "... - UNPROCESSED.jpg" watermarked exports, and offered them as finished work.
    """
    blue = blue if blue is not None else blue_labelled(session_dir)
    out = {}
    for p in session_dir.rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        rel = str(p.relative_to(session_dir))
        if NEG.search(rel):
            continue                                  # excludes its own folder only
        m = re.match(r"(BLD_\d+|[A-Za-z_]*\d+)", p.stem)
        if not m:
            continue
        stem = m.group(1)
        if "UNPROCESSED" in p.stem.upper():
            continue                                  # watermarked export, never a final
        edited = bool(re.search(r"E(-\d)?E?$", p.stem))
        in_processed = re.search(r"(^|/)processed", rel.lower()) is not None
        rank = (2 if edited else 0) + (1 if in_processed else 0) + (1 if stem in blue else 0)
        if rank == 0:
            continue                                  # no signal that this is finished
        if stem not in out or rank > out[stem][0]:
            out[stem] = (rank, p)
    return {k: v[1] for k, v in out.items()}


def blue_labelled(session_dir):
    import sqlite3
    con = sqlite3.connect(str(CATALOG))
    rows = con.execute(
        """SELECT p.filename FROM photos p JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.lr_color_label = 'Blue'""",
        (f"%{session_dir.name}%",)).fetchall()
    con.close()
    return {Path(r[0]).stem for r in rows}


def session_count(session_dir):
    import sqlite3
    con = sqlite3.connect(str(CATALOG))
    r = con.execute("SELECT photo_count FROM sessions WHERE folder_path LIKE ?",
                    (f"%{session_dir.name}%",)).fetchone()
    con.close()
    return r[0] if r else len(list(session_dir.glob("*.CR3")))


def set_frames(session_dir, stem, count=6):
    """Consecutive frames from the DB set that contains this one."""
    import sqlite3
    con = sqlite3.connect(str(CATALOG))
    row = con.execute(
        """SELECT p.set_id FROM photos p JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.filename LIKE ?""",
        (f"%{session_dir.name}%", f"{stem}.%")).fetchone()
    if not row:
        con.close()
        return []
    names = [r[0] for r in con.execute(
        "SELECT filename FROM photos WHERE set_id=? ORDER BY taken_at", (row[0],))]
    con.close()
    nums = sorted({int(re.search(r"(\d+)", n).group(1)) for n in names if re.search(r"\d", n)})
    target = int(re.search(r"(\d+)", stem).group(1))
    i = min(range(len(nums)), key=lambda k: abs(nums[k] - target)) if nums else 0
    lo = max(0, i - count + 1)
    picked = nums[lo:i + 1]
    return [session_dir / f"BLD_{n}.CR3" for n in picked
            if (session_dir / f"BLD_{n}.CR3").exists()]


def render_video(segments, out_path):
    """segments: list of (base_png, text_png|None, seconds, zoom_to, text_fade)."""
    exe = ffmpeg()
    if not exe:
        sys.exit("ffmpeg not found")
    cmd = [exe, "-y"]
    for base, text, secs, _z, _tf in segments:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{secs}", "-i", str(base)]
        if text:
            cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{secs}", "-i", str(text)]
    parts, labels, idx = [], [], 0
    for n, (base, text, secs, zoom, tf) in enumerate(segments):
        b = idx; idx += 1
        vid = f"[{b}:v]"
        if zoom and zoom > 1.0:
            frames = int(secs * FPS)
            # d=1 with the zoom driven by the OUTPUT frame counter. The obvious
            # d=<frames> emits that many frames PER input frame, and since the input
            # is a looped still that turns a 5s segment into ~750s.
            parts.append(f"{vid}fps={FPS},zoompan="
                         f"z='min(1+({zoom - 1:.4f}*on/{frames}),{zoom})'"
                         f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                         f":d=1:s={W}x{H}[z{n}]")
            vid = f"[z{n}]"
        else:
            parts.append(f"{vid}fps={FPS},scale={W}:{H}:flags=lanczos[z{n}]")
            vid = f"[z{n}]"
        if text:
            t = idx; idx += 1
            kind, st, dur = tf
            parts.append(f"[{t}:v]format=rgba,scale={W}:{H}:flags=lanczos,"
                         f"fade=t={kind}:st={st}:d={dur}:alpha=1[t{n}]")
            parts.append(f"{vid}[t{n}]overlay=format=auto[s{n}]")
        else:
            parts.append(f"{vid}null[s{n}]")
        labels.append(f"[s{n}]")
    parts.append("".join(labels) + f"concat=n={len(segments)}:v=1:a=0[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr.strip()[-700:])
        sys.exit("ffmpeg failed")
    return out_path


def write_txt(path, model, session_dir, fmt, sources, set_id, hook, alts, caption, kw, flags):
    path.write_text(
        f"CAPTION\n{caption}\n\nKEYWORDS\n{', '.join(kw)}\n\n"
        f"HOOK USED\n{hook}\n\nALTERNATIVE HOOKS\n" + "\n".join(f"- {a}" for a in alts) +
        f"\n\nPOST AS: Trial Reel\nMUSIC: add in-app\n"
        f"FORMAT: {fmt}\nMODEL: {model}\nSESSION: {session_dir}\nSET: {set_id}\n"
        f"SOURCE FILES:\n" + "\n".join(f"  {s}" for s in sources) +
        (f"\n\nFLAGS: {', '.join(flags)}\n" if flags else "\n"), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True, help="folder name or glob under _archive")
    p.add_argument("--format", choices=["A", "B"], required=True)
    p.add_argument("--final", help="final filename; otherwise the best SFW-looking one")
    p.add_argument("--raw-mode", choices=["flat", "camera", "both"], default="both")
    p.add_argument("--frames", type=int, default=6, help="format B: frames before the keeper")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    hits = sorted(ARCHIVE.glob(f"*/{args.session}"))
    if not hits:
        sys.exit(f"no session matching {args.session!r} under {ARCHIVE}")
    sd = hits[0]
    model = model_from_session(sd.name)
    ok, verdict = consent_ok(model)
    print(f"session : {sd.name}\nmodel   : {model}\nconsent : {verdict}")
    if not ok:
        sys.exit("consent gate: not clear to publish — stopping")

    finals = find_finals(sd)
    blue = blue_labelled(sd)
    total = session_count(sd)
    print(f"finals  : {len(finals)} candidates, {len(blue)} blue-labelled, session has {total} frames")
    if args.final:
        stem = re.match(r"(BLD_\d+)", Path(args.final).stem).group(1)
        final_path = finals.get(stem) or (sd / args.final)
    else:
        stem, final_path = sorted(finals.items())[0]
    raw_path = sd / f"{stem}.CR3"
    if not raw_path.exists():
        sys.exit(f"no RAW for {stem}")
    print(f"final   : {final_path.name}\nraw     : {raw_path.name}")
    if args.dry_run:
        return

    stamp = datetime.now().strftime("%Y-%m-%d")
    tag = f"{stamp}_{re.sub(r'[^A-Za-z0-9]+','',model)}_{'raw2final' if args.format=='A' else 'contactsheet'}_{stem}"
    out_dir = Path(args.out_root) / tag
    # Visible: the scanner groups by top-level directory, so the mp4 and these stills
    # are one card — the video plays first, the frames step behind it.
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    final_img = to_reel(np.asarray(Image.open(final_path).convert("RGB")))
    final_img.save(frames_dir / f"{stem}_final.jpg", quality=95)

    modes = ["flat", "camera"] if args.raw_mode == "both" else [args.raw_mode]
    made = []
    for mode in modes:
        if args.format == "A":
            raw_img = to_reel(develop(raw_path, mode), gamma=1.25 if mode == "flat" else 1.0)
            raw_img.save(frames_dir / f"{stem}_raw_{mode}.jpg", quality=95)
            segs = [(frames_dir / f"{stem}_raw_{mode}.png", frames_dir / "t_raw.png", 1.6, None, ("out", 1.2, 0.4)),
                    (frames_dir / f"{stem}_final.png", frames_dir / "t_final.png", 5.0, 1.09, ("in", 2.0, 0.5))]
            raw_img.save(frames_dir / f"{stem}_raw_{mode}.png")
            final_img.save(frames_dir / f"{stem}_final.png")
            text_layer(["what the camera saw"], 220, 72).save(frames_dir / "t_raw.png")
            text_layer(["what I sent her"], 220, 72).save(frames_dir / "t_final.png")
            hook = "what the camera saw / what I sent her"
        else:
            raws = set_frames(sd, stem, args.frames)
            segs = []
            for i, rp in enumerate(raws):
                im = to_reel(develop(rp, mode), gamma=1.9 if mode == "flat" else 1.2)
                base = frames_dir / f"{rp.stem}_{mode}.png"
                im.save(base); im.save(frames_dir / f"{rp.stem}_{mode}.jpg", quality=93)
                tl = frames_dir / f"lab_{i}.png"
                if i == 0:
                    text_layer([f"{total} frames.", "One gets published."], 220, 72).save(tl)
                    segs.append((base, tl, 1.4, None, ("out", 1.0, 0.4)))
                else:
                    text_layer([f"frame {re.search(r'(\d+)', rp.stem).group(1)}"], 1720, 44, bold=False).save(tl)
                    segs.append((base, tl, 0.55, None, ("out", 0.45, 0.1)))
            final_img.save(frames_dir / f"{stem}_final.png")
            text_layer(["this one."], 220, 72).save(frames_dir / "t_final.png")
            segs.append((frames_dir / f"{stem}_final.png", frames_dir / "t_final.png",
                         4.5, 1.06, ("in", 1.5, 0.5)))
            hook = f"{total} frames. One gets published. / this one."
        mp4 = out_dir / f"{tag}{'' if len(modes) == 1 else '_' + mode}.mp4"
        render_video(segs, mp4)
        made.append(mp4)
        print(f"  → {mp4.name}")

    sources = [str(final_path)] + ([str(raw_path)] if args.format == "A"
                                   else [str(r) for r in set_frames(sd, stem, args.frames)])
    import ig_meta
    ratio, sfw_note = ig_meta.sfw_flags(frames_dir / f"{stem}_final.jpg")
    faces = ig_meta.face_count(frames_dir / f"{stem}_final.jpg")
    flags = [sfw_note] + ([f"{faces} faces detected — is anyone else in frame?"]
                          if faces > 1 else [])
    meta = ig_meta.caption_and_hooks(model, sd.name, args.format, total)
    write_txt(out_dir / f"{tag}.txt", model, sd, args.format, sources,
              "see DB", hook, meta["hooks"], meta["caption"], meta["keywords"], flags)
    for m in made:
        ig_meta.write_sidecar(m, {"model": model, "session_date": sd.name[:10],
                                  "format": args.format, "hooks": meta["hooks"],
                                  "caption": meta["caption"], "skin_ratio": round(ratio, 3),
                                  "faces": faces, "sfw": sfw_note,
                                  "caption_source": meta["source"], "status": "candidate"})
        ig_meta.write_log({"date": stamp, "model": model, "session": sd.name,
                           "format": args.format, "set": stem, "files": m.name,
                           "skin_ratio": round(ratio, 3), "faces": faces,
                           "status": "candidate"})
    for m in made:
        flat = Path(args.out_root) / "_flat"
        flat.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m, flat / m.name)
    print(f"\n{out_dir}")


if __name__ == "__main__":
    main()
