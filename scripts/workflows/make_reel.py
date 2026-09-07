#!/usr/bin/env python3
"""Build a reel: N frames from the SAME SET as a seed photo, all stylized identically.

The point of a reel is that the frames look like one continuous moment, so they must
come from one burst — same camera position, same light, same pose. Set membership is
decided by CAPTURE TIME, not by frame numbering or by visual similarity: a gap larger
than --gap seconds means the photographer moved and a new set began. Pose-similarity
ranking looks reasonable but happily pulls in frames from an hour earlier.

Timestamps come from the photo catalog, keyed on frame number — Lightroom strips EXIF
from exported JPEGs, so the JPEG itself cannot be asked when it was taken. The catalog
indexes the RAWs (BLD_9941.CR3), and the export (BLD_9941E.jpg) shares its stem.

  make_reel.py --seed PHOTO.jpg --style REF.jpg --name reel2 --prompt "..."

Outputs land in <shared>/faces-candidates/{input,output}_<Model>_<style>_<name>/.
"""
import argparse, os, re, sqlite3, subprocess, sys, shutil, glob
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_align as FA

CATALOG = Path("~/gitrep/photo-catalogging/data/photo-catalog.db").expanduser()
SHARED = Path("~/.openclaw/workspace/shared/faces-candidates").expanduser()
HERE = Path(__file__).resolve().parent
PY = os.environ.get("REEL_PYTHON", sys.executable)


def clean(name):
    """Filenames become part of every output name, so strip Lightroom noise."""
    return re.sub(r"[^A-Za-z0-9_.-]", "", name.replace(" - UNPROCESSED", "").replace(" ", ""))


def frame_no(path):
    m = re.search(r"(\d{3,})", Path(path).stem)
    return int(m.group(1)) if m else None


def shot_at(path):
    """Capture time from the export's own EXIF, or None.

    This module used to assert that Lightroom strips EXIF from exports and key
    everything on frame numbers instead. That is false for these files — the exports
    carry DateTimeOriginal — and the assumption caused the damage: IMG_ numbers repeat
    across cameras, so `IMG_7011` matched six different sessions equally well and the
    reel was built from whichever one sorted first.
    """
    try:
        ex = Image.open(path).getexif()
        raw = ex.get_ifd(0x8769).get(0x9003) or ex.get(0x0132)
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S") if raw else None
    except Exception:
        return None


def resolve_session(seed_path):
    """The catalog session a photo belongs to, keyed on FRAME NUMBER + CAPTURE TIME.

    Either key alone is ambiguous — numbers repeat across cameras, and many frames
    share a second. Together they are effectively unique. The match must also be
    unambiguous: if a second session fits nearly as well, this returns nothing rather
    than guessing, because guessing is what put strangers' frames in a reel.

    Returns (session_id, folder_path, note) or (None, None, reason).
    """
    n = frame_no(seed_path)
    t = shot_at(seed_path)
    if not CATALOG.is_file() or n is None:
        return None, None, "no catalog or unreadable frame number"
    con = sqlite3.connect(str(CATALOG))
    rows = con.execute(
        """SELECT s.id, s.folder_path, p.taken_at FROM photos p
           JOIN sessions s ON s.id = p.session_id
           WHERE p.filename LIKE ? AND p.taken_at IS NOT NULL""", (f"%{n}%",)).fetchall()
    con.close()
    cands = []
    for sid, folder, taken in rows:
        try:
            cands.append((sid, folder, datetime.fromisoformat(taken)))
        except ValueError:
            continue
    if not cands:
        return None, None, f"frame {n} is not in the catalog"
    if t is None:
        if len({c[0] for c in cands}) > 1:
            return None, None, (f"frame {n} appears in {len({c[0] for c in cands})} sessions "
                                f"and the JPEG carries no EXIF time to disambiguate")
        return cands[0][0], cands[0][1], "matched on frame number (single candidate)"
    scored = sorted(cands, key=lambda c: abs((c[2] - t).total_seconds()))
    best = scored[0]
    dt = abs((best[2] - t).total_seconds())
    if dt > 172800:
        return None, None, f"closest catalog frame is {dt/3600:.1f}h from the JPEG's EXIF time"
    rival = next((c for c in scored[1:] if c[0] != best[0]), None)
    if rival is not None and abs((rival[2] - t).total_seconds()) - dt < 60:
        return None, None, ("two sessions match this frame equally well "
                            f"({Path(best[1]).name} vs {Path(rival[1]).name})")
    return best[0], best[1], f"matched on frame number + capture time (±{dt:.0f}s)"


def catalog_times(session_id):
    """{frame_number: (datetime, set_id)} for ONE session — never model-wide.

    Scoping to the session is what stops a frame shot an hour later, in another set,
    being offered as part of this burst.
    """
    if not CATALOG.is_file() or session_id is None:
        return {}
    con = sqlite3.connect(str(CATALOG))
    rows = con.execute(
        """SELECT p.filename, p.taken_at, p.set_id FROM photos p
           WHERE p.session_id = ? AND p.taken_at IS NOT NULL""", (session_id,)).fetchall()
    con.close()
    out = {}
    for fn, t, sid in rows:
        n = frame_no(fn)
        if n is None:
            continue
        try:
            out.setdefault(n, (datetime.fromisoformat(t), sid))
        except ValueError:
            pass
    return out


def same_set(times, seed_n, gap):
    """Frame numbers sharing the seed's burst.

    The catalog's own set_id is authoritative when present; the gap walk is only a
    fallback, and it now runs INSIDE the set so it cannot wander into the next one.
    """
    if seed_n not in times:
        return None
    seed_set = times[seed_n][1]
    if seed_set is not None:
        members = {n: t for n, (t, sid) in times.items() if sid == seed_set}
        if members:
            return members
    ordered = sorted(((n, t) for n, (t, _) in times.items()), key=lambda kv: kv[1])
    idx = next(i for i, (n, _) in enumerate(ordered) if n == seed_n)
    lo = idx
    while lo > 0 and (ordered[lo][1] - ordered[lo - 1][1]).total_seconds() <= gap:
        lo -= 1
    hi = idx
    while hi < len(ordered) - 1 and (ordered[hi + 1][1] - ordered[hi][1]).total_seconds() <= gap:
        hi += 1
    return {n: t for n, t in ordered[lo:hi + 1]}


def _gamma(a, g=0.45):
    """Lift shadows before detection. These frames are lit by one gelled source and
    sit far down the curve; the detector needs a face it can actually see."""
    return np.clip(((a / 255.0) ** g) * 255, 0, 255).astype(np.uint8)


def face_ratio(path, tiles=(1, 3, 5), overlap=0.5):
    """Face height as a fraction of image height, or None if no face is found.

    Detection runs on overlapping TILES as well as the whole frame. The detector
    downsamples whatever it is handed, so in a wide full-body shot the face shrinks
    below what it can resolve — which is why 98 of 101 frames here were reported as
    faceless when a person could plainly see the faces. Cropping to tiles keeps the
    face large relative to the input.
    """
    im = Image.open(path)
    im.draft("RGB", (2000, 2000))
    a = _gamma(np.asarray(im.convert("RGB")))
    H, W = a.shape[:2]
    for g in tiles:
        th, tw = max(1, H // g), max(1, W // g)
        sy, sx = max(1, int(th * (1 - overlap))), max(1, int(tw * (1 - overlap)))
        for y in range(0, max(1, H - th + 1), sy):
            for x in range(0, max(1, W - tw + 1), sx):
                pts = FA.landmarks(np.ascontiguousarray(a[y:y + th, x:x + tw]))
                if pts:
                    # express the face height against the FULL frame, so the
                    # scale-tolerance test still compares like with like
                    return float(np.linalg.norm(pts["forehead"] - pts["chin"])) / H
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True, help="the photo whose look you liked")
    p.add_argument("--style", required=True, help="style reference image")
    p.add_argument("--name", required=True, help="reel name, e.g. reel2")
    p.add_argument("--prompt", required=True)
    p.add_argument("--count", type=int, default=8, help="frames in the reel, seed included")
    p.add_argument("--gap", type=float, default=120.0, help="seconds that separate two sets")
    p.add_argument("--seed-value", type=int, default=1234,
                   help="diffusion seed — SHARED by every frame so the style stays coherent")
    p.add_argument("--model-name", default=None, help="override catalog model lookup")
    p.add_argument("--scale-tolerance", type=float, default=1.8,
                   help="keep frames whose face is within this factor of the seed's face "
                        "size; stops a wide full-body frame landing in a close-up reel")
    p.add_argument("--no-crop", action="store_true", help="skip the 9:16 story crop")
    p.add_argument("--only", default=None,
                   help="comma-separated frame numbers to use, instead of auto-picking. "
                        "Auto-pick judges scale and faces, never nudity — so when frames "
                        "have been vetted by eye, pin them here rather than hoping the "
                        "picker lands on the same ones.")
    p.add_argument("--dry-run", action="store_true", help="pick frames and stop")
    args = p.parse_args()

    seed = Path(args.seed).expanduser().resolve()
    if not seed.is_file():
        sys.exit(f"missing seed: {seed}")
    session_dir = seed.parent.parent          # .../<session>/Processed/x.jpg
    model = args.model_name or seed.relative_to(
        Path("~/.openclaw/workspace/_photos").expanduser()).parts[0]

    # every JPEG of this session, by frame number
    jpgs = {}
    for f in sorted(glob.glob(str(session_dir) + "/**/*.jpg", recursive=True)):
        n = frame_no(f)
        if n is not None:
            jpgs.setdefault(n, f)         # first hit wins; Processed sorts before Unprocessed
    seed_n = frame_no(seed)

    sid, folder, why = resolve_session(str(seed))
    if sid is None:
        sys.exit(f"cannot identify the session for this seed: {why}")
    print(f"session: {Path(folder.replace(chr(92), '/')).name}  ({why})")
    times = catalog_times(sid)
    members = same_set(times, seed_n, args.gap)
    if members:
        t0 = times[seed_n][0]
        span = (max(members.values()) - min(members.values())).total_seconds()
        print(f"set: {len(members)} catalogued frames over {span:.0f}s "
              f"(gap threshold {args.gap:.0f}s)")
        pool = sorted((n for n in members if n in jpgs),
                      key=lambda n: abs((times[n][0] - t0).total_seconds()))
    else:
        sys.exit(f"seed frame {seed_n} is not in the catalog for this session — "
                 f"cannot establish which set it belongs to")

    seed_ratio = face_ratio(jpgs[seed_n]) or 0.0
    tol = args.scale_tolerance
    print(f"seed face ratio {seed_ratio:.3f}; keeping "
          f"{seed_ratio/tol:.3f}-{seed_ratio*tol:.3f}")
    if args.only:
        want = [int(x) for x in re.findall(r"\d+", args.only)]
        missing = [n for n in want if n not in jpgs]
        outside = [n for n in want if members and n not in members]
        if missing:
            sys.exit(f"--only names frames with no JPEG in this session: {missing}")
        if outside:
            sys.exit(f"--only names frames outside the seed's set: {outside} — "
                     f"a reel must be one continuous moment")
        picked = [jpgs[n] for n in want]
        no_face, off_scale = [], []
        print(f"using {len(picked)} pinned frames (auto-pick skipped)")
    else:
        picked, no_face, off_scale = [], [], []
        for n in pool:
            if len(picked) >= args.count:
                break
            f = jpgs[n]
            if n == seed_n:
                picked.append(f)
                continue
            r = face_ratio(f)
            if r is None:
                no_face.append(n)
            elif seed_ratio and not (seed_ratio / tol <= r <= seed_ratio * tol):
                off_scale.append((n, round(r, 3)))
            else:
                picked.append(f)
    if not args.only:
        print(f"picked {len(picked)} frames")
    if no_face:
        print(f"  no face: {no_face}")
    if off_scale:
        print(f"  wrong scale for this reel: {off_scale}")
    if len(picked) < args.count and not args.only:
        raw_only = len(members) - len([n for n in members if n in jpgs])
        print(f"\nSHORT: {len(picked)} of {args.count} requested. In this set — "
              f"{len(no_face)} no face, {len(off_scale)} wrong scale, "
              f"{raw_only} RAW-only (no JPEG export).")
        print("Every frame above comes from the seed's own set; the reel is short "
              "rather than padded from elsewhere.")
    for f in picked:
        print("   ", Path(f).name)
    if len(picked) < 2:
        sys.exit("a reel needs at least 2 frames — nothing to build")
    if args.dry_run:
        return

    # One folder per reel: finals/ is the only thing meant for browsing, and the
    # leading underscore on _intermediates/ is load-bearing — project-hub's scanner
    # skips directories starting with "_", so intermediates stay out of the UI.
    style_stem = Path(args.style).stem.replace(" ", "-").replace("_", "-")
    idx = args.name if not args.name.startswith("reel") else args.name
    m = re.match(r"reel(\d+)$", idx)
    if m:
        idx = f"reel{int(m.group(1)):02d}"
    reel_dir = SHARED / "reels" / f"{idx}_{clean(model)}_{style_stem}"
    in_dir = reel_dir / "_sources"
    out_dir = reel_dir / "finals"
    src_dir = in_dir
    src_dir.mkdir(parents=True, exist_ok=True)

    for f in picked:
        shutil.copyfile(f, src_dir / f"{clean(model)}_{clean(Path(f).stem)}.jpg")

    if args.no_crop:
        run_dir = src_dir
    else:
        # Crop BEFORE stylizing: become-image renders ~1024px on the long edge, so a
        # 9:16 crop taken afterwards would throw most of that away on landscape frames.
        run_dir = reel_dir / "_sources" / "crop_916"
        run_dir.mkdir(exist_ok=True)
        finals = Path("~/.openclaw/workspace/shared/finals").expanduser()
        for f in sorted(src_dir.glob("*.jpg")):
            subprocess.run([PY, str(HERE / "smart-crop.py"), "--source", str(f), "--story"],
                           check=False, capture_output=True)
            hits = sorted(finals.glob(f"{f.stem}_crop_9*story*.jpg"))
            if hits:
                shutil.copyfile(hits[-1], run_dir / f.name)
            else:
                print(f"  crop failed, using original: {f.name}")
                shutil.copyfile(f, run_dir / f.name)

    for f in sorted(run_dir.glob("*.jpg")):
        cmd = [PY, str(HERE / "surreal_with_face.py"),
               "--relit", str(f), "--style", str(Path(args.style).expanduser()),
               "--out-dir", str(out_dir), "--prompt", args.prompt,
               "--color", "--align-face", "--match-scope", "hybrid", "--match-strength", "1.0",
               "--depth-strength", "0.95", "--denoising-strength", "0.85",
               "--mask-inner-mult", "0.9", "--mask-outer-mult", "2.8",
               "--mask-falloff-power", "1.0",
               "--seed", str(args.seed_value), "--upscale", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        last = [l for l in r.stdout.splitlines() if "→" in l or "no face" in l]
        print(f"  {f.stem}: {last[-1].strip() if last else 'FAILED'}")

    # Keep only the finals in view; intermediates still enable free recompose_face.py runs.
    inter = reel_dir / "_intermediates"
    inter.mkdir(exist_ok=True)
    for f in (list(out_dir.glob("*__surreal.jpg")) + list(out_dir.glob("*__src.jpg"))
              + list(out_dir.glob("*__bw_relit.jpg")) + list(out_dir.glob("*.json"))):
        shutil.move(str(f), str(inter / f.name))

    n = len(list(out_dir.glob("*__final.jpg")))
    print(f"\n{n} frames → {out_dir}")
    print(f"intermediates → {inter}")


if __name__ == "__main__":
    main()
