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


def catalog_times(model_hint):
    """{frame_number: datetime} for every catalogued frame of this model."""
    if not CATALOG.is_file():
        return {}
    con = sqlite3.connect(str(CATALOG))
    rows = con.execute(
        """SELECT p.filename, p.taken_at FROM photos p
           JOIN sessions s ON s.id = p.session_id
           JOIN models m ON m.id = s.model_id
           WHERE m.name LIKE ? AND p.taken_at IS NOT NULL""",
        (f"%{model_hint}%",)).fetchall()
    con.close()
    out = {}
    for fn, t in rows:
        n = frame_no(fn)
        if n is None:
            continue
        try:
            out.setdefault(n, datetime.fromisoformat(t))
        except ValueError:
            pass
    return out


def same_set(times, seed_n, gap):
    """Frame numbers sharing the seed's burst: walk out until a gap > `gap` seconds."""
    if seed_n not in times:
        return None
    ordered = sorted(times.items(), key=lambda kv: kv[1])
    idx = next(i for i, (n, _) in enumerate(ordered) if n == seed_n)
    lo = idx
    while lo > 0 and (ordered[lo][1] - ordered[lo - 1][1]).total_seconds() <= gap:
        lo -= 1
    hi = idx
    while hi < len(ordered) - 1 and (ordered[hi + 1][1] - ordered[hi][1]).total_seconds() <= gap:
        hi += 1
    return {n: t for n, t in ordered[lo:hi + 1]}


def face_ratio(path):
    """Face height as a fraction of image height, or None if no face is found.

    The normalized retry is left ON — dim, heavily gelled frames (a BDSM room lit by one
    red source) fail the first pass and are exactly the frames such a reel is made of.
    """
    im = Image.open(path)
    im.draft("RGB", (900, 900))
    im = im.convert("RGB")
    pts = FA.landmarks(np.asarray(im))
    if not pts:
        return None
    return float(np.linalg.norm(pts["forehead"] - pts["chin"])) / im.size[1]


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
    for f in glob.glob(str(session_dir) + "/**/*.jpg", recursive=True):
        n = frame_no(f)
        if n is not None:
            jpgs.setdefault(n, f)         # first hit wins; Processed sorts before Unprocessed
    seed_n = frame_no(seed)

    times = catalog_times(model)
    members = same_set(times, seed_n, args.gap)
    if members:
        t0 = times[seed_n]
        span = (max(members.values()) - min(members.values())).total_seconds()
        print(f"set: {len(members)} catalogued frames over {span:.0f}s "
              f"(gap threshold {args.gap:.0f}s)")
        pool = sorted((n for n in members if n in jpgs),
                      key=lambda n: abs((times[n] - t0).total_seconds()))
    else:
        print("no catalog timestamps — falling back to frame-number adjacency")
        pool = sorted(jpgs, key=lambda n: abs(n - seed_n))

    seed_ratio = face_ratio(jpgs[seed_n]) or 0.0
    tol = args.scale_tolerance
    print(f"seed face ratio {seed_ratio:.3f}; keeping "
          f"{seed_ratio/tol:.3f}-{seed_ratio*tol:.3f}")
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
    print(f"picked {len(picked)} frames")
    if no_face:
        print(f"  no face: {no_face}")
    if off_scale:
        print(f"  wrong scale for this reel: {off_scale}")
    if len(picked) < args.count:
        print(f"NOTE: only {len(picked)} of {args.count} available as JPEG in this set "
              f"(the rest are RAW-only)")
    for f in picked:
        print("   ", Path(f).name)
    if args.dry_run:
        return

    style_stem = Path(args.style).stem.replace(" ", "_").replace("-", "_")
    tag = f"{clean(model)}_{style_stem}_{args.name}"
    in_dir, out_dir = SHARED / f"input_{tag}", SHARED / f"output_{tag}"
    (in_dir / "originals").mkdir(parents=True, exist_ok=True)
    src_dir = in_dir / "originals"

    for f in picked:
        shutil.copyfile(f, src_dir / f"{clean(model)}_{clean(Path(f).stem)}.jpg")

    if args.no_crop:
        run_dir = src_dir
    else:
        # Crop BEFORE stylizing: become-image renders ~1024px on the long edge, so a
        # 9:16 crop taken afterwards would throw most of that away on landscape frames.
        run_dir = in_dir / "crop_916"
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
    inter = out_dir / "_intermediates"
    inter.mkdir(exist_ok=True)
    for f in list(out_dir.glob("*__surreal.jpg")) + list(out_dir.glob("*__src.jpg")) \
            + list(out_dir.glob("*__bw_relit.jpg")):
        shutil.move(str(f), str(inter / f.name))

    n = len(list(out_dir.glob("*__final.jpg")))
    print(f"\n{n} frames → {out_dir}")
    print(f"intermediates → {inter}")


if __name__ == "__main__":
    main()
