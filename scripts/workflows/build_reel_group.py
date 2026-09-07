#!/usr/bin/env python3
"""Assemble one CapCut-ready group: burst frames + unedited + edited + stylized.

Outputs a numbered folder so CapCut imports it in timeline order. Nothing is
rendered into a video -- the edit happens by hand, which is the point.

The burst run is found by background similarity, not by frame adjacency: a shoot
moves between setups every few dozen frames, and neighbouring frame numbers
routinely cross that boundary.
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_transform_reel import (locate_crop, load_raw, censor, lace_hem, W, H)
from PIL import Image
import numpy as np


FACE = os.path.expanduser("~/openclaw-venv/mediapipe_models/face_landmarker.task")
_DET = None
_POSE = None
POSE_MODEL = os.path.expanduser("~/openclaw-venv/mediapipe_models/pose_landmarker.task")


def eye_anchor(im):
    """Midpoint between the eye centres, in pixels, plus the inter-ocular
    distance. Returns None if no face is found."""
    global _DET
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    if _DET is None:
        _DET = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=FACE), num_faces=1))
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    res = _DET.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
    if not res.face_landmarks:
        return pose_eyes(im)          # face mesh misses profiles; pose does not
    L = res.face_landmarks[0]
    W_, H_ = im.width, im.height
    def c(i, j):
        return ((L[i].x + L[j].x) / 2 * W_, (L[i].y + L[j].y) / 2 * H_)
    lx, ly = c(33, 133)          # one eye
    rx, ry = c(362, 263)         # the other
    iod = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    return ((lx + rx) / 2, (ly + ry) / 2, iod)


def pose_eyes(im):
    """Fallback anchor from the pose model, which still finds the eyes on a
    profile where the face mesh returns nothing."""
    global _POSE
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    if "_POSE" not in globals() or _POSE is None:
        globals()["_POSE"] = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=POSE_MODEL),
                num_poses=1))
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    r = _POSE.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
    if not r.pose_landmarks:
        return None
    L = r.pose_landmarks[0]
    lx, ly = L[2].x * im.width, L[2].y * im.height
    rx, ry = L[5].x * im.width, L[5].y * im.height
    iod = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    return ((lx + rx) / 2, (ly + ry) / 2, iod or 1.0)


def crop_to_anchor(im, anchor, target, size, scale=None):
    """Crop `size` out of `im` so that `anchor` lands exactly on `target`.
    Pads with edge pixels rather than clamping -- clamping would silently
    break the alignment this exists to guarantee."""
    w, h = size
    ax, ay = anchor
    if scale and scale != 1.0:
        nw, nh = round(im.width * scale), round(im.height * scale)
        im = im.resize((nw, nh), Image.LANCZOS)
        ax, ay = ax * scale, ay * scale
    left, top = round(ax - target[0]), round(ay - target[1])
    pad = max(0, -left, -top, left + w - im.width, top + h - im.height)
    if pad:
        bg = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad))
        bg.paste(im, (pad, pad))
        # edge-extend so padding does not read as a black bar
        for x, box in ((0, (pad, pad, pad + 1, pad + im.height)),
                       (pad + im.width, (pad + im.width - 1, pad,
                                         pad + im.width, pad + im.height))):
            strip = bg.crop(box).resize((pad, im.height))
            bg.paste(strip, (x if x == 0 else x, pad))
        for y, box in ((0, (0, pad, bg.width, pad + 1)),
                       (pad + im.height, (0, pad + im.height - 1,
                                          bg.width, pad + im.height))):
            strip = bg.crop(box).resize((bg.width, pad))
            bg.paste(strip, (0, y))
        im, left, top = bg, left + pad, top + pad
    return im.crop((left, top, left + w, top + h))


def decode(p, size):
    return load_raw(p, size)


def sig(im, bins=8):
    """Coarse colour signature of the frame's outer border -- the background,
    mostly, which is what changes when the setup changes."""
    a = np.asarray(im.resize((64, 96)).convert("RGB"), dtype=np.uint8)
    border = np.concatenate([a[:12].reshape(-1, 3), a[-12:].reshape(-1, 3),
                             a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    h, _ = np.histogramdd(border, bins=(bins,) * 3,
                          range=((0, 256),) * 3)
    h = h.ravel().astype(float)
    return h / max(h.sum(), 1)


def similarity(a, b):
    return float(np.minimum(a, b).sum())                 # histogram intersection


def split_by_gap(scored, floor=0.45, want=1):
    """Keep the frames that belong to the hero's setup.

    A fixed threshold does not work: measured same-setup similarity runs
    0.87-0.93 in one shoot but 0.70 in another, while a *different* setup can
    still score 0.66 when both share a grey wall. What is stable is the gap --
    the scores fall into a tight high cluster and a low one, and the largest
    drop between consecutive sorted scores is the setup boundary."""
    if not scored:
        return []
    vals = sorted((s for _, s, _ in scored), reverse=True)
    if len(vals) == 1:
        return scored if vals[0] >= floor else []
    gaps = sorted(((vals[i] - vals[i + 1], i) for i in range(len(vals) - 1)),
                  reverse=True)
    # The biggest gap is the setup boundary. Relaxing to the next-biggest gap to
    # gather more frames was tried and is wrong: it walks straight back across
    # the boundary (BLD_4960 went from 3 correct frames to 4944..4959, three
    # setups deep). Default want=1 keeps it strict; raise --min-burst only when
    # you have checked the run by eye.
    for _, i in gaps:
        cut = vals[i]
        kept = [t for t in scored if t[1] >= cut and t[1] >= floor]
        if len(kept) >= want:
            return kept
    cut = vals[gaps[0][1]]
    return [t for t in scored if t[1] >= cut and t[1] >= floor]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stylized", required=True)
    ap.add_argument("--edited", help="the Lightroom export. Omit when the frame "
                                     "was never developed -- the group is then a "
                                     "3-beat: burst -> unedited -> stylized.")
    ap.add_argument("--sizeref", help="image whose pixel dimensions the RAW is "
                                      "resized to (default: --edited)")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--crop", help="the 9:16 crop that was fed to the stylizer")
    ap.add_argument("--burst", type=int, default=6, help="frames before the hero")
    ap.add_argument("--min-burst", type=int, default=1,
                    help="relax the setup cut until at least this many frames "
                         "are found. Above 1 this WILL cross setup boundaries.")
    ap.add_argument("--window", type=int, default=25,
                    help="how many preceding frames to score before clustering")
    ap.add_argument("--censor", action="store_true")
    ap.add_argument("--align-eyes", action="store_true",
                    help="crop every burst frame so the midpoint between the "
                         "eyes sits exactly where it sits in the chosen frame")
    ap.add_argument("--align-scale", action="store_true",
                    help="with --align-eyes, also normalise head size by the "
                         "inter-ocular distance")
    ap.add_argument("--iod-tol", type=float, default=0.30,
                    help="reject an inter-ocular measurement this far from the "
                         "run's median and substitute the median instead")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    out = Path(a.outdir) / a.label
    out.mkdir(parents=True, exist_ok=True)
    rawdir = Path(a.raw_dir)
    prefix, num = re.match(r"([A-Za-z_]+)(\d+)", a.stem).groups()
    width, n0 = len(num), int(num)
    ref = a.edited or a.sizeref or a.crop or a.stylized
    export_size = Image.open(ref).size

    def rawp(n):
        for ext in (".CR3", ".CR2"):
            p = rawdir / f"{prefix}{n:0{width}d}{ext}"
            if p.exists():
                return p
        return None

    hero_raw = rawp(n0)
    if not hero_raw:
        sys.exit(f"no RAW for the hero frame {a.stem} in {rawdir}")
    hero = decode(hero_raw, export_size)
    href = sig(hero)

    rect = None
    if a.crop and Path(a.crop).exists():
        loc = locate_crop(Path(ref), Path(a.crop))
        if loc and loc[-1] > 0.90:
            rect = loc[:4]

    # score a window of preceding frames, then keep only the hero's setup.
    # Scoring a window rather than stopping at the first miss also survives
    # culled sequences, where the same setup resumes a few frame numbers back.
    scored = []
    for n in range(n0 - 1, n0 - 1 - a.window, -1):
        p = rawp(n)
        if not p:
            continue
        im = decode(p, export_size)
        scored.append((n, similarity(href, sig(im)), im))
    kept = split_by_gap(scored, want=a.min_burst)
    kept.sort(key=lambda t: -t[0])
    run = [(n, im) for n, _, im in kept[:a.burst]]
    run.reverse()
    rejected = [(n, round(s, 3)) for n, s, _ in scored if (n, s) not in
                {(k[0], k[1]) for k in kept}]

    notes = [f"label       : {a.label}",
             f"hero frame  : {a.stem}",
             f"raw session : {rawdir.name}",
             f"burst found : {len(run)} frames "
             f"({run[0][0] if run else '-'}..{run[-1][0] if run else '-'})",
             f"setup scores: kept {[ (n, round(s,3)) for n,s,_ in kept ]}",
             f"              rejected {rejected[:8]}",
             f"crop rect   : {rect or 'none - centre cover-fit'}",
             f"censored    : {'yes' if a.censor else 'no'}",
             f"beats       : {'4 (unedited -> edited -> stylized)' if a.edited else '3 (unedited -> stylized)'}"]

    # the chosen frame's own eye position, in crop coordinates, is the target
    target = None
    if a.align_eyes:
        if not rect:
            sys.exit("--align-eyes needs the hero crop rect; pass --crop")
        x, y, w, h = rect
        # The target comes from the EDITED export, because the stylized output
        # was rendered from that crop. Lightroom's export is not pixel-aligned
        # with the RAW (its crop differs by a couple of hundred pixels here), so
        # taking the target from the RAW would leave 10_unedited misframed
        # against 20_edited and 30_stylized -- which is exactly what it did.
        ref_im = Image.open(a.edited or ref).convert("RGB")
        ha = eye_anchor(ref_im.crop((x, y, x + w, y + h)))
        if not ha:
            sys.exit("no face found in the chosen frame; cannot align")
        target, hero_iod = (ha[0], ha[1]), ha[2]
        notes.append(f"eye anchor  : ({target[0]:.0f}, {target[1]:.0f}) "
                     f"in the {w}x{h} crop, iod={hero_iod:.1f}px")

    # measure every frame first, so a single bad reading can be recognised as
    # an outlier against the run rather than trusted on its own. One frame here
    # measured iod 47.6 against a median of 83 -- a misdetection, and scaling on
    # it blew that frame up 1.9x into a close-up that broke the sequence.
    anchors, median_iod = {}, None
    if a.align_eyes:
        anchors = {n: eye_anchor(im) for n, im in run}
        anchors["hero"] = eye_anchor(hero)
        iods = sorted(v[2] for v in anchors.values() if v)
        median_iod = iods[len(iods) // 2] if iods else None
        bad = [k for k, v in anchors.items()
               if v and median_iod and abs(v[2] - median_iod) / median_iod > a.iod_tol]
        if bad:
            notes.append(f"iod median   : {median_iod:.1f}px; outliers replaced "
                         f"with it: {sorted(str(b) for b in bad)}")

    def save(im, name, align=False, key=None):
        if align and target:
            x, y, w, h = rect
            an = anchors.get(key) if key in anchors else eye_anchor(im)
            if not an:
                notes.append(f"WARNING: no face in {name} - fell back to the "
                             f"fixed crop, this frame will not be aligned")
            else:
                iod = an[2]
                if median_iod and abs(iod - median_iod) / median_iod > a.iod_tol:
                    iod = median_iod
                sc = (hero_iod / iod) if a.align_scale else None
                im = crop_to_anchor(im, (an[0], an[1]), target, (w, h), sc)
                notes.append(f"  {name}: eyes at ({an[0]:.0f}, {an[1]:.0f})"
                             + (f", scaled x{sc:.3f}" if sc else "")
                             + f", iod={an[2]:.1f}")
                im.save(out / name, quality=95)
                return
        if rect:
            x, y, w, h = rect
            im = im.crop((x, y, x + w, y + h))
        im.save(out / name, quality=95)

    for i, (n, im) in enumerate(run, 1):
        if a.censor:
            hem = lace_hem(im)
            if hem:
                im = censor(im, hem - 0.20, hem - 0.01)
            else:
                notes.append(f"WARNING: no hem found on {prefix}{n} - not censored")
        save(im, f"0{i}_burst_{prefix}{n}.jpg", align=True, key=n)
    save(hero, "10_unedited.jpg", align=True, key="hero")
    if a.edited:
        save(Image.open(a.edited).convert("RGB"), "20_edited.jpg")
    else:
        notes.append("NOTE: no Lightroom-developed version of this frame exists, "
                     "so the group is a 3-beat (no 20_edited.jpg).")
    shutil.copyfile(a.stylized, out / "30_stylized.jpg")

    (out / "notes.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
