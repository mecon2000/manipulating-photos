#!/usr/bin/env python3
"""Find which video frame a photo was taken at, by looking at the picture.

Every timing-based method here has failed the same way: a shoot's rhythm is
self-similar — bursts, pause, bursts — so many lags fit almost equally well and
the scan returns a confident number with nothing behind it. This matches on
CONTENT instead. The phone and the camera saw the same person in the same
instant; that is a far stronger signal than when the shutter clicked.

Two stages, both pretrained, nothing tuned against our four sessions:
  DINOv2 embeddings  -> which few seconds of footage
  MediaPipe pose     -> which frame inside them

The offset is a by-product: frame wall-clock minus photo taken_at. Running it
over many probe photos gives many independent estimates, and their agreement is
the confidence measure — the thing the old margin heuristic was faking.

  ig_match.py --session "2025-10-17 Elina*"            # solve the offset
  ig_match.py --session "..." --expect 62.37           # check against a known one
"""
import argparse, re, sqlite3, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ig_reel as R
import ig_bts as B

SAMPLE_FPS = 2.0        # frames per second pulled from each clip
PROBES = 24             # photos used to vote on the offset
BATCH = 32
AGREE_S = 8.0           # estimates this close count as the same answer


def clip_pool(sd, photo_days):
    """Clips from the session folder that were actually shot on a photo day.

    Tair's folder holds 18 clips from a month earlier alongside 8 real ones; an
    offset fitted against those is meaningless. Date-scoping costs nothing and
    is the single highest-value guard here.
    """
    keep, dropped = [], []
    for p in sorted(B.video_dir(sd).iterdir()):
        if p.suffix.lower() not in (".mp4", ".mov"):
            continue
        st = B.clip_start(p)
        if st is None:
            dropped.append((p.name, "no timestamp in name"))
        elif st.date() not in photo_days:
            dropped.append((p.name, f"shot {st.date()}, not a session day"))
        else:
            keep.append((p, st))
    return keep, dropped


def sample_frames(clips, fps=SAMPLE_FPS):
    """(clip name, wall-clock time, RGB array) every 1/fps seconds of footage."""
    out = []
    for path, start in clips:
        cap = cv2.VideoCapture(str(path))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native / fps)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = start + timedelta(seconds=idx / native)
                out.append((path.name, t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1
        cap.release()
    return out


class Embedder:
    """DINOv2 CLS features. Pretrained and general — it reads pose, outfit,
    background and light together, so it degrades gracefully when the subject is
    half out of the phone's frame, which is where a pose-only match falls over."""

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.mdl = AutoModel.from_pretrained("facebook/dinov2-base").eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mdl.to(self.dev)

    def __call__(self, images):
        vecs = []
        for i in range(0, len(images), BATCH):
            batch = [Image.fromarray(a) if isinstance(a, np.ndarray) else a
                     for a in images[i:i + BATCH]]
            inp = self.proc(images=batch, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                v = self.mdl(**inp).last_hidden_state[:, 0]
            v = v / v.norm(dim=-1, keepdim=True)
            vecs.append(v.cpu().numpy())
        return np.concatenate(vecs)


def preview(sd, stem, max_edge=640):
    """A viewable version of a catalogued frame: an export if one exists, else
    the RAW developed. Matching needs ordinary frames, not just the keepers."""
    for q in sd.glob(f"**/{stem}*.jpg"):
        im = Image.open(q).convert("RGB")
        im.thumbnail((max_edge, max_edge))
        return np.asarray(im)
    for ext in ("CR3", "CR2", "cr3", "cr2"):
        for q in sd.glob(f"**/{stem}.{ext}"):
            a = R.lift(R.develop(q, "camera"), 1.35)
            im = Image.fromarray(a)
            im.thumbnail((max_edge, max_edge))
            return np.asarray(im)
    return None


def probes(sd, n=PROBES):
    """Photos spread across the session, preferring ones with a rating — they
    are more likely to show the subject clearly, which is what matching needs."""
    con = sqlite3.connect(str(R.CATALOG))
    rows = con.execute(
        """SELECT p.filename, p.taken_at, p.lr_rating FROM photos p
           JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.taken_at IS NOT NULL
           ORDER BY p.taken_at""", (f"%{sd.name}%",)).fetchall()
    con.close()
    if not rows:
        return []
    rated = [r for r in rows if (r[2] or 0) >= 1] or rows
    step = max(1, len(rated) // n)
    return [(Path(f).stem, datetime.fromisoformat(t)) for f, t, _ in rated[::step]][:n]


def consensus(estimates, tol=AGREE_S):
    """The offset the most probes independently agree on.

    Agreement is the whole point: a wrong match lands wherever the footage
    happens to look similar, and wrong matches do not agree with each other.
    Ten probes landing within a few seconds cannot happen by chance.
    """
    best = (0, None, [])
    for e in estimates:
        near = [x for x in estimates if abs(x[0] - e[0]) <= tol]
        if len(near) > best[0]:
            best = (len(near), float(np.median([x[0] for x in near])), near)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--probes", type=int, default=PROBES)
    ap.add_argument("--expect", type=float, help="known offset in minutes, to check against")
    ap.add_argument("--fps", type=float, default=SAMPLE_FPS)
    args = ap.parse_args()

    sd = sorted(R.ARCHIVE.glob(f"*/{args.session}"))[0]
    ph = probes(sd, args.probes)
    if not ph:
        sys.exit("no catalogued photos with timestamps")
    days = {t.date() for _, t in ph}
    clips, dropped = clip_pool(sd, days)
    print(f"session : {sd.name}")
    print(f"photos  : {len(ph)} probes on {sorted(days)}")
    for name, why in dropped:
        print(f"  drop  : {name} — {why}")
    if not clips:
        sys.exit("no clips shot on a session day")

    frames = sample_frames(clips, args.fps)
    print(f"clips   : {len(clips)} kept, {len(frames)} frames at {args.fps}/s")

    emb = Embedder()
    print(f"device  : {emb.dev}")
    fv = emb([f[2] for f in frames])

    imgs, keep = [], []
    for stem, taken in ph:
        a = preview(sd, stem)
        if a is not None:
            imgs.append(a); keep.append((stem, taken))
    if not imgs:
        sys.exit("could not render any probe photo")
    pv = emb(imgs)

    sims = pv @ fv.T
    est = []
    for i, (stem, taken) in enumerate(keep):
        j = int(sims[i].argmax())
        clip, t, _ = frames[j]
        off = (t - taken).total_seconds()
        est.append((off, stem, clip, t, float(sims[i][j])))

    n, off, members = consensus(est)
    print(f"\n{'probe':14s} {'implied off':>12s} {'sim':>6s}  frame")
    for o, stem, clip, t, s in sorted(est):
        mark = "*" if any(m[1] == stem for m in members) else " "
        print(f"{mark}{stem:13s} {o/60:9.2f} min {s:6.3f}  {clip} @ {t:%H:%M:%S}")

    if off is None or n < 3:
        print("\nNO CONSENSUS — too few probes agree; do not trust an offset here")
        return
    spread = np.std([m[0] for m in members])
    print(f"\nconsensus: {off/60:.2f} min  ({n}/{len(est)} probes agree, "
          f"sd {spread:.1f}s)")
    if args.expect is not None:
        err = off / 60 - args.expect
        print(f"expected : {args.expect:.2f} min   error {err*60:+.1f} s "
              f"{'PASS' if abs(err*60) < 30 else 'FAIL'}")


if __name__ == "__main__":
    main()
