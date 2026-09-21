#!/usr/bin/env python3
"""Assemble a burst+finished group: burst frames -> unedited hero -> edited hero.

  01..NN_burst_<frame>.jpg   RAW frames shot just before the hero, same set
  10_unedited.jpg            the hero, from RAW
  20_edited.jpg              the hero, Lightroom export
  (30_stylized.jpg)          optional

Burst frames come from the catalog: same SET as the hero, preceding it by
capture time. Frame numbers are not adjacency — consecutive numbers can be
minutes and a whole setup apart.

All framing (no distortion, whole subject, eye lock) is done by framing.py.
"""
import argparse, json, os, re, shutil, sqlite3, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import framing
from PIL import Image

CATALOG = os.path.expanduser("~/gitrep/photo-catalogging/data/photo-catalog.db")


def burst_by_time(session_dir, stem, want, window):
    """Stems shot just before the hero inside the hero's SET, playback order.

    Ordered by whole second, then by frame name: the catalog lost the sub-second
    on some rows (8575 reads 09:34:10.65, 8576 a bare 09:34:10), so trusting
    the fraction shuffles the burst. The frame counter never lies.
    """
    folder = os.path.basename(str(session_dir).rstrip("/"))
    try:
        db = sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)
        row = db.execute(
            """SELECT p.set_id, p.taken_at FROM photos p
               JOIN sessions s ON s.id = p.session_id
               WHERE s.folder_path LIKE ? AND p.filename LIKE ?""",
            (f"%{folder}", stem + ".%")).fetchone()
        if not row or row[0] is None or not row[1]:
            return None
        set_id, t0 = row[0], datetime.fromisoformat(row[1])
        rows = db.execute("SELECT filename, taken_at FROM photos WHERE set_id = ? "
                          "AND taken_at IS NOT NULL", (set_id,)).fetchall()
    except Exception:
        return None
    near = []
    for fn, t in rows:
        k = os.path.splitext(fn)[0]
        if k == stem:
            continue
        try:
            tt = datetime.fromisoformat(t)
        except Exception:
            continue
        dt = (t0 - tt).total_seconds()
        if 0 < dt <= window:
            near.append((dt, int(tt.timestamp()), k))
    near.sort()
    chosen = near[:want]
    chosen.sort(key=lambda x: (x[1], x[2]))
    return [k for _, _, k in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edited", required=True, help="the Lightroom export of the hero")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--stylized")
    ap.add_argument("--burst", type=int, default=10)
    ap.add_argument("--burst-window", type=float, default=900.0,
                    help="seconds back to reach inside the hero's SET")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    rawdir = Path(a.raw_dir)
    prefix = re.match(r"([A-Za-z_]+)\d+", a.stem).group(1)

    def rawp(stem):
        for ext in (".CR3", ".CR2", ".cr3", ".cr2"):
            p = rawdir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    hero_raw = rawp(a.stem)
    if not hero_raw:
        sys.exit(f"no RAW for the hero {a.stem} in {rawdir}")
    hero = framing.load_raw(hero_raw)
    edited = Image.open(a.edited).convert("RGB")
    corr = framing.same_photo(hero, edited)
    if corr < 0.45:
        sys.exit(f"REFUSING: {rawdir.name}/{a.stem} is not the photo in "
                 f"{Path(a.edited).name} (corr {corr:+.3f}). Frame numbers recycle "
                 f"across sessions — the burst would be of someone else.")

    stems = burst_by_time(rawdir, a.stem, a.burst, a.burst_window) or []
    items = []
    for st in stems:
        p = rawp(st)
        if p:                      # number by what was found, so no gaps
            items.append((f"{len(items) + 1:02d}_burst_{st}.jpg", framing.load_raw(p)))
    # 80/90 sort after any burst length and never collide with a burst index
    items.append(("80_unedited.jpg", hero))
    items.append(("90_edited.jpg", edited))

    out = Path(a.outdir) / a.label
    out.mkdir(parents=True, exist_ok=True)
    plan, fnotes = framing.frame_sequence(items, str(out))
    if a.stylized:
        shutil.copyfile(a.stylized, out / "30_stylized.jpg")

    nburst = len(items) - 2
    notes = [f"label       : {a.label}",
             f"reel type   : burst+finished",
             f"hero frame  : {a.stem}",
             f"raw session : {rawdir.name}",
             f"burst found : {nburst} frames"
             + (f" ({stems[0]}..{stems[-1]})" if stems else ""),
             f"burst source: same set, within {a.burst_window:.0f}s before the hero",
             f"hero match  : corr {corr:+.3f}"] + fnotes
    ow, oh = plan["out"]
    (out / "crop.json").write_text(json.dumps(dict(aspect=plan["aspect"], w=ow, h=oh)))
    (out / "notes.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
