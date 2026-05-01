#!/usr/bin/env python3
"""Crop the rightmost panel out of a side-by-side composite, replacing the
original in favorites/ (with a backup).

Reads `~/.openclaw/workspace/shared/_sxs_review/manifest.json` produced by
`find_side_by_side.py` (each entry has `file` + detected `panels` count).
For each: opens the original from favorites/, slices the rightmost N-th
panel, backs the original up to `favorites/_pre_split/`, and writes the
cropped version over the favorites/ filename.

Convention: the original side-by-side tools always pasted the stylized
output on the RIGHT (or rightmost for 3-panel "orig | mid | out"). Crop
the rightmost panel by default. Override with `--keep left|middle|right`
or per-file in the manifest with a `keep` field.

Usage:
    ./scripts/workflows/split_side_by_side.py            # process all in manifest
    ./scripts/workflows/split_side_by_side.py --dry-run  # preview only
    ./scripts/workflows/split_side_by_side.py --only 1,3,5  # by manifest #
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

FAV = Path("~/.openclaw/workspace/shared/favorites").expanduser()
REVIEW = Path("~/.openclaw/workspace/shared/_sxs_review").expanduser()
BACKUP = FAV / "_pre_split"


def crop_panel(img: Image.Image, panels: int, keep: str) -> Image.Image:
    W, H = img.size
    pw = W // panels
    if keep == "left":
        i = 0
    elif keep == "middle":
        i = panels // 2
    else:  # right (default)
        i = panels - 1
    return img.crop((i * pw, 0, (i + 1) * pw if i < panels - 1 else W, H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", choices=["left", "middle", "right"], default="right")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="",
                    help="comma-separated 1-based indices from manifest")
    args = ap.parse_args()

    mf_path = REVIEW / "manifest.json"
    if not mf_path.is_file():
        raise SystemExit(f"manifest missing: {mf_path}")
    items = json.loads(mf_path.read_text())
    only = {int(x) for x in args.only.split(",") if x.strip()}

    BACKUP.mkdir(parents=True, exist_ok=True)
    print(f"backups → {BACKUP}")

    for idx, it in enumerate(items, 1):
        if only and idx not in only:
            continue
        fname = it["file"]
        panels = it.get("panels", 2)
        keep = it.get("keep", args.keep)
        src = FAV / fname
        if not src.is_file():
            print(f"  [skip] #{idx} {fname}: file gone")
            continue
        if args.dry_run:
            img = Image.open(src)
            W, H = img.size
            pw = W // panels
            print(f"  [dry] #{idx} {fname}  {W}x{H} → {panels}p, keep={keep}, "
                  f"out ≈ {pw}x{H}")
            continue
        # Backup
        bk = BACKUP / fname
        if not bk.exists():
            shutil.copyfile(src, bk)
        try:
            img = Image.open(src).convert("RGB")
            cropped = crop_panel(img, panels, keep)
            cropped.save(src, quality=92)
            print(f"  ✔ #{idx} {fname}  → {cropped.size[0]}x{cropped.size[1]} (kept {keep} of {panels}p)")
        except Exception as e:
            print(f"  ✗ #{idx} {fname}: {e}")


if __name__ == "__main__":
    main()
