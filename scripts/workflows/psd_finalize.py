#!/usr/bin/env python3
"""Re-export the JPEGs from a hand-edited layered PSD.

The PSD is deliberately pre-upscale: Real-ESRGAN downsamples anything over ~2.1M
pixels before it runs, so 4x layers would cost bytes and buy nothing. That leaves a
gap — after fixing a mask in Photoshop, the flat `__final.jpg` and `__final_4x.jpg`
next to it are stale, and exporting by hand replaces one and forgets the other.

This closes it: composite the PSD, overwrite the final, redo the upscale, and note
in the sidecar that the mask was edited by hand.

  psd_finalize.py <tag>__layers.psd [--no-upscale]
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parent))


FAVORITES = Path("~/.openclaw/workspace/shared/favorites").expanduser()


def sync_favorites(psd_path, final, tag):
    """Refresh any favourite made from this output.

    Faving COPIES the image, so a favourite goes stale the moment the PSD is
    edited — the copy keeps showing the machine's version of a picture that was
    hand-corrected precisely because that version was wrong. Nothing else notices,
    which makes this exactly the kind of thing to do in the flow rather than
    remember to do.
    """
    db_path = FAVORITES / "favorites.json"
    if not db_path.is_file():
        return
    db = json.loads(db_path.read_text())
    entries = db.get("favorites", db if isinstance(db, list) else [])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hits = 0
    for e in entries:
        if not isinstance(e, dict) or not e.get("file"):
            continue                                  # some old entries lack it
        # match on the recorded PSD path; fall back to the tag appearing in the
        # reconstruction command, for favourites saved before `psd` existed
        if e.get("psd"):
            match = e["psd"] == str(psd_path)
        else:
            match = tag in (e.get("command") or "")
        if not match:
            continue
        dst = FAVORITES / e["file"]
        try:
            Image.open(final).convert("RGB").save(dst, quality=95)
        except Exception as err:
            print(f"  fav refresh FAILED for {e['file']}: {err}")
            continue
        e["hand_edited"] = True
        e["refreshed_at"] = stamp
        e["note"] = "mask hand-fixed in Photoshop; re-exported via psd_finalize.py"
        hits += 1
        print(f"  favourite refreshed → {e['file']}")
    if hits:
        db_path.write_text(json.dumps(db, indent=2))
    else:
        print("  no favourite references this output")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("psd")
    p.add_argument("--upscale", type=int, default=4, choices=[0, 2, 4])
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    psd_path = Path(args.psd).expanduser().resolve()
    if not psd_path.is_file():
        sys.exit(f"missing: {psd_path}")
    tag = psd_path.name[:-len("__layers.psd")] if psd_path.name.endswith("__layers.psd") \
        else psd_path.stem
    out_dir = psd_path.parent
    final = out_dir / f"{tag}__final.jpg"

    img = PSDImage.open(str(psd_path)).composite()
    if img is None:
        sys.exit("could not composite the PSD")
    img = img.convert("RGB")

    # Did the PSD actually diverge from the final sitting next to it? Re-exporting an
    # untouched PSD must not leave a "hand_edited" mark behind — that would put a
    # false provenance claim on every output this is ever run against.
    # Measured on real data: a JPEG round-trip moves the mean by ~1.0, so a mean
    # threshold cannot separate "re-exported" from "edited". Counting pixels that
    # move more than JPEG noise can: a genuine mask fix shifted 37.7% of them,
    # re-encoding shifted 0.005%.
    changed = True
    if final.exists():
        import numpy as np
        try:
            prev = Image.open(final).convert("RGB")
            if prev.size != img.size:
                changed = True
            else:
                d = np.abs(np.asarray(prev, np.int16)
                           - np.asarray(img, np.int16)).max(axis=2)
                changed = bool((d > 12).mean() > 0.002)
        except Exception:
            changed = True
    if not changed:
        # The final is current, but the upscale may not be — it is a separate
        # derived file and an earlier run may have skipped it. Returning here
        # would make a stale 4x permanently unfixable.
        print(f"  PSD matches {final.name} — final already current")

    # Keep the machine's version before overwriting it with the hand-edited one;
    # the whole point of the edit is that the two differ.
    if final.exists() and not args.no_backup:
        keep = out_dir / f"{tag}__final__pre_handedit.jpg"
        if not keep.exists():
            # PIL, not shutil: copy2 sets mtime, which drvfs refuses (WSL 9p mount)
            Image.open(final).convert("RGB").save(keep, quality=95)
            print(f"  kept original → {keep.name}")

    if changed:
        img.save(final, quality=95)
        print(f"→ {final}")

    if args.upscale:
        big = out_dir / f"{tag}__final_{args.upscale}x.jpg"
        stale = (not big.exists()
                 or big.stat().st_mtime < final.stat().st_mtime
                 or changed)
        if stale:
            from surreal_with_face import upscale_replicate
            big.write_bytes(upscale_replicate(str(final), scale=args.upscale))
            print(f"→ {big}")
        else:
            print(f"  {big.name} already current")

    if changed:
        sync_favorites(psd_path, final, tag)

    meta_path = final.with_suffix(".json")
    if changed and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["hand_edited"] = {"psd": str(psd_path),
                               "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  sidecar marked hand-edited")


if __name__ == "__main__":
    main()
