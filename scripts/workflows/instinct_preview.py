#!/usr/bin/env python3
"""Crop a proposal's previews the way its pack will be built.

Ron approves what instinct will receive, so the review thumbnails ARE the
crop: themed and set-vibe get one shared aspect with each photo fitted around
its own subject; bursts get the eye-locked framing. The chosen aspect is saved
in the proposal and the build reuses it, so approval and output can't drift.

  instinct_preview.py --all        every open proposal not yet cropped
  instinct_preview.py --name X     one proposal (re-crops, e.g. after a top-up)
"""
import argparse, json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instinct_pack as P
import framing
from PIL import Image


def load(f):
    src = f.get("src")
    if not src:
        return None
    return framing.load_raw(src) if f["role"] == "raw" else Image.open(src).convert("RGB")


def crop_proposal(pj):
    prop = json.loads(pj.read_text())
    tdir = P.PROPOSALS / prop["name"]
    items = prop["files"] + prop.get("reserve", [])
    imgs = [(f["file"], load(f)) for f in items]
    imgs = [(n, im) for n, im in imgs if im is not None]
    if not imgs:
        return
    if prop["type"] == "burst+finished":
        own = [(n, im) for n, im in imgs if not n.startswith("r")]
        with tempfile.TemporaryDirectory() as td:
            plan, _ = framing.frame_sequence(own, td)
            for n, _ in own:
                t = Image.open(Path(td) / n); t.thumbnail((1200, 1200)); t.save(tdir / n, quality=82)
        aspect = plan["aspect"]
        rest = [(n, im) for n, im in imgs if n.startswith("r")]
        if rest:          # reserve frames: same aspect, subject-fit (not in the lock yet)
            framed, _, _ = framing.frame_independent(rest, force=aspect)
            for n, im in framed:
                im.thumbnail((1200, 1200)); im.save(tdir / n, quality=82)
    else:
        framed, aspect, _ = framing.frame_independent(imgs, force=prop.get("aspect"))
        for n, im in framed:
            im.thumbnail((1200, 1200)); im.save(tdir / n, quality=82)
    prop["aspect"] = aspect
    pj.write_text(json.dumps(prop, indent=1))
    print(f"{prop['name']}: previews cropped to {aspect}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--name")
    a = ap.parse_args()
    for pj in sorted(P.PROPOSALS.glob("*.json")):
        prop = json.loads(pj.read_text())
        if prop.get("status") != "proposed":
            continue
        if a.name and prop["name"] != a.name:
            continue
        if a.all and prop.get("aspect") and not a.name:
            continue
        crop_proposal(pj)


if __name__ == "__main__":
    main()
