#!/usr/bin/env python3
"""Build the review page for staged instinct packs.

Reads send-to-instinct-review/*.json (written by instinct_pack.py), embeds a
small thumbnail of every staged file, and writes review.html next to them.
The page stores Ron's own checks (consent confirmed, SFW confirmed, pack
approved) in the artifact's db under review/<pack>, so they survive reloads
and can be read back with read_db.
"""
import base64, io, json, sys
from pathlib import Path
from PIL import Image

HOME = Path.home()
REVIEW = HOME / ".openclaw/workspace/shared/send-to-instinct-review"
DECISIONS = REVIEW / "decisions"   # read_db dump of review/<slug>
OUTBOX = HOME / ".openclaw/workspace/shared/send-to-instinct"


def thumb(path, side=820):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=60)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    props = []
    pdir = REVIEW / "proposals"
    for j in sorted(pdir.glob("*.json")):
        p = json.loads(j.read_text())
        dec = DECISIONS / f"{p['name']}.json"
        if dec.exists() and json.loads(dec.read_text()).get("dismissed"):
            continue                      # Ron removed or hid it; don't ship it again
        tdir = pdir / p["name"]
        for f in p["files"] + p.get("reserve", []):
            t = tdir / f["file"]
            f["thumb"] = thumb(t) if t.exists() else ""
            f.pop("src", None)            # local paths never go into the page
        props.append(p)
    if not props:
        sys.exit("no proposals — run instinct_run.py first")
    html = (Path(__file__).with_name("instinct_review.html").read_text()
            .replace("/*PACKS*/[]", json.dumps(props)))
    out = REVIEW / "review.html"
    out.write_text(html)
    print(f"{out}  ({len(props)} proposals, {sum(len(p['files']) for p in props)} photos, "
          f"{out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
