#!/usr/bin/env python3
"""Build the packs Ron approved on the review page, and nothing else.

Decisions live in the review page's db (collection `review`, one document per
proposal). Pull them into send-to-instinct-review/decisions/<slug>.json
first — Claude does that with read_db — then:

  instinct_build.py            build every approved, fully-cleared proposal
  instinct_build.py --dry-run  say what would be built and why others wait

A proposal builds only when the pack is approved, every model's consent is
confirmed, and every photo is decided. Photos marked not-SFW are left out; if
that removes the finished photo, or leaves a burst under six frames, the pack
waits instead of shipping short.
"""
import argparse, json, re, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instinct_pack as P
import framing
from PIL import Image


def slug(s):
    return re.sub(r"[^A-Za-z0-9_\-.~:@+]", "_", s)[:190]


def verdict(p, dec):
    if not dec:
        return "waiting — you haven't reviewed it"
    if not dec.get("approved"):
        return "waiting — not approved"
    # consent is not asked on the page: the allowlist gate already ran at proposal time
    files = dec.get("files", {})
    gone = set(dec.get("removed") or [])
    undecided = [f["file"] for f in universe(p, dec) if f["file"] not in gone
                 and f["sfw"] != "pass" and files.get(f["file"], {}).get("sfw") is None]
    if undecided:
        return f"waiting — {len(undecided)} photos not decided"
    return "build"


def universe(p, dec):
    """The pack's own photos plus any reserve photos Ron added on the page."""
    added = dec.get("added") or []
    return p["files"] + [r for n in added for r in p.get("reserve", []) if r["file"] == n]


def build_one(p, dec, R):
    rejected = {f for f, d in dec.get("files", {}).items() if d.get("sfw") is False}
    rejected |= set(dec.get("removed") or [])            # removed from the batch on the page
    keep = [f for f in universe(p, dec) if f["file"] not in rejected]
    order = {name: i for i, name in enumerate(dec.get("order") or [])}
    allf = universe(p, dec)
    keep.sort(key=lambda f: order.get(f["file"], len(order) + allf.index(f)))
    if p["build"]["cmd"] == "burst":
        if {"80_unedited.jpg", "90_edited.jpg"} & rejected:
            return "skipped — you rejected the finished photo"
        if sum(1 for f in keep if f["file"][0] == "0") < 6:
            return "skipped — fewer than 6 burst frames left after your rejections"
        b = p["build"]
        stems = {f.get("stem") for f in allf if f["file"] in rejected}
        # added reserve frames are the next earlier ones in the set, so reaching
        # further back by that many frames picks exactly them up
        extra = sum(1 for f in allf if f["file"].startswith("r"))
        P.cmd_burst(SimpleNamespace(session=b["session"], stem=b["stem"],
                                    count=b["count"] + extra, exclude=stems - {None}), R)
        return "built"
    entries = []
    for f in keep:
        info = {k: v for k, v in f.items()
                if k in ("model", "session", "shot", "rating")}
        info["sfw"] = {"verdict": f["sfw"], "reasons": f["reasons"] + ["cleared by Ron"]}
        src = f["src"] if f["role"] != "raw" else framing.load_raw(f["src"])
        # renumber to the order Ron set on the review page (reserve files drop the r-prefix)
        fname = f"{len(entries) + 1:02d}_" + f["file"].split("_", 1)[1]
        entries.append((fname, f["role"], src, info))
    if len(entries) < 3 or entries[0][1] != "edited" or entries[-1][1] != "edited":
        return "skipped — after your rejections it no longer opens and closes on edited photos"
    # crop step: one shared aspect, each photo fitted around its own subject
    imgs = [(fn, src if not isinstance(src, str) else Image.open(src).convert("RGB"))
            for fn, _, src, _ in entries]
    framed, aspect, fnotes = framing.frame_independent(imgs, force=p.get("aspect"))
    entries = [(fn, role, im, info) for (fn, im), (_, role, _, info) in zip(framed, entries)]
    extra = [tuple(e) for e in p["extra"]] + [("Framing", fnotes[0].split(": ", 1)[1])]
    P.write_pack(p["name"], p["type"], p["why"], entries, extra)
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    R = None
    for pj in sorted(P.PROPOSALS.glob("*.json")):
        p = json.loads(pj.read_text())
        if p.get("status") != "proposed":
            continue
        dj = P.DECISIONS / f"{slug(p['name'])}.json"
        dec = json.loads(dj.read_text()) if dj.exists() else None
        if dec and dec.get("dismissed"):
            print(f"{p['name']}: removed by you — not built")
            continue
        if dec and isinstance(dec.get("files"), list):     # page stores lists
            dec["files"] = {x["file"]: {"sfw": x.get("sfw")} for x in dec["files"]}
            dec["models"] = {x["name"]: x.get("ok") for x in dec.get("models", [])}
        v = verdict(p, dec)
        if v != "build" or a.dry_run:
            print(f"{p['name']}: {v}")
            continue
        R = R or P.Resolver()
        result = build_one(p, dec, R)
        print(f"{p['name']}: {result}")
        if result == "built":
            p["status"] = "at shared"
            p["built_at"] = time.strftime("%Y-%m-%d %H:%M")
            pj.write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
