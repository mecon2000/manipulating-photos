#!/usr/bin/env python3
"""Find more photos for proposals that already exist, without disturbing them.

The review page's "Add photo" pulls from a proposal's `reserve`. This fills it:
extra candidates found by the same rules as the original pack (same theme or
set or burst, consent-gated, verified against the RAW, SFW-pre-filtered),
never repeating a photo already in the pack or its reserve.

  instinct_topup.py --name themed-photos__cool --count 6
  instinct_topup.py --all --count 6           every open proposal
  instinct_topup.py --requested               packs where Ron pressed "Ask for more"
"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instinct_pack as P
import framing
from PIL import Image

num = lambda st: int((re.findall(r"\d{3,}", st or "") or ["0"])[0])


def existing_stems(prop):
    return {f.get("stem") or P.core(f["file"]) for f in prop["files"] + prop.get("reserve", [])}


def extras(prop):
    return dict(prop.get("extra") or [])


def find_themed(prop, R, want):
    theme = extras(prop)["Theme"]
    have = existing_stems(prop)
    per_session = {}
    for f in prop["files"] + prop.get("reserve", []):
        if f.get("session"):
            per_session[f["session"]] = per_session.get(f["session"], 0) + 1
    fav = P.favored()
    rows = P.db().execute(f"""
        SELECT p.filename, p.taken_at, p.lr_rating, p.lr_pick, s.folder_path, m.name
        FROM photos p JOIN sessions s ON s.id=p.session_id JOIN models m ON m.id=s.model_id
        WHERE (p.lr_rating>=1 OR p.lr_pick=1) AND {P.THEMES[theme]}
          AND NOT EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id
                          AND t.dimension='boldness' AND t.value='explicit')
        ORDER BY p.lr_rating DESC NULLS LAST, p.lr_pick DESC""").fetchall()
    out = []
    for fn, t, r, pk, fpath, model in rows:
        if len(out) >= want:
            break
        stem, folder = os.path.splitext(fn)[0], Path(fpath).name
        if stem in have or stem not in R.idx or per_session.get(folder, 0) >= 3:
            continue
        if not R.allowed(model):
            continue
        e = R.export(folder, stem, "edited")
        if not e:
            continue
        if theme in ("dark", "bright"):
            lum = sum(Image.open(e[0]).convert("L").resize((64, 64)).getdata()) / 4096 / 255
            if (theme == "dark") != (lum < 0.30) or (theme == "bright" and lum < 0.60):
                continue
        v = P.sfw_of(e[0])
        if v["verdict"] == "fail":
            continue
        have.add(stem)
        per_session[folder] = per_session.get(folder, 0) + 1
        out.append(("edited", e, stem, v, {"model": model, "session": folder,
                    "shot": (t or "")[:10], "rating": P.stars(r or 0, pk or 0, stem in fav)}))
    return out


def find_setvibe(prop, R, want):
    x = extras(prop)
    folder = x["Session"]
    sidx = int(re.search(r"set (\d+)", x["Set"]).group(1))
    con = P.db()
    row = con.execute("""SELECT st.id FROM sets st JOIN sessions s ON s.id=st.session_id
                         WHERE s.folder_path LIKE ? AND st.set_index=?""",
                      (f"%{folder}", sidx)).fetchone()
    if not row:
        return []
    fav = P.favored()
    photos = [(os.path.splitext(f)[0], t, r or 0, pk or 0) for f, t, r, pk in con.execute(
        "SELECT filename, taken_at, lr_rating, lr_pick FROM photos WHERE set_id=?", (row[0],))]
    photos.sort(key=lambda p: p[2] * 2 + (p[3] == 1) + (2 if p[0] in fav else 0), reverse=True)
    have = existing_stems(prop)
    out = []
    for p in photos:
        if len(out) >= want:
            break
        if p[0] in have or any(abs(num(p[0]) - num(h)) < 4 for h in have):
            continue
        e, role = R.export(folder, p[0], "edited"), "edited"
        if not e:
            e, role = R.export(folder, p[0], "unedited"), "unedited"
        if not e:
            rr = R.raw(folder, p[0])
            e, role = (rr, "raw") if rr else (None, None)
        if not e:
            continue
        v = P.sfw_of(e[1] if role == "raw" else e[0])
        if v["verdict"] == "fail":
            continue
        have.add(p[0])
        out.append((role, e, p[0], v, {"shot": (p[1] or "")[:19],
                    "rating": P.stars(p[2], p[3], p[0] in fav)}))
    return out


def find_burst(prop, R, want):
    from build_reel_group import burst_by_time
    import glob
    b = prop["build"]
    raws = glob.glob(str(P.ARCHIVE / "*" / glob.escape(b["session"])))
    have = existing_stems(prop)
    n_have = sum(1 for s in have if s != b["stem"])
    out = []
    # earlier frames of the same set, nearest the existing burst first
    for st in reversed(burst_by_time(raws[0], b["stem"], n_have + want * 3, 900.0) or []):
        if len(out) >= want:
            break
        if st in have:
            continue
        rr = R.raw(b["session"], st)
        if not rr:
            continue
        v = P.sfw_of(rr[1])
        if v["verdict"] == "fail":
            continue
        have.add(st)
        out.append(("raw", rr, st, v, {}))
    return out


def topup(pj, R, want):
    prop = json.loads(pj.read_text())
    finder = {"themed-photos": find_themed, "set-vibe": find_setvibe,
              "burst+finished": find_burst}[prop["type"]]
    found = finder(prop, R, want)
    tdir = P.PROPOSALS / prop["name"]
    tdir.mkdir(parents=True, exist_ok=True)
    reserve = prop.setdefault("reserve", [])
    for role, e, stem, v, info in found:
        kind = "burst" if prop["type"] == "burst+finished" else role
        fn = f"r{len(reserve) + 1:02d}_{kind}_{stem}.jpg"
        im = e[1]                    # every finder hands over (path, image)
        t = im.copy()
        t.thumbnail((1200, 1200))
        t.save(tdir / fn, quality=80)
        src = e[0]
        reserve.append(dict(file=fn, role=role, src=str(src), stem=stem,
                            sfw=v["verdict"], reasons=v["reasons"], **info))
    prop["reserve_at"] = time.strftime("%Y-%m-%d %H:%M")
    pj.write_text(json.dumps(prop, indent=1))
    print(f"{prop['name']}: +{len(found)} in reserve (now {len(reserve)})")
    if found:                                   # keep reserve previews in the pack's crop
        import instinct_preview
        instinct_preview.crop_proposal(pj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--requested", action="store_true")
    ap.add_argument("--count", type=int, default=6)
    a = ap.parse_args()
    targets = []
    for pj in sorted(P.PROPOSALS.glob("*.json")):
        prop = json.loads(pj.read_text())
        if prop.get("status") != "proposed":
            continue
        if a.name and prop["name"] != a.name:
            continue
        if a.requested:
            dj = P.DECISIONS / f"{re.sub(r'[^A-Za-z0-9_.~:@+-]', '_', prop['name'])[:190]}.json"
            if not (dj.exists() and json.loads(dj.read_text()).get("wantMore")):
                continue
        targets.append(pj)
    if not targets:
        sys.exit("nothing to top up")
    R = P.Resolver()
    for pj in targets:
        topup(pj, R, a.count)


if __name__ == "__main__":
    main()
