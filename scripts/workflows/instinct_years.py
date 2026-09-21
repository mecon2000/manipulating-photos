#!/usr/bin/env python3
"""model-through-years: one model, one photo per shoot, oldest shoot to newest.

The story is time, so the rules follow from it: every photo is edited and shows
her face, each comes from a different session, and the sessions span at least
two years. Within a session the strongest photo wins (rating, pick, fave).

  instinct_years.py --propose --model "Nitsan Perry"
  instinct_years.py --propose --auto 3        models with the longest history first
"""
import argparse, os, re, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instinct_pack as P
import framing


def history(model_names=None):
    """{model: [(session folder, date, [(stem, taken, rating, pick)])]}"""
    rows = P.db().execute("""
        SELECT m.name, s.folder_path, s.date, p.filename, p.taken_at, p.lr_rating, p.lr_pick
        FROM photos p JOIN sessions s ON s.id=p.session_id JOIN models m ON m.id=s.model_id
        WHERE (p.lr_rating>=1 OR p.lr_pick=1) AND s.date IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id
                          AND t.dimension='boldness' AND t.value='explicit')""").fetchall()
    by = defaultdict(lambda: defaultdict(list))
    dates = {}
    for model, fpath, d, fn, t, r, pk in rows:
        if model_names and model not in model_names:
            continue
        folder = Path(fpath).name
        by[model][folder].append((os.path.splitext(fn)[0], t or "", r or 0, pk or 0))
        dates[(model, folder)] = d
    return {m: sorted(((f, dates[(m, f)], ph) for f, ph in s.items()), key=lambda x: x[1])
            for m, s in by.items()}


def year_of(d):
    try:
        return int(str(d)[:4])
    except Exception:
        return None


def build_for(model, sessions, R, a, fav):
    picks = []
    for folder, d, photos in sessions:
        photos.sort(key=lambda p: p[2] * 2 + (p[3] == 1) + (2 if p[0] in fav else 0), reverse=True)
        for stem, t, r, pk in photos[:25]:
            if stem not in R.idx:
                continue
            e = R.export(folder, stem, "edited")
            if not e:
                continue
            m = framing.measure(e[1])
            if not (m and m.get("eyes")):              # her face is the point
                continue
            v = P.sfw_of(e[0])
            if v["verdict"] == "fail":
                continue
            picks.append((folder, d, stem, t, r, pk, e, v))
            break
    years = sorted({year_of(p[1]) for p in picks if year_of(p[1])})
    if len(picks) < a.min_sessions or not years or years[-1] - years[0] < a.min_span:
        span = (years[-1] - years[0]) if years else 0
        print(f"skip {model}: {len(picks)} usable shoots over {span} years")
        return False
    if len(picks) > a.count:                     # keep the ends, thin the middle evenly
        idx = sorted({round(i * (len(picks) - 1) / (a.count - 1)) for i in range(a.count)})
        picks = [picks[i] for i in idx]
    entries = []
    for i, (folder, d, stem, t, r, pk, e, v) in enumerate(picks, 1):
        entries.append((f"{i:02d}_edited_{stem}.jpg", "edited", e[0],
                        {"model": model, "session": folder, "shot": str(d)[:10],
                         "rating": P.stars(r, pk, stem in fav), "sfw": v, "stem": stem}))
    span = f"{years[0]}–{years[-1]}"
    P.write_pack(f"model-years__{re.sub(r'[^A-Za-z0-9]+', '', model)}", "model-years",
                 f"{model} through the years — one photo per shoot, oldest to newest ({span})",
                 entries, [("Model", model), ("Years", span), ("Shoots", len(entries))],
                 build=dict(cmd="files"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--auto", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--min-sessions", type=int, default=4)
    ap.add_argument("--min-span", type=int, default=2, help="years between first and last shoot")
    a = ap.parse_args()
    P.PROPOSE = a.propose
    R = P.Resolver()
    fav = P.favored()
    hist = history({a.model} if a.model else None)
    if a.model:
        if a.model not in hist:
            sys.exit(f"no liked photos for {a.model} in the catalog")
        build_for(a.model, hist[a.model], R, a, fav)
        return
    use, npacks = P.model_usage()
    cands = [(m, s) for m, s in hist.items()
             if R.allowed(m) and len(s) >= a.min_sessions
             and (year_of(s[-1][1]) or 0) - (year_of(s[0][1]) or 0) >= a.min_span]
    # longest histories first, models with fewer packs ahead of busy ones
    cands.sort(key=lambda c: (npacks.get(c[0].lower(), 0), -len(c[1])))
    made = 0
    for m, s in cands:
        if made >= a.auto:
            break
        made += build_for(m, s, R, a, fav)


if __name__ == "__main__":
    main()
