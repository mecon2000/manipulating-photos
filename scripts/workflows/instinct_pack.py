#!/usr/bin/env python3
"""Stage photo packs for the reel builder ("instinct"): one folder = one reel.

Three reel types, each stated in the pack's meta.txt:

  burst+finished  a run of RAW frames shot just before a photo, ending on its
                  edited version. Eye-locked framing (framing.py).
  set-vibe        the strongest frames of ONE set, deliberately not adjacent,
                  so the reel feels like what that setup produced. Opens and
                  closes on edited photos; the middle uses edited where it
                  exists, else the unprocessed export, else the RAW.
  themed-photos   edited photos from DIFFERENT sessions sharing a theme (dark,
                  bright, bw, cool, red, outdoor, studio, lowlight, shibari,
                  bdsm, closeup). Opens and closes on photos rated 5 or faved.

set-vibe and themed-photos ship full, uncropped frames — the builder frames
them. Every photo used is verified to be the photo the catalog row describes
(frame numbers recycle across sessions), consent-gated on its folder, and
single-person.

  instinct_pack.py burst  --session "<folder>" --stem BLD_8580
  instinct_pack.py setvibe --session "<folder>" --set-index 3
  instinct_pack.py setvibe --auto 2
  instinct_pack.py themed --theme dark
"""
import argparse, glob, json, os, re, sqlite3, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/.claude/skills/candidates"))
import framing
import sfw
import consent as C
from PIL import Image

HOME = Path.home()
CATALOG = HOME / "gitrep/photo-catalogging/data/photo-catalog.db"
PHOTOS = HOME / ".openclaw/workspace/_photos"
ARCHIVE = HOME / ".openclaw/workspace/_archive"
OUTBOX = HOME / ".openclaw/workspace/shared/send-to-instinct"
REVIEW = HOME / ".openclaw/workspace/shared/send-to-instinct-review"   # never inside OUTBOX
PROPOSALS = REVIEW / "proposals"
DECISIONS = REVIEW / "decisions"     # Ron's checks, pulled from the review page's db
PROPOSE = False                      # set by --propose: thumbnails + proposal only
FAVS = HOME / ".openclaw/workspace/shared/favorites/favorites.json"
INDEX = HOME / ".cache/instinct_pack/photo_index.json"
SKIP_DIRS = ("onlyfans", "don't publish", "dont publish", "not for publish")
EXCLUDE_FILE = HOME / ".config/instinct/exclude_models.txt"


def model_usage():
    """How often each model already appears across live proposals (proposed,
    built, or reel received). Selection ranks by it so a strong, highly-rated
    model doesn't headline every theme — the first pass put one model in six
    of fourteen packs."""
    use, packs = {}, {}
    for j in PROPOSALS.glob("*.json"):
        try:
            p = json.loads(j.read_text())
        except Exception:
            continue
        x = dict(p.get("extra") or [])
        for f in p.get("files", []):
            m = (f.get("model") or x.get("Model") or "").lower()
            if m:
                use[m] = use.get(m, 0) + 1
                packs.setdefault(m, set()).add(p["name"])
    return use, {m: len(v) for m, v in packs.items()}


def excluded_models():
    """Models Ron has ruled out of instinct reels, matched on the exact name
    (case-insensitive) — 'Eden (Neta)' must not also remove 'Eden Layla'."""
    try:
        return {l.strip().lower() for l in EXCLUDE_FILE.read_text().splitlines()
                if l.strip() and not l.startswith("#")}
    except FileNotFoundError:
        return set()

THEMES = {   # SQL condition on photos p / sessions s
    "bw": "EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id AND "
          "t.dimension='color_mode' AND t.value='bw')",
    "cool": "EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id AND "
            "t.dimension='color_temperature' AND t.value='cool')",
    "red": "EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id AND "
           "t.dimension='prominent_color' AND t.value='red')",
    "blue": "EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id AND "
            "t.dimension='prominent_color' AND t.value='blue')",
    "outdoor": "s.location_type IN ('outdoor_nature','outdoor_urban','beach')",
    "studio": "s.location_type='studio'",
    "lowlight": "p.iso>=3200",
    "shibari": "(lower(s.folder_path) LIKE '%shibari%' OR lower(p.lr_keywords) LIKE '%shibari%')",
    "bdsm": "(lower(s.folder_path) LIKE '%bdsm%' OR lower(p.lr_keywords) LIKE '%bdsm%')",
    "closeup": "p.focal_length_mm>=85",
    # Lightroom keywords are a JSON list, so '"word' anchors to the start of a
    # keyword — that keeps out "watermelon carpet" and "Meteor shower"
    "water": "(" + " OR ".join(f"lower(p.lr_keywords) LIKE '%{k}%'" for k in (
        '"bath"', '"bathroom"', '"shower', 'in shower', 'near shower', '"wet"',
        '"water"', '"pool"', 'near water')) + ")",
    "smoke": "(lower(p.lr_keywords) LIKE '%\"smoke%' OR lower(p.lr_keywords) LIKE '%&smoke%'"
             " OR lower(s.folder_path) LIKE '%smoke%')",
    "mirror": "lower(p.lr_keywords) LIKE '%mirror%'",
    "pink": "EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id AND "
            "t.dimension='prominent_color' AND t.value='pink')",
    "white": "EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id AND "
             "t.dimension='prominent_color' AND t.value='white')",
    "bodypaint": "lower(p.lr_keywords) LIKE '%\"body paint%'",
    "glitter": "lower(p.lr_keywords) LIKE '%glitter%'",
    "projector": "lower(p.lr_keywords) LIKE '%projector%'",
    "oil": "lower(p.lr_keywords) LIKE '%oil & droplets%'",
    "beach": "s.location_type='beach'",
    "bar": "s.location_type IN ('bar','club')",
    "bed": "lower(p.lr_keywords) LIKE '%\"bed\"%'",
    "dark": "1=1", "bright": "1=1",           # decided on the pixels
}


def db():
    return sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)


def core(name):
    m = re.search(r"([A-Z0-9]{3,4}_\d{3,5})", name.upper())
    return m.group(1) if m else None


def photo_index():
    """stem -> [paths] for every JPEG under _photos (cached a day)."""
    if INDEX.exists() and time.time() - INDEX.stat().st_mtime < 86400:
        return json.loads(INDEX.read_text())
    idx = {}
    for root, dirs, files in os.walk(PHOTOS):
        if any(s in root.lower() for s in SKIP_DIRS):
            dirs[:] = []
            continue
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")) and (k := core(f)):
                idx.setdefault(k, []).append(os.path.join(root, f))
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(idx))
    return idx


class Resolver:
    """Find, verify and consent-check the files behind one catalog row."""

    def __init__(self):
        self.idx = photo_index()
        self.models, self.aliases = C.load()
        self.excluded = excluded_models()
        self._raw = {}

    def raw(self, folder, stem):
        key = (folder, stem)
        if key not in self._raw:
            im = None
            for d in glob.glob(str(ARCHIVE / "*" / glob.escape(folder))):
                for ext in (".CR3", ".CR2", ".cr3", ".cr2"):
                    if os.path.exists(p := os.path.join(d, stem + ext)):
                        im = (p, framing.load_raw(p))
                        break
                if im:
                    break
            self._raw[key] = im
        return self._raw[key]

    def consented(self, path):
        top = Path(path).relative_to(PHOTOS).parts[0]
        if top.lower() in self.excluded:
            return False
        return C.decide(top, self.models, self.aliases)[0] == "face_ok"

    def allowed(self, model):
        return (model.lower() not in self.excluded
                and C.decide(model, self.models, self.aliases)[0] == "face_ok")

    def export(self, folder, stem, kind):
        """kind 'edited' or 'unedited'. Returns (path, image) verified against
        the RAW, or None. Without a RAW there is nothing to verify against, so
        nothing is returned — guessing is how strangers got into bursts."""
        cands = [p for p in self.idx.get(stem, [])
                 if (kind == "edited") != ("unprocess" in p.lower())
                 and self.consented(p)]
        if not cands:                       # skip the RAW decode when pointless
            return None
        r = self.raw(folder, stem)
        if not r:
            return None
        for p in cands:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            if framing.same_photo(r[1], im) >= 0.45:
                return p, im
        return None


def favored():
    try:
        favs = json.loads(FAVS.read_text())["favorites"]
    except Exception:
        return set()
    return {core(f.get("source") or "") for f in favs} - {None}


def sfw_of(src):
    """SFW verdict for a path or a PIL image (written to a temp file first)."""
    if not isinstance(src, (str, Path)):
        tmp = REVIEW / ".sfw.jpg"
        REVIEW.mkdir(parents=True, exist_ok=True)
        src.save(tmp, quality=90)
        src = tmp
    return sfw.check(src)


def write_proposal(name, rtype, why, entries, extra, build):
    """Nothing is prepared yet: small thumbnails for the review page and enough
    to build the pack later, once Ron has cleared the model and every photo."""
    tdir = PROPOSALS / name
    tdir.mkdir(parents=True, exist_ok=True)
    files, models = [], []
    for fn, role, src, info in entries:
        im = Image.open(src).convert("RGB") if isinstance(src, (str, Path)) else src
        t = im.copy()
        t.thumbnail((1200, 1200))
        t.save(tdir / fn, quality=80)
        v = info.pop("sfw", None) or sfw.check(tdir / fn)
        srcref = str(src) if isinstance(src, (str, Path)) else info.pop("src_raw", None)
        info.pop("src_raw", None)
        m = info.get("model") or dict(extra).get("Model")
        if m and m not in models:
            models.append(m)
        files.append(dict(file=fn, role=role, src=srcref, sfw=v["verdict"],
                          reasons=v["reasons"], **info))
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    (PROPOSALS / f"{name}.json").write_text(json.dumps(dict(
        name=name, type=rtype, why=why, extra=[list(e) for e in extra], models=models,
        status="proposed", proposed_at=time.strftime("%Y-%m-%d %H:%M"),
        build=build, files=files), indent=1))
    worst = ("check" if any(f["sfw"] != "pass" for f in files) else "pass")
    # previews show the crop the pack will ship with (see instinct_preview.py)
    try:
        import instinct_preview
        instinct_preview.crop_proposal(PROPOSALS / f"{name}.json")
    except Exception as e:
        print(f"   (preview crop failed: {e})")
    print(f"-> proposal {name}  ({len(files)} photos, sfw {worst})")


def write_pack(name, rtype, why, entries, extra, stage_images=True, build=None):
    """entries: [(filename, role, PIL or path, dict info)]. info may carry an
    'sfw' verdict dict; missing verdicts are computed here from the written file."""
    if PROPOSE:
        return write_proposal(name, rtype, why, entries, extra, build)
    out = OUTBOX / name
    out.mkdir(parents=True, exist_ok=True)
    counts, review = {}, []
    lines = [f"Reel type  : {rtype}", f"Why        : {why}"] + \
            [f"{k:<11}: {v}" for k, v in extra] + [""]
    for fn, role, src, info in entries:
        if isinstance(src, (str, Path)):
            (out / fn).write_bytes(Path(src).read_bytes())
        else:
            src.save(out / fn, quality=94)
        counts[role] = counts.get(role, 0) + 1
        v = info.pop("sfw", None) or sfw.check(out / fn)
        info["sfw"] = v["verdict"] + (f" ({'; '.join(v['reasons'])})" if v["reasons"] else "")
        review.append(dict(file=fn, role=role, sfw=v["verdict"], reasons=v["reasons"],
                           **{k: w for k, w in info.items() if k != "sfw"}))
    lines.insert(2, "Contents   : " + f"{len(entries)} files — " +
                 ", ".join(f"{n} {r}" for r, n in counts.items()))
    lines.append("In order:")
    for fn, role, _, info in entries:
        lines.append(f"  {fn:34s} {role:9s} " +
                     "  ".join(f"{k}={v}" for k, v in info.items() if v not in (None, "")))
    worst = ("fail" if any(r["sfw"] == "fail" for r in review) else
             "check" if any(r["sfw"] == "check" for r in review) else "pass")
    lines.insert(3, f"SFW (tool)  : {worst} — a human confirms before posting")
    lines.insert(4, "Consent    : face_ok on every file's folder (consent_allowlist.yaml)")
    (out / "meta.txt").write_text("\n".join(lines) + "\n")
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / f"{name}.json").write_text(json.dumps(dict(
        name=name, type=rtype, why=why, extra=[list(e) for e in extra],
        sfw=worst, consent="face_ok", status="staged — awaiting instinct",
        staged_at=time.strftime("%Y-%m-%d %H:%M"), files=review), indent=1))
    print(f"-> {out}  ({len(entries)} files, sfw {worst})")
    return out


def stars(r, pick, fav):
    return f"★{int(r)}" if r else ("pick" if pick == 1 else ("fav" if fav else ""))


# ---------------------------------------------------------------- burst
def cmd_burst(a, R):
    rows = db().execute(
        """SELECT m.name, s.date, s.location FROM photos p JOIN sessions s
           ON s.id=p.session_id JOIN models m ON m.id=s.model_id
           WHERE s.folder_path LIKE ? AND p.filename LIKE ?""",
        (f"%{a.session}", a.stem + ".%")).fetchone()
    ed = R.export(a.session, a.stem, "edited")
    if not ed:
        sys.exit("no verified, consented edited export for that frame")
    raws = glob.glob(str(ARCHIVE / "*" / glob.escape(a.session)))
    label = f"burst+finished__{re.sub(r'[^A-Za-z0-9]+', '', rows[0] if rows else 'x')}_{a.stem}"
    extra0 = [("Model", rows[0] if rows else "?"), ("Shot", rows[1] if rows else "?"),
              ("Session", a.session), ("Hero", a.stem)]
    why = "a burst shot just before the photo, landing on its edited version"
    if PROPOSE:
        from build_reel_group import burst_by_time
        entries = []
        for st in burst_by_time(raws[0], a.stem, a.count, 900.0) or []:
            rr = R.raw(a.session, st)
            if not rr:
                continue
            v = sfw_of(rr[1])
            if v["verdict"] == "fail":
                continue
            entries.append((f"{len(entries) + 1:02d}_burst_{st}.jpg", "raw", rr[1],
                            {"sfw": v, "src_raw": rr[0], "stem": st}))
        if len(entries) < 6:
            sys.exit(f"REJECTED: fewer than 6 SFW burst frames ({len(entries)})")
        hero = R.raw(a.session, a.stem)
        entries.append(("80_unedited.jpg", "raw", hero[1],
                        {"sfw": sfw_of(hero[1]), "src_raw": hero[0], "stem": a.stem}))
        ve = sfw_of(ed[0])
        if ve["verdict"] == "fail":
            sys.exit(f"REJECTED: the edited photo fails SFW ({'; '.join(ve['reasons'])})")
        entries.append(("90_edited.jpg", "edited", ed[0], {"sfw": ve, "stem": a.stem}))
        return write_pack(label, "burst+finished", why, entries, extra0,
                          build=dict(cmd="burst", session=a.session, stem=a.stem,
                                     count=a.count))
    tmp = OUTBOX / ".build"
    r = subprocess.run([sys.executable, str(HERE / "build_reel_group.py"),
                        "--edited", ed[0], "--raw-dir", raws[0], "--stem", a.stem,
                        "--burst", str(a.count), "--label", label, "--outdir", str(tmp)],
                       capture_output=True, text=True)
    g = tmp / label
    if not (g / "notes.txt").exists():
        sys.exit(r.stdout[-800:] + r.stderr[-800:])
    notes = (g / "notes.txt").read_text()
    entries, dropped = [], []
    for f in sorted(g.glob("*.jpg")):
        if any(x in f.name for x in getattr(a, "exclude", ())):
            continue                         # Ron rejected this frame on the review page
        role = "edited" if f.name.startswith("90_") else "raw"
        v = sfw.check(f)
        if v["verdict"] == "fail":
            if not f.name.startswith("0"):          # the finished photo itself
                sys.exit(f"REJECTED: {f.name} fails SFW ({'; '.join(v['reasons'])})")
            dropped.append(f"{f.name}: {'; '.join(v['reasons'])}")
            continue
        entries.append((f.name, role, f, {"sfw": v}))
    if sum(1 for e in entries if e[0].startswith("0")) < 6:
        sys.exit(f"REJECTED: fewer than 6 SFW burst frames left ({len(dropped)} dropped)")
    for d in dropped:
        print("  dropped", d)
    extra = [("Model", rows[0] if rows else "?"), ("Shot", rows[1] if rows else "?"),
             ("Session", a.session), ("Hero", a.stem),
             ("Framing", re.search(r"framing\s*:\s*(.+)", notes).group(1)),
             ("Dropped", "; ".join(dropped) or "none")]
    write_pack(label, "burst+finished",
               "a burst shot just before the photo, landing on its edited version",
               entries, extra)
    import shutil
    shutil.rmtree(g, ignore_errors=True)


# ---------------------------------------------------------------- set-vibe
def cmd_setvibe(a, R):
    con = db()
    sets = []
    if a.auto:
        cand = con.execute(
            """SELECT st.id, s.folder_path, st.set_index, m.name, s.date, st.photo_count
               FROM sets st JOIN sessions s ON s.id=st.session_id
               JOIN models m ON m.id=s.model_id
               WHERE st.processed_count>=4 AND st.photo_count>=20
               ORDER BY st.starred_count DESC""").fetchall()
        use, npacks = model_usage()
        for c in cand:
            if R.allowed(c[3]):
                sets.append(c)
        # models that already have a pack go to the back of the queue
        sets.sort(key=lambda c: npacks.get(c[3].lower(), 0))
        seen_models = set()
        sets = [c for c in sets if not (c[3] in seen_models or seen_models.add(c[3]))]
    else:
        sets = con.execute(
            """SELECT st.id, s.folder_path, st.set_index, m.name, s.date, st.photo_count
               FROM sets st JOIN sessions s ON s.id=st.session_id
               JOIN models m ON m.id=s.model_id
               WHERE s.folder_path LIKE ? AND st.set_index=?""",
            (f"%{a.session}", a.set_index)).fetchall()
    fav = favored()
    made = 0
    for set_id, fpath, sidx, model, date, pcount in sets:
        folder = Path(fpath).name
        rows = con.execute(
            """SELECT filename, taken_at, lr_rating, lr_pick FROM photos
               WHERE set_id=? ORDER BY taken_at, filename""", (set_id,)).fetchall()
        photos = [(os.path.splitext(f)[0], t, r or 0, pk or 0) for f, t, r, pk in rows]
        score = lambda x: x[2] * 2 + (x[3] == 1) + (2 if x[0] in fav else 0)
        edited = []
        for p in sorted(photos, key=score, reverse=True):
            if p[0] in R.idx and (e := R.export(folder, p[0], "edited")):
                v = sfw_of(e[0])
                if v["verdict"] != "fail":
                    edited.append((p, e, "edited", v))
            if len(edited) >= a.count:
                break
        if len(edited) < 2:
            print(f"skip {folder} set {sidx}: fewer than 2 verified edited photos")
            continue
        last = edited[0]                                # the strongest closes
        first = next((e for e in edited[1:] if e[0][1] < last[0][1]), edited[1])
        chosen = [first, last]
        # frame counter, tolerating suffixed names like BLD_5932E or IMG_0390-2
        num = lambda st: int((re.findall(r"\d{3,}", st) or ["0"])[0])
        pool = [e for e in edited if e not in chosen] + \
               [(p, None, None, None) for p in sorted(photos, key=score, reverse=True)]
        for p, e, role, v in pool:
            if len(chosen) >= a.count:
                break
            if any(abs(num(p[0]) - num(c[0][0])) < 4 for c in chosen):
                continue                                # not adjacent
            if e is None:
                e, role = R.export(folder, p[0], "unedited"), "unedited"
                if not e:
                    rr = R.raw(folder, p[0])
                    e, role = (rr, "raw") if rr else (None, None)
                if not e:
                    continue
                v = sfw_of(e[0] if role != "raw" else e[1])
                if v["verdict"] == "fail":
                    continue
            chosen.append((p, e, role, v))
        mid = sorted(chosen[2:], key=lambda c: c[0][1])
        seq = [first] + mid + [last]
        entries = []
        for i, (p, e, role, v) in enumerate(seq, 1):
            src = e[0] if role != "raw" else e[1]
            info = {"shot": p[1][:19], "rating": stars(p[2], p[3], p[0] in fav), "sfw": v,
                    "stem": p[0]}
            if role == "raw":
                info["src_raw"] = e[0]
            entries.append((f"{i:02d}_{role}_{p[0]}.jpg", role, src, info))
        name = f"set-vibe__{re.sub(r'[^A-Za-z0-9]+', '', model)}_{folder[:10]}_set{sidx}"
        write_pack(name, "set-vibe",
                   "the strongest non-adjacent frames of one set; opens and closes edited",
                   entries, [("Model", model), ("Shot", date), ("Session", folder),
                             ("Set", f"(unnamed) set {sidx}, {pcount} photos")],
                   build=dict(cmd="files"))
        made += 1
        if a.auto and made >= a.auto:
            break


# ---------------------------------------------------------------- themed
TIERS = [(3, 1), (2, 1), (2, 2), (1, 2), (1, 3)]   # (min rating, max per session)


def cmd_themed(a, R):
    """Aim for a.count photos (default 14, never fewer than a.min_count=12).
    Starts strict — rated 3+, one photo per session — and relaxes one step at a
    time only while the pack is short. Work already done carries across steps:
    a photo checked once is never re-verified."""
    cond = THEMES[a.theme]
    fav = favored()
    use, npacks = model_usage()
    loved, liked, per_session, tried, per_model = [], [], {}, set(), {}
    tiers = TIERS if a.min_rating is None else [(a.min_rating, a.per_session)]
    used = None
    for min_rating, cap in tiers:
        rows = db().execute(f"""
            SELECT p.filename, p.taken_at, p.lr_rating, p.lr_pick, s.folder_path, m.name
            FROM photos p JOIN sessions s ON s.id=p.session_id JOIN models m ON m.id=s.model_id
            WHERE (p.lr_rating>={min_rating} OR p.lr_pick=1) AND {cond}
              AND NOT EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_id=p.id
                              AND t.dimension='boldness' AND t.value='explicit')
            ORDER BY p.lr_rating DESC NULLS LAST, p.lr_pick DESC""").fetchall()
        # fresh models first: rating still matters, but each pack a model is
        # already in costs it more than a whole star
        rows.sort(key=lambda r: -((r[2] or 0) * 2 + (r[3] == 1) + (2 if os.path.splitext(r[0])[0] in fav else 0)
                                  - 2.5 * npacks.get((r[5] or "").lower(), 0)))
        used = (min_rating, cap)
        for fn, t, r, pk, fpath, model in rows:
            if len(loved) >= 2 and len(loved) + len(liked) >= a.count:
                break
            stem, folder = os.path.splitext(fn)[0], Path(fpath).name
            if (folder, stem) in tried or stem not in R.idx:
                continue
            if per_session.get(folder, 0) >= cap or not R.allowed(model):
                continue
            if per_model.get(model, 0) >= a.per_model:
                continue
            tried.add((folder, stem))
            e = R.export(folder, stem, "edited")
            if not e:
                continue
            if a.theme in ("dark", "bright"):
                lum = sum(Image.open(e[0]).convert("L").resize((64, 64)).getdata()) / 4096 / 255
                if (a.theme == "dark") != (lum < 0.30) or (a.theme == "bright" and lum < 0.60):
                    continue
            v = sfw_of(e[0])
            if v["verdict"] == "fail":
                continue
            item = (stem, t or "", r or 0, pk or 0, folder, model, e[0], v)
            (loved if (r or 0) >= 5 or stem in fav else liked).append(item)
            per_session[folder] = per_session.get(folder, 0) + 1
            per_model[model] = per_model.get(model, 0) + 1
        total = len(loved) + len(liked)
        print(f"  tier rated {min_rating}+, {cap}/session: {total} photos")
        if len(loved) >= 2 and total >= a.count:
            break
    total = len(loved) + len(liked)
    if total < 6:
        sys.exit(f"theme {a.theme}: not enough verified photos ({total})")
    notes = []
    if len(loved) < 2:
        loved, liked = (loved + liked)[:2], (loved + liked)[2:]
        notes.append("fewer than 2 rated-5/faved photos matched, so it opens and closes on the next best")
    if used != TIERS[0]:
        notes.append(f"relaxed to rated {used[0]}+ and up to {used[1]} per session to reach the count")
    if total < a.min_count:
        notes.append(f"only {total} photos qualify — short of {a.min_count}")
    # the two most-loved bookend it; any further loved photos join the middle
    middle = (loved[2:] + liked)[: a.count - 2]
    seq = [loved[0]] + middle + [loved[1]]
    entries = [(f"{i:02d}_edited_{x[0]}.jpg", "edited", x[6],
                {"model": x[5], "session": x[4], "shot": x[1][:10],
                 "rating": stars(x[2], x[3], x[0] in fav), "sfw": x[7]})
               for i, x in enumerate(seq, 1)]
    write_pack(f"themed-photos__{a.theme}" + (f"-{a.variant}" if a.variant else ""), "themed-photos",
               f"edited photos from different sessions, theme '{a.theme}'; "
               f"first and last are the most-loved" + (f" ({'; '.join(notes)})" if notes else ""),
               entries, [("Theme", a.theme), ("Photos", len(seq)),
                         ("Sessions", len({x[4] for x in seq})),
                         ("Models", len({x[5] for x in seq}))],
               build=dict(cmd="files"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true",
                    help="choose and pre-filter only: write thumbnails and a proposal "
                         "for the review page; build later with instinct_build.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("burst"); b.add_argument("--session", required=True)
    b.add_argument("--stem", required=True); b.add_argument("--count", type=int, default=10)
    b.set_defaults(fn=cmd_burst)
    s = sub.add_parser("setvibe"); s.add_argument("--session"); s.add_argument("--set-index", type=int)
    s.add_argument("--auto", type=int, default=0); s.add_argument("--count", type=int, default=8)
    s.set_defaults(fn=cmd_setvibe)
    t = sub.add_parser("themed"); t.add_argument("--theme", required=True, choices=sorted(THEMES))
    t.add_argument("--count", type=int, default=14, help="target photos in the pack")
    t.add_argument("--min-count", type=int, default=12,
                   help="relax the bar until at least this many qualify")
    t.add_argument("--per-model", type=int, default=2, help="most photos from one model")
    t.add_argument("--variant", help="suffix for a second pack of the same theme")
    t.add_argument("--min-rating", type=int, default=None,
                   help="pin one bar instead of relaxing (picks always count)")
    t.add_argument("--per-session", type=int, default=1,
                   help="with --min-rating: most photos from one session")
    t.set_defaults(fn=cmd_themed)
    args = ap.parse_args()
    PROPOSE = args.propose
    args.fn(args, Resolver())
