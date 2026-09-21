#!/usr/bin/env python3
"""End-to-end run: choose models, build packs of every reel type, stage them.

  instinct_run.py --bursts 4 --setvibes 3 --themes dark,bright,bw,outdoor   (add --fresh to retire open proposals)

1. Clears the outbox (to trash — never deleted) except files that aren't packs.
2. burst+finished: finds heroes itself — a rated/picked photo whose SET holds
   at least 8 frames shot before it, from a face_ok model. One model per pack.
3. set-vibe and themed-photos via instinct_pack.
Every pack is consent-gated per folder, SFW-filtered per file, and leaves a
review record in send-to-instinct-review/ for the human pass.
"""
import argparse, json, os, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/.claude/skills/candidates"))
import consent as C
from instinct_pack import OUTBOX, REVIEW, PROPOSALS, CATALOG, photo_index

PY = sys.executable
SHARED = Path.home() / ".openclaw/workspace/shared"


def clear_outbox():
    """Retire the previous round of PROPOSALS (to trash). Built packs already at
    shared are left alone — they are waiting for instinct."""
    trash = SHARED / "trash" / f"instinct-proposals-{int(time.time())}"
    moved = 0
    for d in list(PROPOSALS.glob("*")):
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(trash / d.name))
        moved += 1
    print(f"cleared {moved} old packs -> {trash}" if moved else "outbox empty")


def burst_heroes(limit):
    models, aliases = C.load()
    from instinct_pack import excluded_models
    excluded = excluded_models()
    idx = photo_index()
    db = sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)
    rows = db.execute("""
        SELECT p.filename, s.folder_path, m.name,
               (SELECT COUNT(*) FROM photos q WHERE q.set_id=p.set_id
                  AND q.taken_at < p.taken_at) AS before
        FROM photos p JOIN sessions s ON s.id=p.session_id JOIN models m ON m.id=s.model_id
        WHERE (p.lr_rating>=3 OR p.lr_pick=1) AND p.set_id IS NOT NULL
        ORDER BY p.lr_rating DESC NULLS LAST, p.taken_at DESC""").fetchall()
    seen, out = set(), []
    for fn, fpath, model, before in rows:
        stem = os.path.splitext(fn)[0]
        if before < 8 or model in seen or stem not in idx:
            continue
        if not any("unprocess" not in p.lower() for p in idx[stem]):
            continue
        if C.decide(model, models, aliases)[0] != "face_ok" or model.lower() in excluded:
            continue
        seen.add(model)
        out.append((Path(fpath).name, stem, model))
        if len(out) >= limit:
            break
    return out


def run(cmd):
    r = subprocess.run([PY, str(HERE / "instinct_pack.py"), "--propose"] + cmd,
                       capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if l.startswith(("->", "REJECTED", "  dropped", "skip", "no verified",
                              "theme", "not enough"))]
    for l in lines:
        print("   ", l)
    return any(l.startswith("->") for l in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", type=int, default=4)
    ap.add_argument("--setvibes", type=int, default=3)
    ap.add_argument("--themes", default="dark,bright,bw,outdoor")
    ap.add_argument("--fresh", action="store_true", help="retire all open proposals to trash first")
    a = ap.parse_args()
    if a.fresh:
        clear_outbox()

    made = 0
    print(f"== burst+finished (want {a.bursts})")
    for folder, stem, model in burst_heroes(a.bursts * 4):
        if made >= a.bursts:
            break
        print(f"  {model} {stem} ({folder})")
        made += run(["burst", "--session", folder, "--stem", stem])

    print(f"== set-vibe (want {a.setvibes})")
    run(["setvibe", "--auto", str(a.setvibes)])

    for t in [t for t in a.themes.split(",") if t]:
        print(f"== themed-photos: {t}")
        run(["themed", "--theme", t])

    props = [json.loads(p.read_text()) for p in sorted(PROPOSALS.glob("*.json"))]
    print(f"\n{len(props)} proposals awaiting your OK:")
    for p in props:
        flagged = sum(1 for f in p["files"] if f["sfw"] != "pass")
        print(f"  {p['name']}  ({len(p['files'])} photos, {flagged} flagged)")


if __name__ == "__main__":
    main()
