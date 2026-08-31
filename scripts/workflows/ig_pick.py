#!/usr/bin/env python3
"""Choose a session to make IG reel candidates from, and ask before spending on it.

Scores every session in the archive that passes the consent gate, proposes the best
one by pushing the actual final frame to the phone, and renders once approved. The
picture is the proposal — a filename is not something anyone can judge.

  ig_pick.py --propose              # score, pick, push, write a pending file
  ig_pick.py --render               # render the pending proposal
  ig_pick.py --propose --auto       # skip the asking (once it has earned trust)
"""
import argparse, json, re, subprocess, sqlite3, sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ig_reel as R
import ig_meta

PENDING = R.OUT_ROOT / "_pending.json"
RECENT = 3                      # do not repeat a model within this many candidates


def recent_models(n=RECENT):
    """Models already used, newest first — so a run does not keep picking the same one."""
    if not ig_meta.LOG.exists():
        return []
    import csv
    rows = list(csv.DictReader(ig_meta.LOG.open(encoding="utf-8")))
    seen, out = set(), []
    for r in reversed(rows):
        m = r.get("model")
        if m and m not in seen:
            seen.add(m); out.append(m)
        if len(out) >= n:
            break
    return out


def score_sessions(limit=40, skip_models=()):
    """Rank sessions by how much usable material they hold.

    Wants: a decent number of finals to choose from, blue-labelled frames (the
    photographer's own keeper marks), and BTS footage, which unlocks format D.
    """
    con = sqlite3.connect(str(R.CATALOG))
    rows = con.execute(
        """SELECT s.folder_path, s.photo_count, m.name
           FROM sessions s JOIN models m ON m.id = s.model_id
           WHERE s.photo_count > 40""").fetchall()
    con.close()
    out = []
    for folder, count, model in rows:
        if model in skip_models:
            continue
        name = Path(folder.replace("\\", "/")).name
        hits = sorted(R.ARCHIVE.glob(f"*/{name}"))
        if not hits:
            continue                                   # catalogued but not on this disk
        sd = hits[0]
        ok, verdict = R.consent_ok(R.model_from_session(sd.name))
        if not ok:
            continue
        finals = R.find_finals(sd)
        if not finals:
            continue
        blue = R.blue_labelled(sd)
        has_bts = any(d.is_dir() and re.search(r"video|bts|from my phone", d.name, re.I)
                      for d in sd.iterdir())
        score = min(len(finals), 30) + 2 * min(len(blue), 20) + (15 if has_bts else 0)
        out.append({"session": sd.name, "dir": str(sd), "model": model, "frames": count,
                    "finals": len(finals), "blue": len(blue), "bts": has_bts,
                    "score": score, "consent": verdict})
        if len(out) >= limit:
            break
    return sorted(out, key=lambda d: -d["score"])


def best_final(sd):
    """The final least likely to be unusable: fewest skin pixels, one face."""
    finals = R.find_finals(sd)
    blue = R.blue_labelled(sd)
    ranked = sorted(finals.items(), key=lambda kv: (kv[0] not in blue, kv[0]))
    tmp = R.OUT_ROOT / "_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    best = None
    for stem, path in ranked[:12]:
        try:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            im.thumbnail((900, 900))
            probe = tmp / "probe.jpg"
            im.save(probe, quality=88)
            ratio, note = ig_meta.sfw_flags(probe)
            faces = ig_meta.face_count(probe)
        except Exception:
            continue
        cand = {"stem": stem, "path": str(path), "skin": ratio, "faces": faces, "sfw": note}
        if faces == 1 and ratio < 0.30:
            return cand                                # clean enough, stop looking
        # Rank on face count FIRST. Sorting by skin alone picks the safest-looking
        # frame, which is often a detail shot with no one recognisable in it — safe,
        # and useless as the payoff of a reel.
        key = (0 if cand["faces"] == 1 else 1, cand["skin"])
        if best is None or key < (0 if best["faces"] == 1 else 1, best["skin"]):
            best = cand
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--propose", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--auto", action="store_true", help="render without asking first")
    p.add_argument("--formats", default="A,B", help="comma list: A,B,D")
    args = p.parse_args()

    if args.render and PENDING.exists():
        pend = json.loads(PENDING.read_text())
        run_formats(Path(pend["dir"]), pend["final"], pend["formats"].split(","))
        PENDING.unlink()
        return

    skip = recent_models()
    print(f"skipping recently used: {skip or 'nothing yet'}")
    ranked = score_sessions(skip_models=skip)
    if not ranked:
        sys.exit("no eligible session found")
    top = ranked[0]
    sd = Path(top["dir"])
    print(f"\npicked  : {top['session']}\nmodel   : {top['model']} ({top['consent']})\n"
          f"material: {top['finals']} finals, {top['blue']} blue, "
          f"{'BTS available' if top['bts'] else 'no BTS'}, {top['frames']} frames")
    cand = best_final(sd)
    if not cand:
        sys.exit("no usable final in that session")
    print(f"final   : {cand['stem']}  skin={cand['skin']:.2f} faces={cand['faces']}\n"
          f"          {cand['sfw']}")

    formats = args.formats if top["bts"] else args.formats.replace(",D", "").replace("D,", "")
    if args.auto:
        run_formats(sd, cand["path"], formats.split(","))
        return

    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps({"dir": str(sd), "final": cand["path"],
                                   "formats": formats, "proposed_at": datetime.now().isoformat(),
                                   "skin": cand["skin"], "faces": cand["faces"]}, indent=2))
    ig_meta.propose(f"IG reel candidate — {top['model']}",
                    f"{cand['stem']} · {cand['sfw']} · formats {formats}. "
                    f"Run --render to make it.", cand["path"])
    print(f"\nproposed. confirm, then: ig_pick.py --render")


def run_formats(sd, final_path, formats):
    for f in formats:
        f = f.strip().upper()
        if not f:
            continue
        script = "ig_bts.py" if f == "D" else "ig_reel.py"
        cmd = [sys.executable, str(HERE / script), "--session", sd.name,
               "--final", Path(final_path).name]
        if f != "D":
            cmd += ["--format", f, "--raw-mode", "flat"]
        print(f"\n--- format {f}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        for line in (r.stdout or "").splitlines():
            if "→" in line or "segments" in line or "final " in line:
                print(line)
        if r.returncode != 0:
            print((r.stderr or "").strip()[-300:])


if __name__ == "__main__":
    main()
