#!/usr/bin/env python3
"""Weekly housekeeping report for the shared content tree.

Reports; it does not delete. Only files that are provably reproducible — pipeline
intermediates whose recipe is recorded in the sidecar beside them — get moved, and
they move to the project's trash/, never rm. Everything else is listed for Ron to
decide on, because "large and old" is not the same as "unwanted": vids-to-censor
sat at 16GB for two months and still held the one asset worth keeping.

  tidy_shared.py            # report only
  tidy_shared.py --sweep    # also move clear-cut intermediates to trash/
"""
import argparse, os, shutil, time
from pathlib import Path

SHARED = Path("~/.openclaw/workspace/shared").expanduser()
TRASH = SHARED / "faces-candidates" / "trash"

# Reproducible pipeline leavings: regenerating them costs no API call, because the
# generation output (__surreal.jpg) and the recipe (__final.json) both survive.
SWEEPABLE = ["**/*__src.jpg", "**/*__pre_upscale_resized.jpg", "**/*__bw_relit.jpg"]
STALE_DAYS = 30
BIG_MB = 500


def human(n):
    for u in ("B", "K", "M", "G"):
        if n < 1024 or u == "G":
            return f"{n:.0f}{u}" if u in ("B", "K") else f"{n:.1f}{u}"
        n /= 1024


def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="move reproducible intermediates to trash/ (still never rm)")
    ap.add_argument("--notify", action="store_true", help="push the summary via ntfy")
    args = ap.parse_args()

    lines, moved, freed = [], 0, 0

    # 1 — reproducible intermediates
    hits = []
    for pat in SWEEPABLE:
        hits += [p for p in SHARED.glob(pat) if p.is_file() and TRASH not in p.parents]
    if hits:
        size = sum(p.stat().st_size for p in hits)
        lines.append(f"reproducible intermediates: {len(hits)} files, {human(size)}")
        if args.sweep:
            TRASH.mkdir(parents=True, exist_ok=True)
            for p in hits:
                try:
                    dest = TRASH / p.name
                    i = 1
                    while dest.exists():
                        dest = TRASH / f"{p.stem}__{i}{p.suffix}"; i += 1
                    shutil.move(str(p), str(dest))
                    moved += 1; freed += 1
                except OSError as e:
                    lines.append(f"  could not move {p.name}: {e}")
            lines.append(f"  → moved {moved} to trash/")

    # 2 — loose files sitting at the shared root
    loose = [p for p in SHARED.iterdir() if p.is_file()]
    if loose:
        lines.append(f"loose files at shared root: {len(loose)}, "
                     f"{human(sum(p.stat().st_size for p in loose))} — should live in a subfolder")

    # 3 — big or stale directories, reported only
    cutoff = time.time() - STALE_DAYS * 86400
    for d in sorted(SHARED.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        size = dir_size(d)
        if size < BIG_MB * 1024 * 1024:
            continue
        newest = 0
        for root, _, files in os.walk(d):
            for f in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
        age = (time.time() - newest) / 86400 if newest else 999
        flag = "  STALE" if newest and newest < cutoff else ""
        lines.append(f"{d.name}: {human(size)}, untouched {age:.0f}d{flag}")

    # 4 — trash that has been sitting long enough to be reviewed
    for t in SHARED.glob("**/trash"):
        size = dir_size(t)
        if size > 100 * 1024 * 1024:
            lines.append(f"{t.relative_to(SHARED)}: {human(size)} — review and empty")

    report = "shared tidy — " + time.strftime("%Y-%m-%d") + "\n" + "\n".join("  " + l for l in lines)
    print(report)
    if args.notify:
        try:
            import subprocess
            subprocess.run([os.path.expanduser("~/bin/ntfy"), "publish", "hub",
                            report[:1500]], check=False, timeout=20)
        except Exception as e:
            print(f"  (no notify: {e})")
    return report


if __name__ == "__main__":
    main()
