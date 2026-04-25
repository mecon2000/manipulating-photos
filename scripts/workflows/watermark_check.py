#!/usr/bin/env python3
"""Warn if any source photo's filename suggests a burned-in 'UNPROCESSED' watermark.

LR exports unprocessed/work-in-progress JPEGs typically include 'UNPROCESSED'
in the filename (Ronnie's catalog convention). When such a JPEG passes through
style-transfer, the watermark text leaks into the output.

This script doesn't try to clean the watermark — Ronnie re-exports clean from
LR himself. It just lists which inputs would taint the output so he can fix
them before kicking off a batch.

Usage:
  watermark_check.py [DIR ...]                # check one or more dirs
  watermark_check.py FILE.jpg                 # check one file
  watermark_check.py --json [DIR]             # machine-readable output
"""
import argparse, json, os, sys
from pathlib import Path

PATTERNS = ("unprocessed", "do not publish", "do_not_publish", "donotpublish",
            "wip", "watermark")


def is_suspect(name):
    low = name.lower()
    return any(p in low for p in PATTERNS)


def scan(target):
    p = Path(target).expanduser()
    suspects, clean = [], []
    if p.is_file():
        (suspects if is_suspect(p.name) else clean).append(str(p))
        return suspects, clean
    if not p.is_dir():
        return [], []
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            (suspects if is_suspect(f.name) else clean).append(str(f))
    return suspects, clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*",
                   default=["~/.openclaw/workspace/shared/candidates-for-motion-streak"])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    all_suspect, all_clean = [], []
    for t in args.targets:
        s, c = scan(t)
        all_suspect.extend(s)
        all_clean.extend(c)

    if args.json:
        print(json.dumps({"suspect": sorted(all_suspect),
                          "clean":   sorted(all_clean)}, indent=2))
        return

    if all_suspect:
        print(f"⚠️  {len(all_suspect)} file(s) likely have UNPROCESSED watermark — re-export from LR before batch:")
        for f in sorted(all_suspect):
            print(f"  {f}")
    else:
        print(f"✅ no suspect filenames in {len(all_clean)} file(s)")


if __name__ == "__main__":
    main()
