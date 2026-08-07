#!/usr/bin/env python3
"""Phase 1 acceptance test (library-level, no HTTP).

Runs a 3-step preview graph (color-bath → time-corruption → ink-dissolution)
on a photo, twice — the second pass must be all cache hits. Then edits step 2's
params and verifies step 1 still hits while steps 2-3 recompute.

Usage:  ~/openclaw-venv/bin/python3 -m studio.smoke_test [--source PHOTO]
"""
import argparse
import os
import sys
import time
from pathlib import Path

from . import runner
from .graph import Session

CANDIDATES = Path(os.path.expanduser("~/.openclaw/workspace/shared/candidates"))

STEPS = [
    ("color-bath", {"preset": "red-film", "strength": 0.7}, 1),
    ("time-corruption", {"preset": "ghost", "intensity": 0.6}, 2),
    ("ink-dissolution", {"preset": "ink-wash"}, 3),
]


def show(label, results, t):
    hits = sum(r["cache_hit"] for r in results)
    print(f"\n{label}  ({t:.1f}s, {hits}/{len(results)} cache hits)")
    for r in results:
        mark = "HIT " if r["cache_hit"] else "RUN "
        extra = "" if r["cache_hit"] else f"({r.get('wall_time', 0)}s)"
        print(f"  {mark} {r['tool']:16s} out={r['output'][:12]} {extra}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="photo path (default: first candidate)")
    args = ap.parse_args()

    src = Path(args.source) if args.source else next(
        iter(sorted(CANDIDATES.glob("*.jp*g"))), None)
    if not src or not src.exists():
        sys.exit("no source photo found — pass --source")
    print(f"source: {src}")

    from . import cache
    session = Session.create(str(src), cache.put_file(src))
    for tool, params, seed in STEPS:
        session.add_step(tool, params, seed=seed, preview=True)

    t0 = time.time(); r1 = runner.evaluate(session); t1 = time.time() - t0
    show("PASS 1 (cold)", r1, t1)

    t0 = time.time(); r2 = runner.evaluate(session); t2 = time.time() - t0
    hits2 = show("PASS 2 (warm)", r2, t2)
    assert hits2 == len(STEPS), f"expected {len(STEPS)} hits, got {hits2}"

    node2 = session.chain()[1]["id"]
    session.edit_step(node2, params={"preset": "ghost", "intensity": 0.9})
    t0 = time.time(); r3 = runner.evaluate(session); t3 = time.time() - t0
    hits3 = show("PASS 3 (edited step 2 → descendants recompute)", r3, t3)
    assert r3[0]["cache_hit"], "step 1 should still hit"
    assert not r3[1]["cache_hit"] and not r3[2]["cache_hit"], \
        "steps 2-3 should recompute"

    print(f"\nOK — session {session.data['id']}: cold {t1:.0f}s, "
          f"warm {t2:.1f}s, edit {t3:.0f}s")


if __name__ == "__main__":
    main()
