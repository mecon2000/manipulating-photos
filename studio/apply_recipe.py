#!/usr/bin/env python3
"""Batch recipe apply — runs as a normal hub job (manifest action), so batches
show in the hub Jobs tab and ntfy like everything else.

For each --source: creates a Studio session, appends the recipe's general
steps, evaluates in preview mode, and drops the result jpg into both the hub
job dir (--local-output-dir) and shared/studio/batches/<batch-id>/ with a
batch.json manifest. Tapping a result in the Studio contact sheet opens the
pre-loaded session (already created here) for delta chat.

Usage:
  studio/apply_recipe.py --recipe <slug> --source A.jpg B.jpg [--local-output-dir DIR]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import cache, recipes, runner            # noqa: E402
from studio.graph import Session                     # noqa: E402
from studio.paths import SHARED                      # noqa: E402

BATCHES_DIR = SHARED / "studio" / "batches"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--local-output-dir", default="")
    args = ap.parse_args()

    recipe = recipes.get(args.recipe)
    if recipe is None:
        sys.exit(f"no recipe {args.recipe!r}")

    batch_id = f"{args.recipe}-{time.strftime('%Y%m%d-%H%M%S')}"
    batch_dir = BATCHES_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(args.local_output_dir) if args.local_output_dir else None

    entries, failures = [], 0
    for src in args.source:
        src = Path(src)
        print(f"[apply] {recipe['name']!r} on {src.name}", flush=True)
        try:
            s = Session.create(str(src), cache.put_file(src))
            recipes.apply_to_session(s, recipe)
            results = runner.evaluate(s)
            out_path = cache.object_path(results[-1]["output"])
            dest = batch_dir / f"{src.stem}{out_path.suffix.lower()}"
            shutil.copyfile(out_path, dest)
            if job_dir:
                shutil.copyfile(out_path, job_dir / dest.name)
            entries.append({"source": str(src), "session": s.data["id"],
                            "output": dest.name, "ok": True})
            print(f"  -> {dest}", flush=True)
        except Exception as e:                        # keep batch going
            failures += 1
            entries.append({"source": str(src), "ok": False, "error": str(e)})
            print(f"  FAILED: {e}", flush=True)

    (batch_dir / "batch.json").write_text(json.dumps({
        "id": batch_id, "recipe": args.recipe, "recipe_name": recipe["name"],
        "created": time.time(), "entries": entries}, indent=2))
    print(f"batch {batch_id}: {len(entries) - failures}/{len(entries)} ok",
          flush=True)
    sys.exit(1 if failures == len(entries) else 0)


if __name__ == "__main__":
    main()
