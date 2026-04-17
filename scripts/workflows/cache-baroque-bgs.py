#!/home/rong/openclaw-venv/bin/python3
"""
Pre-generate a pool of baroque BGs for reuse by baroque-surround.py --use-cached-bg.

Generates Flux Schnell BGs across preset+artifact combinations at two aspect ratios
(portrait 2:3 and landscape 3:2). Saves to ~/.openclaw/workspace/shared/bg_cache/
with an index.json listing preset/artifact/aspect metadata.

Usage:
  ./cache-baroque-bgs.py                      # default: 3 per combo, favorites only
  ./cache-baroque-bgs.py --per-combo 5        # more variety
  ./cache-baroque-bgs.py --presets ink-water,silk --artifacts butterflies,petals
  ./cache-baroque-bgs.py --all-presets --all-artifacts  # large run
"""
import os, sys, json, time, random, argparse
from io import BytesIO
from pathlib import Path

_env = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env):
    for ln in open(_env):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Import PRESETS / ARTIFACTS from baroque-surround.py (module name has a dash, so use importlib)
import importlib.util
_spec = importlib.util.spec_from_file_location("baroque_surround",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "baroque-surround.py"))
_bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bm)
PRESETS = _bm.PRESETS
ARTIFACTS = _bm.ARTIFACTS

CACHE_DIR = Path(os.path.expanduser("~/.openclaw/workspace/shared/bg_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = CACHE_DIR / "index.json"

# Favorite combos (from feedback_baroque_combos memory)
FAV_PRESETS = ["ink-water", "silk", "aurora", "curtains", "ethereal", "dark-romantic"]
FAV_ARTIFACTS = ["butterflies", "ribbons", "petals", "wings", "flames", "feathers"]

# Aspect presets (w, h) at 1024 long edge
ASPECTS = {
    "portrait": (680, 1024),   # ~2:3
    "landscape": (1024, 680),  # ~3:2
    "square":   (1024, 1024),
}

def load_index():
    if INDEX_PATH.exists():
        try: return json.loads(INDEX_PATH.read_text())
        except: return []
    return []

def save_index(entries):
    INDEX_PATH.write_text(json.dumps(entries, indent=2))

def gen_one(prompt, w, h, seed):
    handle = fal_client.submit("fal-ai/flux/schnell", arguments={
        "prompt": prompt,
        "image_size": {"width": w, "height": h},
        "num_inference_steps": 4,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
        "seed": seed,
    })
    res = handle.get()
    url = res["images"][0]["url"]
    r = requests.get(url, timeout=60); r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def build_prompt(preset_name, artifact_name):
    p = PRESETS[preset_name]["prompt"]
    if artifact_name and artifact_name != "none":
        p += f", {ARTIFACTS[artifact_name]}"
    p += ", no central focal point, scattered elements around edges"
    p += ", NO person, NO figure, NO human face, just abstract painterly forms"
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-combo", type=int, default=3, help="BGs per (preset,artifact,aspect) triple")
    ap.add_argument("--presets", default=None, help="Comma-separated presets (default: favorites)")
    ap.add_argument("--artifacts", default=None, help="Comma-separated artifacts (default: favorites)")
    ap.add_argument("--all-presets", action="store_true")
    ap.add_argument("--all-artifacts", action="store_true")
    ap.add_argument("--aspects", default="portrait,landscape", help="Comma-separated: portrait,landscape,square")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all_presets: presets = list(PRESETS.keys())
    elif args.presets:   presets = [p.strip() for p in args.presets.split(",")]
    else:                presets = FAV_PRESETS

    if args.all_artifacts: artifacts = list(ARTIFACTS.keys())
    elif args.artifacts:   artifacts = [a.strip() for a in args.artifacts.split(",")]
    else:                  artifacts = FAV_ARTIFACTS

    aspects = [a.strip() for a in args.aspects.split(",")]
    aspects = [a for a in aspects if a in ASPECTS]

    total = len(presets) * len(artifacts) * len(aspects) * args.per_combo
    est_cost = total * 0.003
    print(f"Will generate {total} BGs  (presets={len(presets)} × artifacts={len(artifacts)} × aspects={len(aspects)} × {args.per_combo})")
    print(f"Estimated cost: ${est_cost:.2f}  (@ $0.003/image, Flux Schnell)")
    if args.dry_run:
        return
    resp = input("Proceed? [y/N] ").strip().lower()
    if resp != "y":
        print("Cancelled."); return

    entries = load_index()
    done = 0
    for preset in presets:
        if preset not in PRESETS:
            print(f"  skip unknown preset: {preset}"); continue
        for artifact in artifacts:
            if artifact not in ARTIFACTS:
                print(f"  skip unknown artifact: {artifact}"); continue
            prompt = build_prompt(preset, artifact)
            for aspect in aspects:
                w, h = ASPECTS[aspect]
                aspect_ratio = w / h
                for i in range(args.per_combo):
                    seed = random.randint(0, 2**32 - 1)
                    try:
                        t0 = time.time()
                        img = gen_one(prompt, w, h, seed)
                        fname = f"{preset}__{artifact}__{aspect}__{seed}.jpg"
                        fpath = CACHE_DIR / fname
                        img.save(fpath, "JPEG", quality=92)
                        entries.append({
                            "file": fname, "preset": preset, "artifact": artifact,
                            "aspect": aspect_ratio, "aspect_name": aspect,
                            "w": w, "h": h, "seed": seed,
                        })
                        save_index(entries)  # save after each in case of interrupt
                        done += 1
                        print(f"  [{done}/{total}] {fname}  ({time.time()-t0:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL {preset}/{artifact}/{aspect}: {e}")
    print(f"\nDone. {done} BGs cached at {CACHE_DIR}")
    print(f"Total in index: {len(entries)}")

if __name__ == "__main__":
    main()
