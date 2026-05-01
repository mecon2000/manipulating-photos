#!/usr/bin/env python3
"""Group favorites into IG-post folders by theme.

Reads `~/.openclaw/workspace/shared/favorites/favorites.json`, applies a set
of theme rules (tool / preset / model based), and writes one subfolder per
theme to `~/.openclaw/workspace/shared/ig-groups/<NN-slug>/` with a
`manifest.json` describing the theme blurb and the photos picked. Files are
copied (not moved); favorites are untouched.

Idempotent: each run wipes the output dir and rebuilds. Safe to rerun after
adding new favorites.

Picking is greedy with diversity caps: round-robin across models so a single
model doesn't dominate, and per-style caps to keep the visual mix varied.
Spotlight groups (single model) relax the model cap.

Usage:
    ./scripts/workflows/build_ig_groups.py
    ./scripts/workflows/build_ig_groups.py --out /tmp/test-groups
    ./scripts/workflows/build_ig_groups.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

FAV = Path("~/.openclaw/workspace/shared/favorites").expanduser()
OUT_DEFAULT = Path("~/.openclaw/workspace/shared/ig-groups").expanduser()


def style_of(f):
    return (f.get("style") or
            (f.get("knobs") or {}).get("lighting") or
            (f.get("knobs") or {}).get("preset") or
            (f.get("knobs") or {}).get("medium") or
            (f.get("knobs") or {}).get("material") or
            f.get("preset") or f.get("medium") or "")


def model_of(f):
    return f.get("model") or "?"


def pick(matches, k, max_per_model=2, max_per_style=3):
    """Greedy round-robin diverse pick."""
    by_model = defaultdict(list)
    for f in matches:
        by_model[model_of(f)].append(f)
    picked, mused, sused = [], defaultdict(int), defaultdict(int)
    while len(picked) < k:
        added = False
        for m in sorted(by_model.keys(), key=lambda x: mused[x]):
            for f in by_model[m]:
                if f in picked:
                    continue
                if mused[m] >= max_per_model:
                    continue
                if sused[style_of(f)] >= max_per_style:
                    continue
                picked.append(f)
                mused[m] += 1
                sused[style_of(f)] += 1
                added = True
                if len(picked) >= k:
                    break
            if len(picked) >= k:
                break
        if not added:
            break
    return picked


def write_group(out_dir: Path, name: str, picks, blurb: str, dry: bool):
    if not picks:
        print(f"  [skip] {name}")
        return
    if dry:
        print(f"  [dry] {name}: {len(picks)}")
        return
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"theme": name, "blurb": blurb, "photos": []}
    for i, f in enumerate(picks, 1):
        src = FAV / f["file"]
        dst = d / f"{i:02d}__{f['file']}"
        # copyfile (no chmod) — shared/ is sometimes a 9p mount where
        # copy/copy2 fail with EPERM on metadata.
        shutil.copyfile(src, dst)
        manifest["photos"].append({
            "n": i, "file": dst.name, "src": f["file"],
            "model": model_of(f), "style": style_of(f),
            "tool": f.get("tool", ""),
        })
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  ✔ {name}: {len(picks)}")


def build_groups(favs, out_dir: Path, dry: bool):
    def of_tool(t):
        return [f for f in favs if f.get("tool", "").startswith(t)]

    def style_in(styles):
        s = set(styles)
        return [f for f in favs if style_of(f) in s]

    def model_eq(m):
        return [f for f in favs if model_of(f) == m]

    write_group(out_dir, "01-ink-dissolved",
                pick(of_tool("ink-dissolution"), 9, max_per_style=3),
                "When the body forgets where it ends — ink, watercolor, charcoal dissolutions of the figure.",
                dry)
    write_group(out_dir, "02-baroque-dark",
                pick([f for f in of_tool("baroque-surround") if style_of(f) in
                      {"smoke", "dark-romantic", "ethereal", "velvet-fog",
                       "embers", "neon-smoke-rings", "burning-silk", "torn-cloud"}], 9),
                "Renaissance, but moody. Subjects floating in painted smoke, embers, velvet fog.",
                dry)
    write_group(out_dir, "03-baroque-flow",
                pick([f for f in of_tool("baroque-surround") if style_of(f) in
                      {"silk", "ink-water", "aurora", "curtains", "bubbles",
                       "spun-sugar", "whipped-cream", "powdered-pigment",
                       "silk + petals", "aurora + ribbons", "ink-water + ribbons",
                       "ink-water + butterflies"}], 9),
                "Painted in motion. Silk, ink water, ribbons of light around the figure.",
                dry)
    write_group(out_dir, "04-red-fever",
                pick([f for f in favs if style_of(f) in {"red-film", "Red Drama", "rose", "amber"}
                      or "red" in f.get("file", "").lower()], 9),
                "All-red feed slot. Film stock, neon, gel, ember.",
                dry)
    write_group(out_dir, "05-soft-window",
                pick(style_in(["Window Light", "Golden Hour", "Overcast Soft",
                               "Beauty Dish", "Candlelight", "High Key"]), 9),
                "Light through a window, light through skin. Soft directional naturalism.",
                dry)
    write_group(out_dir, "06-colored-gels",
                pick(style_in(["Neon Gels", "Teal & Orange", "Neon Signs", "Laser",
                               "Stained Glass", "TV Glow", "Stage Backlight",
                               "Lightning Flash", "Spotlight", "Dramatic Rim"]), 9),
                "Color thrown by lights. Gels, neons, lasers.",
                dry)
    write_group(out_dir, "07-time-broken",
                pick([f for f in favs if "time-corruption" in f.get("tool", "")
                      or any(k in f.get("file", "").lower()
                             for k in ["glitch", "melt", "ghost"])], 8),
                "Long exposure of a body refusing to settle.",
                dry)
    write_group(out_dir, "08-material-world",
                pick(of_tool("material-swap"), 9),
                "Ice, glass, marble, gold — same body, nine bodies.",
                dry)
    write_group(out_dir, "09-spotlight-noie",
                pick(model_eq("Noie Macklin"), 9, max_per_model=99, max_per_style=2),
                "Noie Macklin — across nine treatments.",
                dry)
    write_group(out_dir, "10-spotlight-roni",
                pick(model_eq("Roni Frid"), 8, max_per_model=99, max_per_style=2),
                "Roni Frid — eight different lights.",
                dry)
    write_group(out_dir, "11-color-bath",
                pick(of_tool("color-bath"), 9, max_per_style=2),
                "Single-color rooms. LAB color washes pulling whole scenes into one note.",
                dry)
    write_group(out_dir, "12-style-imitations",
                pick([f for f in favs if "surreal" in f.get("tool", "")
                      or "decorated" in f.get("tool", "")
                      or "surreal-with-face" in f.get("file", "")], 9),
                "Identity preserved through reference-driven generative pipelines.",
                dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    favs_path = FAV / "favorites.json"
    if not favs_path.is_file():
        raise SystemExit(f"favorites.json missing at {favs_path}")
    favs = [f for f in json.loads(favs_path.read_text()).get("favorites", [])
            if f.get("file") and (FAV / f["file"]).is_file()]
    print(f"loaded {len(favs)} favorites with files on disk")

    if not args.dry_run:
        if args.out.exists():
            shutil.rmtree(args.out)
        args.out.mkdir(parents=True)

    build_groups(favs, args.out, args.dry_run)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
