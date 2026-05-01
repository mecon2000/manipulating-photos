#!/usr/bin/env python3
"""Build per-preset thumbnails for the relighting tool.

Strategy:
1. For each preset, look in favorites/ for an existing relight image whose
   filename matches the preset name (case-insensitive token match) — copy +
   resize to ~/.openclaw/workspace/shared/preset_thumbs/relighting/<slug>.jpg.
2. For presets with no fav match, run relighting.py on BLD_4863.jpg
   (Gali) once, with that preset, and resize the output.

Idempotent — skips presets whose thumb already exists.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
SHARED = Path("~/.openclaw/workspace/shared").expanduser()
FAV = SHARED / "favorites"
THUMBS = SHARED / "preset_thumbs" / "relighting"
THUMBS.mkdir(parents=True, exist_ok=True)
PYTHON = str(Path("~/openclaw-venv/bin/python3").expanduser())
RELIGHTING = REPO / "scripts/workflows/relighting.py"
SOURCE_PHOTO = Path("/home/rong/.openclaw/workspace/_photos/Gali/for strmr.com/BLD_4863.jpg")

PRESETS = [
    "Dramatic Rim", "Spotlight", "Low Key", "High Key", "Neon Gels",
    "Teal & Orange", "Red Drama", "Golden Hour", "Window Light",
    "Overcast Soft", "Candlelight", "Butterfly", "Split Light",
    "Beauty Dish", "Underwater Caustics", "Moonlight", "Neon Signs",
    "Firelight", "Laser",
    "Hard Midday Sun", "Stage Backlight", "Blue Hour",
    "Projector Patterns", "Lightning Flash", "TV Glow",
    "Stained Glass", "Practical Bulb",
]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def make_thumb(src: Path, dst: Path):
    img = Image.open(src).convert("RGB")
    # Square crop to center, then resize to 480x480 — bigger so chips can show
    # title overlay legibly. UI is responsible for layout sizing.
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img.thumbnail((480, 480))
    img.save(dst, quality=88)


def find_finals_for_preset(preset: str) -> Path | None:
    """Find the latest BLD_4863 finals file for a given preset (preset name in filename)."""
    preset_safe = preset.replace(" & ", "_").replace(" ", "_")
    # relighting.py saves like: Gali_BLD_4863_<ts>_<Preset_Name>_NN.jpg
    matches = sorted(
        (SHARED / "finals").glob(f"*BLD_4863*{preset_safe}*.jpg"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return matches[0] if matches else None


def find_fav(preset: str) -> Path | None:
    """Look for a relighting fav matching this preset."""
    try:
        data = json.loads((FAV / "favorites.json").read_text())
    except Exception:
        data = {"favorites": []}
    norms = [
        preset.lower().replace(" & ", " ").replace(" ", "_"),
        preset.lower().replace(" ", "_"),
        preset.lower().replace(" ", ""),
    ]
    for f in data.get("favorites", []):
        if f.get("tool") != "relighting":
            continue
        name = (f.get("file") or "").lower()
        if any(n in name for n in norms):
            p = FAV / f["file"]
            if p.is_file():
                return p
    # Fallback: filename scan including older favs without tool field
    for p in FAV.glob("*.jpg"):
        n = p.name.lower()
        if "relight" not in n:
            continue
        if any(nn in n for nn in norms):
            return p
    return None


def generate(preset: str, out: Path) -> bool:
    # Reuse existing finals for this preset before paying fal again
    existing = find_finals_for_preset(preset)
    if existing:
        make_thumb(existing, out)
        print(f"  [reuse] {preset} ← finals/{existing.name}")
        return True
    if not SOURCE_PHOTO.is_file():
        print(f"  [skip] source photo missing: {SOURCE_PHOTO}")
        return False
    print(f"  [gen] running relighting → {preset}")
    cmd = [PYTHON, str(RELIGHTING), "--source", str(SOURCE_PHOTO),
           "--lighting", preset, "--output-to", "local",
           "--local-output-dir", str(SHARED)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "")[-300:]
        print(f"  [fail] {preset}: {tail}")
        return False
    produced = find_finals_for_preset(preset)
    if not produced:
        print(f"  [fail] {preset}: no output file found")
        return False
    make_thumb(produced, out)
    print(f"  [ok] {out.name}")
    return True


def main():
    only_missing = "--regen" not in sys.argv
    print(f"Building thumbs in {THUMBS}")
    for preset in PRESETS:
        out = THUMBS / f"{slug(preset)}.jpg"
        if out.exists() and only_missing:
            print(f"  [skip] {preset} (exists)")
            continue
        fav = find_fav(preset)
        if fav:
            make_thumb(fav, out)
            print(f"  [fav] {preset} ← {fav.name}")
            continue
        if not generate(preset, out):
            continue


if __name__ == "__main__":
    main()
