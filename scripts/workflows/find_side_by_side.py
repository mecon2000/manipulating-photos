#!/usr/bin/env python3
"""Find side-by-side composites in favorites/ — review-only, doesn't delete.

Heuristic: an image whose left and right halves
  (a) have HIGH structural similarity (same composition mirrored), and
  (b) have meaningfully DIFFERENT colors (different stylization),
is almost certainly a "original | output" comparison saved by an older
version of one of the side-by-side tools (ink-dissolution, risograph,
paper-cutout, hatching, torn-reveal).

For aspect ≈ 3 we also probe a triple-split (orig | mid | out).

Output:
  - Prints a sorted score table.
  - Copies the top N matches to ~/.openclaw/workspace/shared/_sxs_review/
    so you can browse them in one folder and decide what to crop. Originals
    in favorites/ are untouched.

Usage:
    ./scripts/workflows/find_side_by_side.py
    ./scripts/workflows/find_side_by_side.py --top 20
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

FAV = Path("~/.openclaw/workspace/shared/favorites").expanduser()
REVIEW = Path("~/.openclaw/workspace/shared/_sxs_review").expanduser()
SUSPECT_TOOLS = ("ink-dissolution", "risograph", "paper-cutout",
                 "hatching", "torn-reveal", "comparison")


def split_score(img, n_panels):
    """Return (mean edge_corr across pairs, max color_diff). Higher = more
    likely panels."""
    W, H = img.size
    pw = W // n_panels
    panels_l, panels_rgb = [], []
    for i in range(n_panels):
        p = img.crop((i * pw, 0, (i + 1) * pw, H))
        panels_l.append(np.array(p.convert("L").resize((128, 128))) / 255.0)
        panels_rgb.append(np.array(p.convert("RGB").resize((64, 64))).reshape(-1, 3).mean(axis=0))
    edge_maps = [np.abs(sobel(a, 0)) + np.abs(sobel(a, 1)) for a in panels_l]
    pair_corrs, pair_diffs = [], []
    for i in range(n_panels - 1):
        c = np.corrcoef(edge_maps[i].flatten(), edge_maps[i + 1].flatten())[0, 1]
        pair_corrs.append(float(c) if not np.isnan(c) else 0.0)
        pair_diffs.append(float(np.linalg.norm(panels_rgb[i] - panels_rgb[i + 1])))
    return (sum(pair_corrs) / len(pair_corrs),
            max(pair_diffs))


def detect_seam_panels(img):
    """Find the panel count by locating the strongest vertical-gradient
    column. Side-by-side composites have a sharp seam between panels. We
    score the candidate seam positions for 2- and 3-panel layouts and
    return the count whose expected seam columns match the actual peak.
    """
    g = np.array(img.convert("L"))
    W = g.shape[1]
    col_grad = np.abs(np.diff(g.astype(int), axis=1)).mean(axis=0)
    peak_col = int(np.argmax(col_grad))
    peak_strength = col_grad[peak_col] / (col_grad.mean() + 1e-6)
    # Distance from peak to expected seams
    d2 = abs(peak_col - W / 2) / W
    d3a = abs(peak_col - W / 3) / W
    d3b = abs(peak_col - 2 * W / 3) / W
    # Within ~1% of expected seam = match
    if d2 < 0.01 and peak_strength > 3:
        return 2, peak_strength
    if min(d3a, d3b) < 0.01 and peak_strength > 3:
        return 3, peak_strength
    return 1, peak_strength  # no clear seam


def score(path):
    try:
        img = Image.open(path)
    except Exception:
        return None
    W, H = img.size
    aspect = W / H
    if aspect < 0.95:
        return {"file": path.name, "aspect": round(aspect, 2),
                "panels": 1, "score": 0.0,
                "skip": "tall — single portrait"}
    # Authoritative panel count from the strongest vertical gradient seam
    n_seam, seam_strength = detect_seam_panels(img)
    if n_seam == 1:
        return {"file": path.name, "aspect": round(aspect, 2),
                "panels": 1, "score": 0.0,
                "skip": f"no clear seam (peak={seam_strength:.1f}×)"}
    edge_corr, color_diff = split_score(img, n_seam)
    # Real side-by-sides have HIGH structural similarity (same composition).
    # Negative or near-zero corr means the seam is internal to a single
    # photo (e.g., a strong body silhouette), not between two panels.
    if edge_corr < 0.5:
        return {"file": path.name, "aspect": round(aspect, 2),
                "panels": 1, "score": 0.0,
                "skip": f"low corr ({edge_corr:.2f}) — internal edge"}
    return {"file": path.name, "aspect": round(aspect, 2),
            "panels": n_seam, "edge_corr": round(edge_corr, 3),
            "color_diff": round(color_diff, 1),
            "seam": round(float(seam_strength), 1),
            "score": round(edge_corr * color_diff / 30.0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    favs = json.loads((FAV / "favorites.json").read_text())["favorites"]
    suspects = []
    for f in favs:
        if not f.get("file"):
            continue
        fn = f["file"].lower()
        tool = (f.get("tool") or "").lower()
        if any(t in fn or t in tool for t in SUSPECT_TOOLS) or "comparison" in fn:
            p = FAV / f["file"]
            if p.is_file():
                suspects.append((f, p))
    print(f"suspect candidates: {len(suspects)}")

    rows = []
    for f, p in suspects:
        r = score(p)
        if r and "skip" not in r:
            r["tool"] = f.get("tool", "")
            rows.append((p, r))
    rows.sort(key=lambda x: -x[1]["score"])

    print(f"\n{'#':<3}{'score':>7}{'panels':>8}{'aspect':>8}{'corr':>7}{'colordiff':>11}  file")
    for i, (p, r) in enumerate(rows[:args.top], 1):
        print(f"{i:<3}{r['score']:>7}{r['panels']:>8}{r['aspect']:>8}{r['edge_corr']:>7}{r['color_diff']:>11}  {r['file']}")

    REVIEW.mkdir(parents=True, exist_ok=True)
    # Wipe prior review batch
    for f in REVIEW.glob("*.jpg"):
        f.unlink()
    (REVIEW / "manifest.json").write_text(json.dumps(
        [r for _, r in rows[:args.top]], indent=2))
    for i, (p, r) in enumerate(rows[:args.top], 1):
        dst = REVIEW / f"{i:02d}__{r['panels']}p__{p.name}"
        shutil.copyfile(p, dst)
    print(f"\n→ Copied top {min(args.top, len(rows))} to {REVIEW}/")
    print("Browse and tell me which are real side-by-sides;")
    print("then I'll write a cropper that splits on panel count.")


if __name__ == "__main__":
    main()
