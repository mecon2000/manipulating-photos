#!/usr/bin/env python3
"""Prepare a portrait for the depth-plane editor: 25 plates, exported as flat layers.

Splits the expensive, slow half of depth-planes.py away from the interactive half.
Everything that costs money or seconds — generating decoration plates, cutting the
subject out, keying, blurring, building the relief masks — happens once, here. The
editor then only stacks finished RGBA layers, so pressing a button is instant.

  prepare_layers.py --portrait FRAME.jpg --style retro.jpg --prompt "..." [--options 5]

Writes <shared>/faces-candidates/depth-planes/_prep/<stem>/
    plates/plane{1..5}_{a..e}.jpg    full-resolution styled plates (for the final render)
    preview/plane{1..5}_{a..e}.webp  pre-blurred, keyed, relieved RGBA for the editor
    preview/portrait.webp            the cut-out subject
    preview/shadow.webp              her shadow at full strength (editor scales the alpha)
    meta.json
"""
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
PY = os.environ.get("REEL_PYTHON", sys.executable)
SHARED = Path("~/.openclaw/workspace/shared/faces-candidates").expanduser()
PLANES = [1, 2, 3, 4, 5]                       # 3 is the portrait
DECOR = [1, 2, 4, 5]
LETTERS = "abcde"


def _dp():
    spec = importlib.util.spec_from_file_location("dp", HERE / "depth-planes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rgba_webp(rgb, alpha, path, width):
    im = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                                    (np.clip(alpha, 0, 1) * 255).astype(np.uint8)]), "RGBA")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(path, "WEBP", quality=82, method=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--portrait", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--options", type=int, default=5)
    p.add_argument("--preview-width", type=int, default=760)
    p.add_argument("--scale", type=float, default=0.80)
    p.add_argument("--front-relief", type=float, default=0.90)
    p.add_argument("--subject-relief", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--skip-plates", action="store_true", help="reuse plates already on disk")
    args = p.parse_args()

    dp = _dp()
    portrait = Path(args.portrait).expanduser()
    stem = portrait.stem
    out = SHARED / "depth-planes" / "_prep" / stem
    (out / "plates").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(exist_ok=True)

    port = np.asarray(Image.open(portrait).convert("RGB"))
    H, W = port.shape[:2]

    # 1. plates ------------------------------------------------------------
    jobs = [(pl, LETTERS[o]) for pl in DECOR for o in range(args.options)]
    if not args.skip_plates:
        src_dir = out / "plates" / "_src"
        src_dir.mkdir(exist_ok=True)
        for i, (pl, letter) in enumerate(jobs):
            dst = out / "plates" / f"plane{pl}_{letter}.jpg"
            if dst.exists():
                continue
            s = src_dir / f"plane{pl}_{letter}.jpg"
            Image.fromarray(dp.noise_source(W, H, pl, args.seed + i)).save(s, quality=94)
            r = subprocess.run([PY, str(HERE / "style_transfer_replicate.py"),
                                "--source", str(s), "--style", str(Path(args.style).expanduser()),
                                "--prompt", args.prompt, "--denoising-strength", "0.85",
                                "--out-dir", str(out / "plates")],
                               capture_output=True, text=True)
            made = sorted((out / "plates").glob(f"plane{pl}_{letter}__style_*.jpg"))
            if made:
                made[-1].replace(dst)
                print(f"  plate {pl}{letter}", flush=True)
            else:
                print(f"  plate {pl}{letter} FAILED: {r.stdout.strip()[-120:]}", flush=True)

    # 2. subject + masks ---------------------------------------------------
    stem_src = stem.split("__style")[0]
    cand = portrait.parent.parent / "_sources" / "crop_916" / f"{stem_src}.jpg"
    mask_img = np.asarray(Image.open(cand).convert("RGB")) if cand.is_file() else port
    if mask_img.shape[:2] != (H, W):
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    subj = dp.refine_alpha(mask_img, dp.subject_alpha(mask_img), shrink=1.5)
    pts = dp.mesh(port)

    sw, sh = int(W * args.scale), int(H * args.scale)
    ox, oy = (W - sw) // 2, H - sh
    small = cv2.resize(port, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
    small_a = cv2.resize(subj, (sw, sh), interpolation=cv2.INTER_LINEAR)
    fade = max(6, int(min(sw, sh) * 0.03))
    ramp = np.ones((sh, sw), np.float32)
    g = np.linspace(0, 1, fade, dtype=np.float32)
    ramp[:, :fade] *= g; ramp[:, -fade:] *= g[::-1]; ramp[:fade, :] *= g[:, None]
    small_a = small_a * ramp
    port_l = np.zeros_like(port); subj_l = np.zeros((H, W), np.float32)
    port_l[oy:oy + sh, ox:ox + sw] = small
    subj_l[oy:oy + sh, ox:ox + sw] = small_a
    if pts is not None:
        pts = pts * args.scale + np.array([ox, oy], np.float32)

    hole = dp.eye_hole((H, W), pts)
    relief = dp.face_relief((H, W), pts, args.front_relief)
    body = cv2.GaussianBlur(subj_l, (0, 0), max(W, H) * 0.02)
    front = hole * relief * (1.0 - args.subject_relief * np.clip(body, 0, 1))

    rgba_webp(port_l, subj_l, out / "preview" / "portrait.webp", args.preview_width)

    k = max(3, int(min(W, H) * 0.035)) | 1
    sh_a = cv2.GaussianBlur(np.clip(subj_l, 0, 1), (k, k), 0)
    sh_a = np.roll(np.roll(sh_a, 14, axis=0), 10, axis=1)
    sh_a = np.clip(sh_a - np.clip(subj_l, 0, 1), 0, 1)
    rgba_webp(np.zeros_like(port), sh_a, out / "preview" / "shadow.webp", args.preview_width)

    # ground first: the haze pass blends far planes toward it
    ground = [238, 238, 238]
    far = out / "plates" / f"plane5_{LETTERS[0]}.jpg"
    if far.is_file():
        fa = np.asarray(Image.open(far).convert("RGB")).astype(np.float32)
        q = (fa // 16).astype(np.int32)
        flat = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
        sel = flat == np.bincount(flat.ravel()).argmax()
        ground = [int(v) for v in np.median(fa[sel].reshape(-1, 3), axis=0)]
        print(f"  ground colour from plane5: {ground}")

    # 3. layers ------------------------------------------------------------
    made = []
    for pl, letter in jobs:
        src = out / "plates" / f"plane{pl}_{letter}.jpg"
        if not src.is_file():
            continue
        im = np.asarray(Image.open(src).convert("RGB"))
        im, a = dp.render_plate(im, pl, W, H, ground)
        if pl in (1, 2):
            a = a * front                     # near planes keep clear of her and the eyes
        rgba_webp(im, a, out / "preview" / f"plane{pl}_{letter}.webp", args.preview_width)
        made.append(f"plane{pl}_{letter}")

    meta = {"stem": stem, "ground": ground, "portrait": str(portrait), "style": str(Path(args.style).expanduser()),
            "prompt": args.prompt, "w": W, "h": H, "preview_width": args.preview_width,
            "scale": args.scale, "front_relief": args.front_relief,
            "subject_relief": args.subject_relief, "options": args.options,
            "layers": made, "planes": DECOR, "letters": LETTERS[:args.options]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(made)} layers → {out}")


if __name__ == "__main__":
    main()
