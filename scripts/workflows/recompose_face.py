#!/usr/bin/env python3
"""Redo just the face-composite steps on an ALREADY-generated stylized image.

`surreal_with_face.py` saves its become-image result as `<tag>__surreal.jpg`.
Tuning the mask or the color transfer by re-running the whole pipeline costs an
API call each time; this reruns steps 2-4 locally and free.

  recompose_face.py --source PHOTO --surreal TAG__surreal.jpg --out OUT.jpg
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import face_align as FA
from psd_layers import write_psd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="original photo (the real face)")
    p.add_argument("--surreal", required=True, help="stylized image to composite onto")
    p.add_argument("--out", required=True)
    p.add_argument("--inner", type=float, default=0.9)
    p.add_argument("--outer", type=float, default=2.6)
    p.add_argument("--power", type=float, default=1.0)
    p.add_argument("--match-scope", choices=["face", "frame", "hybrid"], default="hybrid",
                   help="where the target colour stats come from (see face_align)")
    p.add_argument("--match-strength", type=float, default=1.0,
                   help="0 = keep source tones, 1 = full transfer to stylized face")
    p.add_argument("--no-psd", action="store_true",
                   help="skip the layered PSD written next to --out")
    args = p.parse_args()

    src_pil = Image.open(args.source).convert("RGB")
    sur_pil = Image.open(args.surreal).convert("RGB")
    if sur_pil.size != src_pil.size:
        sur_pil = sur_pil.resize(src_pil.size, Image.LANCZOS)

    src_arr, sur_arr = np.asarray(src_pil), np.asarray(sur_pil)
    src_pts, dst_pts = FA.landmarks(src_arr), FA.landmarks(sur_arr)
    if src_pts is None or dst_pts is None:
        raise SystemExit(f"face not found in {'source' if src_pts is None else 'surreal'}")

    M = FA.similarity_transform(src_pts, dst_pts)
    warped = FA.warp(src_arr, M, src_pil.size) if M is not None else src_arr

    c, semi, ang = FA.ellipse_from_landmarks(dst_pts)
    mask = FA.radial_mask(src_pil.size, c, semi, ang, args.inner, args.outer, args.power)
    matched = FA.match_in_region(warped, sur_arr, mask, strength=args.match_strength,
                                 scope=args.match_scope)

    m = mask[..., None]
    out = np.clip(sur_arr * (1 - m) + matched * m, 0, 255).astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out, quality=95)
    print(f"→ {args.out}")

    if not args.no_psd:
        # The flattened JPEG is one fixed answer to "how much face comes back".
        # The PSD ships the same result as two layers with the mask live, so the
        # call can be revised per-image — around the eyes especially, where a
        # radial mask has no way to know the gaze differs between the two.
        # match the main pipeline's naming: <tag>__layers.psd, not <tag>__final.psd
        stem = Path(args.out).stem
        psd = Path(args.out).with_name(
            (stem[:-len("__final")] if stem.endswith("__final") else stem) + "__layers.psd")
        write_psd(str(psd),
                  [dict(name="stylized", rgb=sur_arr, mask=None),
                   dict(name="photo (aligned + colour-matched)", rgb=matched,
                        mask=(mask * 255).astype(np.uint8))],
                  src_pil.size, out)
        print(f"→ {psd}")


if __name__ == "__main__":
    main()
