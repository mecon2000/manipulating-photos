#!/usr/bin/env python3
"""Add clothing to a photo before it enters a stylization pipeline.

become-image regularly reinterprets a partly-open shirt as no shirt: the stylizer sees
skin and paints skin. Covering the exposed area first gives it something to keep, which
is far more reliable than asking a prompt not to undress someone.

The mask is body-skin BELOW the chin only — MediaPipe already classifies face and hair
separately, so the face is never inpainted and identity is untouched.

  dress.py --source PHOTO.jpg --garment "buttoned white linen shirt" --out OUT.jpg
"""
import argparse, importlib.util, io, os, sys, urllib.request
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
# Version-pinned: replicate.run() on the bare name 404s for this model, and pinning
# also keeps results reproducible from the sidecar.
MODEL = ("lucataco/sdxl-inpainting:"
         "a5b13068cc81a89a4fbeefeccc774869fcb34df4dbc92c1555e0f2771d49dde7")


def _mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def skin_below_chin(img, include_clothes=False):
    """Mask the exposed torso: body-skin, under the chin, excluding face and hair.

    With include_clothes the existing garments join the mask, which is what you want
    when the thing to cover IS clothing — a bra masked as "clothes" survives a
    skin-only mask and the result is a shirt layered over lingerie.
    """
    bs = _mod("body_segment", "body-segment.py")
    fa = _mod("face_align", "face_align.py")
    cats = bs.segment_body(np.asarray(img))
    names = {v: k for k, v in bs.CATEGORIES.items()}
    body_id = names.get("body-skin")
    m = (cats == body_id).astype(np.float32)
    if include_clothes:
        for extra in ("clothes", "others"):
            eid = names.get(extra)
            if eid is not None:
                m = np.maximum(m, (cats == eid).astype(np.float32))
    pts = fa.landmarks(np.asarray(img))
    if pts is not None:
        chin_y = float(pts["chin"][1])
        m[:int(chin_y), :] = 0.0            # never touch the face
    k = np.ones((9, 9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.dilate(m, k, iterations=2)      # cover the skin/fabric boundary
    return cv2.GaussianBlur(m, (21, 21), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--garment", default="a buttoned white linen shirt, fully closed, "
                                        "soft natural folds, matching the existing fabric")
    p.add_argument("--strength", type=float, default=0.92)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--include-clothes", action="store_true",
                   help="also repaint existing garments (needed to replace lingerie)")
    p.add_argument("--save-mask", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, str(HERE))
    from become_image_replicate import load_token
    os.environ["REPLICATE_API_TOKEN"] = load_token()
    import replicate

    src = Path(args.source).expanduser()
    img = Image.open(src).convert("RGB")
    mask = skin_below_chin(img, args.include_clothes)
    cover = float((mask > 0.5).mean())
    print(f"  mask covers {cover*100:.1f}% of the frame")
    if cover < 0.005:
        sys.exit("almost nothing to cover — is she already dressed?")

    mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    if args.save_mask:
        mask_img.save(Path(args.out).with_name(Path(args.out).stem + "_mask.png"))

    def buf(im, mode="RGB"):
        b = io.BytesIO()
        im.convert(mode).save(b, "PNG")
        b.seek(0)
        return b

    out = replicate.run(MODEL, input={
        "image": buf(img), "mask": buf(mask_img, "L"),
        "prompt": args.garment,
        "negative_prompt": "nude, topless, bare skin, cleavage, lingerie, underwear, "
                           "deformed, extra limbs, blurry",
        "strength": args.strength, "steps": args.steps,   # this model calls it "steps"
        "guidance_scale": 8.0, "seed": args.seed,
    })
    item = out[0] if isinstance(out, list) else out
    data = item.read() if hasattr(item, "read") else urllib.request.urlopen(str(item)).read()
    got = Image.open(io.BytesIO(data)).convert("RGB")
    if got.size != img.size:
        got = got.resize(img.size, Image.LANCZOS)

    # Composite through the mask: the model returns a whole frame, and everything
    # outside the garment area must stay bit-identical to the photograph.
    a = np.clip(mask, 0, 1)[..., None]
    blend = np.asarray(img).astype(np.float32) * (1 - a) + np.asarray(got).astype(np.float32) * a
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8)).save(args.out, quality=95)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
