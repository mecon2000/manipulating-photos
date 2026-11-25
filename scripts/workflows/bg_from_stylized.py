#!/home/rong/openclaw-venv/bin/python3
"""Lift the background out of an AI-stylized version of a photo and use it as a compose layer.

The AI style imitation (surreal_with_face) paints a wilder background than anything we
generate standalone, but it also redraws her body. So: cut the AI figure out (BiRefNet on
the stylized image, unioned with the piece's own subject mask so the hole covers both
silhouettes), fill the hole with LaMa, and add the result as a `backdrop` layer (bottom of
the stack) under the real, untouched photo.

  bg_from_stylized.py --stylized STYLIZED.jpg --piece PIECE_DIR [--id ai_backdrop] [--dilate 0.02]

Writes PIECE_DIR/bg_ai.jpg (+ bg_ai_mask.png). ~$0.002 (BiRefNet + LaMa on fal).
"""
import argparse, base64, json, os, subprocess, sys
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking  # noqa: E402


def _b64(im, fmt="JPEG"):
    buf = BytesIO()
    im.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def lama(image, mask):
    """LaMa on fal, at <=1024 long edge (its sweet spot), result scaled back."""
    W, H = image.size
    sc = min(1.0, 1024 / max(W, H))
    w, h = (int(W * sc) // 8) * 8, (int(H * sc) // 8) * 8
    r = requests.post("https://fal.run/fal-ai/lama",
                      headers={"Authorization": f"Key {masking._get_fal_key()}",
                               "Content-Type": "application/json"},
                      json={"image_url": "data:image/jpeg;base64," + _b64(image.resize((w, h), Image.LANCZOS)),
                            "mask_image_url": "data:image/png;base64," + _b64(mask.resize((w, h), Image.NEAREST), "PNG")},
                      timeout=180)
    r.raise_for_status()
    out = Image.open(BytesIO(requests.get(r.json()["image"]["url"], timeout=60).content)).convert("RGB")
    return out.resize((W, H), Image.LANCZOS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stylized", required=True)
    p.add_argument("--piece", required=True)
    p.add_argument("--id", default="ai_backdrop")
    p.add_argument("--dilate", type=float, default=0.02, help="grow the hole, fraction of short edge")
    a = p.parse_args()

    d = Path(a.piece).expanduser()
    src = Image.open(d / "source.jpg").convert("RGB")
    W, H = src.size
    sty = Image.open(a.stylized).convert("RGB").resize((W, H), Image.LANCZOS)

    m_ai, _ = masking.build_mask(sty, "subject", feather=0)
    m = np.asarray(m_ai.resize((W, H)), np.float32) / 255 > 0.5
    if (d / "mask.png").exists():
        m |= np.asarray(Image.open(d / "mask.png").convert("L").resize((W, H)), np.float32) / 255 > 0.5
    m = binary_dilation(m, iterations=max(1, int(a.dilate * min(W, H))))
    mask = Image.fromarray((m * 255).astype(np.uint8))
    mask.save(d / "bg_ai_mask.png")

    filled = lama(sty, mask)
    soft = Image.fromarray((gaussian_filter(m.astype(np.float32), 0.004 * min(W, H)) * 255).astype(np.uint8))
    out = Image.composite(filled, sty, soft)
    out.save(d / "bg_ai.jpg", quality=95)
    print(f"→ {d / 'bg_ai.jpg'}")

    P = json.loads((d / "piece.json").read_text())
    lc = str(HERE / "layer_compose.py")
    if any(L["id"] == a.id for L in P["layers"]):
        subprocess.run([lc, "set", str(d), f"{a.id}.image={d / 'bg_ai.jpg'}", "--no-render"], check=True)
    else:
        subprocess.run([lc, "add", str(d), "--id", a.id, "--kind", "backdrop", "--z", "under",
                        "--param", f"image={d / 'bg_ai.jpg'}", "opacity=1", "brightness=1",
                        "--why", "background lifted from the AI-stylized version (figure cut + LaMa)",
                        "--no-render"], check=True)
    # bottom of the stack: under everything else
    P = json.loads((d / "piece.json").read_text())
    under = [L["id"] for L in P["layers"] if L["z"] == "under" and L["id"] != a.id]
    over = [L["id"] for L in P["layers"] if L["z"] == "over"]
    subprocess.run([lc, "order", str(d), ",".join([a.id] + under + ["subject"] + over), "--no-render"],
                   check=True)
    print(f"layer {a.id} added at the bottom (render to see it)")


if __name__ == "__main__":
    main()
