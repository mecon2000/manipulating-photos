#!/usr/bin/env python3
"""Contact sheets for review, with labels you can actually read.

Two things this exists to get right, both learned the hard way:
labels were rendered in PIL's ~11px bitmap default and were unreadable once the
sheet was scaled to fit a screen; and sheets were written to the session
scratchpad, which lives on ext4 and is invisible from Windows, the hub, and the
terminal client. Default output is the shared review folder.
"""
import os
from PIL import Image, ImageDraw, ImageFont

REVIEW = os.path.expanduser("~/.openclaw/workspace/shared/faces-candidates/review")
_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]


def font(px):
    for f in _FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, px)
    return ImageFont.load_default()


def contact(items, out, cols=4, width=440, ratio=16 / 9, bg=(16, 18, 22),
            label_px=None, label_fill=(255, 215, 80)):
    """items: (label, path) pairs. Label size scales with the tile, so a sheet
    stays legible whatever width it is built at."""
    label_px = label_px or max(18, width // 16)
    pad = label_px + 14
    h = int(width * ratio) if ratio else width
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (h + pad)), bg)
    d = ImageDraw.Draw(sheet)
    f = font(label_px)
    for i, (label, path) in enumerate(items):
        x, y = (i % cols) * width, (i // cols) * (h + pad)
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue
        im.thumbnail((width - 10, h - 10))
        sheet.paste(im, (x + 5 + (width - 10 - im.size[0]) // 2, y + pad))
        d.text((x + 8, y + 6), str(label), fill=label_fill, font=f)
    if not os.path.isabs(out):
        os.makedirs(REVIEW, exist_ok=True)
        out = os.path.join(REVIEW, out)
    sheet.save(out, quality=90)
    return out
