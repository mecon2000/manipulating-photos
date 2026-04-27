"""Shared helper for exporting pipeline stages as a multi-page (layered) TIFF.

Photoshop opens multi-page TIFFs as a Layers stack. We prepend the layer name
to the image's ImageDescription (tag 270) so it's discoverable; Photoshop also
honors PageName (tag 285) on some versions.

Usage:
    from _layered_tiff import save_stack
    save_stack(out_path, [
        ("00_original", pil_img),
        ("01_relit",    pil_img),
        ("99_final_4x", upscaled_pil_img),
    ])

Behaviour:
- All 8-bit RGB.
- All layers are normalized to the SAME dimensions = the largest layer's
  dimensions, capped at MAX_EDGE on the long edge. Smaller layers are
  upsampled (LANCZOS) to match. This keeps Photoshop layers aligned when
  one of them is an upscaled final.
- None images are skipped silently.
"""

from PIL import Image, TiffImagePlugin

MAX_EDGE = 4096
DEFAULT_COMPRESSION = "tiff_lzw"


def _to_rgb(img):
    return img if img.mode == "RGB" else img.convert("RGB")


def _normalize_to_target(img, target_w, target_h):
    img = _to_rgb(img)
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


def _name_ifd(name):
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[270] = str(name)  # ImageDescription
    ifd[285] = str(name)  # PageName
    return ifd


def save_stack(out_path, layers, compression=DEFAULT_COMPRESSION):
    """Save (name, PIL.Image) pairs as a multi-page TIFF, all at the same
    dimensions (largest layer wins, capped at MAX_EDGE long edge).
    Returns path or None."""
    pairs = [(str(n), _to_rgb(im)) for (n, im) in layers if im is not None]
    if not pairs:
        return None

    # Pick target dimensions = largest layer, capped at MAX_EDGE long edge.
    target_w, target_h = 0, 0
    for _, im in pairs:
        w, h = im.size
        if max(w, h) > max(target_w, target_h):
            target_w, target_h = w, h
    long_edge = max(target_w, target_h)
    if long_edge > MAX_EDGE:
        scale = MAX_EDGE / float(long_edge)
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))

    cleaned = [(name, _normalize_to_target(im, target_w, target_h))
               for (name, im) in pairs]

    first_name, first = cleaned[0]
    rest = cleaned[1:]

    try:
        with TiffImagePlugin.AppendingTiffWriter(str(out_path), True) as tw:
            for name, im in cleaned:
                im.save(tw, format="TIFF", compression=compression,
                        tiffinfo=_name_ifd(name))
                tw.newFrame()
    except Exception:
        first.save(
            str(out_path),
            save_all=True,
            append_images=[im for (_, im) in rest],
            compression=compression,
            tiffinfo=_name_ifd(first_name),
        )

    return str(out_path)
