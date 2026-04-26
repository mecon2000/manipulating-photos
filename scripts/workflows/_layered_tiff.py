"""Shared helper for exporting pipeline stages as a multi-page (layered) TIFF.

Photoshop opens multi-page TIFFs as a Layers stack. We prepend the layer name
to the image's ImageDescription (tag 270) so it's discoverable; Photoshop also
honors PageName (tag 285) on some versions.

Usage:
    from _layered_tiff import save_stack
    save_stack(out_path, [
        ("00_original", pil_img),
        ("01_relit",    pil_img),
        ...
    ])

Notes:
- Always 8-bit RGB.
- Any layer with long edge > 4096 is downsampled to 4096 to keep TIFF size sane.
- None images in the list are skipped silently (so callers can pass optional
  intermediates without guarding).
"""

from PIL import Image, TiffImagePlugin

MAX_EDGE = 4096
DEFAULT_COMPRESSION = "tiff_lzw"


def _prep(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_EDGE:
        scale = MAX_EDGE / float(long_edge)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def _name_ifd(name):
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[270] = str(name)  # ImageDescription
    ifd[285] = str(name)  # PageName
    return ifd


def save_stack(out_path, layers, compression=DEFAULT_COMPRESSION):
    """Save (name, PIL.Image) pairs as a multi-page TIFF. Returns path or None."""
    cleaned = [(str(n), _prep(im)) for (n, im) in layers if im is not None]
    if not cleaned:
        return None

    first_name, first = cleaned[0]
    rest = cleaned[1:]

    # Stash per-page tags via the encoderinfo channel; PIL's save_all will
    # honor tiffinfo on the first page. For named layers across pages we
    # rely on AppendingTiffWriter.
    try:
        with TiffImagePlugin.AppendingTiffWriter(str(out_path), True) as tw:
            for name, im in cleaned:
                im.save(tw, format="TIFF", compression=compression,
                        tiffinfo=_name_ifd(name))
                tw.newFrame()
    except Exception:
        # Fallback: simple multi-page save without per-page names.
        first.save(
            str(out_path),
            save_all=True,
            append_images=[im for (_, im) in rest],
            compression=compression,
            tiffinfo=_name_ifd(first_name),
        )

    return str(out_path)
