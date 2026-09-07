#!/usr/bin/env python3
"""Write a layered PSD with real, editable layer masks.

Why hand-rolled: pytoshop is the obvious library and it is broken against modern
numpy (its channel-id handling underflows on uint8 and its RLE path references an
undefined `packbits`). The PSD layer format is stable and well documented, and we
only need one narrow slice of it — RGB, 8-bit, uncompressed channels, one optional
user mask per layer — so writing it directly is less fragile than patching a dead
dependency.

Layers arrive bottom-first, the order PSD itself stores them in.
"""
import struct
import numpy as np


def _pascal4(s):
    """Pascal string padded so the whole field is a multiple of 4 bytes."""
    b = s.encode("latin-1", "replace")[:255]
    out = bytes([len(b)]) + b
    return out + b"\x00" * (-len(out) % 4)


def _even(b):
    return b + b"\x00" * (len(b) % 2)


def write_psd(path, layers, size, composite):
    """layers: list of dicts, bottom first:
         name  – layer name
         rgb   – (H,W,3) uint8, full canvas
         mask  – (H,W) uint8 or None; 255 shows the layer, 0 hides it
       composite: (H,W,3) uint8 flattened preview, what a viewer shows before
       it parses layers — so it must match the layered result, or the file lies
       about itself in every thumbnail."""
    W, H = size
    hdr = (b"8BPS" + struct.pack(">H", 1) + b"\x00" * 6 +
           struct.pack(">HIIHH", 3, H, W, 8, 3))

    records, chan_blobs = b"", b""
    for L in layers:
        rgb = np.ascontiguousarray(L["rgb"])
        mask = L.get("mask")
        ids = [(0, rgb[..., 0]), (1, rgb[..., 1]), (2, rgb[..., 2])]
        if mask is not None:
            ids.append((-2, np.ascontiguousarray(mask)))
        chans = b""
        for cid, plane in ids:
            data = struct.pack(">H", 0) + plane.astype(np.uint8).tobytes()   # 0 = raw
            chans += struct.pack(">hI", cid, len(data))
            chan_blobs += data
        # layer mask record: 20 bytes when present (rect, default colour, flags, pad)
        if mask is not None:
            lm = struct.pack(">iiii", 0, 0, H, W) + bytes([0]) + bytes([0]) + b"\x00\x00"
            mask_field = struct.pack(">I", len(lm)) + lm
        else:
            mask_field = struct.pack(">I", 0)
        extra = mask_field + struct.pack(">I", 0) + _pascal4(L["name"])
        records += (struct.pack(">iiii", 0, 0, H, W) + struct.pack(">H", len(ids)) +
                    chans + b"8BIM" + b"norm" +
                    bytes([255, 0, 0, 0]) + struct.pack(">I", len(extra)) + extra)

    layer_info = _even(struct.pack(">h", len(layers)) + records + chan_blobs)
    layer_info = struct.pack(">I", len(layer_info)) + layer_info
    lmi = layer_info + struct.pack(">I", 0)                 # empty global mask info
    body = struct.pack(">I", len(lmi)) + lmi

    comp = np.ascontiguousarray(composite)
    img = struct.pack(">H", 0) + b"".join(
        comp[..., c].astype(np.uint8).tobytes() for c in range(3))

    with open(path, "wb") as f:
        f.write(hdr + struct.pack(">I", 0) + struct.pack(">I", 0) + body + img)
    return path
