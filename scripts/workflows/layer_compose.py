#!/home/rong/openclaw-venv/bin/python3
"""Compose: rebuild a deconstructed technique on one of Ronnie's photos (plans/deconstruct.md).

A *piece* = one photo x one recipe, in shared/deconstruct/pieces/<photo>__<recipe>/:
  piece.json      the editable state (layers from the recipe + riffs, decisions, seeds)
  source.jpg      the (smart-cropped) photo        mask.png  subject mask (BiRefNet)
  final.jpg       latest render                    layers/   per-layer RGBA PNGs
  piece.psd       layered file                     history/  NNN.jpg + history.jsonl —
                                                   every render and every pick, for the
                                                   making-of reel later
  questions/      choice sets (A/B/C renders + sheet) for the review room

  layer_compose.py init  --photo P --recipe R [--wildness 0.5] [--crop N | --custom x1,y1,x2,y2]
                         [--disable id,id] [--set id.param=value ...] [--context "..."]
  layer_compose.py render PIECE [--full]
  layer_compose.py set    PIECE id.param=value ... [--note "why"]   (then renders)
  layer_compose.py options PIECE --question "..." --vary id.param=a,b,c
         vary keys: id.<param> | id.z | id.enabled | id.seed | base.<k> | background.<k>
  layer_compose.py pick   PIECE --question qN --choice B
  layer_compose.py variant PIECE <layer-id> [--count 3 | --seed N ...]  one layer, other seeds → layers/variants/
  layer_compose.py order  PIECE id,id,subject,id,...       bottom→top; left of subject = under
  layer_compose.py save   PIECE --name "..."               savepoint (piece.json + final.jpg)
  layer_compose.py restore PIECE --savepoint N             (savepoint 0 = start, always there)
  layer_compose.py fav    PIECE [--name "..."]             full-res + <photo>__<recipe>.psd →
                                                           shared/favorites + favorites.json
  set/order take --no-render (fast client edits; render later). Every render writes
  layers/manifest.json + one RGBA PNG per layer (disabled ones too, + paper and finish).

Cropping is smart-crop.py's job (choose with `smart-crop.py --source P --show-options`).
"""
import argparse, copy, json, math, os, re, subprocess, sys, time, zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter, distance_transform_edt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wall_layers as wl  # noqa: E402

ROOT = Path("~/.openclaw/workspace/shared/deconstruct").expanduser()
PIECES = ROOT / "pieces"
PREVIEW = 1600
LETTERS = "ABCDEFGH"


# ---------------------------------------------------------------- piece state
def load(piece):
    d = Path(piece)
    if not d.is_absolute() and not d.exists():
        d = PIECES / piece
    return d, json.loads((d / "piece.json").read_text())


def save(d, P):
    (d / "piece.json").write_text(json.dumps(P, indent=2, ensure_ascii=False))


def hist(d, P, event, **kw):
    """Append to history.jsonl; returns the history index for this event."""
    h = d / "history"
    h.mkdir(exist_ok=True)
    n = len(list(h.glob("*.jpg")))
    rec = {"i": n, "t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}
    with (h / "history.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n


def parse_val(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def apply_set(P, key, val):
    """id.param=value  |  id.z / id.enabled / id.seed / id.region  |  base.k / background.k"""
    head, _, k = key.partition(".")
    if head == "finish" and k == "enabled":       # the grain & scratches pass
        P["base"]["finish"] = val
        return
    if head in ("base", "background"):
        P[head][k] = val
        return
    if k in ("bbox", "params.bbox"):
        raise SystemExit("bbox only marks where the element sat in the REFERENCE photo — nothing "
                         "reads it. To keep a layer off/on part of the picture use "
                         f"{head}.exclude=x0,y0,x1,y1 or {head}.include=x0,y0,x1,y1 "
                         "(normalized, ';' between boxes; empty value clears).")
    for L in P["layers"]:
        if L["id"] == head:
            if k in ("z", "enabled", "removed", "seed", "region", "kind"):
                L[k] = val
            else:
                L["params"][k] = val
            return
    raise SystemExit(f"no layer '{head}' (have: {', '.join(L['id'] for L in P['layers'])})")


# ---------------------------------------------------------------- init
def smart_crop(photo, d, crop=None, custom=None):
    if crop is None and custom is None:
        return photo
    cmd = [sys.executable, str(HERE / "smart-crop.py"), "--source", str(photo), "--local-output-dir", str(d)]
    cmd += ["--crop", str(crop)] if crop is not None else ["--custom", custom]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "NOTIFY_DISABLE": "1"})
    m = re.search(r"→ (.+\.jpg)", out.stdout)
    if not m:
        raise SystemExit(f"smart-crop failed:\n{out.stdout[-800:]}{out.stderr[-800:]}")
    return Path(m.group(1).strip())


def cmd_init(a):
    import riff as R
    photo = Path(a.photo).expanduser()
    rp = ROOT / "recipes" / f"{a.recipe}.json"
    recipe = json.loads(rp.read_text())
    d = PIECES / f"{photo.stem}__{a.recipe}"
    d.mkdir(parents=True, exist_ok=True)
    src = smart_crop(photo, d, a.crop, a.custom)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.save(d / "source.jpg", quality=95)
    seed = a.seed if a.seed is not None else int(time.time()) % 100000
    # mix the photo in: the same --seed on two photos must not give two pieces the
    # same scraps / glyph layouts / brush strokes (Elina came out as Elly's twin)
    import zlib
    seed = (seed + zlib.crc32(photo.stem.encode())) % 100000
    P = {"schema": 1, "photo": str(photo), "crop": {"n": a.crop, "custom": a.custom},
         "recipe": a.recipe, "seed": seed, "wildness": a.wildness,
         "palette": recipe.get("palette", []), "base": copy.deepcopy(recipe["base"]),
         "background": copy.deepcopy(recipe["background"]),
         "layers": copy.deepcopy(recipe["layers"]), "themes": [], "decisions": []}
    P["base"]["edge_dissolve"] = 0   # Ronnie: no torn-edge effect — clean cut-out
    for i, L in enumerate(P["layers"]):
        L.setdefault("seed", seed + 101 * (i + 1))
        if L.get("kind") == "paper_scraps":
            L.setdefault("params", {})["unreadable"] = True
        if L.get("kind") == "ink_smear":      # own set of Flux brush strokes per piece
            L.setdefault("params", {}).setdefault("brush_seed", 1000 + (seed % 9000) * 4)
    for lid in filter(None, (a.disable or "").split(",")):
        apply_set(P, f"{lid}.enabled", False)
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        apply_set(P, k, parse_val(v))
    if a.wildness > 0.15:
        print(f"riffing (wildness {a.wildness}) ...")
        rf = R.riff(photo, P, a.wildness, a.context)
        P["themes"] = rf.get("themes", [])
        for j, L in enumerate(rf.get("riffs", [])):
            L["seed"] = seed + 977 * (j + 1)
            ids = {x["id"] for x in P["layers"]}
            while L["id"] in ids:
                L["id"] += "_r"
            P["layers"].append(L)
    P["decisions"].append({"t": time.strftime("%F %T"), "init": {
        "disable": a.disable, "set": a.set, "wildness": a.wildness, "context": a.context}})
    save(d, P)
    print(f"Piece: {d}")
    for L in P["layers"]:
        print(f"  {'✓' if L.get('enabled', True) else '·'} {L['id']:<32} {L['kind']:<13} "
              f"{L['z']:<5} {'riff' if L.get('riff') else ''}")
    render(d, P, event="init")
    ensure_start(d)
    return d


# ---------------------------------------------------------------- render
def bw(rgb, base, mask=None):
    lum = rgb @ np.array([0.27, 0.62, 0.11], np.float32)
    if mask is not None and (mask > 0.5).any():
        # levels from the SUBJECT only (backlit shots otherwise go murky), then lift
        # the midtones so skin reads pale/statue-like as in the refs
        v = lum[mask > 0.5]
        lo, hi = np.percentile(v, [1.5, 99.5])
        lum = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)
        med = float(np.median(lum[mask > 0.5]))
        target = float(base.get("subject_mid", 0.55))
        if 0.02 < med < 0.98:
            lum = lum ** (np.log(target) / np.log(med))
    preset = base.get("preset", "tri-x-punch")
    c = float(base.get("contrast", 1.5))
    if preset == "high-key":
        lum = lum ** 0.8
    elif preset == "low-key":
        lum = lum ** 1.3
    elif preset == "soft-matte":
        c *= 0.7
    k = 4 + 6 * (c - 1)                       # S-curve steepness
    out = 1 / (1 + np.exp(-k * (lum - 0.5)))
    lo, hi = 1 / (1 + np.exp(k * 0.5)), 1 / (1 + np.exp(-k * 0.5))
    out = (out - lo) / (hi - lo)
    if preset in ("tri-x-punch", "newsprint"):
        out = np.clip((out - 0.05) / 0.9, 0, 1)   # crush blacks, chalk highlights
    return out.astype(np.float32)


def _lifted(im):
    """Brightened copy for masking only: gamma lift + CLAHE pulls dark hair / shoulders
    out of a dark room so the matting model sees their edge. Never shown."""
    import cv2
    a = np.asarray(im.convert("RGB"), np.float32) / 255
    a = np.power(a, 0.45)
    lab = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def _person_mask(im):
    """MediaPipe multiclass 'anything that's her' (skin/hair/clothes). Coarse edges, but it
    keeps skin the matting model drops in low light (chest between harness straps)."""
    import cv2
    import mediapipe as mp
    W, H = im.size
    small = im.convert("RGB").copy()
    small.thumbnail((1024, 1024))
    model = Path("~/openclaw-venv/mediapipe_models/selfie_multiclass.tflite").expanduser()
    with mp.tasks.vision.ImageSegmenter.create_from_options(mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            output_category_mask=True)) as seg:
        cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(np.asarray(small)))).category_mask
        cm = cm.numpy_view().squeeze()
    person = ((cm > 0) * 255).astype(np.uint8)
    person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    person = cv2.resize(person, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(person, (0, 0), max(1.0, W * 0.002))


def subject_mask(d, im, boost=False):
    """BiRefNet subject mask, cached. boost (base.mask_boost): union with BiRefNet on a
    brightened copy and a MediaPipe person mask — recovers hair/skin lost in a dark room."""
    mp = d / "mask.png"
    if not mp.exists():
        from masking import build_mask
        m, _ = build_mask(im, affect="subject", exclude="", output_dir=str(d), feather=0)
        m = np.asarray(m.convert("L").resize(im.size), np.uint8)
        if boost:
            mb, _ = build_mask(_lifted(im), affect="subject", exclude="", output_dir=str(d), feather=0)
            m = np.maximum(m, np.asarray(mb.convert("L").resize(im.size), np.uint8))
            m = np.maximum(m, _person_mask(im))
        Image.fromarray(m).save(mp)
    return Image.open(mp).convert("L")


def head_point(mask, sbox):
    x0, y0, x1, y1 = sbox
    band = mask[y0:y0 + max(2, (y1 - y0) // 6)]
    xs = np.nonzero(band > 0.5)[1]
    hx = xs.mean() if len(xs) else (x0 + x1) / 2
    return hx, y0 + (y1 - y0) * 0.1


def face_points(d, src_full):
    """Face landmarks on the full-res source, cached in face.json."""
    fp = d / "face.json"
    if not fp.exists():
        from face_align import landmarks
        pts = landmarks(np.asarray(src_full))
        fp.write_text(json.dumps({k: [float(v[0]), float(v[1])] for k, v in (pts or {}).items()}))
    return json.loads(fp.read_text())


def eye_protect(pts, W, H, scale):
    """Soft band over both eyes. House rule: nothing ever covers her eyes."""
    if not pts or "eye_l" not in pts or "eye_r" not in pts:
        return np.zeros((H, W), np.float32)
    (lx, ly), (rx, ry) = [(v[0] * scale, v[1] * scale) for v in (pts["eye_l"], pts["eye_r"])]
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    dist = math.hypot(rx - lx, ry - ly)
    ang = math.atan2(ry - ly, rx - lx)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
    v = -(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
    r = np.sqrt((u / (dist * 1.1)) ** 2 + (v / (dist * 0.2)) ** 2)   # eyes are short: keep the band tight
    return np.clip((1.5 - r) / 0.5, 0, 1).astype(np.float32)


def focus_map(mask, s):
    """1 on/near the figure, falling off into the margins — decorations pile up
    behind her and the edges of the paper stay quiet (as in the refs)."""
    out = distance_transform_edt(mask < 0.5)
    return np.exp(-(out / (s * 0.22)) ** 2).astype(np.float32)


def edge_dissolve(mask, amt, s, rng, keep_layout=False):
    """keep_layout: a piece built WITH the dissolve drew its noise from the shared rng, so
    every later layer's placement depends on it. Turning the effect off must still draw
    the same noise, or the whole layout reshuffles (base.edge_keep_layout)."""
    if amt <= 0:
        if keep_layout:
            wl.fbm(mask.shape, rng, s * 0.006, 3)
        return mask
    inside = distance_transform_edt(mask > 0.5)
    band = s * 0.035 * amt
    n = wl.fbm(mask.shape, rng, s * 0.006, 3)
    t = np.clip(inside / max(1, band), 0, 1)
    keep = np.clip(t * 1.6 - 0.3 + 0.6 * n * (1 - t), 0, 1)   # noise only in the edge band
    keep[t >= 1] = 1
    return (mask * keep).astype(np.float32)


def parse_boxes(v):
    """'x0,y0,x1,y1;x0,y0,x1,y1' or [[..],[..]] or [..] (normalized 0-1) -> list of boxes."""
    if v in (None, "", [], "none"):
        return []
    if isinstance(v, str):
        v = [[float(t) for t in b.split(",")] for b in v.split(";") if b.strip()]
    if v and not isinstance(v[0], (list, tuple)):
        v = [v]
    return [tuple(float(t) for t in b[:4]) for b in v if len(b) >= 4]


def box_mask(boxes, W, H, s):
    m = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in boxes:
        m[int(min(y0, y1) * H):int(max(y0, y1) * H), int(min(x0, x1) * W):int(max(x0, x1) * W)] = 1
    return gaussian_filter(m, s * 0.012) if boxes else m


def apply_boxes(L, lay, W, H, s):
    """params.include: the layer only shows inside these boxes; params.exclude: never inside
    them. Soft-edged. This is THE way to keep a layer off part of the picture."""
    p = L.get("params", {})
    inc, exc = parse_boxes(p.get("include")), parse_boxes(p.get("exclude"))
    if inc:
        lay.alpha = lay.alpha * box_mask(inc, W, H, s)
    if exc:
        lay.alpha = lay.alpha * (1 - box_mask(exc, W, H, s))
    return lay


def composite(dst, L):
    a = L.alpha[..., None]
    if L.blend == "multiply":
        return dst * (1 - a) + dst * L.rgb * a
    if L.blend == "screen":
        return dst * (1 - a) + (1 - (1 - dst) * (1 - L.rgb)) * a
    return dst * (1 - a) + L.rgb * a


def build(d, P, full=False, all_layers=False, only=None, only_seeds=None):
    """Returns (final HxWx3 float, stack) at preview or full res.
    stack: bottom→top list of dicts {id, z, enabled, layer} — paper, under…, subject, over…, finish.
    all_layers: also generate disabled layers (not composited) so a client can toggle them.
    only: id of one layer — returns (None, [that entry]) without compositing (variants)."""
    src = Image.open(d / "source.jpg").convert("RGB")
    mask_im = subject_mask(d, src, boost=bool(P["base"].get("mask_boost")))
    src_full_size = src.size
    face = face_points(d, src) if P["base"].get("protect_eyes", True) else {}
    if not full:
        src.thumbnail((PREVIEW, PREVIEW))
    W, H = src.size
    mask = np.asarray(mask_im.resize((W, H)), np.float32) / 255
    rgb = np.asarray(src, np.float32) / 255
    sc = float(P["base"].get("subject_scale", 1.0))
    ox, oy = 0, 0                 # where her scaled photo sits (for face landmarks)
    if sc < 0.999:   # shrink her inside the canvas (bottom-anchored) — BG is paper anyway
        import cv2
        w2, h2 = int(W * sc), int(H * sc)
        r2 = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(mask, (w2, h2), interpolation=cv2.INTER_AREA)
        rgb, mask = np.ones_like(rgb), np.zeros_like(mask)
        x0 = (W - w2) // 2 + int(float(P["base"].get("subject_dx", 0)) * W)
        rgb[H - h2:, x0:x0 + w2], mask[H - h2:, x0:x0 + w2] = r2, m2
        ox, oy = x0, H - h2
    s = min(W, H)
    rng0 = np.random.default_rng(P["seed"])
    base, bgc = P["base"], P["background"]
    lum = bw(rgb, base, mask) if base.get("bw", True) else None
    look = base.get("look", "paper")     # paper: she is printed on paper | photo: dark photograph
    photo_t = None
    if look == "photo":
        lu = lum if lum is not None else rgb.mean(2)
        pal = P.get("palette") or []
        lo_c, hi_c = (str(base.get("duotone") or "").split(",") + ["", ""])[:2]
        lo = wl.hex_rgb(lo_c or (pal[0] if pal else "#0a0a12"))
        hi = wl.hex_rgb(hi_c or "#dfe1e8")
        photo_t = (lo + (hi - lo) * lu[..., None]).astype(np.float32)
        soft = float(base.get("soften", 0)) * min(W, H)     # refs are never tack-sharp
        if soft > 0:
            from scipy.ndimage import gaussian_filter as _gf
            photo_t = _gf(photo_t, (soft, soft, 0))

    ys, xs = np.nonzero(mask > 0.5)
    sbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else (0, 0, W, H)
    focus = focus_map(mask, s)
    ctx_args = dict(W=W, H=H, sbox=sbox, head=head_point(mask, sbox), mask=mask,
                    lum=lum if lum is not None else rgb.mean(2), palette=P.get("palette", []),
                    focus=focus, photo=photo_t,
                    face={k: (ox + v[0] * W / src_full_size[0] * min(sc, 1.0),
                              oy + v[1] * W / src_full_size[0] * min(sc, 1.0))
                          for k, v in face.items()} if face else None)
    # (face landmarks follow her when subject_scale shrinks/moves her)
    # eyes are protected at any subject_scale: use the landmarks already moved with her
    protect = eye_protect(ctx_args["face"], W, H, 1.0) if ctx_args.get("face") else np.zeros((H, W), np.float32)
    quiet = float(P["base"].get("margin_quiet", 0.7))

    paper = wl.hex_rgb(bgc.get("color", "#f1eee8"))
    out = np.empty((H, W, 3), np.float32)
    out[:] = paper
    tone = 1 + 0.025 * wl.fbm((H, W), rng0, s * 0.15, 3)       # paper unevenness
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.12 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    out *= (tone * vig)[..., None]
    if bgc.get("mode") == "keep":
        out = rgb.copy()
    if photo_t is not None:     # her own background, defocused and pushed down into the dark
        from scipy.ndimage import gaussian_filter as _gf
        bl = float(bgc.get("blur", 0.02)) * s
        out = _gf(photo_t, (bl, bl, 0)) * float(bgc.get("brightness", 0.55))
    stack = [{"id": "paper", "z": "base", "enabled": True,
              "layer": wl.Layer(out.copy(), np.ones((H, W), np.float32))}]

    def run(L):
        ctx = wl.Ctx(rng=np.random.default_rng(int(L.get("seed", 1))), **ctx_args)
        g = wl.GENERATORS.get(L["kind"])
        if g is None:
            return None
        return g(L, ctx)

    live = [L for L in P["layers"] if L.get("enabled", True) and not L.get("removed")]
    gen_set = P["layers"] if (all_layers or only) else live
    if only:
        gen_set = [L for L in P["layers"] if L["id"] == only]
        if only_seeds:     # several variants of one layer in one pass (ctx built once)
            gen_set = [{**gen_set[0], "seed": sd, "enabled": True} for sd in only_seeds]

    def emit(L, lay):
        nonlocal out
        lay = apply_boxes(L, lay, W, H, s)
        on = L.get("enabled", True) and not L.get("removed")
        if on and not only:
            out = composite(out, lay)
        stack.append({"id": L["id"], "z": L["z"], "enabled": on, "layer": lay, "seed": L.get("seed")})

    # under-layers aim for the ring just OUTSIDE her silhouette (inside, she hides them)
    ctx_args["focus"] = focus * (1 - 0.85 * (mask > 0.5))
    for L in [x for x in gen_set if x["z"] == "under"]:
        lay = run(L)
        if lay is not None:
            lay.alpha = lay.alpha * (1 - quiet + quiet * focus)
            emit(L, lay)

    # subject: B&W, highlights fall to paper (multiply onto paper colour), ragged edge
    m = edge_dissolve(mask, float(base.get("edge_dissolve", 0)), s, rng0,
                      keep_layout=bool(base.get("edge_keep_layout", False)))
    subj_rgb = photo_t if photo_t is not None else (lum[..., None] * paper) if lum is not None else rgb
    cut = next((L for L in live if L["kind"] == "subject_cut"), None)
    if cut is not None:
        subj_rgb, m = slice_subject(subj_rgb, m, sbox, np.random.default_rng(int(cut.get("seed", 1))), s)
    subj = wl.Layer(subj_rgb.astype(np.float32), m)
    if not only:
        out = composite(out, subj)
        stack.append({"id": "subject", "z": "subject", "enabled": True, "layer": subj})

    ctx_args["focus"] = focus
    for L in [x for x in gen_set if x["z"] == "over"]:
        lay = run(L)
        if lay is not None:
            fade = 1.0 if L["kind"] in ("dot_rows", "micro_text", "print") else \
                (1 - quiet * 0.5 + quiet * 0.5 * focus)
            lay.alpha = lay.alpha * (1 - protect) * fade
            emit(L, lay)
    if only:
        return None, stack
    before_finish = out.copy()

    # finish: grain + scratches (the 'grain' layer kind, else base.grain)
    gl = next((L for L in live if L["kind"] in ("grain", "scratches")), None)
    g_amt = float(base.get("grain", 0.06))
    if gl is not None:
        g_amt = max(g_amt, float(gl["params"].get("opacity", 0.25)) * 0.3)
    if g_amt > 0:
        grain = rng0.standard_normal((H, W)).astype(np.float32) * g_amt * 0.5
        out = out + grain[..., None] * (0.4 + 0.6 * (1 - out.mean(2, keepdims=True)))
    if gl is not None:
        import cv2
        sc = np.zeros((H, W), np.float32)
        r = np.random.default_rng(int(gl.get("seed", 7)))
        for _ in range(int(25 * float(gl["params"].get("coverage", 1.0)))):
            x = r.uniform(0, W)
            cv2.line(sc, (int(x), int(r.uniform(0, H))), (int(x + r.normal(0, s * 0.03)),
                     int(r.uniform(0, H))), float(r.choice([-1, 1])), 1, cv2.LINE_AA)
        out = out + sc[..., None] * 0.18
    # finish as an 'overlay' layer for client compositing: 0.5 grey = neutral
    delta = (out - before_finish).mean(2)
    fin = np.repeat(np.clip(0.5 + delta, 0, 1)[..., None], 3, 2).astype(np.float32)
    fin_on = base.get("finish", True) not in (False, "false", 0, "0")
    stack.append({"id": "finish", "z": "finish", "enabled": fin_on,
                  "layer": wl.Layer(fin, np.ones((H, W), np.float32), blend="overlay")})
    return np.clip(out if fin_on else before_finish, 0, 1), stack


def slice_subject(rgb, m, sbox, rng, s):
    x0, y0, x1, y1 = sbox
    k = int(rng.integers(5, 10))
    edges = np.sort(rng.uniform(x0, x1, k - 1)).astype(int)
    xs = [x0, *edges, x1]
    o_rgb, o_m = rgb.copy(), np.zeros_like(m)
    for a, b in zip(xs, xs[1:]):
        dy = int(rng.normal(0, s * 0.012))
        o_rgb[:, a:b] = np.roll(rgb[:, a:b], dy, 0)
        o_m[:, a:b] = np.roll(m[:, a:b], dy, 0)
        o_m[:, b - 1:b + 1] *= 0.2          # hairline seam
    o_m[:, :x0] = m[:, :x0]
    o_m[:, x1:] = m[:, x1:]
    return o_rgb, o_m


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def psd_path(d, P):
    """<original photo>__<recipe>.psd — named after what it is, not 'piece'."""
    return d / f"{Path(P['photo']).stem}__{P['recipe']}.psd"


def save_rgba(path, L):
    rgba = np.dstack([L.rgb, L.alpha])
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(path)


def render(d, P, full=False, event="render", **meta):
    t = time.time()
    final, stack = build(d, P, full, all_layers=not full)
    img = to_img(final)
    img.save(d / ("final_full.jpg" if full else "final.jpg"), quality=93)
    if not full:
        ld = d / "layers"
        ld.mkdir(exist_ok=True)
        byid = {L["id"]: L for L in P["layers"]}
        entries, written = [], set()
        for i, e in enumerate(stack):
            fn = f"{i:02d}_{e['id']}.png"
            tmp = ld / f".{fn}.tmp.png"
            save_rgba(tmp, e["layer"])
            os.replace(tmp, ld / fn)                  # atomic: readers never see a gap
            written.add(fn)
            L = byid.get(e["id"], {})
            entries.append({"id": e["id"], "file": fn, "kind": L.get("kind", e["id"]),
                            "z": e["z"], "enabled": bool(L.get("enabled", True)) if L else bool(e["enabled"]),
                            "removed": bool(L.get("removed")), "blend": e["layer"].blend,
                            "riff": bool(L.get("riff")), "note": L.get("note", ""),
                            "seed": L.get("seed"), "variants": L.get("variants", [])})
    if full:
        from psd_layers import write_psd
        W, H = img.size
        base = np.asarray(Image.open(d / "source.jpg").convert("RGB"))
        layers = [{"name": "photo (original)", "rgb": base, "mask": np.zeros((H, W), np.uint8)}]
        for e in stack:
            if not e["enabled"] or e["id"] == "finish":
                continue
            L = e["layer"]
            layers.append({"name": e["id"], "rgb": (np.clip(L.rgb, 0, 1) * 255).astype(np.uint8),
                           "mask": (np.clip(L.alpha, 0, 1) * 255).astype(np.uint8)})
        write_psd(str(psd_path(d, P)), layers, (W, H), np.asarray(img))
    n = hist(d, P, event, full=full, **meta)
    if not full:
        mf = d / "layers" / "manifest.json"
        tmp = mf.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rev": n, "W": img.width, "H": img.height, "layers": entries},
                                  ensure_ascii=False))
        os.replace(tmp, mf)
        for f in (d / "layers").glob("*.png"):         # stale files only after the new manifest
            if f.name not in written:
                f.unlink()
    img.save(d / "history" / f"{n:03d}.jpg", quality=88)
    (d / "history" / f"{n:03d}.json").write_text(json.dumps(P, ensure_ascii=False))
    print(f"rendered in {time.time() - t:.1f}s  (history #{n})")
    print(f"Final: {d / ('final_full.jpg' if full else 'final.jpg')}")
    return img


def cmd_render(a):
    d, P = load(a.piece)
    render(d, P, full=a.full, event="render-full" if a.full else "render")


def cmd_set(a):
    d, P = load(a.piece)
    changes = {}
    for kv in a.kv:
        k, v = kv.split("=", 1)
        apply_set(P, k, parse_val(v))
        changes[k] = parse_val(v)
    P["decisions"].append({"t": time.strftime("%F %T"), "set": changes, "note": a.note})
    save(d, P)
    if a.no_render:
        hist_event(d, "set", changes=changes, note=a.note)
        print("saved (no render)")
        return
    render(d, P, event="set", changes=changes, note=a.note)


def cmd_recrop(a):
    """New crop for an existing piece (zoom out / reframe): new source, mask and face
    landmarks; every layer keeps its settings (positions are relative to the frame)."""
    d, P = load(a.piece)
    if a.boost:
        P["base"]["mask_boost"] = True
    src = smart_crop(Path(P["photo"]), d, a.crop, a.custom)
    ImageOps.exif_transpose(Image.open(src)).convert("RGB").save(d / "source.jpg", quality=95)
    for f in ("mask.png", "mask_affect.png", "face.json"):
        (d / f).unlink(missing_ok=True)
    P["crop"] = {"n": a.crop, "custom": a.custom}
    P["decisions"].append({"t": time.strftime("%F %T"), "recrop": P["crop"],
                           "mask_boost": P["base"].get("mask_boost", False), "note": a.note})
    save(d, P)
    render(d, P, event="recrop", crop=P["crop"], note=a.note)


def cmd_add(a):
    """Add a NEW layer (its own row in the panel) instead of repurposing an existing one."""
    d, P = load(a.piece)
    if not re.match(r"^[a-z0-9_]+$", a.id):
        raise SystemExit("id: lowercase letters, digits, _")
    if any(L["id"] == a.id for L in P["layers"]):
        raise SystemExit(f"layer {a.id} already exists — use set")
    L = {"id": a.id, "theme": a.theme, "kind": a.kind, "z": a.z, "region": a.region,
         "params": {}, "enabled": not a.off, "riff": True, "note": a.why,
         "seed": (int(P.get("seed", 1)) + 7919 * (len(P["layers"]) + 1)) % 100000}
    for kv in a.param or []:
        k, v = kv.split("=", 1)
        L["params"][k] = parse_val(v)
    idx = next((i + 1 for i, x in enumerate(P["layers"]) if x["id"] == a.after), None)
    if idx is None:   # end of its z group: right after the last layer with the same z
        same = [i for i, x in enumerate(P["layers"]) if x.get("z") == a.z]
        idx = same[-1] + 1 if same else len(P["layers"])
    P["layers"].insert(idx, L)
    P["decisions"].append({"t": time.strftime("%F %T"), "add": L, "note": a.why})
    save(d, P)
    if a.no_render:
        hist_event(d, "add", layer=a.id)
        print(f"added {a.id} (no render)")
        return
    render(d, P, event="add", layer=a.id, note=a.why)


def hist_event(d, event, **kw):
    """History line without an image (fast client-side edits; the next render snapshots)."""
    (d / "history").mkdir(exist_ok=True)
    with (d / "history" / "history.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw},
                           ensure_ascii=False) + "\n")


def cmd_variant(a):
    """Render one layer with other seeds (default: 3 random) for side-by-side choosing;
    the active seed is unchanged. Prints a Variant:/Seed: pair per variant."""
    d, P = load(a.piece)
    L = next((x for x in P["layers"] if x["id"] == a.id), None)
    if L is None:
        raise SystemExit(f"no layer {a.id}")
    rng = np.random.default_rng()
    seeds = list(a.seed) if a.seed else [int(rng.integers(1, 99_999)) for _ in range(a.count)]
    vd = d / "layers" / "variants" / a.id
    vd.mkdir(parents=True, exist_ok=True)
    cur = L.get("seed")
    todo = seeds + ([cur] if cur is not None and not (vd / f"s{cur}.png").exists() else [])
    _, stack = build(d, P, only=a.id, only_seeds=todo)
    for e in stack:
        if e["id"] == a.id and e.get("seed") in todo:
            save_rgba(vd / f"s{e['seed']}.png", e["layer"])
    L.setdefault("variants", [])
    for sd in [cur] + seeds:
        if sd is not None and sd not in L["variants"]:
            L["variants"].append(sd)
    save(d, P)
    hist_event(d, "variant", id=a.id, seeds=seeds)
    mf = d / "layers" / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        for e in m["layers"]:
            if e["id"] == a.id:
                e["variants"] = L["variants"]
        mf.write_text(json.dumps(m, ensure_ascii=False))
    for sd in seeds:
        print(f"Variant: {vd / f's{sd}.png'}\nSeed: {sd}")


def cmd_order(a):
    """Full bottom→top order of content layers with 'subject' as the divider."""
    d, P = load(a.piece)
    ids = [x.strip() for x in a.ids.split(",") if x.strip()]
    if "subject" not in ids:
        raise SystemExit("order must include 'subject'")
    byid = {L["id"]: L for L in P["layers"]}
    unknown = [i for i in ids if i != "subject" and i not in byid]
    if unknown:
        raise SystemExit(f"unknown layers: {unknown}")
    cut = ids.index("subject")
    listed = []
    for i, lid in enumerate(ids):
        if lid == "subject":
            continue
        byid[lid]["z"] = "under" if i < cut else "over"
        listed.append(byid[lid])
    # listed layers fill their own old slots in the new order; unlisted ones (subject_cut,
    # grain — not drawn as separate layers) stay exactly where they were
    it = iter(listed)
    P["layers"] = [next(it) if L["id"] in byid and L["id"] in ids else L for L in P["layers"]]
    P["decisions"].append({"t": time.strftime("%F %T"), "order": ids})
    save(d, P)
    if a.no_render:
        hist_event(d, "order", order=ids)
        print("saved (no render)")
        return
    render(d, P, event="order", order=ids)


def make_savepoint(d, P, name, n=None, final=None, kind="manual"):
    """savepoints/<NN>_<slug>/{piece.json, final.jpg} + index.json. n=0 is the start."""
    import shutil
    sd = d / "savepoints"
    sd.mkdir(exist_ok=True)
    idx_f = sd / "index.json"
    idx = json.loads(idx_f.read_text()) if idx_f.exists() else []
    if n is None:
        n = max([e["n"] for e in idx] + [0]) + 1
    if any(e["n"] == n for e in idx):
        return None
    slug = re.sub(r"[^\w-]+", "-", name.lower()).strip("-")[:40] or "save"
    out = sd / f"{n:02d}_{slug}"
    out.mkdir(exist_ok=True)
    (out / "piece.json").write_text(json.dumps(P, indent=2, ensure_ascii=False))
    src = Path(final) if final else d / "final.jpg"
    if src.exists():
        shutil.copyfile(src, out / "final.jpg")
    idx.append({"n": n, "name": name, "t": time.strftime("%F %T"), "dir": out.name, "kind": kind})
    idx.sort(key=lambda e: e["n"])
    idx_f.write_text(json.dumps(idx, indent=2, ensure_ascii=False))
    hist_event(d, "savepoint", n=n, name=name, kind=kind)
    print(f"Savepoint {n}: {out}")
    return n


def ensure_start(d):
    """Savepoint 0 = the piece as first rendered (rebuilt from history/000 for old pieces)."""
    h0j, h0i = d / "history" / "000.json", d / "history" / "000.jpg"
    if h0j.exists():
        make_savepoint(d, json.loads(h0j.read_text()), "start", n=0, final=h0i, kind="start")


def cmd_save(a):
    d, P = load(a.piece)
    ensure_start(d)
    make_savepoint(d, P, a.name)


def cmd_fav(a):
    """Final: full-res render + PSD, copy into shared/favorites with a favorites.json
    entry (reconstruction = restore this savepoint + render --full)."""
    import shutil, subprocess as sp
    d, P = load(a.piece)
    ensure_start(d)
    render(d, P, full=True, event="fav")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{Path(P['photo']).stem}__{P['recipe']}__{stamp}"
    n = make_savepoint(d, P, a.name or f"fav {stamp}", kind="fav")
    fav_dir = Path("~/.openclaw/workspace/shared/favorites").expanduser()
    shutil.copyfile(d / "final_full.jpg", fav_dir / f"{name}.jpg")   # copyfile: 9p rejects copy2
    try:
        commit = sp.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    fj = fav_dir / "favorites.json"
    favs = json.loads(fj.read_text()) if fj.exists() else {"favorites": []}
    model = None
    try:
        model = Path(P["photo"]).resolve().relative_to(
            Path("~/.openclaw/workspace/_photos").expanduser().resolve()).parts[0]
    except ValueError:
        pass
    favs["favorites"].append({
        "file": f"{name}.jpg", "source": P["photo"], "model": model,
        "style": f"compose: {P['recipe']} (wildness {P.get('wildness')})",
        "tool": "layer_compose", "score": None, "git_commit": commit,
        "favorited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": (f"cd ~/gitrep/manipulating-photos && ./scripts/workflows/layer_compose.py "
                    f"restore '{d}' --savepoint {n} && ./scripts/workflows/layer_compose.py "
                    f"render '{d}' --full"),
        "piece": str(d), "savepoint": n, "psd": str(psd_path(d, P)), "project": "photo-tools"})
    fj.write_text(json.dumps(favs, indent=2, ensure_ascii=False))
    print(f"Fav: {fav_dir / (name + '.jpg')}\nPSD: {psd_path(d, P)}")


def cmd_restore(a):
    d, _ = load(a.piece)
    idx = json.loads((d / "savepoints" / "index.json").read_text())
    e = next((x for x in idx if x["n"] == int(a.savepoint)), None)
    if e is None:
        raise SystemExit(f"no savepoint {a.savepoint}")
    P = json.loads((d / "savepoints" / e["dir"] / "piece.json").read_text())
    P["decisions"].append({"t": time.strftime("%F %T"), "restore": e["n"], "name": e["name"]})
    save(d, P)
    render(d, P, event="restore", savepoint=e["n"], name=e["name"])


# ---------------------------------------------------------------- choice sets
def cmd_options(a):
    d, P = load(a.piece)
    key, _, vals = a.vary.partition("=")
    # '|' separates options whenever a value itself has commas (lists, JSON, boxes);
    # splitting JSON on ',' produced fragment options that all rendered the same
    vals = [parse_val(v.strip()) for v in (vals.split("|") if "|" in vals else vals.split(","))]
    if key.endswith(".params") or any(isinstance(v, str) and v.lstrip().startswith("{") for v in vals):
        raise SystemExit("vary ONE param (e.g. stamp.center=A|B|C), not a whole params object")
    qd = d / "questions"
    qd.mkdir(exist_ok=True)
    qn = f"q{len(list(qd.glob('q*.json'))) + 1}"
    opts, tiles = [], []
    for letter, v in zip(LETTERS, vals):
        Q = copy.deepcopy(P)
        apply_set(Q, key, v)
        final, _ = build(d, Q)
        img = to_img(final)
        fp = qd / f"{qn}_{letter}.jpg"
        img.save(fp, quality=90)
        opts.append({"letter": letter, "value": v, "file": fp.name})
        tiles.append((letter, v, img))
        print(f"  {letter}: {key}={v}")
    sheet = choice_sheet(a.question, key, tiles)
    sheet.save(qd / f"{qn}_sheet.jpg", quality=88)
    q = {"id": qn, "question": a.question, "key": key, "options": opts,
         "status": "open", "t": time.strftime("%F %T")}
    (qd / f"{qn}.json").write_text(json.dumps(q, indent=2, ensure_ascii=False))
    n = hist(d, P, "question", question=a.question, key=key, values=vals, qid=qn)
    sheet.convert("RGB").save(d / "history" / f"{n:03d}.jpg", quality=85)
    print(f"Question {qn}: {a.question}")
    print(f"Final: {qd / f'{qn}_sheet.jpg'}")


def choice_sheet(question, key, tiles):
    th = 900
    ims = []
    for letter, v, im in tiles:
        im = im.copy()
        im.thumbnail((th, th))
        ims.append((letter, v, im))
    W = sum(i.width for *_, i in ims) + 20 * (len(ims) + 1)
    H = max(i.height for *_, i in ims) + 150
    sheet = Image.new("RGB", (W, H), (22, 22, 22))
    dr = ImageDraw.Draw(sheet)
    f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    f2 = ImageFont.truetype(wl.FONTS["mono"], 22)
    dr.text((20, 18), question, fill=(240, 240, 240), font=f1)
    x = 20
    for letter, v, im in ims:
        sheet.paste(im, (x, 70))
        dr.rectangle([x, 70, x + 56, 120], fill=(230, 57, 70))
        dr.text((x + 16, 74), letter, fill=(0, 0, 0), font=f1)
        dr.text((x, 80 + im.height), f"{key.split('.')[-1]} = {v}", fill=(180, 180, 180), font=f2)
        x += im.width + 20
    return sheet


def cmd_pick(a):
    d, P = load(a.piece)
    qf = d / "questions" / f"{a.question}.json"
    q = json.loads(qf.read_text())
    opt = next(o for o in q["options"] if o["letter"] == a.choice.upper())
    apply_set(P, q["key"], opt["value"])
    q.update(status="answered", choice=opt["letter"], comment=a.comment)
    qf.write_text(json.dumps(q, indent=2, ensure_ascii=False))
    P["decisions"].append({"t": time.strftime("%F %T"), "question": q["question"],
                           "key": q["key"], "choice": opt["value"], "comment": a.comment})
    save(d, P)
    render(d, P, event="pick", qid=q["id"], key=q["key"], value=opt["value"], comment=a.comment)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--photo", required=True)
    i.add_argument("--recipe", required=True)
    i.add_argument("--wildness", type=float, default=0.5)
    i.add_argument("--context", default="")
    i.add_argument("--crop", type=int)
    i.add_argument("--custom")
    i.add_argument("--disable")
    i.add_argument("--set", nargs="*", action="extend")
    i.add_argument("--seed", type=int)
    r = sub.add_parser("render"); r.add_argument("piece"); r.add_argument("--full", action="store_true")
    s = sub.add_parser("set"); s.add_argument("piece"); s.add_argument("kv", nargs="+"); s.add_argument("--note", default="")
    s.add_argument("--no-render", action="store_true")
    v = sub.add_parser("variant"); v.add_argument("piece"); v.add_argument("id")
    v.add_argument("--seed", type=int, nargs="*"); v.add_argument("--count", type=int, default=3)
    od = sub.add_parser("order"); od.add_argument("piece"); od.add_argument("ids"); od.add_argument("--no-render", action="store_true")
    sv = sub.add_parser("save"); sv.add_argument("piece"); sv.add_argument("--name", required=True)
    fv = sub.add_parser("fav"); fv.add_argument("piece"); fv.add_argument("--name", default="")
    rs = sub.add_parser("restore"); rs.add_argument("piece"); rs.add_argument("--savepoint", required=True)
    o = sub.add_parser("options"); o.add_argument("piece"); o.add_argument("--question", required=True); o.add_argument("--vary", required=True)
    rc = sub.add_parser("recrop"); rc.add_argument("piece"); rc.add_argument("--crop")
    rc.add_argument("--custom"); rc.add_argument("--boost", action="store_true"); rc.add_argument("--note", default="")
    ad = sub.add_parser("add"); ad.add_argument("piece"); ad.add_argument("--id", required=True)
    ad.add_argument("--kind", required=True); ad.add_argument("--z", default="over", choices=["under", "over"])
    ad.add_argument("--region", default="anywhere"); ad.add_argument("--param", nargs="*")
    ad.add_argument("--after", default=""); ad.add_argument("--theme", default="added")
    ad.add_argument("--why", default=""); ad.add_argument("--off", action="store_true")
    ad.add_argument("--no-render", action="store_true")
    p = sub.add_parser("pick"); p.add_argument("piece"); p.add_argument("--question", required=True)
    p.add_argument("--choice", required=True); p.add_argument("--comment", default="")
    a = ap.parse_args()
    {"init": cmd_init, "render": cmd_render, "set": cmd_set, "options": cmd_options,
     "pick": cmd_pick, "variant": cmd_variant, "order": cmd_order, "save": cmd_save,
     "restore": cmd_restore, "fav": cmd_fav, "add": cmd_add, "recrop": cmd_recrop}[a.cmd](a)


if __name__ == "__main__":
    main()
