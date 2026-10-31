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

Cropping is smart-crop.py's job (choose with `smart-crop.py --source P --show-options`).
"""
import argparse, copy, json, os, re, subprocess, sys, time, zlib
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
    if head in ("base", "background"):
        P[head][k] = val
        return
    for L in P["layers"]:
        if L["id"] == head:
            if k in ("z", "enabled", "seed", "region", "kind"):
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
    P = {"schema": 1, "photo": str(photo), "crop": {"n": a.crop, "custom": a.custom},
         "recipe": a.recipe, "seed": seed, "wildness": a.wildness,
         "palette": recipe.get("palette", []), "base": copy.deepcopy(recipe["base"]),
         "background": copy.deepcopy(recipe["background"]),
         "layers": copy.deepcopy(recipe["layers"]), "themes": [], "decisions": []}
    P["base"].setdefault("edge_dissolve", 0.45)
    for i, L in enumerate(P["layers"]):
        L.setdefault("seed", seed + 101 * (i + 1))
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


def subject_mask(d, im):
    mp = d / "mask.png"
    if not mp.exists():
        from masking import build_mask
        m, _ = build_mask(im, affect="subject", exclude="", output_dir=str(d), feather=0)
        m.convert("L").resize(im.size).save(mp)
    return Image.open(mp).convert("L")


def head_point(mask, sbox):
    x0, y0, x1, y1 = sbox
    band = mask[y0:y0 + max(2, (y1 - y0) // 6)]
    xs = np.nonzero(band > 0.5)[1]
    hx = xs.mean() if len(xs) else (x0 + x1) / 2
    return hx, y0 + (y1 - y0) * 0.1


def edge_dissolve(mask, amt, s, rng):
    if amt <= 0:
        return mask
    inside = distance_transform_edt(mask > 0.5)
    band = s * 0.035 * amt
    n = wl.fbm(mask.shape, rng, s * 0.006, 3)
    t = np.clip(inside / max(1, band), 0, 1)
    keep = np.clip(t * 1.6 - 0.3 + 0.6 * n * (1 - t), 0, 1)   # noise only in the edge band
    keep[t >= 1] = 1
    return (mask * keep).astype(np.float32)


def composite(dst, L):
    a = L.alpha[..., None]
    if L.blend == "multiply":
        return dst * (1 - a) + dst * L.rgb * a
    return dst * (1 - a) + L.rgb * a


def build(d, P, full=False):
    """Returns (final HxWx3 float, [(name, Layer)] bottom→top) at preview or full res."""
    src = Image.open(d / "source.jpg").convert("RGB")
    mask_im = subject_mask(d, src)
    if not full:
        src.thumbnail((PREVIEW, PREVIEW))
    W, H = src.size
    mask = np.asarray(mask_im.resize((W, H)), np.float32) / 255
    rgb = np.asarray(src, np.float32) / 255
    sc = float(P["base"].get("subject_scale", 1.0))
    if sc < 0.999:   # shrink her inside the canvas (bottom-anchored) — BG is paper anyway
        import cv2
        w2, h2 = int(W * sc), int(H * sc)
        r2 = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(mask, (w2, h2), interpolation=cv2.INTER_AREA)
        rgb, mask = np.ones_like(rgb), np.zeros_like(mask)
        x0 = (W - w2) // 2 + int(float(P["base"].get("subject_dx", 0)) * W)
        rgb[H - h2:, x0:x0 + w2], mask[H - h2:, x0:x0 + w2] = r2, m2
    s = min(W, H)
    rng0 = np.random.default_rng(P["seed"])
    base, bgc = P["base"], P["background"]
    lum = bw(rgb, base, mask) if base.get("bw", True) else None

    ys, xs = np.nonzero(mask > 0.5)
    sbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else (0, 0, W, H)
    ctx_args = dict(W=W, H=H, sbox=sbox, head=head_point(mask, sbox), mask=mask,
                    lum=lum if lum is not None else rgb.mean(2), palette=P.get("palette", []))

    paper = wl.hex_rgb(bgc.get("color", "#f1eee8"))
    out = np.empty((H, W, 3), np.float32)
    out[:] = paper
    tone = 1 + 0.025 * wl.fbm((H, W), rng0, s * 0.15, 3)       # paper unevenness
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.12 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    out *= (tone * vig)[..., None]
    if bgc.get("mode") == "keep":
        out = rgb.copy()
    stack = []

    def run(L):
        ctx = wl.Ctx(rng=np.random.default_rng(int(L.get("seed", 1))), **ctx_args)
        g = wl.GENERATORS.get(L["kind"])
        if g is None:
            return None
        return g(L, ctx)

    live = [L for L in P["layers"] if L.get("enabled", True)]
    for L in [x for x in live if x["z"] == "under"]:
        lay = run(L)
        if lay is not None:
            out = composite(out, lay)
            stack.append((L["id"], lay))

    # subject: B&W, highlights fall to paper (multiply onto paper colour), ragged edge
    m = edge_dissolve(mask, float(base.get("edge_dissolve", 0.45)), s, rng0)
    subj_rgb = (lum[..., None] * paper) if lum is not None else rgb
    cut = next((L for L in live if L["kind"] == "subject_cut"), None)
    if cut is not None:
        subj_rgb, m = slice_subject(subj_rgb, m, sbox, np.random.default_rng(int(cut.get("seed", 1))), s)
    subj = wl.Layer(subj_rgb.astype(np.float32), m)
    out = composite(out, subj)
    stack.append(("subject", subj))

    for L in [x for x in live if x["z"] == "over"]:
        lay = run(L)
        if lay is not None:
            out = composite(out, lay)
            stack.append((L["id"], lay))

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
    return np.clip(out, 0, 1), stack


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


def render(d, P, full=False, event="render", **meta):
    t = time.time()
    final, stack = build(d, P, full)
    img = to_img(final)
    img.save(d / ("final_full.jpg" if full else "final.jpg"), quality=93)
    ld = d / "layers"
    ld.mkdir(exist_ok=True)
    for f in ld.glob("*.png"):
        f.unlink()
    for i, (name, L) in enumerate(stack):
        rgba = np.dstack([L.rgb, L.alpha])
        Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(
            ld / f"{i:02d}_{name}.png")
    if full:
        from psd_layers import write_psd
        W, H = img.size
        base = np.asarray(Image.open(d / "source.jpg").convert("RGB"))
        layers = [{"name": "photo (original)", "rgb": base, "mask": np.zeros((H, W), np.uint8)}]
        for name, L in stack:
            layers.append({"name": name, "rgb": (np.clip(L.rgb, 0, 1) * 255).astype(np.uint8),
                           "mask": (np.clip(L.alpha, 0, 1) * 255).astype(np.uint8)})
        write_psd(str(d / "piece.psd"), layers, (W, H), np.asarray(img))
    n = hist(d, P, event, full=full, **meta)
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
    render(d, P, event="set", changes=changes, note=a.note)


# ---------------------------------------------------------------- choice sets
def cmd_options(a):
    d, P = load(a.piece)
    key, _, vals = a.vary.partition("=")
    vals = [parse_val(v) for v in vals.split(",")]
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
    i.add_argument("--set", nargs="*")
    i.add_argument("--seed", type=int)
    r = sub.add_parser("render"); r.add_argument("piece"); r.add_argument("--full", action="store_true")
    s = sub.add_parser("set"); s.add_argument("piece"); s.add_argument("kv", nargs="+"); s.add_argument("--note", default="")
    o = sub.add_parser("options"); o.add_argument("piece"); o.add_argument("--question", required=True); o.add_argument("--vary", required=True)
    p = sub.add_parser("pick"); p.add_argument("piece"); p.add_argument("--question", required=True)
    p.add_argument("--choice", required=True); p.add_argument("--comment", default="")
    a = ap.parse_args()
    {"init": cmd_init, "render": cmd_render, "set": cmd_set, "options": cmd_options,
     "pick": cmd_pick}[a.cmd](a)


if __name__ == "__main__":
    main()
