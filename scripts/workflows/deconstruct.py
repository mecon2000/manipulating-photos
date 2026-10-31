#!/home/rong/openclaw-venv/bin/python3
"""Reverse-engineer a reference image into a layer recipe (see plans/deconstruct.md).

  deconstruct.py                        # every shared/refs/<technique>/ subfolder -> recipe <technique>
  deconstruct.py --dir DIR --name NAME  # another folder of refs (one technique)
  deconstruct.py --ref a.jpg [b.jpg]    # explicit refs, one recipe
  deconstruct.py --force                # redo even if the recipe exists

Several refs of the same technique are read together, so the recipe captures what
they share (and per-ref variations), not one image's accidents.
Writes shared/deconstruct/recipes/<name>.json and <name>__breakdown.jpg.
Vision: Claude via the local `claude -p` CLI on the subscription (ANTHROPIC_API_KEY
unset for that call — a set key shadows the login and is credit-exhausted);
Gemini 2.5 Flash fallback. The recipe describes *how* the look is built, not the
ref's pixels — compose generates its own elements.
"""
import argparse, base64, io, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SHARED = Path("~/.openclaw/workspace/shared").expanduser()
ROOT = SHARED / "deconstruct"
REFS, RECIPES = SHARED / "refs", ROOT / "recipes"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

KINDS = ["paint_pattern", "calligraphy", "splatter", "ink_smear", "dot_rows", "pen_loops",
         "scribbles", "paper_scraps", "micro_text", "skin_overlay", "subject_cut",
         "scratches", "stain", "grain"]
REGIONS = ["behind-head", "around", "left", "right", "top", "bottom", "edges", "anywhere"]
BW_PRESETS = ["tri-x-punch", "high-key", "low-key", "soft-matte", "newsprint"]

PROMPT = f"""You are reverse-engineering a mixed-media portrait so a pipeline can rebuild
the same LOOK on a different photo. Read these {{n}} reference image(s) — they all use
ONE technique; capture what they share, note per-ref variation in "note":
{{paths}}

Break it into an ordered stack of layers, bottom to top. Describe how each is made
(medium, color, density, where it sits relative to the subject), not what the
specific marks are — we will generate our own.

Return ONLY a JSON object, no prose, no code fence:
{{{{
 "summary": "one sentence: the look in photographer/painter terms",
 "palette": ["#hex", ...],                       // 3-6 dominant colors
 "aspect": "4:5" | "1:1" | "9:16" | "3:4" | "other",
 "base": {{{{"bw": true|false, "preset": one of {BW_PRESETS},
            "contrast": 1.0-2.0, "grain": 0.0-0.15, "note": "tone treatment"}}}},
 "background": {{{{"mode": "flat"|"keep"|"texture", "color": "#hex", "note": "..."}}}},
 "layers": [
   {{{{"id": "short_snake_case", "kind": one of {KINDS},
      "z": "under"|"over",                       // under = behind the subject
      "region": one of {REGIONS},
      "ref": int,                                // 1-based index of the ref that shows it best
      "bbox": [x0, y0, x1, y1],                  // normalized 0-1 in THAT ref
      "params": {{{{"color": "#hex", "coverage": 0-1, "opacity": 0-1, ...kind-specific:
                  paint_pattern: "drips": bool, "pattern": "words describing the marks";
                  calligraphy: "script": "blackletter|gothic|nordic runes|japanese|latin brush",
                               "words": "suggested text or theme", "orientation": "vertical|horizontal";
                  dot_rows: "dot": "halftone|braille", "rows": int;
                  pen_loops: "weight": "hairline|thin", "count": int;
                  micro_text: "content": "tiny typewriter/technical captions";
                  skin_overlay: "content": "what is overlaid on the subject's skin (text, tattoo,
                               ink texture)", "blend": "multiply|overlay";
                  subject_cut: "how the subject is cut/sliced/displaced";
                  splatter: "dot_count": "few|many", "size": "small|large";
                  scribbles: "subject": "what is doodled, e.g. groups of eyes",
                             "medium": "ballpoint|marker|charcoal";
                  paper_scraps: "count": int, "content": "what is on the paper";
                  stain: "type": "coffee|water"; ink_smear: "direction": "deg"}}}},
      "note": "why it works visually"}}}}
 ],
 "questions": ["things a human should decide when rebuilding this, 1-4 items"]
}}}}
Include every distinct decorative layer you see (typically 6-14), including any
treatment applied to the subject itself (skin_overlay, subject_cut)."""


def _env():
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def _prep(refs: list[Path], tmp: Path) -> list[Path]:
    outs = []
    for i, ref in enumerate(refs, 1):
        img = Image.open(ref).convert("RGB")
        img.thumbnail((1568, 1568))
        out = tmp / f"ref{i}.jpg"          # plain names: refs often contain spaces/&
        img.save(out, quality=92)
        outs.append(out)
    return outs


def _prompt(paths) -> str:
    listing = "\n".join(f"  {i}. {p}" for i, p in enumerate(paths, 1))
    return PROMPT.format(n=len(paths), paths=listing)


def _parse(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def ask_claude(imgs: list[Path], model: str) -> dict:
    cmd = ["claude", "-p", _prompt(imgs), "--model", model,
           "--allowedTools", "Read", "--add-dir", str(imgs[0].parent),
           "--output-format", "text"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(),
                       cwd=str(imgs[0].parent))
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed: {(r.stderr or r.stdout)[-400:]}")
    return _parse(r.stdout)


def ask_gemini(imgs: list[Path]) -> dict:
    import requests
    key = os.environ.get("GOOGLE_API_KEY") or _dotenv().get("GOOGLE_API_KEY")
    parts = [{"text": _prompt([f"(attached image {i})" for i in range(1, len(imgs) + 1)])}]
    for p in imgs:
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(p.read_bytes()).decode()}})
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"response_mime_type": "application/json"},
            "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]}
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={key}")
    j = requests.post(url, json=body, timeout=180).json()
    return _parse(j["candidates"][0]["content"]["parts"][0]["text"])


def _dotenv():
    out = {}
    p = Path("~/sol/.env").expanduser()
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def normalize(recipe: dict, refs: list[Path], name: str) -> dict:
    recipe = {"schema": 1, "name": name, "refs": [str(r) for r in refs], **recipe}
    recipe.setdefault("base", {}).setdefault("preset", "tri-x-punch")
    recipe.setdefault("background", {"mode": "flat", "color": "#f1eee8"})
    seen = set()
    for i, L in enumerate(recipe.get("layers", [])):
        if L.get("kind") not in KINDS:
            L["kind_raw"], L["kind"] = L.get("kind"), "paint_pattern"
        L["z"] = L.get("z") if L.get("z") in ("under", "over") else "over"
        L["region"] = L.get("region") if L.get("region") in REGIONS else "anywhere"
        L.setdefault("params", {})
        L["enabled"] = True
        try:
            L["ref"] = max(1, min(len(refs), int(L.get("ref", 1))))
        except (TypeError, ValueError):
            L["ref"] = 1
        lid = L.get("id") or f"{L['kind']}_{i}"
        while lid in seen:
            lid += "_b"
        seen.add(lid)
        L["id"] = lid
        bb = L.get("bbox") or [0, 0, 1, 1]
        L["bbox"] = [max(0.0, min(1.0, float(v))) for v in (list(bb) + [0, 0, 1, 1])[:4]]
    return recipe


COLORS = [(230, 57, 70), (42, 157, 143), (233, 196, 106), (69, 123, 157),
          (244, 162, 97), (131, 56, 236), (6, 214, 160), (255, 0, 110),
          (255, 255, 255), (120, 200, 255), (200, 255, 120), (255, 140, 200)]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except OSError:
        return ImageFont.load_default()


def breakdown_sheet(refs: list[Path], recipe: dict, out: Path):
    """Refs in a grid with numbered layer boxes, legend + open questions on the right."""
    cell = 900
    thumbs = []
    for r in refs:
        im = Image.open(r).convert("RGB")
        im.thumbnail((cell, cell))
        thumbs.append(im)
    cols = 2 if len(thumbs) > 1 else 1
    rows = (len(thumbs) + cols - 1) // cols
    grid_w, grid_h = cols * cell, rows * cell
    legend_w = 760
    sheet = Image.new("RGB", (grid_w + legend_w, max(grid_h, 1100)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    font, small = _font(26), _font(19)
    origins = []
    for i, im in enumerate(thumbs):
        ox = (i % cols) * cell + (cell - im.width) // 2
        oy = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (ox, oy))
        d.text((ox + 8, oy + 6), f"ref {i + 1}", fill=(255, 255, 0), font=font)
        origins.append((ox, oy, im.width, im.height))
    for i, L in enumerate(recipe["layers"]):
        c = COLORS[i % len(COLORS)]
        ox, oy, W, H = origins[L["ref"] - 1]
        x0, y0, x1, y1 = L["bbox"]
        box = [ox + x0 * W, oy + y0 * H, ox + x1 * W, oy + y1 * H]
        d.rectangle(box, outline=c, width=4)
        d.rectangle([box[0], box[1], box[0] + 38, box[1] + 32], fill=c)
        d.text((box[0] + 6, box[1] + 2), str(i + 1), fill=(0, 0, 0), font=font)
    x, y = grid_w + 24, 20
    for chunk in _wrap(recipe.get("summary", ""), 58):
        d.text((x, y), chunk, fill=(240, 240, 240), font=small); y += 24
    b = recipe.get("base", {})
    y += 8
    d.text((x, y), f"base: {'B&W ' if b.get('bw') else ''}{b.get('preset')}  contrast "
           f"{b.get('contrast')}  grain {b.get('grain')}", fill=(180, 180, 180), font=small)
    y += 24
    bg = recipe["background"]
    d.text((x, y), f"bg: {bg.get('mode')} {bg.get('color')}   palette: "
           + " ".join(recipe.get("palette", [])[:6]), fill=(180, 180, 180), font=small)
    y += 36
    for i, L in enumerate(recipe["layers"]):
        c = COLORS[i % len(COLORS)]
        d.rectangle([x, y + 3, x + 18, y + 21], fill=c)
        d.text((x + 28, y), f"{i + 1}. {L['id']}  [{L['kind']} · {L['z']} · {L['region']}]",
               fill=(240, 240, 240), font=small)
        y += 23
        for chunk in _wrap(L.get("note", ""), 64)[:2]:
            d.text((x + 28, y), chunk, fill=(150, 150, 150), font=small); y += 21
        y += 6
    if recipe.get("questions"):
        y += 6
        d.text((x, y), "Questions:", fill=(240, 200, 120), font=small); y += 24
        for q in recipe["questions"]:
            for chunk in _wrap("• " + q, 64)[:3]:
                d.text((x, y), chunk, fill=(220, 190, 140), font=small); y += 21
    sheet.save(out, quality=88)


def _wrap(s, n):
    words, lines, cur = (s or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def deconstruct(refs: list[Path], name: str, model: str, force: bool) -> Path:
    out_json = RECIPES / f"{name}.json"
    if out_json.exists() and not force:
        print(f"skip (recipe exists, --force to redo): {out_json}")
        return out_json
    tmp = Path(tempfile.mkdtemp(prefix="decon_"))
    try:
        small = _prep(refs, tmp)
        try:
            raw = ask_claude(small, model)
            raw["_vision"] = f"claude:{model}"
        except Exception as e:
            print(f"  claude failed ({str(e)[:200]}); falling back to Gemini")
            raw = ask_gemini(small)
            raw["_vision"] = "gemini-2.5-flash"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    recipe = normalize(raw, refs, name)
    RECIPES.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(recipe, indent=2, ensure_ascii=False))
    sheet = RECIPES / f"{name}__breakdown.jpg"
    breakdown_sheet(refs, recipe, sheet)
    print(f"Recipe: {out_json}  ({len(recipe['layers'])} layers, {recipe['_vision']})")
    print(f"Final: {sheet}")
    return out_json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="*", help="explicit ref image(s)")
    ap.add_argument("--dir", default=str(REFS), help="folder of refs (default shared/refs)")
    ap.add_argument("--name", help="recipe name (default: folder name)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = []
    if a.ref:
        refs = [Path(p).expanduser() for p in a.ref]
        jobs.append((a.name or refs[0].stem, refs))
    else:
        d = Path(a.dir).expanduser()
        imgs = lambda folder: sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
        if d == REFS:
            # shared/refs/<technique>/ — one subfolder per technique, one recipe each
            loose = imgs(d)
            if loose:
                print(f"WARN: {len(loose)} loose image(s) in {d} ignored — put them in a "
                      f"subfolder named after the technique")
            jobs = [(sub.name, imgs(sub)) for sub in sorted(d.iterdir())
                    if sub.is_dir() and not sub.name.startswith("_") and imgs(sub)]
        else:
            jobs.append((a.name or d.name, imgs(d)))
    if not jobs:
        print(f"No refs. Make a subfolder per technique under {REFS} and drop images in it")
        sys.exit(1)
    for name, refs in jobs:
        name = re.sub(r"[^\w.-]+", "_", name).strip("_")
        print(f"== {name}: {len(refs)} ref(s)")
        deconstruct(refs, name, a.model, a.force)


if __name__ == "__main__":
    main()
