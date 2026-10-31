#!/home/rong/openclaw-venv/bin/python3
"""Riffs — creative decorations ADJACENT to a reference technique, grounded in the photo.

A recipe (deconstruct.py) says how the ref look is built. Imitating it 1:1 is the
weak version; this proposes extra layers that belong to the same visual language
but come from *this* photo / model's story (e.g. Elly: Soviet ration vouchers,
smoke because she smokes). Riffs are grouped into a few themes so the piece reads
as a story, plus a controlled number of unrelated "wildcards" (the refs do mix
unrelated things — that is part of the look, in moderation).

  riff.py --photo P.jpg --recipe red-calligraffiti --wildness 0.5 [--context "..."]

--wildness 0..1  (the "going wild" slider)
  0.0  no riffs — pure ref imitation
  0.3  1 theme, 2 riffs, tightly tied to the photo/model; same media as the ref
  0.5  2 themes, 3-4 riffs, + 1 loose wildcard
  0.8  2-3 themes, 5 riffs, 2 wildcards, freer media
  1.0  3 themes, 6+ riffs, several wildcards, surprising media
Context comes from --context plus shared/deconstruct/models/<Model>.md (notes on the
model: habits, background, what fits her). Prints JSON; compose.py merges it.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("~/.openclaw/workspace/shared/deconstruct").expanduser()
PHOTOS = Path("~/.openclaw/workspace/_photos").expanduser()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deconstruct import KINDS, REGIONS, _env, _parse  # noqa: E402

# kinds the compositor can render for riffs; "asset" ones are generated images
# (Flux on white, keyed to alpha) — the prompt describes the element itself.
RIFF_KINDS = KINDS + ["glyph_rain", "asset", "print"]


def model_name(photo: Path) -> str | None:
    try:
        rel = photo.resolve().relative_to(PHOTOS.resolve())
        return rel.parts[0]
    except ValueError:
        return None


def model_notes(name: str | None) -> str:
    if not name:
        return ""
    d = ROOT / "models"
    for cand in (d / f"{name}.md", *d.glob(f"{name.split(' ')[0]}*.md")):
        if cand.exists():
            return cand.read_text()
    return ""


def plan(wild: float) -> str:
    if wild <= 0.15:
        return "0 riffs."
    if wild <= 0.35:
        return "1 theme, 2 riffs, 0 wildcards. Tightly tied to the photo/model; same media as the ref."
    if wild <= 0.6:
        return "2 themes, 3-4 riffs total, plus 1 loosely related wildcard."
    if wild <= 0.85:
        return "2-3 themes, ~5 riffs, plus 2 wildcards; media may stray from the ref."
    return "3 themes, 6+ riffs, plus 3 wildcards; surprising media and scale are welcome."


PROMPT = """You are the creative director of a mixed-media portrait series. Look at the
photo at {photo}. We are rebuilding it in this technique (recipe JSON below), but pure
imitation is weak: propose ADJACENT decorations — same visual language (palette,
hand-made marks, collage, print ephemera), new content that comes from THIS photo
and model. Good riffs are grounded: something in the photo, her story, the era or
place the image evokes, or the mood. Group them into themes so the piece tells a
story; wildcards are deliberately unrelated (the refs mix unrelated things — keep
that in moderation).

Wildness {wild:.2f} -> {plan}
Model: {model}
Notes about the model / photo (from the photographer):
{notes}

Recipe (the technique):
{recipe}

Layer kinds the compositor can render: {kinds}.
- "asset": a generated element (illustration/ephemera/object) — give params.prompt
  describing ONLY the element, isolated on plain white paper, no people, e.g.
  "1980s Soviet bread ration coupon, perforated edge, faded print, stamp".
- TEXT MUST BE REAL. Generated images ("asset") render fake lettering, so never ask an
  asset for text. For printed ephemera with words use kind "print": params.variant one of
  tobacco_talon | sugar_talon | soap_talon | papirosy | stamp (stamp: params.ring = list of
  real words around the ring, params.center = real word). Any Russian must be correct Russian.
- "glyph_rain": columns of characters (params.charset: "katakana|latin|cyrillic|digits|ascii").
- "scribbles": params.subject = what is doodled, params.medium.
- "paper_scraps": params.content = what KIND of printed matter (texture only — it renders as
  unreadable small print; never name specific words, they come out misspelled).
Regions: {regions}. z: under (behind her) | over (on top of her).
Colors must stay inside the recipe palette unless wildness > 0.8.

Return ONLY JSON, no prose, no code fence:
{{"themes": [{{"name": "...", "why": "how it connects to the photo/model"}}],
  "riffs": [{{"id": "snake_case", "theme": "theme name or 'wildcard'", "kind": "...",
             "z": "under|over", "region": "...", "params": {{"color": "#hex",
             "coverage": 0-1, "opacity": 0-1, ...}}, "why": "one line"}}]}}"""


def riff(photo: Path, recipe: dict, wild: float, context: str = "", model: str = "sonnet") -> dict:
    if wild <= 0.15:
        return {"wildness": wild, "themes": [], "riffs": []}
    name = model_name(photo)
    notes = "\n".join(x for x in (model_notes(name), context) if x) or "(none)"
    slim = {k: recipe[k] for k in ("summary", "palette", "base", "background") if k in recipe}
    slim["layers"] = [{k: L[k] for k in ("id", "kind", "z", "region", "params")}
                      for L in recipe.get("layers", []) if L.get("enabled", True)]
    tmp = Path(tempfile.mkdtemp(prefix="riff_"))
    try:
        im = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
        im.thumbnail((1280, 1280))
        p = tmp / "photo.jpg"
        im.save(p, quality=90)
        prompt = PROMPT.format(photo=p, wild=wild, plan=plan(wild), model=name or "unknown",
                               notes=notes, recipe=json.dumps(slim, ensure_ascii=False),
                               kinds=", ".join(RIFF_KINDS), regions=", ".join(REGIONS))
        r = subprocess.run(["claude", "-p", prompt, "--model", model, "--allowedTools", "Read",
                            "--add-dir", str(tmp), "--output-format", "text"],
                           capture_output=True, text=True, timeout=600, env=_env(), cwd=str(tmp))
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout)[-400:])
        out = _parse(r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for R in out.get("riffs", []):
        R["kind"] = R.get("kind") if R.get("kind") in RIFF_KINDS else "asset"
        R["z"] = R.get("z") if R.get("z") in ("under", "over") else "over"
        R["region"] = R.get("region") if R.get("region") in REGIONS else "anywhere"
        R.setdefault("params", {})
        R["enabled"] = True
        R["riff"] = True
        R["note"] = R.pop("why", "")
    out["wildness"] = wild
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--recipe", required=True, help="recipe name or path")
    ap.add_argument("--wildness", type=float, default=0.5)
    ap.add_argument("--context", default="")
    ap.add_argument("--model", default="sonnet")
    a = ap.parse_args()
    rp = Path(a.recipe) if a.recipe.endswith(".json") else ROOT / "recipes" / f"{a.recipe}.json"
    print(json.dumps(riff(Path(a.photo).expanduser(), json.loads(rp.read_text()),
                          a.wildness, a.context, a.model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
