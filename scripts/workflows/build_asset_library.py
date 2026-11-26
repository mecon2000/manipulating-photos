#!/home/rong/openclaw-venv/bin/python3
"""Build the compose cut-out library: ~120 varied elements (botanical, creatures, classical
fragments, objects, celestial, ephemera…), each generated once on white (Flux dev on fal)
and cut out with BiRefNet into a real-alpha RGBA PNG.

  shared/deconstruct/assets/library/<category>/<slug>.png   + library/index.json

Riffs pick from it (riff.py offers the least-used items, so pieces don't repeat the same
wolf), and a layer uses one with kind "asset", params.library = "<category>/<slug>".
Idempotent: existing files are skipped. ~$0.02/item (Flux dev 768² + BiRefNet).

  build_asset_library.py              # build what's missing
  build_asset_library.py --list       # show categories + counts
  build_asset_library.py --only birds # one category
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wall_layers as wl  # noqa: E402

LIB = wl.ASSETS / "library"

ENGRAVE = "antique copperplate engraving illustration, fine black ink hatching"
PHOTO = "studio product photograph, crisp detail"
PAINT = "painted gouache illustration, bold flat colour"

LIBRARY = {
    "botanical": (ENGRAVE, [
        "a pressed fern frond", "a single peony bloom with leaves", "a magnolia branch with two flowers",
        "a dried rose with long stem", "a poppy seed head on a curved stem", "a sprig of eucalyptus",
        "a monstera leaf", "a lotus flower", "a spiky thistle", "a dandelion clock seed head",
        "a trailing ivy vine", "a moth orchid stem", "a lily of the valley sprig", "a pomegranate split open",
        "a fig cut in half", "a bunch of grapes with a vine leaf",
    ]),
    "creatures": (ENGRAVE, [
        "a luna moth with wings spread", "a stag beetle", "a dragonfly", "a morpho butterfly",
        "a honey bee", "a cicada", "a scorpion", "a snail shell spiral", "a jellyfish with long tentacles",
        "a seahorse", "a curling octopus tentacle", "a koi fish", "a coiled snake", "a chameleon on a twig",
    ]),
    "birds": (ENGRAVE, [
        "a crow in flight", "a swallow diving", "a grey heron standing", "a single peacock feather",
        "a barn owl with wings open", "a hummingbird", "a raven skull", "a pair of white dove wings",
        "a flamingo", "a magpie on a branch",
    ]),
    "animals": (ENGRAVE, [
        "a hare leaping", "a pair of deer antlers", "a horse head in profile", "a black panther walking",
        "a tiger head", "a ram skull with curled horns", "a sleeping cat curled", "a swan gliding",
        "a jackal head in profile", "a bat with wings spread",
    ]),
    "classical": (PHOTO + ", white marble", [
        "a marble statue hand reaching", "a broken marble bust fragment of a woman", "a fluted greek column fragment",
        "a marble angel wing", "a headless marble venus torso", "a cracked marble face mask",
        "a roman bronze coin", "a marble foot fragment", "a stone cherub", "a weathered stone lion head",
    ]),
    "anatomy": (ENGRAVE + ", vintage medical textbook plate", [
        "an anatomical human heart", "a human eye cross-section", "a hand skeleton",
        "a pair of lungs", "a human brain side view", "a ribcage", "a human skull in profile",
        "a spine vertebrae column",
    ]),
    "objects": (PHOTO, [
        "an antique brass key", "an open gold pocket watch", "a strip of 35mm film negatives",
        "a blank vintage polaroid frame", "an ornate gilded hand mirror", "a single chandelier crystal drop",
        "a pearl necklace", "a cut-glass perfume bottle", "a red stiletto high heel", "a red lipstick open",
        "a silk ribbon bow", "a white chess queen", "a red wax seal", "a pair of dice", "a vintage fountain pen",
        "an old rotary telephone receiver", "a pair of scissors open", "a lit match", "a cracked teacup",
        "a vintage glass syringe", "a spool of red thread", "a feather quill", "a brass compass",
        "a broken mirror shard", "a dark red velvet glove",
    ]),
    "celestial": (ENGRAVE, [
        "a crescent moon with a sleeping face", "a sun with a face and rays", "the planet saturn with rings",
        "a comet with a long tail", "a constellation star chart fragment", "an eclipse", "a cluster of stars",
        "an armillary sphere",
    ]),
    "architecture": (ENGRAVE, [
        "a gothic arched window", "a spiral staircase", "a stone archway", "an ornate baroque frame corner",
        "a wrought iron gate", "a classical temple facade", "a lighthouse", "a crumbling brick wall fragment",
    ]),
    "ephemera": (PHOTO + ", aged paper", [
        "a torn vintage postage stamp with no text", "a torn theatre ticket stub", "a torn sheet music fragment",
        "a torn antique map fragment", "a lace doily", "a torn velvet fabric swatch",
        "a torn vintage botanical label", "a torn page of a dictionary", "a playing card back with ornate pattern",
        "a torn tarot card with a moon illustration and no text",
    ]),
    "paint": (PAINT, [
        "one thick impasto brush stroke in cadmium red", "one thick impasto brush stroke in cobalt blue",
        "one thick impasto brush stroke in hot pink", "one thick impasto brush stroke in ochre yellow",
        "one thick impasto brush stroke in black", "one thick impasto brush stroke in white",
        "a torn piece of paper painted magenta with heavy brush strokes",
        "a torn piece of paper painted ultramarine with heavy brush strokes",
        "a gold leaf fragment", "a splash of red paint",
    ]),
    "sea": (ENGRAVE, [
        "a conch shell", "a scallop shell", "a branch of red coral", "a nautilus shell cut in half",
        "a starfish", "a sea urchin", "a single pearl in an open oyster",
    ]),
}


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:48]


def build_one(cat, style, item, seed):
    out = LIB / cat / f"{slug(item)}.png"
    if out.exists():
        return out, "skip"
    import masking
    prompt = (f"{item}, {style}, single isolated object centred, plain pure white background, "
              "no text, no frame, no border, nothing else in the image")
    im = wl._fal_flux(prompt, seed, 768, 768)
    m, _ = masking.build_mask(im, "subject", feather=0)
    a = np.asarray(m.convert("L").resize(im.size), np.uint8)
    rgba = np.dstack([np.asarray(im), a])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out)
    return out, "new"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=3, help="fal allows ~3 concurrent")
    a = ap.parse_args()
    if a.list:
        for c, (_, items) in LIBRARY.items():
            have = len(list((LIB / c).glob("*.png"))) if (LIB / c).exists() else 0
            print(f"{c:14s} {have}/{len(items)}")
        return
    jobs = [(c, st, it, 1000 + i) for c, (st, items) in LIBRARY.items() if not a.only or c == a.only
            for i, it in enumerate(items)]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build_one, *j) for j in jobs]
        for j, f in zip(jobs, futs):
            try:
                p, st = f.result()
                print(f"{st:4s} {j[0]}/{slug(j[2])}", flush=True)
            except Exception as e:
                print(f"FAIL {j[0]}/{slug(j[2])}: {str(e)[:160]}", flush=True)
    index = [{"id": f"{c}/{slug(it)}", "category": c, "what": it}
             for c, (_, items) in LIBRARY.items() for it in items if (LIB / c / f"{slug(it)}.png").exists()]
    (LIB / "index.json").write_text(json.dumps(index, indent=1))
    print(f"library: {len(index)} items → {LIB / 'index.json'}")


if __name__ == "__main__":
    main()
