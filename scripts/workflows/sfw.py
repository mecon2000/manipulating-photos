#!/usr/bin/env python3
"""SFW pre-filter for Instagram: reject the obvious, flag the rest for a human.

Rule (consent doc): no exposed nipples, genitals or full buttocks; implied
nudity is fine; one person in frame unless every participant consented.

This is a FILTER, not a judge. Measured against eyes on real packs:
  * NudeNet at 320px caught one exposure that 640px missed, and 640px caught
    things 320px missed — so both run and a hit in either counts.
  * Neither caught body paint over bare breasts.
  * A person detector counted a man entangled with the model as one person.
So every image gets a verdict of "fail", "check" or "pass", and nothing that
isn't "fail" is treated as cleared — a human confirms.
"""
import os
from functools import lru_cache

EXPOSED = {"FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
           "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"}
SKIN_HINT = {"FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
             "FEMALE_GENITALIA_COVERED", "BELLY_EXPOSED"}
OD = os.path.expanduser("~/openclaw-venv/mediapipe_models/efficientdet_lite2.tflite")


@lru_cache(maxsize=1)
def _models():
    from nudenet import NudeDetector
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    od = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=OD), score_threshold=0.35,
        category_allowlist=["person"], max_results=6))
    return NudeDetector(inference_resolution=320), NudeDetector(inference_resolution=640), od


def check(path):
    """-> dict(verdict, reasons, people). `path` must be an image file."""
    import mediapipe as mp
    n320, n640, od = _models()
    reasons, fail, check_ = [], False, False
    hits = {}
    for det in (n320, n640):
        for x in det.detect(str(path)):
            c, s = x["class"], x["score"]
            if c in EXPOSED and s >= 0.30:
                hits[c] = max(hits.get(c, 0), s)
            elif c in SKIN_HINT and s >= 0.45:
                check_ = True
            elif c in ("FACE_MALE", "MALE_BREAST_EXPOSED") and s >= 0.45:
                # the person detector merged an entangled couple into one; a
                # man's face or chest is the cheap tell that someone else is there
                check_ = True
                if "a man may be in frame" not in reasons:
                    reasons.append("a man may be in frame")
    for c, s in sorted(hits.items()):
        name = c.replace("_EXPOSED", "").lower()
        if s >= 0.50:
            fail = True
            reasons.append(f"{name} exposed ({s:.2f})")
        else:
            check_ = True
            reasons.append(f"possible {name} ({s:.2f})")
    im = mp.Image.create_from_file(str(path))
    people = sum(1 for d in od.detect(im).detections
                 if d.bounding_box.height > im.height * 0.25)
    if people > 1:
        fail = True
        reasons.append(f"{people} people in frame")
    if check_ and not reasons:
        reasons.append("skin-heavy — confirm")
    verdict = "fail" if fail else ("check" if check_ else "pass")
    return dict(verdict=verdict, reasons=reasons, people=people)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = check(p)
        print(f"{r['verdict']:5s} {os.path.basename(p)}  {'; '.join(r['reasons'])}")
