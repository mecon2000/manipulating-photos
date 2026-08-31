#!/usr/bin/env python3
"""Judgement and bookkeeping around an IG reel candidate: SFW check, caption, log.

Split out of ig_reel.py, which does the rendering — that file was at the repo's
400-line ceiling and these concerns are separable anyway.
"""
import csv, json, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

CATALOG = Path("~/gitrep/photo-catalogging/data/photo-catalog.db").expanduser()
HUB = Path("~/gitrep/project-hub").expanduser()
LOG = Path("~/.openclaw/workspace/shared/ig-reels/log.csv").expanduser()

# YCrCb skin gate. Deliberately crude: this decides what to LOOK AT, never what is
# safe to post — the frame always goes to Ronnie before anything is published.
_SKIN_LO = np.array([0, 133, 77], np.uint8)
_SKIN_HI = np.array([255, 173, 127], np.uint8)


def skin_ratio(img_path, max_edge=900):
    """Fraction of the frame that reads as skin, ignoring anything outside a face-less
    body region is beyond a heuristic — so this is a flag, not a verdict."""
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((max_edge, max_edge))
    a = np.asarray(im)
    ycrcb = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, _SKIN_LO, _SKIN_HI)
    return float((mask > 0).mean())


def sfw_flags(img_path, warn=0.30, high=0.45):
    r = skin_ratio(img_path)
    if r >= high:
        return r, "SFW: HIGH skin area — check carefully before posting"
    if r >= warn:
        return r, "SFW: borderline skin area — confirm visually"
    return r, "SFW: looks clothed (heuristic only — still confirm)"


def face_count(img_path, max_edge=900):
    """How many faces are in the frame. Used to catch the photographer being in shot."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mediapipe as mp
    import face_align as FA
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((max_edge, max_edge))
    base = mp.tasks.BaseOptions(model_asset_path=str(FA.FACE_MODEL))
    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=base, num_faces=5)
    det = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(np.asarray(im))))
    det.close()
    return len(res.face_landmarks or [])


def caption_and_hooks(model, session_name, fmt, total_frames, handle=None):
    """Caption + alternative hooks from the local claude CLI, with a usable fallback.

    Generation can fail (CLI missing, timeout); a candidate is still worth having, so
    the fallback is a real caption rather than an error string.
    """
    fallback = {
        "caption": (f"Shot {total_frames} frames that day. This is the one that made it — "
                    f"and what it looked like before I touched it. "
                    f"Which would you have picked?"),
        "hooks": ["what the camera saw / what I sent her",
                  "before the edit / after the edit",
                  f"{total_frames} frames, one keeper"],
        "keywords": ["portrait", "before and after", "photography", "editing"],
        "source": "fallback",
    }
    try:
        sys.path.insert(0, str(HUB))
        from hub.llm import run_claude
    except Exception:
        return fallback
    kind = "a raw file next to the finished edit" if fmt == "A" else \
           "a run of near-identical frames ending on the keeper" if fmt == "B" else \
           "behind-the-scenes video cut to the finished photo"
    prompt = (
        f"Write an Instagram caption for a reel by a portrait photographer.\n"
        f"The reel shows {kind}. The session had {total_frames} frames.\n"
        f"Voice: first person, plain, understated, no hype, no emoji, no hashtags.\n"
        f"It must end with a genuine question. Two sentences maximum.\n"
        f"{'Credit the model as ' + handle if handle else 'Do not name the model.'}\n"
        f"Then give 3 alternative one-line on-screen hooks (max 5 words each), "
        f"and 4 plain keywords.\n"
        f'Reply as JSON only: {{"caption": "...", "hooks": ["..."], "keywords": ["..."]}}')
    try:
        out = run_claude(prompt, timeout=90)
        m = re.search(r"\{.*\}", out or "", re.S)
        if not m:
            return fallback
        d = json.loads(m.group(0))
        if not d.get("caption"):
            return fallback
        d.setdefault("hooks", fallback["hooks"])
        d.setdefault("keywords", fallback["keywords"])
        d["source"] = "claude"
        return d
    except Exception:
        return fallback


def write_log(row):
    """Append one candidate to the run log; status starts as 'candidate'."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "model", "session", "format", "set",
                                          "files", "skin_ratio", "faces", "status"])
        if new:
            w.writeheader()
        w.writerow(row)


def write_sidecar(mp4_path, data):
    """Sidecar next to the mp4 so the hub card can show hooks, consent and SFW state."""
    Path(str(mp4_path) + ".json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def tag_boldness(session_dir, stem, value, source="ronnie-review"):
    """Record an SFW/NSFW judgement as a catalog boldness tag.

    The dimension covered 511 of 130,760 photos, so the automatic gate had almost
    nothing to read; every judgement widens it. Never overwrites an earlier human call.
    """
    import sqlite3
    con = sqlite3.connect(str(CATALOG))
    row = con.execute(
        """SELECT p.id FROM photos p JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.filename LIKE ?""",
        (f"%{session_dir.name}%", f"{stem}.%")).fetchone()
    if not row:
        con.close()
        return False
    pid = row[0]
    prior = con.execute(
        "SELECT value, source FROM photo_tags WHERE photo_id=? AND dimension='boldness'",
        (pid,)).fetchone()
    if prior and prior[0] != value and prior[1] not in (None, "", source):
        con.close()
        return False
    con.execute("DELETE FROM photo_tags WHERE photo_id=? AND dimension='boldness'", (pid,))
    con.execute("INSERT INTO photo_tags (photo_id, dimension, value, source) VALUES (?,?,?,?)",
                (pid, "boldness", value, source))
    con.commit(); con.close()
    return True


def propose(title, body, image_path):
    """Push the actual frame to the phone. A filename is not something you can judge."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from notify import push_image
        return bool(push_image(str(image_path), title=title, body=body))
    except Exception as e:
        print(f"  (no push: {e})")
        return False
