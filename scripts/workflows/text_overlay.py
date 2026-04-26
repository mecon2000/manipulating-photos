#!/usr/bin/env python3
"""Overlay a literary line on a stylized image.

Quote source modes:
  --text "..."        explicit
  --text auto         pick via embedded literary DB (requires
                      literary_quotes.{json,npy} produced by build_quote_db.py)

Style:                quote (serif) | subtitle (sans)
Orientation:          horizontal | body (rotates to body axis from MediaPipe pose)
                      | manual <degrees>
Color:                opposite-luminance gray sampled from local bbox.
Placement:            score multiple candidate bboxes by luminance variance
                      (low = flatter light/shadow → easier to read), pick best.
                      If even the best is mixed, add subtle outer glow.

Used standalone or imported by surreal_with_face.py as step 4.5.
"""
import argparse, json, os, random, re, sys, time, urllib.request
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

POSE_MODEL = Path("~/openclaw-venv/mediapipe_models/pose_landmarker.task").expanduser()
QUOTES_JSON = Path(__file__).resolve().parent / "literary_quotes.json"
QUOTES_NPY  = Path(__file__).resolve().parent / "literary_quotes.npy"
QUOTES_META = Path(__file__).resolve().parent / "literary_quotes.meta.json"

# System fonts (DejaVu ships everywhere). Replaceable via --font-quote / --font-subtitle.
DEFAULT_QUOTE_FONT    = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
DEFAULT_SUBTITLE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ---- Pick a quote --------------------------------------------------------

def load_key():
    k = os.environ.get("GOOGLE_API_KEY")
    if k: return k
    env = Path("~/sol/.env").expanduser()
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def gemini_mood_phrase(image_path, api_key):
    """Return a short free-text mood phrase for the image."""
    import base64
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    body = {
      "contents": [{"parts": [
        {"text": "Describe this image's emotional mood in one short phrase "
                 "(max 8 words). Use evocative literary language, not generic "
                 "labels. Just the phrase, no preamble."},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
      ]}],
      "generationConfig": {"maxOutputTokens": 60, "temperature": 0.4},
      "safetySettings": [
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT",         "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_NONE"},
      ],
    }
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={api_key}")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    if "candidates" not in d:
        # blocked despite safety relaxation — fall back to generic
        reason = d.get("promptFeedback", {}).get("blockReason", "unknown")
        print(f"  [text] mood-phrase blocked ({reason}) — using generic")
        return "intimate solitude, soft sorrow, longing"
    return d["candidates"][0]["content"]["parts"][0]["text"].strip().strip('"')


def pick_quote_auto(image_path, top_k=20):
    """Semantic NN over local literary DB."""
    if not (QUOTES_JSON.is_file() and QUOTES_NPY.is_file()):
        raise RuntimeError(
            f"literary DB not built yet. Run build_quote_db.py first, or use --text \"...\"")
    api_key = load_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing — needed for mood phrase")
    mood = gemini_mood_phrase(image_path, api_key)
    print(f"  [text] mood: {mood!r}")
    from sentence_transformers import SentenceTransformer
    meta = json.loads(QUOTES_META.read_text())
    m = SentenceTransformer(meta["model"])
    q_vec = m.encode([mood], normalize_embeddings=True)[0]
    db_vecs = np.load(QUOTES_NPY)             # (N, 384), pre-normalized
    sims = db_vecs @ q_vec                     # cosine since both normalized
    top_idx = np.argsort(-sims)[:top_k]
    lines = json.loads(QUOTES_JSON.read_text())
    pick = lines[int(random.choice(top_idx))]
    print(f"  [text] picked: \"{pick['text']}\"  ({pick.get('author','')})")
    return pick["text"]


# ---- MediaPipe pose for body axis -----------------------------------------

def detect_pose(rgb_arr):
    import mediapipe as mp
    H, W = rgb_arr.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_arr.astype(np.uint8))
    base = mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL))
    opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
    det = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    res = det.detect(mp_img); det.close()
    if not res.pose_landmarks: return None
    lms = res.pose_landmarks[0]
    return {"W": W, "H": H, "lms": lms,
            "P": lambda i: (lms[i].x * W, lms[i].y * H)}


def body_axis_degrees(pose):
    """Angle (degrees, image convention 0=horizontal) of the shoulder line."""
    if pose is None: return 0.0
    P = pose["P"]
    ls, rs = np.array(P(11)), np.array(P(12))
    v = ls - rs                         # right shoulder → left shoulder
    return float(np.degrees(np.arctan2(v[1], v[0])))


def face_bbox(pose):
    if pose is None: return None
    P = pose["P"]
    pts = np.array([P(0), P(2), P(5), P(7), P(8), P(9), P(10)])
    return (pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max())


# ---- Placement -----------------------------------------------------------

def candidate_bboxes(W, H, pose, text_w, text_h):
    """Return a list of (x, y) candidate top-left positions, ordered preferred-first.

    Heuristic: if face detected, place opposite the face horizontally at chin
    height; otherwise four corner / center candidates.
    """
    pad = int(min(W, H) * 0.04)
    out = []
    if pose:
        bb = face_bbox(pose)
        if bb:
            fx0, fy0, fx1, fy1 = bb
            face_cx = (fx0 + fx1) / 2
            chin_y = fy1                              # chin line approx
            # opposite-face horizontal at chin height
            if face_cx > W / 2:
                # face on right → text on left
                x_left = pad
            else:
                x_left = W - text_w - pad
            out.append((int(x_left), int(min(chin_y, H - text_h - pad))))
            # also try below-knees
            P = pose["P"]
            try:
                knee_y = max(P(25)[1], P(26)[1])
                out.append((max(pad, (W - text_w)//2), int(min(knee_y + pad, H - text_h - pad))))
            except Exception:
                pass
    # generic fallbacks (4 corners + center-bottom)
    out += [
        (pad, pad),
        (W - text_w - pad, pad),
        (pad, H - text_h - pad),
        (W - text_w - pad, H - text_h - pad),
        (max(pad, (W - text_w)//2), H - text_h - pad),
    ]
    # filter in-bounds
    return [(x, y) for (x, y) in out if 0 <= x <= W - text_w and 0 <= y <= H - text_h]


def bbox_score(luma, x, y, w, h):
    """Lower is better. Penalize variance (mixed light/shadow)."""
    patch = luma[y:y+h, x:x+w]
    return float(patch.std())


def text_color(mean_l):
    """Opposite-luminance gray, clamped 40-220."""
    g = int(round(255 - mean_l))
    return (max(40, min(220, g)),) * 3


# ---- Render --------------------------------------------------------------

def render_text(pil_rgb, text, font_path, font_size, pos, color, glow=False, glow_radius=3):
    """Compose text (with optional outer glow) onto pil_rgb. Returns new RGB image."""
    out = pil_rgb.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype(font_path, font_size)
    if glow:
        # Layer-based glow: draw text in opposite-direction gray on a transparent
        # canvas, blur, paste underneath. Simpler: draw shadow text 4 directions
        # each with low opacity, then text on top. PIL doesn't have a clean
        # outer-glow primitive — emulate via Gaussian-blurred text alpha.
        glow_layer = Image.new("RGBA", pil_rgb.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer)
        # opposite-of-text gray for the glow (so glow shows on the edge)
        glow_color = (255 - color[0], 255 - color[1], 255 - color[2], 200)
        gd.text(pos, text, font=font, fill=glow_color)
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(glow_radius))
        out = Image.alpha_composite(out.convert("RGBA"), glow_layer).convert("RGB")
        draw = ImageDraw.Draw(out)
    draw.text(pos, text, font=font, fill=color)
    return out


def overlay(rgb_arr, text, *, style="quote", orientation="horizontal",
             font_size_pct=3.0, manual_angle_deg=None,
             font_quote=DEFAULT_QUOTE_FONT, font_subtitle=DEFAULT_SUBTITLE_FONT,
             debug=False):
    """Apply text overlay. orientation in {horizontal, body, manual}."""
    H, W = rgb_arr.shape[:2]
    short = min(W, H)
    font_path = font_quote if style == "quote" else font_subtitle
    font_size = max(14, int(short * font_size_pct / 100.0))
    font = ImageFont.truetype(font_path, font_size)

    # Measure text
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
    except Exception:
        r, b = draw.textsize(text, font=font); l = t = 0
    text_w, text_h = (r - l), (b - t)

    # Decide orientation angle
    angle = 0.0
    pose = None
    if orientation in ("body", "horizontal"):
        try:
            pose = detect_pose(rgb_arr)
        except Exception as e:
            print(f"  [text] pose failed: {e}")
    if orientation == "body" and pose is not None:
        ang = body_axis_degrees(pose)
        # only auto-rotate if tilt > 5°; otherwise keep horizontal (looks weird at small angles)
        if abs(ang) > 5: angle = -ang   # negate: PIL rotation positive=CCW
    elif orientation == "manual" and manual_angle_deg is not None:
        angle = float(manual_angle_deg)

    # Pick best bbox by luminance-variance score
    luma = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[..., 0]
    cands = candidate_bboxes(W, H, pose, text_w, text_h)
    if not cands:
        # Fallback: just bottom-left
        cands = [(int(W*0.04), int(H*0.92 - text_h))]
    scored = [(bbox_score(luma, x, y, text_w, text_h), x, y) for (x, y) in cands]
    scored.sort()
    score, bx, by = scored[0]
    print(f"  [text] best bbox @ ({bx},{by}) std={score:.1f}, "
          f"font={Path(font_path).stem}, size={font_size}px, angle={angle:.1f}°")

    # Color: opposite-luminance gray
    mean_l = float(luma[by:by+text_h, bx:bx+text_w].mean())
    color = text_color(mean_l)
    glow = score > 25.0   # high variance (mixed light/shadow) → glow

    # Render. For non-zero angle, render text on a separate transparent canvas,
    # rotate, then paste with alpha.
    pil_rgb = Image.fromarray(rgb_arr)
    if abs(angle) < 0.5:
        out = render_text(pil_rgb, text, font_path, font_size, (bx, by), color,
                          glow=glow, glow_radius=int(font_size * 0.15))
    else:
        # Render onto transparent layer, rotate, paste
        layer = Image.new("RGBA", (text_w + 80, text_h + 80), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        if glow:
            glow_color = (255 - color[0], 255 - color[1], 255 - color[2], 200)
            gl = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            ImageDraw.Draw(gl).text((40 - l, 40 - t), text, font=font, fill=glow_color)
            gl = gl.filter(ImageFilter.GaussianBlur(int(font_size * 0.15)))
            layer = Image.alpha_composite(layer, gl)
            ld = ImageDraw.Draw(layer)
        ld.text((40 - l, 40 - t), text, font=font, fill=color + (255,))
        rot = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
        # paste onto pil_rgb at (bx, by) offset for new layer's bbox center
        paste_x = bx - (rot.size[0] - text_w) // 2
        paste_y = by - (rot.size[1] - text_h) // 2
        out_rgba = pil_rgb.convert("RGBA")
        out_rgba.alpha_composite(rot, (paste_x, paste_y))
        out = out_rgba.convert("RGB")

    return np.asarray(out)


# ---- CLI -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--text", required=True,
                   help='quote text, or "auto" for semantic NN over literary DB')
    p.add_argument("--style", choices=["quote", "subtitle"], default="quote")
    p.add_argument("--orientation", choices=["horizontal", "body", "manual"],
                   default="body",
                   help="body = align with shoulder line if tilted >5°")
    p.add_argument("--manual-angle", type=float, default=None)
    p.add_argument("--font-size-pct", type=float, default=3.0,
                   help="font size as %% of short edge")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    img = Image.open(args.source).convert("RGB")
    arr = np.asarray(img)
    text = args.text
    if text == "auto":
        text = pick_quote_auto(args.source)

    out = overlay(arr, text, style=args.style, orientation=args.orientation,
                   manual_angle_deg=args.manual_angle,
                   font_size_pct=args.font_size_pct)
    out_path = args.out or str(Path(args.source).with_suffix(".__text.jpg"))
    Image.fromarray(out).save(out_path, quality=92)
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
