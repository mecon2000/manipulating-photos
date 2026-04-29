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


def rank_quotes_by_mood(mood, top_k=50):
    """Return ordered list of {text,author,title} ranked by similarity to mood."""
    if not (QUOTES_JSON.is_file() and QUOTES_NPY.is_file()):
        raise RuntimeError("literary DB not built yet")
    from sentence_transformers import SentenceTransformer
    meta = json.loads(QUOTES_META.read_text())
    m = SentenceTransformer(meta["model"])
    q_vec = m.encode([mood], normalize_embeddings=True)[0]
    db_vecs = np.load(QUOTES_NPY)
    sims = db_vecs @ q_vec
    order = np.argsort(-sims)[:top_k].tolist()
    lines = json.loads(QUOTES_JSON.read_text())
    return [lines[int(i)] for i in order], order


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

MIN_MARGIN = 40   # min pixels from any image edge


def candidate_bboxes(W, H, pose, text_w, text_h, margin=MIN_MARGIN):
    """Return a list of (x, y) candidate top-left positions, ordered preferred-first.
    All candidates respect at least `margin` px from any edge.
    """
    out = []
    if pose:
        bb = face_bbox(pose)
        if bb:
            fx0, fy0, fx1, fy1 = bb
            face_cx = (fx0 + fx1) / 2
            chin_y = fy1
            # opposite-face horizontal at chin height
            if face_cx > W / 2:
                x = margin                             # text on left
            else:
                x = W - text_w - margin                # text on right
            y = int(min(chin_y, H - text_h - margin))
            y = max(margin, y)
            out.append((int(x), int(y)))
            # also try below-knees, centered
            P = pose["P"]
            try:
                knee_y = max(P(25)[1], P(26)[1])
                cy = int(min(knee_y + margin, H - text_h - margin))
                cy = max(margin, cy)
                out.append((max(margin, (W - text_w)//2), cy))
            except Exception:
                pass
    # generic fallbacks (4 corners + center-bottom)
    out += [
        (margin, margin),
        (W - text_w - margin, margin),
        (margin, H - text_h - margin),
        (W - text_w - margin, H - text_h - margin),
        (max(margin, (W - text_w)//2), H - text_h - margin),
    ]
    # filter: must fit inside image with margin
    return [(x, y) for (x, y) in out
            if margin <= x <= W - text_w - margin
            and margin <= y <= H - text_h - margin]


def bbox_score(luma, x, y, w, h):
    """Lower is better. Penalize variance (mixed light/shadow)."""
    patch = luma[y:y+h, x:x+w]
    return float(patch.std())


def region_blur_score(luma, x, y, w, h):
    """Laplacian variance — higher = sharper, lower = blurrier."""
    patch = luma[y:y+h, x:x+w]
    return float(cv2.Laplacian(patch, cv2.CV_32F).var())


def text_blur_sigma_to_match(lapvar, sharper_factor=0.7):
    """Estimate how much to blur the text so it reads slightly sharper than
    the photo region behind it. Heuristic mapping LapVar → sigma:

      LapVar  ~50  (very blurry)  → sigma ≈ 2.5
      LapVar ~200  (mid)          → sigma ≈ 1.2
      LapVar ~1000 (sharp)        → sigma ≈ 0.3
      LapVar ~5000 (very sharp)   → sigma ≈ 0   (no blur)

    `sharper_factor` (0..1): 1.0 = match photo blur exactly,
                             0.7 = text is 30%% sharper than photo (default).
    """
    if lapvar <= 0: return 0.0
    sigma_photo_est = 16.0 / (lapvar ** 0.5 + 5.0)   # empirical, smooth
    sigma_text = sigma_photo_est * sharper_factor
    return float(max(0.0, min(3.0, sigma_text)))


def text_color_from_highlights(luma, top_pct=15):
    """Mean luminance of the brightest top_pct% of the image.
    Text gets this gray so it matches the photo's actual highlight tone
    (not pure white — feels integrated rather than pasted on)."""
    flat = luma.flatten()
    cutoff = np.percentile(flat, 100 - top_pct)
    highs = flat[flat >= cutoff]
    g = int(round(highs.mean()))
    return (g, g, g)


def wrap_text(text, font, max_width_px):
    """Wrap into 1-3 lines fitting max_width_px. Uses simple greedy word-wrap.
    Adds soft breaks at commas/em-dashes when line is still too long.
    Honors explicit linebreaks: "|" or literal "\\n" force a new line."""
    import textwrap
    # Normalize explicit breaks first so each segment wraps independently.
    raw_segments = re.split(r"\s*\|\s*|\\n|\n", text or "")
    raw_segments = [s for s in (seg.strip() for seg in raw_segments) if s]
    if len(raw_segments) > 1:
        out = []
        for seg in raw_segments:
            out.extend(wrap_text(seg, font, max_width_px))
        if len(out) > 3:
            out = out[:2] + [" ".join(out[2:])]
        return out
    words = (raw_segments[0] if raw_segments else text).split()
    if not words:
        return [text]

    # Greedy line-break by pixel width
    def measure(s):
        try:
            l, t, r, b = font.getbbox(s)
            return r - l
        except Exception:
            return font.getsize(s)[0]

    lines = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if measure(candidate) <= max_width_px:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # If we still have a single line over limit, force-split at ~half by punctuation
    if len(lines) == 1 and measure(lines[0]) > max_width_px:
        s = lines[0]
        for sep in [", ", "; ", " — ", " - "]:
            if sep in s:
                i = s.index(sep) + len(sep)
                lines = [s[:i].rstrip(), s[i:]]
                break

    # Cap at 3 lines: collapse extras by re-joining last segments
    if len(lines) > 3:
        lines = lines[:2] + [" ".join(lines[2:])]
    return lines


def count_candidate_bboxes(rgb_arr, text, *, font_size_pct=3.0, max_width_pct=42.0,
                            font_path=DEFAULT_QUOTE_FONT):
    """Return number of candidate bbox positions for a given text/image."""
    H, W = rgb_arr.shape[:2]
    short = min(W, H)
    font_size = max(14, int(short * font_size_pct / 100.0))
    font = ImageFont.truetype(font_path, font_size)
    pose = None
    try:
        pose = detect_pose(rgb_arr)
    except Exception:
        pass
    max_line_w = int(W * max_width_pct / 100.0)
    lines = wrap_text(text or "x", font, max_line_w)
    def measure(s):
        try:
            l, t, r, b = font.getbbox(s); return (r - l), (b - t)
        except Exception:
            return font.getsize(s)
    line_sizes = [measure(s) for s in lines]
    line_h = max(h for _, h in line_sizes)
    leading = int(line_h * 0.25)
    block_w = max(w for w, _ in line_sizes)
    block_h = len(lines) * line_h + (len(lines) - 1) * leading
    return len(candidate_bboxes(W, H, pose, block_w, block_h))


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
             max_width_pct=42.0, blur_match_factor=0.5, debug=False,
             force_bbox_idx=None, force_align=None,
             force_xy_pct=None):
    """Apply text overlay. Always horizontal by default.

    - Wraps long text into up to 3 lines fitting `max_width_pct`% of image width.
    - Bbox candidate must clear MIN_MARGIN from every edge.
    - Multiline alignment: flush to whichever side (L/R) is closer to the image border.
    - Color: mean luminance of the photo's top-15% highlights, so text matches
      the actual highlight tone (not pure white).
    - Faint outer glow only if the chosen bbox spans mixed light/shadow.
    """
    H, W = rgb_arr.shape[:2]
    short = min(W, H)
    font_path = font_quote if style == "quote" else font_subtitle
    font_size = max(14, int(short * font_size_pct / 100.0))
    font = ImageFont.truetype(font_path, font_size)

    # Decide orientation angle (default horizontal; --orientation body is opt-in)
    angle = 0.0
    pose = None
    try:
        pose = detect_pose(rgb_arr)
    except Exception as e:
        print(f"  [text] pose failed: {e}")
    if orientation == "body" and pose is not None:
        ang = body_axis_degrees(pose)
        if abs(ang) > 5: angle = -ang
    elif orientation == "manual" and manual_angle_deg is not None:
        angle = float(manual_angle_deg)

    # Wrap into lines fitting max_width_pct% of image
    max_line_w = int(W * max_width_pct / 100.0)
    lines = wrap_text(text, font, max_line_w)
    print(f"  [text] wrapped to {len(lines)} line(s): {lines}")

    # Measure block
    def measure(s):
        try:
            l, t, r, b = font.getbbox(s)
            return (r - l), (b - t)
        except Exception:
            return font.getsize(s)
    line_sizes = [measure(s) for s in lines]
    line_h = max(h for _, h in line_sizes)
    leading = int(line_h * 0.25)
    block_w = max(w for w, _ in line_sizes)
    block_h = len(lines) * line_h + (len(lines) - 1) * leading

    # Pick best bbox
    luma = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[..., 0]
    cands = candidate_bboxes(W, H, pose, block_w, block_h)
    if not cands:
        cands = [(MIN_MARGIN, max(MIN_MARGIN, H - block_h - MIN_MARGIN))]
    scored = [(bbox_score(luma, x, y, block_w, block_h), x, y) for (x, y) in cands]
    scored.sort()
    score, bx, by = scored[0]
    if force_bbox_idx is not None and 0 <= int(force_bbox_idx) < len(cands):
        bx, by = cands[int(force_bbox_idx)]
        score = bbox_score(luma, bx, by, block_w, block_h)
    if force_xy_pct is not None:
        # User clicked on the preview: place block CENTERED at that point,
        # then clamp inside MIN_MARGIN.
        xp, yp = force_xy_pct
        cx = int(W * float(xp))
        cy = int(H * float(yp))
        bx = max(MIN_MARGIN, min(W - block_w - MIN_MARGIN, cx - block_w // 2))
        by = max(MIN_MARGIN, min(H - block_h - MIN_MARGIN, cy - block_h // 2))
        score = bbox_score(luma, bx, by, block_w, block_h)

    # Decide alignment: flush to whichever edge is closer
    bbox_cx = bx + block_w // 2
    align_right = (bbox_cx > W / 2)
    if force_align in ("left", "right", "center"):
        align_right = (force_align == "right")
        align_center = (force_align == "center")
    else:
        align_center = False
    blur_score = region_blur_score(luma, bx, by, block_w, block_h)
    print(f"  [text] block {block_w}×{block_h} @ ({bx},{by}) std={score:.1f} "
          f"blur(LapVar)={blur_score:.1f} align={'right' if align_right else 'left'} "
          f"angle={angle:.1f}°")

    # Compute how much to blur the text layer so it matches the photo's local
    # blur but stays slightly sharper (so text reads without feeling pasted-on).
    text_blur_sigma = text_blur_sigma_to_match(blur_score, blur_match_factor)
    print(f"  [text] text blur sigma={text_blur_sigma:.2f}px (match factor={blur_match_factor})")

    # Color from photo highlights (not pure white)
    color = text_color_from_highlights(luma, top_pct=15)

    # Glow only if mixed light/shadow at the chosen bbox
    glow = score > 25.0
    glow_radius = max(1, int(font_size * 0.15))
    # glow color = the OPPOSITE end of the gray ramp from text color, so it
    # contrasts against text and renders as a soft halo
    glow_color = (255 - color[0], 255 - color[1], 255 - color[2])

    pil_rgb = Image.fromarray(rgb_arr)

    # Build the text block on a transparent layer, then paste (with optional rotate)
    pad = max(20, glow_radius * 3)
    layer = Image.new("RGBA", (block_w + pad*2, block_h + pad*2), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    def line_x(line_w):
        if align_center:
            return (block_w - line_w) // 2
        return (block_w - line_w) if align_right else 0

    if glow:
        gl = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        gld = ImageDraw.Draw(gl)
        cy = pad
        for line, (lw, lh) in zip(lines, line_sizes):
            gld.text((pad + line_x(lw), cy), line, font=font,
                     fill=glow_color + (200,))
            cy += line_h + leading
        gl = gl.filter(ImageFilter.GaussianBlur(glow_radius))
        layer = Image.alpha_composite(layer, gl)
        ld = ImageDraw.Draw(layer)

    cy = pad
    for line, (lw, lh) in zip(lines, line_sizes):
        ld.text((pad + line_x(lw), cy), line, font=font, fill=color + (255,))
        cy += line_h + leading

    # Optionally blur the text layer to nearly match the photo's local blur
    # (text stays slightly sharper — see text_blur_sigma_to_match).
    if text_blur_sigma > 0.05:
        layer = layer.filter(ImageFilter.GaussianBlur(text_blur_sigma))

    if abs(angle) < 0.5:
        out_rgba = pil_rgb.convert("RGBA")
        out_rgba.alpha_composite(layer, (bx - pad, by - pad))
        return np.asarray(out_rgba.convert("RGB"))
    else:
        rot = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
        paste_x = bx - (rot.size[0] - block_w) // 2
        paste_y = by - (rot.size[1] - block_h) // 2
        out_rgba = pil_rgb.convert("RGBA")
        out_rgba.alpha_composite(rot, (paste_x, paste_y))
        return np.asarray(out_rgba.convert("RGB"))


# ---- CLI -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--text", required=True,
                   help='quote text, or "auto" for semantic NN over literary DB')
    p.add_argument("--style", choices=["quote", "subtitle"], default="quote")
    p.add_argument("--orientation", choices=["horizontal", "body", "manual"],
                   default="horizontal",
                   help="default horizontal; body = align with shoulder line if tilted >5°")
    p.add_argument("--max-width-pct", type=float, default=42.0,
                   help="max line width as %% of image width before wrapping")
    p.add_argument("--blur-match-factor", type=float, default=0.5,
                   help="0.0-1.0: text blur as fraction of photo's local blur. "
                        "1.0=match exactly, 0.7=text 30%% sharper than photo, 0=crisp text")
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
                   font_size_pct=args.font_size_pct,
                   max_width_pct=args.max_width_pct,
                   blur_match_factor=args.blur_match_factor)
    out_path = args.out or str(Path(args.source).with_suffix(".__text.jpg"))
    Image.fromarray(out).save(out_path, quality=92)
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
