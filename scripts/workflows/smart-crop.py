#!/home/rong/openclaw-venv/bin/python3
"""
Smart Crop — Intelligent photo cropping with composition-aware suggestions.

Analyzes subject position (BiRefNet mask + MediaPipe pose), generates multiple
crop suggestions including standard compositions and unusual/artistic crops.
Supports outpainting to extend canvas for aspect ratio changes.

Usage:
    python smart-crop.py --source photo.jpg --show-options
    python smart-crop.py --source photo.jpg --crop 3
    python smart-crop.py --source photo.jpg --crop 3 --outpaint
    python smart-crop.py --source photo.jpg --auto-align --show-options
    python smart-crop.py --source photo.jpg --custom 100,200,900,1800
"""

import os
import sys
import math
import argparse
import random

_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())
os.environ['FAL_KEY'] = os.environ.get('FAL_API_KEY', '')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask
sys.stdout.reconfigure(line_buffering=True)

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")


def detect_pose_landmarks(img):
    """Detect body landmarks via MediaPipe pose. Returns dict of named points or None."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        img_rgb = np.array(img)
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results = pose.process(img_rgb)
            if not results.pose_landmarks:
                return None
            w, h = img.size
            landmarks = {}
            names = {
                0: "nose", 2: "left_eye", 5: "right_eye",
                7: "left_ear", 8: "right_ear",
                11: "left_shoulder", 12: "right_shoulder",
                13: "left_elbow", 14: "right_elbow",
                15: "left_wrist", 16: "right_wrist",
                23: "left_hip", 24: "right_hip",
                25: "left_knee", 26: "right_knee",
                27: "left_ankle", 28: "right_ankle",
            }
            for idx, name in names.items():
                lm = results.pose_landmarks.landmark[idx]
                if lm.visibility > 0.3:
                    landmarks[name] = (int(lm.x * w), int(lm.y * h))
            return landmarks if landmarks else None
    except Exception:
        return None


def detect_face_bbox(landmarks):
    """Get face bounding box from pose landmarks."""
    face_keys = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    pts = [landmarks[k] for k in face_keys if k in landmarks]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Expand face bbox
    cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    r = max(face_w, face_h)
    return (cx - r, cy - r, cx + r, cy + r)


def auto_align_angle(img, landmarks):
    """Estimate rotation angle to straighten the subject."""
    # Use shoulders or eyes for alignment
    if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
        lx, ly = landmarks["left_shoulder"]
        rx, ry = landmarks["right_shoulder"]
    elif "left_eye" in landmarks and "right_eye" in landmarks:
        lx, ly = landmarks["left_eye"]
        rx, ry = landmarks["right_eye"]
    else:
        return 0.0
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    # Only correct small tilts
    if abs(angle) > 15:
        return 0.0
    return -angle


def story_crop(w, h, face_bbox, height_in_faces=5.2, face_from_top=0.34, ar=9/16):
    """Aspect-exact 9:16 head-and-shoulders crop, anchored on the face.

    The other options here clamp to the frame and lose their ratio in the
    process; a story crop has to stay exactly 9:16, so this shrinks the box to
    fit rather than clipping one side of it.
    """
    if not face_bbox:
        ch = min(h, w / ar)
        cw = ch * ar
        return int((w - cw) / 2), int((h - ch) / 2), int(cw), int(ch)
    fx1, fy1, fx2, fy2 = face_bbox
    fh = max(fy2 - fy1, 1)
    cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    ch = min(fh * height_in_faces, h, w / ar)
    cw = ch * ar
    x1 = int(round(min(max(cx - cw / 2, 0), w - cw)))
    y1 = int(round(min(max(cy - ch * face_from_top, 0), h - ch)))
    return x1, y1, int(cw), int(ch)


def generate_crop_options(w, h, mask_binary, landmarks, face_bbox):
    """Generate multiple crop suggestions. Returns list of (name, x1, y1, x2, y2)."""
    # Subject bounding box from mask
    ys, xs = np.where(mask_binary > 0)
    if len(ys) == 0:
        # No mask — use full image
        return [("Full image", 0, 0, w, h)]

    subj_x1, subj_y1 = int(xs.min()), int(ys.min())
    subj_x2, subj_y2 = int(xs.max()), int(ys.max())
    subj_cx = (subj_x1 + subj_x2) // 2
    subj_cy = (subj_y1 + subj_y2) // 2
    subj_w = subj_x2 - subj_x1
    subj_h = subj_y2 - subj_y1
    short_edge = min(w, h)

    options = []

    def clamp(x1, y1, x2, y2):
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 50, min(x2, w))
        y2 = max(y1 + 50, min(y2, h))
        return int(x1), int(y1), int(x2), int(y2)

    # --- Standard crops ---

    # 1. Tight subject with 5% padding
    pad = int(short_edge * 0.05)
    options.append(("Tight subject",
                    *clamp(subj_x1 - pad, subj_y1 - pad, subj_x2 + pad, subj_y2 + pad)))

    # 2. Subject with rule-of-thirds positioning (subject at right third)
    thirds_w = int(subj_w * 1.8)
    thirds_h = int(subj_h * 1.3)
    # Place subject at left third
    thirds_x1 = subj_cx - int(thirds_w * 0.67)
    thirds_y1 = subj_y1 - int(subj_h * 0.15)
    options.append(("Rule of thirds (left)",
                    *clamp(thirds_x1, thirds_y1, thirds_x1 + thirds_w, thirds_y1 + thirds_h)))

    # 3. Square crop centered on subject
    sq_size = max(subj_w, subj_h) + int(short_edge * 0.1)
    sq_x1 = subj_cx - sq_size // 2
    sq_y1 = subj_cy - sq_size // 2
    options.append(("Square centered",
                    *clamp(sq_x1, sq_y1, sq_x1 + sq_size, sq_y1 + sq_size)))

    # 4. 4:5 portrait (Instagram)
    p_h = int(subj_h * 1.25)
    p_w = int(p_h * 0.8)
    p_x1 = subj_cx - p_w // 2
    p_y1 = subj_y1 - int(subj_h * 0.1)
    options.append(("4:5 portrait",
                    *clamp(p_x1, p_y1, p_x1 + p_w, p_y1 + p_h)))

    # 5. 16:9 cinematic
    cin_h = int(subj_h * 1.1)
    cin_w = int(cin_h * 16 / 9)
    cin_x1 = subj_cx - cin_w // 2
    cin_y1 = subj_cy - cin_h // 2
    options.append(("16:9 cinematic",
                    *clamp(cin_x1, cin_y1, cin_x1 + cin_w, cin_y1 + cin_h)))

    # 6. 9:16 story (head & shoulders) — aspect-exact, for IG/TikTok verticals
    sx, sy, scw, sch = story_crop(w, h, face_bbox)
    options.append(("9:16 story (head+shoulders)", sx, sy, sx + scw, sy + sch))

    # --- Unusual/artistic crops ---
    # Estimate body segments from mask when pose detection fails
    # Use vertical distribution of mask pixels to find head/torso/legs regions
    chin_y = None
    knee_y = None
    shoulder_y = None
    hip_y = None

    if landmarks:
        # Use pose landmarks if available
        if "nose" in landmarks:
            if face_bbox:
                chin_y = face_bbox[3]
            else:
                chin_y = landmarks["nose"][1] + int(subj_h * 0.05)
        knee_pts = [landmarks.get("left_knee"), landmarks.get("right_knee")]
        knee_pts = [p for p in knee_pts if p]
        if knee_pts:
            knee_y = max(p[1] for p in knee_pts) + int(short_edge * 0.03)
        shoulder_pts = [landmarks.get("left_shoulder"), landmarks.get("right_shoulder")]
        shoulder_pts = [p for p in shoulder_pts if p]
        if shoulder_pts:
            shoulder_y = min(p[1] for p in shoulder_pts)
        hip_pts = [landmarks.get("left_hip"), landmarks.get("right_hip")]
        hip_pts = [p for p in hip_pts if p]
        if hip_pts:
            hip_y = max(p[1] for p in hip_pts)

    # Fallback: estimate from mask vertical profile
    if chin_y is None or knee_y is None:
        # Sum mask pixels per row to find vertical distribution
        row_sums = mask_binary.sum(axis=1)
        occupied = np.where(row_sums > 0)[0]
        if len(occupied) > 10:
            top_y = int(occupied[0])
            bot_y = int(occupied[-1])
            body_h = bot_y - top_y
            # Estimate: head=top 18%, shoulders=22%, hips=55%, knees=78%
            if chin_y is None:
                chin_y = top_y + int(body_h * 0.18)
            if shoulder_y is None:
                shoulder_y = top_y + int(body_h * 0.22)
            if hip_y is None:
                hip_y = top_y + int(body_h * 0.55)
            if knee_y is None:
                knee_y = top_y + int(body_h * 0.78)

    # 6. Face close-up
    if face_bbox:
        fx1, fy1, fx2, fy2 = face_bbox
        face_r = max(fx2 - fx1, fy2 - fy1)
        face_pad = int(face_r * 0.8)
        options.append(("Face close-up",
                        *clamp(fx1 - face_pad, fy1 - face_pad, fx2 + face_pad, fy2 + face_pad)))
    elif chin_y:
        # Estimate face from mask top to chin
        face_h = chin_y - subj_y1
        face_pad = int(face_h * 0.5)
        options.append(("Face close-up (est.)",
                        *clamp(subj_cx - face_h, subj_y1 - face_pad, subj_cx + face_h, chin_y + face_pad)))

    # 7. Chin-down (headless body)
    if chin_y:
        options.append(("Chin-down (headless)",
                        *clamp(subj_x1 - pad, chin_y, subj_x2 + pad, subj_y2 + pad)))

    # 8. Knee-up
    if knee_y:
        options.append(("Knee-up",
                        *clamp(subj_x1 - pad, subj_y1 - pad, subj_x2 + pad, knee_y)))

    # 9. Chin-to-knee (no head, no feet)
    if chin_y and knee_y:
        options.append(("Chin-to-knee",
                        *clamp(subj_x1 - pad * 2, chin_y, subj_x2 + pad * 2, knee_y)))

    # 10. Torso only (shoulders to hips)
    if shoulder_y and hip_y:
        torso_pad = int(short_edge * 0.08)
        options.append(("Torso only",
                        *clamp(subj_x1 - torso_pad, shoulder_y - torso_pad,
                               subj_x2 + torso_pad, hip_y + torso_pad)))

    # 11. Waist-down
    if hip_y:
        options.append(("Waist-down",
                        *clamp(subj_x1 - pad * 2, hip_y - int(short_edge * 0.05),
                               subj_x2 + pad * 2, subj_y2 + pad)))

    # 12. Off-center dramatic (subject pushed to edge)
    dramatic_w = int(subj_w * 2.0)
    dramatic_h = int(subj_h * 1.2)
    # Push subject to right 20%
    d_x1 = subj_x2 - int(dramatic_w * 0.2)
    d_y1 = subj_y1 - int(subj_h * 0.1)
    options.append(("Off-center right",
                    *clamp(d_x1 - dramatic_w, d_y1, d_x1, d_y1 + dramatic_h)))

    return options


def draw_options_overlay(img, options):
    """Draw numbered crop rectangles split across two side-by-side (or stacked) panels.
    Options are sorted top-to-bottom by upper-left Y, then split so spatially
    close crops land on different panels."""
    w, h = img.size

    # Sort options by upper-left Y coordinate, renumber
    sorted_opts = sorted(enumerate(options), key=lambda t: t[1][2])  # sort by y1
    # Renumber 1..N in top-to-bottom order
    renumbered = []
    for new_idx, (orig_idx, opt) in enumerate(sorted_opts):
        renumbered.append((new_idx + 1, opt))  # (display_number, (name, x1, y1, x2, y2))

    # Split into two groups: alternate so nearby crops go to different panels
    group_a = [renumbered[i] for i in range(0, len(renumbered), 2)]  # odd positions
    group_b = [renumbered[i] for i in range(1, len(renumbered), 2)]  # even positions

    colors = [
        (255, 50, 50), (50, 255, 50), (80, 80, 255), (255, 255, 50),
        (255, 50, 255), (50, 255, 255), (255, 150, 50), (150, 50, 255),
        (50, 255, 150), (255, 100, 100), (100, 255, 100), (150, 150, 255),
    ]

    font_size = max(14, int(min(w, h) * 0.018))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    def draw_panel(base_img, group):
        panel = base_img.copy()
        draw = ImageDraw.Draw(panel)
        used_rects = []  # track both label AND border rects for overlap

        for num, (name, x1, y1, x2, y2) in group:
            color = colors[(num - 1) % len(colors)]
            thickness = max(1, int(min(w, h) * 0.0012))

            # Draw border
            for t in range(thickness):
                draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)

            # Label
            label = f"{num}. {name}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Start position: inside top-left, slightly offset down
            lx = x1 + thickness + 4
            ly = y1 + thickness + 4 + int(th * 0.3)

            # Nudge down to avoid overlapping any existing label or border line
            for _ in range(30):
                label_rect = (lx - 2, ly - 2, lx + tw + 4, ly + th + 4)
                overlap = False
                for ur in used_rects:
                    if not (label_rect[2] < ur[0] or label_rect[0] > ur[2] or
                            label_rect[3] < ur[1] or label_rect[1] > ur[3]):
                        overlap = True
                        break
                if not overlap:
                    break
                ly += th + 4

            # Clamp
            pw, ph = base_img.size
            if ly + th > ph:
                ly = y1 - th - 4
            if lx + tw > pw:
                lx = x2 - tw - 4
            lx = max(2, lx)
            ly = max(2, ly)

            label_rect = (lx - 2, ly - 2, lx + tw + 4, ly + th + 4)
            used_rects.append(label_rect)
            draw.rectangle(label_rect, fill=(0, 0, 0, 200))
            draw.text((lx, ly), label, fill=color, font=font)

        return panel

    panel_a = draw_panel(img, group_a)
    panel_b = draw_panel(img, group_b)

    # Combine: portrait → side by side, landscape → stacked
    is_portrait = h > w
    if is_portrait:
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(panel_a, (0, 0))
        combined.paste(panel_b, (w, 0))
    else:
        combined = Image.new("RGB", (w, h * 2))
        combined.paste(panel_a, (0, 0))
        combined.paste(panel_b, (0, h))

    return combined, renumbered


def apply_crop(img, x1, y1, x2, y2):
    """Crop the image. If coords extend beyond image, mirror-fill the padded area
    so the outpainter has context instead of pure black."""
    w, h = img.size
    # If fully within bounds, just crop
    if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
        return img.crop((x1, y1, x2, y2))

    # Needs outpainting — create canvas with mirrored content for context
    new_w = x2 - x1
    new_h = y2 - y1
    img_arr = np.array(img)

    # Create output array
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Paste the original region
    paste_x = max(0, -x1)
    paste_y = max(0, -y1)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    region = img_arr[src_y1:src_y2, src_x1:src_x2]
    canvas[paste_y:paste_y + region.shape[0], paste_x:paste_x + region.shape[1]] = region

    # Mirror-fill extended areas with flipped content from the edge
    # Bottom extension
    if y2 > h:
        ext_h = y2 - h
        src_strip = img_arr[max(0, h - ext_h):h, src_x1:src_x2]
        flipped = src_strip[::-1]  # vertical flip
        if flipped.shape[0] > ext_h:
            flipped = flipped[:ext_h]
        fill_y = paste_y + (h - max(0, y1))
        fill_h = min(flipped.shape[0], new_h - fill_y)
        if fill_h > 0:
            canvas[fill_y:fill_y + fill_h, paste_x:paste_x + flipped.shape[1]] = flipped[:fill_h]

    # Top extension
    if y1 < 0:
        ext_h = -y1
        src_strip = img_arr[0:min(ext_h, h), src_x1:src_x2]
        flipped = src_strip[::-1]
        fill_h = min(flipped.shape[0], ext_h)
        if fill_h > 0:
            start_y = ext_h - fill_h
            canvas[start_y:ext_h, paste_x:paste_x + flipped.shape[1]] = flipped[:fill_h]

    # Left extension
    if x1 < 0:
        ext_w = -x1
        src_strip = img_arr[src_y1:src_y2, 0:min(ext_w, w)]
        flipped = src_strip[:, ::-1]
        fill_w = min(flipped.shape[1], ext_w)
        if fill_w > 0:
            start_x = ext_w - fill_w
            canvas[paste_y:paste_y + flipped.shape[0], start_x:ext_w] = flipped[:, :fill_w]

    # Right extension
    if x2 > w:
        ext_w = x2 - w
        src_strip = img_arr[src_y1:src_y2, max(0, w - ext_w):w]
        flipped = src_strip[:, ::-1]
        fill_w = min(flipped.shape[1], ext_w)
        fill_x = paste_x + (w - max(0, x1))
        if fill_w > 0 and fill_x + fill_w <= new_w:
            canvas[paste_y:paste_y + flipped.shape[0], fill_x:fill_x + fill_w] = flipped[:, :fill_w]

    return Image.fromarray(canvas)


def outpaint_fill(img, original, crop_coords, output_dir, prompt=None):
    """Fill black/padded areas using fal.ai inpainting."""
    import fal_client
    import requests
    import tempfile
    from io import BytesIO

    w, h = img.size
    img_arr = np.array(img)

    # Build mask from crop coordinates (not brightness — mirror-fill means no black areas)
    x1, y1, x2, y2 = crop_coords
    orig_w, orig_h = original.size
    paste_x = max(0, -x1)
    paste_y = max(0, -y1)
    orig_region_w = min(orig_w, x2) - max(0, x1)
    orig_region_h = min(orig_h, y2) - max(0, y1)

    # Binary: 255 = fill area, 0 = keep
    fill_binary = np.ones((h, w), dtype=np.uint8) * 255
    fill_binary[paste_y:paste_y + orig_region_h, paste_x:paste_x + orig_region_w] = 0

    # Gradient feather at the boundary
    from scipy.ndimage import distance_transform_edt
    feather_px = max(40, int(min(w, h) * 0.04))
    dist = distance_transform_edt(fill_binary > 0)
    mask = np.clip(dist / feather_px, 0, 1) * 255
    mask = mask.astype(np.uint8)

    # Auto-detect outpaint prompt from the visible content if not given
    if prompt is None:
        from PIL import ImageStat
        entropy = img.convert("L").entropy()
        if entropy > 6.0:
            prompt = ("seamless continuation of artistic painting, "
                      "matching colors and brushwork style, "
                      "flowing abstract forms, ink watercolor oil painting texture, "
                      "same color palette and mood, NO person NO body parts")
        else:
            prompt = "natural continuation of the photograph, matching lighting color and style, NO person NO body parts"
    print(f"  Outpaint prompt: {prompt[:80]}...")

    fill_pct = np.mean(mask > 10) * 100
    if fill_pct < 1:
        print("  No outpainting needed (< 1% fill area)")
        return img

    print(f"  Outpainting {fill_pct:.1f}% of canvas (feather={feather_px}px)...")

    # Upload image and mask
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp, format="JPEG", quality=95)
        img_url = fal_client.upload_file(tmp.name)
        os.unlink(tmp.name)

    mask_pil = Image.fromarray(mask, "L")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        mask_pil.save(tmp, format="PNG")
        mask_url = fal_client.upload_file(tmp.name)
        os.unlink(tmp.name)

    # Try multiple inpainting endpoints
    endpoints = [
        "fal-ai/flux-general/inpainting",
        "fal-ai/inpaint",
    ]
    for endpoint in endpoints:
        try:
            print(f"  Trying {endpoint}...")
            args = {
                "image_url": img_url,
                "mask_url": mask_url,
                "prompt": prompt,
                "negative_prompt": "person, human, body parts, feet, hands, face, legs, arms, fingers, toes, skin",
                "strength": 0.95,
                "num_images": 1,
                "output_format": "jpeg",
                "enable_safety_checker": False,
            }
            handle = fal_client.submit(endpoint, arguments=args)
            result = handle.get()
            images = result.get("images", [])
            if images:
                url = images[0].get("url", "")
                if url:
                    resp = requests.get(url, timeout=60)
                    filled = Image.open(BytesIO(resp.content)).convert("RGB")
                    if filled.size != (w, h):
                        filled = filled.resize((w, h), Image.LANCZOS)
                    # Check it's not black
                    from PIL import ImageStat
                    brightness = ImageStat.Stat(filled.convert("L")).mean[0]
                    if brightness > 10:
                        print(f"  Outpaint OK (brightness={brightness:.0f})")
                        return filled
                    print(f"  Result too dark ({brightness:.0f}), trying next...")
            else:
                print(f"  No images returned")
        except Exception as e:
            print(f"  {endpoint} failed: {e}")
            continue

    print("  All outpaint endpoints failed — returning unmodified crop")
    return img


def main():
    parser = argparse.ArgumentParser(description="Smart Crop — Intelligent photo cropping")
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--show-options", action="store_true",
                        help="Show numbered crop options overlaid on the image")
    parser.add_argument("--crop", type=int, default=None,
                        help="Apply crop number N (from --show-options)")
    parser.add_argument("--custom", type=str, default=None,
                        help="Custom crop: x1,y1,x2,y2")
    parser.add_argument("--story", action="store_true",
                        help="apply the aspect-exact 9:16 head-and-shoulders crop directly "
                             "(no option list needed — usable in batch)")
    parser.add_argument("--outpaint", action="store_true",
                        help="Fill extended canvas areas with AI-generated content")
    parser.add_argument("--outpaint-prompt", type=str, default=None,
                        help="Custom prompt for outpainting (auto-detected if not given)")
    parser.add_argument("--auto-align", action="store_true",
                        help="Auto-straighten based on shoulder/eye alignment")
    parser.add_argument("--output-to", choices=["local"], default="local")
    parser.add_argument("--local-output-dir", default=None)
    args = parser.parse_args()

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    img = Image.open(source).convert("RGB")
    w, h = img.size
    short_edge = min(w, h)
    print(f"Source: {w}x{h} — {source}")

    # Detect pose landmarks
    print("Detecting pose...")
    landmarks = detect_pose_landmarks(img)
    if landmarks:
        print(f"  Found {len(landmarks)} landmarks: {', '.join(sorted(landmarks.keys()))}")
    else:
        print("  No pose detected")

    face_bbox = detect_face_bbox(landmarks) if landmarks else None
    if face_bbox:
        print(f"  Face bbox: {face_bbox}")

    # Auto-align
    if args.auto_align and landmarks:
        angle = auto_align_angle(img, landmarks)
        if abs(angle) > 0.3:
            print(f"  Auto-align: rotating {angle:.1f}°")
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
            w, h = img.size
            # Re-detect after rotation
            landmarks = detect_pose_landmarks(img)
            face_bbox = detect_face_bbox(landmarks) if landmarks else None

    # Extract mask
    print("Extracting mask...")
    mask, mask_info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.LANCZOS)
    mask_binary = (np.array(mask) > 127).astype(np.uint8)
    print(f"  Coverage: {mask_info['coverage_pct']}%")

    # Generate options
    options = generate_crop_options(w, h, mask_binary, landmarks, face_bbox)

    os.makedirs(FINALS, exist_ok=True)
    src_name = os.path.splitext(os.path.basename(source))[0]

    # Show options mode — draw overlay, get renumbered list
    renumbered = None
    if args.show_options:
        combined, renumbered = draw_options_overlay(img, options)
        out_path = os.path.join(FINALS, f"{src_name}_crop_options.jpg")
        combined.save(out_path, quality=95)
        print(f"\nGenerated {len(options)} crop options (sorted top→bottom):")
        for num, (name, x1, y1, x2, y2) in renumbered:
            cw, ch = x2 - x1, y2 - y1
            extends = " [OUTPAINT]" if (x1 < 0 or y1 < 0 or x2 > w or y2 > h) else ""
            print(f"  {num:>2}. {name:<25} {cw}x{ch}{extends}")
        print(f"\nOptions overlay saved: {out_path}")
        try:
            from notify import push_image
            push_image(out_path, "Crop options", f"{len(options)} suggestions")
        except Exception:
            pass
    else:
        # Just print unsorted for reference
        print(f"\nGenerated {len(options)} crop options:")
        for i, (name, x1, y1, x2, y2) in enumerate(options):
            cw, ch = x2 - x1, y2 - y1
            print(f"  {i + 1:>2}. {name:<25} {cw}x{ch}")

    # Apply crop — use renumbered order if show-options was used
    crop_coords = None
    crop_name = None

    if args.story:
        # Prefer the 468-point face mesh over the pose model's coarse face box:
        # the pose estimate can sit far enough off-centre to crop half a face
        # out of the frame, which a story crop cannot recover from.
        story_bbox = face_bbox
        try:
            import numpy as _np
            import face_align as _FA
            _pts = _FA.landmarks(_np.asarray(img.convert("RGB")))
            if _pts:
                _xs = [p[0] for p in _pts.values()]
                _ys = [p[1] for p in _pts.values()]
                story_bbox = (min(_xs), min(_ys), max(_xs), max(_ys))
                print("  story crop: using face-mesh bbox")
        except Exception as e:
            print(f"  story crop: face-mesh unavailable ({e}) — using pose bbox")
        sx, sy, scw, sch = story_crop(w, h, story_bbox)
        crop_coords = (sx, sy, sx + scw, sy + sch)
        crop_name = "9:16 story (head+shoulders)"

    elif args.custom:
        parts = [int(x) for x in args.custom.split(",")]
        if len(parts) == 4:
            crop_coords = tuple(parts)
            crop_name = "custom"
        else:
            print("ERROR: --custom needs x1,y1,x2,y2")
            sys.exit(1)

    elif args.crop is not None:
        # Match against renumbered list if available, else original order
        if renumbered:
            match = [opt for num, opt in renumbered if num == args.crop]
            if match:
                crop_name, *crop_coords = match[0]
                crop_coords = tuple(crop_coords)
                print(f"\nApplying crop {args.crop}: {crop_name}")
            else:
                print(f"ERROR: Crop {args.crop} not found (1-{len(options)})")
                sys.exit(1)
        else:
            idx = args.crop - 1
            if 0 <= idx < len(options):
                crop_name, *crop_coords = options[idx]
                crop_coords = tuple(crop_coords)
                print(f"\nApplying crop {args.crop}: {crop_name}")
            else:
                print(f"ERROR: Crop {args.crop} out of range (1-{len(options)})")
                sys.exit(1)

    if crop_coords:
        x1, y1, x2, y2 = crop_coords
        cropped = apply_crop(img, x1, y1, x2, y2)

        # Outpaint if needed and requested
        needs_outpaint = x1 < 0 or y1 < 0 or x2 > w or y2 > h
        if needs_outpaint and args.outpaint:
            cropped = outpaint_fill(cropped, img, crop_coords, FINALS, prompt=args.outpaint_prompt)
        elif needs_outpaint:
            print("  Note: crop extends beyond image. Use --outpaint to fill.")

        safe_name = crop_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(FINALS, f"{src_name}_crop_{safe_name}.jpg")
        cropped.save(out_path, quality=95)
        print(f"Cropped: {cropped.size[0]}x{cropped.size[1]} → {out_path}")
        try:
            from notify import push_image
            push_image(out_path, f"Crop: {crop_name}", f"{cropped.size[0]}x{cropped.size[1]}")
        except Exception:
            pass

    if not args.show_options and args.crop is None and args.custom is None:
        print("\nUse --show-options to see suggestions, --crop N to apply one.")


if __name__ == "__main__":
    main()
