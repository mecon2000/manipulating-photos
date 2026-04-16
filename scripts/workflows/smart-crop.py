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

    # --- Unusual/artistic crops ---

    # 6. Face close-up (if face detected)
    if face_bbox:
        fx1, fy1, fx2, fy2 = face_bbox
        face_r = max(fx2 - fx1, fy2 - fy1)
        face_pad = int(face_r * 0.8)
        options.append(("Face close-up",
                        *clamp(fx1 - face_pad, fy1 - face_pad, fx2 + face_pad, fy2 + face_pad)))

    # 7. Chin-down (cut above chin, include body)
    chin_y = None
    if landmarks and "nose" in landmarks:
        nose_y = landmarks["nose"][1]
        # Chin is roughly 15% of face height below nose
        if face_bbox:
            face_h_est = face_bbox[3] - face_bbox[1]
            chin_y = nose_y + int(face_h_est * 0.4)
    if chin_y is None and face_bbox:
        chin_y = face_bbox[3]
    if chin_y:
        options.append(("Chin-down (headless)",
                        *clamp(subj_x1 - pad, chin_y, subj_x2 + pad, subj_y2 + pad)))

    # 8. Knee-up (cut below knees)
    knee_y = None
    if landmarks:
        knee_pts = [landmarks.get("left_knee"), landmarks.get("right_knee")]
        knee_pts = [p for p in knee_pts if p]
        if knee_pts:
            knee_y = max(p[1] for p in knee_pts) + int(short_edge * 0.03)
    if knee_y:
        top_y = subj_y1 - pad if not chin_y else chin_y
        options.append(("Knee-up",
                        *clamp(subj_x1 - pad, subj_y1 - pad, subj_x2 + pad, knee_y)))

    # 9. Chin-to-knee (the Daniel crop — no head, no feet)
    if chin_y and knee_y:
        hand_xs = []
        if landmarks:
            for k in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
                if k in landmarks:
                    hand_xs.append(landmarks[k][0])
        left_bound = min([subj_x1] + hand_xs) - pad
        right_bound = max([subj_x2] + hand_xs) + pad
        options.append(("Chin-to-knee (hands in)",
                        *clamp(left_bound, chin_y, right_bound, knee_y)))

    # 10. Torso only (shoulders to hips)
    if landmarks:
        shoulder_pts = [landmarks.get("left_shoulder"), landmarks.get("right_shoulder")]
        hip_pts = [landmarks.get("left_hip"), landmarks.get("right_hip")]
        shoulder_pts = [p for p in shoulder_pts if p]
        hip_pts = [p for p in hip_pts if p]
        if shoulder_pts and hip_pts:
            torso_top = min(p[1] for p in shoulder_pts) - int(short_edge * 0.05)
            torso_bot = max(p[1] for p in hip_pts) + int(short_edge * 0.08)
            torso_left = min(p[0] for p in shoulder_pts + hip_pts) - int(short_edge * 0.1)
            torso_right = max(p[0] for p in shoulder_pts + hip_pts) + int(short_edge * 0.1)
            options.append(("Torso only",
                            *clamp(torso_left, torso_top, torso_right, torso_bot)))

    # 11. Bottom half (waist down)
    if landmarks:
        hip_pts = [landmarks.get("left_hip"), landmarks.get("right_hip")]
        hip_pts = [p for p in hip_pts if p]
        if hip_pts:
            waist_y = min(p[1] for p in hip_pts) - int(short_edge * 0.05)
            options.append(("Waist-down",
                            *clamp(subj_x1 - pad * 2, waist_y, subj_x2 + pad * 2, subj_y2 + pad)))

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
    """Draw numbered crop rectangles on the image."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    colors = [
        (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
        (255, 50, 255), (50, 255, 255), (255, 150, 50), (150, 50, 255),
        (50, 255, 150), (255, 100, 100), (100, 255, 100), (100, 100, 255),
    ]

    w, h = img.size
    font_size = max(16, int(min(w, h) * 0.02))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Track label positions to avoid overlap
    used_label_rects = []

    for i, (name, x1, y1, x2, y2) in enumerate(options):
        color = colors[i % len(colors)]
        # Thinner border
        thickness = max(1, int(min(w, h) * 0.0015))
        for t in range(thickness):
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)

        # Label with number and name
        label = f"{i + 1}. {name}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Position label inside top-left of crop rect, offset down 20%
        label_offset_y = int(th * 0.2)
        lx = x1 + thickness + 4
        ly = y1 + thickness + 4 + label_offset_y

        # Nudge down if overlapping with existing labels
        for _ in range(20):
            label_rect = (lx - 2, ly - 2, lx + tw + 4, ly + th + 4)
            overlap = False
            for ur in used_label_rects:
                if not (label_rect[2] < ur[0] or label_rect[0] > ur[2] or
                        label_rect[3] < ur[1] or label_rect[1] > ur[3]):
                    overlap = True
                    break
            if not overlap:
                break
            ly += th + 6  # push down below the overlapping label

        # Clamp to image bounds
        if ly + th > h:
            ly = y1 - th - 4
        if lx + tw > w:
            lx = x2 - tw - 4

        label_rect = (lx - 2, ly - 2, lx + tw + 4, ly + th + 4)
        used_label_rects.append(label_rect)
        draw.rectangle(label_rect, fill=(0, 0, 0, 200))
        draw.text((lx, ly), label, fill=color, font=font)

    return overlay


def apply_crop(img, x1, y1, x2, y2):
    """Crop the image. If coords extend beyond image, pad with black (outpaint later)."""
    w, h = img.size
    # If fully within bounds, just crop
    if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
        return img.crop((x1, y1, x2, y2))

    # Needs outpainting — create padded canvas
    new_w = x2 - x1
    new_h = y2 - y1
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    paste_x = max(0, -x1)
    paste_y = max(0, -y1)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    region = img.crop((src_x1, src_y1, src_x2, src_y2))
    canvas.paste(region, (paste_x, paste_y))
    return canvas


def outpaint_fill(img, original, crop_coords, output_dir):
    """Fill black/padded areas using fal.ai inpainting."""
    import fal_client
    import requests
    import tempfile
    from io import BytesIO

    w, h = img.size
    img_arr = np.array(img)

    # Create mask: white where we need to fill (black/padded areas)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    mask = (gray < 5).astype(np.uint8) * 255

    # Check if there's actually anything to fill
    fill_pct = np.mean(mask > 0) * 100
    if fill_pct < 1:
        print("  No outpainting needed (< 1% fill area)")
        return img

    print(f"  Outpainting {fill_pct:.1f}% of canvas...")

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

    # Inpaint using SDXL (works better than Flux for outpainting)
    try:
        handle = fal_client.submit("fal-ai/stable-diffusion-v35-large/inpainting", arguments={
            "image_url": img_url,
            "mask_url": mask_url,
            "prompt": "natural continuation of the photograph, matching lighting and style, seamless extension",
            "negative_prompt": "different style, different lighting, text, watermark",
            "strength": 0.95,
            "num_images": 1,
            "output_format": "jpeg",
        })
        result = handle.get()
        images = result.get("images", [])
        if images:
            resp = requests.get(images[0]["url"], timeout=60)
            filled = Image.open(BytesIO(resp.content)).convert("RGB")
            if filled.size != (w, h):
                filled = filled.resize((w, h), Image.LANCZOS)
            return filled
    except Exception as e:
        print(f"  Outpainting failed: {e}")

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
    parser.add_argument("--outpaint", action="store_true",
                        help="Fill extended canvas areas with AI-generated content")
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
    print(f"\nGenerated {len(options)} crop options:")
    for i, (name, x1, y1, x2, y2) in enumerate(options):
        cw, ch = x2 - x1, y2 - y1
        extends = ""
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            extends = " [NEEDS OUTPAINT]"
        print(f"  {i + 1:>2}. {name:<25} ({x1},{y1})-({x2},{y2}) = {cw}x{ch}{extends}")

    os.makedirs(FINALS, exist_ok=True)
    src_name = os.path.splitext(os.path.basename(source))[0]

    # Show options mode
    if args.show_options:
        overlay = draw_options_overlay(img, options)
        out_path = os.path.join(FINALS, f"{src_name}_crop_options.jpg")
        overlay.save(out_path, quality=95)
        print(f"\nOptions overlay saved: {out_path}")
        try:
            from notify import push_image
            push_image(out_path, "Crop options", f"{len(options)} suggestions")
        except Exception:
            pass

    # Apply crop
    crop_coords = None
    crop_name = None

    if args.custom:
        parts = [int(x) for x in args.custom.split(",")]
        if len(parts) == 4:
            crop_coords = tuple(parts)
            crop_name = "custom"
        else:
            print("ERROR: --custom needs x1,y1,x2,y2")
            sys.exit(1)

    elif args.crop is not None:
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
            cropped = outpaint_fill(cropped, img, crop_coords, FINALS)
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
