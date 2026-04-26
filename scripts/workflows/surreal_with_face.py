#!/usr/bin/env python3
"""Identity-preserving surreal stylization (0010x0010 inspired).

Pipeline:
  0.   relit.jpg       (from relighting.py — passed in or run separately)
  0.5  bw_with_curve   → bw_relit.jpg                             (anchor)
  1.   become-image    → surreal.jpg (color from Replicate)
  1.5  → grayscale     → surreal_gray.jpg
  2.   MediaPipe face  → radial gradient mask on bw_relit dims
  3.   skimage histogram-match  → bw_relit_matched (matches surreal_gray's tone)
  4.   composite       → final = surreal_gray*(1-mask) + bw_relit_matched*mask

Usage:
  surreal_with_face.py --relit RELIT.jpg --style STYLE_REF.jpg
  # output: ~/.openclaw/workspace/shared/surreal-with-face/<stem>__style_<n>.jpg

Reads REPLICATE_API_TOKEN from env or ~/sol/.env.
"""
import argparse, base64, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from skimage.exposure import match_histograms

OUT_DIR = Path("~/.openclaw/workspace/shared/surreal-with-face").expanduser()
POSE_MODEL = Path("~/openclaw-venv/mediapipe_models/pose_landmarker.task").expanduser()
BECOME_MODEL = "fofr/become-image:8d0b076a2aff3904dfcec3253c778e0310a68f78483c4699c7fd800f3051d2b3"
UPSCALE_MODEL = "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b"

# --- Token ----------------------------------------------------------------

def load_token():
    t = os.environ.get("REPLICATE_API_TOKEN")
    if t: return t
    env = Path("~/sol/.env").expanduser()
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

# --- Step 0.5: B&W with crushed-blacks/lifted-mids curve ------------------

def bw_with_curve(img_pil):
    arr = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    lum = 0.2126*arr[..., 0] + 0.7152*arr[..., 1] + 0.0722*arr[..., 2]
    x = lum / 255.0
    y = np.where(x < 0.15, x * 0.4, 0.06 + (x - 0.15) * (1.05 / 0.85))
    y = np.clip(y, 0, 1)
    out = (y * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([out, out, out], axis=-1))

# --- Step 1: become-image API call ---------------------------------------

def to_data_uri(path):
    suf = Path(path).suffix.lower()
    mime = {".png":"image/png", ".webp":"image/webp"}.get(suf, "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def run_become_image(bw_relit_path, style_path, prompt, neg_prompt,
                      instant_id_strength, image_to_become_strength,
                      denoising_strength, depth_strength, prompt_strength, seed,
                      max_retries=8):
    import replicate
    for attempt in range(max_retries):
        try:
            out = replicate.run(BECOME_MODEL, input={
                "image": to_data_uri(bw_relit_path),
                "image_to_become": to_data_uri(style_path),
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "instant_id_strength": instant_id_strength,
                "image_to_become_strength": image_to_become_strength,
                "denoising_strength": denoising_strength,
                "control_depth_strength": depth_strength,
                "prompt_strength": prompt_strength,
                "number_of_images": 1,
                "disable_safety_checker": True,
                "seed": seed if seed is not None else int(time.time()) % 100000,
            })
            item = out[0] if isinstance(out, list) and out else out
            if hasattr(item, "read"): return item.read()
            if isinstance(item, str):
                with urllib.request.urlopen(item, timeout=60) as r: return r.read()
            raise RuntimeError(f"unexpected output: {type(out)}")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "throttled" in msg.lower():
                import re as _re
                m = _re.search(r"resets in ~(\d+)s", msg)
                wait = int(m.group(1)) + 1 if m else min(60, 2**attempt + 5)
                time.sleep(wait); continue
            raise
    raise RuntimeError("max retries exceeded")

# --- Step 2: face mask via MediaPipe pose landmarks ----------------------

def face_ellipse_mask(img_size, source_arr, inner_mult=1.0, outer_mult=4.0,
                       falloff_power=0.5):
    """Build an oriented-ellipse mask aligned with face axis.

    Inner ellipse fits the face tightly (eyes/ears horizontal extent, eyes-to-
    chin vertical extent). Outer ellipse is `feather_mult` × inner along the
    face axis (downward toward shoulders/chest gets caught) and ~1.5× sideways.
    Falloff is smooth (squared) so seam is invisible.

    Returns float32 mask 0-1 same H×W as img_size = (W, H).
    """
    import mediapipe as mp
    W, H = img_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Fallback ellipse: top-third, square-ish
    cx, cy = W * 0.5, H * 0.33
    a, b = min(W, H) * 0.13, min(W, H) * 0.13
    angle_deg = 0.0

    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=source_arr.astype(np.uint8))
        base = mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL))
        opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=base, num_poses=1)
        det = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        res = det.detect(mp_img); det.close()
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            def P(i): return np.array([lms[i].x * W, lms[i].y * H], dtype=np.float32)
            # MediaPipe pose face indices:
            # 0=nose, 2=left-eye, 5=right-eye, 7=left-ear, 8=right-ear, 9=left-mouth, 10=right-mouth
            nose, le, re, lear, rear, lm, rm = P(0), P(2), P(5), P(7), P(8), P(9), P(10)
            eye_mid   = (le + re) / 2
            mouth_mid = (lm + rm) / 2
            face_pts = np.stack([nose, le, re, lm, rm])  # ears optional — they extend hair
            cx, cy = face_pts.mean(axis=0)
            # Major axis = direction from mouth toward top-of-head (above eyes)
            axis_vec = eye_mid - mouth_mid
            axis_norm = axis_vec / (np.linalg.norm(axis_vec) + 1e-6)
            angle_deg = float(np.degrees(np.arctan2(-axis_norm[1], axis_norm[0]))) - 90.0  # 0=face up
            # Semi-axes (face-local: a=horizontal, b=vertical-along-face-axis).
            # Aim: foci land at forehead and lips → b ≈ eye-to-mouth (covers chin+forehead),
            #      a ≈ ear-distance × 0.4 (tighter than the ears so hair stays out of inner mask).
            eye_to_mouth = float(np.linalg.norm(eye_mid - mouth_mid))
            face_h = eye_to_mouth * 1.0
            face_w = float(np.linalg.norm(lear - rear)) * 0.4
            a, b = face_w, face_h     # a = semi-x in face-local coords, b = semi-y
            print(f"  face: center=({cx:.0f},{cy:.0f})  semi=({a:.0f}×{b:.0f})  angle={angle_deg:.1f}°  inner={inner_mult} outer={outer_mult} power={falloff_power}")
        else:
            print("  pose: NOT detected — fallback ellipse")
    except Exception as e:
        print(f"  pose error: {e} — fallback")

    # Distance in face-local coords: rotate (xx-cx, yy-cy) by -angle, then scale by (a,b).
    # angle_deg=0 means upright face → no rotation → image-x maps to local-x, image-y to local-y.
    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(-rad), np.sin(-rad)
    dx, dy = xx - cx, yy - cy
    lx = (dx * cos_t - dy * sin_t) / a
    ly = (dx * sin_t + dy * cos_t) / b
    d = np.sqrt(lx**2 + ly**2)
    # Falloff shape: 1 − t^falloff_power.
    #   power=1   → linear (LR-style)
    #   power=0.5 → fast initial drop, long faint tail (small visible bright region + halo)
    #   power=2   → slow drop near face, sharp cliff toward outer
    t = np.clip((d - inner_mult) / max(outer_mult - inner_mult, 1e-3), 0, 1)
    mask = 1.0 - t ** falloff_power
    return mask.astype(np.float32)


def upscale_replicate(in_path, scale=4, max_retries=8, max_input_pixels=2_000_000):
    """Real-ESRGAN 2x or 4x upscale via Replicate. Auto-downsamples input that
    exceeds Real-ESRGAN's GPU memory limit (~2.1M pixels). Returns bytes."""
    import replicate
    src = Image.open(in_path)
    pixels = src.size[0] * src.size[1]
    actual_path = in_path
    if pixels > max_input_pixels:
        ratio = (max_input_pixels / pixels) ** 0.5
        new_w, new_h = int(src.size[0] * ratio), int(src.size[1] * ratio)
        small = src.resize((new_w, new_h), Image.LANCZOS)
        small_path = Path(in_path).with_name(Path(in_path).stem + "__pre_upscale_resized.jpg")
        small.save(small_path, quality=95)
        actual_path = str(small_path)
        print(f"  resize for upscale: {src.size}→{(new_w,new_h)} ({pixels//1000}K→{(new_w*new_h)//1000}K px)")
    for attempt in range(max_retries):
        try:
            with open(actual_path, "rb") as f:
                out = replicate.run(UPSCALE_MODEL, input={
                    "image": f, "scale": scale, "face_enhance": False})
            item = out[0] if isinstance(out, list) and out else out
            if hasattr(item, "read"): return item.read()
            if isinstance(item, str):
                with urllib.request.urlopen(item, timeout=60) as r: return r.read()
            raise RuntimeError(f"unexpected upscale output: {type(out)}")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "throttled" in msg.lower():
                import re as _re
                m = _re.search(r"resets in ~(\d+)s", msg)
                wait = int(m.group(1)) + 1 if m else min(60, 2**attempt + 5)
                time.sleep(wait); continue
            raise
    raise RuntimeError("upscale: max retries exceeded")


def add_grain(arr, strength=0.025, seed=None, grain_mask=None, inside_pct=1.0):
    """Add gaussian grain. If grain_mask given (0=outside grain ellipse, 1=inside),
    grain strength is `strength * inside_pct` inside, `strength` outside, linear lerp
    in between."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(arr.shape[:2]).astype(np.float32) * 255  # base noise
    if grain_mask is None:
        per_px = np.full(arr.shape[:2], strength, dtype=np.float32)
    else:
        per_px = strength * (inside_pct * grain_mask + (1.0 - grain_mask))
    n = noise * per_px
    out = arr.astype(np.float32) + (n[..., None] if arr.ndim == 3 else n)
    return np.clip(out, 0, 255).astype(np.uint8)

# --- Steps 3 + 4: histogram-match + composite ---------------------------

def grayscale(img_pil):
    return np.asarray(img_pil.convert("L"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--relit", required=True, help="relit color photo (output of relighting.py)")
    p.add_argument("--style", required=True, help="style reference image")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--prompt", default="long-exposure black and white motion-blur dance photograph, dramatic side lighting")
    p.add_argument("--negative-prompt", default="lowres, blurry artifacts, deformed body, mutation, ugly")
    p.add_argument("--instant-id-strength",      type=float, default=1.0)
    p.add_argument("--image-to-become-strength", type=float, default=0.75)
    p.add_argument("--denoising-strength",       type=float, default=1.0)
    p.add_argument("--depth-strength",           type=float, default=0.8)
    p.add_argument("--prompt-strength",          type=float, default=2.0)
    p.add_argument("--mask-inner-mult",          type=float, default=0.7,
                   help="inner sharp region as multiple of face ellipse")
    p.add_argument("--mask-outer-mult",          type=float, default=3.0,
                   help="outer 0-edge as multiple of face ellipse")
    p.add_argument("--mask-falloff-power",       type=float, default=1.0,
                   help="0.5=fast-then-tail, 1=linear, 2=slow-then-cliff")
    p.add_argument("--grain",                    type=float, default=0.0,
                   help="film-grain strength outside the grain ellipse (0=off, 0.02-0.04 typical)")
    p.add_argument("--grain-inside-pct",         type=float, default=0.25,
                   help="grain strength inside grain inner ellipse, as fraction of --grain")
    p.add_argument("--grain-ellipse-scale",      type=float, default=2.0,
                   help="grain ellipse size = face ellipse × this (bigger = wider face-protect zone)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-intermediates", action="store_true",
                   help="save bw_relit, surreal_gray, mask, matched, alongside final")
    p.add_argument("--save-stack", action="store_true",
                   help="export each pipeline step as a layer in a multi-page TIFF "
                        "(saved alongside final as <tag>__stack.tif)")
    p.add_argument("--no-face-overlay", action="store_true",
                   help="skip face mask + composite (use when source has no detectable face)")
    p.add_argument("--upscale", type=int, default=4, choices=[0, 2, 4],
                   help="post Real-ESRGAN upscale (0=off, 2x, 4x). Default 4x.")
    # Step 4.5 — text overlay (before grading so grade sweeps text too)
    p.add_argument("--text", default=None,
                   help='overlay text; "auto" = semantic NN over literary DB; '
                        'or explicit "..." string. Empty = no text.')
    p.add_argument("--text-style", choices=["quote","subtitle"], default="quote")
    p.add_argument("--text-orientation", choices=["horizontal","body","manual"],
                   default="body")
    p.add_argument("--text-angle", type=float, default=None)
    p.add_argument("--text-size-pct", type=float, default=3.0)
    # Step 5 — color grade
    p.add_argument("--grade", default="off",
                   help='warm-cool | split | wash:<color> | off')
    p.add_argument("--grade-strength", type=float, default=0.25)
    args = p.parse_args()

    token = load_token()
    if not token:
        print("REPLICATE_API_TOKEN missing", file=sys.stderr); sys.exit(2)
    os.environ["REPLICATE_API_TOKEN"] = token

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    relit_path = Path(args.relit).expanduser()
    style_path = Path(args.style).expanduser()
    if not relit_path.is_file(): print(f"missing relit: {relit_path}"); sys.exit(2)
    if not style_path.is_file(): print(f"missing style: {style_path}"); sys.exit(2)

    # Stack of pipeline stages for --save-stack
    stack_layers = []
    def _stash(name, img):
        if not args.save_stack: return
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.fromarray(img)
        stack_layers.append((name, img.copy() if hasattr(img, "copy") else img))

    stem = relit_path.stem.replace(" ", "_").replace("-", "_")[:60]
    style_stem = style_path.stem.replace(" ", "_").replace("-", "_")[:30]
    tag = f"{stem}__style_{style_stem}"

    # 0.5 — B&W with curve
    relit_pil = Image.open(relit_path).convert("RGB")
    _stash("00_relit", relit_pil)
    bw_relit = bw_with_curve(relit_pil)
    bw_relit_path = out_dir / f"{tag}__bw_relit.jpg"
    bw_relit.save(bw_relit_path, quality=95)
    print(f"step 0.5: bw_with_curve → {bw_relit_path.name}")
    _stash("01_bw_relit", bw_relit)

    # 1 — become-image
    print("step 1: become-image …")
    t0 = time.time()
    surreal_blob = run_become_image(
        bw_relit_path, style_path, args.prompt, args.negative_prompt,
        args.instant_id_strength, args.image_to_become_strength,
        args.denoising_strength, args.depth_strength,
        args.prompt_strength, args.seed)
    surreal_path = out_dir / f"{tag}__surreal.jpg"
    surreal_path.write_bytes(surreal_blob)
    print(f"  surreal → {surreal_path.name}  ({time.time()-t0:.1f}s)")

    # 1.5 — grayscale the surreal so we composite in B&W
    surreal_pil = Image.open(surreal_path).convert("RGB")
    # Resize surreal to bw_relit dimensions if mismatched
    if surreal_pil.size != bw_relit.size:
        print(f"  resizing surreal {surreal_pil.size} → {bw_relit.size}")
        surreal_pil = surreal_pil.resize(bw_relit.size, Image.LANCZOS)
    _stash("02_surreal", surreal_pil)
    surreal_gray = grayscale(surreal_pil)        # H×W uint8
    bw_arr      = grayscale(bw_relit)            # H×W uint8
    _stash("03_surreal_gray", surreal_gray)

    # 2 — face radial mask on bw_relit (skipped in --no-face-overlay)
    relit_arr = np.asarray(relit_pil)
    if relit_arr.shape[:2] != bw_arr.shape:
        relit_arr = np.asarray(relit_pil.resize(bw_relit.size, Image.LANCZOS))
    if args.no_face_overlay:
        print("step 2: SKIPPED (--no-face-overlay)")
        mask = np.zeros(bw_arr.shape, dtype=np.float32)
    else:
        print("step 2: face mask …")
        mask = face_ellipse_mask(bw_relit.size, relit_arr,
                                  inner_mult=args.mask_inner_mult,
                                  outer_mult=args.mask_outer_mult,
                                  falloff_power=args.mask_falloff_power)

    _stash("04_face_mask", (mask * 255).astype(np.uint8))

    # 3 — histogram-match bw_relit's tone to surreal_gray
    print("step 3: histogram match …")
    bw_matched = match_histograms(bw_arr, surreal_gray).astype(np.uint8)
    _stash("05_bw_matched", bw_matched)

    # 4 — composite
    print("step 4: composite …")
    final = surreal_gray.astype(np.float32) * (1 - mask) \
            + bw_matched.astype(np.float32) * mask
    final_u8 = np.clip(final, 0, 255).astype(np.uint8)
    if args.grain > 0:
        # Build a larger ellipse mask for grain attenuation: mask=1 inside (low grain),
        # 0 outside (full grain), linear lerp in the falloff ring.
        scale = args.grain_ellipse_scale
        grain_mask = face_ellipse_mask(
            bw_relit.size, relit_arr,
            inner_mult=args.mask_inner_mult * scale,
            outer_mult=args.mask_outer_mult * scale,
            falloff_power=1.0,   # always linear for grain to match user's mental model
        )
        final_u8 = add_grain(final_u8, strength=args.grain, seed=args.seed,
                              grain_mask=grain_mask, inside_pct=args.grain_inside_pct)
        if args.save_intermediates:
            Image.fromarray((grain_mask * 255).astype(np.uint8)).save(
                out_dir / f"{tag}__grain_mask.png")
        print(f"  added grain ({args.grain} outside, {args.grain*args.grain_inside_pct:.3f} inside, ellipse-scale={scale})")
    final_arr = np.stack([final_u8]*3, axis=-1)
    _stash("06_composite", final_arr)

    # Step 4.5 — optional text overlay (BEFORE grading so grade sweeps text too)
    if args.text:
        try:
            from text_overlay import overlay as _txt_overlay, pick_quote_auto
            tmp_path = out_dir / f"{tag}__pre_text.jpg"
            Image.fromarray(final_arr).save(tmp_path, quality=92)
            text = pick_quote_auto(tmp_path) if args.text == "auto" else args.text
            tmp_path.unlink(missing_ok=True)
            final_arr = _txt_overlay(final_arr, text,
                style=args.text_style, orientation=args.text_orientation,
                manual_angle_deg=args.text_angle, font_size_pct=args.text_size_pct)
            print(f"  step 4.5 text overlay applied")
            _stash("07_post_text", final_arr)
        except Exception as e:
            print(f"  text overlay FAIL: {e}")

    # Step 4.7 — optional color grade
    if args.grade and args.grade != "off":
        try:
            from color_grade import grade as _grade
            final_arr = _grade(final_arr, args.grade, strength=args.grade_strength)
            print(f"  step 4.7 grade applied ({args.grade})")
            _stash("08_post_grade", final_arr)
        except Exception as e:
            print(f"  grade FAIL: {e}")

    final_pil = Image.fromarray(final_arr)
    final_path = out_dir / f"{tag}__final.jpg"
    final_pil.save(final_path, quality=92)
    print(f"  → {final_path}")
    _stash("09_final", final_pil)

    # Write multi-page TIFF stack alongside final
    if args.save_stack and stack_layers:
        try:
            from _layered_tiff import save_stack
            stack_path = out_dir / f"{tag}__stack.tif"
            save_stack(stack_path, stack_layers)
            print(f"  → stack: {stack_path}  ({len(stack_layers)} layers)")
        except Exception as e:
            print(f"  save-stack FAIL: {e}")

    # Step 5 — optional Real-ESRGAN upscale
    if args.upscale and args.upscale > 0:
        print(f"step 5: upscale {args.upscale}× via Real-ESRGAN …")
        t0 = time.time()
        try:
            up_blob = upscale_replicate(final_path, scale=args.upscale)
            up_path = out_dir / f"{tag}__final_{args.upscale}x.jpg"
            up_path.write_bytes(up_blob)
            print(f"  → {up_path}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  upscale FAIL: {e}")

    # Sidecar metadata
    meta = {
        "relit": str(relit_path), "style": str(style_path),
        "tag": tag, "final": str(final_path),
        "params": {
            "instant_id_strength": args.instant_id_strength,
            "image_to_become_strength": args.image_to_become_strength,
            "denoising_strength": args.denoising_strength,
            "depth_strength": args.depth_strength,
            "prompt_strength": args.prompt_strength,
            "mask_inner_mult": args.mask_inner_mult,
            "mask_outer_mult": args.mask_outer_mult,
            "grain": args.grain,
            "seed": args.seed,
            "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        },
        "tool": "surreal_with_face",
    }
    final_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    if args.save_intermediates:
        Image.fromarray(surreal_gray).save(out_dir / f"{tag}__surreal_gray.jpg", quality=92)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{tag}__mask.png")
        Image.fromarray(bw_matched).save(out_dir / f"{tag}__bw_matched.jpg", quality=92)


if __name__ == "__main__":
    main()
