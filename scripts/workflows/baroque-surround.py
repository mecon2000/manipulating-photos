#!/home/rong/openclaw-venv/bin/python3
"""
Baroque Surround — Generative Painterly Background

Extracts subject, generates a dramatic BG via Flux text-to-image (with optional
story elements like roses, wings, chains), composites using Laplacian pyramid
blending + light wrap + LAB edge match + LAB color wash for unified color.

Pipeline:
  1. Extract subject mask (BiRefNet)        } parallel
  2. Generate BG (Flux text-to-image)       }
  3. Laplacian pyramid blend (6 levels)
  4. Light wrap (BG spill on subject edges)
  5. LAB edge color match
  6. Full-image LAB color wash (60%)

Usage:
    python baroque-surround.py --source photo.jpg --preset smoke
    python baroque-surround.py --source photo.jpg --preset roses --artifact petals
    python baroque-surround.py --source photo.jpg --preset baroque --artifact random
    python baroque-surround.py --list-presets
    python baroque-surround.py --list-artifacts
"""

import os
import sys

_env_file = os.path.expanduser("~/sol/.env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())
os.environ['FAL_KEY'] = os.environ.get('FAL_API_KEY', '')

import re
import json
import time
import random
import base64
import argparse
import shutil
import tempfile
import threading
from io import BytesIO
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat
import fal_client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_mask
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Presets — base atmosphere for the BG
# ---------------------------------------------------------------------------
PRESETS = {
    "baroque": {
        "prompt": "large flowing amorphous organic shapes and billowing drapery, baroque oil painting, dramatic chiaroscuro, luminous glazing, Bouguereau and Caravaggio, warm ochre cool blue-grey cream, smooth blended brushwork, sweeping undulating forms",
        "negative": "modern, digital, sharp edges, text, watermark, flat colors, cartoon, solid color background",
    },
    "renaissance": {
        "prompt": "large soft amorphous forms of golden light and flowing draped silk fabric, sfumato Renaissance oil painting, Raphael da Vinci, olive warm brown soft blue, luminous atmospheric depth, billowing organic forms",
        "negative": "modern, digital, harsh lighting, text, watermark, flat background, solid color",
    },
    "dark-romantic": {
        "prompt": "large swirling amorphous storm forms and turbulent abstract shapes, dark romantic oil painting, Delacroix Turner, dark blue warm amber charcoal copper, flowing organic masses, dramatic atmospheric turbulence",
        "negative": "bright, cheerful, flat, text, watermark, cartoon, solid background",
    },
    "ethereal": {
        "prompt": "large flowing amorphous luminous cloud forms and soft ethereal mist, dreamy angelic, billowing organic shapes in pearl ivory pale gold soft blue, divine radiance, sweeping undulating cloud-like masses",
        "negative": "dark, gritty, harsh, text, watermark, modern, flat background",
    },
    "smoke": {
        "prompt": "large visible swirling smoke plumes and flowing amorphous grey volumetric forms, dramatic single light source illuminating billowing smoke, warm grey amber cream emerging from shadows, Caravaggio chiaroscuro, dense volumetric smoke clouds",
        "negative": "flat black, solid black, empty background, text, watermark, plain background",
    },
    "underwater": {
        "prompt": "deep underwater scene with volumetric light rays penetrating dark ocean water, large flowing organic jellyfish-like forms and bioluminescent particles, swirling ocean currents carrying soft blue green teal glowing shapes, deep sea atmosphere",
        "negative": "text, watermark, surface, sky, dry, land, flat",
    },
    "ink-water": {
        "prompt": "large flowing ink drops dissolving in water, organic amorphous spreading ink forms in deep indigo black and warm sienna, mesmerizing fluid dynamics, billowing ink tendrils and blooming clouds of pigment in clear water",
        "negative": "text, watermark, flat, solid color, dry, paper",
    },
    "aurora": {
        "prompt": "sweeping northern lights aurora borealis forms, large flowing luminous curtains of green teal purple pink light against dark starry sky, organic undulating ribbons of light, atmospheric glow",
        "negative": "text, watermark, flat, daylight, sun, bright",
    },
    "silk": {
        "prompt": "large flowing luxurious silk fabric forms billowing in wind, organic draping shapes in rich burgundy gold ivory, volumetric folds catching dramatic light, Renaissance drapery study, sensual flowing textile",
        "negative": "text, watermark, flat, modern, digital, hard edges",
    },
    "embers": {
        "prompt": "swirling embers and warm smoke forms rising in dramatic updraft, glowing orange sparks and flowing ash shapes against dark background, volumetric fire glow, warm amber red black, cinematic atmosphere",
        "negative": "text, watermark, flat, bright, daylight, cold",
    },
    "curtains": {
        "prompt": "large crumpled heavy velvet curtains and draped theatrical fabric, rich burgundy crimson deep purple and gold fabric folds, dramatic stage lighting from above, luxurious textile wrinkles and creases, baroque theater",
        "negative": "text, watermark, flat, modern, smooth, digital",
    },
    "whipped-cream": {
        "prompt": "massive mountains and peaks of glossy white whipped cream, organic flowing cream swirls and peaks, soft volumetric meringue forms, creamy vanilla and soft pink highlights, dreamy confection landscape",
        "negative": "text, watermark, flat, dark, gritty, dry",
    },
    "bubbles": {
        "prompt": "large floating iridescent soap bubbles and heavy foam clusters, translucent spheres with rainbow reflections, thick soapy lather and bubble masses, soft diffused light through transparent orbs, dreamy bathroom atmosphere",
        "negative": "text, watermark, flat, dry, dark, harsh",
    },
    "velvet-fog": {
        "prompt": "deep burgundy and navy mist in soft heavy velvet folds, dense moody fog draped in layered folds, low-frequency volumetric haze, rich jewel tones dissolving into shadow, dark romantic atmosphere",
        "negative": "text, watermark, flat, bright, daylight, sharp edges, solid color",
    },
    "coral-smoke": {
        "prompt": "warm terracotta and dusty rose smoke in loose organic coils, flowing amorphous smoke plumes, soft peach cream ochre swirling through warm air, gentle turbulent coils, sunset-lit haze",
        "negative": "text, watermark, flat, blue, cold, sharp, solid color",
    },
    "neon-smoke-rings": {
        "prompt": "layered rings and curls of electric pink and cyan neon smoke unraveling in darkness, glowing magenta and turquoise smoke forms, volumetric neon glow on black, dreamy cyberpunk haze, luminous colored vapor",
        "negative": "text, watermark, flat, daylight, sun, pastel, solid color",
    },
    "burning-silk": {
        "prompt": "sheer white fabric mid-disintegration with glowing orange-red burning edges on deep black background, ember-lit holes in translucent silk, smoldering fabric dissolving into sparks, dramatic high contrast",
        "negative": "text, watermark, flat, bright daylight, intact fabric, solid color",
    },
    "torn-cloud": {
        "prompt": "thick white cloud blanket ripped open at center with warm golden light pouring through the tear, billowing cloud edges curling around a luminous gap, sunlit rays breaking through overcast sky, Turner dramatic heaven",
        "negative": "text, watermark, flat, clear sky, solid color, digital",
    },
    "spun-sugar": {
        "prompt": "translucent white and blush pink fibrous wisps of spun sugar dissolving at edges, fine candy floss strands stretched across soft pastel haze, delicate confection threads, pale lavender highlights",
        "negative": "text, watermark, flat, dark, gritty, sharp, solid color",
    },
    "powdered-pigment": {
        "prompt": "dense clouds of dry ochre cobalt and crimson pigment caught mid-explosion, powder paint particles scattering in mid-air, layered holi powder bursts, fine colored dust in slow motion, rich saturated pigment clouds",
        "negative": "text, watermark, flat, wet, liquid, solid color, sharp",
    },
}

# ---------------------------------------------------------------------------
# Artifacts — story elements added to the BG prompt (1 per image)
# ---------------------------------------------------------------------------
ARTIFACTS = {
    "wings": "huge dark crow wings spreading outward, dark feathers dissolving into smoke at the tips",
    "petals": "hundreds of dark red rose petals floating and swirling through the air, wilting baroque roses with thorny stems",
    "hands": "ghostly pale hands reaching upward through the clouds, ethereal fingers emerging from mist",
    "faces": "tortured amorphic faces emerging from the smoke with open mouths, anguished expressions dissolving into mist",
    "chains": "thick ornate iron chains hanging and draping through the smoke, dark iron links dissolving into mist",
    "serpents": "a large serpent coiling through the haze with iridescent scales, sinuous body dissolving into smoke",
    "butterflies": "dozens of dark moths and butterflies with translucent wings, scattered and floating through the haze",
    "thorns": "twisted thorny vines creeping and weaving through the smoke, dark brambles with sharp barbs",
    "feathers": "loose dark feathers floating and drifting through the air, individual plumes caught in slow motion",
    "flames": "ethereal blue and amber flames dancing and flickering through the darkness, ghostly fire",
    "flowers": "large baroque flowers blooming and wilting in the haze, peonies and dahlias with heavy petals",
    "skulls": "ornate baroque skulls partially emerging from the smoke, vanitas memento mori, gilded bone",
    "ribbons": "flowing silk ribbons and fabric strips twisting through the air, caught in wind",
    "eyes": "multiple ethereal eyes peering from within the smoke, mysterious watchful presences",
}

_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(output_dir, message, level="INFO"):
    israel_time = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{israel_time}] [{level}] {message}"
    print(formatted)
    if output_dir:
        with _log_lock:
            try:
                with open(os.path.join(output_dir, "workflow.log"), "a") as f:
                    f.write(formatted + "\n")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def check_image_quality(img, label, output_dir):
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness, contrast, entropy = stat.mean[0], stat.stddev[0], gray.entropy()
    reasons = []
    if brightness < 10:
        reasons.append(f"nearly black (brightness={brightness:.1f})")
    elif brightness > 245:
        reasons.append(f"nearly white (brightness={brightness:.1f})")
    if contrast < 5:
        reasons.append(f"flat/uniform (contrast={contrast:.1f})")
    if entropy < 1.0:
        reasons.append(f"zero-entropy (entropy={entropy:.2f})")
    ok = len(reasons) == 0
    if not ok:
        log(output_dir, f"QUALITY FAIL [{label}]: {'; '.join(reasons)}", "WARN")
    else:
        log(output_dir, f"Quality OK [{label}]: brightness={brightness:.1f} contrast={contrast:.1f} entropy={entropy:.2f}")
    return {"ok": ok, "brightness": round(brightness, 1), "contrast": round(contrast, 1), "entropy": round(entropy, 2)}


# ---------------------------------------------------------------------------
# Gemini Evaluation
# ---------------------------------------------------------------------------
_EVAL_PROMPT = """\
You are a professional art director evaluating a composite photograph where the subject is photographic \
and the background has been replaced with generative painterly forms (baroque/classical oil painting style).

Evaluate the image on these criteria:
1. Subject integrity: does the person look untouched, photorealistic, and anatomically correct?
2. Surround quality: does the painterly background look like convincing oil painting?
3. Transition: is the blend between photographic subject and painted surround smooth and natural?
4. Color harmony: do the subject and surround share the same color temperature?
5. Overall cohesion: does it feel like one unified image, not a cutout on a painted background?

Respond ONLY with valid JSON (no markdown fences):
{"score": <int 1-10>, "critique": "<2-3 sentences>", "issues": [<zero or more from: "subject_altered", "harsh_transition", "flat_background", "color_clash", "artifacts", "too_dark", "too_bright", "cutout_look", "incoherent">]}"""


def evaluate_with_gemini(img, output_dir, original_img=None):
    import requests
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log(output_dir, "GOOGLE_API_KEY not set — skipping Gemini evaluation")
        return None
    try:
        def _img_to_b64(im, max_size=1024):
            im_resized = im.copy()
            im_resized.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = BytesIO()
            im_resized.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        parts = []
        if original_img is not None:
            parts.append({"text": "ORIGINAL:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _img_to_b64(original_img)}})
        parts.append({"text": "RESULT:" if original_img else ""})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _img_to_b64(img)}})
        parts.append({"text": _EVAL_PROMPT})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096, "responseMimeType": "application/json"},
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json=payload, timeout=60)
        if response.status_code != 200:
            log(output_dir, f"Gemini API error ({response.status_code}): {response.text[:200]}", "WARN")
            return None
        resp_json = response.json()
        candidates = resp_json.get("candidates", [])
        if not candidates:
            log(output_dir, f"Gemini blocked: {resp_json.get('promptFeedback', {}).get('blockReason', '?')}", "WARN")
            return None
        raw = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```")).strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                result = json.loads(raw[start:end + 1])
            else:
                log(output_dir, f"Gemini JSON parse failed: {raw[:200]}", "WARN")
                return None
        log(output_dir, f"Gemini score: {result.get('score', '?')}/10 — {result.get('critique', '')}")
        return result
    except Exception as e:
        log(output_dir, f"Gemini evaluation failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Step 1: Extract mask (runs in thread)
# ---------------------------------------------------------------------------
def extract_mask_thread(img_orig, w, h, output_dir, result_dict):
    try:
        mask, mask_info = build_mask(img_orig, affect="subject", exclude="", output_dir=output_dir, feather=0)
        if mask is None:
            result_dict["error"] = "Subject extraction failed"
            return
        if mask.size != (w, h):
            mask = mask.resize((w, h), Image.LANCZOS)
        mask.save(os.path.join(output_dir, "1_mask.png"))
        result_dict["mask"] = mask
        result_dict["mask_info"] = mask_info
    except Exception as e:
        result_dict["error"] = str(e)


BG_CACHE_DIR = os.path.expanduser("~/.openclaw/workspace/shared/bg_cache")

def load_cached_bg(preset_name, artifact_name, w, h):
    """Pick a cached BG matching preset (+artifact if possible). Returns PIL.Image or None."""
    idx_path = os.path.join(BG_CACHE_DIR, "index.json")
    if not os.path.isfile(idx_path):
        return None
    try:
        with open(idx_path) as f:
            entries = json.load(f)
    except Exception:
        return None
    target_aspect = w / h
    def score(e):
        s = 0
        if e.get("preset") == preset_name: s += 100
        if artifact_name and artifact_name != "none" and e.get("artifact") == artifact_name: s += 50
        # aspect penalty
        a = e.get("aspect", 1.0)
        s -= abs(a - target_aspect) * 10
        return s
    matches = [e for e in entries if e.get("preset") == preset_name and os.path.isfile(os.path.join(BG_CACHE_DIR, e["file"]))]
    if not matches:
        return None
    matches.sort(key=score, reverse=True)
    # sample from top 3 to vary
    top = matches[:3]
    pick = random.choice(top)
    try:
        img = Image.open(os.path.join(BG_CACHE_DIR, pick["file"])).convert("RGB")
        # center-crop to target aspect, then resize
        iw, ih = img.size
        src_aspect = iw / ih
        if src_aspect > target_aspect:
            new_w = int(ih * target_aspect)
            left = (iw - new_w) // 2
            img = img.crop((left, 0, left + new_w, ih))
        else:
            new_h = int(iw / target_aspect)
            top_ = (ih - new_h) // 2
            img = img.crop((0, top_, iw, top_ + new_h))
        img = img.resize((w, h), Image.LANCZOS)
        return img, pick["file"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 2: Generate BG (runs in thread)
# ---------------------------------------------------------------------------
def generate_bg_thread(prompt, w, h, output_dir, seed, result_dict, flux_model="dev"):
    import requests as req_lib
    try:
        # Flux dimensions: cap at 1024 on long edge, maintain aspect ratio
        flux_w = min(w, 1024)
        flux_h = int(flux_w * h / w)
        flux_h = (flux_h // 8) * 8
        flux_w = (flux_w // 8) * 8

        endpoint = "fal-ai/flux/schnell" if flux_model == "schnell" else "fal-ai/flux/dev"
        steps = 4 if flux_model == "schnell" else 28
        log(output_dir, f"Generating BG: {flux_w}x{flux_h} via {endpoint} ({steps} steps)")
        fa = {
            "prompt": prompt,
            "image_size": {"width": flux_w, "height": flux_h},
            "num_inference_steps": steps,
            "num_images": 1,
            "output_format": "jpeg",
            "enable_safety_checker": False,
            "seed": seed,
        }
        if flux_model != "schnell":
            fa["guidance_scale"] = 3.5
        handle = fal_client.submit(endpoint, arguments=fa)
        result = handle.get()
        images = result.get("images", [])
        if not images:
            result_dict["error"] = "BG generation returned no images"
            return
        bg_url = images[0].get("url", "")
        resp = req_lib.get(bg_url, timeout=60)
        bg_img = Image.open(BytesIO(resp.content)).convert("RGB")
        if bg_img.size != (w, h):
            bg_img = bg_img.resize((w, h), Image.LANCZOS)
        bg_img.save(os.path.join(output_dir, "2_bg.jpg"), "JPEG", quality=95)
        result_dict["bg"] = bg_img
    except Exception as e:
        result_dict["error"] = str(e)


# ---------------------------------------------------------------------------
# Steps 3-6: Composite pipeline (local, fast)
# ---------------------------------------------------------------------------
def composite_pipeline(src_f, bg_img, mask_binary, w, h, short_edge, output_dir, stages=None):
    """Laplacian pyramid blend + light wrap + LAB edge match + LAB 60% wash.

    If `stages` is a dict, it is populated with PIL.Image snapshots after each
    sub-stage keyed by 'lap_blend', 'light_wrap', 'lab_edge_match', 'lab_full_wash'.
    """
    def _snap(arr):
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    bg_f = np.array(bg_img).astype(np.float32)

    # --- Step 3: Laplacian pyramid blend ---
    log(output_dir, "Step 3: Laplacian pyramid blend (6 levels)")

    def lap_pyr(img_f, levels=6):
        pyr, cur = [], img_f.copy()
        for _ in range(levels - 1):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cur - up)
            cur = down
        pyr.append(cur)
        return pyr

    def gauss_pyr(m, levels=6):
        pyr, cur = [m.copy()], m.copy()
        for _ in range(levels - 1):
            cur = cv2.pyrDown(cur)
            pyr.append(cur)
        return pyr

    def reconstruct(pyr):
        cur = pyr[-1]
        for i in range(len(pyr) - 2, -1, -1):
            cur = cv2.pyrUp(cur, dstsize=(pyr[i].shape[1], pyr[i].shape[0])) + pyr[i]
        return cur

    levels = 6
    s_pyr = lap_pyr(src_f, levels)
    b_pyr = lap_pyr(bg_f, levels)
    m3 = np.stack([mask_binary.astype(np.float32)] * 3, axis=-1)
    m_pyr = gauss_pyr(m3, levels)
    blended = [s * m + b * (1 - m) for s, b, m in zip(s_pyr, b_pyr, m_pyr)]
    result = np.clip(reconstruct(blended), 0, 255).astype(np.float32)
    if stages is not None:
        stages["lap_blend"] = _snap(result)

    # --- Step 4: Light wrap ---
    log(output_dir, "Step 4: Light wrap")
    blur_r = max(30, int(short_edge * 0.08))
    bg_blur = cv2.GaussianBlur(bg_f, (0, 0), blur_r)
    ks = max(5, int(short_edge * 0.025))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(mask_binary, kern, iterations=1)
    edge_band = ((dilated - mask_binary) > 0).astype(np.float32)
    edge_soft = cv2.GaussianBlur(edge_band, (0, 0), max(3, ks // 2))[:, :, np.newaxis]
    result = result * (1 - edge_soft * 0.25) + bg_blur * (edge_soft * 0.25)
    if stages is not None:
        stages["light_wrap"] = _snap(result)

    # --- Step 5: LAB edge color match ---
    log(output_dir, "Step 5: LAB edge color match")
    bg_lab = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    res_lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ew = max(10, int(short_edge * 0.05))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ew, ew))
    eroded = cv2.erode(mask_binary, ke, iterations=1)
    inner_edge = ((mask_binary - eroded) > 0).astype(np.float32)
    inner_soft = cv2.GaussianBlur(inner_edge, (0, 0), max(3, ew // 2))
    for ch in range(3):
        bg_near = bg_lab[:, :, ch][edge_band > 0.3]
        subj_edge = res_lab[:, :, ch][inner_soft > 0.3]
        if len(bg_near) == 0 or len(subj_edge) == 0:
            continue
        res_lab[:, :, ch] += (bg_near.mean() - subj_edge.mean()) * 0.4 * inner_soft
    result = cv2.cvtColor(np.clip(res_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    if stages is not None:
        stages["lab_edge_match"] = _snap(result)

    # --- Step 6: Full-image LAB 60% wash ---
    log(output_dir, "Step 6: LAB 60% color wash")
    comp_lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in range(3):
        c_mean = comp_lab[:, :, ch].mean()
        c_std = comp_lab[:, :, ch].std() + 1e-8
        b_mean = bg_lab[:, :, ch].mean()
        b_std = bg_lab[:, :, ch].std() + 1e-8
        new_mean = c_mean + (b_mean - c_mean) * 0.6
        new_std = c_std + (b_std - c_std) * 0.18
        comp_lab[:, :, ch] = (comp_lab[:, :, ch] - c_mean) * (new_std / c_std) + new_mean
    final = cv2.cvtColor(np.clip(comp_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if stages is not None:
        stages["lab_full_wash"] = Image.fromarray(final)

    return Image.fromarray(final)


# ---------------------------------------------------------------------------
# Tensor Art tile-refine — ControlNet Tile workflow for de-AI-ification
# ---------------------------------------------------------------------------
TENSOR_BASE_URL = "https://ap-east-1.tensorart.cloud/v1"
TILE_REFINE_TEMPLATE_ID = "761883020545458610"

def _get_tensor_key():
    return os.environ.get("TENSOR_API_KEY", "")

def _get_fal_key():
    return os.environ.get("FAL_API_KEY", "") or os.environ.get("FAL_KEY", "")


def tensor_upload_image(image_pil, output_dir):
    """Upload a PIL image to Tensor Art, return resourceId."""
    import requests as req
    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    headers = {"Authorization": f"Bearer {_get_tensor_key()}", "Content-Type": "application/json"}
    try:
        res = req.post(f"{TENSOR_BASE_URL}/resource/image", json={}, headers=headers, timeout=30)
        if res.status_code != 200:
            log(output_dir, f"Tensor upload init failed ({res.status_code}): {res.text[:200]}", "ERROR")
            return None
        data = res.json()
        put = req.put(data["putUrl"], data=buf.getvalue(), headers=data["headers"], timeout=120)
        if put.status_code not in (200, 201):
            log(output_dir, f"Tensor upload PUT failed ({put.status_code})", "ERROR")
            return None
        return data["resourceId"]
    except Exception as e:
        log(output_dir, f"Tensor upload failed: {e}", "ERROR")
        return None


def tensor_tile_refine(image_pil, denoise, positive_prompt, output_dir):
    """Run the Ultimate SD Upscale + ControlNet Tile workflow. Returns refined PIL.Image or None.

    Denoise 0.15-0.35 range. Higher = more refinement but more face drift.
    """
    import requests as req
    resource_id = tensor_upload_image(image_pil, output_dir)
    if not resource_id:
        return None
    w, h = image_pil.size
    # Tile sizes must be <= image dims, multiples of 8
    tile_w = max(512, min(1024, (w // 16) * 8))
    tile_h = max(512, min(1024, (h // 16) * 8))
    seed = random.randint(1, 2**31 - 1)

    import hashlib as _h
    payload = {
        "requestId": _h.md5(f"{time.time()}-{seed}".encode()).hexdigest(),
        "templateId": TILE_REFINE_TEMPLATE_ID,
        "fields": {"fieldAttrs": [
            {"nodeId": "1", "fieldName": "upscale_by", "fieldValue": 1},
            {"nodeId": "1", "fieldName": "seed", "fieldValue": seed},
            {"nodeId": "1", "fieldName": "steps", "fieldValue": 20},
            {"nodeId": "1", "fieldName": "cfg", "fieldValue": 6},
            {"nodeId": "1", "fieldName": "sampler_name", "fieldValue": "dpmpp_2m"},
            {"nodeId": "1", "fieldName": "scheduler", "fieldValue": "karras"},
            {"nodeId": "1", "fieldName": "denoise", "fieldValue": float(denoise)},
            {"nodeId": "1", "fieldName": "mode_type", "fieldValue": "Linear"},
            {"nodeId": "1", "fieldName": "tile_width", "fieldValue": tile_w},
            {"nodeId": "1", "fieldName": "tile_height", "fieldValue": tile_h},
            {"nodeId": "1", "fieldName": "mask_blur", "fieldValue": 8},
            {"nodeId": "1", "fieldName": "tile_padding", "fieldValue": 32},
            {"nodeId": "1", "fieldName": "seam_fix_mode", "fieldValue": "None"},
            {"nodeId": "1", "fieldName": "seam_fix_denoise", "fieldValue": 1},
            {"nodeId": "1", "fieldName": "seam_fix_width", "fieldValue": 64},
            {"nodeId": "1", "fieldName": "seam_fix_mask_blur", "fieldValue": 8},
            {"nodeId": "1", "fieldName": "seam_fix_padding", "fieldValue": 16},
            {"nodeId": "1", "fieldName": "force_uniform_tiles", "fieldValue": True},
            {"nodeId": "1", "fieldName": "tiled_decode", "fieldValue": False},
            {"nodeId": "2", "fieldName": "image", "fieldValue": resource_id},
            {"nodeId": "3", "fieldName": "ckpt_name", "fieldValue": "600315593373002855"},
            {"nodeId": "4", "fieldName": "text", "fieldValue": positive_prompt},
            {"nodeId": "5", "fieldName": "text", "fieldValue": "plastic, cgi, painting, illustration, blurry, low quality, deformed, extra fingers"},
            {"nodeId": "6", "fieldName": "model_name", "fieldValue": "4x-UltraSharp.pth"},
            {"nodeId": "7", "fieldName": "filename_prefix", "fieldValue": "baroque-refine"},
            {"nodeId": "12", "fieldName": "strength", "fieldValue": 1},
            {"nodeId": "12", "fieldName": "start_percent", "fieldValue": 0},
            {"nodeId": "12", "fieldName": "end_percent", "fieldValue": 1},
            {"nodeId": "15", "fieldName": "image", "fieldValue": resource_id},
            {"nodeId": "20", "fieldName": "control_net_name", "fieldValue": "control_v11f1e_sd15_tile.pth"},
        ]},
    }
    headers = {"Authorization": f"Bearer {_get_tensor_key()}", "Content-Type": "application/json"}
    try:
        r = req.post(f"{TENSOR_BASE_URL}/jobs/workflow/template", headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            log(output_dir, f"Tile-refine submit failed ({r.status_code}): {r.text[:300]}", "ERROR")
            return None
        job_id = r.json().get("job", {}).get("id")
        if not job_id:
            log(output_dir, f"Tile-refine no job id: {r.text[:200]}", "ERROR")
            return None
    except Exception as e:
        log(output_dir, f"Tile-refine submit exception: {e}", "ERROR")
        return None

    # Poll
    for i in range(90):
        time.sleep(3)
        try:
            res = req.get(f"{TENSOR_BASE_URL}/jobs/{job_id}", headers=headers, timeout=15).json()
        except Exception:
            continue
        job = res.get("job", {})
        status = job.get("status")
        if status == "SUCCESS":
            images = job.get("successInfo", {}).get("images") or job.get("resultSets") or []
            # Try common shapes
            url = None
            if isinstance(images, list) and images:
                first = images[0]
                url = first.get("url") if isinstance(first, dict) else None
            if not url:
                # fallback: scan dict for any url
                import json as _j
                txt = _j.dumps(job)
                import re as _re
                m = _re.search(r'"url"\s*:\s*"([^"]+)"', txt)
                if m:
                    url = m.group(1)
            if not url:
                log(output_dir, f"Tile-refine SUCCESS but no url: {str(job)[:300]}", "ERROR")
                return None
            try:
                img = Image.open(BytesIO(req.get(url, timeout=60).content)).convert("RGB")
                if img.size != image_pil.size:
                    img = img.resize(image_pil.size, Image.LANCZOS)
                return img
            except Exception as e:
                log(output_dir, f"Tile-refine download failed: {e}", "ERROR")
                return None
        if status in ("FAILED", "CANCELED"):
            log(output_dir, f"Tile-refine {status}: {job.get('failedInfo', {})}", "ERROR")
            return None
    log(output_dir, "Tile-refine poll timeout", "ERROR")
    return None


def fal_face_swap(source_path, target_pil, output_dir):
    """Swap source face onto target. Returns PIL or None."""
    import requests as req
    # Save target to temp
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        target_pil.save(tf.name, "JPEG", quality=95)
        target_path = tf.name
    try:
        with open(source_path, "rb") as f:
            src_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(target_path, "rb") as f:
            tgt_b64 = base64.b64encode(f.read()).decode("utf-8")
        headers = {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}
        payload = {
            "base_image_url": f"data:image/jpeg;base64,{tgt_b64}",
            "swap_image_url": f"data:image/jpeg;base64,{src_b64}",
        }
        r = req.post("https://fal.run/fal-ai/face-swap", headers=headers, json=payload, timeout=180)
        if r.status_code == 200:
            url = r.json().get("image", {}).get("url")
            if url:
                return Image.open(req.get(url, stream=True, timeout=30).raw).convert("RGB")
        log(output_dir, f"Face swap failed ({r.status_code}): {r.text[:200]}", "ERROR")
        return None
    finally:
        try: os.unlink(target_path)
        except OSError: pass


# ---------------------------------------------------------------------------
# Grain match — extract high-freq from original, add to final to unify grain
# ---------------------------------------------------------------------------
def apply_grain_match(final_img, original_img, strength, output_dir):
    """Extract sensor grain from original (high-pass) and add to final.

    Fixes the 'AI-glued-on' look — the photo's sensor noise covers both
    subject and BG, so the composite reads as one capture instead of
    'clean generated BG + noisy photo subject'.
    """
    orig_f = np.array(original_img).astype(np.float32)
    final_f = np.array(final_img).astype(np.float32)

    # High-pass grain from original (two-level: fine texture + slightly coarser)
    blur_fine = cv2.GaussianBlur(orig_f, (0, 0), 1.2)
    grain_fine = orig_f - blur_fine  # sub-pixel sensor noise

    blur_coarse = cv2.GaussianBlur(orig_f, (0, 0), 3.0)
    grain_coarse = orig_f - blur_coarse - grain_fine * 0.5  # mid-freq texture

    grain = grain_fine + grain_coarse * 0.4

    # Normalize to zero-mean so we're only adding noise, not shifting brightness
    grain = grain - grain.mean(axis=(0, 1), keepdims=True)

    out = final_f + grain * strength
    result = np.clip(out, 0, 255).astype(np.uint8)

    # Debug: save grain layer visualized
    try:
        g_vis = np.clip(grain * 6 + 128, 0, 255).astype(np.uint8)
        Image.fromarray(g_vis).save(os.path.join(output_dir, "8_grain_layer.jpg"), "JPEG", quality=85)
    except Exception:
        pass

    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# Foreground wisp — 2nd BG variant, heavily blurred, masked to avoid face
# ---------------------------------------------------------------------------
def apply_foreground_wisp(final_img, fg_bg_img, mask_binary, w, h, short_edge,
                          opacity, num_holes, seed, output_dir):
    """Composite a heavily-blurred 2nd BG variant on top, with face/shoulders protected."""
    # Find head position: top-center of subject mask bbox
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0:
        cx, cy = w // 2, int(h * 0.3)
    else:
        top_y = int(np.percentile(ys, 5))
        top_band = mask_binary[top_y:top_y + max(1, h // 20)]
        band_xs = np.where(top_band > 0)[1]
        cx = int(band_xs.mean()) if len(band_xs) else int(xs.mean())
        cy = top_y + short_edge // 30

    # Build FG alpha mask: start at full opacity white, subtract face-protect radial, add holes
    fg_alpha = np.ones((h, w), dtype=np.float32)

    # Face/shoulders protection: soft radial gradient, radius covers head + shoulders
    protect_r = short_edge * 0.32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 0.85) ** 2)
    protect = np.clip(1.0 - (dist / protect_r), 0.0, 1.0)
    protect = protect ** 1.5  # sharper center, softer falloff
    fg_alpha *= (1.0 - protect)

    # Random feathered holes so bg peeks through
    rng = random.Random(seed + 99)
    min_hole = int(short_edge * 0.04)
    max_hole = int(short_edge * 0.14)
    for _ in range(num_holes):
        hx = rng.randint(0, w - 1)
        hy = rng.randint(0, h - 1)
        hr = rng.randint(min_hole, max_hole)
        hd = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        hole = np.clip(1.0 - (hd / hr), 0.0, 1.0) ** 1.2
        fg_alpha *= (1.0 - hole)

    # Heavy blur on both FG image and alpha for soft "out-of-focus" feel
    blur_r = max(15, int(short_edge * 0.025))
    fg_f = np.array(fg_bg_img).astype(np.float32)
    fg_blur = cv2.GaussianBlur(fg_f, (0, 0), blur_r)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), max(5, blur_r // 2))

    # Save mask for debugging
    Image.fromarray((fg_alpha * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "4_fgwisp_mask.png"))
    fg_bg_img.save(os.path.join(output_dir, "4_fgwisp_bg.jpg"), "JPEG", quality=90)

    # Composite: final * (1 - a*opacity) + fg_blur * (a*opacity)
    a = (fg_alpha * opacity)[:, :, np.newaxis]
    final_f = np.array(final_img).astype(np.float32)
    out = final_f * (1 - a) + fg_blur * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baroque Surround — Generative Painterly Background")
    parser.add_argument("--source", required=True, help="Input photo path")
    parser.add_argument("--preset", default="baroque", help="Preset name (default: baroque)")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides preset)")
    parser.add_argument("--negative", default=None, help="Custom negative prompt")
    parser.add_argument("--artifact", default=None,
                        help="Story element in BG: wings, petals, hands, faces, chains, serpents, "
                             "butterflies, thorns, feathers, flames, flowers, skulls, ribbons, eyes, "
                             "random, none (default: none)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--auto-correct", action="store_true", help="Enable Gemini evaluation")
    parser.add_argument("--output-to", choices=["local", "gdrive", "both"], default="local")
    parser.add_argument("--local-output-dir", default=None, help="Custom local output directory")
    parser.add_argument("--list-presets", action="store_true", help="List all presets")
    parser.add_argument("--list-artifacts", action="store_true", help="List all artifacts")
    parser.add_argument("--flux-model", choices=["dev", "schnell"], default="schnell", help="Flux model: schnell (default, ~$0.003) or dev (quality, ~$0.04)")
    parser.add_argument("--use-cached-bg", action="store_true", help="Use a pre-generated BG from bg_cache/ (~$0 for BG, falls back to Flux if no match)")
    parser.add_argument("--tile-refine", type=float, default=0.0,
                        help="Tensor Art ControlNet Tile refine denoise (0.0=off, 0.15-0.35 typical). De-AI-ifies composite. Face-swap runs after if denoise>=0.2. Adds ~$0.02 Tensor + ~$0.03 fal = ~$0.05.")
    parser.add_argument("--tile-refine-prompt", default="raw photo, detailed skin texture, photographic, film grain, natural lighting, sharp focus, masterpiece, best quality",
                        help="Positive prompt for tile refine pass")
    parser.add_argument("--grain-match", type=float, default=0.5,
                        help="Grain match strength 0.0-1.0 (default 0.5). Extracts sensor grain from original and adds to final to unify 'AI-glued-on' look. Set 0 to disable.")
    parser.add_argument("--foreground-wisp", type=float, default=0.0,
                        help="Foreground wisp opacity 0.0-1.0 (default 0.0=off). Generates 2nd BG, blurs, masks to avoid face. Good: 0.3-0.5")
    parser.add_argument("--fg-holes", type=int, default=5, help="Number of feathered bg-showthrough holes in FG wisp mask (default 5)")
    parser.add_argument("--save-stack", action="store_true",
                        help="export pipeline stages as a multi-page TIFF (<finals>__stack.tif)")
    args = parser.parse_args()

    if args.list_presets:
        print(f"\n{'Preset':<20} Description")
        print("=" * 80)
        for name, p in PRESETS.items():
            desc = p["prompt"][:60] + "..." if len(p["prompt"]) > 60 else p["prompt"]
            print(f"  {name:<18} {desc}")
        print(f"\nTotal: {len(PRESETS)} presets")
        sys.exit(0)

    if args.list_artifacts:
        print(f"\n{'Artifact':<15} Description")
        print("=" * 80)
        for name, desc in ARTIFACTS.items():
            print(f"  {name:<13} {desc[:65]}")
        print(f"\nTotal: {len(ARTIFACTS)} artifacts. Use --artifact random for random selection.")
        sys.exit(0)

    source = os.path.expanduser(args.source)
    if not os.path.isfile(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Resolve preset
    if args.prompt:
        bg_base_prompt = args.prompt
        bg_negative = args.negative or "modern, digital, text, watermark"
        preset_name = "Custom"
    else:
        if args.preset not in PRESETS:
            print(f"ERROR: Unknown preset '{args.preset}'. Use --list-presets.")
            sys.exit(1)
        preset = PRESETS[args.preset]
        bg_base_prompt = preset["prompt"]
        bg_negative = args.negative or preset.get("negative", "")
        preset_name = args.preset

    # Resolve artifact
    artifact_name = "none"
    artifact_text = ""
    if args.artifact and args.artifact != "none":
        if args.artifact == "random":
            artifact_name = random.choice(list(ARTIFACTS.keys()))
        elif args.artifact in ARTIFACTS:
            artifact_name = args.artifact
        else:
            print(f"ERROR: Unknown artifact '{args.artifact}'. Use --list-artifacts.")
            sys.exit(1)
        artifact_text = ARTIFACTS[artifact_name]

    # Build full BG prompt
    bg_prompt = bg_base_prompt
    if artifact_text:
        bg_prompt += f", {artifact_text}"
    bg_prompt += ", no central focal point, scattered elements around edges"
    bg_prompt += ", NO person, NO figure, NO human face, just abstract painterly forms"

    # Derive names
    source_basename = os.path.splitext(os.path.basename(source))[0]
    path_parts = os.path.normpath(source).split(os.sep)
    model_name = "Unknown"
    for i, part in enumerate(path_parts):
        if part == "_photos" and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    timestamp = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    artifact_tag = f"_{artifact_name}" if artifact_name != "none" else ""
    folder_name = f"{model_name}_{source_basename}_{timestamp}_baroque_{preset_name}{artifact_tag}_{seed % 100:02d}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)

    if args.local_output_dir:
        output_dir = os.path.join(os.path.expanduser(args.local_output_dir), folder_name)
    else:
        output_dir = os.path.join(os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates"), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    log(output_dir, "=" * 60)
    log(output_dir, "BAROQUE SURROUND v2")
    log(output_dir, f"Source:    {source}")
    log(output_dir, f"Preset:    {preset_name}")
    log(output_dir, f"Artifact:  {artifact_name}")
    log(output_dir, f"Seed:      {seed}")
    log(output_dir, f"Prompt:    {bg_prompt[:120]}...")
    log(output_dir, "=" * 60)

    # Load original
    img_orig = Image.open(source).convert("RGB")
    img_orig.save(os.path.join(output_dir, "0_original.jpg"), "JPEG", quality=95)
    w, h = img_orig.size
    short_edge = min(w, h)
    src_f = np.array(img_orig).astype(np.float32)

    # --- Steps 1 & 2: PARALLEL — mask extraction + BG generation ---
    t0 = time.time()
    log(output_dir, "Steps 1+2: Extracting mask + generating BG (parallel)...")

    mask_result = {}
    bg_result = {}
    t_mask = threading.Thread(target=extract_mask_thread, args=(img_orig, w, h, output_dir, mask_result))
    t_mask.start()

    cached = None
    if args.use_cached_bg:
        cached = load_cached_bg(preset_name, artifact_name, w, h)
    fg_bg_result = {}
    t_fg = None
    if cached is not None:
        bg_img_cached, cached_file = cached
        bg_img_cached.save(os.path.join(output_dir, "2_bg.jpg"), "JPEG", quality=95)
        bg_result["bg"] = bg_img_cached
        log(output_dir, f"Using cached BG: {cached_file}")
        if args.foreground_wisp > 0:
            t_fg = threading.Thread(target=generate_bg_thread, args=(bg_prompt, w, h, output_dir, seed + 1, fg_bg_result, args.flux_model))
            t_fg.start()
        t_mask.join()
    else:
        if args.use_cached_bg:
            log(output_dir, "No cached BG match — falling back to Flux")
        t_bg = threading.Thread(target=generate_bg_thread, args=(bg_prompt, w, h, output_dir, seed, bg_result, args.flux_model))
        t_bg.start()
        if args.foreground_wisp > 0:
            t_fg = threading.Thread(target=generate_bg_thread, args=(bg_prompt, w, h, output_dir, seed + 1, fg_bg_result, args.flux_model))
            t_fg.start()
        t_mask.join()
        t_bg.join()
    if t_fg is not None:
        t_fg.join()
        # Re-save main BG to canonical path (threads may have collided on 2_bg.jpg)
        if "bg" in bg_result:
            bg_result["bg"].save(os.path.join(output_dir, "2_bg.jpg"), "JPEG", quality=95)

    if "error" in mask_result:
        log(output_dir, f"Mask extraction failed: {mask_result['error']}", "ERROR")
        sys.exit(1)
    if "error" in bg_result:
        log(output_dir, f"BG generation failed: {bg_result['error']}", "ERROR")
        sys.exit(1)

    mask = mask_result["mask"]
    mask_info = mask_result["mask_info"]
    bg_img = bg_result["bg"]

    # Check BG quality
    bg_quality = check_image_quality(bg_img, "BG", output_dir)
    if not bg_quality["ok"]:
        log(output_dir, "BG quality check failed — retrying with different seed", "WARN")
        bg_result2 = {}
        generate_bg_thread(bg_prompt, w, h, output_dir, seed + 1, bg_result2, args.flux_model)
        if "bg" in bg_result2:
            bg_img = bg_result2["bg"]
        else:
            log(output_dir, "BG retry also failed — proceeding with original", "WARN")

    mask_binary = (np.array(mask) > 127).astype(np.uint8)
    t_parallel = time.time() - t0
    log(output_dir, f"Steps 1+2 done ({t_parallel:.1f}s) — mask={mask_info['coverage_pct']}%")

    # --- Steps 3-6: Composite pipeline (local, fast) ---
    t0 = time.time()
    composite_stages = {} if args.save_stack else None
    final_img = composite_pipeline(src_f, bg_img, mask_binary, w, h, short_edge, output_dir, stages=composite_stages)
    t_composite = time.time() - t0
    log(output_dir, f"Steps 3-6 done ({t_composite:.1f}s)")

    # Save composite
    final_path = os.path.join(output_dir, "3_final.jpg")
    final_img.save(final_path, "JPEG", quality=95)

    # Step 7: foreground wisp (optional)
    if args.foreground_wisp > 0:
        if "bg" not in fg_bg_result:
            log(output_dir, f"FG wisp BG generation failed: {fg_bg_result.get('error', '?')} — skipping", "WARN")
        else:
            t0 = time.time()
            log(output_dir, f"Step 7: FG wisp (opacity={args.foreground_wisp}, holes={args.fg_holes})")
            final_img = apply_foreground_wisp(
                final_img, fg_bg_result["bg"], mask_binary, w, h, short_edge,
                opacity=max(0.0, min(1.0, args.foreground_wisp)),
                num_holes=args.fg_holes, seed=seed, output_dir=output_dir,
            )
            final_img.save(final_path, "JPEG", quality=95)
            log(output_dir, f"Step 7 done ({time.time()-t0:.1f}s)")

    # Step 7b: Tensor Art tile-refine (optional — de-AI-ify via ControlNet Tile)
    if args.tile_refine > 0:
        t0 = time.time()
        log(output_dir, f"Step 7b: Tensor Art tile-refine (denoise={args.tile_refine})")
        refined = tensor_tile_refine(final_img, args.tile_refine, args.tile_refine_prompt, output_dir)
        if refined is not None:
            refined.save(os.path.join(output_dir, "7b_tile_refined.jpg"), "JPEG", quality=95)
            # Face-swap original back on if denoise was high enough to drift face
            if args.tile_refine >= 0.2:
                log(output_dir, "Step 7c: Face-swap to restore identity")
                swapped = fal_face_swap(source, refined, output_dir)
                if swapped is not None:
                    refined = swapped
                    refined.save(os.path.join(output_dir, "7c_face_swapped.jpg"), "JPEG", quality=95)
                else:
                    log(output_dir, "Face-swap failed — keeping tile-refined without swap", "WARN")
            final_img = refined
            final_img.save(final_path, "JPEG", quality=95)
            log(output_dir, f"Step 7b+c done ({time.time()-t0:.1f}s)")
        else:
            log(output_dir, "Tile-refine failed — skipping", "WARN")

    # Step 8: grain match (default on)
    if args.grain_match > 0:
        t0 = time.time()
        log(output_dir, f"Step 8: grain match (strength={args.grain_match})")
        final_img = apply_grain_match(final_img, img_orig, args.grain_match, output_dir)
        final_img.save(final_path, "JPEG", quality=95)
        log(output_dir, f"Step 8 done ({time.time()-t0:.2f}s)")

    # Quality check
    quality_final = check_image_quality(final_img, "FINAL", output_dir)

    # Gemini evaluation
    eval_result = None
    if args.auto_correct:
        eval_result = evaluate_with_gemini(final_img, output_dir, original_img=img_orig)

    # Copy to finals
    local_out = args.local_output_dir or os.path.expanduser("~/.openclaw/workspace/shared/tool-outputs-intermediates")
    finals_dir = os.path.expanduser("~/.openclaw/workspace/shared/finals")
    os.makedirs(finals_dir, exist_ok=True)
    finals_name = os.path.basename(output_dir) + ".jpg"
    finals_dest = os.path.join(finals_dir, finals_name)
    final_img.save(finals_dest, "JPEG", quality=95)
    log(output_dir, f"Final: {finals_dest}")

    # --save-stack: aggregate intermediates into a multi-page TIFF
    if args.save_stack:
        try:
            from _layered_tiff import save_stack
            stage_files = [
                ("00_original",   "0_original.jpg"),
                ("01_mask",       "1_mask.png"),
                ("02_bg",         "2_bg.jpg"),
            ]
            layers = []
            for name, fname in stage_files:
                fp = os.path.join(output_dir, fname)
                if os.path.isfile(fp):
                    layers.append((name, Image.open(fp)))
            # In-memory composite sub-stages (03-06)
            substage_order = [
                ("03_lap_blend",      "lap_blend"),
                ("04_light_wrap",     "light_wrap"),
                ("05_lab_edge_match", "lab_edge_match"),
                ("06_lab_full_wash",  "lab_full_wash"),
            ]
            if composite_stages:
                for name, key in substage_order:
                    if key in composite_stages:
                        layers.append((name, composite_stages[key]))
            post_files = [
                ("07_fgwisp_bg",   "4_fgwisp_bg.jpg"),
                ("08_fgwisp_mask", "4_fgwisp_mask.png"),
                ("09_tile_refined","7b_tile_refined.jpg"),
                ("10_face_swapped","7c_face_swapped.jpg"),
                ("11_grain_layer", "8_grain_layer.jpg"),
            ]
            for name, fname in post_files:
                fp = os.path.join(output_dir, fname)
                if os.path.isfile(fp):
                    layers.append((name, Image.open(fp)))
            layers.append(("99_final", final_img))
            stack_path = os.path.join(finals_dir, os.path.basename(output_dir) + "__stack.tif")
            save_stack(stack_path, layers)
            log(output_dir, f"Stack: {stack_path} ({len(layers)} layers)")
        except Exception as e:
            log(output_dir, f"save-stack failed: {e}", "WARN")

    # Push to phone
    try:
        from notify import push_image
        src_name = os.path.splitext(os.path.basename(args.source))[0]
        art_label = f" +{artifact_name}" if artifact_name != "none" else ""
        push_image(finals_dest, title=f"Baroque {preset_name}{art_label}", body=f"{src_name}")
    except Exception as e:
        log(output_dir, f"Push failed: {e}", "WARN")

    # Copy script
    try:
        shutil.copy2(os.path.abspath(__file__), os.path.join(output_dir, "baroque-surround.py"))
    except Exception:
        pass

    # Summary
    total = t_parallel + t_composite
    score_str = f"{eval_result['score']}/10 — {eval_result.get('critique', '')}" if eval_result else "N/A"
    print(f"""
{'='*60}
  BAROQUE SURROUND v2 — DONE
{'='*60}
  Source:    {source}
  Preset:    {preset_name}
  Artifact:  {artifact_name}
  Seed:      {seed}

  Timing:
    Steps 1+2 (mask+BG parallel)  {t_parallel:>6.1f}s
    Steps 3-6 (composite local)   {t_composite:>6.1f}s
    TOTAL                         {total:>6.1f}s

  Quality:   brightness={quality_final['brightness']} contrast={quality_final['contrast']} entropy={quality_final['entropy']}
  Aesthetic:  {score_str}
  Output:    {finals_dest}
{'='*60}""")


if __name__ == "__main__":
    main()
