#!/home/rong/openclaw-venv/bin/python3
"""
Test baroque BG generation with different story elements,
then run full pipeline (Laplacian + light wrap + LAB edge + LAB 60% wash).
"""
import os, sys, numpy as np, cv2, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for line in open(os.path.expanduser("~/sol/.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["FAL_KEY"] = os.environ.get("FAL_API_KEY", "")

import fal_client, requests
from PIL import Image, ImageFilter
from masking import build_mask
from notify import push_image
from io import BytesIO

FINALS = os.path.expanduser("~/.openclaw/workspace/shared/finals")

# --- Load source + mask (reuse across all rounds) ---
src_path = os.path.expanduser("~/.openclaw/workspace/_photos/Nastia Tsoy/Processed/BLD_5147.jpg")
img = Image.open(src_path).convert("RGB")
w, h = img.size
short_edge = min(w, h)

print("Extracting mask...")
mask, info = build_mask(img, affect="subject", exclude="", output_dir="/tmp", feather=0)
if mask.size != (w, h):
    mask = mask.resize((w, h), Image.LANCZOS)
mask_binary = (np.array(mask) > 127).astype(np.uint8)
src_f = np.array(img).astype(np.float32)
print(f"  Coverage: {info['coverage_pct']}%")


def full_pipeline(bg_pil, tag):
    """Steps 3-6: Laplacian blend + light wrap + LAB edge + LAB 60% wash."""
    bg_resized = bg_pil.resize((w, h), Image.LANCZOS) if bg_pil.size != (w, h) else bg_pil
    bg_f = np.array(bg_resized).astype(np.float32)

    # Step 3: Laplacian pyramid blend
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

    # Step 4: Light wrap
    blur_r = max(30, int(short_edge * 0.08))
    bg_blur = cv2.GaussianBlur(bg_f, (0, 0), blur_r)
    ks = max(5, int(short_edge * 0.025))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(mask_binary, kern, iterations=1)
    edge = ((dilated - mask_binary) > 0).astype(np.float32)
    edge_s = cv2.GaussianBlur(edge, (0, 0), max(3, ks // 2))[:, :, np.newaxis]
    result = result * (1 - edge_s * 0.25) + bg_blur * (edge_s * 0.25)

    # Step 5: LAB edge color match
    bg_lab = cv2.cvtColor(np.array(bg_resized), cv2.COLOR_RGB2LAB).astype(np.float32)
    res_lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ew = max(10, int(short_edge * 0.05))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ew, ew))
    eroded = cv2.erode(mask_binary, ke, iterations=1)
    inner = ((mask_binary - eroded) > 0).astype(np.float32)
    inner_s = cv2.GaussianBlur(inner, (0, 0), max(3, ew // 2))
    for ch in range(3):
        bg_near = bg_lab[:, :, ch][edge > 0.3]
        subj_edge = res_lab[:, :, ch][inner_s > 0.3]
        if len(bg_near) == 0 or len(subj_edge) == 0:
            continue
        res_lab[:, :, ch] += (bg_near.mean() - subj_edge.mean()) * 0.4 * inner_s
    result = cv2.cvtColor(np.clip(res_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Step 6: Full-image LAB 60% wash
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

    out = os.path.join(FINALS, f"baroque_elem_{tag}.jpg")
    Image.fromarray(final).save(out, quality=95)
    push_image(out, f"Baroque: {tag}", "Laplacian+wrap+LAB edge+LAB60%")
    print(f"  Final: {out}")

    # Also save raw BG for reference
    bg_out = os.path.join(FINALS, f"baroque_elem_{tag}_bg.jpg")
    bg_resized.save(bg_out, quality=95)
    return out


# --- Flux BG dimensions ---
flux_w = min(w, 1024)
flux_h = int(flux_w * h / w)
flux_h = (flux_h // 8) * 8
flux_w = (flux_w // 8) * 8

# --- 3 rounds with different story elements ---
rounds = [
    ("faces_hands",
     "surreal volumetric smoke and dark haze, tortured amorphic faces emerging "
     "from the smoke with open mouths, ghostly pale hands reaching upward through "
     "the clouds, anguished expressions dissolving into mist, warm golden and "
     "deep charcoal tones, baroque oil painting atmosphere, chiaroscuro, "
     "Caravaggio inspired, soft directional light from right"),

    ("roses_petals",
     "surreal volumetric clouds and ethereal haze, hundreds of dark red rose "
     "petals floating and swirling through the air, wilting baroque roses with "
     "thorny stems emerging from dense smoke, scattered loose petals caught in "
     "wind, warm amber and deep crimson tones, romantic dark oil painting "
     "atmosphere, soft golden light, vanitas still life inspiration"),

    ("serpents_chains",
     "surreal volumetric smoke and dark atmospheric haze, thick ornate chains "
     "hanging and draping through the smoke, a large serpent coiling through "
     "the chains with iridescent scales, dark iron links dissolving into mist, "
     "warm copper and deep teal tones, dark baroque atmosphere, dramatic "
     "chiaroscuro lighting, Caravaggio and Rubens inspired"),
]

for tag, prompt in rounds:
    print(f"\n{'='*60}")
    print(f"Round: {tag}")
    print(f"{'='*60}")

    print("  Generating BG via Flux...")
    handle = fal_client.submit("fal-ai/flux/dev", arguments={
        "prompt": prompt,
        "image_size": {"width": flux_w, "height": flux_h},
        "num_images": 1,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "output_format": "jpeg",
        "enable_safety_checker": False,
    })
    result = handle.get()
    bg_url = result["images"][0]["url"]
    resp = requests.get(bg_url, timeout=60)
    bg_img = Image.open(BytesIO(resp.content)).convert("RGB")
    print(f"  BG generated: {bg_img.size}")

    full_pipeline(bg_img, tag)

print("\n=== ALL DONE ===")
