"""
Lux - Contemporary Art Creation
Session: Ruby, Hanuka (Adi Levi / Session 64)
5 unique art pieces from shibari+candles performance
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

SRC = "/home/openclaw/gdrive/Adi Levi/64 Ruby, Hanuka/Choose from these/"
OUT_FINAL = "/home/openclaw/.openclaw/workspace/art_output/Final/"
OUT_WORK  = "/home/openclaw/.openclaw/workspace/art_output/Working/"

# Source images (chosen for art potential + permission safety)
SOURCES = {
    "body_close":   SRC + "BLD_2135.jpg",  # close-up back/skin, no face, rope+candles
    "silhouette1":  SRC + "BLD_2496.jpg",  # full silhouette extended pose
    "silhouette2":  SRC + "BLD_2094.jpg",  # inverted figure blue light
    "curved":       SRC + "BLD_2112.jpg",  # curved body, blue+gold
    "dynamic":      SRC + "BLD_2121.jpg",  # kinetic, no face
}

def save_working(img, name):
    path = OUT_WORK + name
    img.save(path, quality=90)
    print(f"  [working] {name}")
    return path

def save_final(img, name):
    path = OUT_FINAL + name
    img.save(path, quality=95)
    print(f"  [FINAL]   {name}")
    return path

def apply_curves_np(arr, in_low, in_high, gamma=1.0):
    """Simple curve adjustment: remap pixel range."""
    arr = arr.astype(np.float32)
    arr = np.clip((arr - in_low) / (in_high - in_low), 0, 1)
    arr = np.power(arr, gamma)
    return (arr * 255).astype(np.uint8)

def color_grade_np(arr, shadows, midtones, highlights):
    """Apply color grade: each is (r,g,b) multiplier for shadows/mids/highs."""
    result = arr.astype(np.float32)
    lum = (result[:,:,0]*0.2126 + result[:,:,1]*0.7152 + result[:,:,2]*0.0722) / 255.0
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (
            shadows[c] * (1-lum)**2 +
            midtones[c] * 4*lum*(1-lum)**1 +
            highlights[c] * lum**2
        )
    return np.clip(result, 0, 255).astype(np.uint8)

def add_film_grain(arr, intensity=25):
    grain = np.random.normal(0, intensity, arr.shape).astype(np.float32)
    return np.clip(arr.astype(np.float32) + grain, 0, 255).astype(np.uint8)

def vignette(arr, strength=0.7):
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    dist = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
    mask = np.clip(1 - dist * strength, 0.15, 1.0)
    result = arr.astype(np.float32) * mask[:,:,np.newaxis]
    return result.astype(np.uint8)

print("=== LUX CONTEMPORARY ART ENGINE ===\n")
print("Loading source images...\n")

imgs = {}
for name, path in SOURCES.items():
    img = Image.open(path).convert("RGB")
    imgs[name] = img
    print(f"  Loaded: {name} {img.size}")
    # Save originals to Working
    img.save(OUT_WORK + f"source_{name}.jpg", quality=90)

print()

# ─────────────────────────────────────────────
# PIECE 1: "Vessel" — Chiaroscuro / Old Master
# Close-up skin+rope+candles, painterly gold/shadow treatment
# ─────────────────────────────────────────────
print("Creating Piece 1: VESSEL (Chiaroscuro)...")
img1 = imgs["body_close"].copy()

# Boost contrast + warmth
arr = np.array(img1)

# Color grade: warm gold shadows, amber midtones, bright ivory highlights
arr = color_grade_np(arr,
    shadows   = (0.6, 0.35, 0.1),    # dark umber shadows
    midtones  = (1.3, 0.9, 0.4),     # warm amber mids
    highlights= (1.1, 1.0, 0.7)      # ivory highlights
)

# Push blacks deeper
lum_mask = np.array(Image.fromarray(arr).convert("L"))
dark_mask = (lum_mask < 60).astype(np.float32)
arr[:,:,0] = np.clip(arr[:,:,0] * (1 - dark_mask*0.4), 0, 255)
arr[:,:,1] = np.clip(arr[:,:,1] * (1 - dark_mask*0.5), 0, 255)
arr[:,:,2] = np.clip(arr[:,:,2] * (1 - dark_mask*0.6), 0, 255)

# Add glow to bright candle areas
bright_mask = (lum_mask > 180).astype(np.float32)
arr[:,:,0] = np.clip(arr[:,:,0] + bright_mask*30, 0, 255)
arr[:,:,1] = np.clip(arr[:,:,1] + bright_mask*15, 0, 255)

# Film grain + vignette
arr = add_film_grain(arr, intensity=8)
arr = vignette(arr, strength=0.5)

vessel = Image.fromarray(arr)
# Final touch: slight warm blur to soften (painterly)
vessel_soft = vessel.filter(ImageFilter.GaussianBlur(radius=0.5))
# Blend 70% sharp + 30% soft
vessel_final = Image.blend(vessel, vessel_soft, 0.3)

save_final(vessel_final, "01_Vessel.jpg")

# ─────────────────────────────────────────────
# PIECE 2: "Eclipse" — Graphic Silhouette / Print Art
# Full silhouette → stark B&W graphic, radial glow behind
# ─────────────────────────────────────────────
print("Creating Piece 2: ECLIPSE (Graphic Silhouette)...")
img2 = imgs["silhouette1"].copy()

arr = np.array(img2)

# Convert to high-contrast B&W
gray = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
gray = np.clip(gray, 0, 255).astype(np.uint8)

# Apply S-curve: crush shadows, blow highlights
gray_f = gray.astype(np.float32)/255
# Mid-tone S-curve
gray_f = np.where(gray_f < 0.5,
    2*gray_f**2,
    1 - 2*(1-gray_f)**2
)
# Push to extreme contrast
gray_f = np.power(gray_f, 0.7) 
gray_ex = (np.clip(gray_f, 0, 1) * 255).astype(np.uint8)

# Create radial background glow (deep purple to black)
h, w = gray_ex.shape
Y, X = np.ogrid[:h, :w]
cx, cy = w/2, h/3  # center-top (where candles would be)
dist = np.sqrt(((X-cx)/w)**2 + ((Y-cy)/h)**2)
glow = np.clip(1 - dist*1.5, 0, 1)

# Build color image: figure stays near-B&W, background gets purple radial
result = np.zeros((h, w, 3), dtype=np.float32)

# Background: deep blue-purple
bg_r = 0.05 + glow * 0.25
bg_g = 0.02 + glow * 0.08
bg_b = 0.08 + glow * 0.45

# Merge: where figure is dark (original), use background; where bright use white
fig_mask = gray_ex.astype(np.float32)/255
result[:,:,0] = (bg_r * (1-fig_mask) + fig_mask * 0.95) * 255
result[:,:,1] = (bg_g * (1-fig_mask) + fig_mask * 0.92) * 255
result[:,:,2] = (bg_b * (1-fig_mask) + fig_mask * 0.88) * 255

result = np.clip(result, 0, 255).astype(np.uint8)
arr_eq = add_film_grain(result, intensity=5)
eclipse = Image.fromarray(arr_eq)

save_final(eclipse, "02_Eclipse.jpg")

# ─────────────────────────────────────────────  
# PIECE 3: "Devotion" — Double Exposure / Spiritual Overlay
# Blend silhouette2 (inverted figure) + body_close at different blend modes
# ─────────────────────────────────────────────
print("Creating Piece 3: DEVOTION (Double Exposure)...")
img_a = imgs["silhouette2"].copy()
img_b = imgs["body_close"].copy()

# Resize b to match a
img_b_r = img_b.resize(img_a.size, Image.LANCZOS)

arr_a = np.array(img_a).astype(np.float32) / 255
arr_b = np.array(img_b_r).astype(np.float32) / 255

# Screen blend: result = 1 - (1-a)(1-b)  — ghostly double exposure
screened = 1 - (1 - arr_a) * (1 - arr_b)

# Desaturate slightly for ethereal feel
gray_ch = (screened[:,:,0]*0.2126 + screened[:,:,1]*0.7152 + screened[:,:,2]*0.0722)
# Mix 40% gray + 60% color
screened[:,:,0] = gray_ch * 0.4 + screened[:,:,0] * 0.6
screened[:,:,1] = gray_ch * 0.4 + screened[:,:,1] * 0.6
screened[:,:,2] = gray_ch * 0.4 + screened[:,:,2] * 0.6

# Push to slightly cooler, more spiritual palette
screened[:,:,0] *= 0.85  # reduce red
screened[:,:,2] *= 1.15  # boost blue

# Increase gamma (brighten)
screened = np.power(np.clip(screened, 0, 1), 0.8)

result = (screened * 255).astype(np.uint8)
result = add_film_grain(result, intensity=12)
result = vignette(result, strength=0.4)

devotion = Image.fromarray(result)
save_final(devotion, "03_Devotion.jpg")

# ─────────────────────────────────────────────
# PIECE 4: "Threshold" — Split Toning / Francis Bacon
# Curved body with extreme color split: teal shadows / orange highlights
# ─────────────────────────────────────────────
print("Creating Piece 4: THRESHOLD (Split Toning)...")
img4 = imgs["curved"].copy()

arr = np.array(img4).astype(np.float32)

# Compute luminance
lum = (arr[:,:,0]*0.2126 + arr[:,:,1]*0.7152 + arr[:,:,2]*0.0722) / 255
lum3 = lum[:,:,np.newaxis]

# Shadow color: teal (#003344)  
sh_r, sh_g, sh_b = 0.0, 0.2, 0.27
# Highlight color: deep gold (#CC8800)
hi_r, hi_g, hi_b = 0.8, 0.53, 0.0

# Shadow mask (low lum), highlight mask (high lum)
sh_mask = np.clip(1 - lum*2.5, 0, 1)[:,:,np.newaxis]
hi_mask = np.clip(lum*2.0 - 0.8, 0, 1)[:,:,np.newaxis]
mid_mask = 1 - sh_mask - hi_mask + sh_mask*hi_mask  # what's left = midtones

# Start from desaturated base
gray = (arr[:,:,0]*0.2126 + arr[:,:,1]*0.7152 + arr[:,:,2]*0.0722)[:,:,np.newaxis]
base = np.concatenate([gray, gray, gray], axis=2)

# Apply split toning
toned = base.copy()
toned[:,:,0] = (base[:,:,0] * (sh_mask[:,:,0]*sh_r + hi_mask[:,:,0]*hi_r + mid_mask[:,:,0]*0.4))
toned[:,:,1] = (base[:,:,1] * (sh_mask[:,:,0]*sh_g + hi_mask[:,:,0]*hi_g + mid_mask[:,:,0]*0.4))
toned[:,:,2] = (base[:,:,2] * (sh_mask[:,:,0]*sh_b + hi_mask[:,:,0]*hi_b + mid_mask[:,:,0]*0.5))

# Blend 60% toned + 40% original for depth
orig_norm = arr / 255.0
result_f = toned/255.0 * 0.65 + orig_norm * 0.35
result_f = np.clip(result_f, 0, 1)

# Extra contrast push
result_f = np.where(result_f < 0.5, 2*result_f**2, 1-2*(1-result_f)**2)

result = (result_f * 255).astype(np.uint8)
result = add_film_grain(result, intensity=18)
result = vignette(result, strength=0.6)

threshold = Image.fromarray(result)
save_final(threshold, "04_Threshold.jpg")

# ─────────────────────────────────────────────
# PIECE 5: "Suspended Light" — Abstract / Negative Space
# Take dynamic pose, isolate to near-silhouette, 
# then create dramatic abstract color field with figure as anchor
# ─────────────────────────────────────────────
print("Creating Piece 5: SUSPENDED LIGHT (Abstract Color Field)...")
img5 = imgs["dynamic"].copy()

arr = np.array(img5).astype(np.float32)

# 1) Extract near-silhouette of figure by darkening background and isolating form
gray = (arr[:,:,0]*0.2126 + arr[:,:,1]*0.7152 + arr[:,:,2]*0.0722)

# 2) Build abstract color gradient background (horizontal sweep)
h, w = gray.shape
# Gradient: deep crimson left → near-black center → deep indigo right
X = np.linspace(0, 1, w)
Y = np.linspace(0, 1, h)
Xm, Ym = np.meshgrid(X, Y)

# Horizontal gradient
grad_r = (0.5*(1-Xm) + 0.05*Xm) * 255
grad_g = (0.02 + Ym * 0.05) * 255
grad_b = (0.05*(1-Xm) + 0.5*Xm) * 255
gradient = np.stack([grad_r, grad_g, grad_b], axis=2)

# Add radial light burst at top-center (candle position)
cx, cy = w*0.5, h*0.05
dist_from_light = np.sqrt(((np.arange(w)-cx)/w)**2 + ((np.arange(h)[:,np.newaxis]-cy)/h)**2)
light_falloff = np.clip(1.0/(1 + dist_from_light*4), 0, 1)
gradient[:,:,0] = np.clip(gradient[:,:,0] + light_falloff*80, 0, 255)
gradient[:,:,1] = np.clip(gradient[:,:,1] + light_falloff*50, 0, 255)
gradient[:,:,2] = np.clip(gradient[:,:,2] + light_falloff*20, 0, 255)

# 3) Figure: where dark in original, show gradient; where figure (mid-bright), show as luminous form
fig_lum = gray / 255.0
# Isolate figure: areas with lum between 0.2 and 0.85
fig_mask = np.clip((fig_lum - 0.15) / 0.5, 0, 1)  # ramp up from shadows

# Original figure contribution (stylized)
orig_styled = arr.copy()
orig_styled[:,:,0] = np.clip(arr[:,:,0] * 1.2, 0, 255)
orig_styled[:,:,1] = np.clip(arr[:,:,1] * 0.8, 0, 255) 
orig_styled[:,:,2] = np.clip(arr[:,:,2] * 1.4, 0, 255)

# Composite: gradient background + luminous figure overlay
result_f = gradient.astype(np.float32)
for c in range(3):
    result_f[:,:,c] = (
        gradient[:,:,c] * (1 - fig_mask * 0.7) +
        orig_styled[:,:,c] * fig_mask * 0.7
    )

result = np.clip(result_f, 0, 255).astype(np.uint8)
result = add_film_grain(result, intensity=10)
result = vignette(result, strength=0.4)

suspended = Image.fromarray(result)
save_final(suspended, "05_Suspended_Light.jpg")

print("\n=== ALL 5 PIECES COMPLETE ===")
print(f"Final pieces: {OUT_FINAL}")
print(f"Working files: {OUT_WORK}")
