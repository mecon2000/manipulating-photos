# OpenClaw Scripts

Photo transformation pipeline for portrait/boudoir photography. Eleven tools with unified `--affect`/`--exclude` masking.

## Environment

- **Python venv**: `~/openclaw-venv/` (numpy, Pillow, requests, anthropic, mediapipe, fal_client)
- **API keys**: `~/sol/.env` (FAL_API_KEY, TENSOR_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)
- **Photos**: `~/.openclaw/workspace/_photos/` — subfolders per model name, each has `Processed/` and/or `Unprocessed/`
- **Shared folder** (visible from Windows): `~/.openclaw/workspace/shared/`
- **Styles**: `scripts/workflows/styles.json` — 111 art styles with prompt additions
- **Style guide**: `scripts/workflows/style-guide.json` — per-category strength recommendations
- **Favorites**: `~/.openclaw/workspace/shared/favorites/favorites.json` — liked outputs with full reconstruction commands
- **Impasto experiments**: `~/.openclaw/workspace/shared/impasto_experiments/` — shelved stroke direction research with examples
- **MediaPipe models**: `~/openclaw-venv/mediapipe_models/` — selfie_multiclass.tflite, hand_landmarker.task, pose_landmarker.task

## Unified Masking (`--affect` / `--exclude`)

All tools share `scripts/workflows/masking.py` for mask building. Two masking engines, auto-selected:

**BiRefNet** (fal.ai API, ~5s, excellent edges — especially hair):
- `--affect subject` — whole person vs background
- `--affect bg` — background only (inverted subject mask)

**MediaPipe body-segment** (local, ~0.5s, 6 categories):
- `--affect skin` — face-skin + body-skin (default for shibari — ropes auto-excluded)
- `--affect face-skin` / `--affect body-skin` / `--affect hair` / `--affect clothes` / `--affect others`
- Any comma-separated combination: `--affect face-skin,body-skin,hair`

**Special:**
- `--affect all` — full image, no masking
- `--exclude hands` — subtract detected hands (MediaPipe hand landmarker)
- `--exclude ropes` — subtract ropes via HSV detection (aggressive, may eat skin — usually not needed since MediaPipe auto-classifies ropes as clothes)

**Defaults per tool:**
- time-corruption: `skin` (shibari-safe, ropes untouched)
- relighting: `subject` (IC-Light needs full subject on black)
- material-swap: `subject` (transform entire subject)
- pose-geometry: `subject` (silhouette-based effects)
- foreground-framing: not applicable (uses subject mask for avoidance, not effect application)

## Active Scripts

### `scripts/workflows/stylizing-bg-model-separately.py`
**The main workflow script.** Takes a photo, separates subject from background, stylizes each independently (in parallel via Tensor Art), composites back, optionally face-swaps, evaluates quality via Gemini Vision, and auto-corrects if needed.

**Key features:**
- Separate BG/model stylization with different styles (`--bg-style`, `--model-style`)
- Auto-switches to whole-image mode if mask is too small (<5%) or too large (>70%)
- Quality gates: brightness/contrast/entropy checks with auto-retry on failure
- SSIM validation between original and stylized BG
- Gemini Flash aesthetic evaluation with structured issue detection
- Auto-correction loop: adjusts strength, tries different styles, re-runs LaMa, skips faceswap, brightens subject, color-matches — based on judge feedback
- Output to GDrive, local shared folder, or both
- Auto-loads API keys from `~/sol/.env`

**Usage:**
```bash
./scripts/workflows/stylizing-bg-model-separately.py --source photo.jpg --style "Old Dutch Master" --bg-strength 0.35 --model-strength 0.2 --output-to local --local-output-dir ~/.openclaw/workspace/shared --auto-correct
```

**All flags:** `--source`, `--style`, `--bg-style`, `--model-style`, `--bg-strength` (default 0.6), `--model-strength` (default 0.4), `--cfg-scale`, `--separate/--no-separate`, `--up-to-step 1-7`, `--faceswap/--no-faceswap`, `--tensor-model`, `--seed`, `--max-retries`, `--auto-correct`, `--max-corrections`, `--output-to` (gdrive/local/both), `--local-output-dir`, `--list-styles`, `--bg-fill` (blur/lama, default blur), `--mask-model` (birefnet/rembg, default birefnet), `--posterize` (2-8 tone levels), `--downscale` (target long-edge px), `--stroke-angle` (degrees), `--stroke-length` (px), `--mask-expand` (px), `--prompt-extra` (appended to style prompt)

**Steps:** 1. Extract mask (fal.ai BiRefNet/rembg) → 2. Clean BG (blur fill or fal.ai LaMa) → 3. Stylize BG+Model in parallel (Tensor Art) → 4. Composite → 5. Face swap (fal.ai) → 6. Quality eval + auto-correct (Gemini 2.5 Flash) → 7. Upload

### `scripts/workflows/relighting.py`
**Lighting re-imagination.** Instead of stylizing, re-lights photos using IC-Light V2 (fal.ai). Extracts subject (BiRefNet), then applies new lighting: rim light, spotlight, colored gels, etc. Results look photographic, not painterly.

**Usage:**
```bash
./scripts/workflows/relighting.py --source photo.jpg --lighting "Dramatic Rim" --auto-correct
./scripts/workflows/relighting.py --source photo.jpg --lighting "Neon Gels" --highres-denoise 0.7
./scripts/workflows/relighting.py --list-presets
```

**All flags:** `--source`, `--lighting` (preset name), `--prompt` (custom, overrides preset), `--negative`, `--lowres-denoise` (default 0.85), `--highres-denoise` (default 0.5, lower = more faithful), `--guidance-scale` (default 2.5), `--steps` (default 28), `--seed`, `--no-hr`, `--bg-blend` (0.0-1.0, blend original BG back), `--bg-blend-blur` (mask blur px, 0=hard edge), `--auto-correct`, `--max-corrections`, `--output-to`, `--local-output-dir`, `--list-presets`

**20 presets:** Dramatic Rim, Spotlight, Low Key, High Key, Neon Gels, Teal & Orange, Red Drama, Golden Hour, Window Light, Overcast Soft, Candlelight, Butterfly, Split Light, Beauty Dish, Underwater Caustics, Moonlight, Neon Signs, Firelight, Laser

### `scripts/workflows/foreground-framing.py`
**Foreground depth framing.** Adds blurry foreground elements to photo edges, simulating the "shoot-through" technique at shallow depth of field (f/1.4-2.8, 35-50mm). Uses fal.ai SDXL inpainting for contextual foreground generation, then blurs + darkens + color-matches to the original.

**Usage:**
```bash
./scripts/workflows/foreground-framing.py --source photo.jpg --framing "doorframe" --auto-correct
./scripts/workflows/foreground-framing.py --source photo.jpg --framing "foliage" --coverage 0.25 --darken 0.4
./scripts/workflows/foreground-framing.py --list-presets
```

**All flags:** `--source`, `--framing` (preset or "auto" for Gemini scene detection, default auto), `--prompt` (custom), `--negative`, `--coverage` (0.1-0.4, default 0.20), `--sides` (left-right/top-bottom/all/auto/smart, default smart — L-shaped based on subject position), `--blur-radius` (auto, capped at 60px), `--darken` (0.0-1.0, default 0.55), `--irregularity` (0-1, default 0.5), `--guidance-scale` (default 9.0), `--steps` (default 30), `--seed`, `--auto-correct`, `--output-to`, `--local-output-dir`, `--list-presets`

**10 presets:** foliage, warm foliage, doorframe, curtain, dark curtain, flowers, fairy lights, metal, smoke, brick

**Steps:** 1. Create organic edge mask → 2. Inpaint foreground via SDXL (fal.ai) → 3. Heavy Gaussian blur + color match + darken + composite → 4. Gemini eval + output

**Tips:**
- Works best with wide-to-normal focal lengths (24-50mm feel). Telephoto compression doesn't suit this effect.
- `--coverage 0.15-0.25` is the sweet spot. Higher = more dramatic but risks obscuring subject.
- `--darken 0.4-0.6` keeps framing subtle. Lower = darker framing.

### `scripts/workflows/time-corruption.py`
**Temporal decay effects.** Simulates time corruption: ghosting (multiple exposure), motion trails, diffusion melting, chromatic aberration. Especially suited for shibari/movement photography.

**Usage:**
```bash
./scripts/workflows/time-corruption.py --source photo.jpg --effect ghost --intensity 0.7
./scripts/workflows/time-corruption.py --source photo.jpg --effect full --direction 45
./scripts/workflows/time-corruption.py --source photo.jpg --effect melt --intensity 0.8
```

**5 effects:** ghost, melt, trails, glitch, full (combines all). PIL/numpy/scipy based — no API calls for effects.

**3 modes:** `--mode dissolve` (default, ropes stay sharp, body gets effect — best for shibari), `--mode float` (subject sharp, BG dissolves — "in space" feeling), `--mode normal` (effects on full subject).

**Shibari flags:** `--rope-color` (auto/red/beige/black/white), `--arc-angle` (ghost arc curve, default 30°)

Ghost in dissolve mode uses exponential arc offsets (scaled to image size) for visible, artistic body echoes while ropes stay perfectly sharp.

### `scripts/workflows/material-swap.py`
**Material transformation.** Changes the subject's skin/body material to glass, marble, metal, etc. Uses BiRefNet for subject extraction + Tensor Art img2img for material transformation. Background stays pristine.

**Usage:**
```bash
./scripts/workflows/material-swap.py --source photo.jpg --material "wet glass" --strength 0.4
./scripts/workflows/material-swap.py --source photo.jpg --material "cracked glass" --auto-correct
./scripts/workflows/material-swap.py --list-presets
```

**10 presets:** wet glass, cracked glass, oily glass, frosted glass, marble, liquid metal, porcelain, ice, gold, obsidian

### `scripts/workflows/pose-geometry.py`
**Geometric pose art.** Extracts subject silhouette and reconstructs as geometric art, then blends back with the original. "Art gallery" aesthetic. BiRefNet + local PIL/scipy processing.

**Usage:**
```bash
./scripts/workflows/pose-geometry.py --source photo.jpg --geometry wireframe --blend-mode overlay
./scripts/workflows/pose-geometry.py --source photo.jpg --geometry lowpoly --blend-opacity 0.6
./scripts/workflows/pose-geometry.py --list-presets
```

**7 presets:** wireframe, lowpoly, crystal (edge-aware Delaunay), shatter (gradient-straddling), refine (error-minimizing iterative — best quality, adaptive sizing), blocks, contour. `--num-points` controls triangle density. **4 blend modes:** overlay, multiply, screen, alpha.

### `scripts/workflows/body-segment.py`
**Fine-grained body part segmentation.** Uses MediaPipe multiclass selfie segmentation to separate body into face-skin, body-skin, hair, clothes, and others. Can also detect and subtract hands (MediaPipe hand landmarker) and ropes (HSV thresholding). Runs locally — no API calls.

**Usage:**
```bash
./scripts/workflows/body-segment.py --source photo.jpg --include face-skin,body-skin --exclude hands
./scripts/workflows/body-segment.py --source photo.jpg --include skin --exclude hands,ropes
./scripts/workflows/body-segment.py --source photo.jpg --include hair
./scripts/workflows/body-segment.py --source photo.jpg --include all --exclude background
```

**All flags:** `--source`, `--include` (comma-separated: face-skin, body-skin, hair, clothes, others, skin, all), `--exclude` (comma-separated: hands, ropes, hair, clothes, others, background), `--rope-color` (auto/red/beige/black/white), `--feather` (edge blur % of short edge, default 0.5), `--cleanup` (close/open/smooth/none), `--debug` (save individual masks), `--bg-color` (black/white/transparent), `--output-to`, `--local-output-dir`

**Categories (from MediaPipe):** background, hair, body-skin, face-skin, clothes, others. Ropes/blindfolds/gags are auto-classified as clothes/others (not skin), so they're excluded from skin masks without needing HSV detection.

**Tips:**
- For shibari: `--include skin --exclude hands` is usually enough — ropes are already excluded by the segmenter
- HSV rope detection (`--exclude ropes`) is aggressive and may eat skin — only use when ropes are misclassified as skin
- Hand detection works well for subtracting another person's hands touching the subject
- `--debug` saves individual mask PNGs for each category

### `scripts/workflows/noir-paint.py`
**High-contrast painterly effect** inspired by @pulpbrother's gouache/acrylic style. Extracts subject, relights with harsh directional light, posterizes to 2-3 tones, vectorizes boundaries into smooth curved contours, overlays coarse canvas texture.

**Usage:**
```bash
./scripts/workflows/noir-paint.py --source photo.jpg
./scripts/workflows/noir-paint.py --source photo.jpg --tones warm --num-tones 3
./scripts/workflows/noir-paint.py --source photo.jpg --light-angle 135 --canvas-strength 0.25
./scripts/workflows/noir-paint.py --list-palettes
```

**All flags:** `--source`, `--tones` (cool/warm/cold/sepia, default cool), `--num-tones` (2/3/4, default 2), `--light-angle` (degrees, default auto from body axis), `--highres-denoise` (default 0.45), `--paint-strength` (Tensor Art img2img, default 0.18, 0=skip), `--canvas-strength` (default 0.22, 0=skip), `--edge-roughness` (default 0.5, 0=skip), `--seed`, `--auto-correct`, `--output-to`, `--local-output-dir`

**4 palettes:** cool (blue-grey, classic), warm (skin tones), cold (steel-blue), sepia (vintage)

**Pipeline:** 1. Extract subject (BiRefNet) + scene context (Gemini) → 2. Body axis detection (MediaPipe pose) → perpendicular light direction → 3. Relight (IC-Light, harsh directional) → 4. [Shadow casting — parked] → 5. Bilateral presmooth + Otsu posterize → 6. Vectorize tones (Douglas-Peucker + slight bezier curves) → 7. Edge roughening → 8. Paint texture (optional Tensor Art img2img) → 9. Coarse canvas texture overlay

**Light direction convention (XZ/XY clock):**
- XZ clock (floor plane, bird's eye): 12=behind model, 6=camera, 3=model's left, 9=model's right
- XY clock (wall behind model): 12=above, 6=below, 3=left, 9=right
- Example: xz-3 = side light from model's left, xy-10+xz-6 = above camera slightly left

**Key techniques:**
- **Bilateral presmooth** before posterizing eliminates gradient jitter at tone boundaries
- **Smooth curved contours** (cv2 + Douglas-Peucker + slight bezier bow) — decisive paint-stroke-like edges, not jagged pixels
- **Coarse canvas texture** — real burlap texture generated via Flux, seamlessly tiled with feathered overlapping + random flips
- **Gemini scene context** — detects ground surface for grounding shapes (trapezoid for floor, rectangle for bed)
- **Body axis perpendicular lighting** — MediaPipe pose landmarks determine body orientation, light placed perpendicular biased toward empty space

**Tips:**
- `--num-tones 2` gives the boldest, most graphic result (like pulpbrother)
- `--paint-strength 0` skips Tensor Art pass — faster, slightly less painterly
- Shadow casting is implemented but parked — procedural approach doesn't generalize well across poses
- IC-Light prompt direction can be unintuitive — describe what's illuminated, not light position

### `scripts/workflows/ink-dissolution.py`
**Frequency-band dissolution.** Decomposes photo into Laplacian pyramid, suppresses texture bands, overlays medium texture. Radial gradient from face: face stays sharp, dissolution increases with distance. Background fully dissolved.

**Usage:**
```bash
./scripts/workflows/ink-dissolution.py --source photo.jpg --medium ink-wash
./scripts/workflows/ink-dissolution.py --source photo.jpg --medium charcoal --dissolve-strength 0.5
./scripts/workflows/ink-dissolution.py --source photo.jpg --medium canvas --fade-distance 7.0
./scripts/workflows/ink-dissolution.py --list-media
```

**5 media:** ink-wash (user favorite), watercolor, canvas, charcoal, graphite. All local PIL/numpy/scipy + BiRefNet mask + MediaPipe body segmentation. No Tensor Art calls.

**Key flags:** `--dissolve-strength` (0-1, default 0.85), `--face-preserve` (0-1, default 0.85), `--fade-distance` (multiplier for preserve radius, default 3.5 — higher = more body preserved), `--levels` (pyramid depth, default 5), `--seed`

### `scripts/workflows/torn-reveal.py`
**Two-layer portrait composite with paper tear.** Top: color photo. Bottom: high-contrast B&W. A torn-paper strip across the eye area reveals the B&W layer beneath. Fractal midpoint displacement for realistic tear edges, cone-shaped tear, fiber burst zones, drop shadow, film grain.

**Usage:**
```bash
./scripts/workflows/torn-reveal.py --top color.jpg --bottom bw.jpg
./scripts/workflows/torn-reveal.py --top photo.jpg --bottom photo.jpg --tear-angle 15
./scripts/workflows/torn-reveal.py --top photo.jpg --bottom photo.jpg --extra-tear-y 0.72 --extra-tear-fill dark
```

**Key flags:** `--tear-height` (fraction, default 0.10), `--tear-angle` (degrees or "auto"), `--tear-jitter` (0-1, default 0.5), `--bw-contrast` (default 1.5), `--grain` (default 0.04), `--extra-tear-y` (fraction for second tear), `--extra-tear-fill` (dark/bw/black)

**Eye detection:** FaceLandmarker (iris landmarks) → body-segment face-skin fallback → center fallback. Alignment capped at ±15° rotation. Translation-only when detection methods differ.

**Best with:** Front-facing portraits where FaceLandmarker detects both faces. Adjacent file numbers (same shoot) give best alignment.

### `scripts/workflows/baroque-surround.py`
**Generative painterly background.** Extracts subject, generates a new BG from scratch via Flux text-to-image, composites with tight mask + spot bleeds + foreground overlay + LAB color matching.

**STATUS: Work in progress.** The generate-BG-from-scratch approach works (visible amorphic forms) but the composited result still looks like a cutout on many photos, especially dark-BG sources. The core unsolved problem: making subject and BG feel like they belong to the same image (same lighting, same color temperature, same painterly quality).

**What works:** ethereal and silk presets, LAB 70%+ color transfer, the relit-then-composite approach (relight original → extract → LAB match → composite onto generated BG).

**What doesn't work:** Flux inpainting on dark BGs (generates black), Flux img2img on dark photos (also black), IC-Light relighting (replaces BG entirely, doesn't relight the scene).

**Options being explored:**
1. Gradient-mask inpainting (like Photoshop generative fill) — soft radial mask instead of binary cutout
2. Tensor Art img2img instead of Flux (different model may handle dark inputs better)
3. SDXL inpainting with gradient masks
4. Whole-image img2img at moderate strength with selective blend-back (tested, Flux gives black)
5. The approach from reference artist: likely heavy PS work with Firefly generative fill + manual blending

**13 presets:** baroque, renaissance, dark-romantic, ethereal, smoke, underwater, ink-water, aurora, silk, embers, curtains, whipped-cream, bubbles

**Usage:**
```bash
./scripts/workflows/baroque-surround.py --source photo.jpg --preset ethereal
./scripts/workflows/baroque-surround.py --source photo.jpg --preset silk --method generate
./scripts/workflows/baroque-surround.py --list-presets
```

**All flags:** `--source`, `--preset`, `--prompt` (custom), `--negative`, `--strength`, `--method` (generate/inpaint), `--transition`, `--noise`, `--seed`, `--auto-correct`, `--output-to`, `--local-output-dir`, `--list-presets`

### `scripts/workflows/find-candidates.py`
**Candidate photo picker.** Scans `_photos/` directory, picks random processed photos from different models, copies them to a candidates folder with metadata manifest.

**Usage:**
```bash
./scripts/workflows/find-candidates.py                          # 5 random
./scripts/workflows/find-candidates.py --count 10 --prefer-full-body
./scripts/workflows/find-candidates.py --models "Anya,Jana"
./scripts/workflows/find-candidates.py --list-models
```

Output goes to `shared/candidates/` with a `candidates.json` manifest.

### `scripts/workflows/styles.json`
111 art styles with names and prompt additions. Loaded automatically by the stylization script. Use `--list-styles` to see all available styles.

## Lessons Learned (from testing, April 2026)

### What works well — do more of this
- **Ink dissolution ink-wash** — user's favorite new tool. Radial gradient from face, texture overlay. All favorited.
- **Baroque: generate-BG-from-scratch** — generating BG independently via Flux text-to-image, then compositing. Much better than inpainting (which generates black on dark photos). Ethereal and silk presets work best.
- **Baroque: relit-then-composite** — relight original photo to bake warm tones → extract subject → LAB match → composite onto generated BG. Best color integration so far.
- **Torn reveal** — fractal tear edges + cone shape + fiber bursts. Works best with front-facing portraits, adjacent file numbers.
- **Bioluminescent + underwater photos** — natural fit, consistently scores 9. The style matches the water context.
- **Colored gel relighting on shibari** — neon gels, pink+cyan, amber+violet. Very photographic, dramatic.
- **Ghost dissolve** (time-corruption, dissolve mode) — body echoes along arc while ropes stay sharp. Powerful for shibari.
- **BG blend in relighting** (`--bg-blend 0.4-0.5`) — keeps the original scene context, prevents "pasted on BG" look.
- **Oil Impasto with posterize+downscale** — chunky visible brush strokes. posterize 3 + downscale 768.
- **Style matching to scene context** — underwater style for pool photos, doorframe framing for indoor, etc.
- **Combining tools** — e.g., ghost dissolve → relight with window light. Layer effects for unique results.

### What went wrong repeatedly — avoid these pitfalls
- **Pixel values not scaled to image size.** Absolute pixel offsets (5px, 15px) are invisible on 2048px images. ALWAYS scale effects to percentage of image dimensions (1-5% of short edge). This was fixed 3+ times.
- **Gemini JSON parsing.** maxOutputTokens too low → truncated JSON → lost scores. responseMimeType=application/json is mandatory. Set maxOutputTokens to 4096. Check finishReason for MAX_TOKENS.
- **SDXL inpainting ignores prompts.** Asked for "doorframe" → got foliage. The model preserves surrounding context more than following the text prompt. Strength=0.95 helps but doesn't fully solve it.
- **Effects that blur-in-place are invisible.** Motion trails and melt applied within a body mask just blur the body in the same location — no visible change. Effects need to SPREAD BEYOND the mask boundary to be visible.
- **Relighting removes the original BG entirely.** IC-Light generates a new scene. Hair edges look cut. Fix: --bg-blend 0.4-0.5 with soft mask blur (~2% of image). But even then, hair can look pasted.
- **Flux inpainting/img2img on dark-BG photos → black.** Tried BG lifting, noise, blur, multiple strengths (0.5-0.93). Flux always gravitates back to black on dark inputs. Not usable for photos with dark backgrounds.
- **Binary mask compositing always looks like a cutout.** Even with feathering, bleed spots, light wrap, LAB color transfer — two separately-created elements don't feel like one image. The reference artist likely uses gradient masks + generative fill (Photoshop Firefly) for unified lighting/color.
- **LAB color transfer needs 70%+ strength.** Below 70% the shift is invisible. Apply to entire subject (stronger at edges, 40% minimum at center). RGB shift needs 50%+ to be visible.
- **Complementary color wash must be on entire composite, not just subject.** Washing only the subject is invisible. Full-image wash creates overall tonal unity.
- **Auto blur radius on large images.** 2048+ px images get blur radius 120+ which makes everything look like abstract blobs. Cap blur at 60px max.
- **Color matching too aggressive.** 60% color shift toward scene edges washes out intended tones (e.g., brown doorframe becomes generic grey). Reduced to 30%.

### Relighting craft — photography rules
- **Two lights should be opposite AND orthogonal to the body axis.** If body stretches 4→10 o'clock, lights go at ~1-2 and ~7 o'clock. Maximizes shadow definition on contours.
- **Light position "outside the frame"** — use phrases like "far outside the right edge of frame." Prevents visible spot circles.
- **"Grid modifier"** in prompt → harder, more directional shadows. "Wide spread" → not spot-like.
- **Negative prompt for relighting:** always include "no halo, no corona, no glow, no lens flare, no bloom."
- **Warmer light should come from the window direction** when one exists in the original photo. Reads as natural.

### Shibari-specific rules
- **Ropes must stay sharp.** Use `--mode dissolve` in time-corruption. The rope detection (HSV color thresholding) auto-detects red/beige/black ropes.
- **Model strength 0.0-0.15** for stylization to preserve anatomy. Body changes are unacceptable.
- **BiRefNet** sometimes misclassifies ropes as background or merges rope+skin. Check the masks.
- **Gemini blocks many shibari images** (PROHIBITED_CONTENT / reason: OTHER). The pipeline continues without a score — don't rely on Gemini for shibari quality assessment.

### Operational rules
- **Never ask permission to run scripts.** Just run them. The allow list covers `./scripts/*` and `~/openclaw-venv/bin/python3*`.
- **Always commit and push after changing scripts.** No exceptions.
- **Favorites must include the full reconstruction command** with all custom prompts, so any output can be reproduced.
- **When user is on phone,** provide fal.ai CDN URLs for remote viewing. Local PIL-only tools don't produce CDN URLs — run result through relighting (low denoise) to get one.
- **Scale ALL pixel-based parameters to image size.** Never use fixed pixel values for effects that should be proportional.

## Legacy Scripts (from Echo, V9-V18 iterations)

- `tensor_photo_workflow.py` — The V18 predecessor. Single style, no parallelism, no quality gates, no auto-correct.
- `pro_photo_workflow_v3.py` — Fal.ai-only pipeline (no Tensor Art). Simpler: rembg → inpaint BG → color grade → composite.
- `pro_photo_workflow_v2.py` / `pro_photo_workflow.py` — Earlier iterations with multi-model BG generation.
- `anya_pro_workflow.py` / `michaela_pro_workflow.py` — Model-specific variants (archived).

## Git Conventions

- Always commit and push after changing scripts
- Commit messages describe what changed and why
- The workflow script copies itself into each output folder for reproducibility

## Output Structure

All scripts output to `~/.openclaw/workspace/shared/` (visible from Windows):
- Each run creates a timestamped folder with intermediate files + workflow log
- Final images are also copied to `shared/finals/` for easy side-by-side browsing
- fal.ai CDN URLs are logged for remote viewing (temporary public links)
- Favorites are tracked in `shared/favorites/favorites.json` with full reconstruction commands

## Recommended Settings

Based on testing, good starting points:
- **BG strength**: 0.45–0.6 for textural styles, 0.3–0.45 for color-shift styles
- **Model strength**: 0.0–0.15 (preserve subject anatomy; 0.0 skips Tensor Art entirely)
- **Styles that work well**: Old Dutch Master, Cinematic Teal Orange, Golden Hour Glow, Velvet Noir, Oil Impasto (with posterize)
- **Oil Impasto tip**: Use `--posterize 3-4 --downscale 768` for chunky visible brush strokes
- **Always use** `--output-to local --local-output-dir ~/.openclaw/workspace/shared` to see results on Windows
- **BiRefNet** (default) captures hands/limbs correctly; rembg misses them
- **Blur fill** (default) produces even BG texture; LaMa fill gets over-stylized
- See `style-guide.json` for per-category strength recommendations

### Relighting
- **Dramatic Rim** and **Spotlight** work on almost any photo
- **Underwater themes** (Ocean Blue, Underwater Caustics) are magic on pool/water photos
- `--highres-denoise 0.4-0.5` keeps subject faithful; higher = more creative freedom
- Custom prompts via `--prompt` are very effective for specific lighting setups

### Foreground Framing
- **Doorframe** preset is ideal for indoor hallway/room shots
- **Curtain/dark curtain** works for window scenes
- **Foliage** works for outdoor shots
- Best on wide-to-normal lens photos (24-50mm). Telephoto doesn't suit this effect.
