# OpenClaw Scripts

Photo transformation pipeline for portrait/boudoir photography. Three tools: art stylization, lighting re-imagination, and foreground depth framing.

## Environment

- **Python venv**: `~/openclaw-venv/` (numpy, Pillow, requests, anthropic)
- **API keys**: `~/sol/.env` (FAL_API_KEY, TENSOR_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)
- **Photos**: `~/.openclaw/workspace/_photos/` — subfolders per model name, each has `Processed/` and/or `Unprocessed/`
- **Shared folder** (visible from Windows): `~/.openclaw/workspace/shared/`
- **Styles**: `scripts/workflows/styles.json` — 111 art styles with prompt additions
- **Style guide**: `scripts/workflows/style-guide.json` — per-category strength recommendations
- **Favorites**: `~/.openclaw/workspace/shared/favorites/favorites.json` — liked outputs with full reconstruction commands
- **Impasto experiments**: `~/.openclaw/workspace/shared/impasto_experiments/` — shelved stroke direction research with examples

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

**All flags:** `--source`, `--lighting` (preset name), `--prompt` (custom, overrides preset), `--negative`, `--lowres-denoise` (default 0.85), `--highres-denoise` (default 0.5, lower = more faithful), `--guidance-scale` (default 2.5), `--steps` (default 28), `--seed`, `--no-hr`, `--auto-correct`, `--max-corrections`, `--output-to`, `--local-output-dir`, `--list-presets`

**20 presets:** Dramatic Rim, Spotlight, Low Key, High Key, Neon Gels, Teal & Orange, Red Drama, Golden Hour, Window Light, Overcast Soft, Candlelight, Butterfly, Split Light, Beauty Dish, Underwater Caustics, Moonlight, Neon Signs, Firelight, Laser

### `scripts/workflows/foreground-framing.py`
**Foreground depth framing.** Adds blurry foreground elements to photo edges, simulating the "shoot-through" technique at shallow depth of field (f/1.4-2.8, 35-50mm). Uses fal.ai SDXL inpainting for contextual foreground generation, then blurs + darkens + color-matches to the original.

**Usage:**
```bash
./scripts/workflows/foreground-framing.py --source photo.jpg --framing "doorframe" --auto-correct
./scripts/workflows/foreground-framing.py --source photo.jpg --framing "foliage" --coverage 0.25 --darken 0.4
./scripts/workflows/foreground-framing.py --list-presets
```

**All flags:** `--source`, `--framing` (preset name), `--prompt` (custom, overrides preset), `--negative`, `--coverage` (0.1-0.4, default 0.20), `--sides` (left-right/top-bottom/all/auto), `--blur-radius` (auto based on image size), `--darken` (0.0-1.0, default 0.55), `--irregularity` (0-1, default 0.5), `--guidance-scale` (default 9.0), `--steps` (default 30), `--seed`, `--auto-correct`, `--output-to`, `--local-output-dir`, `--list-presets`

**10 presets:** foliage, warm foliage, doorframe, curtain, dark curtain, flowers, fairy lights, metal, smoke, brick

**Steps:** 1. Create organic edge mask → 2. Inpaint foreground via SDXL (fal.ai) → 3. Heavy Gaussian blur + color match + darken + composite → 4. Gemini eval + output

**Tips:**
- Works best with wide-to-normal focal lengths (24-50mm feel). Telephoto compression doesn't suit this effect.
- `--coverage 0.15-0.25` is the sweet spot. Higher = more dramatic but risks obscuring subject.
- `--darken 0.4-0.6` keeps framing subtle. Lower = darker framing.

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
