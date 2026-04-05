# OpenClaw Scripts

Photo stylization pipeline for fine-art transformations of portrait/boudoir photography.

## Environment

- **Python venv**: `~/openclaw-venv/` (numpy, Pillow, requests, anthropic)
- **API keys**: `~/sol/.env` (FAL_API_KEY, TENSOR_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)
- **Photos**: `~/.openclaw/workspace/_photos/` — subfolders per model name, each has `Processed/` and/or `Unprocessed/`
- **Shared folder** (visible from Windows): `~/.openclaw/workspace/shared/`
- **Styles**: `scripts/workflows/styles.json` — 100 art styles with prompt additions

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

**All flags:** `--source`, `--style`, `--bg-style`, `--model-style`, `--bg-strength` (default 0.6), `--model-strength` (default 0.4), `--cfg-scale`, `--separate/--no-separate`, `--up-to-step 1-7`, `--faceswap/--no-faceswap`, `--tensor-model`, `--seed`, `--max-retries`, `--auto-correct`, `--max-corrections`, `--output-to` (gdrive/local/both), `--local-output-dir`, `--list-styles`

**Steps:** 1. Extract mask (fal.ai rembg) → 2. Clean BG (fal.ai LaMa) → 3. Stylize BG+Model in parallel (Tensor Art) → 4. Composite → 5. Face swap (fal.ai) → 6. Quality eval + auto-correct (Gemini) → 7. Upload

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
100 art styles with names and prompt additions. Loaded automatically by the main workflow script. Use `--list-styles` to see all available styles.

## Legacy Scripts (from Echo, V9-V18 iterations)

- `tensor_photo_workflow.py` — The V18 predecessor. Single style, no parallelism, no quality gates, no auto-correct.
- `pro_photo_workflow_v3.py` — Fal.ai-only pipeline (no Tensor Art). Simpler: rembg → inpaint BG → color grade → composite.
- `pro_photo_workflow_v2.py` / `pro_photo_workflow.py` — Earlier iterations with multi-model BG generation.
- `anya_pro_workflow.py` / `michaela_pro_workflow.py` — Model-specific variants (archived).

## Git Conventions

- Always commit and push after changing scripts
- Commit messages describe what changed and why
- The workflow script copies itself into each output folder for reproducibility

## Recommended Settings

Based on testing, good starting points:
- **BG strength**: 0.3–0.4 (higher = more artistic but risks losing the scene)
- **Model strength**: 0.15–0.25 (higher = risks distorting the subject)
- **Styles that work well**: Old Dutch Master, Cinematic Teal Orange, Golden Hour Glow, Velvet Noir
- **Always use** `--output-to local --local-output-dir ~/.openclaw/workspace/shared` to see results on Windows
