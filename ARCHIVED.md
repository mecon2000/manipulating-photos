# Archived / Parked / Superseded

Reference doc for tools and notes that are no longer active but might come up in questions or git archaeology. `CLAUDE.md` points here by name — read this file when any of the following are mentioned:

**Trigger names:** `silhouette-backdrop`, `style_transfer_replicate`, `tensor_photo_workflow`, `pro_photo_workflow`, `pro_photo_workflow_v2`, `pro_photo_workflow_v3`, `anya_pro_workflow`, `michaela_pro_workflow`, removed presets (`glacial-veil`, `frozen-breath`), shadow-casting in `noir-paint`.

---

## Parked tools (code present, not in active rotation)

### `scripts/workflows/silhouette-backdrop.py` ⏸ paused
**Silhouette on simple backdrop with graphic element** (moon, pedestal, sunset, etc.). Works, but the suitability filter (pose extension + clothing coverage) segfaults under parallel MediaPipe+BiRefNet load — currently not in batch-runner rotation. Usable as a single-run tool with `--force`.

### `scripts/workflows/style_transfer_replicate.py` (mostly superseded)
**Replicate `fofr/style-transfer` wrapper.** IPAdapter Plus + DreamShaperXL Lightning + depth ControlNet on Replicate (NSFW-friendly community SDXL, no Flux filter). ~$0.0063/run, ~7s. Single or batch. Less identity-preserving than `become-image`; mostly superseded by `surreal_with_face.py` for portrait work.

**Usage:**
```bash
./scripts/workflows/style_transfer_replicate.py --source PHOTO --style STYLE_REF
./scripts/workflows/style_transfer_replicate.py --batch  # every source × every style
```

**Flags:** `--source`, `--style`, `--batch`, `--source-dir`, `--style-dir`, `--prompt`, `--denoising-strength` (0-1, default 0.65), `--depth-strength` (0-1, default 1.0), `--seed`, `--out-dir`. Reads `REPLICATE_API_TOKEN` from env or `~/sol/.env`. Outputs to `shared/style-transfer-finals/` with sidecar JSON per file.

---

## Parked features inside active tools

### `noir-paint.py` — shadow casting
Implemented but parked. Procedural shadow casting doesn't generalize across poses. The pipeline step is bypassed; if revisiting, expect to redesign it generatively (e.g., relight pass with a hard key) rather than geometric.

### `baroque-surround.py` — removed presets
`glacial-veil` and `frozen-breath` were dropped — too tame visually, didn't read as dramatic backdrops. Don't reintroduce them by name without changing the prompts substantially.

---

## Legacy scripts (Echo, V9–V18 iterations)

Earlier iterations of the main stylization pipeline. Kept in the repo for reference / output-folder reproducibility but not run anymore.

- `tensor_photo_workflow.py` — V18 predecessor of `stylizing-bg-model-separately.py`. Single style, no parallelism, no quality gates, no auto-correct.
- `pro_photo_workflow_v3.py` — Fal.ai-only pipeline (no Tensor Art). rembg → inpaint BG → color grade → composite.
- `pro_photo_workflow_v2.py` / `pro_photo_workflow.py` — Earlier iterations with multi-model BG generation.
- `anya_pro_workflow.py` / `michaela_pro_workflow.py` — Model-specific variants, archived.

If a favorite's reconstruction command references one of these, the script is still on disk; favorites are reproducible. New work should use `stylizing-bg-model-separately.py` or `surreal_with_face.py`.

---

## Lessons that are now baked into code

These were real lessons from testing (April 2026), but the fixes are now in the codebase. Listed here so a future agent doesn't "re-discover" them and re-edit code that already handles them:

- **Pixel values scaled to image size** — all effects use percentage of short edge, not absolute px. Fixed 3+ times across tools.
- **Gemini JSON parsing** — `responseMimeType=application/json` is set; `maxOutputTokens=4096`; `finishReason` checked.
- **Auto blur radius capped at 60px** — prevented abstract-blob outputs on 2048+ px images.
- **Color matching reduced to 30%** at scene edges — earlier 60% washed out intended tones.
- **LAB color wash on full composite, not just subject** — implemented in baroque-surround v2.
