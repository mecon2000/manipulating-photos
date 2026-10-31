# Deconstruct → Compose (reverse-engineer a reference look, rebuild it on Ronnie's photos)

Two pipelines, one shared vocabulary (the **layer recipe**):

1. **Deconstruct** — reference image → layer recipe JSON + annotated breakdown sheet.
2. **Compose** — Ronnie's photo + recipe → layered render (+ PSD). Iterative: the agent
   raises questions as choice cards (A/B/C renders), Ronnie taps or pins comments on
   the canvas, the recipe updates, re-render.

Everything lives in the hub → Studio, so new art is made in one place.

## Folders (`~/.openclaw/workspace/shared/deconstruct/`)
| Folder | What |
|---|---|
| `refs/` | **Drop reference photos here** (any number). Each is deconstructed once. |
| `recipes/<ref>.json` | Layer recipe (schema v1) |
| `recipes/<ref>__breakdown.jpg` | Ref with numbered layer boxes + legend |
| `assets/<kind>/` | Reusable transparent elements (generated once, reused across photos) |
| `renders/<photo>__<ref>/` | Compose outputs: final, per-layer PNGs, PSD, `recipe.json` used |

## Recipe schema v1
```json
{"schema": 1, "ref": "x.jpg", "summary": "...", "palette": ["#c8102e", ...],
 "aspect": "4:5",
 "base": {"bw": true, "preset": "tri-x-punch", "contrast": 1.5, "grain": 0.05},
 "background": {"mode": "flat", "color": "#f1eee8"},
 "layers": [{"id": "red_paint", "kind": "paint_pattern", "z": "under",
             "region": "behind-head", "bbox": [x0,y0,x1,y1],
             "params": {"color": "#b5121b", "coverage": 0.3, "drips": true},
             "note": "why it is there / how it reads"}]}
```
`kind` vocabulary (one generator each): `paint_pattern` (painted wall marks + drips),
`splatter` (thick-brush splotches + dots), `scratches`, `paper_scraps` (torn B&W cut-offs),
`scribbles` (doodled objects — `params.subject`, e.g. "groups of eyes"), `ink_smear`,
`stain` (coffee / water rings + spills), `grain`.
`z`: `under` (between BG and subject) | `over` (above subject).
`region`: `behind-head | around | left | right | top | bottom | edges | anywhere`,
resolved against the subject bbox of the *target* photo (not the ref's pixels).

## Phases
1. **`deconstruct.py`** — Claude vision (subscription via `claude -p`, Gemini fallback)
   reads each ref in `refs/`, writes recipe + breakdown sheet. Idempotent (skips done refs
   unless `--force`). Ronnie confirms/edits the layer list (first ping-pong).
2. **`layer_compose.py`** — B&W presets, subject cut (BiRefNet) onto flat BG, procedural
   generators (free, seeded): splatter, drips, scratches, grain, paper scraps (fractal tear
   edges from torn-reveal), ink smears, stains. Generative elements (scribbles, paint
   patterns) via Flux on white → keyed to alpha → cached in `assets/` (generated standalone,
   never from the photo, so no NSFW block). **Cropping = existing `smart-crop.py`**
   (`--crop N` / `--custom`) run first; compose never crops on its own. Outputs layer
   PNGs + PSD (`psd_layers.write_psd`). `--variants layer.param=a,b,c` renders a choice set.
3. **Studio "compose" sessions** — hub entry: pick ref (from `refs/`) + pick photo(s)
   (catalog / candidates / favorites). Layer panel (toggle, reorder under/over, re-seed),
   **choice cards** (agent posts 2-4 renders for one question; tap = pick, logged), pinned
   canvas comments routed to the agent as region-scoped notes → recipe edits.
4. **Taste** — picks + comments feed `studio/mine_taste.py`; hub tile; agent-knowledge
   entry.

## Status
- Phase 1: DONE (2026-10-31) — `deconstruct.py`, refs folder = `shared/refs/` (whole folder = one technique). First recipe: `red-calligraffiti` (14 layers, claude sonnet, ~35s). Note: sandboxed runs need api.anthropic.com allowed.
- Phase 2: next — first target Elly `01 hotel/Processed/BLD_5611E.jpg`
