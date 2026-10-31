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
| `../refs/<technique>/` | **Drop reference photos here** — one subfolder per technique (`shared/refs/red-calligraffiti/`). Each subfolder → one recipe. |
| `recipes/<ref>.json` | Layer recipe (schema v1) |
| `recipes/<ref>__breakdown.jpg` | Ref with numbered layer boxes + legend |
| `assets/<kind>/` | Reusable transparent elements (generated once, reused across photos) |
| `models/<Model>.md` | Notes on the model that ground the riffs (Elly: smokes, USSR ephemera, dreads) |
| `pieces/<photo>__<recipe>/` | `piece.json` state, `final.jpg`, `layers/`, `piece.psd` (--full), `questions/qN_{A,B,C}.jpg + qN_sheet.jpg`, **`history/NNN.jpg + NNN.json + history.jsonl`** — every render, question and pick, kept for the making-of reel |

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

## Riffs + the wildness slider
`riff.py` (called by `layer_compose.py init --wildness`) adds layers ADJACENT to the technique,
grounded in the photo + `models/<Model>.md` notes, grouped into 1-3 themes, plus a controlled
number of unrelated wildcards. 0 = pure imitation · 0.3 one tight theme · 0.5 two themes + 1
wildcard · 0.8 freer media · 1.0 surprising. Extra kinds for riffs: `glyph_rain`, `asset`.

## Making-of reel (later)
history/ holds every render (numbered) + the piece state + the question/pick that caused it;
layers/ of the final give a build-up sequence. Reel = build-up of the final, then the iteration
trail (choice sheets flash, the picked option holds).

## Status
- Phase 1: DONE (2026-10-31) — `deconstruct.py`, refs folder = `shared/refs/` (whole folder = one technique). First recipe: `red-calligraffiti` (14 layers, claude sonnet, ~35s). Note: sandboxed runs need api.anthropic.com allowed.
- Phase 2: DONE (2026-10-31) — `layer_compose.py` (init/render/set/options/pick), `wall_layers.py`
  (generators), `riff.py`. ~9s preview render after first-run asset generation. First piece:
  `pieces/BLD_5611E__red-calligraffiti` (Elly). Knobs: `base.subject_mid` (subject midtone),
  `base.edge_dissolve`, `base.subject_scale` + `subject_dx` (shrink/reposition her on the paper —
  valid, unused so far), any `layer.param`.
- Phase 3: **STATUS: DONE (2026-10-31).** Studio `/compose` (pieces grid + New piece: recipe
  with breakdown thumb × photo from candidates/favorites/finals or pasted path × "imitate ↔ go
  wild" slider → `init` in a background thread, activity + ntfy) and `/compose/p/<piece>` (canvas
  on final.jpg, layer panel top→bottom with on/off · under/over · 🎲 re-seed · riff badge · note on
  tap, choice cards with tap-to-enlarge + pick + comment, pinned notes, history, chat). Files:
  `studio/compose.py` (CLI wrapper, per-piece lock, comments.jsonl append-only, thumbs on ext4),
  `studio/compose_routes.py`, `studio/compose_agent.py` (tools compose_get/set/options/render +
  resolve_comment/look/journal; transcript at `studio/state/compose/<piece>/chat.jsonl`),
  `web/templates/compose*.html`, `web/static/compose*.{js,css}` (reuses canvas.js + chat.js).
  Pins: VLM region label ("near her left cheek") runs in the background (cold Ollama ~60s) and the
  agent waits ≤25s for it. Hub: `links.compose` chip in hub-project.yaml. Verified via curl on a
  spare port: set (toggle + restore), options → q3 → pick, pin → agent compose_get +
  resolve_comment, background init. Needs `systemctl --user restart studio` to go live on :8701.
  **Round 2 (2026-10-31):** hands-on + fast — the browser composites `layers/manifest.json`
  (per-layer blend), so on/off, drag-reorder (subject = drag-through divider) and 🎲 variant
  chips redraw instantly; edits persist via `--no-render` and `studio/compose_sync.py` debounces
  a background render (2s idle; "syncing" dot). Savepoints (start first, ★ favs) with wipe /
  side-by-side compare + restore; "vs start" one tap; **Fav ★** = `fav` in the background (PSD
  path + size shown). Files: `compose_sync.py`, `web/static/compose_{layers,panel,side}.js`,
  `canvas.js` (`setImageSource`). Client composite vs final.jpg: mean diff 4.8/255.
