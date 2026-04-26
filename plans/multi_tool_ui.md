# Multi-tool UI plan

**Status:** spec, not implemented.
**Last updated:** 2026-04-26.
**Implements:** extends the unified tabbed UI (`plans/ui_revamp.md`) to support **all** workflow tools, not just `surreal_with_face`. Adds an **Auto** tab that revives the legacy batch-runner autonomous gallery flow.

## Why

The Phase 1-5 UI assumes one pipeline (surreal-with-face). Other workflow tools are equally important to Ronnie:
- `baroque-surround` (top tool, 20+ favs)
- `ink-dissolution`, `time-corruption`, `material-swap`, `relighting`, `color-bath`, `foreground-framing`, `noir-paint`
- Plus future tools

These need both *deliberate* invocation (pick a tool + tune knobs + hit Go) and the *autonomous* gallery flow (random tool × random preset × random photo, weighted toward favorites — what `batch-runner.py:generation_loop` already does).

## New tab layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ [📋 Candidates] [🎬 Run] [🤖 Auto] [🗳 Vote] [⭐ Favs] [🌳 Tree]   │
└─────────────────────────────────────────────────────────────────────┘
```

(Inserts **Auto** between Run and Vote. Now 6 tabs.)

## 🎬 Run tab (extended — deliberate mode)

Top of left column: **tool picker** dropdown. Default `surreal_with_face` (current behaviour). Other choices populated from the tool registry (see below). Each choice changes the right side of the left column to show that tool's controls.

Below the tool picker: standard controls per tool. The registry tells the UI:
- which preset/style multi-select widgets to show (or text inputs for raw prompts)
- which slider params (with min/max/default) to expose
- which boolean flags (checkboxes) to expose
- whether the tool needs a **style ref image** (only `surreal_with_face`, `style_transfer_replicate`, `become_image_replicate`)

**Examples by tool:**
- `surreal_with_face` — current behaviour. Style multi-select (9 cleaned 0010 refs).
- `baroque-surround` — preset multi-select (20 presets), artifact multi-select (14), strength slider, optional `--foreground-wisp` toggle.
- `ink-dissolution` — medium dropdown (5 options), strength slider.
- `time-corruption` — effect dropdown (5), mode (dissolve/float/normal), intensity slider, rope-color, arc-angle.
- `material-swap` — material multi-select (10).
- `relighting` — lighting preset multi-select (20).
- `color-bath` — color multi-select (10), strength slider, preserve-shadows toggle.
- `foreground-framing` — framing preset multi-select (10), coverage slider, sides dropdown.
- `noir-paint` — palette dropdown, num-tones, light angle slider.

Run button text adapts: "Run baroque-surround on 5 candidates × 4 presets = 20 jobs · ~$0.20".

Jobs board on right is unchanged — same chip + thumbnails + Decorate / Bad / Download / Quick-fav. Decorate works for any tool's output (text + grade are tool-agnostic post-steps).

## 🤖 Auto tab (new)

Revives the legacy autonomous gallery flow that `batch-runner.py:generation_loop` already implements but the new UI dropped. UI:

```
┌──────────────────────────────────────────┐
│  Status: 🟢 running · 12 / 15 in queue   │
│  [⏸ Pause]  [⏭ Skip current]             │
│                                           │
│  Per-tool weights:                        │
│   baroque-surround    [████████░░] 50%    │
│   ink-dissolution     [██░░░░░░░░] 15%    │
│   ...                                     │
│  [Reset to defaults]                      │
│                                           │
│  Cost today: $0.42 / cap $2.00            │
│                                           │
│  [Live feed — last 6 generated photos]    │
└──────────────────────────────────────────┘
```

Reuses the existing `STATE` (gallery state) + `generation_loop` thread. New endpoints `/api/auto/{state, pause, resume, weights}` (`pause`/`resume` already exist).

Live feed = newest-first scrolling list of recently-generated outputs, polling `/api/queue` (which already exists). Each item has Decorate / Bad / Quick-fav / Download — same as Run tab's jobs board but already-completed.

## Tool registry

A new `manipulating-photos-with-ui/tool_registry.json` (or hardcoded dict at top of `batch-runner.py`):

```json
{
  "surreal_with_face": {
    "label": "Surreal with face (0010x0010)",
    "needs_style_ref": true,
    "style_ref_dir": "shared/0010x0010/cleaned",
    "presets": null,
    "params": {
      "instant_id_strength": {"type":"float","min":0,"max":2,"default":1.0},
      "image_to_become_strength": {"type":"float","min":0,"max":1,"default":0.75},
      ...
    },
    "flags": ["--no-face-overlay"],
    "cost_estimate_usd": 0.07,
    "wall_time_estimate_sec": 240
  },
  "baroque-surround": {
    "label": "Baroque surround",
    "needs_style_ref": false,
    "presets": ["baroque","ink-water","silk", ...],
    "artifacts": ["wings","petals","butterflies", ...],
    "params": { "strength": {"type":"float","min":0,"max":1,"default":0.6} },
    "flags": ["--foreground-wisp"],
    "cost_estimate_usd": 0.01,
    "wall_time_estimate_sec": 60
  },
  ...
}
```

Backend uses this to (a) generate `/api/run/tools` for the dropdown, (b) construct subprocess commands for `/api/run/tool/<name>/run`.

## New endpoints

```
GET  /api/run/tools                     → tool registry
POST /api/run/tool/<name>/run           → body: {candidates, params, flags, style_refs?, presets?, artifacts?}
                                          spawns N subprocesses, returns job IDs
                                          (re-uses the same PIPELINE_JOBS dict as surreal_with_face)
GET  /api/auto/state                    → {running, paused, queue_size, cost_today, recent_outputs}
POST /api/auto/{pause,resume}           → toggles
POST /api/auto/weights                  → {tool_name: weight, ...}  for tuning auto picker
```

## Phasing

1. **Tool registry** — define `tool_registry.json`, expose via `GET /api/run/tools`. ~30 min.
2. **Run tab tool picker** — UI: dropdown + dynamic per-tool form. Bind to `/api/run/tool/<name>/run`. ~1h.
3. **Auto tab** — revive `generation_loop`, expose state + controls + live feed. ~1h.

Total: ~2-3h human-equivalent → ~15 min agent wall, three sequential dispatches.

## Out of scope

- Tool *parameter inference* (auto-pick best params per source). Stays manual.
- Cross-tool chaining (matswap → baroque → crop in one run). Separate feature.
- Per-user preference learning. No.
- New tools — only existing ones in `scripts/workflows/` get registry entries.

## Acceptance criteria

- Run tab: pick `baroque-surround` from dropdown → preset/artifact selectors appear → run on 1 candidate × 2 presets → 2 jobs land in board → Decorate works on the outputs.
- Auto tab: pause/resume affects the live feed (no new items when paused).
- Switching back to `surreal_with_face` in dropdown returns to current Phase-3 behaviour without breakage.
