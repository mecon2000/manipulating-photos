# UI revamp plan — unified tabbed interface

**Status:** spec, not yet implemented.
**Last updated:** 2026-04-26.
**Implements:** consolidates `/`, `/gallery`, `/candidates`, `/style-transfer`, `/pipeline`, `/tree` into one tabbed page; adds Decorate modal that lets Ronnie choose text + position + alignment + color grade + layered-TIFF save before favoriting.

## Goal

One tabbed page (`/`) replaces five separate pages. Every tab uses a shared shell (top bar, mode chip, common keyboard shortcuts). The Decorate modal opens from the Run tab whenever Ronnie wants to finalize a stylized output before favoriting.

## Tab layout

```
┌────────────────────────────────────────────────────────────────┐
│  [📋 Candidates] [🎬 Run] [🗳 Vote] [⭐ Favs] [🌳 Tree]         │
└────────────────────────────────────────────────────────────────┘
```

### 📋 Candidates

Grid of `~/.openclaw/workspace/shared/candidates-for-motion-streak/` photos. Per candidate:
- Auto MediaPipe face-quality chip: 🟢 clear / 🟡 partial / 🔴 none. Cached in `pipeline_state.json` keyed by `(filename, mtime)`.
- "has_face" toggle (overrides auto, persisted).
- 🛂 watermark warn chip if `watermark_check.py` flags it (lazy-run on first display per candidate, cached).
- Crop button → opens existing inline crop UI (calls `/api/pipeline/candidate/<f>/crop-options` + `/crop-batch`, mutates the file in place).
- Filter pills at top: "all / clear-face only / no-face only / has-watermark only / unmarked".
- Multi-select checkbox per candidate (persisted) — selection feeds the Run tab.

### 🎬 Run

Two columns on desktop, stacked on mobile.

**Left:** Style ref picker — 9-image grid from `~/.openclaw/workspace/shared/0010x0010/cleaned/`. Multi-select. Default 4 chosen. "Apply to all selected candidates" button kicks off `surreal_with_face.py` once per candidate × style.

**Right:** Job board — list of running/done jobs with progress chip. Each finished output is a thumbnail card with these buttons:
- ✨ **Decorate** (opens modal — see below) — primary action
- 👎 Bad → moves to `surreal-with-face-bad/`
- ⬇️ Download
- ⭐ Quick-fav (skips decorate, current default behaviour)

### 🗳 Vote

Sub-tabs per output pool — same good/bad/fav UX, persisted votes per pool:
1. `motion-streak` (legacy candidates)
2. `style-transfer` (legacy fofr/style-transfer outputs)
3. `surreal-with-face` (new pipeline raw outputs)
4. `decorated` (post-Decorate-modal saves)

Scoreboard panel: vote totals per style across pools.

### ⭐ Favs

Browser of `~/.openclaw/workspace/shared/favorites/`. Click a fav → side panel shows reproduction-command from the sidecar JSON, big preview, source/style links.

### 🌳 Tree

Existing `/tree` view. Mermaid graph + mindmap toggle.

## Decorate modal — full spec

Opened from a Run-tab thumbnail's ✨ Decorate button.

```
┌─ Decorate (BLD_4863 × style_6) ──────────────────────────┐
│ TEXT                                                       │
│  ○ none                                                    │
│  ○ "Soft echoes of a borrowed name." (Wordsworth)          │
│  ○ "Where the silver branches meet." (Tennyson)            │
│  ○ "I would not let you go." (Rossetti)                    │
│  ○ "What softer voice is hushed." (Shelley)                │
│  ○ "She was the dream within the dream." (Poe)             │
│  [↻ refresh suggestions]                                   │
│  ● my own:  [_______________________] [save as new]        │
│  [✏ edit selected text]                                    │
│                                                            │
│ POSITION (auto-found low-variance bboxes — 4-6 thumbs)     │
│  [thumb1] [thumb2] [thumb3] [thumb4] [thumb5]              │
│  (rendered server-side, ~0.5s each, cached per output)     │
│                                                            │
│ ALIGN  ○ auto-edge  ○ left  ○ right  ○ center              │
│                                                            │
│ COLOR GRADE                                                │
│  ● off  ○ warm-cool  ○ split  ○ wash:[teal▼]               │
│  strength [────●──────] 0.30                               │
│                                                            │
│ EXPORT                                                     │
│  ☐ Save layered TIFF stack (intermediate steps as layers)  │
│                                                            │
│ [LIVE PREVIEW — updates on any change, debounced 500ms]    │
│                                                            │
│              [Discard]  [Save & Fav]                       │
└────────────────────────────────────────────────────────────┘
```

### Behaviours

- **Text suggestions** — top-5 from `text_overlay.pick_quote_auto` (semantic NN over `literary_quotes.json`/`.npy`). Refresh button re-rolls (next-K from the same NN ranking). Edit button populates "my own" with the selected text for tweaking.
- **Position thumbnails** — server runs `text_overlay.candidate_bboxes` (already implemented), renders the chosen text at each top-K bbox at ~25% size, returns 4-6 thumbnails. Cached by (filename, text-hash) so re-clicking is instant.
- **Align** — overrides `text_overlay.overlay`'s align logic. `auto-edge` keeps existing behaviour.
- **Color grade** — passes mode + strength to `color_grade.grade()`.
- **Layered TIFF checkbox** — when ticked, the save also writes `<base>__stack.tif` via `_layered_tiff.save_stack` with all named intermediates (text-on, text-off, grade-on, grade-off, etc).
- **Live preview** — every knob change → debounced POST `/api/decorate/preview` → returns new image URL → swaps the preview img. Backend caches by knob-tuple to avoid re-rendering identical states.
- **Save & Fav** — finalizes, copies to `favorites/`, writes sidecar JSON with full reconstruction params (text + position + align + grade mode + grade strength + tiff flag + source + style), removes the Decorate item from Run-tab board.

## Backend endpoints (new)

```
GET  /api/pipeline/output/<filename>/text-suggestions?n=5
     → { suggestions: [{text, author, title}, ...], next_offset }
GET  /api/pipeline/output/<filename>/text-suggestions?n=5&offset=N
     → next-K (for refresh)

POST /api/decorate/preview
     body: { filename, text, position_idx, align, grade_mode, grade_strength }
     → { preview_url } (cached)

POST /api/decorate/save
     body: { ...same as preview..., save_tiff: bool }
     → { fav_path, tiff_path?, sidecar_path }

GET  /api/pipeline/candidate/<f>/watermark
     → { has_watermark: bool, text?, location?, model_version }
     (lazy + cached)
```

## Frontend decisions

- **No new framework.** Stays vanilla JS + fetch. Existing pages do this already.
- **Single template** `templates/index.html` with tab content as `<section>` blocks, switched by JS routing on hash (`#candidates`, `#run`, etc).
- **Shared menu partial** `templates/_modemenu.html` already exists — repurpose as the tab bar.
- **Mobile breakpoint** 700px (matches existing). Below: tabs become vertical strip.
- **Keyboard:** `1-5` switch tab. Within Decorate modal: `g`/`b`/`f` reserved for vote, `↵` saves, `Esc` discards.

## Backwards compatibility

- Old routes (`/gallery`, `/candidates`, `/pipeline`, `/style-transfer`, `/tree`) keep working via redirects to the new tabbed page.
- Existing API endpoints stay. New ones are additive.
- Existing `pipeline_state.json` schema stays.

## Implementation phases

Roughly 6-8 hours of agent work, split into:

1. **Skeleton + tab routing** (~1h) — shell, mode menu becomes tabs, hash routing, redirects.
2. **Candidates tab** (~1h) — port + add watermark chip + filter pills.
3. **Run tab** (~1h) — port from `/pipeline` Pane 2+3, replace Quick-fav with Decorate primary.
4. **Decorate modal + preview pipeline** (~2-3h) — biggest piece. Text suggestions, position thumbnails, live-preview rendering, debounce, caching.
5. **Vote tab** (~1h) — pool sub-tabs over the four output folders.
6. **Favs tab** (~30m) — browser + reproduction panel.
7. **Polish, smoke tests, commit** (~30m).

## Out of scope (this revamp)

- ComfyUI integration (separate plan).
- Pushbullet replacement.
- Multi-user / multi-session UX (single-user assumption stays).
- Full text editor inside the modal (just the my-own input + edit-selected).
- Position drag-and-drop (auto bboxes only, MVP).

## Acceptance criteria

- All 5 tabs work on desktop + mobile.
- Decorate modal: text picks change live preview within 1s, position click changes preview within 0.5s (cached), Save&Fav writes both image + sidecar JSON.
- Layered-TIFF checkbox: when ticked, saved stack opens in Photoshop with at least 5 named layers visible.
- Old routes redirect cleanly. All existing endpoints still work.
- Smoke test: full flow on 1 candidate × 2 styles → Decorate one of them with text + warm-cool grade + TIFF → verify favorite saved with all params + tiff readable.
