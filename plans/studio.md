# Studio — conversational photo tweaking (plan)

**Status:** approved spec, ready to implement.
**Written:** 2026-08-07, by Claude (Cowork session with Ronnie), after reading BOTH repos: `manipulating-photos` (CLAUDE.md, tool_registry.json, plans/, commit history) and `project-hub` (CLAUDE.md, README architecture, manifests.py, app.py surface).
**Supersedes:** `plans/ui_revamp.md` and `plans/multi_tool_ui.md` conceptually (both predate the hub migration). Do not implement those.
**Target repos:** `~/gitrep/project-hub` (UI + services) and `~/gitrep/manipulating-photos` (tools). Follow existing conventions: commit+push after every change, never ask permission to run scripts, scale pixel params to image size.

## Why (the problem)

Today's loop is: random candidates → pre-made pipeline → thumbs up/down. When an output is *almost* great ("grain is awesome but not on her hand", "text should sit in the empty space left of her"), fixing it means editing a pipeline script via Claude Code and re-running everything. Slow, generic, per-photo tweaks are painful. Ronnie wants a buddy: a chat + canvas in one place, where he and the LLM look at the same photo, he points and talks, the LLM suggests and executes, iterations take seconds not minutes, and a good look can be saved as a recipe and stamped onto more photos (then tweaked per-photo with small deltas).

## 0. Before writing any code

1. Read `manipulating-photos/CLAUDE.md` fully — especially "Lessons Learned", shibari rules, relighting craft, and the NSFW routing facts (Flux/Gemini block NSFW; Tensor Art `Z-Image-Uncensored` accepts it). These become part of the agent's brain (§3.6). Read `project-hub/CLAUDE.md` + README too — its non-negotiables bind this plan (see below).
2. **Hub facts this plan is built on** (verify still true): Flask on 127.0.0.1:8700 behind `tailscale serve`; vanilla-JS frontend; manifest-driven generic engine (`hub-project.yaml` glob + `projects/*.yaml`), `actions_from_registry` already converts `tool_registry.json` into hub actions; subprocess-per-job runner + APScheduler; self-hosted ntfy :8093; **"generic engine only — zero project-specific code in the hub, ever"** with `hub/ext/` as the single sanctioned exception; ≤400-line modules; no symlinks; delete = move to `trash/`; all file serving through `safepath.py`; **runtime state must live on ext4** — `shared/` is a drvfs/9p mount where SQLite WAL corrupts and small-file churn is slow; `_photos/` is a READ-ONLY mount; photo catalog: `~/gitrep/photo-catalogging/data/photo-catalog.db` (read-only SQLite, 130k photos — the NSFW/consent brain).
3. **Because of the hub's generic-engine rule, Studio is NOT hub code.** It's a standalone service (§2.0) that the hub merely links to. Do not add a second `hub/ext/` module; if a "link tile" concept doesn't exist in the manifest schema yet, add it as a *generic* manifest feature (e.g. `links: {studio: {url, label}}`) — generic mechanism, project-specific content.
4. Confirm environment: WSL2 + 3070 Ti (8GB VRAM) GPU access from the venv (`nvidia-smi`), Ollama installed or installable (`ollama_install.sh` exists in the tools repo).
5. Put this file at `manipulating-photos/plans/studio.md`, keep it updated as phases land (status per phase).

## 1. Product spec — the Studio experience

### 1.1 Entry points
- Existing candidates flow stays: N random consent-filtered photos, "order more randoms" button.
- Every photo/output card anywhere in the hub gets a **"Tweak"** action → opens a Studio session for it.
- Studio sessions are **parallel**: several photos open at once, a swipeable tab strip (mobile) / tab bar (desktop) to move between them. Slow steps run in the background; ntfy pings on completion; a badge on the session tab shows "ready".

### 1.2 The Studio screen (mobile-first — design for phone width first, desktop is the enhancement)
- **Canvas** (top on mobile, left on desktop):
  - Pinch/scroll zoom + pan.
  - **Mask tools:** brush/lasso paint; tap-to-segment (SAM 2, §2.4); existing semantic masks (`--affect` categories: subject/bg/skin/hair/clothes…) selectable as chips. Masks render as translucent overlay, toggleable, invertible, combinable (union/subtract).
  - **Numbered markers:** tap to drop `#1`, `#2`… draggable, removable. Chat can reference them ("the text should appear at #1"). Markers are sent to the agent as normalized coordinates + a VLM description of what's under them.
  - **Crop mode** (heavily used — first-class): drag a crop rectangle on the canvas, or tap through suggestion chips from the existing `smart-crop.py` (12 subject-aware crops, `--outpaint` canvas extension, `--auto-align` straighten — already implemented, just wire it in as a step; the hub's queue-Crop montage picker is prior art). Crop is a normal graph step: undoable, recipe-able, and it invalidates downstream steps like any other change.
  - Variant compare: when a step has multiple variants, swipe/arrow between them on the canvas; long-press for side-by-side.
- **Chat** (below canvas on mobile, right on desktop):
  - Powered by the studio agent (§2.2). **Short responses.** Suggestive, opinionated about aesthetics — composition, color harmony, subject/background contrast, where the eye travels. Never a wall of text; one idea + optionally "want me to try it?".
- **Params panel** (collapsible, between canvas and chat):
  - Dynamic per current step/tool (§3.3): sliders / inputs / toggles. Seeded from `tool_registry.json` schemas.
  - **Hidden drawer**: "more params…" opens the full list with a search bar. Hiding/showing is learned from usage (§3.3) but always manually overridable.
- **Steps strip** (collapsible, hidden by default — Ronnie doesn't always use it):
  - Horizontal list of steps applied to this photo; a step with multiple variants shows as a stack — expand vertically to see `[step2.v1, step2.v2, step2.v3]`, tap to choose the active variant. Tapping any step shows its intermediate output on the canvas (crucial for "why did this come out weird").
  - Per-step actions: redo (same params/new seed), redo-from-here, edit params, delete.
- **Top bar actions:** Undo/redo (full history, §3.2) · **Lock** (finalize, §3.4) · ⭐ Fav · "Save as recipe" (§3.5) · a small **daily-cost chip** in a corner (today's API spend across all Studio sessions, same spirit as the hub's cost display; fed by the tool server's per-call accounting).

### 1.3 Recipes
- Every Studio session **auto-saves** its step graph — reopening a photo restores everything.
- "Save as recipe": the agent pre-marks each step/param as **general** (style, strengths, presets) vs **photo-specific** (masks, marker positions, crop boxes, per-photo negative prompts); Ronnie confirms/flips via checkboxes; photo-specific parts are excluded (or kept as "re-derive per photo" placeholders — e.g. "mask: skin" re-derives, "mask: brush stroke at x,y" is dropped).
- Agent suggests a recipe name; Ronnie can rename.
- **Recipes view**: grid of recipes with a sample output thumbnail, rename/delete, and "Apply to…" → same session's N photos / same model's N photos / N random candidate photos (consent-filtered). N is configurable, default 4.
- **Batch apply → contact-sheet review:** results appear as a grid; tapping one opens a Studio session pre-loaded with the recipe steps, where a short note ("less contrast", "dreamier") becomes a **delta** stored on top of the recipe for that photo. When the same delta recurs across ≥half the batch, the agent proposes folding it into the base recipe.

## 2. Architecture

Six pieces. Reuse hub infrastructure where it exists (ntfy, tailscale serve, batch jobs via hub actions), but Studio itself is its own app.

### 2.0 Placement & ports
- **Studio app** lives in `manipulating-photos` as a new top-level `studio/` package (it's inseparable from the tools + registry; the hub stays generic). Own Flask/FastAPI process on **127.0.0.1:8701**, exposed with `tailscale serve --set-path /studio` (same tailnet URL, phone-friendly). Follow hub repo hygiene even here: ≤400-line modules, safepath-style boundary checks on every file route, trash-not-rm, no symlinks.
- **Hub integration is thin:** a generic link tile from the photo-tools manifest (§0.3) pointing at `/studio`, plus "Tweak" deep-links from hub galleries (`/studio/new?src=<path>` — a generic per-area `link_template` manifest field). Batch recipe-apply (Phase 5) reuses the hub's existing job runner via a normal registry/manifest action, so batches show in the hub Jobs tab like everything else.
- **Storage split (9p rule):** session graphs, agent transcripts, param-usage stats, taste profile, and the step cache live on **ext4** at `manipulating-photos/studio/state/` (gitignored), mirroring the hub's `state/` pattern. Only human-facing media exports cross to `shared/` (drvfs): locked finals → `shared/finals/`, favorites → `shared/favorites/`, recipes → `shared/studio/recipes/` (small JSON, Windows-visible on purpose). Source photos are read-only — never write into `_photos/`.
- **Ports:** studio :8701, tool server :8702, SAM 2 :8703, Ollama :11434 (default), ntfy :8093 (existing).

### 2.1 Tool server (speed backbone) — `manipulating-photos`
The single biggest latency win. Today every run is a cold subprocess (venv import, model init) running a whole pipeline. Build a small **FastAPI service** (in the tools repo, runs in `~/openclaw-venv`) that imports the workflow modules **once** and exposes them as **steps**:
- `POST /step/run` — `{tool, step, params, input_ref, mask_ref?, seed, preview: bool}` → output ref. Refactor incrementally: start by wrapping whole tools (subprocess parity), then split the high-traffic ones (`stylizing-bg-model-separately` already has `--up-to-step`; masking.py, color_grade, text_overlay, time-corruption are natural step libraries).
- `POST /mask/build` — semantic masks via existing `masking.py` (BiRefNet/MediaPipe) + combine ops.
- **Preview mode:** `preview: true` downscales input to 1024px long-edge before the step. ALL Studio iteration runs in preview mode. Full-res only via Lock (§3.4).
- **Content-addressed cache:** output ref = hash(tool, step, params, input_ref, mask_ref, seed). Re-running a mid-graph step only recomputes invalidated descendants — and only when actually viewed/needed (lazy).
- Cost accounting per call (reuse `_pipeline_accrue`).

### 2.2 Studio agent — Claude Agent SDK
One agent session per Studio session (persisted transcript so reopening resumes the conversation). Use the **Python Claude Agent SDK** inside `~/openclaw-venv` (custom tools + streaming beat the shell-out approach; `hub/llm.py`'s env-scrubbed `claude` CLI pattern is the fallback if SDK streaming misbehaves).
**Billing/auth: Ronnie has NO API billing — the SDK must run on his Claude subscription.** This is supported: SDK usage draws from the subscription's usage limits (per support.claude.com article 15036540; the planned separate credit pool was paused). Mechanics: the SDK rides the Claude Code CLI's stored subscription login (or a `claude setup-token` long-lived OAuth token). **Ensure `ANTHROPIC_API_KEY` is scrubbed from the studio agent's environment** (same env-scrub trick as `hub/llm.py`) so the SDK can't silently fall back to pay-per-token API billing if a key ever lands in `~/sol/.env`. Note the tradeoff: Studio chat shares the plan's usage limits with Ronnie's interactive Claude use — if limits are hit, iterations pause until reset; the agent should degrade gracefully (queue the request, say so briefly) rather than erroring. Tools:
- `run_step / rerun_step / run_variants(n)` → tool server. `run_variants` fans out n seeds in parallel for cheap steps (§3.1).
- `build_mask(spec)` — semantic (`skin`, `bg`…), SAM point/box (from taps/markers), brush (from canvas), boolean combos.
- `look(image_ref | region | marker)` → eyes service (§2.4). Returns description/critique. The agent calls this after every render so it always knows what the current image looks like.
- `get_graph / set_active_variant / undo` — session graph ops.
- `save_recipe / apply_recipe / update_recipe` — §1.3.
- `propose_params(step)` — returns param schema additions for the panel (§3.3).
- `get_taste_profile` — §3.6.
- `upscale(ref, scale)` — existing `upscale_replicate.py`.
- `fav(ref)` — writes to `shared/favorites/` with full reconstruction data (existing format).
The chat UI streams the agent's text; tool calls surface as compact status chips ("relighting… 12s"), not prose.

### 2.3 SAM 2 microservice (tap-to-mask)
FastAPI + `facebook/sam2` (**sam2.1-hiera-small**, fits ~2GB VRAM), model loaded once, endpoints: `/embed` (per-image, cached), `/segment` (points/box → mask PNG + score). Sub-second after embed. Complements — doesn't replace — BiRefNet/MediaPipe semantic masks: SAM is for "this thing right here" (a prop, her hand, one rope span, the thing she's holding).

### 2.4 Eyes service (VLM routing)
One internal endpoint `describe(image, question?, region?)` that routes by a per-image NSFW level (from `photo-catalogging/data/photo-catalog.db` — photo-level explicit tags + set-level + LR keywords already exist there — with a cheap local classifier fallback for fresh outputs):
- **Local VLM via Ollama** (e.g. `qwen2.5-vl:7b` q4 — fits 8GB) → default for anything explicit; never leaves the machine; never refuses.
- **Gemini Flash** → SFW-only judging (existing integration; it blocks shibari — documented).
- **Claude (agent's own vision)** → attach pixels directly when content level permits; best reasoning about composition.
The agent never needs to know which backend answered. **8GB VRAM note:** SAM 2 small (~2GB) + Qwen-VL 7B q4 (~5-6GB) is tight; configure Ollama `keep_alive` short and load/unload gracefully, or use `sam2.1-hiera-tiny`. Verify empirically in Phase 3.

### 2.5 Frontend (Studio app, hub-adjacent)
- Vanilla JS, same idioms as the hub's `web/js/` modules (the hub is vanilla; don't introduce a framework).
- Canvas: **Konva.js** (battle-tested, mobile touch/pinch, layers, draggable shapes — covers zoom/pan/brush/markers/overlays out of the box). Chat: plain streaming (SSE) + markdown-lite rendering.
- Session state (graph, params panel state, markers) lives server-side; the client is thin — refreshing a phone mid-session must lose nothing.

## 3. Key behaviors

### 3.1 Variants (2D intermediates)
**Only for stochastic steps** — ones whose output varies run-to-run: generative model calls (Tensor/fal/Replicate) and seed-dependent local effects. Deterministic local steps (same params → same pixels) never fan out; the content-addressed cache would just return the identical output N times. Mark each step `deterministic: true/false` in the tool-server step metadata (derived from the registry / presence of a seed param).
For stochastic steps whose marginal cost is small (default threshold: ≤ $0.03 or ≤ 15s — configurable), the agent defaults to `run_variants(3)` with different seeds and says "pick one". The graph stores them as sibling variants of the same step: `[step1, [step2.v1..v3], step3…]`. Downstream steps hang off the *active* variant; switching active variant lazily recomputes descendants (cache makes revisits free). Seed-sensitivity is real (documented for anime hands) — variants are the cheap fix.

### 3.2 History / undo
The graph is append-only; undo/redo moves a pointer. Nothing is destroyed until an explicit "clear session" (which trashes, never rms). Session graph JSON + cached step artifacts live on **ext4** under `studio/state/sessions/<session-id>/` (§2.0 — the 9p mount would corrupt SQLite and crawl under cache churn); anything Ronnie should see from Windows Explorer gets exported to `shared/`.

### 3.3 Dynamic params panel
- Seed: registry param schemas for the current tool/step.
- The agent can **propose** new panel controls when it thinks a knob is worth direct manipulation (warmth, brightness, edge spikiness, its own creativity/temperature) — `propose_params` maps them to real step params or micro-steps. **Agent-proposed params carry a visible "✨ new" badge** so Ronnie knows the buddy added them (badge clears after first use or explicit dismiss).
- **Usage learning:** every panel interaction is logged (`studio/state/params_usage.json`, ext4). A param untouched in its last ~10 appearances auto-hides into the drawer; the drawer (with search) can pin anything back. Show a subtle "n hidden" hint so nothing feels lost.

### 3.4 Draft ladder + Lock
- All iteration on 1024px previews via fast/turbo endpoints where a choice exists.
- **Lock** offers two paths: **(default) Upscale this exact result** — Real-ESRGAN 2×/4× on the chosen draft, look preserved; **(option) Re-render at full res** — same graph, same seeds, full-res source, with an explicit "generative steps may come out different" warning. **Lock = fav.** The locked output lands in `shared/favorites/` with a `favorites.json` entry + full reconstruction sidecar (existing convention) — favorites is THE important folder (IG posting feeds from it), `finals/` is incidental (tools may still drop copies there per their pinned behavior).

### 3.5 Recipes & deltas
Recipe = ordered steps + general params + re-derivable mask specs, stored `shared/studio/recipes/<slug>.json` (schema versioned, includes a sample thumbnail ref). Per-photo deltas from contact-sheet review are stored with the batch result, layered over the recipe at apply time. Existing `favorites.json` reconstruction commands are the spiritual ancestor — provide a one-time importer that converts a favorite into a starter recipe.

### 3.6 The buddy's brain
**All Studio sessions share one "project brain"** (like chats in a shared claude.ai project): the same system prompt, the same knowledge docs, the same taste profile, plus a **cross-session journal** (`studio/state/journal.md`) the agent appends notable events/insights to ("whole batch wanted less contrast", "Ronnie hates warm cast on skin in the red-fever set") — so any session, including a fresh one on a new photo, knows what happened in the others. Per-photo *transcripts* stay separate (context stays small); the shared brain is what makes them feel like one continuous collaboration.
System prompt assembled from: the distilled "Lessons Learned" + shibari rules + relighting craft from CLAUDE.md (keep a `studio/agent-knowledge.md` the agent re-reads, so tool lessons keep accruing); **photography fundamentals — the full craft, not a shortlist**: composition (thirds, leading lines, negative space, balance/visual weight, framing), tonal range and subject/background contrast, color theory and harmony, eye-path and focal hierarchy, gestalt/figure-ground, texture and pattern, cropping and aspect, lighting quality and direction (the examples Ronnie gave — composition, colors, contrast — were examples, not the list); and a **taste profile** (`studio/state/taste.json`, ext4 — regenerable runtime data) periodically regenerated by an agent job that mines favorites + votes + accumulated deltas ("consistently favors muted grades, grain, sharp ropes; dislikes warm casts on skin; crops tighter than suggested"). Response style: concise, concrete, proposes 1-3 directions with draft previews rather than describing them.

## 4. Phases (each independently shippable; commit+push per phase)

**Phase 1 — Tool server + step graph (backbone).** FastAPI service wrapping registry tools as steps with preview mode, content-addressed cache, session graph JSON. CLI smoke test: run a 3-step graph on a photo twice — second run is ~100% cache hits; a mid-step param change recomputes only descendants. *No UI yet.*

> **STATUS: DONE (2026-08-07).** `studio/` package: `server.py` (FastAPI :8702 — /tools, /session/*, /step/run, /mask/build, /object/{ref}, /costs/today), `runner.py` (subprocess-wrapped tools, 1024px preview downscale, stdout `Final:`-dialect parsing + newest-file fallback, NOTIFY_DISABLE=1), `cache.py` (content-addressed objects + step records under `studio/state/cache/`), `graph.py` (append-only nodes, undo/redo pointer, sibling-branch edit), `registry.py` (registry→argv, hub-compatible param mapping; deterministic set = color-bath/time-corruption/ink-dissolution), `costs.py` (day ledger), `masks.py` (in-process masking.build_mask). Smoke test (`python3 -m studio.smoke_test`) passed: cold 14s → warm 3/3 hits 0.0s → step-2 edit recomputes only steps 2-3. Notes: `surreal_with_face` excluded (needs --relit + style ref; no finals copy — wire in when step-splitting lands); preview runs still copy into `shared/finals/` (tools have no skip flag — known wart, revisit as an additive env-var check in the tools); stochastic tools get a random seed assigned at node creation so re-roll vs cache-hit semantics are explicit. §0 verified: GPU is a 3060 Ti (8GB, ~5GB in use at idle — Phase 3 VRAM budget even tighter than planned), Ollama 0.32 installed but not running, repo on ext4, shared on 9p as documented.

**Phase 2 — Studio screen MVP.** Standalone studio app (:8701, `tailscale serve --set-path /studio`, hub link tile + gallery "Tweak" deep-links per §2.0): canvas (Konva: zoom/pan/markers/brush) + chat wired to a minimal agent (run_step, build_mask semantic+brush, get_graph, undo) + static params panel from registry + steps strip. Acceptance: on the phone over Tailscale, open a candidate from the hub → "make it moodier" → agent runs color-bath preview in <10s → brush-mask her hand → "grain everywhere except the mask" → done in <10s → undo works → phone refresh mid-session loses nothing.

> **STATUS: DONE (2026-08-08), pending phone walkthrough.** `studio/app.py` (FastAPI :8701 — pages + api: session/step/eval/edit/undo/redo/brush-mask/ui/chat NDJSON stream/chat-history/object/costs), `studio/agent.py` (Python Claude Agent SDK on the **subscription** — ANTHROPIC_API_KEY scrubbed at process level since options.env *merges* over the parent env; in-process MCP tools run_step/build_mask/get_graph/undo; sonnet; graph summary injected into system prompt each round), `studio/web/` (vanilla JS + vendored Konva 9: canvas zoom/pan/brush/markers with explicit mode toggle, NDJSON chat, params panel from registry schema, collapsible steps strip, dark mobile-first). Verified end-to-end: chat "dissolving ink-wash feel, keep her face intact" → agent ran ink-dissolution(ink-wash) in 4s preview; "remind me what we've done + undo last step" → correct recap + undo executed. Tailnet live at `https://desktop-ddrctuq.tail4fbebb.ts.net/studio/` (tailscale serve **strips** the /studio mount path — the app detects proxied requests via X-Forwarded-Host and prefixes URLs). Hub: generic `links:` + per-area `link_template:` manifest fields (routes.py passthrough + home-card chips + lightbox button), photo-tools manifest wires Studio + Tweak on finals/favorites/anime/candidates/edit-later. **Continuity note:** SDK-side resume doesn't survive per-request clients (in-memory session store) — Studio keeps its own `chat.jsonl` per session and replays the last 24 messages; real transcript resume stays Phase 6. **Manual steps for Ronnie (permission-gated here):** `cp studio/systemd/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now studio.service studio-tools.service`, and `systemctl --user restart project-hub` to pick up the hub frontend changes. Until then a background uvicorn from this session is serving :8701.

**Phase 3 — Local perception.** SAM 2 service (tap-to-mask into canvas) + Ollama VLM + eyes routing; agent auto-`look`s after each render; markers get VLM labels. Acceptance: tap her hand → clean mask <1s (post-embed); on an explicit photo, "what would you improve?" gets a specific, non-refused answer citing what's actually in the frame; VRAM stays within 8GB (measure and record the model-size choices that fit).

> **STATUS: DONE (2026-08-08).** `studio/sam_service.py` (:8703, `sam2.1-hiera-tiny` via HF from_pretrained, per-ref embedding LRU) — cold segment 2.9s, warm tap **20ms**, well under the <1s bar. WSL gotcha: torch 2.12 cu13 wheels throw `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` → `torch.backends.cudnn.enabled = False` (native kernels: embed 0.23s). `studio/eyes.py`: describe() routed by the catalog's `boldness` tag (photo→set fallback; renamed candidates usually miss → treated unknown → local); local = Ollama **gemma3:4b** (already on disk, vision-capable, ~10s/look), Gemini 2.5 Flash only for catalog-confirmed 'safe'. **VRAM measured:** desktop idles ~4.5GB of 8GB; gemma3:4b loaded → 7.5GB total; SAM tiny ~1GB — the -small SAM the plan suggested does NOT fit alongside, tiny does; Ollama keep_alive 3m keeps the overlap window short. Agent: `look` tool + auto-look after every run_step (result text includes what the output actually looks like); verified on a boudoir frame — specific, non-refused critique naming actual frame content. App endpoints: `/api/session/{id}/point-mask` (SAM proxy against current canvas), `/api/session/{id}/describe` (marker labels / ad-hoc looks). Frontend: Select mode (tap-to-mask overlay, +/− points) + marker VLM labels. Torch bump 2.11→2.12.1 (sam2 dep): mediapipe + sentence_transformers still import clean. sam2 + claude-agent-sdk added to requirements.txt; `studio-sam.service` unit added (manual install as Phase 2).

**Phase 4 — Variants + dynamic params + Lock.** run_variants fan-out with canvas compare; params usage-learning + drawer + search; agent-proposed params; Lock (upscale default / re-render option) → finals + fav. Acceptance: a cheap step auto-yields 3 variants; an unused param auto-hides and is findable in the drawer; Lock produces a 4× final matching the chosen draft.

> **STATUS: DONE (2026-08-08).** Backend: `/api/session/{id}/variants` (parallel fan-out, ThreadPoolExecutor, shared prefix materialized first; **deterministic tools refused** — identical outputs, per §3.1), `/head` (set_active_variant / arbitrary head jump), `graph.variant_group()` (same parent+tool+params, differing seed), session API exposes `variants` + sibling outputs; `params_usage.py` (appear/touch counters → `hidden` hints on /api/tools, ≥10 untouched appearances hides); `lock.py` (upscale via upscale_replicate.py subprocess / rerender as preview=False branch → shared/favorites + finals + favorites.json entry with chain reconstruction — **shutil.copyfile only; copy2's utime and copy's chmod both EPERM on the 9p mount**); agent tools run_variants (auto-looks at each variant and reports differences), propose_params (validated against registry, ✨ entries in ui.json), lock. Frontend: variant stacks in steps strip + canvas ‹ › flip + swipe, params drawer with search + usage logging + pin-back, ✨ badges with dismiss, Lock dialog (upscale 2×/4× default / rerender with drift warning). Fixed en route: scratch-dir name collision under parallel runs (uuid suffix). Verified: 3-seed fan-out grouped correctly, head switch, hidden-param hint flips, lock 2× → favorites entry `tool: studio` with full chain.

**Phase 5 — Recipes + batch + contact sheet.** Save-as-recipe with general/specific marking UI, recipes view (rename/apply-to), batch apply through the hub's existing subprocess job runner (a normal manifest action, so batches appear in the hub Jobs tab + ntfy like everything else), contact-sheet review with per-photo delta chat, fold-into-recipe suggestions, favorites→recipe importer. Acceptance: build a look on one photo → save recipe → apply to 5 photos of another model → tweak 2 via short notes → agent proposes folding a recurring delta.

**Phase 6 — Parallelism + polish.** Multi-session tab strip, background-job badges + ntfy, taste-profile miner job, agent-knowledge doc, transcript persistence/resume, mobile polish pass (thumb-reach, gesture conflicts between pan/brush — use an explicit mode toggle).

## 5. Non-goals (now)
- Replacing the random-candidates/auto-gen flow (stays; Studio is the "Tweak" path out of it).
- Multi-user; public exposure beyond the existing Tailscale setup.
- Video tools in Studio (stills_to_video/parallax stay registry-only for now).
- Training/fine-tuning anything.

## 6. Risks & notes
- **NSFW routing is load-bearing.** Encode the documented service matrix (Flux/Gemini/Replicate-SDXL-filter block; Tensor `Z-Image-Uncensored` accepts) as data the agent checks *before* dispatching a step, so it routes or warns instead of burning a call. Local steps are always safe.
- **8GB VRAM** is the tightest constraint (§2.4) — resolve empirically in Phase 3, prefer smaller checkpoints over swapping.
- **Preview→full-res drift** for generative steps is inherent; that's why Lock defaults to upscaling the draft.
- **Cost visibility, not policing:** no budgets or nagging — just the daily-cost chip in the UI corner (§1.2), aggregated from the tool server's per-call accounting (reuse `_pipeline_accrue` semantics / the auto-gen daily-cost pattern). The agent may still consider cost when choosing between equivalent routes, silently.
- Keep `tool_registry.json` the single source of truth for tool schemas — the tool server, the params panel, AND the hub's existing `actions_from_registry` converter all read it; no duplication, and don't break the hub's converter when extending the schema (additive fields only).
- **Hub non-negotiables apply across the board:** no project-specific code in the hub repo; studio runtime state on ext4, never on the 9p `shared/` mount; `_photos/` is read-only; safepath-style boundary checks on every studio file route; delete = trash.
