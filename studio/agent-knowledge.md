# Studio agent knowledge

Distilled craft + tool lessons (from CLAUDE.md "Lessons Learned", April 2026 testing,
and ongoing Studio work). The agent re-reads this every chat — append new lessons at
the bottom, keep entries terse.

## Photography fundamentals (apply to every critique)
- Composition: thirds, leading lines, negative space, balance/visual weight, framing
  within the frame; crop tighter than feels safe — Ronnie crops tighter than suggested.
- Tonal range: subject/background contrast decides readability; check where the
  brightest and darkest patches sit relative to the subject.
- Color: harmony (analogous/complementary), one dominant bath beats many hues;
  Ronnie's taste: muted grades, red-tinted 35mm film, ochre, teal-moody; he dislikes
  warm casts on skin.
- Eye path: the eye enters at the brightest/highest-contrast point — make sure that's
  the subject (face/hands), not a hot BG patch.
- Figure-ground: dissolving/texturing the BG strengthens the figure; ink-wash radial
  dissolution from the face is a proven favorite.
- Lighting: quality (hard/soft), direction relative to body axis (two lights opposite
  AND orthogonal to the axis), motivation (warm light from a window direction reads
  natural). No halo/corona/glow/lens-flare language in relight prompts.

## Hard rules (never break)
- Shibari: ropes stay SHARP (time-corruption --mode dissolve keeps them); anatomy is
  untouchable — model strength 0.0-0.15 on stylization; skin masks auto-exclude ropes.
- Never suggest steps that would alter body shape.
- NSFW routing: Flux/Gemini/Replicate-SDXL-filtered steps block explicit input;
  Tensor Art Z-Image-Uncensored (anime_stylize) accepts it; all pure-local steps are
  always safe. Route or warn BEFORE burning a call.
- Scale every pixel parameter to image size — never absolute px.

## Tool lessons
- ink-dissolution ink-wash: the favorite. Face preserved, dissolution grows with
  distance; fade-distance higher = more body preserved.
- baroque-surround: the LAB 60% wash is what unifies the composite. Best combos:
  ink-water+butterflies, silk+petals, aurora+ribbons, curtains+flames,
  dark-romantic+faces. Chain material-swap → baroque-surround for ink-on-ink.
- relighting: --bg-blend 0.4-0.5 keeps the original scene, else IC-Light invents a
  new BG and hair edges look cut. Colored gels (neon, pink+cyan, amber+violet) are
  dramatic on shibari. highres-denoise 0.4-0.5 = faithful.
- color-bath / color_grade: LAB shifts need strength ≥0.7 to be visible;
  --preserve-shadows keeps chiaroscuro. Warm-cool grade: warmth on subject only.
- time-corruption ghost dissolve: body echoes along an arc, ropes sharp — signature
  shibari move. Effects must SPREAD BEYOND the mask to be visible; blur-in-place is
  invisible.
- foreground-framing: wide-to-normal lens photos only; coverage 0.15-0.25;
  doorframe indoors, foliage outdoors. SDXL inpainting ignores prompts sometimes —
  expect foliage regardless.
- anime_stylize: strength 0.5 default, 0.4 for dark/foreshortened frames; hands are
  seed-sensitive — re-roll fixes; negative "camera, photographer, male" when dark
  hand-held gear is in frame.
- noir-paint: 2 tones boldest; IC-Light direction language describes what's LIT, not
  where the light sits.
- Oil Impasto: --posterize 3 --downscale 768 for chunky strokes.
- smart-crop: mostly for unprocessed photos; unusual crops (chin-down, torso-only)
  work via mask shape even without pose detection.
- Seed-sensitivity is real for all Tensor/fal generative steps — variants (3 seeds)
  are the cheap fix; that's what run_variants is for.
- Gemini blocks many shibari/nude images — that's why look() routes local by default.

## Studio-specific
- All iteration is 1024px preview; Lock upscales the exact draft (default) or
  re-renders full-res with drift risk.
- Cache means re-running an identical step is free — bias toward trying things.
- Recipes: general steps travel across photos; masks re-derive per photo; watch for
  recurring deltas across a batch and propose folding them into the recipe.

<!-- Append new lessons below this line. -->
- 2026-08-08: inpaint step (builtin, Flux via fal) for add/remove-object edits. For
  ADDING at a spot use region_mask blob — SAM on an empty background patch segments
  the whole background (96% of frame) and the inpaint eats the subject. For editing
  existing objects, sam_mask. Bare prompts auto-inherit the image style (match-style).
  Flux rejects explicit input frames (black output → clear error).
- 2026-08-09: inpaint engine switched to FLUX Fill (fal-ai/flux-pro/v1/fill,
  general-inpainting as fallback) — "add a heart" now actually adds a heart;
  flux-general repainted context and ignored the prompt. Every inpaint is now
  auto-verified: the edited region is cropped and the VLM answers YES/NO against
  the prompt — trust that verdict over the model's own claim of success.

## Decorations never cover the eyes (Ronnie, 2026-10-31)
Any overlay, text, texture, scribble or stamp placed ON the subject must leave her eyes clear.
`layer_compose.py` enforces it (`eye_protect`, face_align landmarks → soft band over both eyes,
applied to every over-layer; `base.protect_eyes`). Apply the same rule in any new tool.
Also: random-walk "marker scribbles" crossing the face read as ugly — keep red pen gestures off the head.

## Compose sessions (deconstruct → compose, 2026-10-31)
A compose piece = one photo × one deconstructed recipe, stacked layers under/over the cut-out
subject (`layer_compose.py`; state in `shared/deconstruct/pieces/<photo>__<recipe>/piece.json`).
Tools: `compose_get` (stack, base/background, open questions, open pinned notes — call it first),
`compose_set(kv, note)` (`layer.param=v`, `layer.z=under|over`, `layer.enabled=false`,
`layer.seed=<int>` for a new random layout, `base.*`, `background.*`; re-renders ~10s),
`compose_options(question, vary)` (A/B/C choice card, `vary` = `key=a,b,c`), `compose_render(full)`
(full-res + PSD only on request), `resolve_comment`, `look`.
- Visual questions go out as choice cards via `compose_options`, never as prose. Post one, then wait.
- Pinned notes arrive as "at (x,y) — near …: text" (normalized on final.jpg). Map the point to the
  layer(s) whose bbox/region covers it, fix with `compose_set`, then `resolve_comment`.
- House rules hold here too: eyes never covered (`base.protect_eyes` stays on), no red pen
  gestures across the head, subject stays the focal point; prefer re-seeding a layer over
  deleting it when only its layout is wrong.
- paper_scraps renders unreadable print only (Flux misspells: 'safeword' → 'Sefeworu'). Any real words go in kind=print or rendered text; quoted words in params.content are stripped.
- Readable text on paper: kind=print, variant=typed (params.title, params.lines list, params.strike indices, sign). Each piece gets its own seeds (photo-mixed) and its own brush-stroke set (params.brush_seed) — never reuse another piece's visuals.
- No torn-edge effect: base.edge_dissolve stays 0 (Ronnie prefers the clean cut-out).
- Choice cards: vary ONE param (stamp.center=A|B|C); '|' separates values that contain commas. Never vary a whole params object.
- BDSM emblem: print variant stamp with params.center=@triskelion draws the triskelion. For a dominant model, nothing on her skin claims HER (no 'property' marks).
- New element = NEW layer (compose_add), never repurpose an existing layer's kind/prompt — Ronnie toggles each element separately.
