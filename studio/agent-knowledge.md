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
- Calligraphy can spell real words: red_calligraphy*.text (Cyrillic ok, Ruslan Display font; '/' = new line; .at=x,y pins it; .text_scale). There is NO 'words' param. compose_set now errors if a layer renders identically — never claim a change you didn't see.
- Bad source framing (signature, cut-off shoulder): `layer_compose.py recrop PIECE --custom x0,y0,x1,y1 [--boost]` regenerates source/mask/face. base.mask_boost unions BiRefNet + brightened BiRefNet + MediaPipe person mask (recovers dark skin under straps/hair). base.subject_scale <1 shrinks her, but exposes straight photo edges — prefer recrop.
- Size dials: fingerprint_stamps.scale/.count, ink_splatter.scale, print/asset .angle (tilt), print .coverage (0.4+ = a readable sheet). Braille with .text spells real words (English; '/' = new line, .at = top-left).
- Calligraphy text tilt: .angle (whole block) or .line_angles=a,b (one tilt per '/' line, stacked). Big text behind her (z=under) gets hidden by hair — use z=over above the eyes.
- Photographic look (recipe 'smears'): base.look=photo keeps her as a dark duotone photo (base.duotone='#shadow,#highlight', base.soften, base.subject_mid) over her own blurred BG (background.blur, .brightness). Kinds: haze (screen glow behind), ghost (blurred doubles: direction/offset/count; z=over at ~0.3 for double exposure), motion_smear (direction deg, length, ribs, keep_body, coverage), light_leak (prism bands: count, width, opacity ~0.3), vignette (inner, opacity).
- smears dials: motion_smear.strength and ghost.strength (1 = subtle, 4 = Ronnie's 'way stronger'); show a 1/2/3/4/6 choice card before settling.
- base.eye_feather (default 0.5): how softly the eye protection fades. Under heavy smear/ghost use 2-4 — a tight band 'punches' a hard window around her eyes.
- motion_smear arc mode: .mode=arc .pivot=x,y (centre of the body's curve, normalized) .sweep=deg (sign = direction) — the smear swings along a curved body (arched back) instead of a straight line.
- Sparks / embers / fire specks: kind=embers (count, size, trail 0-2, spread 0.05-0.8 default 0.3, chaos 0-1 = how loosely they follow the wind (default 0.45; some strays tumble their own way), saturation 0-1 = colour (default 0.12, near white — Ronnie wants embers almost black-and-white), opacity (default 0.55), inside = how far they may push into her outline (default 0.025 of short edge, feathered — embers never sit on her body); same .mode=arc .pivot .sweep as motion_smear for the general wind, else .direction). NEVER fake an element with motion_smear — it only smears HER photo and draws nothing new. The layer panel has ⚙ sliders for embers/smear/ghost/prism/haze/vignette.
- compose_add now refuses unknown kinds (e.g. 'red_calligraphy' — the kind is 'calligraphy') and errors when the new layer renders nothing visible. If it errors, fix it; never tell Ronnie it was added.
- Calligraphy text wider than the frame is shrunk to fit and clamped, so .at/.text_scale seem to do nothing — lower text_scale (~0.8) first.
- Big headline words ("burn like / there is no tomorrow"): kind=calligraphy (text with '/' between lines, font "Ruslan Display", color #d81020, text_scale ~0.7-0.85, line_angles small like -4,2, at=x,y). kind=micro_text only ever renders TINY caption lines — the smears recipe's `headline_type` layer is micro_text, so switch its kind to calligraphy when Ronnie wants a headline. Put it z=over when smear/ghost layers sit over that area, or they eat the letters.

## rim_light layer + base.palette_ref (2026-11-25, red-blue-paint recipe)
- `rim_light` (kind): coloured one-sided rim on the subject edges facing the light. params: color, opacity, direction (deg light comes FROM; 180=left, 0=right, 90=top), direction2 (optional, makes it two-sided — usually too much, reads as an outline), width (0.012-0.02), breakup (0-1 dry-brush gaps, 0.7 looks painted), lit_only. Use it instead of `light_leak` for rim light; light_leak renders a rainbow prism.
- `base.palette_ref=<image path>`: recolours her through that image's own luminance→colour gradient (main hue only), so duotone refs get the real gradient not 2 flat stops. Pair with base.subject_mid≈0.36 or it blows out to pale.
- For this recipe, `subject_cut` slices her into strips even when its note says "no cut" — disable it; `skin_overlay` with a texture description draws faint letters — disable unless text is wanted.
- `backdrop` (kind, z=under, put it at the very bottom): full-frame Flux painted BG from params.prompt, cached by prompt+`bg_seed`. bg_seed is recipe-level on purpose → every piece of a recipe shares the same BG, so a carousel reads as one series. brightness/opacity to sink it.
- `dry_brush` (kind, z=over, subject-clipped): her shadows break into solid black with ragged bristle edges + strokes eating the outline from inside. coverage (how far into midtones), roughness, edge, direction, length. Put `rim_light` ABOVE it or the brush eats the rim.
- Recipes own their layers; riffs (wildness, default 0.5) are random extra layers per piece, tagged riff=True — Ronnie likes them, keep them on. After tuning a piece, `layer_compose.py to-recipe PIECE` folds it back (old recipe kept in recipes/_history/<name>.vN.json; disabled riffs dropped, kept riffs become recipe layers) so the next pose/model starts from the tuned state. Other pieces of the recipe are never changed — they only get the recipe layers they lack, added OFF and noted "[new in recipe, off]".
- AI-version background: `scripts/workflows/bg_from_stylized.py --stylized <surreal_with_face output> --piece PIECE` cuts the AI figure (BiRefNet ∪ piece mask, dilated), LaMa-fills the hole, adds `ai_backdrop` (kind backdrop, params.image) at the bottom. Much wilder than a Flux `backdrop` — Ronnie prefers it. Turn grunge_backdrop off (it's opaque and sits above) and keep the vignette ≤0.3 or the BG goes muted again. Per photo it needs one style-imitation run (~$0.01) + ~$0.002.

## Compose — palettes, new layer kinds (2026-11-26)
- **Named palettes.** A recipe may carry `"palettes": {name: {role: "#hex"}}`; any param (or base/background value) written as `"$role"` resolves through the active one. Switch with `layer_compose.py set PIECE base.palette_name=<name>` (the piece screen shows palette chips when there are ≥2). cracked-statue ships 6: red-yellow, cobalt-orange, teal-coral, magenta-acid, ultramarine-gold, oxblood-sepia. pen-outline: neon, acid, sunset.
- **base.posterize** `"#c1,#c2,…"` (look=photo): flat colour patches by tone, dark→light; `posterize_swap` (0-1) lets some patches take the next colour — the screenprint/pop-art read.
- **background.mode "flat"** (look=photo): flat colour ground instead of her own darkened BG. `background.lift` (0-1) pushes her own BG toward white.
- **shutter_drag** — slow-shutter smear of HER (premultiplied, so her silhouette trails into the BG, her old BG doesn't): mode linear|rotate|zoom, direction, length, angle, strength, face_keep (face stays sharp), spread.
- **pen_outline** — black pen drawing from her own edges (contour + inner lines): weight, detail, jitter, breaks.
- **line_screen** — diagonal engraving hatch: where (highlights|shadows|subject|background|all), threshold, angle, period.
- **shards** — collage triangles in `colors` (comma list), coverage, size, hatch (share hatched).
- **cracks** — branching stone cracks across her (eyes protected): count, length, width, branch.
- **grunge_frame** — rough black border: width, roughness.
- **ink_smear `burst: true`** — strokes radiate from her head (the poster "explosion").
- **Riff limits.** recipe `riff_kinds` (allow-list) + `riff_note` keep riffs in the recipe's world — blurred-bw allows only photographic riffs (ghost/haze/motion/grain/scratches/vignette); stickers and drawings wrecked it.
- deconstruct drafts may write words for numbers ("long", "wide, uneven"); the renderer maps spread/length/offset/ribs words to numbers.
- **pen_outline v3 = a human ballpoint sketch**, not filtered edges: vector strokes on her silhouette + a few long inner lines + face features from the 468-pt mesh (upper lids whole, lower lids/jaw partial, brows, nostrils, lips). Strokes wobble, taper, overshoot, get redrawn (`repeat`), have `gaps`; hair skipped (MediaPipe hair mask, cached hair.png). Default blue ballpoint. Exempt from the eye guard (drawing her eyes doesn't cover them).
- **shards v2**: imperfect torn almost-triangles (tips cut off, rips, odd quads), `paint` share = thick brush-painted paper with a lit/shaded edge, `shadow` share casts a drop shadow, white torn paper core on some edges. On `over` layers shards never touch the face. paper_scraps take `shadow` (0-1) too.
- **spike_burst** (cracked-statue's red explosion): jagged spikes from behind the head, dry-brush streaks along the rays, rough core, flecks. Exempt from the margin fade. **scrape_streaks**: squeegee/scraped dry paint bands.
- **Cut-out library**: `build_asset_library.py` → `assets/library/<category>/<slug>.png` (RGBA, ~130 items: botanical, creatures, birds, animals, classical marble, anatomy, objects, celestial, architecture, ephemera, paint, sea). Use with kind `asset` + `params.library=<id>` (+ shadow, angle, gray). Riffs are offered the least-used items (usage.json) so pieces don't repeat.
- **Render cue (piece screen)**: when any layer image changed on the server, two high beeps → ~700ms → new picture + low beep. Instant changes are silent.
- **blurred-bw v2 (selective drag + depth)**: `depth_trail` (under her) = copies of her along a curved 3D path that ends behind her: shrinking with depth, defocusing, fading into the backdrop, darks dominate (end, bend, depth, defocus, haze, darks). `shutter_drag` is now selective: `hair`, `body`, `edges` weights (0-1; MediaPipe hair mask) + face_keep — blur lands only where asked, and trails leave her silhouette. `studio_arcs` = soft shaded arches in the backdrop (smeared studio gear): count 1-2 reads like the ref, 3+ looks like ripples.
- **Paint where a layer shows (🖌)**: any layer can carry `params.paint_mask = masks/<id>.png` (white = show), painted on the canvas in the piece screen (🖌 on the layer row → paint → Apply; "No mask" removes it). The compositor multiplies the layer's alpha by the softened mask — Ronnie's way to say "drag only along her left outline". Prefer suggesting it over guessing geometry.
- shutter_drag selective extras: `hands` (MediaPipe hand landmarker, cached hands.png) and a sharp zone `keep_side` (-1 left … 0 middle … +1 right of her bbox) + `keep_width` (0 = off). Naming any of hair/body/edges/hands makes the others default to 0. Drag + depth_trail steps are jittered so copies never stair-step.
