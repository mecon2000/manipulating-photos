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
