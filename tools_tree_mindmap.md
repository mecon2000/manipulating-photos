# OpenClaw Tools Tree — Org-chart View

```mermaid
graph TD
    classDef happy fill:#1a6b1a,stroke:#4f4,color:#fff,stroke-width:3px
    classDef done fill:#2d5a1e,stroke:#4a9,color:#fff
    classDef wip fill:#8a6d1b,stroke:#fc3,color:#fff
    classDef tired fill:#5a3a1e,stroke:#a86,color:#ccc
    classDef paused fill:#4a2a4a,stroke:#a6a,color:#ccc
    classDef dropped fill:#5a1e1e,stroke:#a44,color:#ccc,stroke-dasharray:5 5
    classDef idea fill:#1a1a3a,stroke:#66f,color:#99f
    classDef foundation fill:#1a3a5a,stroke:#48c,color:#fff
    classDef root fill:#222,stroke:#fff,color:#fff,stroke-width:3px

    START[START]:::root

    START --> FOUNDATION[Foundation]:::foundation
    START --> COMPOSITING[Compositing]:::done
    START --> LIGHTING[Lighting]:::done
    START --> EFFECTS[Effects]:::done
    START --> FRAMING[Framing]:::done
    START --> INACTIVE[Paused / Dropped]:::paused
    START --> RESEARCH[Research / Ideas]:::idea

    %% Foundation
    FOUNDATION --> MASK[masking]:::foundation
    FOUNDATION --> SEGMENT[body-segment]:::foundation
    FOUNDATION --> NOTIFY[notify.py]:::foundation
    FOUNDATION --> PHOTODB[photo catalog DB]:::foundation
    FOUNDATION --> CAND[find-candidates]:::foundation
    FOUNDATION --> CROP[smart-crop]:::foundation
    FOUNDATION --> BATCHRUN[batch-runner]:::happy

    %% Compositing
    COMPOSITING --> STYLIZE[stylizing-bg-model-separately]:::done
    COMPOSITING --> BAROQUE["baroque-surround v2<br/>⭐ 20+ favs"]:::happy
    COMPOSITING --> COLORBATH[color-bath]:::happy

    BAROQUE --> BGCACHE[cache-baroque-bgs]:::done
    BAROQUE --> I_REFARTIST[idea: reference artist style]:::idea
    BAROQUE --> I_HARMONIZE[idea: deep harmonization]:::idea
    BAROQUE --> I_PAINTER[idea: painterly brushstroke]:::idea
    BAROQUE --> I_SCENEMATCH[idea: auto scene-matching]:::idea
    COLORBATH --> I_SFUMATO[idea: sfumato edge]:::idea

    %% Lighting
    LIGHTING --> RELIGHT["relighting<br/>⭐ 16 favs"]:::happy
    LIGHTING --> NOIR[noir-paint 😴]:::tired

    %% Effects
    EFFECTS --> TIMECORR["time-corruption<br/>⭐ shibari ghost"]:::happy
    EFFECTS --> INKDISS["ink-dissolution<br/>⭐ 11 favs"]:::happy
    EFFECTS --> MATSWAP[material-swap]:::done
    MATSWAP --> I_CHAIN[idea: tool chaining]:::idea
    INKDISS --> I_MONO[idea: mono+accent]:::idea

    %% Framing
    FRAMING --> FGFRAME[foreground-framing 😴]:::tired
    FRAMING --> TORN[torn-reveal 😴]:::tired
    FRAMING --> GEOM[pose-geometry 😴]:::tired

    %% Inactive
    INACTIVE --> SILH[silhouette-backdrop ⏸]:::paused
    INACTIVE --> BOTAN[botanical-overlay ❌]:::dropped
    INACTIVE --> INKSPL[ink-splash ❌]:::dropped
    SILH --> I_SILH2[idea: silhouette v2]:::idea
    BOTAN --> I_BOTAN2[idea: botanical v2]:::idea

    %% Research
    RESEARCH --> I_DIFFDIFF[differential diffusion]:::idea
    RESEARCH --> I_COMFYUI[ComfyUI on fal.ai]:::idea
    RESEARCH --> I_NSFW[NSFW-safe inpainting]:::idea
    RESEARCH --> I_SCOREBOARD[style scoreboard]:::idea
```
