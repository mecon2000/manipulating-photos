# OpenClaw Scripts

Photo transformation pipeline for portrait/boudoir photography. Runs as a tabbed web UI (Flask + cloudflared tunnel) with multiple workflow tools, an autonomous gallery, and a Decorate modal that adds literary text + color grading.

- Tool map / status: [`tools_tree.md`](tools_tree.md)
- Tool reference: [`CLAUDE.md`](CLAUDE.md)
- Plans: [`plans/`](plans/)

## Quick start (fresh machine)

1. **Clone:** `git clone https://github.com/mecon2000/manipulating-photos.git && cd manipulating-photos`
2. **Run the bootstrap:** `./scripts/setup.sh` — creates the venv, installs deps, downloads MediaPipe models, scaffolds folders, builds the literary quote DB.
3. **Fill API keys** in `~/sol/.env` (template written by setup if missing). Required: `FAL_API_KEY`, `REPLICATE_API_TOKEN`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`. Optional: `PUSHBULLET_TOKEN` (image-share via S3 only — Pushbullet's free tier no longer supports push notifications).
4. **Drop photos** under `~/.openclaw/workspace/_photos/<model-name>/...` — any subfolder layout works (`Processed/`, `<session>/processed/`, `bts/`, etc — tools walk recursively). Typically a symlink to your GDrive library: `ln -s "/mnt/g/My Drive/photos/Anya" ~/.openclaw/workspace/_photos/Anya`.
5. **Drop style references** under `~/.openclaw/workspace/style-refs/<family>/` (e.g. `0010x0010/`, future `pulpbrother/`). The Run tab auto-discovers families on each load.
6. **Start it:** `./manipulating-photos-with-ui/start-gallery.sh` (with cloudflared tunnel) or `./manipulating-photos-with-ui/batch-runner.py --no-tunnel --port 5555` for local-only.

Open `http://localhost:5555` (or the tunnel URL pushed to your phone). 6 tabs: Candidates / Run / Auto / Vote / Favs / Tree.

## What `setup.sh` does

- Creates `~/openclaw-venv/` and installs from `requirements.txt`.
- Creates `~/.openclaw/workspace/{shared,_photos}/` with all the subfolders the tools expect (`finals/`, `favorites/`, `surreal-with-face/`, `tool-outputs-intermediates/`, `style-refs/`, `data/`, etc).
- Downloads MediaPipe models to `~/openclaw-venv/mediapipe_models/` (pose / hand / face / selfie_multiclass).
- Writes a template `~/sol/.env` if absent.
- Symlinks the legacy `0010x0010/cleaned/` folder into the new `style-refs/0010x0010/` location if it exists.
- Runs `build_quote_db.py` once to produce the 10k literary line embeddings used by `text_overlay.py --text auto`.
- Warns if `cloudflared` (for the public tunnel) is missing.

The script is idempotent — re-run any time without harm.

## What the bootstrap can't do for you

These bits are external and must be supplied manually:

- **`~/.openclaw/workspace/shared/data/photo-catalog.db`** — SQLite catalog built externally from your Lightroom library. ~13MB. Tools that pick photos by metadata (`find-candidates.py`, the Auto worker) need it. Manual workflows (`--source <path>`) work fine without it.
- **API keys** — fill in `~/sol/.env` with real tokens.
- **Photos** — copy or symlink your library (often a GDrive mount) to `~/.openclaw/workspace/_photos/<model>/...`. Layout under `<model>/` is flexible.
- **Style references** — drop cleaned reference images into `~/.openclaw/workspace/shared/style-refs/<family>/`.

## API key sources

| Var | Where to get it |
|---|---|
| `FAL_API_KEY` | https://fal.ai/dashboard/keys |
| `REPLICATE_API_TOKEN` | https://replicate.com/account/api-tokens |
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey |
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| `PUSHBULLET_TOKEN` | https://www.pushbullet.com/#settings/account |
| `TENSOR_API_KEY` | https://tensor.art/ (legacy, optional) |

## Repo layout

- `scripts/workflows/` — workflow tools (relighting, baroque-surround, surreal_with_face, ink-dissolution, time-corruption, material-swap, color_grade, text_overlay, etc.)
- `scripts/setup.sh` — bootstrap script (above).
- `manipulating-photos-with-ui/` — Flask app + templates for the tabbed UI.
- `plans/` — design docs for major features (UI revamp, multi-tool UI).
- `tools_tree.md` / `tools_tree_mindmap.md` — the live tool map (graph + org-chart views, viewable in the UI Tree tab).
- `CLAUDE.md` — full per-tool reference and lessons learned.

## Cost ceiling

The Auto tab tracks today's accrued spend. Default cap = $2/day. Past that, the gallery generator pauses automatically. Override per-session via the Auto tab UI.
