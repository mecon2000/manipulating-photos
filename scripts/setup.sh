#!/usr/bin/env bash
# OpenClaw setup — one-shot bootstrap to bring a fresh machine to a state
# where `manipulating-photos-with-ui/start-gallery.sh` boots and works.
#
# Idempotent: re-running is safe; it skips steps that are already done.
#
# Run from the repo root:  ./scripts/setup.sh

set -e
cd "$(dirname "$0")/.."

REPO="$(pwd)"
VENV="$HOME/openclaw-venv"
ENV_FILE="$HOME/sol/.env"
SHARED="$HOME/.openclaw/workspace/shared"
PHOTOS="$HOME/.openclaw/workspace/_photos"
MP_MODELS="$VENV/mediapipe_models"
QUOTES_NPY="$REPO/scripts/workflows/literary_quotes.npy"

echo "============================================================"
echo "  OpenClaw setup — repo: $REPO"
echo "============================================================"

# 1. Python check ---------------------------------------------------------
if ! command -v python3 >/dev/null; then
  echo "❌ python3 not found. Install Python 3.12+ first."
  exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ python3 $PYV"

# 2. Virtualenv -----------------------------------------------------------
if [ ! -x "$VENV/bin/python3" ]; then
  echo "→ creating venv at $VENV"
  python3 -m venv "$VENV"
fi
echo "✅ venv at $VENV"

# 3. Python deps ----------------------------------------------------------
echo "→ installing/updating Python deps from requirements.txt"
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install -r "$REPO/requirements.txt" | tail -5

# 4. Workspace folders ----------------------------------------------------
mkdir -p "$SHARED"/{finals,favorites,candidates-for-motion-streak,style-refs,style-transfer-finals,surreal-with-face,surreal-with-face-bad,decorate_previews,tool-outputs-intermediates,data,bg_cache,edit-later,text-grade-tests,become-image-finals,motion-streak-finals,style-transfer-bad,decorated_votes_data}
mkdir -p "$PHOTOS"
mkdir -p "$MP_MODELS"
echo "✅ folders under ~/.openclaw/workspace/{shared,_photos}"

# 5. MediaPipe models -----------------------------------------------------
download_mp() {
  local file="$1" url="$2"
  if [ -f "$MP_MODELS/$file" ]; then
    echo "✅ $file"
  else
    echo "→ downloading $file"
    curl -L -o "$MP_MODELS/$file" "$url"
  fi
}
download_mp pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
download_mp hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
download_mp face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
download_mp selfie_multiclass.tflite \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

# 6. .env file ------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<EOF
# Required for the full pipeline. Fill in your tokens.
FAL_API_KEY=
REPLICATE_API_TOKEN=
GOOGLE_API_KEY=
ANTHROPIC_API_KEY=

# Optional. Pushbullet — image-share via S3 still works without Pro,
# but push notifications require Pro and will silently fail on free tier.
PUSHBULLET_TOKEN=

# Legacy / unused by main pipeline:
TENSOR_API_KEY=
GITHUB_API_KEY=
EOF
  echo "⚠️  Wrote $ENV_FILE — fill in API keys before running anything."
else
  echo "✅ env file exists at $ENV_FILE"
fi

# 7. SQLite photo catalog DB ---------------------------------------------
DB="$SHARED/data/photo-catalog.db"
if [ ! -f "$DB" ]; then
  cat <<EOF

⚠️  Photo catalog DB not found at $DB.
   This DB is built externally (Lightroom catalog → SQLite). It is NOT in
   the repo. If you have a backup, copy it to:
       $DB
   Without it, find-candidates.py and any DB-driven photo picking will not work.
   Other tools (manual --source paths) will work fine.

EOF
else
  ROWS=$("$VENV/bin/python3" -c "
import sqlite3
c=sqlite3.connect('file:$DB?mode=ro', uri=True)
print(sum(1 for _ in c.execute('SELECT 1 FROM photos')))
" 2>/dev/null || echo 0)
  echo "✅ photo catalog DB ($ROWS photos)"
fi

# 8. Style refs ------------------------------------------------------------
# Style refs live in the repo at <REPO>/style-refs/. We symlink the legacy
# shared path to the repo so any old references still resolve.
REPO_STYLE_REFS="$REPO/style-refs"
LEGACY_STYLE_REFS="$SHARED/style-refs"
if [ -d "$REPO_STYLE_REFS" ]; then
  if [ -L "$LEGACY_STYLE_REFS" ]; then
    echo "✅ style-refs symlink → $REPO_STYLE_REFS"
  elif [ ! -e "$LEGACY_STYLE_REFS" ]; then
    echo "→ symlinking $LEGACY_STYLE_REFS → $REPO_STYLE_REFS"
    ln -s "$REPO_STYLE_REFS" "$LEGACY_STYLE_REFS"
  else
    echo "ℹ️  $LEGACY_STYLE_REFS exists as a real directory; leaving it alone."
    echo "   Repo style-refs are at $REPO_STYLE_REFS — point tools there or"
    echo "   remove the legacy dir and re-run setup to create the symlink."
  fi
else
  echo "⚠️  No style-refs/ in repo. Drop cleaned screenshots into:"
  echo "       $REPO_STYLE_REFS/<family>/"
fi

# 9. Literary quote DB ----------------------------------------------------
if [ ! -f "$QUOTES_NPY" ]; then
  echo "→ building literary quote DB (one-time, ~2-5 min, network needed)"
  "$VENV/bin/python3" "$REPO/scripts/workflows/build_quote_db.py" \
    --target-lines 10000 --poems 200 || \
    echo "⚠️  Quote DB build failed (PoetryDB / HuggingFace may be down). Re-run later."
else
  echo "✅ literary_quotes.npy"
fi

# 10. Cloudflared ---------------------------------------------------------
if ! command -v cloudflared >/dev/null; then
  echo "→ cloudflared not found; attempting install (requires sudo)"
  CF_TMP="/tmp/cloudflared-$$"
  if curl -fL --output "$CF_TMP" \
       https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
     && sudo install -m 755 "$CF_TMP" /usr/local/bin/cloudflared; then
    rm -f "$CF_TMP"
    echo "✅ cloudflared installed: $(cloudflared --version 2>&1 | head -1)"
  else
    rm -f "$CF_TMP"
    cat <<EOF

⚠️  cloudflared install failed (sudo unavailable, network, or curl error).
   start-gallery.sh uses it for the public tunnel. Install manually:
       curl -L --output /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
       sudo chmod +x /usr/local/bin/cloudflared
   Or run with --no-tunnel for local-only access.

EOF
  fi
else
  echo "✅ cloudflared"
fi

# 11. Sentence-transformers cache hint ------------------------------------
HF_CACHE="$HOME/.cache/huggingface/hub"
if [ -d "$HF_CACHE" ]; then
  SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | awk '{print $1}')
  echo "✅ HuggingFace model cache (~$SIZE)"
else
  echo "ℹ️  HuggingFace cache will populate on first text_overlay/build_quote_db run (~80MB)"
fi

cat <<EOF

============================================================
  Setup complete. Next steps:
    1. Fill API keys in $ENV_FILE
    2. Drop your photos under $PHOTOS/<model-name>/{Processed,Unprocessed}/
    3. Drop style references under $SHARED/style-refs/<family>/
    4. Start the gallery:
         ./manipulating-photos-with-ui/start-gallery.sh
       Or local-only:
         ./manipulating-photos-with-ui/batch-runner.py --no-tunnel --port 5555
    5. Visit http://localhost:5555 (or the cloudflared URL pushed to phone)
============================================================
EOF
