#!/usr/bin/env bash
# Start the Studio stack: tool server (:8702) + web app (:8701).
# Expose on the tailnet once:  tailscale serve --set-path /studio http://127.0.0.1:8701
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/openclaw-venv/bin/python3

"$PY" -m uvicorn studio.server:app --host 127.0.0.1 --port 8702 &
TOOL_PID=$!
"$PY" -m uvicorn studio.sam_service:app --host 127.0.0.1 --port 8703 &
SAM_PID=$!
trap 'kill $TOOL_PID $SAM_PID 2>/dev/null' EXIT

exec "$PY" -m uvicorn studio.app:app --host 127.0.0.1 --port 8701
