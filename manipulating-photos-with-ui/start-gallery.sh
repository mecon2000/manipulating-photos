#!/bin/bash
# Launch the autonomous photo gallery (batch-runner).
# Flask on :5555 + cloudflared tunnel + Pushbullet link to phone.
cd "$(dirname "$0")"
exec ./batch-runner.py "$@"
