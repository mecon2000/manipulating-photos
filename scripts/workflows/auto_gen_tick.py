#!/usr/bin/env python3
"""One auto-generation tick — the hub-era replacement for batch-runner's
generation_loop thread. Picks a source photo + tool + preset weighted by your
favorites, runs ONE generation, and respects a daily cost cap. Driven by a hub
schedule (visible/pausable in the Schedules tab); each tick is a normal hub job
with logs, history and ntfy.

Weights: tools that appear more in favorites.json get picked more (+1 smoothing).
Sources: random jpg from shared/candidates/ (top up with find-candidates.py).
Cost: estimated per-tool cost accrues into a daily ledger; tick refuses to run
past --daily-cap-usd.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
REGISTRY = REPO / "manipulating-photos-with-ui" / "tool_registry.json"
SHARED = Path(os.path.expanduser("~/.openclaw/workspace/shared"))
FAVORITES_JSON = SHARED / "favorites" / "favorites.json"
CANDIDATES = SHARED / "candidates"
LEDGER = SHARED / "photo-tools" / "auto_gen_ledger.json"
PYTHON = os.path.expanduser("~/openclaw-venv/bin/python")


def eligible_tools(registry: dict, only: set[str] | None) -> dict:
    out = {}
    for name, t in registry.items():
        if not isinstance(t, dict) or "script" not in t:
            continue
        if t.get("needs_style_ref") or t.get("output_kind") == "video":
            continue
        if only and name not in only:
            continue
        out[name] = t
    return out


def tool_weights(tools: dict) -> dict[str, float]:
    counts = {name: 1.0 for name in tools}          # +1 smoothing
    try:
        favs = json.loads(FAVORITES_JSON.read_text()).get("favorites", [])
        for e in favs:
            t = e.get("tool")
            if t in counts:
                counts[t] += 1.0
    except (OSError, ValueError):
        pass
    return counts


def spend_today() -> float:
    try:
        ledger = json.loads(LEDGER.read_text())
    except (OSError, ValueError):
        ledger = {}
    return float(ledger.get(time.strftime("%Y-%m-%d"), 0.0))


def accrue(usd: float) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger = json.loads(LEDGER.read_text())
    except (OSError, ValueError):
        ledger = {}
    day = time.strftime("%Y-%m-%d")
    ledger[day] = float(ledger.get(day, 0.0)) + usd
    LEDGER.write_text(json.dumps(ledger, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1, help="generations this tick")
    p.add_argument("--daily-cap-usd", type=float, default=1.0)
    p.add_argument("--tools", default="", help="csv filter, e.g. ink-dissolution,color-bath")
    p.add_argument("--local-output-dir", default=str(SHARED / "tool-outputs-intermediates"))
    args = p.parse_args()

    registry = json.loads(REGISTRY.read_text())
    only = {t.strip() for t in args.tools.split(",") if t.strip()} or None
    tools = eligible_tools(registry, only)
    if not tools:
        print("no eligible tools")
        sys.exit(1)
    sources = sorted(CANDIDATES.glob("*.jpg")) + sorted(CANDIDATES.glob("*.jpeg"))
    if not sources:
        print(f"no candidates in {CANDIDATES} — run find-candidates.py")
        sys.exit(1)

    weights = tool_weights(tools)
    rc_total = 0
    for i in range(max(1, args.count)):
        spent = spend_today()
        if spent >= args.daily_cap_usd:
            print(f"daily cap reached (${spent:.2f} >= ${args.daily_cap_usd:.2f}) — skipping")
            break
        name = random.choices(list(tools), weights=[weights[t] for t in tools])[0]
        tool = tools[name]
        src = str(random.choice(sources))
        cmd = [PYTHON, str(REPO / tool["script"]), "--source", src]
        if tool.get("presets") and tool.get("preset_flag"):
            cmd += [tool["preset_flag"], random.choice(tool["presets"])]
        cmd += ["--output-to", "local", "--local-output-dir", args.local_output_dir]
        est = float(tool.get("cost_estimate_usd") or 0.0)
        print(f"[tick {i+1}/{args.count}] {name} on {Path(src).name} "
              f"(est ${est:.2f}, today ${spent:.2f}/{args.daily_cap_usd:.2f})", flush=True)
        print("$ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        accrue(est)
        rc_total += abs(r.returncode)
    sys.exit(1 if rc_total else 0)


if __name__ == "__main__":
    main()
