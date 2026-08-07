#!/usr/bin/env python3
"""Taste-profile miner (§3.6) — regenerates studio/state/taste.json.

Mines favorites.json (tool/preset/param frequencies), the cross-session
journal, and recipe deltas into a compact taste profile every Studio session
reads. Statistical summary always; an LLM distillation pass (Agent SDK on the
subscription, sonnet) turns it into humane guidance when available.

Run manually or as a hub schedule:
  ~/openclaw-venv/bin/python3 studio/mine_taste.py
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(_k, None)

from studio.paths import SHARED, STATE, ensure_dirs   # noqa: E402

TASTE_FILE = STATE / "taste.json"
JOURNAL_FILE = STATE / "journal.md"


def _stats() -> dict:
    try:
        favs = json.loads((SHARED / "favorites" / "favorites.json").read_text()
                          )["favorites"]
    except (OSError, ValueError, KeyError):
        favs = []
    tools = Counter(f.get("tool", "?") for f in favs)
    presets = Counter()
    for f in favs:
        p = f.get("params") or {}
        for key in ("preset", "effect", "medium", "lighting", "style"):
            if p.get(key):
                presets[f"{f.get('tool')}:{p[key]}"] += 1
        for c in f.get("chain") or []:
            if (c.get("params") or {}).get("preset"):
                presets[f"{c['tool']}:{c['params']['preset']}"] += 1
    journal = ""
    try:
        journal = JOURNAL_FILE.read_text()[-3000:]
    except OSError:
        pass
    return {"favorite_count": len(favs),
            "top_tools": tools.most_common(8),
            "top_presets": presets.most_common(12),
            "journal_tail": journal}


def _distill(stats: dict) -> list[str]:
    """LLM pass — 5-8 humane taste statements. Empty list on any failure."""
    try:
        import anyio
        from claude_agent_sdk import ClaudeAgentOptions, query

        prompt = (
            "From this photographer's favorites data and working journal, write 5-8 "
            "short taste statements a photo-editing assistant should follow "
            "(e.g. 'consistently favors muted grades and sharp ropes', 'dislikes "
            "warm casts on skin'). Only statements the data supports. Return a JSON "
            "array of strings, nothing else.\n\n" + json.dumps(stats, indent=1))
        out = []

        async def _run():
            opts = ClaudeAgentOptions(model="sonnet", max_turns=1,
                                      allowed_tools=[],
                                      system_prompt="Return only the JSON array.")
            async for msg in query(prompt=prompt, options=opts):
                text = getattr(msg, "result", None)
                if isinstance(text, str):
                    out.append(text)

        anyio.run(_run)
        text = out[-1] if out else ""
        start, end = text.find("["), text.rfind("]")
        return json.loads(text[start:end + 1]) if start >= 0 else []
    except Exception as e:
        print(f"distill skipped: {e}")
        return []


def main():
    ensure_dirs()
    stats = _stats()
    statements = _distill(stats)
    taste = {"generated": time.strftime("%Y-%m-%d %H:%M"),
             "statements": statements,
             "top_tools": stats["top_tools"],
             "top_presets": stats["top_presets"]}
    TASTE_FILE.write_text(json.dumps(taste, indent=2))
    print(f"taste.json written: {len(statements)} statements, "
          f"{stats['favorite_count']} favorites mined")


if __name__ == "__main__":
    main()
