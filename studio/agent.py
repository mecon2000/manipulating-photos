"""Studio buddy — Claude Agent SDK session per Studio session.

Runs on the Claude subscription (CLI login): ANTHROPIC_API_KEY is scrubbed from
the child env (hub/llm.py trick) so the SDK can never fall back to pay-per-token
API billing. Transcripts resume across chats via the stored SDK session id.

chat() is an async generator yielding event dicts ready for NDJSON streaming:
{type: text|tool_start|tool_end|graph_changed|done|error, ...}
"""
import asyncio
import json
import os
import time

# Scrub API-billing keys from THIS process before the SDK ever spawns the CLI:
# ClaudeAgentOptions.env merges over the parent env, so a key inherited from
# the shell would take precedence over the subscription login (and the key in
# ~/sol/.env is out of credits anyway — see memory/CLAUDE.md).
for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(_k, None)

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                              ClaudeSDKClient, TextBlock,
                              create_sdk_mcp_server, tool)

from . import eyes, masks, registry, runner
from .graph import Session
from .paths import SESSIONS_DIR

MODEL = "sonnet"  # fast + cheap on the subscription; the buddy talks in short bursts

SYSTEM_PROMPT = """You are Studio, Ronnie's photo-tweaking buddy. You are looking at the same
photo he is (a portrait/boudoir/shibari art photo — always consented, his own work).

Style: SHORT responses — one idea, optionally "want me to try it?". Opinionated about
aesthetics: composition, color harmony, subject/background contrast, where the eye travels.
Never a wall of text. When he asks for a change, just do it (run_step) and say what you did
in one line. Iterations are cheap 1024px previews — bias toward trying things.

Craft rules you must respect:
- Shibari: ropes stay sharp; model/anatomy strength stays low; skin masks auto-exclude ropes.
- Preserve anatomy — never suggest steps that would alter the body.
- Relighting: lights orthogonal to body axis; no halo/corona/glow in prompts.
- NSFW routing: Flux/Gemini-backed steps block explicit content; local tools always work.

Use run_step with a tool from the list you're given. Params ride the registry schema;
"preset" and "artifact" are special params. Stochastic tools take a seed — re-roll by
running again with a new seed. build_mask makes semantic masks (subject/bg/skin/hair/
clothes, exclude hands/ropes). If he references numbered markers or a brush mask, their
data is appended to his message.

You have eyes: run_step already tells you what each result looks like, and `look`
re-examines the current image any time (optionally with a question — "is her hand
intact?", "critique the composition"). Ground your opinions in what you actually see,
and check results before declaring success on anatomy-sensitive steps."""


HISTORY_REPLAY = 24  # messages of context replayed into each fresh SDK session


def _history_path(session_id: str):
    return SESSIONS_DIR / session_id / "chat.jsonl"


def load_history(session_id: str) -> list[dict]:
    try:
        return [json.loads(line) for line in
                _history_path(session_id).read_text().splitlines() if line]
    except (OSError, ValueError):
        return []


def _append_history(session_id: str, role: str, text: str) -> None:
    p = _history_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"role": role, "text": text, "ts": time.time()}) + "\n")


def _tools_summary() -> str:
    lines = []
    for name, m in registry.steps_meta().items():
        bits = [f"params: {', '.join(m['params']) or 'none'}"]
        if m["presets"]:
            bits.append(f"presets: {', '.join(m['presets'][:8])}")
        if m["artifacts"]:
            bits.append("artifacts available")
        bits.append("deterministic" if m["deterministic"] else
                    f"stochastic ~${m['cost_estimate_usd']:.2f} ~{m['wall_time_estimate_sec']}s")
        lines.append(f"- {name}: {'; '.join(bits)}")
    return "\n".join(lines)


def _make_server(session_id: str, events: asyncio.Queue):
    """In-process MCP server whose tools are bound to this Studio session."""

    async def _emit(ev: dict):
        await events.put(ev)

    def _look_at(ref: str, question: str | None = None) -> str:
        """Eyes on an output ref, NSFW-routed by the session source. Never
        raises — a blind round shouldn't kill the chat."""
        try:
            s = Session.load(session_id)
            return eyes.describe(ref, question=question,
                                 source_path=s.data["source_path"])["text"]
        except Exception as e:
            return f"(eyes unavailable: {e})"

    def _current_ref() -> str:
        s = Session.load(session_id)
        if s.data["head"]:
            return runner.evaluate(s)[-1]["output"]
        return runner.preview_source(s.data["source_path"])

    @tool("run_step", "Append a tool step to the photo's graph and run it (1024px preview). "
          "params is an object matching the tool's schema; use 'preset'/'artifact' keys for those.",
          {"tool": str, "params": dict, "seed": int})
    async def run_step(args):
        t0 = time.time()
        name = args["tool"]
        await _emit({"type": "tool_start", "name": "run_step", "detail": name})
        try:
            def _work():
                s = Session.load(session_id)
                s.add_step(name, args.get("params") or {}, seed=args.get("seed"),
                           preview=True)
                return runner.evaluate(s)
            results = await asyncio.to_thread(_work)
            last = results[-1]
            await _emit({"type": "graph_changed"})
            await _emit({"type": "tool_end", "name": "run_step",
                         "detail": f"{name} done {time.time()-t0:.0f}s"})
            took = "cache hit" if last["cache_hit"] else f"{last.get('wall_time')}s"
            seen = await asyncio.to_thread(_look_at, last["output"])
            return {"content": [{"type": "text", "text":
                    f"ran {name}; output ref {last['output'][:12]}; {took}. "
                    f"The result is now on the canvas. What it looks like: {seen}"}]}
        except Exception as e:
            await _emit({"type": "tool_end", "name": "run_step", "detail": f"{name} FAILED"})
            return {"content": [{"type": "text", "text": f"step failed: {e}"}], "is_error": True}

    @tool("build_mask", "Build a semantic mask. affect: subject|bg|skin|face-skin|body-skin|"
          "hair|clothes|others|all (comma-combos ok). exclude: hands|ropes|hair|clothes|others.",
          {"affect": str, "exclude": str})
    async def build_mask(args):
        await _emit({"type": "tool_start", "name": "build_mask", "detail": args.get("affect", "")})
        try:
            def _work():
                s = Session.load(session_id)
                input_ref = runner.evaluate(s)[-1]["output"] if s.data["head"] else \
                    runner.preview_source(s.data["source_path"])
                return masks.build_mask(input_ref, args.get("affect", "subject"),
                                        args.get("exclude", ""))
            res = await asyncio.to_thread(_work)
            await _emit({"type": "tool_end", "name": "build_mask",
                         "detail": f"{res['info'].get('coverage_pct', '?')}% coverage"})
            return {"content": [{"type": "text", "text":
                    f"mask ref {res['mask'][:12]}: {json.dumps(res['info'])}"}]}
        except Exception as e:
            await _emit({"type": "tool_end", "name": "build_mask", "detail": "FAILED"})
            return {"content": [{"type": "text", "text": f"mask failed: {e}"}], "is_error": True}

    @tool("look", "Look at the current image with fresh eyes (local VLM, NSFW-safe). "
          "Optional question, e.g. 'is her hand mangled?' or 'critique the composition'.",
          {"question": str})
    async def look(args):
        await _emit({"type": "tool_start", "name": "look", "detail": ""})
        try:
            text = await asyncio.to_thread(
                lambda: _look_at(_current_ref(), args.get("question") or None))
            await _emit({"type": "tool_end", "name": "look", "detail": "done"})
            return {"content": [{"type": "text", "text": text}]}
        except Exception as e:
            await _emit({"type": "tool_end", "name": "look", "detail": "FAILED"})
            return {"content": [{"type": "text", "text": f"look failed: {e}"}],
                    "is_error": True}

    @tool("run_variants", "Fan out N seeds of a stochastic step in parallel and look at "
          "each — for cheap steps (≤$0.03 or ≤15s) default to 3 and let Ronnie pick. "
          "Head lands on variant 1; Ronnie can flip on the canvas.",
          {"tool": str, "params": dict, "n": int})
    async def run_variants(args):
        name = args["tool"]
        n = max(2, min(int(args.get("n") or 3), 6))
        await _emit({"type": "tool_start", "name": "run_variants",
                     "detail": f"{name} ×{n}"})
        try:
            def _work():
                from .app import run_variants_impl
                s = Session.load(session_id)
                return run_variants_impl(s, name, args.get("params") or {}, None, n)
            results = await asyncio.to_thread(_work)
            await _emit({"type": "graph_changed"})
            await _emit({"type": "tool_end", "name": "run_variants",
                         "detail": f"{name} ×{n} done"})
            seen = []
            for i, r in enumerate(results):
                desc = await asyncio.to_thread(
                    _look_at, r["output"],
                    "One sentence: what stands out in this render (flaws included)?")
                seen.append(f"variant {i+1} (node {r['node']}): {desc}")
            return {"content": [{"type": "text", "text":
                    f"ran {n} variants of {name}. " + " | ".join(seen) +
                    " — variant 1 is on canvas; Ronnie can flip between them."}]}
        except Exception as e:
            await _emit({"type": "tool_end", "name": "run_variants", "detail": "FAILED"})
            return {"content": [{"type": "text", "text": f"variants failed: {e}"}],
                    "is_error": True}

    @tool("propose_params", "Surface a registry param in Ronnie's panel with a ✨ badge "
          "when a knob deserves direct manipulation. param must exist in the tool's "
          "schema ('preset' counts).", {"tool": str, "param": str, "note": str})
    async def propose_params(args):
        name, param = args["tool"], args["param"]
        meta = registry.steps_meta().get(name)
        if not meta or (param not in meta["params"] and
                        param not in ("preset", "artifact")):
            return {"content": [{"type": "text", "text":
                    f"no such param {param!r} on {name}"}], "is_error": True}
        def _work():
            import json as _j
            from .paths import SESSIONS_DIR
            p = SESSIONS_DIR / session_id / "ui.json"
            try:
                ui = _j.loads(p.read_text())
            except (OSError, ValueError):
                ui = {"markers": []}
            props = ui.setdefault("proposed_params", [])
            if not any(x["tool"] == name and x["param"] == param for x in props):
                props.append({"tool": name, "param": param,
                              "note": args.get("note", ""), "ts": time.time()})
            p.write_text(_j.dumps(ui, indent=2))
        await asyncio.to_thread(_work)
        await _emit({"type": "graph_changed"})
        return {"content": [{"type": "text", "text":
                f"proposed {name}.{param} to the panel (✨)"}]}

    @tool("lock", "Finalize the current draft — Lock = fav. mode 'upscale' (default; "
          "Real-ESRGAN 2x/4x of the exact draft) or 'rerender' (full-res re-run; "
          "generative steps may drift — warn Ronnie first).",
          {"mode": str, "scale": int})
    async def lock_tool(args):
        mode = args.get("mode") or "upscale"
        await _emit({"type": "tool_start", "name": "lock", "detail": mode})
        try:
            def _work():
                from . import lock as lock_mod
                s = Session.load(session_id)
                if mode == "rerender":
                    return lock_mod.lock_rerender(s)
                return lock_mod.lock_upscale(s, scale=int(args.get("scale") or 4))
            res = await asyncio.to_thread(_work)
            await _emit({"type": "graph_changed"})
            await _emit({"type": "tool_end", "name": "lock", "detail": "faved ⭐"})
            return {"content": [{"type": "text", "text":
                    f"locked ({res['mode']}) → {res['file']} (also in finals/). "
                    "It's in favorites with full reconstruction data."}]}
        except Exception as e:
            await _emit({"type": "tool_end", "name": "lock", "detail": "FAILED"})
            return {"content": [{"type": "text", "text": f"lock failed: {e}"}],
                    "is_error": True}

    @tool("save_recipe", "Save the current chain as a reusable recipe. Give it a short "
          "evocative name. All steps default to 'general'; Ronnie can flip specifics "
          "in the save dialog later.", {"name": str})
    async def save_recipe(args):
        try:
            def _work():
                from . import recipes as rmod
                s = Session.load(session_id)
                draft = rmod.suggest_from_session(s)
                if args.get("name"):
                    draft["name"] = str(args["name"])[:80]
                thumb = runner.evaluate(s)[-1]["output"] if s.data["head"] else None
                return rmod.save(draft, thumbnail_ref=thumb)
            r = await asyncio.to_thread(_work)
            return {"content": [{"type": "text", "text":
                    f"saved recipe {r['name']!r} (slug {r['slug']}) with "
                    f"{len(r['steps'])} steps — it's in the Recipes view."}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"save_recipe failed: {e}"}],
                    "is_error": True}

    @tool("update_recipe", "Fold a recurring delta into a recipe's base: merge params "
          "into one step. Use when the same tweak keeps recurring across a batch "
          "(propose it to Ronnie first).",
          {"slug": str, "step_index": int, "params": dict})
    async def update_recipe(args):
        try:
            def _work():
                from . import recipes as rmod
                return rmod.update_step_params(args["slug"],
                                               int(args["step_index"]),
                                               args.get("params") or {})
            r = await asyncio.to_thread(_work)
            return {"content": [{"type": "text", "text":
                    f"folded into {r['slug']} step {args['step_index']}: "
                    f"{json.dumps(r['steps'][int(args['step_index'])]['params'])}"}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"update_recipe failed: {e}"}],
                    "is_error": True}

    @tool("get_graph", "Current step graph: chain of steps root→head with params.", {})
    async def get_graph(args):
        s = Session.load(session_id)
        chain = [{"id": n["id"], "tool": n["tool"], "params": n["params"],
                  "seed": n["seed"]} for n in s.chain()]
        return {"content": [{"type": "text", "text": json.dumps(
            {"source": s.data["source_path"], "chain": chain}, indent=1)}]}

    @tool("undo", "Undo: move head one step back.", {})
    async def undo(args):
        s = Session.load(session_id)
        head = s.undo()
        await _emit({"type": "graph_changed"})
        return {"content": [{"type": "text", "text": f"head now {head or 'source (no steps)'}"}]}

    return create_sdk_mcp_server("studio",
                                 tools=[run_step, run_variants, build_mask, look,
                                        propose_params, lock_tool, save_recipe,
                                        update_recipe, get_graph, undo])


def _scrubbed_env() -> dict:
    env = dict(os.environ)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    return env


async def chat(session_id: str, message: str, context: dict | None = None):
    """Yield NDJSON-ready event dicts for one user message.

    Continuity: the SDK's session store is in-memory (CLI-side resume doesn't
    survive our per-request clients), so Studio keeps its own chat.jsonl and
    replays recent history into each fresh SDK session. Proper transcript
    resume is a Phase 6 item.
    """
    events: asyncio.Queue = asyncio.Queue()
    s = Session.load(session_id)

    prompt = message
    ctx = context or {}
    if ctx.get("markers"):
        prompt += "\n\n[markers on canvas: " + json.dumps(ctx["markers"]) + "]"
    if ctx.get("brush_mask"):
        prompt += f"\n\n[Ronnie painted a brush mask: ref {ctx['brush_mask']}]"

    history = load_history(session_id)[-HISTORY_REPLAY:]
    history_block = ""
    if history:
        lines = [f"{'Ronnie' if h['role'] == 'user' else 'You'}: {h['text']}"
                 for h in history]
        history_block = ("\n\nConversation so far (you are mid-session; don't re-greet):\n"
                         + "\n".join(lines))

    chain = s.chain()
    graph_block = "\n\nSteps currently applied (root→head): " + (
        " → ".join(f"{n['tool']}({json.dumps(n['params'])}, seed={n['seed']})"
                   for n in chain) if chain else "none yet — pristine photo")
    if s.data.get("recipe"):
        graph_block += (
            f"\n\nThis session was seeded from recipe '{s.data['recipe']}' (part of a "
            "batch). Ronnie's tweaks here are DELTAS on the recipe. If you notice the "
            "same delta recurring across photos of this batch (check chat history), "
            "propose folding it into the base recipe — on his yes, use update_recipe.")

    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT + "\n\nCurrent photo: " + s.data["source_path"]
        + "\n\nAvailable tools for run_step:\n" + _tools_summary()
        + graph_block + history_block,
        mcp_servers={"studio": _make_server(session_id, events)},
        allowed_tools=["mcp__studio__run_step", "mcp__studio__run_variants",
                       "mcp__studio__build_mask", "mcp__studio__look",
                       "mcp__studio__propose_params", "mcp__studio__lock",
                       "mcp__studio__save_recipe", "mcp__studio__update_recipe",
                       "mcp__studio__get_graph", "mcp__studio__undo"],
        disallowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep",
                          "WebFetch", "WebSearch", "Task", "NotebookEdit"],
        permission_mode="bypassPermissions",
        max_turns=16,
        env=_scrubbed_env(),
    )

    async def _run(out: asyncio.Queue):
        replies = []
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                replies.append(block.text)
                                await out.put({"type": "text", "delta": block.text})
        except Exception as e:
            await out.put({"type": "error", "message": str(e)})
        finally:
            _append_history(session_id, "user", message)
            if replies:
                _append_history(session_id, "assistant", "\n".join(replies))
            await out.put({"type": "done"})

    task = asyncio.create_task(_run(events))
    while True:
        ev = await events.get()
        yield ev
        if ev["type"] == "done":
            break
    await task
