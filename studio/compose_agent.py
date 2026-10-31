"""Studio buddy for Compose sessions (one piece = one conversation).

Same subscription-only Agent SDK setup + shared brain as agent.py; different
tools: the layer_compose CLI via studio/compose.py. Transcript per piece at
studio/state/compose/<piece>/chat.jsonl (ext4), replayed into each fresh SDK
session like agent.py does.
"""
import asyncio
import json
import threading
import time

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                              ClaudeSDKClient, TextBlock,
                              create_sdk_mcp_server, tool)

from . import compose, eyes
from .agent import HISTORY_REPLAY, MODEL, _brain_blocks, _scrubbed_env, append_journal

SYSTEM_PROMPT = """You are Studio, Ronnie's art buddy, in a COMPOSE session: one of his photos
rebuilt with a deconstructed reference technique as stacked layers (paint, calligraphy, smears,
scraps, scribbles, stamps, grain…) under/over the cut-out subject. You both look at final.jpg.

Style: SHORT replies — one idea, one line on what you changed. Opinionated about composition,
where the eye travels, density vs breathing room, palette discipline.

How you work:
- compose_get first if you don't know the current layer stack, open questions or pinned comments.
- compose_set changes the recipe and re-renders (~10s). Keys: <layer>.<param>, <layer>.z
  (under|over), <layer>.enabled (true|false), <layer>.seed (new int = new random layout),
  <layer>.region, <layer>.exclude / <layer>.include (normalized boxes "x0,y0,x1,y1", ';' between
  boxes, "" clears — THE way to keep a layer off part of the picture; bbox does nothing),
  base.<k> (bw, preset, contrast, grain, subject_mid, edge_dissolve,
  subject_scale, subject_dx, protect_eyes), background.<k> (color, mode). Always give a short note.
- When the question is VISUAL ("denser or sparser?", "which red?", "under or over her?"), don't
  ask in prose — call compose_options with 2-3 values; Ronnie taps a card. Then stop and wait.
- Pinned comments arrive as region-scoped notes "at (x,y) — near …: text" (normalized 0..1 of
  final.jpg). Find the layer(s) that live there (region in compose_get; use the face landmarks
  it lists to place boxes for anything about the face), act with compose_set, then LOOK at the
  result (look with a region around the pin) and only resolve_comment when the change is
  actually visible — if it isn't, say so and try again.
- look re-examines final.jpg (optionally a region) — check your edit actually landed.
- compose_render(full=true) only when he asks for a full-res render; the finished piece goes
  through his Fav ★ button (full-res + PSD + favorites entry + savepoint).

House rules: decorations NEVER cover her eyes (the engine protects them — never disable
base.protect_eyes); red pen gestures stay off the head; preserve anatomy; the subject stays
the focal point."""


def _hist_path(piece: str):
    return compose.COMPOSE_STATE / piece / "chat.jsonl"


def load_history(piece: str) -> list[dict]:
    try:
        return [json.loads(x) for x in _hist_path(piece).read_text().splitlines() if x]
    except (OSError, ValueError):
        return []


def _append(piece: str, role: str, text: str) -> None:
    p = _hist_path(piece)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"role": role, "text": text, "ts": time.time()}) + "\n")


def compact(piece: str) -> str:
    """Summary the agent reads: stack + open questions + open comments."""
    s = compose.summary(piece)
    lines = [f"piece {piece} — recipe {s['recipe']}, wildness {s['wildness']}",
             "base: " + json.dumps({k: v for k, v in s["base"].items() if k != "note"}),
             "background: " + json.dumps({k: v for k, v in s["background"].items()
                                          if k != "note"}),
             "stack (bottom→top):"]
    P = compose.load_piece(piece)
    full = {L["id"]: L for L in P["layers"]}
    for L in s["layers"]:
        if L["id"] == "subject":
            lines.append("  — subject —")
            continue
        src = full.get(L["id"], {})
        lines.append(f"  {'✓' if L['enabled'] else '·'} {L['id']} [{L['kind']}, {L['z']}"
                     f"{', riff' if L['riff'] else ''}] region={L['region']} "
                     f"seed={L['seed']} params={json.dumps(L['params'])}")
    try:   # face landmarks, normalized to final.jpg (the cropped source), for placing boxes
        pts, lit = _face_geo(piece)
        lines.append("face (normalized x,y): " + ", ".join(
            f"{k}=({v[0]:.3f},{v[1]:.3f})" for k, v in pts.items()) +
            "  — her right cheek is image-LEFT. "
            + (f"Lit (brighter) side of her face: {lit}." if lit else "Face lit evenly."))
    except Exception:
        pass
    oq = [q for q in s["questions"] if q.get("status") == "open"]
    if oq:
        lines.append("open questions: " + "; ".join(
            f"{q['id']} {q['question']} ({q['key']}: " +
            ", ".join(f"{o['letter']}={o['value']}" for o in q["options"]) + ")" for q in oq))
    oc = [c for c in s["comments"] if c.get("status") == "open"]
    if oc:
        lines.append("open pinned comments: " + "; ".join(
            f"[{c['id']}] at ({c['x']:.2f},{c['y']:.2f}) {c.get('label', '')}: {c['text']}"
            for c in oc))
    if s["themes"]:
        lines.append("riff themes: " + "; ".join(t.get("name", "") for t in s["themes"]))
    return "\n".join(lines)


def _face_geo(piece: str):
    """Face landmarks normalized to final.jpg + which side of the face is lit.
    Returns (pts, lit) where lit is 'image-left' / 'image-right' / ''."""
    from PIL import Image as _I
    import numpy as _np
    d = compose.piece_dir(piece)
    src = _I.open(d / "source.jpg").convert("L")
    W0, H0 = src.size
    f = json.loads((d / "face.json").read_text())
    pts = {k: (v[0] / W0, v[1] / H0) for k, v in f.items()}
    a = _np.asarray(src, dtype=_np.float32)
    nx, ny = f["nose"]
    r = max(8, int(abs(f["eye_r"][0] - f["eye_l"][0]) * 0.18))

    def patch(cx):
        cx, cy = int(cx), int(ny)
        return float(a[max(0, cy - r):cy + r, max(0, cx - r):cx + r].mean())
    left = patch((f["eye_l"][0] + f["mouth_l"][0]) / 2)
    right = patch((f["eye_r"][0] + nx) / 2 + (f["eye_r"][0] - nx) * 0.3)
    lit = "" if abs(left - right) < 12 else ("image-left" if left > right else "image-right")
    return pts, lit


def face_label(piece: str, x: float, y: float) -> str:
    """Deterministic label when the pin is on the face — the VLM misnames face spots."""
    try:
        pts, lit = _face_geo(piece)
    except Exception:
        return ""
    xs = [v[0] for v in pts.values()]
    ys = [v[1] for v in pts.values()]
    pad = (max(xs) - min(xs)) * 0.08
    if not (min(xs) - pad <= x <= max(xs) + pad and pts["forehead"][1] - pad <= y <= pts["chin"][1] + pad):
        return ""
    ey = (pts["eye_l"][1] + pts["eye_r"][1]) / 2
    side = "image-left" if x < pts["nose"][0] else "image-right"
    her = "her right" if side == "image-left" else "her left"
    if y < ey - 0.02:
        part = "forehead"
    elif abs(y - ey) <= 0.02:
        part = "eye area"
    elif y > pts["mouth_l"][1] + 0.03:
        part = "chin/jaw"
    elif y > pts["nose"][1]:
        part = "mouth/lower cheek"
    else:
        part = "cheek"
    lit_s = "" if not lit else (", the LIT/brighter side" if lit == side else ", the SHADOWED/darker side")
    return f"her face — {side} {part} ({her} {part.split('/')[-1]}{lit_s})"


def region_label(piece: str, x: float, y: float) -> str:
    """'near her left cheek' — local-VLM look at a crop around the pin."""
    fl = face_label(piece, x, y)
    if fl:
        return fl
    d = compose.piece_dir(piece)
    w = 0.22
    region = [max(0.0, x - w / 2), max(0.0, y - w / 2), w, w]
    try:
        txt = eyes.describe(str(d / "final.jpg"), region=region,
                            source_path=compose.load_piece(piece).get("photo"),
                            question="In 3-7 words, what is at the exact centre of this "
                                     "crop (e.g. 'her left cheek', 'red glyphs above her "
                                     "shoulder')? Answer with the phrase only.")["text"]
        t = txt.strip().strip(".").splitlines()[0][:80]
        return t[:1].lower() + t[1:]
    except Exception:
        return ""


def label_comment_async(piece: str, c: dict) -> None:
    def _w():
        label = region_label(piece, c["x"], c["y"])
        if label:
            compose.set_comment_status(piece, c["id"], "open", label=label)
    threading.Thread(target=_w, daemon=True).start()


async def _with_pin_label(piece: str, message: str, pin_id: str, wait: float = 25.0) -> str:
    """Insert ' — near <label>' into a pinned-comment note once the background
    VLM label lands (bounded wait; coordinates alone are still usable)."""
    deadline = time.time() + wait
    while True:
        c = next((x for x in compose.comments(piece) if x["id"] == pin_id), None)
        if c and c.get("label"):
            head, sep, rest = message.partition("): ")
            return f"{head}) — near {c['label']}: {rest}" if sep else message
        if time.time() > deadline:
            return message
        await asyncio.sleep(1.0)


def _server(piece: str, events: asyncio.Queue):
    gate = {"dirty": False}   # a compose_set happened and nobody looked since

    async def emit(ev):
        await events.put(ev)

    def _ok(text):
        return {"content": [{"type": "text", "text": text}]}

    def _err(text):
        return {"content": [{"type": "text", "text": text}], "is_error": True}

    def _p(args):
        return args.get("piece") or piece

    @tool("compose_get", "Piece state: layer stack (bottom→top, enabled, z, riff, region, "
          "bbox, params), base/background, open choice questions, open pinned comments.",
          {"piece": str})
    async def compose_get(args):
        try:
            return _ok(await asyncio.to_thread(compact, _p(args)))
        except Exception as e:
            return _err(f"compose_get failed: {e}")

    @tool("compose_set", "Change the recipe and re-render (~10s). kv: list of "
          "'id.param=value' strings (layer.z=over, layer.enabled=false, layer.seed=123, "
          "base.contrast=1.4 …). note: short why.", {"piece": str, "kv": list, "note": str})
    async def compose_set(args):
        kv = args.get("kv") or []
        await emit({"type": "tool_start", "name": "compose_set", "detail": ", ".join(map(str, kv))[:80]})
        try:
            await asyncio.to_thread(compose.do_set, _p(args), kv, args.get("note") or "")
            gate["dirty"] = True
            await emit({"type": "graph_changed"})
            await emit({"type": "tool_end", "name": "compose_set", "detail": "rendered"})
            return _ok(f"set {kv} and re-rendered final.jpg (now on Ronnie's canvas).")
        except Exception as e:
            await emit({"type": "tool_end", "name": "compose_set", "detail": "FAILED"})
            return _err(f"compose_set failed: {e}")

    @tool("compose_add", "Add a NEW layer as its own toggleable row — use this for a new element "
          "instead of changing an existing layer's kind/prompt (Ronnie wants to play with each "
          "element separately). kind: asset (params.prompt, isolated on white, no text) | print "
          "(params.variant stamp/typed + ring/center/lines; center=@triskelion for the BDSM emblem) | "
          "any kind from compose_get. z under|over, region e.g. at:0.15,0.6. Renders (~40s).",
          {"id": str, "kind": str, "z": str, "region": str, "params": dict, "why": str})
    async def compose_add(args):
        await emit({"type": "tool_start", "name": "compose_add", "detail": args.get("id", "")})
        try:
            await asyncio.to_thread(compose.do_add, piece, str(args.get("id", "")),
                                    str(args.get("kind", "asset")), args.get("z") or "over",
                                    args.get("region") or "anywhere", args.get("params") or {},
                                    args.get("why") or "")
            gate["dirty"] = True
            await emit({"type": "graph_changed"})
            await emit({"type": "tool_end", "name": "compose_add", "detail": "added"})
            return _ok(f"added layer {args.get('id')} and re-rendered.")
        except Exception as e:
            await emit({"type": "tool_end", "name": "compose_add", "detail": "FAILED"})
            return _err(f"compose_add failed: {e}")

    @tool("compose_options", "Raise a VISUAL question as a choice card: renders A/B/C with "
          "one key varied. vary: 'id.param=a,b,c' (2-4 values; use a|b|c when a value contains "
          "commas, e.g. lists). Vary ONE param, never a whole params object. Ronnie taps one; the pick is "
          "applied + logged automatically. Takes ~10s per option.",
          {"piece": str, "question": str, "vary": str})
    async def compose_options(args):
        await emit({"type": "tool_start", "name": "compose_options", "detail": args.get("vary", "")})
        try:
            res = await asyncio.to_thread(compose.do_options, _p(args),
                                          args.get("question") or "Which one?",
                                          args.get("vary") or "")
            await emit({"type": "graph_changed"})
            await emit({"type": "tool_end", "name": "compose_options", "detail": "card posted"})
            q = res["question"] or {}
            return _ok(f"posted choice card {q.get('id')} — wait for Ronnie's tap; don't "
                       "describe the options at length.")
        except Exception as e:
            await emit({"type": "tool_end", "name": "compose_options", "detail": "FAILED"})
            return _err(f"compose_options failed: {e}")

    @tool("compose_render", "Re-render the piece. full=true writes full-res final_full.jpg "
          "+ the layered PSD (slower; only on request). The real finish is Ronnie's Fav "
          "button (full-res + PSD + favorites + savepoint).", {"piece": str, "full": bool})
    async def compose_render(args):
        await emit({"type": "tool_start", "name": "compose_render",
                    "detail": "full" if args.get("full") else "preview"})
        try:
            await asyncio.to_thread(compose.do_render, _p(args), bool(args.get("full")))
            await emit({"type": "graph_changed"})
            await emit({"type": "tool_end", "name": "compose_render", "detail": "done"})
            return _ok("rendered" + (" full-res + PSD" if args.get("full") else ""))
        except Exception as e:
            await emit({"type": "tool_end", "name": "compose_render", "detail": "FAILED"})
            return _err(f"compose_render failed: {e}")

    @tool("resolve_comment", "Mark a pinned comment done (after you acted on it) or "
          "'wontfix' (say why in chat).", {"id": str, "status": str})
    async def resolve_comment(args):
        if (args.get("status") or "done") == "done" and gate["dirty"]:
            return _err("not resolved: you changed the piece but haven't looked. Call look "
                        "with a region around the pin, confirm the change is visible THERE "
                        "and matches what he asked, then resolve.")
        try:
            await asyncio.to_thread(compose.set_comment_status, piece, args["id"],
                                    args.get("status") or "done")
            await emit({"type": "graph_changed"})
            return _ok("comment updated")
        except Exception as e:
            return _err(f"resolve_comment failed: {e}")

    @tool("look", "Look at final.jpg with fresh eyes (local VLM, NSFW-safe). Optional "
          "question; optional region [x,y,w,h] normalized.",
          {"question": str, "region": list})
    async def look(args):
        await emit({"type": "tool_start", "name": "look", "detail": ""})
        try:
            def _w():
                d = compose.piece_dir(piece)
                return eyes.describe(str(d / "final.jpg"), question=args.get("question") or None,
                                     region=args.get("region") or None,
                                     source_path=compose.load_piece(piece).get("photo"))["text"]
            txt = await asyncio.to_thread(_w)
            gate["dirty"] = False
            await emit({"type": "tool_end", "name": "look", "detail": "done"})
            return _ok(txt)
        except Exception as e:
            await emit({"type": "tool_end", "name": "look", "detail": "FAILED"})
            return _err(f"look failed: {e}")

    @tool("journal", "Append a one-line insight to the shared cross-session journal.",
          {"note": str})
    async def journal(args):
        if not (args.get("note") or "").strip():
            return _err("empty note")
        await asyncio.to_thread(append_journal, args["note"])
        return _ok("noted")

    tools = [compose_get, compose_set, compose_add, compose_options, compose_render,
             resolve_comment, look, journal]
    return create_sdk_mcp_server("compose", tools=tools), \
        [f"mcp__compose__{t.name}" for t in tools]


async def chat(piece: str, message: str, context: dict | None = None):
    """NDJSON-ready events, same protocol as agent.chat (chat.js reuses it)."""
    events: asyncio.Queue = asyncio.Queue()
    if (context or {}).get("pin"):
        message = await _with_pin_label(piece, message, str(context["pin"]))
    hist = load_history(piece)[-HISTORY_REPLAY:]
    hist_block = ""
    if hist:
        hist_block = ("\n\nConversation so far (mid-session; don't re-greet):\n" + "\n".join(
            f"{'Ronnie' if h['role'] == 'user' else 'You'}: {h['text']}" for h in hist))
    try:
        state = await asyncio.to_thread(compact, piece)
    except Exception as e:
        state = f"(state unavailable: {e})"
    server, allowed = _server(piece, events)
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT + _brain_blocks()
        + "\n\n# Current piece\n" + state + hist_block,
        mcp_servers={"compose": server},
        allowed_tools=allowed,
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
                await client.query(message)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                replies.append(block.text)
                                await out.put({"type": "text", "delta": block.text})
        except Exception as e:
            await out.put({"type": "error", "message": str(e)})
        finally:
            _append(piece, "user", message)
            if replies:
                _append(piece, "assistant", "\n".join(replies))
            await out.put({"type": "done"})

    task = asyncio.create_task(_run(events))
    while True:
        ev = await events.get()
        yield ev
        if ev["type"] == "done":
            break
    await task
