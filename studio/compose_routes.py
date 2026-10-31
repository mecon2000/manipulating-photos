"""Compose pages + API (plans/deconstruct.md Phase 3). Mounted by app.py.

Pages:  /compose  (pieces + new piece)   /compose/p/{piece}  (piece screen)
API:    /api/compose/...  — thin wrappers over studio/compose.py (the CLI).
Chat:   /api/compose/session/{piece}/chat[-history] — same shape as Studio's
        session chat so web/static/chat.js is reused unchanged.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from . import activity, compose, compose_agent, compose_sync
from .paths import safe_source

router = APIRouter()
_ctx: dict = {}   # templates + base() injected by app.py


def setup(templates, base_fn):
    _ctx.update(templates=templates, base=base_fn)


def _dir(piece: str) -> Path:
    try:
        return compose.piece_dir(piece)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


def _call(fn, piece: str, *a):
    _dir(piece)
    try:
        return fn(piece, *a)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# -- pages -------------------------------------------------------------------

@router.get("/compose", response_class=HTMLResponse)
def compose_home(request: Request, src: str = ""):
    return _ctx["templates"].TemplateResponse(request, "compose.html", {
        "base": _ctx["base"](request), "src": src})


@router.get("/compose/p/{piece}", response_class=HTMLResponse)
def compose_piece_page(request: Request, piece: str):
    _dir(piece)
    return _ctx["templates"].TemplateResponse(request, "compose_piece.html", {
        "base": _ctx["base"](request), "piece": piece})


# -- listing / pickers ---------------------------------------------------------

@router.get("/api/compose/pieces")
def api_pieces():
    return {"pieces": compose.list_pieces(), "jobs": list(compose.JOBS.values())[-10:]}


@router.get("/api/compose/recipes")
def api_recipes():
    return {"recipes": compose.list_recipes()}


@router.get("/api/compose/recipes/{name}/breakdown")
def api_breakdown(name: str, w: int = 480):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    p = compose.RECIPES / f"{name}__breakdown.jpg"
    if not p.is_file():
        raise HTTPException(404, "no breakdown")
    return FileResponse(compose.thumb(p, w))


@router.get("/api/compose/photos")
def api_photos(src: str = "candidates"):
    if src not in ("candidates", "favorites", "finals"):
        raise HTTPException(400, "src must be candidates|favorites|finals")
    return {"photos": compose.photo_choices(src)}


@router.get("/api/compose/thumb")
def api_thumb(path: str, w: int = 360):
    try:
        p = safe_source(path)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not p.is_file():
        raise HTTPException(404, "no such file")
    try:
        return FileResponse(compose.thumb(p, w))
    except OSError as e:
        raise HTTPException(415, f"not an image: {e}")


@router.post("/api/compose/init")
def api_init(body: dict = Body(...)):
    try:
        photo = safe_source(str(body.get("photo", "")))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not photo.is_file():
        raise HTTPException(404, f"no such photo {photo}")
    try:
        job = compose.start_init(str(photo), str(body.get("recipe", "")),
                                 float(body.get("wildness", 0.5)),
                                 crop=body.get("crop"), custom=body.get("custom") or None,
                                 extra_set=body.get("set") or None)
    except FileExistsError as e:
        raise HTTPException(409, f"piece exists: {e}")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return job


@router.get("/api/compose/jobs/{job_id}")
def api_job(job_id: str):
    job = compose.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job (app restarted?)")
    return job


# -- piece -------------------------------------------------------------------

def _full(piece: str) -> dict:
    s = compose.summary(piece)
    s["manifest"] = compose_sync.manifest(piece)
    s["savepoints"] = compose_sync.savepoints(piece)
    s["sync"] = compose_sync.status(piece)
    if s["manifest"] is None and not s["sync"]["pending"]:
        compose_sync.kick_render(piece, 0.1)   # pre-manifest piece: catch up once
    return s


@router.get("/api/compose/piece/{piece}")
def api_piece(piece: str):
    _dir(piece)
    activity.mark_seen(f"compose:{piece}")
    return _full(piece)


@router.get("/api/compose/piece/{piece}/sync")
def api_sync(piece: str):
    _dir(piece)
    m = compose_sync.manifest(piece)
    return {**compose_sync.status(piece), "rev": m.get("rev") if m else None}


@router.post("/api/compose/piece/{piece}/quick-set")
def api_quick_set(piece: str, body: dict = Body(...)):
    return _call(compose_sync.quick_set, piece, body.get("kv") or [], body.get("note") or "")


@router.post("/api/compose/piece/{piece}/order")
def api_order(piece: str, body: dict = Body(...)):
    return _call(compose_sync.order, piece, body.get("ids") or [])


@router.post("/api/compose/piece/{piece}/variant")
def api_variant(piece: str, body: dict = Body(...)):
    return _call(compose_sync.variant, piece, str(body.get("layer", "")), body.get("seed"),
                 body.get("count", 3))


@router.post("/api/compose/piece/{piece}/save")
def api_save(piece: str, body: dict = Body(default={})):
    return {"savepoints": _call(compose_sync.save, piece, str(body.get("name") or ""))}


@router.post("/api/compose/piece/{piece}/fav")
def api_fav(piece: str, body: dict = Body(default={})):
    return _call(compose_sync.start_fav, piece, str(body.get("name") or ""))


@router.post("/api/compose/piece/{piece}/restore")
def api_restore(piece: str, body: dict = Body(...)):
    _call(compose_sync.restore, piece, body.get("n"))
    return _full(piece)


@router.get("/api/compose/piece/{piece}/file/{rel:path}")
def api_piece_file(piece: str, rel: str, w: int = 0):
    d = _dir(piece).resolve()
    p = (d / rel).resolve()
    try:
        p.relative_to(d)
    except ValueError:
        raise HTTPException(403, "outside piece")
    if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".psd"):
        raise HTTPException(404, "no such file")
    if w and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        return FileResponse(compose.thumb(p, w))
    return FileResponse(p)


@router.post("/api/compose/piece/{piece}/set")
def api_set(piece: str, body: dict = Body(...)):
    _call(compose.do_set, piece, body.get("kv") or [], body.get("note") or "")
    return _full(piece)


@router.post("/api/compose/piece/{piece}/reseed")
def api_reseed(piece: str, body: dict = Body(...)):
    _call(compose.reseed, piece, str(body["layer"]))
    return _full(piece)


@router.post("/api/compose/piece/{piece}/render")
def api_render(piece: str, body: dict = Body(default={})):
    _call(compose.do_render, piece, bool(body.get("full")))
    return _full(piece)


@router.post("/api/compose/piece/{piece}/options")
def api_options(piece: str, body: dict = Body(...)):
    return _call(compose.do_options, piece, str(body.get("question") or "Which one?"),
                 str(body.get("vary") or ""))


@router.post("/api/compose/piece/{piece}/pick")
def api_pick(piece: str, body: dict = Body(...)):
    _call(compose.do_pick, piece, str(body.get("question", "")),
          str(body.get("choice", "")), str(body.get("comment") or ""))
    return _full(piece)


@router.post("/api/compose/piece/{piece}/comments")
def api_comment(piece: str, body: dict = Body(...)):
    _dir(piece)
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "empty comment")
    c = compose.add_comment(piece, body.get("x", 0.5), body.get("y", 0.5), text)
    # VLM label ("her left cheek") is slow on a cold model — label in the
    # background; compose_agent.chat waits briefly for it before prompting.
    compose_agent.label_comment_async(piece, c)
    c["note"] = f"[pinned comment {c['id']}] at ({c['x']:.2f},{c['y']:.2f}): {text}"
    return c


@router.post("/api/compose/piece/{piece}/comments/{cid}")
def api_comment_status(piece: str, cid: str, body: dict = Body(...)):
    _dir(piece)
    status = str(body.get("status") or "done")
    if status not in ("open", "done", "wontfix"):
        raise HTTPException(400, "status must be open|done|wontfix")
    try:
        return compose.set_comment_status(piece, cid, status)
    except KeyError:
        raise HTTPException(404, f"no comment {cid}")


# -- chat (chat.js protocol) ----------------------------------------------------

@router.get("/api/compose/session/{piece}/chat-history")
def api_chat_history(piece: str):
    _dir(piece)
    return {"messages": compose_agent.load_history(piece)}


@router.post("/api/compose/session/{piece}/chat")
async def api_chat(piece: str, body: dict = Body(...)):
    _dir(piece)
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "empty message")

    async def stream():
        async for ev in compose_agent.chat(piece, message, body.get("context")):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
