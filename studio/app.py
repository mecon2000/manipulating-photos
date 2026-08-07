"""Studio web app — FastAPI on 127.0.0.1:8701, tailnet-exposed at /studio.

Run:  ~/openclaw-venv/bin/python3 -m uvicorn studio.app:app --host 127.0.0.1 --port 8701
Expose:  tailscale serve --set-path /studio http://127.0.0.1:8701

The app imports the studio libraries directly (same repo, same state dirs);
the :8702 tool server stays available for hub/CLI callers. Works both at /
and behind a /studio path prefix (middleware strips it), so links must be
built from the injected `base`.
"""
import base64
import binascii
import json
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import agent, cache, costs, registry, runner
from .graph import Session
from .paths import RUNS_DIR, SESSIONS_DIR, ensure_dirs, safe_source

WEB = Path(__file__).parent / "web"
app = FastAPI(title="studio")
ensure_dirs()
templates = Jinja2Templates(directory=WEB / "templates")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

PREFIX = "/studio"


@app.middleware("http")
async def strip_prefix(request: Request, call_next):
    """Serve identically at / and under /studio (tailscale serve set-path).
    Deliberately NOT via root_path — Starlette folds root_path into mount
    matching, which 404s the /static mount behind the prefix."""
    path = request.scope["path"]
    if path == PREFIX or path.startswith(PREFIX + "/"):
        request.scope["path"] = path[len(PREFIX):] or "/"
        request.scope.setdefault("state", {})["studio_base"] = PREFIX
    return await call_next(request)


def _base(request: Request) -> str:
    if request.scope.get("state", {}).get("studio_base"):
        return PREFIX
    # tailscale serve strips the /studio mount but marks proxied requests
    if request.headers.get("x-forwarded-host"):
        return PREFIX
    return ""


def _load_session(session_id: str) -> Session:
    try:
        return Session.load(session_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, f"no session {session_id}")


def _ui_path(session_id: str) -> Path:
    return SESSIONS_DIR / session_id / "ui.json"


def _load_ui(session_id: str) -> dict:
    try:
        return json.loads(_ui_path(session_id).read_text())
    except (OSError, ValueError):
        return {"markers": []}


# -- pages -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    sessions = []
    for sid in reversed(Session.list_ids()):
        try:
            s = Session.load(sid)
        except (OSError, ValueError):
            continue
        sessions.append({
            "id": sid,
            "source_name": Path(s.data["source_path"]).name,
            "created_h": time.strftime("%d %b %H:%M",
                                       time.localtime(s.data["created"])),
            "steps": len(s.chain()),
        })
    return templates.TemplateResponse(request, "home.html",
                                      {"base": _base(request),
                                       "sessions": sessions})


@app.get("/new")
def new_session(request: Request, src: str):
    try:
        p = safe_source(src)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not p.is_file():
        raise HTTPException(404, f"no such file: {p}")
    s = Session.create(str(p), cache.put_file(p))
    return RedirectResponse(f"{_base(request)}/s/{s.data['id']}", status_code=303)


@app.get("/s/{session_id}", response_class=HTMLResponse)
def studio_page(request: Request, session_id: str):
    s = _load_session(session_id)
    return templates.TemplateResponse(request, "studio.html", {
        "base": _base(request),
        "session_id": session_id,
        "source_name": Path(s.data["source_path"]).name,
    })


@app.get("/health")
def health():
    return {"ok": True}


# -- API ---------------------------------------------------------------------

@app.get("/api/tools")
def api_tools():
    return registry.steps_meta()


@app.get("/api/session/{session_id}")
def api_session(session_id: str):
    s = _load_session(session_id)
    chain = s.chain()
    outputs, input_ref = {}, None
    for node in chain:  # peek at cache without running anything
        if node["parent"] is None:
            input_ref = runner.preview_source(Path(s.data["source_path"])) \
                if node["preview"] else s.data["source_ref"]
        key = cache.step_key({"tool": node["tool"], "params": node["params"],
                              "flags": node["flags"], "seed": node["seed"],
                              "preview": node["preview"], "input": input_ref})
        rec = cache.get_step(key)
        if rec is None:
            break
        outputs[node["id"]] = rec["output"]
        input_ref = rec["output"]
    head = s.data["head"]
    canvas_ref = outputs.get(head, s.data["source_ref"])
    return {"graph": s.data, "chain": [n["id"] for n in chain],
            "outputs": outputs, "canvas_ref": canvas_ref,
            "ui": _load_ui(session_id)}


@app.post("/api/session/{session_id}/step")
def api_step(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    tool = body["tool"]
    meta = registry.steps_meta()
    if tool not in meta:
        raise HTTPException(400, f"unknown tool {tool}")
    seed = body.get("seed")
    if seed is None and not meta[tool]["deterministic"]:
        import random
        seed = random.randint(0, 999_999)
    node = s.add_step(tool, body.get("params") or {}, seed=seed,
                      preview=True, flags=body.get("flags"))
    return {"node": node}


@app.post("/api/session/{session_id}/eval")
def api_eval(session_id: str, body: dict = Body(default={})):
    s = _load_session(session_id)
    try:
        return {"results": runner.evaluate(s, body.get("node_id"))}
    except (RuntimeError, KeyError, ValueError, FileNotFoundError) as e:
        raise HTTPException(500, str(e))


@app.post("/api/session/{session_id}/edit")
def api_edit(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    try:
        tail = s.edit_step(body["node_id"], params=body.get("params"),
                           seed=body.get("seed", "KEEP"))
    except KeyError as e:
        raise HTTPException(400, str(e))
    return {"head": tail["id"]}


@app.post("/api/session/{session_id}/undo")
def api_undo(session_id: str):
    return {"head": _load_session(session_id).undo()}


@app.post("/api/session/{session_id}/redo")
def api_redo(session_id: str):
    return {"head": _load_session(session_id).redo()}


@app.post("/api/session/{session_id}/brush-mask")
def api_brush_mask(session_id: str, body: dict = Body(...)):
    _load_session(session_id)
    data_url = body.get("png", "")
    if not data_url.startswith("data:image/png;base64,"):
        raise HTTPException(400, "expected png data URL")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
    except (binascii.Error, ValueError):
        raise HTTPException(400, "bad base64")
    ensure_dirs()
    tmp = RUNS_DIR / f"brush-{session_id}-{int(time.time())}.png"
    tmp.write_bytes(raw)
    ref = cache.put_file(tmp)
    tmp.unlink()
    return {"mask": ref}


@app.post("/api/session/{session_id}/ui")
def api_ui(session_id: str, body: dict = Body(...)):
    _load_session(session_id)
    ui = _load_ui(session_id)
    ui.update({k: v for k, v in body.items() if k in ("markers",)})
    _ui_path(session_id).write_text(json.dumps(ui, indent=2))
    return {"ok": True}


@app.get("/api/session/{session_id}/chat-history")
def api_chat_history(session_id: str):
    _load_session(session_id)
    return {"messages": agent.load_history(session_id)}


@app.post("/api/session/{session_id}/chat")
async def api_chat(session_id: str, body: dict = Body(...)):
    _load_session(session_id)
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "empty message")

    async def stream():
        async for ev in agent.chat(session_id, message, body.get("context")):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/object/{ref}")
def api_object(ref: str):
    p = cache.object_path(ref)
    if p is None:
        raise HTTPException(404, "no such object")
    return FileResponse(p)


@app.get("/api/costs/today")
def api_costs():
    return costs.today()
