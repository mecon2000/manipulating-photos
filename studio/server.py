"""Studio tool server — FastAPI on 127.0.0.1:8702.

Run:  ~/openclaw-venv/bin/python3 -m uvicorn studio.server:app \
          --host 127.0.0.1 --port 8702
(from the repo root; module imports resolve relative to it)

Endpoints (Phase 1):
  GET  /tools                    step metadata from tool_registry.json
  POST /session/new              {source} → session graph
  GET  /session/{id}             graph JSON
  POST /session/{id}/step        {tool, params?, flags?, seed?, preview?} append
  POST /session/{id}/eval        {node_id?} evaluate chain (cache-aware)
  POST /session/{id}/edit        {node_id, params?, seed?} sibling-branch edit
  POST /session/{id}/undo|redo
  POST /step/run                 stateless one-shot step
  POST /mask/build               {input_ref|source, affect, exclude?} → mask ref
  GET  /object/{ref}             serve a cached object
  GET  /costs/today
"""
import random

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import cache, costs, masks, registry, runner
from .graph import Session
from .paths import ensure_dirs, safe_source

app = FastAPI(title="studio-tool-server")
ensure_dirs()


def _seed_or_random(tool: str, seed):
    """Stochastic tools always get a concrete seed so re-rolls (new node, new
    seed) and cache hits (same node, same seed) both behave."""
    if seed is not None:
        return int(seed)
    if tool in registry.DETERMINISTIC_TOOLS:
        return 0
    return random.randint(0, 999_999)


def _load_session(session_id: str) -> Session:
    try:
        return Session.load(session_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, f"no session {session_id}")


@app.get("/tools")
def tools():
    return registry.steps_meta()


@app.post("/session/new")
def session_new(body: dict = Body(...)):
    try:
        src = safe_source(body["source"])
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not src.is_file():
        raise HTTPException(404, f"no such file: {src}")
    s = Session.create(str(src), cache.put_file(src))
    return s.data


@app.get("/session/{session_id}")
def session_get(session_id: str):
    return _load_session(session_id).data


@app.post("/session/{session_id}/step")
def session_step(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    tool = body["tool"]
    if tool not in registry.steps_meta():
        raise HTTPException(400, f"unknown tool {tool}")
    node = s.add_step(tool, body.get("params") or {},
                      seed=_seed_or_random(tool, body.get("seed")),
                      preview=body.get("preview", True),
                      flags=body.get("flags"),
                      parent=body.get("parent", "HEAD"))
    return node


@app.post("/session/{session_id}/eval")
def session_eval(session_id: str, body: dict = Body(default={})):
    s = _load_session(session_id)
    try:
        return {"results": runner.evaluate(s, body.get("node_id"))}
    except (RuntimeError, KeyError, ValueError, FileNotFoundError) as e:
        raise HTTPException(500, str(e))


@app.post("/session/{session_id}/edit")
def session_edit(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    try:
        tail = s.edit_step(body["node_id"], params=body.get("params"),
                           seed=body.get("seed", "KEEP"),
                           flags=body.get("flags"))
    except KeyError as e:
        raise HTTPException(400, str(e))
    return {"head": tail["id"], "graph": s.data}


@app.post("/session/{session_id}/undo")
def session_undo(session_id: str):
    s = _load_session(session_id)
    return {"head": s.undo()}


@app.post("/session/{session_id}/redo")
def session_redo(session_id: str):
    s = _load_session(session_id)
    return {"head": s.redo()}


@app.post("/step/run")
def step_run(body: dict = Body(...)):
    tool = body["tool"]
    if tool not in registry.steps_meta():
        raise HTTPException(400, f"unknown tool {tool}")
    preview = body.get("preview", True)
    if "input_ref" in body:
        input_ref = body["input_ref"]
    else:
        try:
            src = safe_source(body["source"])
        except PermissionError as e:
            raise HTTPException(403, str(e))
        input_ref = runner.preview_source(src) if preview else cache.put_file(src)
    seed = _seed_or_random(tool, body.get("seed"))
    key = cache.step_key({"tool": tool, "params": body.get("params") or {},
                          "flags": body.get("flags") or [], "seed": seed,
                          "preview": preview, "input": input_ref})
    rec = cache.get_step(key)
    if rec:
        return {"output": rec["output"], "cache_hit": True, "key": key}
    try:
        rec = runner.run_step(tool, body.get("params"), body.get("flags"),
                              seed, input_ref, preview)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(500, str(e))
    cache.put_step(key, rec["output"], {k: v for k, v in rec.items() if k != "output"})
    return {"output": rec["output"], "cache_hit": False, "key": key,
            "wall_time": rec["wall_time"]}


@app.post("/mask/build")
def mask_build(body: dict = Body(...)):
    if "input_ref" in body:
        input_ref = body["input_ref"]
    else:
        try:
            input_ref = cache.put_file(safe_source(body["source"]))
        except PermissionError as e:
            raise HTTPException(403, str(e))
    try:
        return masks.build_mask(input_ref, body.get("affect", "subject"),
                                body.get("exclude", ""),
                                rope_color=body.get("rope_color", "auto"),
                                feather=float(body.get("feather", 0.5)),
                                cleanup=body.get("cleanup", "smooth"))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(500, str(e))


@app.get("/object/{ref}")
def object_get(ref: str):
    p = cache.object_path(ref)
    if p is None:
        raise HTTPException(404, "no such object")
    return FileResponse(p)


@app.get("/costs/today")
def costs_today():
    return costs.today()
