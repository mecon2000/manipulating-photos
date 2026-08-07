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

from . import (activity, agent, cache, costs, lock, params_usage, recipes,
               registry, runner)
from .graph import Session
from .paths import RUNS_DIR, SESSIONS_DIR, SHARED, ensure_dirs, safe_source

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
def new_session(request: Request, src: str, recipe: str = ""):
    try:
        p = safe_source(src)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not p.is_file():
        raise HTTPException(404, f"no such file: {p}")
    s = Session.create(str(p), cache.put_file(p))
    if recipe:
        r = recipes.get(recipe)
        if r is None:
            raise HTTPException(404, f"no recipe {recipe}")
        recipes.apply_to_session(s, r)
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


@app.get("/recipes", response_class=HTMLResponse)
def recipes_page(request: Request):
    return templates.TemplateResponse(request, "recipes.html",
                                      {"base": _base(request)})


@app.get("/batches", response_class=HTMLResponse)
def batches_page(request: Request):
    return templates.TemplateResponse(request, "batches.html",
                                      {"base": _base(request)})


# -- recipes API -------------------------------------------------------------

HUB_URL = "http://127.0.0.1:8700"
CANDIDATES_DIR = SHARED / "candidates"


@app.get("/api/recipes")
def api_recipes():
    return {"recipes": recipes.list_all()}


@app.get("/api/recipes/suggest")
def api_recipe_suggest(session_id: str):
    s = _load_session(session_id)
    return recipes.suggest_from_session(s)


@app.post("/api/recipes")
def api_recipe_save(body: dict = Body(...)):
    s = _load_session(body["session_id"])
    draft = recipes.suggest_from_session(s)
    if body.get("name"):
        draft["name"] = str(body["name"])[:80]
    marks = body.get("general_marks")  # list of bools aligned with steps
    if isinstance(marks, list):
        for step, mark in zip(draft["steps"], marks):
            step["general"] = bool(mark)
    thumb = api_session(body["session_id"])["canvas_ref"]
    return recipes.save(draft, thumbnail_ref=thumb)


@app.post("/api/recipes/import-favorite")
def api_recipe_import(body: dict = Body(...)):
    try:
        favs = json.loads((SHARED / "favorites" / "favorites.json").read_text()
                          )["favorites"]
    except (OSError, ValueError, KeyError):
        raise HTTPException(500, "cannot read favorites.json")
    wanted = body.get("file")
    entry = next((f for f in reversed(favs) if f.get("file") == wanted), None)
    if entry is None:
        raise HTTPException(404, f"no favorite {wanted!r}")
    return recipes.save(recipes.from_favorite(entry))


@app.get("/api/favorites")
def api_favorites(limit: int = 30):
    try:
        favs = json.loads((SHARED / "favorites" / "favorites.json").read_text()
                          )["favorites"]
    except (OSError, ValueError, KeyError):
        favs = []
    return {"favorites": [{"file": f.get("file"), "tool": f.get("tool")}
                          for f in reversed(favs[-limit:])]}


@app.patch("/api/recipes/{slug}")
def api_recipe_rename(slug: str, body: dict = Body(...)):
    try:
        return recipes.rename(slug, str(body["name"])[:80])
    except KeyError:
        raise HTTPException(404, f"no recipe {slug}")


@app.delete("/api/recipes/{slug}")
def api_recipe_delete(slug: str):
    try:
        recipes.delete(slug)
    except KeyError:
        raise HTTPException(404, f"no recipe {slug}")
    return {"ok": True}


@app.get("/api/recipes/{slug}/thumb")
def api_recipe_thumb(slug: str):
    r = recipes.get(slug)
    if r is None or not r.get("thumbnail"):
        raise HTTPException(404, "no thumbnail")
    return FileResponse(recipes.RECIPES_DIR / r["thumbnail"])


@app.post("/api/recipes/{slug}/apply")
def api_recipe_apply(slug: str, body: dict = Body(default={})):
    """Kick a batch through the hub's job runner (Jobs tab + ntfy)."""
    import random

    import requests as _rq
    if recipes.get(slug) is None:
        raise HTTPException(404, f"no recipe {slug}")
    n = max(1, min(int(body.get("n", 4)), 12))
    sources = body.get("sources") or []
    if not sources:
        pool = sorted(str(p) for p in CANDIDATES_DIR.glob("*.jp*g"))
        model = (body.get("model") or "").lower()
        if model:
            pool = [p for p in pool if model in Path(p).name.lower()]
        if not pool:
            raise HTTPException(404, "no matching candidate photos")
        sources = random.sample(pool, min(n, len(pool)))
    try:
        r = _rq.post(f"{HUB_URL}/api/p/photo-tools/action/studio-apply-recipe",
                     json={"sources": sources, "params": {"recipe": slug}},
                     timeout=30)
        r.raise_for_status()
        return {"job": r.json().get("job"), "sources": sources,
                "jobs_url": "/#p=photo-tools&tab=jobs"}
    except _rq.RequestException as e:
        raise HTTPException(502, f"hub job runner unreachable: {e}")


@app.get("/api/batches")
def api_batches():
    out = []
    bdir = SHARED / "studio" / "batches"
    if bdir.exists():
        for p in sorted(bdir.glob("*/batch.json"), reverse=True):
            try:
                out.append(json.loads(p.read_text()))
            except ValueError:
                continue
    return {"batches": out[:20]}


@app.get("/api/batches/{batch_id}/{filename}")
def api_batch_file(batch_id: str, filename: str):
    if "/" in batch_id or "/" in filename or ".." in batch_id or ".." in filename:
        raise HTTPException(400, "bad path")
    p = SHARED / "studio" / "batches" / batch_id / filename
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p)


# -- API ---------------------------------------------------------------------

@app.get("/api/tools")
def api_tools():
    meta = registry.steps_meta()
    for tool, m in meta.items():
        hidden = set(params_usage.hidden_params(tool))
        for pname, spec in m["params"].items():
            spec["hidden"] = pname in hidden
    return meta


@app.post("/api/params-usage")
def api_params_usage(body: dict = Body(...)):
    kind = body.get("kind")
    if kind not in ("appear", "touch"):
        raise HTTPException(400, "kind must be appear|touch")
    params_usage.record(str(body["tool"]), str(body["param"]), kind)
    return {"ok": True}


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
    variants = {}
    for node in chain:
        group = s.variant_group(node["id"])
        if len(group) > 1:
            variants[node["id"]] = [g["id"] for g in group]
            for g in group:  # variant outputs are peekable too
                key = cache.step_key({"tool": g["tool"], "params": g["params"],
                                      "flags": g["flags"], "seed": g["seed"],
                                      "preview": g["preview"],
                                      "input": outputs.get(g["parent"]) or
                                      (runner.preview_source(Path(s.data["source_path"]))
                                       if g["parent"] is None else None)})
                rec = cache.get_step(key)
                if rec:
                    outputs[g["id"]] = rec["output"]
    return {"graph": s.data, "chain": [n["id"] for n in chain],
            "outputs": outputs, "canvas_ref": canvas_ref,
            "variants": variants, "ui": _load_ui(session_id)}


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
    label = " → ".join(n["tool"] for n in s.chain()[-2:]) or "eval"
    name = Path(s.data["source_path"]).name

    if body.get("background"):
        import threading

        def _bg():
            try:
                runner.evaluate(s, body.get("node_id"))
                activity.finish(session_id, True, name)
            except Exception:
                activity.finish(session_id, False, name)

        activity.start(session_id, label)
        threading.Thread(target=_bg, daemon=True).start()
        return {"background": True}

    activity.start(session_id, label)
    try:
        results = runner.evaluate(s, body.get("node_id"))
        activity.finish(session_id, True, name)
        activity.mark_seen(session_id)  # the requesting client sees it directly
        return {"results": results}
    except (RuntimeError, KeyError, ValueError, FileNotFoundError) as e:
        activity.finish(session_id, False, name)
        raise HTTPException(500, str(e))


@app.get("/api/sessions-status")
def api_sessions_status():
    """Tab strip data: recent sessions + activity (spinner/ready badges)."""
    act = activity.status()
    sessions = []
    for sid in reversed(Session.list_ids()[-12:]):
        try:
            s = Session.load(sid)
        except (OSError, ValueError):
            continue
        sessions.append({"id": sid,
                         "source_name": Path(s.data["source_path"]).name,
                         "steps": len(s.chain()),
                         **act.get(sid, {"running": False, "ready": False})})
    return {"sessions": sessions}


@app.post("/api/session/{session_id}/seen")
def api_seen(session_id: str):
    activity.mark_seen(session_id)
    return {"ok": True}


@app.post("/api/session/{session_id}/edit")
def api_edit(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    try:
        tail = s.edit_step(body["node_id"], params=body.get("params"),
                           seed=body.get("seed", "KEEP"))
    except KeyError as e:
        raise HTTPException(400, str(e))
    return {"head": tail["id"]}


def run_variants_impl(s: Session, tool: str, params: dict, flags, n: int,
                      seeds=None) -> list[dict]:
    """Fan out n sibling variant nodes (same parent/params, different seeds)
    and evaluate them in parallel. Head lands on the first variant."""
    import random
    from concurrent.futures import ThreadPoolExecutor
    if tool in registry.DETERMINISTIC_TOOLS:
        raise ValueError(f"{tool} is deterministic — variants would be identical; "
                         "run it once instead")
    parent = s.data["head"]
    if parent:  # materialize the shared prefix once, before fanning out
        runner.evaluate(s)
    seeds = list(seeds or [])
    while len(seeds) < n:
        seeds.append(random.randint(0, 999_999))
    nodes = [s.add_step(tool, params, seed=seeds[i], preview=True,
                        flags=flags, parent=parent) for i in range(n)]
    with ThreadPoolExecutor(max_workers=min(n, 4)) as ex:
        futs = [ex.submit(runner.evaluate, s, node["id"]) for node in nodes]
        results = [f.result()[-1] for f in futs]
    s.set_head(nodes[0]["id"])
    return results


@app.post("/api/session/{session_id}/variants")
def api_variants(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    tool = body["tool"]
    meta = registry.steps_meta()
    if tool not in meta:
        raise HTTPException(400, f"unknown tool {tool}")
    n = max(2, min(int(body.get("n", 3)), 6))
    try:
        results = run_variants_impl(s, tool, body.get("params") or {},
                                    body.get("flags"), n)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise HTTPException(500, str(e))
    return {"results": results, "head": s.data["head"]}


@app.post("/api/session/{session_id}/head")
def api_head(session_id: str, body: dict = Body(...)):
    s = _load_session(session_id)
    try:
        return {"head": s.set_head(body.get("node_id"))}
    except KeyError as e:
        raise HTTPException(400, str(e))


@app.post("/api/session/{session_id}/lock")
def api_lock(session_id: str, body: dict = Body(default={})):
    s = _load_session(session_id)
    mode = body.get("mode", "upscale")
    try:
        if mode == "rerender":
            return lock.lock_rerender(s)
        return lock.lock_upscale(s, scale=int(body.get("scale", 4)))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise HTTPException(500, str(e))


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


SAM_URL = "http://127.0.0.1:8703"


def _canvas_ref(s: Session) -> str:
    return api_session(s.data["id"])["canvas_ref"]


@app.post("/api/session/{session_id}/point-mask")
def api_point_mask(session_id: str, body: dict = Body(...)):
    """Tap-to-mask: forward normalized points to the SAM 2 service against
    the current canvas image."""
    import requests as _rq
    s = _load_session(session_id)
    payload = {"ref": body.get("ref") or _canvas_ref(s),
               "points": body.get("points") or [],
               "labels": body.get("labels")}
    if body.get("box"):
        payload["box"] = body["box"]
    try:
        r = _rq.post(f"{SAM_URL}/segment", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except _rq.RequestException as e:
        raise HTTPException(502, f"SAM service unavailable: {e}")


@app.post("/api/session/{session_id}/describe")
def api_describe(session_id: str, body: dict = Body(default={})):
    """Eyes on the current canvas (or a region of it) — used for marker labels
    and ad-hoc looks. Routed local/Gemini by the session source's NSFW level."""
    from . import eyes
    s = _load_session(session_id)
    try:
        return eyes.describe(body.get("ref") or _canvas_ref(s),
                             question=body.get("question"),
                             region=body.get("region"),
                             source_path=s.data["source_path"])
    except Exception as e:
        raise HTTPException(502, f"eyes failed: {e}")


@app.post("/api/session/{session_id}/ui")
def api_ui(session_id: str, body: dict = Body(...)):
    _load_session(session_id)
    ui = _load_ui(session_id)
    ui.update({k: v for k, v in body.items()
               if k in ("markers", "proposed_params")})
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
