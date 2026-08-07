"""Recipes (§1.3/§3.5): a saved look = ordered steps + general params.

Stored as small JSON in shared/studio/recipes/ (Windows-visible on purpose;
schema versioned) with a thumbnail jpg beside each. Steps carry a `general`
flag — photo-specific steps are dropped at apply time; semantic masks
(--affect style params) re-derive per photo by nature. Deletion = move to
shared/studio/trash/.
"""
import json
import re
import shutil
import time

from . import cache
from .paths import SHARED

RECIPES_DIR = SHARED / "studio" / "recipes"
TRASH_DIR = SHARED / "studio" / "trash"
SCHEMA_VERSION = 1


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "recipe"
    base, n = slug, 2
    while (RECIPES_DIR / f"{slug}.json").exists():
        slug = f"{base}-{n}"
        n += 1
    return slug


def suggest_from_session(session) -> dict:
    """Draft recipe from the active chain, pre-marked general/specific.
    Heuristic: steps are general (style/strength/preset travel well);
    a step whose params carry obviously per-photo values (explicit pixel
    coords, custom per-photo prompts) would be marked specific — none of the
    current registry params are per-photo, so the frontend checkboxes are the
    real gate."""
    steps = [{"tool": n["tool"], "params": n["params"], "flags": n["flags"],
              "seed": n["seed"], "general": True}
             for n in session.chain()]
    return {"version": SCHEMA_VERSION,
            "name": " + ".join(dict.fromkeys(s["tool"] for s in steps)) or "empty",
            "steps": steps,
            "source_session": session.data["id"],
            "source_path": session.data["source_path"]}


def save(draft: dict, thumbnail_ref: str | None = None) -> dict:
    RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(draft.get("name", "recipe"))
    recipe = {**draft, "slug": slug, "version": SCHEMA_VERSION,
              "created": time.time(), "updated": time.time()}
    if thumbnail_ref:
        p = cache.object_path(thumbnail_ref)
        if p is not None:
            shutil.copyfile(p, RECIPES_DIR / f"{slug}.jpg")
            recipe["thumbnail"] = f"{slug}.jpg"
    (RECIPES_DIR / f"{slug}.json").write_text(json.dumps(recipe, indent=2))
    return recipe


def list_all() -> list[dict]:
    if not RECIPES_DIR.exists():
        return []
    out = []
    for p in sorted(RECIPES_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return sorted(out, key=lambda r: r.get("created", 0), reverse=True)


def get(slug: str) -> dict | None:
    if not re.fullmatch(r"[a-z0-9-]+", slug or ""):
        return None
    p = RECIPES_DIR / f"{slug}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def rename(slug: str, new_name: str) -> dict:
    r = get(slug)
    if r is None:
        raise KeyError(slug)
    r["name"] = new_name
    r["updated"] = time.time()
    (RECIPES_DIR / f"{slug}.json").write_text(json.dumps(r, indent=2))
    return r


def update_step_params(slug: str, step_index: int, params: dict) -> dict:
    """Fold a delta into the base recipe (agent's update_recipe)."""
    r = get(slug)
    if r is None:
        raise KeyError(slug)
    r["steps"][step_index]["params"] = {**r["steps"][step_index]["params"],
                                        **params}
    r["updated"] = time.time()
    (RECIPES_DIR / f"{slug}.json").write_text(json.dumps(r, indent=2))
    return r


def delete(slug: str) -> None:
    if get(slug) is None:
        raise KeyError(slug)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for suffix in (".json", ".jpg"):
        p = RECIPES_DIR / f"{slug}{suffix}"
        if p.exists():
            shutil.move(str(p), TRASH_DIR / f"{slug}-{stamp}{suffix}")


def apply_to_session(session, recipe: dict) -> list[dict]:
    """Append the recipe's general steps to a session (no evaluation)."""
    added = []
    for step in recipe["steps"]:
        if not step.get("general", True):
            continue
        added.append(session.add_step(step["tool"], step["params"],
                                      seed=step.get("seed"),
                                      preview=True, flags=step.get("flags")))
    session.data["recipe"] = recipe["slug"]
    session.save()
    return added


def from_favorite(entry: dict) -> dict:
    """One-time importer: favorites.json entry → starter recipe draft."""
    if entry.get("tool") == "studio" and entry.get("chain"):
        steps = [{"tool": c["tool"], "params": c.get("params", {}),
                  "flags": c.get("flags", []), "seed": c.get("seed"),
                  "general": True} for c in entry["chain"]]
    else:
        params = {k: v for k, v in (entry.get("params") or {}).items()
                  if isinstance(v, (int, float, str))}
        steps = [{"tool": entry.get("tool"), "params": params, "flags": [],
                  "seed": (entry.get("params") or {}).get("seed"),
                  "general": True}]
    return {"version": SCHEMA_VERSION,
            "name": f"fav: {entry.get('file', 'unknown')[:40]}",
            "steps": steps, "source_favorite": entry.get("file")}
