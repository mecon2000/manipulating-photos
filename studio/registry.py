"""tool_registry.json → step metadata + argv construction.

The registry stays the single source of truth (the hub's converter and the
params panel read the same file). Param names are declared without dashes and
map to `--<name>`; presets/artifacts ride their declared flags. Only additive
interpretation here — nothing the hub's `actions_from_registry` doesn't already
assume.
"""
import json
import os

from .paths import REGISTRY_FILE, VENV_PYTHON, REPO

# Deterministic given --seed (pure local pixels; no generative API in the main
# path). Everything else calls Tensor/fal/Replicate and varies run-to-run.
DETERMINISTIC_TOOLS = {"color-bath", "time-corruption", "ink-dissolution"}

# surreal_with_face is the odd one out: --relit input, --out-dir output,
# no finals copy. Skip until step-splitting lands (needs a style ref anyway).
_UNSUPPORTED = {"surreal_with_face"}


def load_registry() -> dict:
    return json.loads(REGISTRY_FILE.read_text())


def steps_meta() -> dict:
    """Step metadata for /tools: schema, determinism, cost, estimates."""
    out = {}
    for name, t in load_registry().items():
        if not isinstance(t, dict) or "script" not in t or name in _UNSUPPORTED:
            continue
        out[name] = {
            "label": t.get("label", name),
            "params": t.get("params") or {},
            "presets": t.get("presets"),
            "artifacts": t.get("artifacts"),
            "flags": t.get("flags") or [],
            "flag_descriptions": t.get("flag_descriptions") or {},
            "deterministic": name in DETERMINISTIC_TOOLS,
            "output_kind": t.get("output_kind", "image"),
            "cost_estimate_usd": float(t.get("cost_estimate_usd") or 0.0),
            "wall_time_estimate_sec": int(t.get("wall_time_estimate_sec") or 60),
            "needs_style_ref": bool(t.get("needs_style_ref")),
        }
    return out


def _validate_param(name: str, spec: dict, value):
    kind = spec.get("type")
    if kind == "float":
        v = float(value)
    elif kind == "int":
        v = int(value)
    else:
        v = str(value)
    if isinstance(v, (int, float)):
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and v < lo or hi is not None and v > hi:
            raise ValueError(f"param {name}={v} outside [{lo}, {hi}]")
    return str(v)


def build_argv(tool: str, source_path: str, params: dict | None,
               flags: list | None, seed, out_dir: str) -> list[str]:
    reg = load_registry()
    if tool not in reg or tool in _UNSUPPORTED:
        raise KeyError(f"unknown tool {tool}")
    t = reg[tool]
    declared = t.get("params") or {}
    argv = [str(VENV_PYTHON), str(REPO / t["script"]), "--source", str(source_path)]

    params = dict(params or {})
    preset = params.pop("preset", None)
    artifact = params.pop("artifact", None)
    if preset is not None:
        if not t.get("preset_flag") or preset not in (t.get("presets") or []):
            raise ValueError(f"bad preset {preset!r} for {tool}")
        argv += [t["preset_flag"], str(preset)]
    if artifact is not None:
        if not t.get("artifact_flag") or artifact not in (t.get("artifacts") or []):
            raise ValueError(f"bad artifact {artifact!r} for {tool}")
        argv += [t["artifact_flag"], str(artifact)]

    for name, value in params.items():
        spec = declared.get(name)
        if spec is None:
            raise ValueError(f"undeclared param {name!r} for {tool}")
        argv += [spec.get("flag", f"--{name}"), _validate_param(name, spec, value)]

    for fl in flags or []:
        if fl not in (t.get("flags") or []):
            raise ValueError(f"undeclared flag {fl!r} for {tool}")
        argv.append(fl)

    if seed is not None:
        argv += ["--seed", str(int(seed))]

    # Route outputs into the per-run scratch dir. Tools that support
    # --output-to need it set to local or they'll try gdrive.
    script_src = (REPO / t["script"]).read_text()
    if "--output-to" in script_src:
        argv += ["--output-to", "local"]
    argv += ["--local-output-dir", str(out_dir)]
    return argv
