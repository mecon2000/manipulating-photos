"""Step execution: subprocess-wrapped tools + preview downscale + evaluation.

Phase 1 wraps whole tools (subprocess parity with the hub's job runner). Each
run gets a scratch dir under state/runs/; the tool's final output is captured
by parsing its stdout (the tools print `Final: …` / `Saved final: …` in a few
dialects) with a newest-file scan of the scratch dir as fallback, then stored
in the content-addressed cache.

Preview mode downscales the *source* to 1024px long-edge before step 1; the
rest of the chain naturally inherits the small size. Full-res re-render is the
same graph with preview=False (Lock, Phase 4).
"""
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image

from . import cache, costs, registry
from .paths import REPO, RUNS_DIR, ensure_dirs

PREVIEW_LONG_EDGE = 1024
_OUT_EXTS = (".jpg", ".jpeg", ".png", ".mp4", ".tif", ".tiff")
_FINAL_RE = re.compile(
    r"(?:Final|Finals|Final copied to|Copied to finals|Saved final|Saved|Output)"
    r"\s*:?\s+(/.+?\.(?:jpg|jpeg|png|mp4|tiff?))\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def preview_source(path: Path, long_edge: int = PREVIEW_LONG_EDGE) -> str:
    """Downscaled-source object ref (cached as a pseudo-step)."""
    src_ref = cache.put_file(path)
    key = cache.step_key({"step": "__downscale__", "input": src_ref,
                          "long_edge": long_edge})
    rec = cache.get_step(key)
    if rec:
        return rec["output"]
    ensure_dirs()
    img = Image.open(path)
    img.thumbnail((long_edge, long_edge), Image.LANCZOS)
    tmp = RUNS_DIR / f"preview-{src_ref[:12]}.jpg"
    img.convert("RGB").save(tmp, "JPEG", quality=92)
    out_ref = cache.put_file(tmp)
    tmp.unlink()
    cache.put_step(key, out_ref, {"step": "__downscale__"})
    return out_ref


def _capture_output(stdout: str, scratch: Path, started: float) -> Path | None:
    for m in reversed(_FINAL_RE.findall(stdout)):
        p = Path(m)
        if p.exists():
            return p
    newest, newest_ts = None, started
    for p in scratch.rglob("*"):
        if p.suffix.lower() in _OUT_EXTS and p.is_file():
            ts = p.stat().st_mtime
            if ts >= newest_ts:
                newest, newest_ts = p, ts
    return newest


def run_step(tool: str, params: dict | None, flags: list | None, seed,
             input_ref: str, preview: bool) -> dict:
    """Execute one step (no cache check — see evaluate). Returns step record."""
    meta = registry.steps_meta()[tool]
    input_path = cache.object_path(input_ref)
    if input_path is None:
        raise FileNotFoundError(f"input ref {input_ref} not in object store")

    ensure_dirs()
    scratch = RUNS_DIR / (time.strftime("%Y%m%d-%H%M%S") + "-" + tool
                          + "-" + uuid.uuid4().hex[:6])
    scratch.mkdir(parents=True)
    argv = registry.build_argv(tool, input_path, params, flags, seed, scratch)

    env = {**os.environ, "NOTIFY_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
    timeout = min(max(meta["wall_time_estimate_sec"] * 5, 120), 1800)
    started = time.time()
    proc = subprocess.run(argv, cwd=REPO, env=env, timeout=timeout,
                          capture_output=True, text=True)
    (scratch / "run.log").write_text(
        "$ " + " ".join(map(str, argv)) + "\n\n" + proc.stdout +
        ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))

    out = _capture_output(proc.stdout, scratch, started)
    if proc.returncode != 0 or out is None:
        raise RuntimeError(
            f"{tool} failed (rc={proc.returncode}, output "
            f"{'missing' if out is None else out}) — log: {scratch}/run.log")

    costs.accrue(tool, meta["cost_estimate_usd"])
    output_ref = cache.put_file(out)
    return {"output": output_ref, "tool": tool,
            "wall_time": round(time.time() - started, 1),
            "cost_usd": meta["cost_estimate_usd"], "log": str(scratch / "run.log")}


def node_input_ref(session, node: dict) -> str | None:
    """Input ref for a node WITHOUT evaluating: source for root nodes (preview-
    downscaled if the node wants preview), else the parent's cached output."""
    if node["parent"] is None:
        src = Path(session.data["source_path"])
        return preview_source(src) if node["preview"] else cache.put_file(src)
    return None  # caller resolves via evaluation


def evaluate(session, node_id: str | None = None) -> list[dict]:
    """Resolve the chain root→node (default head), running only cache misses.
    Returns one result dict per node: {node, key, output, cache_hit, ...}."""
    results = []
    input_ref = None
    for node in session.chain(node_id):
        if node["parent"] is None:
            input_ref = node_input_ref(session, node)
        key = cache.step_key({
            "tool": node["tool"], "params": node["params"], "flags": node["flags"],
            "seed": node["seed"], "preview": node["preview"], "input": input_ref,
        })
        rec = cache.get_step(key)
        hit = rec is not None
        if not hit:
            rec = run_step(node["tool"], node["params"], node["flags"],
                           node["seed"], input_ref, node["preview"])
            cache.put_step(key, rec["output"],
                           {k: v for k, v in rec.items() if k != "output"})
        results.append({"node": node["id"], "tool": node["tool"], "key": key,
                        "output": rec["output"], "cache_hit": hit,
                        "wall_time": rec.get("wall_time")})
        input_ref = rec["output"]
    return results
