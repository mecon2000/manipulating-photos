#!/home/rong/openclaw-venv/bin/python3
"""
Batch Runner — autonomous photo gallery.

Continuously generates stylized photos with random tools/presets, serves a web
UI (mobile-first). Optionally exposes the UI via cloudflared tunnel so Ronnie
can review from anywhere. Designed to be started and walked away from.

Usage:
    ./batch-runner.py
    ./batch-runner.py --port 5555
    ./batch-runner.py --tools baroque-surround,ink-dissolution
    ./batch-runner.py --no-tunnel
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

SCRIPT_DIR = Path(__file__).resolve().parent
# Workflow tools live in the parent repo at ../scripts/workflows/
WORKFLOWS_DIR = (SCRIPT_DIR.parent / "scripts" / "workflows").resolve()
sys.path.insert(0, str(WORKFLOWS_DIR))

try:
    from notify import push_image, push_text  # type: ignore
except Exception:
    def push_text(*a, **k): return False
    def push_image(*a, **k): return False

try:
    from zoneinfo import ZoneInfo
    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except ImportError:
    ISRAEL_TZ = timezone(timedelta(hours=3))

PHOTOS_DIR = Path("~/.openclaw/workspace/_photos").expanduser()
SHARED_DIR = Path("~/.openclaw/workspace/shared").expanduser()
FAVORITES_DIR = SHARED_DIR / "favorites"
FAVORITES_JSON = FAVORITES_DIR / "favorites.json"
EDIT_LATER_DIR = SHARED_DIR / "edit-later"
STATE_PATH = SHARED_DIR / "batch_state.json"
FINALS_DIR = SHARED_DIR / "finals"

PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
SKIP_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}

PYTHON = "/home/rong/openclaw-venv/bin/python3"

# --- Tool registry ---------------------------------------------------------

# preset_weights: None -> uniform. Otherwise dict name->weight.
TOOLS = {
    "baroque-surround": {
        "weight": 50,
        "preset_flag": "--preset",
        "presets": ["baroque", "renaissance", "dark-romantic", "ethereal", "smoke",
                    "underwater", "ink-water", "aurora", "silk", "embers",
                    "curtains", "whipped-cream", "bubbles"],
        "preset_weights": {"ink-water": 40, "silk": 20, "aurora": 15, "curtains": 10},
        "artifact_flag": "--artifact",
        "artifacts": ["wings", "petals", "hands", "faces", "chains", "serpents",
                      "butterflies", "thorns", "feathers", "flames", "flowers",
                      "skulls", "ribbons", "eyes"],
        "artifact_prob": 0.75,
    },
    "ink-dissolution": {
        "weight": 15,
        "preset_flag": "--medium",
        "presets": ["ink-wash", "watercolor", "canvas", "charcoal", "graphite"],
        "preset_weights": {"ink-wash": 50, "watercolor": 20, "charcoal": 15,
                           "canvas": 10, "graphite": 5},
    },
    "relighting": {
        "weight": 15,
        "preset_flag": "--lighting",
        "presets": ["Dramatic Rim", "Spotlight", "Low Key", "High Key", "Neon Gels",
                    "Teal & Orange", "Red Drama", "Golden Hour", "Window Light",
                    "Overcast Soft", "Candlelight", "Butterfly", "Split Light",
                    "Beauty Dish", "Underwater Caustics", "Moonlight", "Neon Signs",
                    "Firelight", "Laser"],
        "extra_args": ["--auto-correct"],
    },
    "material-swap": {
        "weight": 5,
        "preset_flag": "--material",
        "presets": ["wet glass", "cracked glass", "oily glass", "frosted glass",
                    "marble", "liquid metal", "porcelain", "ice", "gold", "obsidian"],
    },
    "time-corruption": {
        "weight": 5,
        "preset_flag": "--effect",
        "presets": ["ghost", "melt", "trails", "glitch", "full"],
    },
    "noir-paint": {
        "weight": 5,
        "preset_flag": None,  # no required preset
        "presets": [None],
    },
    "pose-geometry": {
        "weight": 5,
        "preset_flag": "--geometry",
        "presets": ["wireframe", "lowpoly", "crystal", "shatter", "refine",
                    "blocks", "contour"],
    },
}

# --- Helpers ---------------------------------------------------------------

def weighted_choice(items, weights=None, default_weight=5):
    """Pick a weighted random from items. weights is dict name->weight."""
    if not items:
        return None
    if weights is None:
        return random.choice(items)
    w = [weights.get(x, default_weight) for x in items]
    return random.choices(items, weights=w, k=1)[0]

def list_photos_for_model(model_dir):
    """Return (processed_photos, unprocessed_photos) as lists of Paths."""
    proc, unproc = [], []
    for sub, bucket in (("Processed", proc), ("processed", proc),
                        ("Unprocessed", unproc), ("unprocessed", unproc)):
        d = model_dir / sub
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in PHOTO_EXTS and f.name.lower() not in SKIP_FILES:
                    bucket.append(f)
    return proc, unproc

def all_models():
    """List model directories under _photos/."""
    out = []
    if not PHOTOS_DIR.is_dir():
        return out
    for d in PHOTOS_DIR.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.lower() in SKIP_FILES or name.endswith(".gdoc") or name.endswith(".bat"):
            continue
        out.append(d)
    return out

def pick_random_photo(avoid_model=None):
    """Pick a random (model_name, Path). 70% processed, 30% unprocessed."""
    models = all_models()
    random.shuffle(models)
    for md in models:
        if avoid_model and md.name == avoid_model:
            continue
        proc, unproc = list_photos_for_model(md)
        if not proc and not unproc:
            continue
        if proc and unproc:
            photos = proc if random.random() < 0.70 else unproc
        else:
            photos = proc or unproc
        return md.name, random.choice(photos)
    return None, None

def pick_tool(allowed=None):
    names = [n for n in TOOLS if not allowed or n in allowed]
    weights = [TOOLS[n]["weight"] for n in names]
    return random.choices(names, weights=weights, k=1)[0]

def pick_preset(tool_name):
    t = TOOLS[tool_name]
    if not t["presets"] or t["presets"] == [None]:
        return None
    return weighted_choice(t["presets"], t.get("preset_weights"))

def pick_artifact(tool_name):
    t = TOOLS[tool_name]
    if "artifacts" not in t:
        return None
    if random.random() > t.get("artifact_prob", 1.0):
        return None
    return random.choice(t["artifacts"])

def now_str():
    return datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")

# --- State -----------------------------------------------------------------

class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.queue = deque()           # pending items (newest appended to right)
        self.history = deque(maxlen=200)  # recently-decided items (for undo/context)
        self.priority = deque()        # priority (front) items
        self.stats = {"generated": 0, "liked": 0, "disliked": 0,
                      "edit_later": 0, "running": True, "paused": False,
                      "failures": 0}
        self.backpressure_notified = False
        self.load()

    def load(self):
        if not STATE_PATH.is_file():
            return
        try:
            data = json.loads(STATE_PATH.read_text())
            for item in data.get("queue", []):
                self.queue.append(item)
            self.stats.update(data.get("stats", {}))
        except Exception:
            pass

    def save(self):
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        data = {
            "queue": list(self.queue),
            "stats": self.stats,
            "saved_at": now_str(),
        }
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(STATE_PATH)

    def add(self, item, priority=False):
        with self.lock:
            if priority:
                self.priority.append(item)
            else:
                self.queue.append(item)
            self.stats["generated"] += 1
            self.save()

    def all_pending(self):
        with self.lock:
            return list(self.priority) + list(self.queue)

    def find(self, item_id):
        with self.lock:
            for lst in (self.priority, self.queue):
                for it in lst:
                    if it["id"] == item_id:
                        return it
        return None

    def remove(self, item_id):
        with self.lock:
            for lst in (self.priority, self.queue):
                for i, it in enumerate(lst):
                    if it["id"] == item_id:
                        del lst[i]
                        self.save()
                        return it
        return None

    def pending_count(self):
        with self.lock:
            return len(self.queue) + len(self.priority)

STATE = State()

# --- Command builder -------------------------------------------------------

def build_command(tool_name, source_path, preset, artifact):
    """Build subprocess args to run a tool. Returns (cmd_list, cmd_str, metadata)."""
    tool = TOOLS[tool_name]
    script = WORKFLOWS_DIR / f"{tool_name}.py"
    cmd = [PYTHON, str(script), "--source", str(source_path)]
    if tool.get("preset_flag") and preset:
        cmd += [tool["preset_flag"], str(preset)]
    if tool.get("artifact_flag") and artifact:
        cmd += [tool["artifact_flag"], str(artifact)]
    # Always output locally to shared/
    cmd += ["--output-to", "local", "--local-output-dir", str(SHARED_DIR)]
    for extra in tool.get("extra_args", []):
        cmd.append(extra)
    cmd_str = " ".join(f"'{c}'" if " " in c else c for c in cmd)
    return cmd, cmd_str

def find_output_after(start_ts, hint_model=None, hint_source=None):
    """Find the newest .jpg in shared/finals/ with mtime >= start_ts."""
    if not FINALS_DIR.is_dir():
        return None
    candidates = []
    for f in FINALS_DIR.iterdir():
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_mtime >= start_ts - 2:
            score = st.st_mtime
            if hint_source and hint_source in f.name:
                score += 100000
            candidates.append((score, f))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

# --- Generation worker -----------------------------------------------------

def generation_loop(allowed_tools=None):
    while STATE.stats.get("running", True):
        try:
            # Backpressure
            pending = STATE.pending_count()
            if STATE.stats.get("paused"):
                time.sleep(2)
                continue
            if pending >= 15:
                if not STATE.backpressure_notified:
                    push_text("Gallery backed up", f"{pending} photos waiting for review")
                    STATE.backpressure_notified = True
                # Wait until queue drops to 10
                while STATE.pending_count() >= 10 and STATE.stats.get("running", True):
                    if STATE.stats.get("paused"):
                        break
                    time.sleep(3)
                if STATE.pending_count() < 10:
                    STATE.backpressure_notified = False
                continue

            # Pick a random job
            tool_name = pick_tool(allowed_tools)
            model_name, photo = pick_random_photo()
            if not photo:
                print("[gen] no photos found, sleeping")
                time.sleep(10)
                continue
            preset = pick_preset(tool_name)
            artifact = pick_artifact(tool_name)

            seed = random.randint(0, 2**32 - 1)
            job_id = uuid.uuid4().hex[:8]
            generate_one(job_id, tool_name, model_name, photo, preset, artifact,
                         seed, priority=False)
        except Exception as e:
            print(f"[gen] loop error: {e}", flush=True)
            STATE.stats["failures"] = STATE.stats.get("failures", 0) + 1
            time.sleep(5)

def generate_one(job_id, tool_name, model_name, photo, preset, artifact,
                 seed, priority=False):
    """Run one tool and add to queue. Returns the item or None on failure."""
    cmd, cmd_str = build_command(tool_name, photo, preset, artifact)
    src_stem = Path(photo).stem
    print(f"[gen] [{job_id}] {tool_name} / {preset} / {artifact} on {model_name}/{src_stem}", flush=True)
    start_ts = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"[gen] [{job_id}] TIMEOUT", flush=True)
        return None
    if res.returncode != 0:
        print(f"[gen] [{job_id}] FAILED rc={res.returncode}", flush=True)
        print((res.stderr or "")[-400:], flush=True)
        STATE.stats["failures"] = STATE.stats.get("failures", 0) + 1
        return None

    # Parse output path from stdout (tools log "Final: <path>")
    out_path = None
    for line in (res.stdout or "").splitlines():
        m = re.search(r"Final:\s+(.+\.(?:jpg|jpeg|png))\s*$", line.strip(), re.IGNORECASE)
        if m:
            cand = Path(m.group(1).strip())
            if cand.is_file():
                out_path = cand
                break
    if out_path is None:
        out_path = find_output_after(start_ts, hint_model=model_name, hint_source=src_stem)
    if out_path is None or not out_path.is_file():
        print(f"[gen] [{job_id}] could not locate output", flush=True)
        return None

    item = {
        "id": job_id,
        "source": str(photo),
        "model": model_name,
        "tool": tool_name,
        "preset": preset,
        "artifact": artifact,
        "seed": seed,
        "command": cmd_str,
        "output": str(out_path),
        "generated_at": now_str(),
        "status": "pending",
    }
    STATE.add(item, priority=priority)
    print(f"[gen] [{job_id}] OK -> {out_path.name}", flush=True)
    return item

# --- Flask app -------------------------------------------------------------

app = Flask(__name__, template_folder=str(SCRIPT_DIR / "templates"))

@app.route("/")
def index():
    return render_template("gallery.html")

@app.route("/api/queue")
def api_queue():
    return jsonify({
        "pending": STATE.all_pending(),
        "pending_count": STATE.pending_count(),
        "stats": STATE.stats,
    })

@app.route("/api/queue/<item_id>")
def api_item(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    return jsonify(it)

@app.route("/api/stats")
def api_stats():
    return jsonify({**STATE.stats, "pending": STATE.pending_count()})

@app.route("/api/image")
def api_image():
    """Serve image file by absolute path. Restricted to SHARED_DIR + PHOTOS_DIR."""
    path = request.args.get("path", "")
    if not path:
        abort(400)
    p = Path(path).expanduser().resolve()
    allowed_roots = [SHARED_DIR.resolve(), PHOTOS_DIR.resolve()]
    if not any(str(p).startswith(str(root)) for root in allowed_roots):
        abort(403)
    if not p.is_file():
        abort(404)
    return send_file(str(p))

@app.route("/api/queue/<item_id>/dislike", methods=["POST"])
def api_dislike(item_id):
    it = STATE.remove(item_id)
    if not it:
        abort(404)
    # Delete the output file
    try:
        Path(it["output"]).unlink()
    except OSError:
        pass
    STATE.stats["disliked"] = STATE.stats.get("disliked", 0) + 1
    STATE.save()
    return jsonify({"ok": True})

@app.route("/api/queue/<item_id>/like", methods=["POST"])
def api_like(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    # Copy to favorites + append to favorites.json
    FAVORITES_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(it["output"])
    model_tag = re.sub(r"[^\w]+", "_", it.get("model") or "").strip("_")
    src_stem = Path(it["source"]).stem
    preset_tag = re.sub(r"[^\w]+", "_", (it.get("preset") or "")).strip("_")
    artifact_tag = re.sub(r"[^\w]+", "_", (it.get("artifact") or "")).strip("_")
    parts = [model_tag, src_stem, it["tool"], preset_tag, artifact_tag]
    parts = [p for p in parts if p]
    fav_name = "__".join(parts) + src.suffix
    fav_name = re.sub(r"[<>:\"/\\|?*]", "_", fav_name)
    fav_path = FAVORITES_DIR / fav_name
    try:
        shutil.copy2(src, fav_path)
    except Exception:
        # Fallback via PIL
        from PIL import Image
        Image.open(src).save(fav_path, quality=95)

    # git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", str(WORKFLOWS_DIR), "rev-parse", "--short", "HEAD"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        git_hash = None

    entry = {
        "file": fav_name,
        "source": it["source"],
        "model": it.get("model"),
        "style": it.get("preset") or it.get("artifact"),
        "tool": it["tool"],
        "score": None,
        "git_commit": git_hash,
        "favorited_at": now_str(),
        "command": it["command"],
        "artifact": it.get("artifact"),
        "seed": it.get("seed"),
    }
    data = {"favorites": []}
    if FAVORITES_JSON.is_file():
        try:
            data = json.loads(FAVORITES_JSON.read_text())
        except Exception:
            pass
    data.setdefault("favorites", []).append(entry)
    FAVORITES_JSON.write_text(json.dumps(data, indent=2))

    STATE.remove(item_id)
    STATE.stats["liked"] = STATE.stats.get("liked", 0) + 1
    STATE.save()
    return jsonify({"ok": True, "file": fav_name})

@app.route("/api/queue/<item_id>/edit", methods=["POST"])
def api_edit(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    EDIT_LATER_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(it["output"])
    dest_img = EDIT_LATER_DIR / src.name
    dest_meta = EDIT_LATER_DIR / (src.stem + ".json")
    try:
        shutil.copy2(src, dest_img)
    except Exception:
        from PIL import Image
        Image.open(src).save(dest_img, quality=95)
    dest_meta.write_text(json.dumps(it, indent=2))
    STATE.remove(item_id)
    STATE.stats["edit_later"] = STATE.stats.get("edit_later", 0) + 1
    STATE.save()
    return jsonify({"ok": True})

@app.route("/api/favorites")
def api_favorites():
    if not FAVORITES_JSON.is_file():
        return jsonify({"favorites": []})
    try:
        return jsonify(json.loads(FAVORITES_JSON.read_text()))
    except Exception:
        return jsonify({"favorites": []})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    STATE.stats["paused"] = True
    STATE.save()
    return jsonify({"paused": True})

@app.route("/api/resume", methods=["POST"])
def api_resume():
    STATE.stats["paused"] = False
    STATE.save()
    return jsonify({"paused": False})

# --- "More like this" ------------------------------------------------------

def _queue_priority_job(tool_name, model_name, photo, preset, artifact):
    """Kick off a job on the priority thread pool (fire-and-forget)."""
    def runner():
        seed = random.randint(0, 2**32 - 1)
        job_id = uuid.uuid4().hex[:8]
        generate_one(job_id, tool_name, model_name, photo, preset, artifact,
                     seed, priority=True)
    t = threading.Thread(target=runner, daemon=True)
    t.start()

@app.route("/api/queue/<item_id>/more/same-style", methods=["POST"])
def api_more_same_style(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    queued = 0
    seen_models = {it.get("model")}
    attempts = 0
    while queued < 4 and attempts < 20:
        attempts += 1
        model_name, photo = pick_random_photo()
        if not photo or model_name in seen_models:
            continue
        seen_models.add(model_name)
        _queue_priority_job(it["tool"], model_name, photo, it.get("preset"), it.get("artifact"))
        queued += 1
    return jsonify({"queued": queued})

@app.route("/api/queue/<item_id>/more/same-photo", methods=["POST"])
def api_more_same_photo(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    photo = Path(it["source"])
    queued = 0
    seen = {(it["tool"], it.get("preset"), it.get("artifact"))}
    attempts = 0
    while queued < 4 and attempts < 20:
        attempts += 1
        tool_name = pick_tool()
        preset = pick_preset(tool_name)
        artifact = pick_artifact(tool_name)
        key = (tool_name, preset, artifact)
        if key in seen:
            continue
        seen.add(key)
        _queue_priority_job(tool_name, it.get("model") or "Unknown", photo, preset, artifact)
        queued += 1
    return jsonify({"queued": queued})

@app.route("/api/queue/<item_id>/more/similar", methods=["POST"])
def api_more_similar(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    src = Path(it["source"])
    model_dir = src.parent.parent  # <model>/Processed/file.jpg
    # Adjacent file numbers in same folder
    m = re.search(r"(\d+)(?=\D*$)", src.stem)
    if not m:
        return jsonify({"queued": 0, "reason": "no number in filename"})
    base_num = int(m.group(1))
    num_span = m.span(1)
    prefix = src.stem[:num_span[0]]
    suffix = src.stem[num_span[1]:]
    digit_w = num_span[1] - num_span[0]

    # Search Processed + Unprocessed siblings
    roots = [p for p in [model_dir / "Processed", model_dir / "processed",
                         model_dir / "Unprocessed", model_dir / "unprocessed"]
             if p.is_dir()]
    found = []
    for offset in [-2, -1, 1, 2, 3, 4]:
        n = base_num + offset
        candidate_stem = f"{prefix}{str(n).zfill(digit_w)}{suffix}"
        for root in roots:
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                p = root / (candidate_stem + ext)
                if p.is_file():
                    found.append(p)
                    break
        if len(found) >= 4:
            break
    queued = 0
    for photo in found[:4]:
        _queue_priority_job(it["tool"], it.get("model") or "Unknown",
                            photo, it.get("preset"), it.get("artifact"))
        queued += 1
    return jsonify({"queued": queued})

# --- Cloudflared tunnel ----------------------------------------------------

def start_cloudflared(port):
    """Start cloudflared quick tunnel, push URL to phone. Returns subprocess.Popen."""
    cf = shutil.which("cloudflared")
    if not cf:
        print("[tunnel] cloudflared not installed — running local-only")
        print("  Install: sudo apt install cloudflared  (or see plan)")
        push_text("Gallery running (local)", f"http://localhost:{port}")
        return None

    proc = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    url_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    found_url = {"url": None}

    def reader():
        for line in proc.stdout:
            line = line.rstrip()
            if not found_url["url"]:
                m = url_re.search(line)
                if m:
                    found_url["url"] = m.group(0)
                    print(f"\n*** GALLERY: {found_url['url']} ***\n", flush=True)
                    push_text("Gallery ready", found_url["url"])
            # Dim output
            if line:
                print(f"[tunnel] {line}", flush=True)
    threading.Thread(target=reader, daemon=True).start()
    return proc

# --- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--tools", default=None,
                   help="Comma-separated subset of tools to use")
    p.add_argument("--no-tunnel", action="store_true")
    p.add_argument("--no-gen", action="store_true", help="Skip generation loop (UI only)")
    args = p.parse_args()

    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    FINALS_DIR.mkdir(parents=True, exist_ok=True)

    allowed = None
    if args.tools:
        allowed = {t.strip() for t in args.tools.split(",") if t.strip()}
        bad = allowed - set(TOOLS)
        if bad:
            print(f"Unknown tools: {bad}. Known: {sorted(TOOLS)}")
            sys.exit(2)

    STATE.stats["running"] = True
    STATE.stats["paused"] = False
    STATE.save()

    # Start cloudflared
    if not args.no_tunnel:
        start_cloudflared(args.port)

    # Start generation thread
    if not args.no_gen:
        t = threading.Thread(target=generation_loop, args=(allowed,), daemon=True)
        t.start()

    print(f"[flask] serving on http://0.0.0.0:{args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
