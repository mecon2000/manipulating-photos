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
BLACKLIST_PATH = SHARED_DIR / "blacklisted_models.json"

# --- Motion-streak candidate mode state ------------------------------------
MS_CANDIDATES_JSON = SHARED_DIR / "motion_streak_candidates.json"
MS_REJECTS_JSON = SHARED_DIR / "motion_streak_rejects.json"
MS_BLACKLIST_PATH = SHARED_DIR / "motion_streak_blacklist.json"

# Global mode flag (set in main). "gallery" (default) or "candidates".
MODE = "gallery"
# Per-session boost multiplier for motion-streak model picking (one-time per session).
MS_BOOSTS = {}


def _load_json_set(path, key):
    if not Path(path).is_file():
        return set()
    try:
        return set(json.loads(Path(path).read_text()).get(key, []))
    except Exception:
        return set()


def _save_json_set(path, key, items):
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({key: sorted(items)}, indent=2))


def load_ms_candidates():
    if not MS_CANDIDATES_JSON.is_file():
        return []
    try:
        return json.loads(MS_CANDIDATES_JSON.read_text()).get("candidates", [])
    except Exception:
        return []


def save_ms_candidates(entries):
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    MS_CANDIDATES_JSON.write_text(json.dumps({"candidates": entries}, indent=2))


def ms_candidate_paths():
    return {e.get("source") for e in load_ms_candidates()}


def ms_load_rejects():
    return _load_json_set(MS_REJECTS_JSON, "paths")


def ms_save_rejects(paths):
    _save_json_set(MS_REJECTS_JSON, "paths", paths)


def ms_load_blacklist():
    return _load_json_set(MS_BLACKLIST_PATH, "models")


def ms_save_blacklist(models):
    _save_json_set(MS_BLACKLIST_PATH, "models", models)


MS_REJECTS = ms_load_rejects()
MS_BLACKLIST = ms_load_blacklist()


def load_blacklist():
    if not BLACKLIST_PATH.is_file():
        return set()
    try:
        return set(json.loads(BLACKLIST_PATH.read_text()).get("models", []))
    except Exception:
        return set()


def save_blacklist(models):
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    BLACKLIST_PATH.write_text(json.dumps({"models": sorted(models)}, indent=2))


BLACKLIST = load_blacklist()

PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
SKIP_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}

PYTHON = "/home/rong/openclaw-venv/bin/python3"

# --- Tool registry ---------------------------------------------------------

# preset_weights: None -> uniform. Otherwise dict name->weight.
# Per-run cost estimate in USD for the typical/default invocation of each tool.
# The dict key maps 1:1 to a tool name; batch-runner adds TOOL_COST[tool] on each job
# without inspecting flags, so under-reports when expensive optional flags are enabled
# (e.g. --tile-refine, --foreground-wisp) and over-reports when opted out.
#
# Calibrated 2026-04-19 from $1.21 spent over known workload (see memory file
# project_fal_calibration.md). Derived unit costs:
#   Flux schnell ~$0.009/call, BiRefNet ~$0.001/call, fal face-swap ~$0.095/call
# baroque-surround: 1 Flux + 1 BiRefNet = ~$0.01. With --foreground-wisp: ~$0.02.
# With --tile-refine (default denoise>=0.2 triggers face-swap): ~$0.10.
# polish: 0 Flux + 0 BiRefNet + 1 face-swap = ~$0.10 (Tensor credits extra).
TOOL_COST = {
    "baroque-surround": 0.01,
    "ink-dissolution": 0.003,
    "relighting": 0.06,
    "material-swap": 0.04,
    "time-corruption": 0.003,
    "noir-paint": 0.10,
    "pose-geometry": 0.003,
    "foreground-framing": 0.04,
    "stylizing-bg-model-separately": 0.12,
    "smart-crop": 0.01,
    "torn-reveal": 0.005,
    "body-segment": 0.0,
    "color-bath": 0.0,
    "polish": 0.10,
}

TOOLS = {
    "baroque-surround": {
        "weight": 45,
        "preset_flag": "--preset",
        "presets": ["baroque", "renaissance", "dark-romantic", "ethereal", "smoke",
                    "underwater", "ink-water", "aurora", "silk", "embers",
                    "curtains", "whipped-cream", "bubbles",
                    "velvet-fog", "coral-smoke", "neon-smoke-rings",
                    "burning-silk", "torn-cloud", "spun-sugar", "powdered-pigment"],
        "preset_weights": {"ink-water": 40, "silk": 20, "aurora": 15, "curtains": 10},
        "artifact_flag": "--artifact",
        "artifacts": ["wings", "petals", "hands", "faces", "chains", "serpents",
                      "butterflies", "thorns", "feathers", "flames", "flowers",
                      "skulls", "ribbons", "eyes"],
        "artifact_prob": 0.75,
    },
    "ink-dissolution": {
        "weight": 30,
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
        "weight": 10,
        "preset_flag": "--effect",
        "presets": ["ghost", "melt", "trails", "glitch", "full"],
        "preset_weights": {"ghost": 50, "full": 20, "melt": 15, "trails": 10, "glitch": 5},
        "extra_args": ["--mode", "dissolve"],
    },
    "color-bath": {
        "weight": 15,
        "preset_flag": "--preset",
        "presets": ["red-film", "ochre", "teal-moody", "amber", "blue-hour",
                    "rose", "sepia", "emerald", "magenta-dusk", "cyan-ice"],
        "preset_weights": {"red-film": 25, "ochre": 20, "teal-moody": 20,
                           "amber": 15, "rose": 10, "blue-hour": 10},
        "extra_args": ["--analog", "--preserve-shadows"],
    },
    # silhouette-backdrop: paused — suitability filter (pose/clothing) unstable with parallel MediaPipe
    # botanical-overlay: dropped — petals landed on clothes, aesthetic didn't work
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

_recent_models = deque(maxlen=15)  # don't repeat these models
_recent_photos = deque(maxlen=60)  # don't repeat these photos

def pick_random_photo(avoid_model=None):
    """Pick a random (model_name, Path) with model-diversity enforcement.

    Two-stage sampling:
      1. Pick a model uniformly among models with photos, excluding the
         last ~15 used. This forces variety across sessions — no model
         repeats until ~15 other models have been tried.
      2. Pick a photo within the chosen model, 70% from Processed / 30%
         from Unprocessed, also avoiding the last ~60 photos used.

    The per-model uniform pick (not by photo count) keeps big folders like
    Neta/Gali & Henia/Omry from dominating the queue.
    """
    candidates = []
    for md in all_models():
        if avoid_model and md.name == avoid_model:
            continue
        if md.name in BLACKLIST:
            continue
        if md.name in _recent_models:
            continue
        proc, unproc = list_photos_for_model(md)
        if proc or unproc:
            candidates.append((md.name, proc, unproc))
    if not candidates:
        # Everything is on cooldown; drain history and retry
        _recent_models.clear()
        return pick_random_photo(avoid_model=avoid_model)

    model_name, proc, unproc = random.choice(candidates)
    want_proc = random.random() < 0.70
    photos = (proc if want_proc else unproc) or proc or unproc
    fresh = [p for p in photos if str(p) not in _recent_photos]
    photo = random.choice(fresh or photos)

    _recent_models.append(model_name)
    _recent_photos.append(str(photo))
    return model_name, photo

def pick_ms_candidate_photo():
    """Pick a photo for motion-streak candidate mode.

    - Skip models in MS_BLACKLIST
    - Skip photos already in MS_REJECTS or already saved as candidates
    - Skip recent models (avoid back-to-back same-session)
    - Weight models by MS_BOOSTS (one-time bump when a model yields a 'good')
    """
    saved_paths = ms_candidate_paths()
    candidates = []
    for md in all_models():
        if md.name in MS_BLACKLIST:
            continue
        if md.name in _recent_models:
            continue
        proc, unproc = list_photos_for_model(md)
        photos = [p for p in (proc + unproc)
                  if str(p) not in MS_REJECTS and str(p) not in saved_paths]
        if photos:
            candidates.append((md.name, photos))
    if not candidates:
        _recent_models.clear()
        # Try again once without recent-model filter; if still empty, give up.
        for md in all_models():
            if md.name in MS_BLACKLIST:
                continue
            proc, unproc = list_photos_for_model(md)
            photos = [p for p in (proc + unproc)
                      if str(p) not in MS_REJECTS and str(p) not in saved_paths]
            if photos:
                candidates.append((md.name, photos))
        if not candidates:
            return None, None

    names = [c[0] for c in candidates]
    weights = [MS_BOOSTS.get(n, 1.0) for n in names]
    chosen_name = random.choices(names, weights=weights, k=1)[0]
    photos = dict(candidates)[chosen_name]
    fresh = [p for p in photos if str(p) not in _recent_photos]
    photo = random.choice(fresh or photos)
    _recent_models.append(chosen_name)
    _recent_photos.append(str(photo))
    return chosen_name, photo


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
                      "failures": 0, "cost_today": 0.0, "cost_date": ""}
        self.backpressure_notified = False
        self.load()

    def load(self):
        if not STATE_PATH.is_file():
            return
        try:
            data = json.loads(STATE_PATH.read_text())
            for item in data.get("queue", []):
                self.queue.append(item)
            for item in data.get("priority", []):
                self.priority.append(item)
            self.stats.update(data.get("stats", {}))
        except Exception:
            pass

    def save(self):
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        data = {
            "queue": list(self.queue),
            "priority": list(self.priority),
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
            self._accrue_cost(item.get("tool"))
            self.save()

    def _accrue_cost(self, tool_name):
        today = datetime.now(ISRAEL_TZ).strftime("%Y-%m-%d")
        if self.stats.get("cost_date") != today:
            self.stats["cost_date"] = today
            self.stats["cost_today"] = 0.0
        self.stats["cost_today"] = round(
            self.stats.get("cost_today", 0.0) + TOOL_COST.get(tool_name, 0.01), 4)

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
    # Intermediates go to shared/tool-outputs-intermediates; finals are pinned in tools to shared/finals/
    cmd += ["--output-to", "local", "--local-output-dir", str(SHARED_DIR / "tool-outputs-intermediates")]
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

def ms_candidate_loop():
    """Continuously generate motion-streak B&W previews for candidate review."""
    while STATE.stats.get("running", True):
        try:
            if STATE.stats.get("paused"):
                time.sleep(2)
                continue
            pending = STATE.pending_count()
            if pending >= 15:
                if not STATE.backpressure_notified:
                    push_text("Candidate queue full", f"{pending} waiting for review")
                    STATE.backpressure_notified = True
                while STATE.pending_count() >= 10 and STATE.stats.get("running", True):
                    if STATE.stats.get("paused"):
                        break
                    time.sleep(3)
                if STATE.pending_count() < 10:
                    STATE.backpressure_notified = False
                continue

            model_name, photo = pick_ms_candidate_photo()
            if not photo:
                print("[ms] no candidate photos available, sleeping", flush=True)
                time.sleep(15)
                continue
            job_id = uuid.uuid4().hex[:8]
            generate_ms_candidate(job_id, model_name, photo)
        except Exception as e:
            print(f"[ms] loop error: {e}", flush=True)
            STATE.stats["failures"] = STATE.stats.get("failures", 0) + 1
            time.sleep(5)


def generate_ms_candidate(job_id, model_name, photo):
    """Run motion-streak --up-to-step 1 (B&W only) and queue the result."""
    script = WORKFLOWS_DIR / "motion-streak.py"
    cmd = [PYTHON, str(script), "--source", str(photo), "--up-to-step", "1"]
    cmd_str = " ".join(f"'{c}'" if " " in c else c for c in cmd)
    src_stem = Path(photo).stem
    print(f"[ms] [{job_id}] {model_name}/{src_stem}", flush=True)
    start_ts = time.time()
    try:
        child_env = {**os.environ, "NOTIFY_DISABLE_IMAGE": "1"}
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                             env=child_env)
    except subprocess.TimeoutExpired:
        print(f"[ms] [{job_id}] TIMEOUT", flush=True)
        return None
    if res.returncode != 0:
        print(f"[ms] [{job_id}] FAILED rc={res.returncode}", flush=True)
        print((res.stderr or "")[-400:], flush=True)
        STATE.stats["failures"] = STATE.stats.get("failures", 0) + 1
        return None

    out_path = find_output_after(start_ts, hint_model=model_name, hint_source=src_stem)
    if out_path is None or not out_path.is_file():
        print(f"[ms] [{job_id}] could not locate output", flush=True)
        return None

    item = {
        "id": job_id,
        "source": str(photo),
        "model": model_name,
        "tool": "motion-streak",
        "preset": "bw-preview",
        "artifact": None,
        "seed": None,
        "command": cmd_str,
        "output": str(out_path),
        "generated_at": now_str(),
        "status": "pending",
    }
    STATE.add(item, priority=False)
    print(f"[ms] [{job_id}] OK -> {out_path.name}", flush=True)
    return item


def generate_one(job_id, tool_name, model_name, photo, preset, artifact,
                 seed, priority=False):
    """Run one tool and add to queue. Returns the item or None on failure."""
    cmd, cmd_str = build_command(tool_name, photo, preset, artifact)
    src_stem = Path(photo).stem
    print(f"[gen] [{job_id}] {tool_name} / {preset} / {artifact} on {model_name}/{src_stem}", flush=True)
    start_ts = time.time()
    try:
        child_env = {**os.environ, "NOTIFY_DISABLE_IMAGE": "1"}
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             env=child_env)
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
    if MODE == "candidates":
        return render_template("candidates.html")
    return render_template("gallery.html")

@app.route("/candidates")
def candidates_page():
    return render_template("candidates.html")

@app.route("/gallery")
def gallery_page():
    return render_template("gallery.html")

@app.route("/tree")
def tree_page():
    return render_template("tree.html")

@app.route("/api/tree")
def api_tree():
    tree_path = Path("~/.openclaw/workspace/shared/tools_tree.md").expanduser()
    if not tree_path.is_file():
        return jsonify({"mermaid": "", "updated": None, "error": "tools_tree.md not found"})
    text = tree_path.read_text()
    import re as _re
    m = _re.search(r"```mermaid\n(.*?)\n```", text, _re.DOTALL)
    mmd = m.group(1) if m else ""
    u = _re.search(r"Last updated:\s*(\S+)", text)
    return jsonify({"mermaid": mmd, "updated": u.group(1) if u else None})

@app.route("/api/candidates/list")
def api_ms_list():
    return jsonify({"candidates": load_ms_candidates(),
                    "rejects": sorted(MS_REJECTS),
                    "blacklist": sorted(MS_BLACKLIST),
                    "boosts": MS_BOOSTS})

@app.route("/api/candidates/<item_id>/good", methods=["POST"])
def api_ms_good(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    source = it["source"]
    model = it.get("model") or ""
    entries = load_ms_candidates()
    if not any(e.get("source") == source for e in entries):
        entries.append({
            "source": source,
            "filename": Path(source).name,
            "model": model,
            "marked_at": now_str(),
        })
        save_ms_candidates(entries)
    # One-time boost per session
    if model and model not in MS_BOOSTS:
        MS_BOOSTS[model] = 1.5
    # Remove preview from queue + delete preview file (path is saved in json)
    try:
        Path(it["output"]).unlink()
    except OSError:
        pass
    STATE.remove(item_id)
    STATE.stats["liked"] = STATE.stats.get("liked", 0) + 1
    STATE.save()
    return jsonify({"ok": True, "saved": len(entries), "boost": MS_BOOSTS.get(model)})

@app.route("/api/candidates/<item_id>/bad", methods=["POST"])
def api_ms_bad(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    MS_REJECTS.add(it["source"])
    ms_save_rejects(MS_REJECTS)
    try:
        Path(it["output"]).unlink()
    except OSError:
        pass
    STATE.remove(item_id)
    STATE.stats["disliked"] = STATE.stats.get("disliked", 0) + 1
    STATE.save()
    return jsonify({"ok": True, "rejects": len(MS_REJECTS)})

@app.route("/api/candidates/<item_id>/bad-session", methods=["POST"])
def api_ms_bad_session(item_id):
    it = STATE.find(item_id)
    if not it:
        abort(404)
    model = it.get("model")
    if not model:
        abort(400)
    MS_BLACKLIST.add(model)
    ms_save_blacklist(MS_BLACKLIST)
    removed = 0
    with STATE.lock:
        for lst in (STATE.priority, STATE.queue):
            i = 0
            while i < len(lst):
                cur = lst[i]
                if cur.get("model") == model:
                    try:
                        Path(cur["output"]).unlink()
                    except OSError:
                        pass
                    del lst[i]
                    removed += 1
                else:
                    i += 1
        STATE.save()
    return jsonify({"ok": True, "model": model, "removed": removed})

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

@app.route("/api/queue/<item_id>/blacklist", methods=["POST"])
def api_blacklist(item_id):
    """Blacklist the item's model (session) — future picks skip it, and
    all currently-queued items from that model are removed + deleted."""
    it = STATE.find(item_id)
    if not it:
        abort(404)
    model = it.get("model")
    if not model:
        abort(400)
    BLACKLIST.add(model)
    save_blacklist(BLACKLIST)
    removed = 0
    with STATE.lock:
        for lst in (STATE.priority, STATE.queue):
            i = 0
            while i < len(lst):
                cur = lst[i]
                if cur.get("model") == model:
                    try:
                        Path(cur["output"]).unlink()
                    except OSError:
                        pass
                    del lst[i]
                    removed += 1
                else:
                    i += 1
        STATE.save()
    return jsonify({"ok": True, "model": model, "removed": removed})


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
    p.add_argument("--mode", choices=["gallery", "candidates"], default="gallery",
                   help="gallery=default stylized queue; candidates=motion-streak candidate picker")
    args = p.parse_args()
    global MODE
    MODE = args.mode

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
        if MODE == "candidates":
            t = threading.Thread(target=ms_candidate_loop, daemon=True)
        else:
            t = threading.Thread(target=generation_loop, args=(allowed,), daemon=True)
        t.start()

    print(f"[flask] serving on http://0.0.0.0:{args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
