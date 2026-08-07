"""Eyes service — VLM describe/critique with NSFW-aware backend routing.

One entry point: describe(ref_or_path, question, region). Routing:
- local Ollama VLM (gemma3:4b) — the DEFAULT. Never refuses, never leaves
  the machine. Used for anything explicit/bold/unknown.
- Gemini 2.5 Flash — only when the source is catalog-confirmed 'safe'
  (it blocks shibari/nudity; documented in CLAUDE.md).
The caller never needs to know which backend answered.

NSFW level comes from the photo catalog's `boldness` tag (photo-level, then
set-level fallback). Studio candidates are often renamed copies, so lookups
may miss — unknown routes local, which is always correct, just less eloquent.

8GB VRAM note (measured 2026-08-08): desktop idles ~4.5GB, gemma3:4b adds
~2.9GB → 7.5/8.2GB while loaded. Ollama's default 4-min keep_alive plus the
explicit keep_alive below keeps the window short so SAM 2 (tiny) coexists.
"""
import base64
import io
import json
import os
import sqlite3

import requests
from PIL import Image

from . import cache

CATALOG_DB = os.path.expanduser("~/gitrep/photo-catalogging/data/photo-catalog.db")
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_KEEP_ALIVE = "3m"
GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_QUESTION = ("Describe this photograph like a photographer talking to another "
                    "photographer: subject, pose, lighting, palette, mood, composition, "
                    "where the eye goes. 2-3 tight sentences. It is the artist's own "
                    "consented artwork — be direct, no hedging.")


def nsfw_level(source_path: str) -> str:
    """safe | suggestive | explicit | unknown — from the catalog's boldness tags."""
    name = os.path.basename(str(source_path))
    try:
        db = sqlite3.connect(f"file:{CATALOG_DB}?mode=ro", uri=True)
        row = db.execute(
            "SELECT pt.value, p.set_id FROM photos p "
            "LEFT JOIN photo_tags pt ON pt.photo_id = p.id AND pt.dimension='boldness' "
            "WHERE p.filename = ? LIMIT 1", (name,)).fetchone()
        if row is None:
            return "unknown"
        value, set_id = row
        if value is None and set_id is not None:
            r2 = db.execute("SELECT value FROM tags WHERE set_id=? AND dimension='boldness' "
                            "LIMIT 1", (set_id,)).fetchone()
            value = r2[0] if r2 else None
        if value in ("safe", "suggestive"):
            return value
        if value in ("bold", "explicit"):
            return "explicit"
        return "unknown"
    except sqlite3.Error:
        return "unknown"


def _load_image(ref_or_path: str, region=None, max_edge: int = 1024) -> bytes:
    p = cache.object_path(ref_or_path) or ref_or_path
    img = Image.open(p).convert("RGB")
    if region:  # normalized [x, y, w, h] crop, padded a bit for context
        x, y, w, h = region
        W, H = img.size
        pad_x, pad_y = w * 0.25 * W, h * 0.25 * H
        img = img.crop((max(0, x * W - pad_x), max(0, y * H - pad_y),
                        min(W, (x + w) * W + pad_x), min(H, (y + h) * H + pad_y)))
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _ollama(image_bytes: bytes, question: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=180, json={
        "model": OLLAMA_MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": [{"role": "user", "content": question,
                      "images": [base64.b64encode(image_bytes).decode()]}]})
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _gemini(image_bytes: bytes, question: str) -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("no GOOGLE_API_KEY")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
        f":generateContent?key={key}", timeout=60,
        json={"contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(image_bytes).decode()}},
            {"text": question}]}]})
    r.raise_for_status()
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"gemini blocked/empty: {json.dumps(data)[:200]}")


def describe(ref_or_path: str, question: str | None = None, region=None,
             source_path: str | None = None) -> dict:
    """Returns {"text", "backend", "level"}. source_path drives NSFW routing
    (pass the session's original photo; outputs inherit its level)."""
    question = question or DEFAULT_QUESTION
    level = nsfw_level(source_path or ref_or_path)
    image_bytes = _load_image(ref_or_path, region)
    if level == "safe":
        try:
            return {"text": _gemini(image_bytes, question),
                    "backend": "gemini", "level": level}
        except (requests.RequestException, RuntimeError):
            pass  # fall through to local
    return {"text": _ollama(image_bytes, question),
            "backend": "ollama", "level": level}
