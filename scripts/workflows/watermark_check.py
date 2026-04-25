#!/usr/bin/env python3
"""Detect 'UNPROCESSED'/'DO NOT PUBLISH'-type watermarks via Gemini Vision.

LR exports of unprocessed/work-in-progress JPEGs sometimes have a burned-in
watermark — corner signature, big half-transparent text across the photo, or
'unprocessed - do not publish'. Filename alone is unreliable (the same file
might or might not carry the watermark).

This script asks Gemini 2.5 Flash per image and warns you. It does not try
to clean — Ronnie re-exports clean from LR himself.

Usage:
  watermark_check.py [DIR ...]              # scan dirs (default: candidates folder)
  watermark_check.py FILE.jpg               # scan one file
  watermark_check.py --json [DIR]           # machine-readable output
  watermark_check.py --suspect-only DIR     # only print flagged files (one per line)
"""
import argparse, base64, json, os, sys, time, urllib.request
from pathlib import Path

PROMPT = (
    "Does this photograph contain a burned-in watermark text such as "
    "'UNPROCESSED', 'DO NOT PUBLISH', 'WIP', a photographer's signature/copyright "
    "text, or any other obvious overlaid label? "
    "Ignore subject names, tattoos, logos on clothing, or text that's part of the "
    "scene (a sign on a wall, etc.). "
    "Respond ONLY with strict JSON: "
    '{"has_watermark": true/false, "text": "<exact watermark text or empty>", '
    '"location": "corner/center/diagonal/top/bottom/none"}'
)


def load_key():
    k = os.environ.get("GOOGLE_API_KEY")
    if k:
        return k
    env = Path("~/sol/.env").expanduser()
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def scan_image(api_key, path, model="gemini-2.5-flash"):
    """Returns dict {has_watermark, text, location} or {error: ...}."""
    try:
        b = Path(path).read_bytes()
    except OSError as e:
        return {"error": f"read: {e}"}
    img_b64 = base64.b64encode(b).decode()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    body = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": mime, "data": img_b64}},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 256,
            "temperature": 0,
        },
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"error": f"api: {e}"}
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except Exception as e:
        return {"error": f"parse: {e}", "raw": str(data)[:200]}


def collect_files(targets):
    files = []
    for t in targets:
        p = Path(t).expanduser()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    files.append(f)
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*",
                   default=["~/.openclaw/workspace/shared/candidates-for-motion-streak"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--suspect-only", action="store_true",
                   help="print only suspect file paths, one per line")
    p.add_argument("--model", default="gemini-2.5-flash")
    args = p.parse_args()

    key = load_key()
    if not key:
        print("GOOGLE_API_KEY not found in env or ~/sol/.env", file=sys.stderr)
        sys.exit(2)

    files = collect_files(args.targets)
    results = []
    for f in files:
        r = scan_image(key, f, model=args.model)
        results.append({"path": str(f), **r})
        if not args.json and not args.suspect_only:
            if r.get("error"):
                print(f"  ❓ {f.name}  ({r['error']})")
            elif r.get("has_watermark"):
                print(f"  ⚠️  {f.name}  '{r.get('text','')}' @ {r.get('location','?')}")
            else:
                print(f"  ✅ {f.name}")
        time.sleep(0.05)   # gentle pacing for Flash

    if args.json:
        print(json.dumps(results, indent=2))
        return

    suspects = [r for r in results if r.get("has_watermark")]
    if args.suspect_only:
        for r in suspects:
            print(r["path"])
        return

    print()
    if suspects:
        print(f"⚠️  {len(suspects)} of {len(results)} flagged — re-export from LR before pipeline.")
    else:
        print(f"✅ {len(results)} clean.")


if __name__ == "__main__":
    main()
