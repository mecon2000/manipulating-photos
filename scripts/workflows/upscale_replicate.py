#!/usr/bin/env python3
"""Upscale style-transfer outputs to IG-print resolution via Replicate Real-ESRGAN.

Default 4x → 1024 in becomes 4096 out. ~$0.005/image, ~3s.

Usage:
  upscale_replicate.py --source FILE.jpg
  upscale_replicate.py --batch                       # all of style-transfer-finals/
  upscale_replicate.py --batch --in-dir DIR --out-dir DIR --scale 2
"""
import argparse, os, sys, time, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_IN  = Path("~/.openclaw/workspace/shared/style-transfer-finals").expanduser()
DEFAULT_OUT = Path("~/.openclaw/workspace/shared/style-transfer-upscaled").expanduser()


def load_token():
    t = os.environ.get("REPLICATE_API_TOKEN")
    if t:
        return t
    env = Path("~/sol/.env").expanduser()
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def upscale_one(src_path, scale, max_retries=8):
    import replicate
    for attempt in range(max_retries):
        try:
            with open(src_path, "rb") as f:
                out = replicate.run(
                    "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                    input={"image": f, "scale": scale, "face_enhance": False},
                )
            item = out[0] if isinstance(out, list) and out else out
            if hasattr(item, "read"):
                return item.read()
            if isinstance(item, str):
                with urllib.request.urlopen(item, timeout=60) as r:
                    return r.read()
            raise RuntimeError(f"unexpected output: {type(out)}")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "throttled" in msg.lower():
                import re as _re
                m = _re.search(r"resets in ~(\d+)s", msg)
                wait = int(m.group(1)) + 1 if m else min(60, 2 ** attempt + 5)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"max retries exceeded for {src_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="single file")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--in-dir", default=str(DEFAULT_IN))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--scale", type=int, default=4, choices=[2, 4],
                   help="upscale factor (Real-ESRGAN supports 2 or 4)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    token = load_token()
    if not token:
        print("REPLICATE_API_TOKEN missing", file=sys.stderr); sys.exit(2)
    os.environ["REPLICATE_API_TOKEN"] = token

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        in_dir = Path(args.in_dir).expanduser()
        files = sorted(p for p in in_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    elif args.source:
        files = [Path(args.source)]
    else:
        print("need --source or --batch", file=sys.stderr); sys.exit(2)

    cost = len(files) * 0.005
    print(f"upscaling {len(files)} file(s), scale={args.scale}x  (~${cost:.2f})")

    def process(f):
        out = out_dir / f.name
        if out.exists():
            return ("skip", f.name, 0.0)
        t0 = time.time()
        try:
            blob = upscale_one(str(f), args.scale)
        except Exception as e:
            return ("fail", f.name, str(e)[:120])
        out.write_bytes(blob)
        return ("ok", f.name, round(time.time() - t0, 1))

    workers = max(1, args.workers if len(files) > 1 else 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, name, info in pool.map(process, files):
            if status == "ok":
                print(f"  ✔ {name}  ({info}s)")
            elif status == "skip":
                print(f"skip exists: {name}")
            else:
                print(f"FAIL {name}: {info}")


if __name__ == "__main__":
    main()
