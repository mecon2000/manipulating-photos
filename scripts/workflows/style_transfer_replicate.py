#!/usr/bin/env python3
"""Run fofr/style-transfer on Replicate: source photo + style ref → stylized output.

Uses IPAdapter Plus + DreamShaperXL Lightning + depth ControlNet on the
Replicate side. NSFW-friendly (DreamShaperXL community model). ~$0.0063/run, 7s.

Single run:
  style_transfer_replicate.py --source PHOTO --style STYLE_REF [--prompt TEXT]

Batch grid (every source × every style):
  style_transfer_replicate.py --batch \\
      --source-dir ~/.openclaw/workspace/shared/candidates-for-motion-streak \\
      --style-dir ~/.openclaw/workspace/shared/0010x0010/cropped

Auth:
  Reads REPLICATE_API_TOKEN from env or ~/sol/.env
"""
import argparse, os, sys, time, json, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT_DIR = Path("~/.openclaw/workspace/shared/style-transfer-finals").expanduser()


def load_token():
    t = os.environ.get("REPLICATE_API_TOKEN")
    if t:
        return t
    env_path = Path("~/sol/.env").expanduser()
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def run_one(source, style, prompt, denoising_strength, depth_strength, seed=None,
            max_retries=8):
    import replicate
    for attempt in range(max_retries):
        try:
            return _run_once(replicate, source, style, prompt,
                             denoising_strength, depth_strength, seed)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "throttled" in msg.lower() or "rate limit" in msg.lower():
                # parse "resets in ~Ns" if present, else exponential backoff
                import re as _re
                m = _re.search(r"resets in ~(\d+)s", msg)
                wait = int(m.group(1)) + 1 if m else min(60, 2 ** attempt + 5)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"max retries exceeded for {source} × {style}")


def _run_once(replicate, source, style, prompt, denoising_strength, depth_strength, seed):
    with open(source, "rb") as fs, open(style, "rb") as fst:
        out = replicate.run(
            "fofr/style-transfer:f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0",
            input={
                "style_image": fst,
                "structure_image": fs,
                "prompt": prompt,
                "structure_depth_strength": depth_strength,
                "structure_denoising_strength": denoising_strength,
                "model": "fast",
                "number_of_images": 1,
                "output_format": "jpg",
                "output_quality": 92,
                "seed": seed if seed is not None else int(time.time()) % 100000,
            },
        )
    # `out` is a list of file objects (newer SDK) or URLs (older). Normalize:
    if isinstance(out, list) and out:
        item = out[0]
    else:
        item = out
    if hasattr(item, "read"):
        return item.read()
    if isinstance(item, str):
        with urllib.request.urlopen(item, timeout=60) as r:
            return r.read()
    raise RuntimeError(f"unexpected replicate output: {type(out)}")


def short(p):
    s = Path(p).stem
    s = s.replace(" ", "_").replace("-", "_")
    return s[:60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="path to one source photo")
    p.add_argument("--style", help="path to one style reference photo")
    p.add_argument("--batch", action="store_true",
                   help="run every source × every style")
    p.add_argument("--source-dir", default="~/.openclaw/workspace/shared/candidates-for-motion-streak")
    p.add_argument("--style-dir", default="~/.openclaw/workspace/shared/0010x0010/cropped")
    p.add_argument("--prompt", default="long-exposure black and white motion-blur dance photograph, dramatic side lighting")
    p.add_argument("--denoising-strength", type=float, default=0.65,
                   help="how much the style overrides the source (0=keep source, 1=full style)")
    p.add_argument("--depth-strength", type=float, default=1.0,
                   help="how strongly the source structure (depth) constrains the output")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--workers", type=int, default=8,
                   help="parallel workers for batch mode (paid account: 8 fine)")
    args = p.parse_args()

    token = load_token()
    if not token:
        print("REPLICATE_API_TOKEN not found in env or ~/sol/.env", file=sys.stderr)
        sys.exit(2)
    os.environ["REPLICATE_API_TOKEN"] = token

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    if args.batch:
        sd = Path(args.source_dir).expanduser()
        std = Path(args.style_dir).expanduser()
        sources = [p for p in sorted(sd.iterdir())
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        styles = [p for p in sorted(std.iterdir())
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                  and not p.name.startswith("_")]
        if not sources or not styles:
            print(f"need files in both {sd} and {std}", file=sys.stderr)
            sys.exit(2)
        print(f"batch: {len(sources)} sources × {len(styles)} styles = {len(sources)*len(styles)} runs"
              f"  (~${len(sources)*len(styles)*0.0063:.2f})")
        for s in sources:
            for st in styles:
                pairs.append((s, st))
    else:
        if not (args.source and args.style):
            print("need --source and --style (or --batch)", file=sys.stderr)
            sys.exit(2)
        pairs.append((Path(args.source), Path(args.style)))

    def process(pair):
        src, style = pair
        tag = f"{short(src)}__style_{short(style)}"
        out_path = out_dir / f"{tag}.jpg"
        if out_path.exists():
            return ("skip", tag, 0.0)
        t0 = time.time()
        try:
            blob = run_one(str(src), str(style), args.prompt,
                           args.denoising_strength, args.depth_strength, args.seed)
        except Exception as e:
            return ("fail", tag, str(e)[:120])
        out_path.write_bytes(blob)
        meta = {
            "source": str(src), "style": str(style), "prompt": args.prompt,
            "denoising_strength": args.denoising_strength,
            "depth_strength": args.depth_strength,
            "seed": args.seed, "tool": "fofr/style-transfer",
            "elapsed_s": round(time.time() - t0, 1),
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return ("ok", tag, round(time.time() - t0, 1))

    workers = max(1, args.workers if len(pairs) > 1 else 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, tag, info in pool.map(process, pairs):
            if status == "ok":
                print(f"  ✔ {tag}  ({info}s)")
            elif status == "skip":
                print(f"skip exists: {tag}")
            else:
                print(f"FAIL {tag}: {info}")


if __name__ == "__main__":
    main()
