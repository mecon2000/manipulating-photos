#!/usr/bin/env python3
"""Replicate fofr/become-image — face + style ref → identity-preserving stylization.

Replaces the fofr/style-transfer + face-swap chain. fofr/become-image uses
InstantID (face) + IPAdapter (style image) + depth ControlNet in one pass,
so identity is preserved natively and there's no "no face detected" failure
mode at the swap stage.

Single:
  become_image_replicate.py --source PHOTO --style STYLE_REF
Batch (every source × every style):
  become_image_replicate.py --batch \\
      --source-dir ~/.openclaw/workspace/shared/candidates-for-motion-streak \\
      --style-dir  ~/.openclaw/workspace/shared/0010x0010/cleaned

Reads REPLICATE_API_TOKEN from env or ~/sol/.env.
"""
import argparse, os, sys, time, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT_DIR = Path("~/.openclaw/workspace/shared/become-image-finals").expanduser()


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


MODEL = "fofr/become-image:8d0b076a2aff3904dfcec3253c778e0310a68f78483c4699c7fd800f3051d2b3"


def _to_data_uri(path):
    import base64
    suf = Path(path).suffix.lower()
    mime = {".png": "image/png", ".webp": "image/webp"}.get(suf, "image/jpeg")
    b = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


def _run_once(replicate, source, style, prompt, neg_prompt,
              instant_id_strength, image_to_become_strength,
              denoising_strength, depth_strength, prompt_strength, seed):
    # become-image inspects file extension; pass data URIs so the container
    # gets a real .jpg/.png file when Replicate decodes them
    out = replicate.run(MODEL, input={
        "image": _to_data_uri(source),
        "image_to_become": _to_data_uri(style),
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "instant_id_strength": instant_id_strength,
        "image_to_become_strength": image_to_become_strength,
        "denoising_strength": denoising_strength,
        "control_depth_strength": depth_strength,
        "prompt_strength": prompt_strength,
        "number_of_images": 1,
        "disable_safety_checker": True,
        "seed": seed if seed is not None else int(time.time()) % 100000,
    })
    item = out[0] if isinstance(out, list) and out else out
    if hasattr(item, "read"):
        return item.read()
    if isinstance(item, str):
        with urllib.request.urlopen(item, timeout=60) as r:
            return r.read()
    raise RuntimeError(f"unexpected output type: {type(out)}")


def run_one(source, style, prompt, neg_prompt, instant_id_strength,
            image_to_become_strength, denoising_strength, depth_strength,
            prompt_strength, seed=None, max_retries=8):
    import replicate
    for attempt in range(max_retries):
        try:
            return _run_once(replicate, source, style, prompt, neg_prompt,
                             instant_id_strength, image_to_become_strength,
                             denoising_strength, depth_strength, prompt_strength, seed)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "throttled" in msg.lower() or "rate limit" in msg.lower():
                import re as _re
                m = _re.search(r"resets in ~(\d+)s", msg)
                wait = int(m.group(1)) + 1 if m else min(60, 2 ** attempt + 5)
                time.sleep(wait); continue
            raise
    raise RuntimeError(f"max retries exceeded for {source} × {style}")


def short(p):
    s = Path(p).stem
    return s.replace(" ", "_").replace("-", "_")[:60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source")
    p.add_argument("--style")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--source-dir", default="~/.openclaw/workspace/shared/candidates-for-motion-streak")
    p.add_argument("--style-dir",  default="~/.openclaw/workspace/shared/0010x0010/cleaned")
    p.add_argument("--prompt", default="long-exposure black and white motion-blur dance photograph, dramatic side lighting")
    p.add_argument("--negative-prompt", default="lowres, blurry, deformed, distorted face, mutation, ugly")
    p.add_argument("--instant-id-strength",       type=float, default=1.0,
                   help="face preservation (0-2; higher = more faithful identity)")
    p.add_argument("--image-to-become-strength",  type=float, default=0.75,
                   help="style strength (0-1; higher = stronger style)")
    p.add_argument("--denoising-strength",        type=float, default=1.0,
                   help="how much of source to keep (1=full restyle)")
    p.add_argument("--depth-strength",            type=float, default=0.8,
                   help="structure preservation via depth ControlNet")
    p.add_argument("--prompt-strength",           type=float, default=2.0,
                   help="CFG scale for text prompt")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    token = load_token()
    if not token:
        print("REPLICATE_API_TOKEN missing", file=sys.stderr); sys.exit(2)
    os.environ["REPLICATE_API_TOKEN"] = token

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    if args.batch:
        sd  = Path(args.source_dir).expanduser()
        std = Path(args.style_dir).expanduser()
        sources = [p for p in sorted(sd.iterdir())
                   if p.suffix.lower() in {".jpg",".jpeg",".png"} and not p.name.startswith("_")]
        styles  = [p for p in sorted(std.iterdir())
                   if p.suffix.lower() in {".jpg",".jpeg",".png"} and not p.name.startswith("_")]
        if not sources or not styles:
            print(f"need files in {sd} and {std}", file=sys.stderr); sys.exit(2)
        cost = len(sources)*len(styles)*0.01   # rough estimate
        print(f"batch: {len(sources)} × {len(styles)} = {len(sources)*len(styles)} runs  (~${cost:.2f} est)")
        for s in sources:
            for st in styles:
                pairs.append((s, st))
    else:
        if not (args.source and args.style):
            print("need --source and --style (or --batch)", file=sys.stderr); sys.exit(2)
        pairs.append((Path(args.source), Path(args.style)))

    def process(pair):
        src, style = pair
        tag = f"{short(src)}__style_{short(style)}"
        out_path = out_dir / f"{tag}.jpg"
        if out_path.exists():
            return ("skip", tag, 0.0)
        t0 = time.time()
        try:
            blob = run_one(str(src), str(style), args.prompt, args.negative_prompt,
                           args.instant_id_strength, args.image_to_become_strength,
                           args.denoising_strength, args.depth_strength,
                           args.prompt_strength, args.seed)
        except Exception as e:
            return ("fail", tag, str(e)[:160])
        out_path.write_bytes(blob)
        meta = {
            "source": str(src), "style": str(style), "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "instant_id_strength": args.instant_id_strength,
            "image_to_become_strength": args.image_to_become_strength,
            "denoising_strength": args.denoising_strength,
            "depth_strength": args.depth_strength,
            "prompt_strength": args.prompt_strength,
            "seed": args.seed, "tool": "fofr/become-image",
            "elapsed_s": round(time.time() - t0, 1),
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return ("ok", tag, round(time.time() - t0, 1))

    workers = max(1, args.workers if len(pairs) > 1 else 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, tag, info in pool.map(process, pairs):
            if status == "ok":   print(f"  ✔ {tag}  ({info}s)")
            elif status == "skip": print(f"skip exists: {tag}")
            else:                print(f"FAIL {tag}: {info}")


if __name__ == "__main__":
    main()
