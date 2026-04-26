#!/usr/bin/env python3
"""Build literary quote DB with semantic embeddings (no mood enum).

Source:  PoetryDB (poetrydb.org) — public-domain poets, free, no auth.
Embed:   sentence-transformers/all-MiniLM-L6-v2 (local CPU, free, ~3s for 5k lines).

Output:
  scripts/workflows/literary_quotes.json   — list of {text, author, title}
  scripts/workflows/literary_quotes.npy    — float32 (N, 384) embeddings, same order
  scripts/workflows/literary_quotes.meta.json — {model, dim, count}

Runtime use (in text_overlay.py):
  - Embed image's free-text mood description (also via all-MiniLM)
  - Cosine-NN over the .npy → top-K nearest lines → pick one randomly

Usage:
  build_quote_db.py [--target-lines 10000]
"""
import argparse, json, os, random, re, sys, time, urllib.request
from pathlib import Path

OUT_JSON = Path(__file__).resolve().parent / "literary_quotes.json"
OUT_NPY  = Path(__file__).resolve().parent / "literary_quotes.npy"
OUT_META = Path(__file__).resolve().parent / "literary_quotes.meta.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def fetch_random_poems(n_per_call=50, max_total=4000):
    """Hit PoetryDB /random/N until we have ~max_total unique poems."""
    seen = set()
    poems = []
    misses = 0
    UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    while len(poems) < max_total and misses < 5:
        try:
            req = urllib.request.Request(
                f"https://poetrydb.org/random/{n_per_call}",
                headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                batch = json.loads(r.read())
        except Exception as e:
            print(f"[poetrydb] fetch fail: {e}, retrying in 5s")
            time.sleep(5); continue
        new = 0
        for p in batch:
            key = (p.get("title", ""), p.get("author", ""))
            if key in seen: continue
            seen.add(key); poems.append(p); new += 1
        if new == 0:
            misses += 1
            print(f"[poetrydb] no new poems this round (miss {misses}/5)")
        else:
            misses = 0
            print(f"[poetrydb] +{new} new (total {len(poems)})")
        time.sleep(0.4)
    return poems


def extract_lines(poems, min_words=5, max_words=14):
    """Pick gem-quality single lines."""
    lines = []
    for p in poems:
        author = p.get("author", "")
        title = p.get("title", "")
        for raw in p.get("lines", []):
            line = raw.strip()
            if not line or len(line) < 8: continue
            # Must end with sentence-terminating punctuation or em-dash/quote
            if not re.search(r"[.!?—\";]\s*$", line): continue
            words = line.split()
            if not (min_words <= len(words) <= max_words): continue
            # Skip lowercase fragment continuations (e.g. starting mid-sentence)
            if line[0].islower() and "," in line[:8]: continue
            # Skip all-caps shouting
            if sum(1 for c in line if c.isupper()) > len(line) * 0.5: continue
            # Strip trailing decoration
            cleaned = line.rstrip(",;:")
            lines.append({"text": cleaned, "author": author, "title": title})
    # dedupe by text
    seen, out = set(), []
    for l in lines:
        if l["text"] in seen: continue
        seen.add(l["text"]); out.append(l)
    return out


def embed_lines(lines, model_name=EMBED_MODEL, batch_size=128):
    """Local CPU embedding via sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    print(f"loading {model_name} …")
    m = SentenceTransformer(model_name)
    texts = [l["text"] for l in lines]
    print(f"embedding {len(texts)} lines (batch={batch_size}) …")
    t0 = time.time()
    vecs = m.encode(texts, batch_size=batch_size, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=True)
    print(f"  done in {time.time()-t0:.1f}s, shape={vecs.shape}, dtype={vecs.dtype}")
    return vecs.astype("float32")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-lines", type=int, default=10000)
    p.add_argument("--poems",        type=int, default=4000,
                   help="ceiling on poems to pull from PoetryDB")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"target: ~{args.target_lines} lines from up to {args.poems} poems")
    poems = fetch_random_poems(max_total=args.poems)
    print(f"got {len(poems)} unique poems")

    lines = extract_lines(poems)
    print(f"extracted {len(lines)} unique gem-quality lines")
    if len(lines) > args.target_lines:
        random.seed(42)
        random.shuffle(lines)
        lines = lines[:args.target_lines]
        print(f"trimmed to {len(lines)}")

    if args.dry_run:
        for l in random.sample(lines, min(15, len(lines))):
            print(f"  {l['text'][:70]:72}  ({l['author']})")
        return

    import numpy as np
    vecs = embed_lines(lines)
    np.save(OUT_NPY, vecs)
    OUT_JSON.write_text(json.dumps(lines, ensure_ascii=False, indent=2))
    OUT_META.write_text(json.dumps({
        "model": EMBED_MODEL, "dim": int(vecs.shape[1]),
        "count": len(lines),
    }, indent=2))
    print(f"\nwrote:")
    print(f"  {OUT_JSON}  ({OUT_JSON.stat().st_size//1024} KB)")
    print(f"  {OUT_NPY}   ({OUT_NPY.stat().st_size//1024} KB)")
    print(f"  {OUT_META}")


if __name__ == "__main__":
    main()
