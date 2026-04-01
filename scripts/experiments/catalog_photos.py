#!/usr/bin/env python3
"""
Photo Catalog Builder for ron.p.wilder Instagram
- Scans gdrive folders recursively
- Creates thumbnails
- Builds SQLite catalog
- Generates HTML gallery
"""

import os
import json
import sqlite3
import base64
import argparse
from pathlib import Path
from PIL import Image

GDRIVE = Path("/home/openclaw/gdrive")
CATALOG_DIR = Path("/home/openclaw/.openclaw/workspace/catalog")
THUMBS_DIR = CATALOG_DIR / "thumbs"
DB_PATH = CATALOG_DIR / "photos.db"
GALLERY_PATH = CATALOG_DIR / "gallery.html"

THUMB_SIZE = (250, 250)
JPEG_QUALITY = 75
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            orig_path TEXT NOT NULL UNIQUE,
            thumb_path TEXT,
            width INTEGER,
            height INTEGER,
            tags TEXT,
            caption TEXT,
            status TEXT DEFAULT 'pending',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_folder ON photos(folder)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON photos(status)")
    conn.commit()
    return conn


def make_thumbnail(orig_path: Path, thumb_path: Path) -> tuple | None:
    """Create thumbnail. Returns (width, height) of original or None on error."""
    try:
        with Image.open(str(orig_path)) as img:
            orig_w, orig_h = img.size
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(thumb_path), "JPEG", quality=JPEG_QUALITY, optimize=True)
            return orig_w, orig_h
    except Exception as e:
        print(f"  ⚠️  {orig_path.name}: {e}")
        return None


def scan_and_build(conn: sqlite3.Connection, limit: int = 0):
    """Scan gdrive recursively and build catalog."""
    print(f"Scanning {GDRIVE} recursively...")

    all_images = []
    io_errors = []

    def handle_error(e):
        io_errors.append(str(e))

    for root, dirs, files in os.walk(str(GDRIVE), onerror=handle_error):
        dirs.sort()
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                all_images.append(Path(root) / fname)

    if io_errors:
        print(f"  ⚠️  {len(io_errors)} I/O errors (skipped those paths)")

    print(f"Found {len(all_images)} images total")

    total_new = 0
    total_skip = 0
    last_top = None

    for img_path in all_images:
        rel = img_path.relative_to(GDRIVE)
        top_folder = rel.parts[0]
        folder_key = str(rel.parent)  # e.g. "Adi Levi/01 Morin, Yad Rambam"

        if top_folder != last_top:
            if last_top is not None:
                conn.commit()
            print(f"  📁 {top_folder}")
            last_top = top_folder

        existing = conn.execute(
            "SELECT id FROM photos WHERE orig_path = ?", (str(img_path),)
        ).fetchone()
        if existing:
            total_skip += 1
            continue

        safe_key = folder_key.replace("/", "__").replace(" ", "_").replace("\\", "__")
        thumb_path = THUMBS_DIR / safe_key / f"{img_path.stem}.jpg"

        dims = make_thumbnail(img_path, thumb_path)
        w, h = dims if dims else (0, 0)

        conn.execute("""
            INSERT OR IGNORE INTO photos
            (folder, filename, orig_path, thumb_path, width, height)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (folder_key, img_path.name, str(img_path),
              str(thumb_path) if dims else None, w, h))

        total_new += 1
        if limit and total_new >= limit:
            conn.commit()
            print(f"\n  Reached limit of {limit}")
            print(f"✅ Done: {total_new} new, {total_skip} skipped")
            return total_new

    conn.commit()
    print(f"\n✅ Done: {total_new} new, {total_skip} skipped")
    return total_new


def generate_gallery(conn: sqlite3.Connection, filter_status: str = None, limit: int = 500):
    """Generate HTML gallery for browsing."""
    query = "SELECT id, folder, filename, thumb_path, tags, status, notes FROM photos"
    params = []
    if filter_status:
        query += " WHERE status = ?"
        params.append(filter_status)
    query += " ORDER BY folder, filename LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()

    # Group by top-level folder
    top_folders: dict[str, list] = {}
    for row in rows:
        fid, folder, filename, thumb_path, tags, status, notes = row
        top = folder.split("/")[0] if "/" in folder else folder
        top_folders.setdefault(top, []).append({
            "id": fid, "folder": folder, "filename": filename,
            "thumb": thumb_path,
            "tags": json.loads(tags) if tags else [],
            "status": status or "pending",
            "notes": notes or ""
        })

    folder_sections = []
    for top_name, photos in top_folders.items():
        cards = []
        for p in photos:
            tag_html = " ".join(f'<span class="tag">{t}</span>' for t in p["tags"])
            sc = p["status"]
            sub = p["folder"].replace(top_name, "").lstrip("/")

            img_src = ""
            if p["thumb"] and Path(p["thumb"]).exists():
                try:
                    with open(p["thumb"], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    img_src = f"data:image/jpeg;base64,{b64}"
                except:
                    pass
            if not img_src:
                img_src = "data:image/gif;base64,R0lGODlhAQABAIAAAMLCwgAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw=="

            cards.append(f"""
            <div class="photo-card status-{sc}" data-id="{p['id']}">
                <img src="{img_src}" alt="{p['filename']}" loading="lazy">
                <div class="info">
                    {f'<div class="subfolder">{sub}</div>' if sub else ''}
                    <div class="filename" title="{p['filename']}">{p['filename'][:28]}</div>
                    <div class="tags">{tag_html}</div>
                    <div class="status-badge {sc}">{sc}</div>
                    {f'<div class="notes">{p["notes"]}</div>' if p["notes"] else ""}
                </div>
            </div>""")

        folder_sections.append(f"""
        <section class="folder">
            <h2>📁 {top_name} <span class="count">({len(photos)})</span></h2>
            <div class="grid">{"".join(cards)}</div>
        </section>""")

    total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM photos WHERE status='pending'").fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM photos WHERE status='approved'").fetchone()[0]
    posted = conn.execute("SELECT COUNT(*) FROM photos WHERE status='posted'").fetchone()[0]

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ron.p.wilder Photo Catalog</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 20px; }}
  h1 {{ font-size: 1.8rem; color: #f90; margin-bottom: 10px; }}
  .stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
  .stat {{ background: #222; padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; }}
  .stat .n {{ font-size: 1.4rem; font-weight: bold; color: #f90; }}
  .folder {{ margin-bottom: 40px; }}
  h2 {{ color: #ccc; margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .count {{ color: #888; font-size: 0.8rem; font-weight: normal; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }}
  .photo-card {{ background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 2px solid transparent; transition: border-color 0.2s; }}
  .photo-card:hover {{ border-color: #f90; }}
  .photo-card img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
  .info {{ padding: 6px; }}
  .subfolder {{ font-size: 0.65rem; color: #f90; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .filename {{ font-size: 0.72rem; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 2px; margin-bottom: 3px; }}
  .tag {{ background: #2a3a4a; color: #7bf; padding: 1px 4px; border-radius: 3px; font-size: 0.65rem; }}
  .status-badge {{ font-size: 0.65rem; padding: 1px 5px; border-radius: 3px; display: inline-block; }}
  .pending {{ border-color: #333; }} .status-badge.pending {{ background: #333; color: #aaa; }}
  .approved {{ border-color: #2a5; }} .status-badge.approved {{ background: #1a3a1a; color: #4f8; }}
  .posted {{ border-color: #259; }} .status-badge.posted {{ background: #1a2a3a; color: #4af; }}
  .skip {{ border-color: #500; opacity: 0.5; }} .status-badge.skip {{ background: #300; color: #f44; }}
  .notes {{ font-size: 0.65rem; color: #fa0; margin-top: 2px; }}
</style>
</head>
<body>
<header style="margin-bottom:30px">
  <h1>📷 ron.p.wilder — Photo Catalog</h1>
  <div class="stats">
    <div class="stat"><div class="n">{total}</div>סה"כ</div>
    <div class="stat"><div class="n">{pending}</div>ממתין</div>
    <div class="stat"><div class="n">{approved}</div>מאושר</div>
    <div class="stat"><div class="n">{posted}</div>פורסם</div>
    <div class="stat"><div class="n">{len(top_folders)}</div>סשנים</div>
  </div>
</header>
{"".join(folder_sections)}
<p style="color:#555;text-align:center;margin-top:40px;font-size:0.8rem">
  Echo 🔊 | {len(rows)} photos shown | {total} total in catalog
</p>
</body>
</html>"""

    GALLERY_PATH.write_text(html, encoding="utf-8")
    size_kb = GALLERY_PATH.stat().st_size // 1024
    print(f"✅ Gallery: {GALLERY_PATH} ({size_kb}KB, {len(rows)} photos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--gallery", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--status", type=str, default=None)
    parser.add_argument("--gallery-limit", type=int, default=500)
    args = parser.parse_args()

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_db(DB_PATH)

    if not args.scan and not args.gallery:
        scan_and_build(conn, limit=args.limit)
        generate_gallery(conn, limit=args.gallery_limit)
    else:
        if args.scan:
            scan_and_build(conn, limit=args.limit)
        if args.gallery:
            generate_gallery(conn, filter_status=args.status, limit=args.gallery_limit)

    conn.close()
