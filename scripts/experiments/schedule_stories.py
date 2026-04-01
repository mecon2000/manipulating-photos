#!/usr/bin/env python3
"""
Posts comeback stories at 20:00 Israel time (UTC+2).
Run this script and leave it running — it will wait and post at the right time.
"""

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from instagrapi import Client

USERNAME = "ron.p.wilder"
PASSWORD = "z69SbesK5DMZmsRU"
SESSION_FILE = Path.home() / ".openclaw/workspace/secrets/ig_session.json"
SLIDES_DIR = Path.home() / ".openclaw/workspace/scripts/story_slides"

ISRAEL_TZ = timezone(timedelta(hours=2))
TARGET_HOUR = 20
TARGET_MIN = 0

slide_paths = sorted(SLIDES_DIR.glob("slide_*.jpg"))

def get_target_time():
    now = datetime.now(ISRAEL_TZ)
    target = now.replace(hour=TARGET_HOUR, minute=TARGET_MIN, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target

def main():
    target = get_target_time()
    now = datetime.now(ISRAEL_TZ)
    wait_sec = (target - now).total_seconds()

    print(f"Current time (Israel): {now.strftime('%H:%M:%S')}")
    print(f"Will post at:          {target.strftime('%H:%M:%S')} ({wait_sec/3600:.1f}h from now)")
    print(f"Slides ready: {[p.name for p in slide_paths]}")
    print("Waiting...")

    time.sleep(wait_sec)

    print("\n20:00 — posting now!")
    cl = Client()
    cl.delay_range = [2, 5]
    cl.load_settings(SESSION_FILE)
    cl.login(USERNAME, PASSWORD)

    for i, path in enumerate(slide_paths):
        print(f"  uploading {path.name}...", end=" ", flush=True)
        cl.photo_upload_to_story(path)
        print("✓")
        if i < len(slide_paths) - 1:
            time.sleep(3)

    cl.dump_settings(SESSION_FILE)
    print("\n✅ Stories posted at 20:00!")

if __name__ == "__main__":
    main()
