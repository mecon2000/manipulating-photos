#!/usr/bin/env python3
"""
Post a single photo to the @ron.p.wilder Instagram feed.

Usage:
    python3 post_feed_photo.py <image_path> <caption_file>

    image_path   — path to JPEG (should be 4:5 portrait or 1:1 square)
    caption_file — path to a .txt file containing the full caption

Session is loaded from ig_session.json (re-authenticated if expired).
"""

import sys
import time
from pathlib import Path
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, TwoFactorRequired, ChallengeRequired

USERNAME = "ron.p.wilder"
PASSWORD = "z69SbesK5DMZmsRU"
SESSION_FILE = Path.home() / ".openclaw" / "workspace" / "secrets" / "ig_session.json"


def login() -> Client:
    cl = Client()
    cl.delay_range = [2, 5]

    if SESSION_FILE.exists():
        print("Loading saved session...")
        cl.load_settings(SESSION_FILE)
        try:
            cl.login(USERNAME, PASSWORD)
            print("  ✓ session restored")
            return cl
        except Exception as e:
            print(f"  Session expired ({e}), re-logging in...")

    print(f"Logging in as {USERNAME}...")
    try:
        cl.login(USERNAME, PASSWORD)
    except TwoFactorRequired:
        code = input("  2FA code (SMS/app): ").strip()
        cl.login(USERNAME, PASSWORD, verification_code=code)
    except ChallengeRequired:
        print("  Challenge required — check your phone/email and confirm login.")
        input("  Press Enter once confirmed...")
        cl.login(USERNAME, PASSWORD)

    cl.dump_settings(SESSION_FILE)
    print(f"  ✓ Logged in, session saved.")
    return cl


def main():
    if len(sys.argv) != 3:
        print("Usage: post_feed_photo.py <image_path> <caption_file>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    caption_file = Path(sys.argv[2])

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    if not caption_file.exists():
        print(f"ERROR: Caption file not found: {caption_file}")
        sys.exit(1)

    caption = caption_file.read_text(encoding="utf-8").strip()

    print(f"Image:   {image_path}")
    print(f"Caption: {caption[:80]}...")
    print()

    cl = login()

    print("Uploading photo to feed...")
    media = cl.photo_upload(image_path, caption)
    print(f"✅ Posted! Media ID: {media.pk}")
    print(f"   https://www.instagram.com/p/{media.code}/")


if __name__ == "__main__":
    main()
