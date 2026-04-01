#!/usr/bin/env python3
"""
Generates comeback story slides and posts them to @ron.p.wilder
"""

import os
import sys
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, TwoFactorRequired, ChallengeRequired

# ── Config ──────────────────────────────────────────────────────────────────
USERNAME = "ron.p.wilder"
PASSWORD = "z69SbesK5DMZmsRU"
SESSION_FILE = Path.home() / ".openclaw" / "workspace" / "secrets" / "ig_session.json"
OUTPUT_DIR = Path.home() / ".openclaw" / "workspace" / "scripts" / "story_slides"

W, H = 1080, 1920
BG_COLOR = (13, 13, 13)       # #0d0d0d
TEXT_COLOR = (245, 245, 245)  # near-white
ACCENT_COLOR = (180, 180, 180) # for smaller text

SLIDES = [
    {
        "lines": ["I'm back."],
        "size": 90,
        "weight": "bold",
    },
    {
        "lines": [
            "My main account had",
            "13,000 followers.",
            "",
            "Then I got hacked.",
            "The attacker sent spam",
            "from my account.",
            "Instagram banned me permanently.",
            "",
            "That account is gone.",
        ],
        "size": 52,
        "weight": "regular",
    },
    {
        "lines": ["This is home now.", "", "@ron.p.wilder"],
        "size": 68,
        "weight": "bold",
        "accent_lines": [2],  # @ron.p.wilder in accent color
    },
    {
        "lines": [
            "If you followed me before —",
            "or know someone who did —",
            "send them this profile.",
            "",
            "Share this Story.",
        ],
        "size": 58,
        "weight": "regular",
    },
    {
        "lines": [
            "New work drops this week.",
            "",
            "You're going to want",
            "to see this.",
        ],
        "size": 62,
        "weight": "bold",
    },
]


def get_font(size, weight="regular"):
    """Try to load a nice font, fall back to default."""
    candidates_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    candidates_regular = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    candidates = candidates_bold if weight == "bold" else candidates_regular
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_slide(slide_data, index):
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    lines = slide_data["lines"]
    size = slide_data["size"]
    weight = slide_data.get("weight", "regular")
    accent_lines = slide_data.get("accent_lines", [])

    font = get_font(size, weight)
    line_height = size * 1.5

    # Calculate total block height
    total_h = len(lines) * line_height
    start_y = (H - total_h) / 2

    padding = 80  # left/right padding

    for i, line in enumerate(lines):
        if not line:
            continue
        color = ACCENT_COLOR if i in accent_lines else TEXT_COLOR
        # Word-wrap long lines
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        if text_w > W - 2 * padding:
            # wrap
            wrapped = textwrap.fill(line, width=int((W - 2*padding) / (size * 0.55)))
            for j, wline in enumerate(wrapped.split("\n")):
                y = start_y + i * line_height + j * line_height
                draw.text((W // 2, y), wline, font=font, fill=color, anchor="mt")
        else:
            y = start_y + i * line_height
            draw.text((W // 2, y), line, font=font, fill=color, anchor="mt")

    # Subtle bottom watermark
    small_font = get_font(28, "regular")
    draw.text((W // 2, H - 60), "@ron.p.wilder", font=small_font, fill=(80, 80, 80), anchor="mt")

    out_path = OUTPUT_DIR / f"slide_{index+1:02d}.jpg"
    img.save(out_path, "JPEG", quality=95)
    print(f"  ✓ slide {index+1} → {out_path.name}")
    return out_path


def login():
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
            print(f"  session expired ({e}), re-logging in...")

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
    print(f"  ✓ logged in, session saved to {SESSION_FILE.name}")
    return cl


def post_stories(cl, slide_paths):
    print(f"\nPosting {len(slide_paths)} story slides...")
    for i, path in enumerate(slide_paths):
        print(f"  uploading slide {i+1}/{len(slide_paths)}...", end=" ", flush=True)
        cl.photo_upload_to_story(path)
        print("✓")
        if i < len(slide_paths) - 1:
            import time
            time.sleep(3)
    print("\n✅ All stories posted!")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating slides ===")
    slide_paths = []
    for i, slide in enumerate(SLIDES):
        path = make_slide(slide, i)
        slide_paths.append(path)

    print("\n=== Logging into Instagram ===")
    cl = login()

    print("\n=== Posting stories ===")
    post_stories(cl, slide_paths)


if __name__ == "__main__":
    main()
