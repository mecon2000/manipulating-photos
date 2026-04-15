"""
Pushbullet notification helper — sends preview images to phone.

Usage:
    from notify import push_image, push_text

    push_image("/path/to/final.jpg", title="Noir Paint — Flora", body="cool palette, 2 tones")
    push_text("Pipeline complete", "3 photos processed in 45s")
"""

import os
import json
import subprocess

# Running counter for push notifications (resets at 99)
_COUNTER_FILE = os.path.expanduser("~/.openclaw/workspace/shared/.push_counter")

def _next_push_number():
    """Get next push number (1-99, wraps around)."""
    try:
        with open(_COUNTER_FILE) as f:
            n = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        n = 0
    n = (n % 99) + 1
    try:
        with open(_COUNTER_FILE, "w") as f:
            f.write(str(n))
    except OSError:
        pass
    return n


# Load token from env (auto-loaded from ~/sol/.env by each script)
def _get_token():
    token = os.environ.get("PUSHBULLET_TOKEN", "")
    if not token:
        # Try loading from .env directly
        env_file = os.path.expanduser("~/sol/.env")
        if os.path.isfile(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PUSHBULLET_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
    return token


def _curl_json(url, headers=None, data=None, file_path=None, form_fields=None, timeout=30):
    """Run curl and return parsed JSON (or None on failure)."""
    cmd = ["curl", "-s", "--max-time", str(timeout)]

    if headers:
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]

    if data:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(data)]
    elif form_fields:
        cmd += ["-X", "POST"]
        for k, v in form_fields.items():
            if k == "file" and v.startswith("@"):
                cmd += ["-F", f"{k}={v}"]
            else:
                cmd += ["-F", f"{k}={v}"]

    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def push_text(title, body=""):
    """Send a text notification to phone."""
    token = _get_token()
    if not token:
        return False

    result = _curl_json(
        "https://api.pushbullet.com/v2/pushes",
        headers={"Access-Token": token},
        data={"type": "note", "title": title, "body": body},
    )
    return result is not None and result.get("active", False)


def push_image(file_path, title="", body=""):
    """Upload an image and send it as a push notification.

    Args:
        file_path: path to the image file
        title: notification title
        body: notification body text

    Returns:
        True if sent successfully, False otherwise
    """
    token = _get_token()
    if not token:
        return False

    file_name = os.path.basename(file_path)
    # Detect mime type
    ext = os.path.splitext(file_name)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")

    # Step 1: Request upload URL
    upload_req = _curl_json(
        "https://api.pushbullet.com/v2/upload-request",
        headers={"Access-Token": token},
        data={"file_name": file_name, "file_type": mime},
    )
    if not upload_req or "upload_url" not in upload_req:
        return False

    upload_url = upload_req["upload_url"]
    file_url = upload_req["file_url"]

    # Step 2: Upload file
    form = {}
    for k, v in upload_req.get("data", {}).items():
        form[k] = v
    form["file"] = f"@{file_path}"

    _curl_json(upload_url, form_fields=form, timeout=60)

    # Step 3: Send push with file
    result = _curl_json(
        "https://api.pushbullet.com/v2/pushes",
        headers={"Access-Token": token},
        data={
            "type": "file",
            "file_name": file_name,
            "file_type": mime,
            "file_url": file_url,
            "title": f"#{_next_push_number():02d} {title or file_name}",
            "body": body,
        },
    )
    return result is not None and result.get("active", False)
