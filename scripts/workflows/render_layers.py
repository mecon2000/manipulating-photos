#!/usr/bin/env python3
"""Render the editor's chosen combination: final image, build-up frames, and a video.

The editor previews at 760px with canvas-approximated blur; this re-renders the same
combination from the full-resolution plates so the saved image is the real thing.

  render_layers.py --settings settings.json [--out-dir DIR] [--no-video]

Build order is far-to-near — plane 5, 4, the portrait (with its shadow), 2, 1 (with the
vignette) — so showing the frames in sequence looks like the picture assembling itself.
"""
import argparse, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
SHARED = Path("~/.openclaw/workspace/shared/faces-candidates").expanduser()
BUILD_MS, HOLD_MS, TEARDOWN_MS = 300, 2500, 100


def _dp():
    spec = importlib.util.spec_from_file_location("dp", HERE / "depth-planes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ffmpeg():
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def make_video(build, hold, teardown, out_path, w, h):
    """Build up at 300ms, hold the finished picture, then tear down at 100ms.

    Teardown is its own frame list, not the build reversed: the build shows planes that
    were tried and dropped, and replaying those backwards would re-introduce work the
    picture had already rejected.
    """
    exe = ffmpeg()
    if not exe:
        print("  ffmpeg not found — skipping video")
        return None
    seq = [(f, BUILD_MS) for f in build]
    seq.append((hold, HOLD_MS))
    seq += [(f, TEARDOWN_MS) for f in teardown]
    lst = out_path.with_suffix(".concat.txt")
    lines = []
    for f, ms in seq:
        lines.append(f"file '{f}'")
        lines.append(f"duration {ms / 1000:.3f}")
    lines.append(f"file '{seq[-1][0]}'")          # concat needs the last frame twice
    lst.write_text("\n".join(lines))
    even = f"scale={w - w % 2}:{h - h % 2}"       # h264 needs even dimensions
    r = subprocess.run([exe, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-vf", f"{even},format=yuv420p", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "20", "-r", "30", str(out_path)],
                       capture_output=True, text=True)
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        print("  ffmpeg failed:", r.stderr.strip()[-200:])
        return None
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    cfg = json.loads(Path(args.settings).read_text())
    prep = SHARED / "depth-planes" / "_prep" / cfg["stem"]
    meta = json.loads((prep / "meta.json").read_text())
    dp = _dp()

    W, H = meta["w"], meta["h"]
    port_img = np.asarray(Image.open(meta["portrait"]).convert("RGB"))

    # subject + masks, exactly as prepare_layers built them
    stem_src = cfg["stem"].split("__style")[0]
    cand = Path(meta["portrait"]).parent.parent / "_sources" / "crop_916" / f"{stem_src}.jpg"
    mimg = np.asarray(Image.open(cand).convert("RGB")) if cand.is_file() else port_img
    if mimg.shape[:2] != (H, W):
        mimg = cv2.resize(mimg, (W, H), interpolation=cv2.INTER_LANCZOS4)
    subj = dp.refine_alpha(mimg, dp.subject_alpha(mimg), shrink=1.5)
    pts = dp.mesh(port_img)

    s = meta["scale"]
    sw, sh = int(W * s), int(H * s)
    ox, oy = (W - sw) // 2, H - sh
    small = cv2.resize(port_img, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
    small_a = cv2.resize(subj, (sw, sh), interpolation=cv2.INTER_LINEAR)
    fade = max(6, int(min(sw, sh) * 0.03))
    ramp = np.ones((sh, sw), np.float32)
    g = np.linspace(0, 1, fade, dtype=np.float32)
    ramp[:, :fade] *= g; ramp[:, -fade:] *= g[::-1]; ramp[:fade, :] *= g[:, None]
    port_l = np.zeros_like(port_img); subj_l = np.zeros((H, W), np.float32)
    port_l[oy:oy + sh, ox:ox + sw] = small
    subj_l[oy:oy + sh, ox:ox + sw] = small_a * ramp
    if pts is not None:
        pts = pts * s + np.array([ox, oy], np.float32)

    hole = dp.eye_hole((H, W), pts)
    relief = dp.face_relief((H, W), pts, meta["front_relief"])
    body = cv2.GaussianBlur(subj_l, (0, 0), max(W, H) * 0.02)
    front = hole * relief * (1.0 - meta["subject_relief"] * np.clip(body, 0, 1))

    def plate(plane, letter):
        f = prep / "plates" / f"plane{plane}_{letter}.jpg"
        im = np.asarray(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS))
        if plane == 1:
            z = 2.6
            big = cv2.resize(im, (int(W * z), int(H * z)), interpolation=cv2.INTER_LANCZOS4)
            cy, cx = big.shape[0] // 2, big.shape[1] // 2
            im = big[cy - H // 2:cy - H // 2 + H, cx - W // 2:cx - W // 2 + W]
        a = dp.key_cream(im)
        if plane == 1:
            im = np.clip(im.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
        b = dp.BLUR[plane] | 1
        im = cv2.GaussianBlur(im.astype(np.float32), (b * 2 + 1, b * 2 + 1), b / 2)
        a = cv2.GaussianBlur(a, (b * 2 + 1, b * 2 + 1), b / 2)
        if plane in (1, 2):
            a = a * front
        return im, a

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else \
        SHARED / "depth-planes" / f"edit_{cfg['stem'][:40]}"
    (out_dir / "finals").mkdir(parents=True, exist_ok=True)
    ground = np.array(meta.get("ground", [238, 238, 238]), np.float32)
    planes = cfg.get("planes", {})
    ORDER = [5, 4, "portrait", 2, 1]          # far to near

    def compose(active, grade=False):
        """Composite an arbitrary set of planes. Removal steps need to rebuild from
        scratch — you cannot un-draw a layer once it is composited."""
        c = np.full((H, W, 3), ground, np.float32)
        for item in ORDER:
            if item not in active:
                continue
            if item == "portrait":
                if cfg.get("shadow", {}).get("on"):
                    c = dp.drop_shadow(c, subj_l, cfg["shadow"].get("amount", 0.18),
                                       0.035, 10, 14, W, H)
                c = dp.over(c, port_l.astype(np.float32), subj_l)
            else:
                im, a = plate(item, planes.get(str(item), {}).get("opt", "a"))
                c = dp.over(c, im, a)
        if grade:
            if cfg.get("pop", "off") != "off":
                soft = cv2.GaussianBlur(np.clip(subj_l, 0, 1), (0, 0), max(W, H) * 0.006)
                c = dp.separate(c, soft, cfg["pop"], cfg.get("pop_amount", 0.4))
            if cfg.get("vignette", {}).get("on"):
                c = dp.vignette(c, cfg["vignette"].get("amount", 0.12), W, H)
        return c

    chosen = [i for i in ORDER
              if (i == "portrait" and cfg.get("portrait_on", True))
              or (i != "portrait" and planes.get(str(i), {}).get("on", True))]
    rejected = [i for i in ORDER if i not in chosen]

    # Every plane appears, then the rejected ones drop away, so the film ends on exactly
    # the chosen picture rather than quietly skipping what was tried and discarded.
    seq, active = [], []
    for item in ORDER:
        active.append(item)
        seq.append((f"add_{item}", list(active)))
    for item in reversed([i for i in ORDER if i in rejected]):
        active = [x for x in active if x != item]
        seq.append((f"drop_{item}", list(active)))

    stages = [(name, compose(a)) for name, a in seq]
    stages.append(("final", compose(chosen, grade=True)))

    # Teardown: strip the chosen picture back down, nearest plane first.
    down = []
    active = list(chosen)
    for item in [i for i in reversed(ORDER) if i in chosen][:-1]:
        active = [x for x in active if x != item]
        down.append((f"undo_{item}", compose(active)))

    frames, tear = [], []
    for i, (name, c) in enumerate(stages, 1):
        f = out_dir / "finals" / f"build_{i:02d}_{name}.jpg"
        Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(f, quality=94)
        frames.append(str(f))
    for i, (name, c) in enumerate(down, 1):
        f = out_dir / "_teardown_{:02d}_{}.jpg".format(i, name)
        Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(f, quality=90)
        tear.append(str(f))
    print(f"  {len(frames)} build stages: {[n for n, _ in stages]}")
    print(f"  {len(tear)} teardown stages")

    shutil.copyfile(frames[-1], out_dir / "finals" / "final.jpg")
    (out_dir / "settings.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {len(frames)} build frames + final → {out_dir/'finals'}")

    if not args.no_video:
        v = make_video(frames[:-1], frames[-1], tear, out_dir / "build.mp4", W, H)
        if v:
            print(f"  video → {v}")


if __name__ == "__main__":
    main()
