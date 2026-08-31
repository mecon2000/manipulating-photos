#!/usr/bin/env python3
"""Work out the clock offset between the camera and the phone that shot the BTS video.

The two devices do not agree. In one session the camera ran 67 minutes behind the phone,
so matching a photo to "the clip covering that moment" by raw timestamps picked footage
from an entirely different part of the day — arithmetically right, completely wrong.

Nothing here asks the user. The video's audio contains the shutter firing, and the photo
timestamps say exactly when each shutter fired. Cross-correlating those two event trains
recovers the offset: burst patterns are distinctive enough that the true alignment stands
well clear of the noise.

  ig_sync.py --session "2025-06-09 Elly*"
"""
import argparse, json, re, subprocess, sqlite3, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ig_reel as R
import ig_bts as B

BIN = 0.25                    # seconds per bin for the correlation
MAX_OFFSET_H = 6              # search +/- this many hours
CACHE = R.OUT_ROOT / "_clock_offsets.json"


def clip_audio(path, sr=16000):
    """Mono PCM for the whole clip, via ffmpeg."""
    r = subprocess.run([R.ffmpeg(), "-v", "error", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None, sr
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0, sr


def shutter_times(path, sr=16000, hop=0.02):
    """Times (seconds into the clip) where a camera shutter probably fired.

    A shutter is a short, broadband, high-frequency transient. Energy above ~3kHz rises
    sharply and falls back; speech and room tone do not do that as abruptly. This is an
    onset detector, not a classifier — false positives are fine, since the correlation
    only needs the overall rhythm to line up.
    """
    y, sr = clip_audio(path, sr)
    if y is None or len(y) < sr:
        return np.array([])
    n = int(hop * sr)
    frames = len(y) // n
    y = y[:frames * n].reshape(frames, n)
    # crude high-band energy: difference from a smoothed copy keeps the transients
    e = np.sqrt((np.diff(y, axis=1) ** 2).mean(axis=1) + 1e-12)
    e = e / (np.median(e) + 1e-9)
    onset = np.clip(np.diff(e, prepend=e[0]), 0, None)
    thr = onset.mean() + 3.0 * onset.std()
    idx = np.where(onset > thr)[0]
    if not len(idx):
        return np.array([])
    keep, last = [], -99
    for i in idx:                                   # one event per burst of frames
        if i - last > 3:
            keep.append(i)
        last = i
    return np.array(keep) * hop


def photo_times(session_dir):
    con = sqlite3.connect(str(R.CATALOG))
    rows = con.execute(
        """SELECT p.taken_at FROM photos p JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.taken_at IS NOT NULL ORDER BY p.taken_at""",
        (f"%{session_dir.name}%",)).fetchall()
    con.close()
    return [datetime.fromisoformat(r[0]).replace(tzinfo=None) for r in rows]


def _train(times_s, lo, hi):
    """Binary event train over [lo, hi) seconds."""
    n = int((hi - lo) / BIN) + 1
    v = np.zeros(n, np.float32)
    for t in times_s:
        i = int((t - lo) / BIN)
        if 0 <= i < n:
            v[i] = 1.0
    return v


def estimate_offset(session_dir, candidates=None, drift_window=20 * 60):
    """Seconds to ADD to a camera timestamp to reach phone/video time."""
    vd = B.video_dir(session_dir)
    if not vd:
        return None, "no video folder"
    clips = [(p, B.clip_start(p)) for p in sorted(vd.iterdir())
             if p.suffix.lower() in (".mp4", ".mov")]
    clips = [(p, s) for p, s in clips if s]
    if not clips:
        return None, "no clips with a readable start time"
    photos = photo_times(session_dir)
    if len(photos) < 20:
        return None, "too few catalogued photos to correlate"

    base = min(min(s for _, s in clips), min(photos))
    shutters = []
    for p, start in clips:
        for t in shutter_times(p):
            shutters.append((start - base).total_seconds() + float(t))
    if len(shutters) < 20:
        return None, f"only {len(shutters)} shutter-like events found"

    span = max(max(shutters), max((t - base).total_seconds() for t in photos))
    a = _train(shutters, 0, span)
    b = _train([(t - base).total_seconds() for t in photos], 0, span)
    # FFT cross-correlation: searching hours of offset a bin at a time is far too slow
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    limit = int(MAX_OFFSET_H * 3600 / BIN)
    pos, neg = corr[:limit], corr[-limit:]
    scores = np.concatenate([neg, pos])
    lags = np.concatenate([np.arange(-limit, 0), np.arange(0, limit)]) * BIN
    # Only look near plausible answers. Over six hours the correlation has thousands of
    # chances to land on a coincidence; a whole-hour candidate plus a small clock drift
    # is the entire realistic space, and searching only that removes most of the noise.
    if candidates:
        near = np.zeros_like(scores, bool)
        for c in candidates:
            near |= np.abs(lags - c) <= drift_window
        scores = np.where(near, scores, -1.0)
    k = int(np.argmax(scores))
    best_lag, best = float(lags[k]), float(scores[k])
    # Sigma above the noise floor is not the right test: the floor is near zero, so a
    # mediocre peak still scores many sigma. What matters is whether the winner beats
    # every rival that would imply a DIFFERENT answer — rivals within a couple of
    # minutes are the same answer, so they are excluded from the comparison.
    far = np.abs(lags - best_lag) > 120
    runner = float(scores[far].max()) if far.any() else 0.0
    margin = (best - runner) / (best + 1e-9)
    note = (f"{len(shutters)} events, best {best:.0f} vs next distinct {runner:.0f} "
            f"({margin*100:.0f}% margin)")
    if margin < 0.25:
        return None, note + " — too close to call, not trusted"
    return best_lag, note


def camera_of(session_dir):
    import sqlite3
    con = sqlite3.connect(str(R.CATALOG))
    row = con.execute(
        """SELECT p.camera, COUNT(*) c FROM photos p JOIN sessions s ON s.id = p.session_id
           WHERE s.folder_path LIKE ? AND p.camera IS NOT NULL
           GROUP BY p.camera ORDER BY c DESC""", (f"%{session_dir.name}%",)).fetchone()
    con.close()
    return row[0] if row else None


def session_date(session_dir):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", session_dir.name)
    return datetime(*[int(x) for x in m.groups()]) if m else None


def drift_from_siblings(session_dir, window_days=400):
    """Borrow the drift measured on another session from the SAME camera.

    A camera's clock error changes slowly, so a session solved once calibrates its
    neighbours. This is what stops the manual step from recurring: measure a body's
    drift a couple of times and every other session it shot inherits it.
    """
    cam, when = camera_of(session_dir), session_date(session_dir)
    if not cam or not when or not CACHE.exists():
        return None, "no camera or date to match on", None
    data = json.loads(CACHE.read_text())
    best = None
    for name, rec in data.items():
        if rec.get("camera") != cam or "drift_s" not in rec:
            continue
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", name)
        if not m:
            continue
        d = abs((datetime(*[int(x) for x in m.groups()]) - when).days)
        if d <= window_days and (best is None or d < best[0]):
            best = (d, rec["drift_s"], name, rec.get("camera_tz"))
    if not best:
        return None, f"no solved session from {cam} within {window_days} days", None
    d, drift, name, cam_tz = best
    return drift, f"drift {drift/60:+.1f} min borrowed from {name} ({d}d away, {cam})", cam_tz


def phone_tz(session_dir):
    """The phone's UTC offset in hours, read from the clips themselves."""
    vd = B.video_dir(session_dir)
    if not vd:
        return None
    for p in sorted(vd.iterdir()):
        if p.suffix.lower() not in (".mp4", ".mov"):
            continue
        start, dur = B.clip_start(p), B.duration(p)
        out = subprocess.run([R.ffprobe(), "-v", "error", "-show_entries",
                              "format_tags=creation_time", "-of", "csv=p=0", str(p)],
                             capture_output=True, text=True).stdout.strip()
        if not (start and out):
            continue
        try:
            utc_end = datetime.fromisoformat(out.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            continue
        return round(((start + timedelta(seconds=dur)) - utc_end).total_seconds() / 3600)
    return None


def dst_candidates(session_dir):
    """Whole-hour candidates, from the data rather than a DST table.

    The phone's UTC offset is recoverable per clip (filename is local start, the
    container tag is UTC end). The camera carries no zone at all, so the remaining
    unknown is only whether its clock sits on summer time — a +/- one hour choice.
    """
    vd = B.video_dir(session_dir)
    if not vd:
        return [0.0]
    for p in sorted(vd.iterdir()):
        if p.suffix.lower() not in (".mp4", ".mov"):
            continue
        start, dur = B.clip_start(p), B.duration(p)
        out = subprocess.run([R.ffprobe(), "-v", "error", "-show_entries",
                              "format_tags=creation_time", "-of", "csv=p=0", str(p)],
                             capture_output=True, text=True).stdout.strip()
        if not (start and out):
            continue
        try:
            utc_end = datetime.fromisoformat(out.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            continue
        tz = round(((start + timedelta(seconds=dur)) - utc_end).total_seconds() / 3600)
        return [0.0, 3600.0, -3600.0, tz * 3600.0]
    return [0.0, 3600.0, -3600.0]


def cached_offset(session_dir, refresh=False):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = session_dir.name
    if not refresh and key in data:
        return data[key]["offset_s"], data[key]["note"] + " (cached)"
    cands = dst_candidates(session_dir)
    off, note = estimate_offset(session_dir, candidates=cands)
    if off is None:
        # nothing measurable here — inherit this camera's drift from a solved session
        drift, why, cam_tz = drift_from_siblings(session_dir)
        tz = phone_tz(session_dir)
        if drift is not None and cam_tz is not None and tz is not None:
            # The camera's clock was set once and never follows DST; the phone always
            # does. So the whole-hour part is simply the difference between the phone's
            # zone today and the zone the camera was set in — computed, not guessed.
            off = (tz - cam_tz) * 3600 + drift
            note = f"inherited: {why}, phone UTC+{tz} vs camera set at UTC+{cam_tz}"
            note = f"inherited: {why}"
    if off is not None:
        hour = round(off / 3600) * 3600
        tz = phone_tz(session_dir)
        data[key] = {"offset_s": off, "drift_s": off - hour, "camera": camera_of(session_dir),
                     "camera_tz": (tz - round(hour / 3600)) if tz is not None else None,
                     "note": note, "measured_at": datetime.now().isoformat()}
        CACHE.write_text(json.dumps(data, indent=2))
    return off, note


def set_offset(session_dir, seconds, note="set by hand"):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    hour = round(float(seconds) / 3600) * 3600
    tz = phone_tz(session_dir)
    data[session_dir.name] = {"offset_s": float(seconds), "drift_s": float(seconds) - hour,
                              "camera": camera_of(session_dir),
                              "camera_tz": (tz - round(hour / 3600)) if tz is not None else None,
                              "note": note, "measured_at": datetime.now().isoformat()}
    CACHE.write_text(json.dumps(data, indent=2))
    return float(seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--set-minutes", type=float,
                   help="record the offset by hand (minutes to ADD to camera times); "
                        "measured once per session, then reused forever")
    args = p.parse_args()
    hits = sorted(R.ARCHIVE.glob(f"*/{args.session}"))
    if not hits:
        sys.exit("no such session")
    sd = hits[0]
    if args.set_minutes is not None:
        off = set_offset(sd, args.set_minutes * 60)
        print(f"session : {sd.name}\noffset  : {off:+.0f}s ({args.set_minutes:+.1f} min), recorded")
        return
    off, note = cached_offset(sd, args.refresh)
    if off is None:
        print(f"could not estimate: {note}")
        return
    print(f"session : {sd.name}\noffset  : {off:+.1f}s ({off/60:+.1f} min) "
          f"to add to camera times\nbasis   : {note}")


if __name__ == "__main__":
    main()
