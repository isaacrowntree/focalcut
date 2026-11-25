"""Camera moves: what the camera is doing over a stretch of footage -- above all a drone's, whose shots are usually
flown for one purpose: a long held horizon (made for a time-lapse), a spin straight down over the action, a dive
in, a pull-out reveal, a fly-over, a pan.

Corners are tracked frame to frame (Lucas-Kanade, as shake.py does) and a RANSAC similarity fit gives the camera's
whole motion per frame: shift (dx, dy in frame widths), rotation (radians) and scale (log zoom: > 0 the picture
grows -- pushing in or descending; < 0 pulling out or climbing), tracked on the background only -- the people the
score step found are masked out, so their moving isn't taken for the camera's. Writes <reel>/scores/<name>_moves.npz (t, dx, dy,
rot, zoom) at RATE fps, from each unit's export file (GPU-decoded H.264) or its source. Resumable.

label() names the dominant move over a stretch, from per-second rates against thresholds (reel.json "moves" may
set any of them):

    held     barely moving: under HELD of a frame width a second, rotation and zoom alike
    spin     turning: over SPIN radians a second (a drone yawing top-down over the action)
    push     the picture growing over ZOOM a second (diving in, descending)
    pull     the picture shrinking (pulling out, climbing: a reveal)
    pan      sideways over PAN of a frame width a second, steadily one way
    tilt     up or down the same way
    fly      moving across steadily in a direction that isn't purely sideways or vertical (a fly-over, a track)
    drift    moving, slower than any of the above
    wobble   moving without a direction (handheld, a gust)

The planner asks for moves by name (reference mode: the move the reference's own shot made, and reel.json
"moves_for": {"<category>": ["spin", "push"]}), and plays a held shot as a time-lapse (reel.json "drone_speed").
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RATE, SIDE = 10, 320
TH = dict(held=0.02, spin=0.06, zoom=0.04, pan=0.05)        # per-second rates (frame widths, radians, log zoom)
KINDS = ("held", "spin", "push", "pull", "pan", "tilt", "fly", "drift", "wobble")


def frames_moves(g, masks=None):
    """Per frame (n, 4): dx, dy (frame widths), rotation (rad), log scale -- the camera's similarity transform
    between neighbouring grey frames g (n, h, w). masks (n, h, w uint8, optional): where to track -- the
    background, people left out, so a crowd moving (a splash, a dunk, a hug filling a long lens) isn't read as
    the camera moving."""
    import cv2
    w = g.shape[2] if len(g) else 1
    out = np.zeros((len(g), 4))
    for i in range(1, len(g)):
        mk = masks[i - 1] if masks is not None else None
        p0 = cv2.goodFeaturesToTrack(g[i - 1], 300, 0.01, 6, mask=mk)
        if p0 is None or len(p0) < 8:                       # (too little background: track the whole frame)
            p0 = cv2.goodFeaturesToTrack(g[i - 1], 300, 0.01, 6)
        if p0 is None or len(p0) < 8:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g[i - 1], g[i], p0, None)
        ok = st[:, 0] == 1
        if ok.sum() < 8:
            continue
        m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if m is None:
            continue
        a, b = m[0, 0], m[1, 0]
        h_, w_ = g.shape[1:]
        # the shift of the frame's centre (the transform is about the top-left corner)
        cx, cy = w_ / 2, h_ / 2
        tx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2] - cx
        ty = m[1, 0] * cx + m[1, 1] * cy + m[1, 2] - cy
        out[i] = [tx / w, ty / w, np.arctan2(b, a), np.log(max(np.hypot(a, b), 1e-6))]
    if len(out) > 1:
        out[0] = out[1]
    return out


def grey(path, ts, dur, gpu=True, rate=RATE):
    import subprocess
    sw, sh = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                               "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    w, h = SIDE, SIDE * sh // sw // 2 * 2
    r = subprocess.run(["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if gpu else ["-threads", "4"]),
                        "-ss", f"{ts:.3f}", "-t", f"{dur:.3f}", "-i", path, "-an",
                        "-vf", f"fps={rate},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    if r.returncode and gpu:
        return grey(path, ts, dur, False, rate)
    raw = r.stdout
    return np.frombuffer(raw, np.uint8)[: len(raw) // (w * h) * w * h].reshape(-1, h, w)


def label(mv, rate=RATE, th=None):
    """The dominant move over a stretch of per-frame moves (n, 4) -> (kind, rates dict)."""
    th = {**TH, **(th or {})}
    if len(mv) < 3:
        return "held", {}
    s = mv.sum(0) * rate / len(mv)                           # net per second (direction kept)
    a = np.abs(mv).sum(0) * rate / len(mv)                   # gross per second
    shift, net_shift = float(np.hypot(*a[:2])), float(np.hypot(*s[:2]))
    rot, zoom = float(abs(s[2])), float(s[3])
    r = dict(shift=round(shift, 4), net=round(net_shift, 4), rot=round(rot, 4), zoom=round(zoom, 4))
    if shift < th["held"] and a[2] < th["spin"] / 3 and abs(a[3]) < th["zoom"] / 3:
        return "held", r
    if rot >= th["spin"]:
        return "spin", r
    if abs(zoom) >= th["zoom"]:
        return ("push" if zoom > 0 else "pull"), r
    if net_shift >= th["pan"]:
        ang = abs(np.degrees(np.arctan2(s[1], s[0])))
        if ang < 25 or ang > 155:
            return "pan", r
        if 65 < ang < 115:
            return "tilt", r
        return "fly", r
    if net_shift < 0.35 * shift and shift >= th["held"] * 2:
        return "wobble", r
    return "drift", r


def people_masks(score, times, shape, pad=0.15):
    """Per time: a mask of the frame with the people (the score step's boxes, padded) left out; None without
    a score."""
    if score is None or not os.path.exists(score):
        return None
    z = np.load(score)
    pt, bx = z["pose_t"], z["boxes"]
    h, w = shape
    k = w / 640.0
    out = np.full((len(times), h, w), 255, np.uint8)
    for n, t in enumerate(times):
        j = int(np.argmin(np.abs(pt - t))) if len(pt) else 0
        if not len(pt) or abs(pt[j] - t) > 0.5:
            continue
        for b in bx[j]:
            if np.isnan(b[0]):
                continue
            x0, y0, x1, y1 = b[:4] * k
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            out[n, max(int(y0 - py), 0):int(y1 + py), max(int(x0 - px), 0):int(x1 + px)] = 0
    return out


def _measure(args):
    path, x0, spans, out, score = args
    T, M = [], []
    for a, b in spans:
        g = grey(path, max(a - x0, 0), b - a)
        tt = a + np.arange(len(g)) / RATE
        mv = frames_moves(g, people_masks(score, tt, g.shape[1:]) if len(g) else None)
        T.append(a + np.arange(len(mv)) / RATE); M.append(mv)
    t = np.concatenate(T) if T else np.zeros(0)
    mv = np.concatenate(M) if M else np.zeros((0, 4))
    np.savez(out, t=t, dx=mv[:, 0], dy=mv[:, 1], rot=mv[:, 2], zoom=mv[:, 3])
    return out


def run(project, workers: int = 3, only=None) -> None:
    """Moves for every unit (reel.json "moves_in": categories or clip patterns to limit it to, e.g. ["DJI_*"])."""
    import fnmatch

    from .shake import export_of
    units = project.load("units.json")
    pats = only or project.config.get("moves_in")
    jobs, by = [], {}
    for u in units:
        if pats and not any(fnmatch.fnmatch(u["clip"], p) or u["category"] == p for p in pats):
            continue
        by.setdefault(u["score"], []).append(u)
    for name, us in by.items():
        out = project.path("scores", name + "_moves.npz")
        if os.path.exists(out):
            continue
        ex = export_of(project, us[0])
        path, x0 = (ex, us[0].get("export_t0", 0.0)) if ex else (os.path.join(project.folder, us[0]["source"]), 0.0)
        jobs.append((path, x0, [(u["t0"], u["t1"]) for u in us], out, project.path("scores", name + ".npz")))
    if jobs:
        with ProcessPoolExecutor(workers) as ex:
            for out in ex.map(_measure, jobs):
                print(f"[reel] moves: {os.path.basename(out)}", flush=True)


_CACHE: dict = {}


def series(project, unit):
    p = project.path("scores", unit["score"] + "_moves.npz")
    if p not in _CACHE:
        _CACHE[p] = dict(np.load(p)) if os.path.exists(p) else None
    return _CACHE[p]


def at(project, unit, a, b):
    """The camera's move over source seconds a..b of a unit: (kind, rates) -- None when unmeasured."""
    z = series(project, unit)
    if z is None:
        return None
    m = (z["t"] >= a) & (z["t"] <= b)
    if m.sum() < 3:
        return None
    mv = np.stack([z["dx"][m], z["dy"][m], z["rot"][m], z["zoom"][m]], 1)
    return label(mv, th=(project.config.get("moves") or None))
