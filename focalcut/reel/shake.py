"""Camera shake per source: handheld jitter, not the camera's deliberate moves.

Corners are tracked frame to frame (Lucas-Kanade) and a RANSAC similarity fit gives the camera's own motion (the
dancers are the outliers). Shake is what is left of that motion after a ~0.1 s smooth -- a pan, a follow or a slow
sway drops out; a wobble doesn't -- as an RMS over 1 s, in frame widths per frame (`hf`). Measured on what the
reel will actually use: each unit's own stretch, from its export file when there is one (H.264 decodes on the GPU
far faster than the cameras' 10-bit 4:2:2 HEVC), else from the source or its proxy.

Typical values (hf): stage tripods 0.0003-0.001; handheld social floor 0.001-0.006; a talking head shot handheld
from the audience or a hand-stabilised master 0.002-0.02. A 9:16 strip cut from a 16:9 frame shows the shake three
times over, so the planner divides by the crop's width before comparing (plan.SHAKE).

Writes <reel>/scores/<name>_shake.npz (t, hf, mv: raw dx/dy per frame). Resumable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RATE, SIDE = 25, 320                              # analysis rate (fps) and width (px)


def motion(path, ts, dur, gpu=False):
    """Camera motion frame to frame (dx, dy in frame widths) over a stretch of a file, at RATE."""
    import cv2
    sw, sh = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                               "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    w, h = SIDE, SIDE * sh // sw // 2 * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if gpu else ["-threads", "4"]),
                          "-ss", f"{ts:.3f}", "-t", f"{dur:.3f}", "-i", path,
                          "-an", "-vf", f"fps={RATE},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    g = np.frombuffer(raw, np.uint8)[: len(raw) // (w * h) * w * h].reshape(-1, h, w)
    mv = np.zeros((len(g), 2))
    for i in range(1, len(g)):
        p0 = cv2.goodFeaturesToTrack(g[i - 1], 200, 0.01, 8)
        if p0 is None or len(p0) < 8:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g[i - 1], g[i], p0, None)
        ok = st[:, 0] == 1
        if ok.sum() < 8:
            continue
        m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if m is not None:
            mv[i] = m[:, 2]
    return mv / w


def hf(mv):
    """High-frequency shake: frame-to-frame camera motion minus its ~0.1 s average, RMS over 1 s."""
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    r = mv - np.stack([np.convolve(np.pad(mv[:, j], 2, mode="edge"), k, mode="valid") for j in range(2)], 1)
    return np.sqrt(np.convolve((r ** 2).sum(1), np.ones(RATE) / RATE, mode="same")).astype(np.float32)


def export_of(project, unit):
    """The unit's own export file (same frames, from export_t0), or None (a joined B-roll piece has none)."""
    f = os.path.join(project.export_dir, unit["category"], unit["id"] + ".mp4")
    return f if "#" not in unit["id"] and os.path.exists(f) else None


def measure(project, name, units, gpu):
    out = project.path("scores", name + "_shake.npz")
    if os.path.exists(out):
        return f"{name}: cached"
    t, mv = [], []
    for u in sorted(units, key=lambda u: u["t0"]):
        f = export_of(project, u)
        if f:
            m = motion(f, u["t0"] - u.get("export_t0", 0.0), u["t1"] - u["t0"], gpu=gpu)
        else:
            src, ts = project.decode_from(u["source"], u["t0"])
            m = motion(src, ts, u["t1"] - u["t0"])
        t.append(u["t0"] + np.arange(len(m)) / RATE); mv.append(m)
    t, mv = np.concatenate(t), np.concatenate(mv)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, t=t, hf=hf(mv), mv=mv.astype(np.float32))
    return f"{name}: {len(t)} frames, median {np.median(hf(mv)) * 1e3:.2f}e-3"


def series(project, name):
    """(t, hf) for a source, or None if it wasn't measured."""
    f = project.path("scores", name + "_shake.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return z["t"], z["hf"]


def jitter(project, unit, a, b):
    """Mean shake over a stretch of a unit's source time, or None if it wasn't measured."""
    z = series(project, unit["score"])
    if z is None:
        return None
    t, j = z
    m = (t >= a) & (t <= b)
    return float(j[m].mean()) if m.any() else None


def run(project, workers: int = 3) -> None:
    by = {}
    for u in project.load("units.json"):
        by.setdefault(u["score"], []).append(u)
    gpu = bool(shutil.which("nvidia-smi")) and not os.environ.get("FOCALCUT_SHAKE_CPU")
    with ProcessPoolExecutor(max(1, workers)) as ex:
        futs = [ex.submit(measure, project, k, us, gpu) for k, us in sorted(by.items(), key=lambda kv: -sum(
            u["t1"] - u["t0"] for u in kv[1]))]
        for f in futs:
            print("[reel] shake", f.result(), flush=True)
