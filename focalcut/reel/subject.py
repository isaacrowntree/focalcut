"""Subject track for every unit: SAM 3 + saliency + depth (subject_engine) on the camera's keyframes (~1 a second)
at 960 px, inside the units' time ranges. Writes <reel>/scores/<name>_subject.npz:
  t (K,)  box (K, 4) subject union x0 y0 x1 y1 as fractions   touch (K, 4) mask at the left/top/right/bottom edge
  n (K,)  people in the subject                               strength (K,)
Frames are in the reel's orientation (rotated clips rotated). Resumable: existing tracks are kept.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time

import numpy as np

LONG, STEP = 960, 0.9


def keyframes(project, unit, spans, out):
    """Decode keyframes inside the spans into the queue as (source time, rgb); None at the end. Uses the source's
    proxy when one is configured."""
    src, _ = project.decode_from(unit["source"], 0.0)
    w, h = (LONG, LONG * 9 // 16) if unit["orient"] == "landscape" else (LONG * 9 // 16, LONG)
    rot = (unit["rotate"] + ",") if unit.get("rotate") else ""
    sw, sh = (h, w) if rot else (w, h)            # scale before the rotation
    p = subprocess.Popen(["ffmpeg", "-v", "info", "-hide_banner", "-nostats", "-skip_frame", "nokey", "-i", src, "-an",
                          "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{STEP})',scale={sw}:{sh},"
                                 f"{rot}showinfo", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times = queue.Queue()

    def err():
        for line in p.stderr:
            m = re.search(rb"pts_time:([\d.]+)", line)
            if m:
                times.put(float(m.group(1)))
    threading.Thread(target=err, daemon=True).start()
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        t = project.source_time(unit["source"], times.get(timeout=30))
        if any(a - 0.5 <= t <= b_ + 0.5 for a, b_ in spans):
            out.put((t, np.frombuffer(b, np.uint8).reshape(h, w, 3)))
    p.wait()
    out.put(None)


def by_score(units):
    by = {}
    for u in units:
        by.setdefault(u["score"], []).append(u)
    return by


def track_one(project, name, units):
    out = project.path("scores", name + "_subject.npz")
    if os.path.exists(out):
        return f"{name}: cached"
    from . import subject_engine as E
    q = queue.Queue(maxsize=16)
    threading.Thread(target=keyframes, args=(project, units[0], [(u["t0"], u["t1"]) for u in units], q),
                     daemon=True).start()
    T, B, TO, N, S = [], [], [], [], []
    t0 = time.time()
    while (item := q.get()) is not None:
        t, rgb = item
        r = E.subject(rgb)
        T.append(t)
        if r is None:
            B.append([np.nan] * 4); TO.append([False] * 4); N.append(0); S.append(0.0)
        else:
            B.append(r["box"]); TO.append(r["touch"]); N.append(r["n"]); S.append(r["strength"])
    np.savez_compressed(out, t=np.array(T), box=np.array(B, np.float32).reshape(-1, 4),
                        touch=np.array(TO, bool).reshape(-1, 4), n=np.array(N), strength=np.array(S, np.float32))
    return f"{name}: {len(T)} frames in {time.time() - t0:.0f}s"


def run(project) -> None:
    os.makedirs(project.path("scores"), exist_ok=True)
    for name, units in by_score(project.load("units.json")).items():
        print("[reel] subject", track_one(project, name, units), flush=True)
