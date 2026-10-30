"""Which of the detected people are in focus: per-person sharpness from the camera's keyframes at 1280 px.

The fallback subject picker when there is no subject track (frame.subject): the couple the camera operator focused
on is usually the sharpest. Unreliable on its own -- dancers in motion blur read as soft and still bystanders as
sharp -- which is why the subject pass exists. Keyframes only (about one a second) keeps this to minutes per shoot.
Per pose frame of scores/<name>.npz, each box gets its sharpness from the nearest keyframe (NaN if none within
0.6 s). Writes <reel>/scores/<name>_focus.npz. Resumable.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np

LONG, STEP = 1280, 0.9                            # decode size; minimum spacing between samples (s)


def box_sharpness(gray, box, sc):
    """Laplacian energy in the middle of a person box (avoids the edge against the background), size-independent."""
    import cv2
    x0, y0, x1, y1 = (box[:4] * sc).astype(int)
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, x0 + w // 5), min(gray.shape[1], x1 - w // 5)
    y0, y1 = max(0, y0 + h // 8), min(gray.shape[0], y1 - h // 3)   # head and torso: faces and costume detail
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.nan
    c = gray[y0:y1, x0:x1]
    return float(np.percentile(np.abs(cv2.Laplacian(c, cv2.CV_32F)), 90))


def focus_one(scores_dir, name, src, rotated):
    import cv2
    cv2.setNumThreads(2)
    out = os.path.join(scores_dir, name + "_focus.npz")
    if os.path.exists(out):
        return f"{name}: cached"
    d = np.load(os.path.join(scores_dir, name + ".npz"))
    w, h = d["size"]
    aw = 640 if w >= h else 640 * w / h            # analysis width the boxes are in
    sw, sh = (LONG, round(LONG * h / w / 2) * 2) if w >= h else (round(LONG * w / h / 2) * 2, LONG)
    sc = sw / aw
    if rotated:                                     # the score pass saw these unrotated
        sw, sh = sh, sw
    p = subprocess.Popen(["ffmpeg", "-v", "info", "-hide_banner", "-nostats", "-skip_frame", "nokey", "-i", src,
                          "-an", "-vf",
                          f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{STEP})',scale={sw}:{sh},showinfo",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times = []

    def err():
        for line in p.stderr:
            m = re.search(rb"pts_time:([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
    th = threading.Thread(target=err); th.start()
    frames, n = [], sw * sh
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        frames.append(np.frombuffer(b, np.uint8).reshape(sh, sw))
    p.wait(); th.join()
    pt, bx = d["pose_t"], d["boxes"]
    foc = np.full(bx.shape[:2], np.nan, np.float32)
    best = {}                                       # pose index -> frame
    for t, g in zip(times, frames):
        i = int(np.argmin(np.abs(pt - t)))
        if abs(pt[i] - t) < 0.15:
            best[i] = g
    kt = np.array(sorted(best), int)
    for i, g in best.items():
        for k in range(bx.shape[1]):
            if not np.isnan(bx[i, k, 0]):
                foc[i, k] = box_sharpness(g, bx[i, k], sc)
    # spread each keyframe's values to the pose frames around it (boxes there are matched by overlap)
    full = np.full_like(foc, np.nan)
    if len(kt):
        for i in range(len(pt)):
            j = kt[np.argmin(np.abs(pt[kt] - pt[i]))]
            if abs(pt[j] - pt[i]) > 0.6:
                continue
            for k in range(bx.shape[1]):
                if np.isnan(bx[i, k, 0]):
                    continue
                ious = [_iou(bx[i, k], bx[j, q]) if not np.isnan(bx[j, q, 0]) else 0 for q in range(bx.shape[1])]
                q = int(np.argmax(ious))
                if ious[q] > 0.3:
                    full[i, k] = foc[j, q]
    np.savez_compressed(out, focus=full, key_t=pt[kt] if len(kt) else np.zeros(0))
    return f"{name}: {len(frames)} keyframes"


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def run(project, workers: int = 3) -> None:
    todo = {}
    for u in project.load("units.json"):
        if u["score"] not in todo and os.path.exists(project.path("scores", u["score"] + ".npz")):
            todo[u["score"]] = (project.decode_from(u["source"], 0.0)[0], bool(u.get("rotate")))
    with ProcessPoolExecutor(max(1, workers)) as ex:
        futs = [ex.submit(focus_one, project.path("scores"), k, src, rot) for k, (src, rot) in todo.items()]
        for f in futs:
            print("[reel] focus", f.result(), flush=True)
