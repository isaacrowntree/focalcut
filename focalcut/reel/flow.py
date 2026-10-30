"""Which way a moment moves, for cutting on the action: a turn cuts to a turn, movement to the right keeps going
right. Optical flow inside the subject's box minus the background's (the camera pans with a dancer, so the room
sliding left means they travel right), just before and just after the moment's peak. Cached in flow_cache.json, so
re-planning never decodes twice.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

WIN, RATE, SIDE = 0.8, 10, 480                    # seconds each side of the peak, frames per second, analysis width


class Flow:
    def __init__(self, project):
        self.project = project
        self.path = project.path("flow_cache.json")
        self.cache = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self._loaded = {}

    def direction(self, unit, t):
        """(dx_in, dx_out): the subject's horizontal travel before and after t, in frame widths per second
        (+ = right)."""
        k = f"{unit['id']}@{t:.2f}"
        if k not in self.cache:
            self.cache[k] = self._measure(unit, t)
        return tuple(self.cache[k])

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.cache, f)

    def _measure(self, unit, t):
        import cv2

        from . import frame as FR
        a = max(unit["t0"], t - WIN)
        path, ts = self.project.decode_from(unit["source"], a)
        land = unit["orient"] == "landscape"
        w, h = (SIDE, SIDE * 9 // 16) if land else (SIDE * 9 // 16, SIDE)
        rot = [unit["rotate"]] if unit.get("rotate") else []
        sw, sh = (h, w) if rot else (w, h)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts:.3f}", "-t", f"{2 * WIN:.3f}", "-i", path, "-an",
                              "-vf", ",".join([f"fps={RATE}", f"scale={sw}:{sh}", *rot]), "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True).stdout
        g = np.frombuffer(raw, np.uint8)[: len(raw) // (w * h) * w * h].reshape(-1, h, w)
        if len(g) < 3:
            return [0.0, 0.0]
        times = a + (np.arange(len(g) - 1) + 0.5) / RATE
        score = self.project.path("scores", unit["score"] + ".npz")
        if not os.path.exists(score):
            return [0.0, 0.0]
        if unit["score"] not in self._loaded:
            self._loaded[unit["score"]] = FR.load(score)
        union, _, _ = FR.track(self._loaded[unit["score"]], times)
        dx = []
        for i in range(len(g) - 1):
            fl = cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., 0]
            inside = np.zeros((h, w), bool)
            if not np.isnan(union[i, 0]):
                x0, y0, x1, y1 = (union[i] * [w, h, w, h]).astype(int)
                inside[max(0, y0):y1, max(0, x0):x1] = True
            if inside.sum() < 50 or (~inside).sum() < 50:
                dx.append(np.nan)
                continue
            dx.append(float(np.median(fl[inside]) - np.median(fl[~inside])) * RATE / w)
        dx = np.array(dx)
        pre, post = dx[times < t], dx[times >= t]

        def med(v):
            return float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
        return [round(med(pre), 4), round(med(post), 4)]


def still(unit, t):
    """No measurement (tests, or no footage to decode): nobody travels."""
    return (0.0, 0.0)
