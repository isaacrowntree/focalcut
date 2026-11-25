"""Who is holding the thing that makes them the subject -- above all the microphone: at a talk, a wedding speech, a
baptism's testimony, the speaker is the person with the mic, not the pastor standing beside them or the face nearest
the lens.

For each unit of the categories reel.json "holding" names ({"speakers": "microphone"}: category -> what is held), its
export file is sampled every STEP seconds and SAM 3's image detector finds the prop by its name; the holder is the
person (the score step's boxes) whose head or hands are nearest it -- the prop inside their box, close to the top
of it (a mic sits at the mouth, a handheld one at the chest). Writes <reel>/scores/<unit>_holding.npz: t, prop box,
holder box (both 0..1 of the frame, NaN where none), prop score. Resumable; runs in the ComfyUI venv (torch).

template.py uses it: a slot of such a category plays only a stretch where the prop is held throughout (`share`), and
a panel of it is framed on the holder.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

STEP = 0.5             # seconds between samples
SIDE = 960             # frames are read this wide
MIN_SCORE = 0.35       # SAM 3 score for the prop
SHARE = 0.8            # a stretch counts as held when the prop is found in this share of its samples


def settings(cfg) -> dict:
    """reel.json "holding": {"<category>": "<what is held>"}; {} when off."""
    return dict(cfg.get("holding") or {})


def _frames(path, a, b, w=SIDE):
    sw, sh = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                               "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    h = w * sh // sw // 2 * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(a, 0):.3f}", "-t", f"{b - a:.3f}", "-i", path,
                          "-vf", f"fps={1 / STEP},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3), (w, h)


NOSE, WRISTS = 0, (9, 10)       # COCO keypoints


def _to_box(x, y, box, aspect):
    """Distance from a point to a box (0..1 units, x scaled by the frame's aspect so both axes count alike)."""
    dx = max(box[0] - x, 0.0, x - box[2]) * aspect
    dy = max(box[1] - y, 0.0, y - box[3])
    return float(np.hypot(dx, dy))


def holder(prop, boxes, kp=None, aspect=16 / 9):
    """The person holding a prop: of the people in one frame (boxes (n, 4+) and their COCO keypoints (n, 17, 3), in
    0..1 of the frame), the one whose hand is at it -- the nearest wrist to the prop's box, over the person's height,
    the bigger (nearer the lens) person winning a close call. A face is no proof: someone standing behind the speaker
    has their face right behind the mic in the picture, so a nose counts only (and less) when no wrist is near.
    Without keypoints, the biggest person whose box holds the prop. None when nobody is near it."""
    if prop is None or boxes is None or not len(boxes):
        return None
    px, py = (prop[0] + prop[2]) / 2, (prop[1] + prop[3]) / 2
    best, cost = None, None
    hmax = max((b[3] - b[1] for b in boxes if not np.isnan(b[0])), default=1.0) or 1.0
    for j, b in enumerate(boxes):
        if np.isnan(b[0]):
            continue
        h = max(b[3] - b[1], 1e-6)
        if kp is not None:
            wr = [_to_box(kp[j, k, 0], kp[j, k, 1], prop, aspect) / h for k in WRISTS
                  if np.nan_to_num(kp[j, k, 2]) >= 0.3]
            no = [_to_box(kp[j, NOSE, 0], kp[j, NOSE, 1], prop, aspect) / h + 0.15] \
                if np.nan_to_num(kp[j, NOSE, 2]) >= 0.3 else []
            d = min(wr) if wr and min(wr) <= 0.3 else min(no, default=None)
            if d is None or d > 0.35:
                continue
            c = d - 0.15 * h / hmax
        else:
            if not (b[0] <= px <= b[2] and b[1] <= py <= b[3]):
                continue
            c = -h
        if cost is None or c < cost:
            best, cost = b, c
    return best


def assign(project, u, z=None):
    """(Re)make a unit's holders from its stored prop boxes and the score step's people (no detector run)."""
    from . import frame as FR
    p = project.path("scores", u["id"].replace("#", "_") + "_holding.npz")
    z = z or dict(np.load(p))
    sc = project.path("scores", u["score"] + ".npz")
    d = FR.load(sc) if os.path.exists(sc) else None
    H = []
    for t, pb in zip(z["t"], z["prop"], strict=True):
        hb = None
        if not np.isnan(pb[0]) and d is not None and len(d["pose_t"]):
            j = int(np.argmin(np.abs(d["pose_t"] - t)))
            if abs(d["pose_t"][j] - t) <= 0.5:
                sx, sy = d["aw"], d["ah"]
                bx = d["boxes"][j][:, :4] / [sx, sy, sx, sy]
                kp = d["kp"][j].astype(float).copy()
                kp[..., 0] /= sx
                kp[..., 1] /= sy
                hb = holder(pb, bx, kp)
        H.append(hb[:4] if hb is not None else [np.nan] * 4)
    z["holder"] = np.array(H, float)
    np.savez(p, **z)
    return z


def run(project) -> None:
    want = settings(project.config)
    if not want:
        return
    if os.environ.get("FOCALCUT_DISABLE_CUDNN"):           # (some CUDA installs ship broken cuDNN engines)
        import torch
        torch.backends.cudnn.enabled = False
    from ..detect import Sam3ImageDetector
    from .shake import export_of
    det = None
    for u in project.load("units.json"):
        prop = want.get(u["category"])
        if not prop:
            continue
        out = project.path("scores", u["id"].replace("#", "_") + "_holding.npz")
        if os.path.exists(out):
            continue
        ex = export_of(project, u)
        path, x0 = (ex, u.get("export_t0", 0.0)) if ex else (os.path.join(project.folder, u["source"]), 0.0)
        frames, (w, h) = _frames(path, u["t0"] - x0, u["t1"] - x0)
        det = det or Sam3ImageDetector()
        found = det.detect(frames, prompt=prop, min_score=MIN_SCORE)
        T, P, S = [], [], []
        for k, dets in enumerate(found):
            d = max(dets, key=lambda x: x.score) if dets else None
            T.append(u["t0"] + k * STEP)
            P.append(np.array(d.box) / [w, h, w, h] if d else [np.nan] * 4)
            S.append(d.score if d else 0.0)
        z = dict(t=np.array(T), prop=np.array(P, float), score=np.array(S))
        np.savez(out, **z, holder=np.full((len(T), 4), np.nan))
        H = assign(project, u, z)["holder"]
        held = float(np.mean(~np.isnan(H[:, 0]))) if len(H) else 0.0
        print(f"[reel] holding: {u['id']} {prop} held in {held:.0%} of {len(T)} samples", flush=True)


class Holding:
    """The holding files loaded: held(unit, a, b) -> share of a..b's samples with the prop held; at(unit, a, b) ->
    the holder's median box over a..b (0..1), or None."""
    def __init__(self, project):
        self.project, self.want, self._z = project, settings(project.config), {}

    def applies(self, u):
        return u["category"] in self.want

    def _load(self, u):
        if u["id"] not in self._z:
            p = self.project.path("scores", u["id"].replace("#", "_") + "_holding.npz")
            self._z[u["id"]] = dict(np.load(p)) if os.path.exists(p) else None
        return self._z[u["id"]]

    def measured(self, u):
        return self._load(u) is not None

    def held(self, u, a, b):
        z = self._load(u)
        if z is None:
            return 1.0
        m = (z["t"] >= a - STEP / 2) & (z["t"] <= b + STEP / 2)
        return float(np.mean(~np.isnan(z["holder"][m, 0]))) if m.any() else 0.0

    def ever(self, u):
        z = self._load(u)
        return z is None or bool((~np.isnan(z["holder"][:, 0])).sum() >= 4)

    def at(self, u, a, b):
        z = self._load(u)
        if z is None:
            return None
        m = (z["t"] >= a - STEP / 2) & (z["t"] <= b + STEP / 2) & ~np.isnan(z["holder"][:, 0])
        return np.median(z["holder"][m], axis=0) if m.any() else None
