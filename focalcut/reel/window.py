"""The night lift as a colourist does it: a tracked window on the faces, not a curve over the frame.

For a night shot whose people sit dark (reel.json "exposure" "lift"), the shot's own picture -- rotated and cropped
exactly as it plays -- is read at SAMPLE_FPS and YuNet finds the faces in it. Their level decides how far the shot
is lifted (toward "target", never more than "stops" of exposure, nothing for a face that is already bright or one
that is lost in black: a silhouette is meant to be one). Each face then has a soft window: an ellipse from the top
of the head to the chest, feathered wide, and a gentler one over the body below it (shoulders to hips), following
the face from sample to sample. Inside it a shadow lift raises the face and its colour together (Cb/Cr scale with the lift, so skin keeps its colour instead of going grey);
outside it the room stays exactly as shot, so the night stays dark.

The window is a small greyscale video (y4m, MASK_W wide) composited by the shot's own ffmpeg pass (render.shot_filter:
split, lutyuv, alphamerge, overlay), so it costs a few seconds per shot and nothing at all for shots it leaves
alone. Analyses and windows are cached in <reel>/work/lift by what shapes them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading

import numpy as np

from . import hw

VERSION = 4
SAMPLE_FPS = 10            # analysis frames per second of source
GRID = 640                 # analysis size: the crop's long side, in pixels
MASK_W = 160               # the window video's long side (soft: it is scaled up with the shot)
FACE_SCORE = 0.7           # YuNet score for a face the window follows
FACE_MIN = 0.045           # smallest face lit, as a share of the crop's height
SEEN = 0.3                 # faces in fewer of the samples than this: no lift (a passing face, a crowd in silhouette)
DARKEST = 0.05             # a face below this level is a silhouette or noise: left dark
TARGET = 0.40              # the face level a lift aims for (legal-range luma, 0..1)
STOPS = 0.6                # the most a face is brightened, in stops
A_MAX = 1.0                # the shadow curve's strength cap (the gain right above black, 1 + a, stays under 2x)
CHROMA = 0.5               # colour rises by this share of the face's luma gain (skin keeps its colour, the dark
                           # room around the face doesn't gain colour noise)
SAT = (0.07, 0.16)         # a face's chroma: up to the first, its colour rises with it; past the second, stays
TOE = (0.02, 0.10)         # the lift fades in over these levels: black and the deep shadows stay as shot
# the window around a face of width w, height h centred (cx, cy): an ellipse centred DROP * h below the face, with
# semi-axes RX * w and RY * h, full strength inside CORE of its radius and gone at its edge
DROP, RX, RY, CORE = 0.6, 1.5, 1.9, 0.45
# and below it the body, from the shoulders to the hips, lifted BODY as much as the face (a little: the face leads)
BODY_DROP, BODY_RX, BODY_RY, BODY_CORE, BODY = 4.0, 2.1, 3.2, 0.35, 0.55


def settings(v) -> dict | None:
    """reel.json "exposure" "lift" as the window's settings: true, or {"target": 0.4, "stops": 0.6}; None when off or
    when "window": false asks for the old curve over the whole frame."""
    if not v:
        return None
    v = v if isinstance(v, dict) else {}
    if v.get("window") is False:
        return None
    return {"window": VERSION, "target": float(v.get("target", TARGET)), "stops": float(v.get("stops", STOPS))}


def toe(y):
    """0 at black rising smoothly to 1 by TOE[1]: the lift leaves the deepest shadows (the night, the noise floor, the
    dark room just around a face) as they were."""
    e = min(max((y - TOE[0]) / (TOE[1] - TOE[0]), 0.0), 1.0)
    return e * e * (3 - 2 * e)


def lifted(y, a):
    """A level y (legal-range luma, 0..1) after a lift of strength a."""
    return y + a * y * (1 - y) ** 2 * toe(y)


def amount(level, target=TARGET, stops=STOPS, sat=0.0):
    """(a, k): the shadow curve's strength and the colour gain that bring a face at level up toward target, by at most
    stops; (0, 1) for a face already there or lost in black. sat: the face's colour (its chroma, 0..0.5): skin under
    coloured stage light (SAT) keeps the colour it has instead of going neon."""
    if level is None or level < DARKEST or level >= target:
        return 0.0, 1.0
    want = min(target, level * 2 ** stops)
    a = min(A_MAX, (want - level) / max(level * (1 - level) ** 2 * toe(level), 1e-6))
    keep = min(max((SAT[1] - sat) / (SAT[1] - SAT[0]), 0.0), 1.0)
    return round(a, 4), round(1 + CHROMA * keep * (lifted(level, a) / level - 1), 4)


def curve(a, k):
    """lutyuv expressions (10-bit) for a lift of strength a: lifted() over legal range (most in the shadows and low
    midtones where a dark face sits; black, the toe and white where they were), colour scaled by k."""
    A = a / 876.0 ** 2
    t0, t1 = 64 + 876 * TOE[0], 876 * (TOE[1] - TOE[0])
    e = f"clip((val-{t0:.4f})/{t1:.4f}\\,0\\,1)"
    y = (f"if(between(val\\,64\\,940)\\,val+{A:.6e}*(val-64)*(940-val)*(940-val)*{e}*{e}*(3-2*{e})\\,val)")
    c = f"clip(512+(val-512)*{k:.4f}\\,64\\,960)"
    return y, c


_local = threading.local()


def _detector():
    """YuNet, one per thread: shots render side by side, and one detector shared between them breaks mid-frame."""
    det = getattr(_local, "det", None)
    if det is None:
        import cv2

        from .people import models
        det = _local.det = cv2.FaceDetectorYN.create(models()["det"], "", (GRID, GRID), FACE_SCORE, 0.3, 5000)
    return det


def _fps(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                        "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip()
    n, d = (r.split("/") + ["1"])[:2]
    return int(n), int(d)


def _vf(*parts):
    return ",".join(p.strip(",") for p in parts if p and p.strip(","))


def _grab(project, s, fmt, dur, crop_chain):
    """The shot's picture as it plays (rotated, cropped), SAMPLE_FPS frames a second, GRID on its long side: BGR."""
    src, t = project.decode_from(s["unit"]["source"], s["src_in"])
    exe, dec, P, pre = hw.ffmpeg(src, True)
    size = subprocess.run([exe, "-v", "info", "-hide_banner", *dec, "-ss", f"{t:.3f}", "-t", "0.05", "-i", P(src), "-an",
                           "-vf", _vf(pre, crop_chain, "showinfo"), "-frames:v", "1", "-f", "null", "-"],
                          capture_output=True, text=True)
    m = re.search(r" s:(\d+)x(\d+)", size.stderr)
    cw, ch = (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)
    sc = GRID / max(cw, ch)
    w, h = max(2, int(round(cw * sc / 2)) * 2), max(2, int(round(ch * sc / 2)) * 2)
    raw = subprocess.run([exe, "-v", "error", *dec, "-ss", f"{t:.3f}", "-t", f"{dur:.3f}", "-i", P(src), "-an",
                          "-vf", _vf(pre, crop_chain, f"fps={SAMPLE_FPS},scale={w}:{h}:flags=area"),
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), src


def _faces(frames):
    """Per sample: faces as (cx, cy, w, h) in pixels, the median level (0..1 luma) and colour (chroma) inside each."""
    det = _detector()
    out = []
    for f in frames:
        h, w = f.shape[:2]
        det.setInputSize((w, h))
        _, found = det.detect(f)
        got = []
        if found is not None:
            rgb = f[..., ::-1].astype(np.float32) / 255.0
            y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            chroma = np.hypot((rgb[..., 2] - y) / 1.8556, (rgb[..., 0] - y) / 1.5748)
            for x0, y0, fw, fh, *_rest, score in found:
                if score < FACE_SCORE or fh < FACE_MIN * h:
                    continue
                ix0, iy0 = int(max(x0 + 0.2 * fw, 0)), int(max(y0 + 0.25 * fh, 0))   # the face, not hair or background
                ix1, iy1 = int(min(x0 + 0.8 * fw, w)), int(min(y0 + 0.85 * fh, h))
                if ix1 - ix0 < 3 or iy1 - iy0 < 3:
                    continue
                got.append((float(x0 + fw / 2), float(y0 + fh / 2), float(fw), float(fh),
                            float(np.median(y[iy0:iy1, ix0:ix1])), float(np.median(chroma[iy0:iy1, ix0:ix1]))))
        out.append(got)
    return out


def _window(faces, w, h):
    """One sample's window, (h, w) float in 0..1: the union of every face's feathered ellipse (head to chest, full
    strength) and its body's below it (shoulders to hips, BODY strength)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = np.zeros((h, w), np.float32)

    def oval(cx, cy, rx, ry, core):
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        e = np.clip((1.0 - d) / (1.0 - core), 0.0, 1.0)
        return e * e * (3 - 2 * e)                            # smoothstep: no edge to see
    for cx, cy, fw, fh, *_ in faces:
        m = np.maximum(m, oval(cx, cy + DROP * fh, RX * fw, RY * fh, CORE))
        m = np.maximum(m, BODY * oval(cx, cy + BODY_DROP * fh, BODY_RX * fw, BODY_RY * fh, BODY_CORE))
    return m


def _write_y4m(path, masks, fps):
    h, w = masks[0].shape
    with open(path + ".part", "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F{fps[0]}:{fps[1]} Ip A1:1 Cmono\n".encode())
        for m in masks:
            f.write(b"FRAME\n")
            f.write(np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
    os.replace(path + ".part", path)


def prepare(project, s, fmt, dur, crop_chain):
    """The window for a shot whose s["lift"] asks for one, over dur seconds of source from s["src_in"]: (mask file, a, k,
    report) or None when the shot is left as shot. Cached in <reel>/work/lift."""
    cfg = s.get("lift")
    if not isinstance(cfg, dict):
        return None
    d = project.path("work", "lift")
    os.makedirs(d, exist_ok=True)
    name = hashlib.md5(json.dumps([s["unit"]["source"], round(s["src_in"], 3), round(dur, 3), crop_chain, cfg],
                                  sort_keys=True).encode()).hexdigest()[:12]
    meta_f, mask_f = os.path.join(d, name + ".json"), os.path.join(d, name + ".y4m")
    if os.path.exists(meta_f):
        meta = json.load(open(meta_f))
        if not meta.get("a") or os.path.exists(mask_f):
            return (mask_f, meta["a"], meta["k"], meta) if meta.get("a") else None
    frames, src = _grab(project, s, fmt, dur, crop_chain)
    meta = {"a": 0.0, "k": 1.0, "level": None, "seen": 0.0}
    if len(frames):
        faces = _faces(frames)
        seen = sum(1 for f in faces if f) / len(faces)
        weights = [x[2] * x[3] for f in faces for x in f]
        level = float(np.average([x[4] for f in faces for x in f], weights=weights)) if weights else None
        sat = float(np.average([x[5] for f in faces for x in f], weights=weights)) if weights else 0.0
        meta.update(seen=round(seen, 2), level=round(level, 3) if level is not None else None, sat=round(sat, 3))
        if seen >= SEEN:
            a, k = amount(level, cfg["target"], cfg["stops"], sat)
            meta.update(a=a, k=k)
        if meta["a"]:
            h, w = frames.shape[1:3]
            sc = MASK_W / max(w, h)
            mw, mh = max(2, int(round(w * sc))), max(2, int(round(h * sc)))
            small = [[(cx * sc, cy * sc, fw * sc, fh * sc) for cx, cy, fw, fh, *_ in f] for f in faces]
            keys = np.array([_window(f, mw, mh) for f in small])
            if len(keys) >= 3:                                # a face missed for a sample doesn't blink
                keys = np.maximum(keys, 0.5 * (np.roll(keys, 1, 0) + np.roll(keys, -1, 0)))
            fps = _fps(src)
            n = int(math.ceil(dur * fps[0] / fps[1])) + 2
            ts = np.arange(n) * fps[1] / fps[0] * SAMPLE_FPS     # in samples
            i0 = np.clip(np.floor(ts).astype(int), 0, len(keys) - 1)
            i1 = np.clip(i0 + 1, 0, len(keys) - 1)
            fr = np.clip(ts - np.floor(ts), 0, 1)[:, None, None]
            _write_y4m(mask_f, (1 - fr) * keys[i0] + fr * keys[i1], fps)
    with open(meta_f, "w") as f:
        json.dump(meta, f)
    return (mask_f, meta["a"], meta["k"], meta) if meta["a"] else None


def graph(a, k, mask_in, tag="w"):
    """Filter-graph text that takes the cropped picture (before it, a chain; after it, the rest of the chain) and lifts
    it through the window from the input labelled mask_in."""
    y, c = curve(a, k)
    return (f"format=yuv422p10le,split[{tag}b][{tag}l];[{tag}l]lutyuv=y={y}:u={c}:v={c}[{tag}x];"
            f"[{mask_in}]format=gray[{tag}m0];[{tag}m0][{tag}x]scale2ref=w=rw:h=rh:flags=bicubic[{tag}m][{tag}y];"
            f"[{tag}y][{tag}m]alphamerge[{tag}a];[{tag}b][{tag}a]overlay=format=yuv422p10:eof_action=repeat,")
