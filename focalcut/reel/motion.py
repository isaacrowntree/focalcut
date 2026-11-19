"""The picture's own movement, shot by shot: how fast the people in it move, where a movement lands, and whether it
flows or punches -- what the planner matches to the music's movement (music.movement, plan.follow_motion).

What is adopted, and from where:

  Davis & Agrawala, "Visual Rhythm and Beat" (2018). Their directogram is the optical flow's magnitude binned by
      direction, frame by frame; its negative flux per direction, summed, is the impact envelope: sudden
      deceleration or a change of direction -- a foot planted, a pose hit, an arm stopping, a turn finishing. Its
      peaks are the visual beats. directogram() and impact() are that, on the flow left once the camera's own
      motion (the median flow) is taken out, inside the boxes of the people in the frame (the score step's).
  Kinematic beats in dance research: the dancer's beat sits where the motion energy falls to a local minimum after
      a burst. settle() scores exactly that at a source time: a burst, then a rest (velocity._rest's idea, on a
      25 fps curve instead of 10).
  M2Beats / AIST-M2B (IJCAI 2024): motion rhythm and music rhythm are not one to one. So nothing here forces every
      beat onto a visual beat: only the song's structural movements are matched (a stop, a close, an entry) and
      its texture.
  Bellini, Kleiman & Cohen-Or, "Dance to the beat" (2018): they time-warp the video, within a bound, so motion
      beats meet music beats. An editor's pick is never sped up, so the planner moves in-points instead, inside the
      editor's own in-out, and leaves the cuts on the beat grid.
  Editing craft: a gesture finishes before the cut; over-syncing every beat reads as a gimmick. Strong musical
      events and texture carry the match; a shot with neither stays where the other rules put it.

Texture: punch() is the share of a stretch's movement that ends in an impact (impact over speed): staccato dancing
-- hits, pops, jumps, quick head moves -- stops and starts, so it is high; a sway, a gliding arm, hair flowing, a
smooth turn keep moving, so it is low. Stillness is soft.

The motion step writes <reel>/scores/<name>_motion.npz (t, dg: the directogram, frame widths a second per
direction) from each unit's own stretch at RATE fps, the way shake does. Without it series() falls back on the
score step's 10 fps subject motion: coarser, the same shapes. Resumable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RATE, SIDE, BINS = 25, 320, 8
NOISE = 0.15              # flow under this many pixels a frame (at SIDE across) is noise
CHUNK = 40.0              # seconds decoded at a time
PUNCH = (0.07, 0.19)      # punch(): impact over speed that reads as all flow / all punch
STILL = 0.004             # a stretch moving less than this (frame widths a second) is still: soft, at rest


def _grey(path, ts, dur, gpu):
    sw, sh = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                               "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    w, h = SIDE, SIDE * sh // sw // 2 * 2
    r = subprocess.run(["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if gpu else ["-threads", "4"]),
                        "-ss", f"{ts:.3f}", "-t", f"{dur:.3f}", "-i", path,
                        "-an", "-vf", f"fps={RATE},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    if r.returncode and gpu:
        return _grey(path, ts, dur, False)
    raw = r.stdout
    return np.frombuffer(raw, np.uint8)[: len(raw) // (w * h) * w * h].reshape(-1, h, w)


def directogram(g, masks=None):
    """The directogram of grey frames g (n, h, w): per frame, the flow's magnitude summed by direction (BINS of
    them) over the pixels of masks[i] (None: the whole frame), the camera's own motion (the median flow) taken out
    first; in frame widths a frame, averaged over the mask. Row 0 repeats row 1."""
    import cv2
    n = len(g)
    dg = np.zeros((n, BINS), np.float32)
    for i in range(1, n):
        f = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f = f - np.median(f.reshape(-1, 2), 0)
        mag = np.hypot(f[..., 0], f[..., 1])
        mag[mag < NOISE] = 0
        b = ((np.arctan2(f[..., 1], f[..., 0]) + np.pi) / (2 * np.pi) * BINS).astype(int) % BINS
        m = masks[i] if masks is not None and masks[i] is not None else None
        if m is not None and m.sum() >= 50:
            dg[i] = np.bincount(b[m], mag[m], BINS) / m.sum()
        else:
            dg[i] = np.bincount(b.ravel(), mag.ravel(), BINS) / mag.size
    if n > 1:
        dg[0] = dg[1]
    return dg / g.shape[2]


def impact(dg, rate=RATE):
    """(speed, impact) per frame from a directogram: speed is all the movement there is; impact the deceleration --
    how much movement in each direction stopped since the frame before, summed over the directions (Davis &
    Agrawala's impact envelope). Both in frame widths a second, lightly smoothed."""
    k = np.array([1, 2, 1], float) / 4
    sm = np.stack([np.convolve(np.pad(dg[:, j], 1, mode="edge"), k, mode="valid") for j in range(dg.shape[1])], 1) \
        if len(dg) else dg
    v = sm.sum(1) * rate
    fl = np.diff(sm, axis=0, prepend=sm[:1]) * rate
    return v, np.clip(-fl, 0, None).sum(1)


def _masks(d, t, shape):
    """Per frame time, the pixels inside the boxes of the people the score step found (the nearest pose frame within
    0.3 s; a little room around each), or None where it found nobody."""
    h, w = shape
    if d is None or not len(d.get("pose_t", ())):
        return [None] * len(t)
    pt, bx = d["pose_t"], d["boxes"]
    sx, sy = w / d["aw"], h / d["ah"]
    out, last = [], (None, None)
    for x in t:
        i = int(np.argmin(np.abs(pt - x)))
        if abs(pt[i] - x) > 0.3:
            out.append(None)
            continue
        if last[0] != i:
            m = np.zeros((h, w), bool)
            for b in bx[i]:
                if np.isnan(b[0]):
                    continue
                px, py = 0.1 * (b[2] - b[0]), 0.05 * (b[3] - b[1])
                m[max(int((b[1] - py) * sy), 0):int((b[3] + py) * sy) + 1,
                  max(int((b[0] - px) * sx), 0):int((b[2] + px) * sx) + 1] = True
            last = (i, m if m.any() else None)
        out.append(last[1])
    return out


def measure(project, name, units, gpu):
    from . import frame as FR
    from .shake import export_of
    out = project.path("scores", name + "_motion.npz")
    if os.path.exists(out):
        return f"{name}: cached"
    f0 = project.path("scores", name + ".npz")
    d = FR.load(f0) if os.path.exists(f0) else None
    T, D = [], []
    for u in sorted(units, key=lambda u: u["t0"]):
        a = u["t0"]
        while a < u["t1"] - 0.05:
            dur = min(CHUNK, u["t1"] - a) + 1.0 / RATE       # (a frame over: the next chunk starts on a known flow)
            f = export_of(project, u)
            if f:
                g = _grey(f, a - u.get("export_t0", 0.0), dur, gpu)
            else:
                src, ts = project.decode_from(u["source"], a)
                g = _grey(src, ts, dur, False)
            if len(g) < 2:
                break
            t = a + np.arange(len(g)) / RATE
            dg = directogram(g, _masks(d, t, g.shape[1:]))
            keep = t < min(a + CHUNK, u["t1"]) - 1e-6
            T.append(t[keep]); D.append(dg[keep])
            a += CHUNK
    if not T:
        return f"{name}: nothing decoded (not cached; will retry)"
    t, dg = np.concatenate(T), np.concatenate(D)
    order = np.argsort(t, kind="stable")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-4] + f".{os.getpid()}.tmp.npz"                # (whole or not at all: a reader never sees half of it)
    np.savez_compressed(tmp, t=t[order], dg=dg[order].astype(np.float32))
    os.replace(tmp, out)
    return f"{name}: {len(t)} frames"


_CACHE = {}


def series(project, unit):
    """A unit's movement over source time: dict(t, v, imp, rate) -- speed and impact (impact()) -- from the motion
    step's directogram, else from the score step's subject motion (10 fps; its fall from one sample to the next
    stands for the impact), else None."""
    name = unit["score"]
    key = (project.reel_dir, name)
    if key in _CACHE:
        return _CACHE[key]
    f, f0 = project.path("scores", name + "_motion.npz"), project.path("scores", name + ".npz")
    M = None
    if os.path.exists(f):
        z = np.load(f)
        v, imp = impact(z["dg"])
        M = dict(t=z["t"], v=v, imp=imp, rate=RATE)
    elif os.path.exists(f0):
        z = np.load(f0)
        if "subj" in z.files and len(z["t"]) > 2:
            s = np.nan_to_num(np.asarray(z["subj"], float))
            s = np.maximum(s, np.roll(s, 1))                 # (measured on alternate samples: fill the gaps)
            v = np.convolve(s, np.ones(3) / 3, mode="same") * 10 / 320.0   # px a sample at 320 across -> widths / s
            M = dict(t=z["t"], v=v, imp=np.clip(-np.diff(v, prepend=v[:1]), 0, None) * 2.5, rate=10)
    _CACHE[key] = M
    return M


def _mean(M, key, a, b):
    m = (M["t"] >= a) & (M["t"] < b)
    return float(M[key][m].mean()) if m.any() else None


def speed(M, a, b):
    """Mean speed over source a..b (frame widths a second), or None where unmeasured."""
    return _mean(M, "v", a, b)


def punch(M, a, b):
    """0 (it flows, or is still) .. 1 (it punches) over source a..b: the share of the movement that ends in an
    impact. None where unmeasured."""
    v, i = _mean(M, "v", a, b), _mean(M, "imp", a, b)
    if v is None:
        return None
    if v < STILL:
        return 0.0
    return float(np.clip((i / v - PUNCH[0]) / (PUNCH[1] - PUNCH[0]), 0, 1))


def settle(M, t, before=0.6, after=0.6):
    """-1..1: how much the movement comes to rest at source time t -- the speed over `before` seconds up to t
    against the `after` seconds from just past it (1: a burst then stillness; 0: no change; under 0: it picks up).
    A stretch still on both sides scores 0.5: the picture is at rest there, though nothing landed. None where
    unmeasured."""
    b, r = _mean(M, "v", t - before, t), _mean(M, "v", t + 0.08, t + 0.08 + after)
    if b is None or r is None:
        return None
    if b < STILL and r < STILL:
        return 0.5
    return float((b - r) / (b + r + 1e-9))


def burst(M, t, before=0.5, after=0.5):
    """-1..1: how much the movement takes off at source time t (settle() the other way round: a jump, arms going
    up, a spin starting)."""
    b, r = _mean(M, "v", t - before, t), _mean(M, "v", t, t + after)
    if b is None or r is None:
        return None
    if b < STILL and r < STILL:
        return 0.0
    return float((r - b) / (b + r + 1e-9))


def beats(M, a, b, n=None):
    """Visual beats in source a..b: [(time, strength)], the impact envelope's local peaks over the stretch's median,
    the strongest first (n of them at most)."""
    m = (M["t"] >= a) & (M["t"] < b)
    t, x = M["t"][m], M["imp"][m]
    if len(x) < 3:
        return []
    floor = np.median(x)
    pk = [(float(t[i]), float(x[i])) for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > floor]
    return sorted(pk, key=lambda p: -p[1])[:n]


def run(project, workers: int = 3) -> None:
    by = {}
    for u in project.load("units.json"):
        by.setdefault(u["score"], []).append(u)
    gpu = bool(shutil.which("nvidia-smi")) and not os.environ.get("FOCALCUT_SHAKE_CPU")
    with ProcessPoolExecutor(max(1, workers)) as ex:
        futs = [ex.submit(measure, project, k, us, gpu) for k, us in sorted(by.items(), key=lambda kv: -sum(
            u["t1"] - u["t0"] for u in kv[1]))]
        for f in futs:
            print("[reel] motion", f.result(), flush=True)
