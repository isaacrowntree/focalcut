"""Render a reel from plan_<fmt>.json: beat-locked shots, effect modules, transitions, the music, the stinger.

Shots are cut from the original camera files (or their configured proxies) at 50 fps and the project's size
(reel.json "resolution": 1080p, Instagram's ceiling, unless it says "4k" or another), each through the plan's crop:
a fixed zoom whose centre glides with the subject. Shots carrying effects run through focalcut.fx (speed curves
first, as they change how much source is read); transitions rewrite the tail of one shot and the head of the next
so the cuts stay on the beat. Output: out/<name>_recap_<fmt>.mp4 at -14 LUFS, plus out/<fmt>_shots.json (the final
file used for each shot, read by the Premiere export).

Where the time goes, and what runs beside what (measured: docs/render.md):

  shots       plain shots several at once (DECODERS: Quick Sync and CPU decodes, each with an NVENC session). A
              stabilised shot first has its camera shake measured (vidstab's pass 1: CPU alone, no encoder,
              STAB_JOBS at once, cached per source stretch in ~/.cache/focalcut/stab), then renders as soon as
              a slot is free. A shot whose only effect is a speed curve streams through it (decode, curve and
              encode each on a thread, a few frames held) in one of the same slots, FX_JOBS at once; the other
              effect shots (SAM 3, BiRefNet) go one at a time beside all of those, from the start. Every shot
              is cached in work/ under a key of all that shapes its pixels, which is also its file's name
  overlay     the MilkDrop layer renders beside the shots (it needs the music and the acts, not the pictures); the
              composite runs on the GPU (composite.py), decode and encode on threads of their own
  the rest    transitions (two shots each, streamed: only the frames inside the blend are held), the mix and the
              mux are seconds; the shots are read as one by the overlay and joined straight into the delivered
              file, with no joined copies written in between

Each stage's wall-clock time is printed at the end and saved to out/<cut>_timings.json.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from . import exposure, hw

RAMP_FX = ("dunk", "rise", "reveal", "drift", "none")   # fx.speed.RAMPS: ramps by what moves (reel.json "ramps")
SPEED_FX = ("hero", "bullet", "montage", "flashin", "flashout", "jumpcut", "fit", "glide", *RAMP_FX)

SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}   # 1080p: the default (reel.json "resolution")
FPS = 50
# shots rendered at once, by decoder: the Intel iGPU (hw.ffmpeg) leaves the CPU free for the filters, so it takes
# the most. Every shot holds an NVENC session while it renders and consumer drivers allow 12 at once: these, the
# effect shot beside them and the MilkDrop layer make 11
DECODERS = {"qsv": 6, "cpu": 3}
# vidstab's motion analysis is the heaviest thing a render does on the CPU (about half a core-second per 4K frame).
# It spreads over every core with OpenMP, at two thirds the efficiency of one thread; so each analysis gets
# STAB_THREADS and STAB_JOBS of them run at once (FOCALCUT_STAB_JOBS). Measured on 24 cores, 2,400 frames:
# 24 x 1 thread 52 s, 12 x 2 72 s, 8 x 3 67 s
STAB_THREADS = 1         # (render_shots gives a few analyses more threads each: a re-render of one changed shot)
STAB_JOBS = max(2, os.cpu_count() or 4)
MEM_FLOOR = 4.0          # GB free below which a new analysis or stabilised render waits for others to finish


def stab_jobs(dim) -> int:
    """Analyses at once for a render at size dim: one per core, but each holds about 0.6 GB, and above 1080p a
    stabilised render holds 1.6 GB and an effect shot 10 GB and more, so there no more than a 2.5 GB share of the
    machine's memory each (16 on a 40 GB machine). Past its memory a render swaps and takes twice as long."""
    n = int(os.environ.get("FOCALCUT_STAB_JOBS") or STAB_JOBS)
    if dim[0] * dim[1] > 1920 * 1080 * 1.5 and not os.environ.get("FOCALCUT_STAB_JOBS"):
        n = max(2, min(n, int((hw.mem_gb("MemTotal") or 64) / 2.5)))
    return n
STAB_ANALYSIS = "full"   # reel.json "stab_analysis": "half" measures the shake on a half-size picture (7x less CPU;
#                          the stabilised picture then sits within a pixel or two of the full-size analysis's)


def __getattr__(name):                                      # render.ENC: the encoder arguments (found on first use)
    if name == "ENC":
        return hw.enc()
    raise AttributeError(name)
I, TP = -14.0, -1.5


def size(project, fmt) -> tuple:
    """(width, height) the project renders a format at: reel.json "resolution", else 1080p."""
    f = getattr(project, "size", None)
    return tuple(f(fmt)) if callable(f) else SIZE[fmt]


class Timer:
    """Wall-clock seconds per stage of a render; stages that run beside others are timed on their own."""

    def __init__(self):
        self.t, self.lock, self.t0 = {}, threading.Lock(), time.time()

    def add(self, stage, seconds):
        with self.lock:
            self.t[stage] = self.t.get(stage, 0.0) + seconds

    def __call__(self, stage):
        timer = self

        class Span:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *exc):
                timer.add(stage, time.time() - self.t)
        return Span()

    def report(self) -> dict:
        return {"total": round(time.time() - self.t0, 1), **{k: round(v, 1) for k, v in self.t.items()}}


def crop_expr(path, k):
    """Piecewise-linear crop centre (fraction of the source) over shot time t, as an ffmpeg expression; k picks x (1)
    or y (2) from the path's [t, x, y] points."""
    pts = [(p[0], p[k]) for p in path]
    pieces = [f"({x0}+({x1}-{x0})*(t-{t0})/{max(t1 - t0, 1e-3)})"
              for (t0, x0), (t1, x1) in zip(pts[:-1], pts[1:])] + [f"{pts[-1][1]}"]
    ends = [p[0] for p in pts[1:]]                 # piece i holds while t < ends[i]; the last one after that

    def tree(lo, hi):                              # a balanced if() tree: ffmpeg refuses nesting past ~100 deep,
        if lo == hi:                               # which a chain hits on a long shot (one point per 0.2 s)
            return pieces[lo]
        mid = (lo + hi) // 2
        return f"if(lt(t\\,{ends[mid]})\\,{tree(lo, mid)}\\,{tree(mid + 1, hi)})"
    return tree(0, len(pieces) - 1)


STAB = os.path.expanduser("~/.cache/focalcut/stab")         # vidstab motion files (a path without spaces for ffmpeg)
_stab_locks, _stab_guard = {}, threading.Lock()


def stab(path, t, dur, rot):
    """Pass 1 of vidstab over exactly the stretch the shot decodes: the camera's shake, frame by frame."""
    os.makedirs(STAB, exist_ok=True)
    half = STAB_ANALYSIS == "half"
    name = hashlib.md5(f"{path}|{t:.3f}|{dur:.3f}|{rot}{'|half' if half else ''}".encode()).hexdigest()[:12] + ".trf"
    trf = os.path.join(STAB, name)
    with _stab_guard:                              # (the same stretch asked for twice at once is analysed once)
        lock = _stab_locks.setdefault(name, threading.Lock())
    with lock:
        if not os.path.exists(trf):                # run in STAB with a bare file name: no filter-string escaping
            pre = (rot + ",") if rot else ""
            pre += "scale=iw/2:ih/2:flags=bilinear," if half else ""
            part = f"{name}.{os.getpid()}.part"    # (another render analysing it too writes its own)
            # (below the effect shots and the encodes in the CPU's queue: they are what the render waits for)
            nice = ["nice", "-n", "10"] if shutil.which("nice") else []
            subprocess.run([*nice, "ffmpeg", "-v", "error", "-y", "-threads", str(STAB_THREADS), "-ss", f"{t:.3f}",
                            "-t", f"{dur:.3f}", "-i", path, "-an",
                            "-vf", f"{pre}vidstabdetect=shakiness=6:accuracy=12:result={part}", "-f", "null", "-"],
                           check=True, cwd=STAB, env={**os.environ, "OMP_NUM_THREADS": str(STAB_THREADS)})
            if half:
                scale_motion(os.path.join(STAB, part), 2)
            os.replace(os.path.join(STAB, part), trf)   # (a render stopped mid-pass leaves no half a motion file)
    return trf


def scale_motion(trf, k):
    """A vidstab motion file measured on a picture 1/k the size, rewritten for the full-size one: every local
    motion's vector, field position and field size times k. Only the text format ("VID.STAB 1") is understood."""
    import re
    with open(trf, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"VID.STAB 1"):
        raise RuntimeError("this ffmpeg's vidstab writes binary motion files: reel.json \"stab_analysis\": \"half\" "
                           "needs the text format (libvidstab 1.1.0)")
    lm = re.compile(r"\(LM (-?\d+) (-?\d+) (-?\d+) (-?\d+) (\d+) ")
    out = lm.sub(lambda m: "(LM " + " ".join(str(int(v) * k) for v in m.groups()) + " ", raw.decode())
    with open(trf, "w") as f:
        f.write(out)


def room(wait=180.0):
    """Hold a memory-heavy step back while the machine has under MEM_FLOOR GB free (others are finishing): for a
    while, never for good."""
    t0 = time.time()
    while (hw.mem_gb() or 1e9) < MEM_FLOOR and time.time() - t0 < wait:
        time.sleep(1.0)


def read_length(s) -> float:
    """Seconds of source a plain shot decodes: its length (more of it when paced) and two frames to spare."""
    return s["length"] * (s.get("speed") or 1.0) + 0.1


LOCK_WAIT = 720          # 5 s tries (an hour) for a locked output file before giving up
LONG_ARGS = 24000         # a command longer than this (Windows' limit is 32,767) reads its filter graph from a file


def from_file(args, P, base):
    """args with any -vf / -filter_complex value moved into a file beside base (ffmpeg's -/option), when the command
    would be too long: a long shot's crop path, steadied frame by frame, is thousands of characters."""
    if sum(len(a) + 1 for a in args) <= LONG_ARGS:
        return args
    out = list(args)
    for i, a in enumerate(out[:-1]):
        if a in ("-vf", "-filter_complex"):
            f = f"{base}.{a.strip('-')}.txt"
            with open(f, "w") as fh:
                fh.write(out[i + 1])
            out[i], out[i + 1] = "-/" + a.strip("-"), P(f)
    return out


def frame_part(s):
    """The rotation and the crop that follows the subject: the picture as the shot plays it, before any grade."""
    u, c = s["unit"], s.get("crop")
    f = [u["rotate"]] if u.get("rotate") else []
    if s.get("gain"):                                       # an exposure matched to its scene's (before the faces
        f.append(exposure.gain_filter(s["gain"], s.get("gain_protect")))          # are judged for a lift)
    if c:
        f.append(f"crop=w=iw*{c['w']}:h=ih*{c['h']}:x=({crop_expr(c['path'], 1)})*iw-ow/2"
                 f":y=({crop_expr(c['path'], 2)})*ih-oh/2")
    return ",".join(f)


STAB_CAP = 0.012         # steady_crop: the most the crop moves against the camera's jitter, of the frame's width
STAB_SIGMA = 0.25        # steady_crop: the camera's path is smoothed over this many seconds (a pan or a follow stays)
STAB_ZOOM = 0.03         # steady_crop: the crop is this much smaller, for the room to move in
STAB_SIGN = 1.0
STAB_GAIN = 0.85         # steady_crop: steadied only when that leaves this share of the shake or less


STEADY = os.path.expanduser("~/.cache/focalcut/steady")    # shot_motion's measurements
STEADY_SIDE = 960        # shot_motion's analysis width


def shot_motion(project, source, a, dur, qsv=True, frames=False):
    """The camera's frame-to-frame motion over exactly the stretch a shot decodes, at FPS: (source times, (dx, dy) in
    frame widths). Measured on the decoded frames themselves (shake.motion: corners, RANSAC), so a correction lands on
    the frame it belongs to -- the stored 25 fps series, timed from the export file, lands a frame or two off, and a
    jitter corrected a frame late is a jitter doubled. Cached in ~/.cache/focalcut/steady."""
    from .shake import frames_motion
    path, t = project.decode_from(source, a)
    exe, dec, P, pre = hw.ffmpeg(path, qsv)                 # (the decoder the shot renders with: Quick Sync and the
    kind = "qsv" if dec else "cpu"                          #  CPU land a seek two frames apart, and a correction
    os.makedirs(STEADY, exist_ok=True)                      #  two frames late doubles the jitter)
    f = os.path.join(STEADY, hashlib.md5(f"{path}|{t:.3f}|{dur:.3f}|{FPS}|{kind}|{STEADY_SIDE}".encode()).hexdigest()[:12] + ".npy")
    if os.path.exists(f) and not frames:
        mv = np.load(f)
        g = None
    else:
        sw, sh = frame_size(path)
        w = STEADY_SIDE                                     # (sub-pixel jitter at shake.py's 320 px is noise: a
        h = w * sh // sw // 2 * 2                           #  correction made of noise adds wobble)
        raw = subprocess.run([exe, "-v", "error", *dec, "-ss", f"{max(t, 0.0):.3f}", "-t", f"{dur:.3f}", "-i", P(path),
                              "-an", "-vf", f"{pre}fps={FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        g = np.frombuffer(raw, np.uint8)[: len(raw) // (w * h) * w * h].reshape(-1, h, w)
        mv = np.load(f) if os.path.exists(f) else frames_motion(g)
        np.save(f, mv)
    return (a + np.arange(len(mv)) / FPS, mv, g) if frames else (a + np.arange(len(mv)) / FPS, mv)


def steadies(project, source, a, dur, qsv, jit):
    """Does the correction steady this shot? Tried on its own frames: each moved by its correction, the camera's
    motion measured again (corners, RANSAC) on the middle of the frame, kept only when that leaves STAB_GAIN of the
    shake or less -- where the jitter is finer than the measurement or the dancers fill the frame, a correction can
    add as much as it takes away (SSMC's animation shots: one of three came out a quarter shakier). Cached."""
    import cv2

    from .shake import frames_motion, hf
    key = hashlib.md5(f"{source}|{a:.3f}|{dur:.3f}|{qsv}|{STAB_SIGMA}|{STAB_CAP}|{STAB_SIGN}|{STEADY_SIDE}".encode())
    f = os.path.join(STEADY, key.hexdigest()[:12] + ".json")
    if os.path.exists(f):
        try:
            with open(f) as fh:
                return json.load(fh)["ok"]
        except (ValueError, KeyError):                       # (a run killed mid-write: measured again)
            os.remove(f)
    _, mv, g = shot_motion(project, source, a, dur, qsv, frames=True)
    n = min(len(g), len(jit))
    if n < 8:
        return False
    hh, ww = g.shape[1:]
    moved = np.stack([cv2.warpAffine(g[i], np.float32([[1, 0, -jit[i, 0] * ww], [0, 1, -jit[i, 1] * hh]]), (ww, hh),
                                     borderMode=cv2.BORDER_REPLICATE) for i in range(n)])
    m = (slice(None), slice(hh // 10, hh - hh // 10), slice(ww // 10, ww - ww // 10))
    before, after = float(hf(frames_motion(g[:n][m])).mean()), float(hf(frames_motion(moved[m])).mean())
    ok = bool(before and after <= STAB_GAIN * before)
    with open(f + ".part", "w") as fh:                      # (whole or not at all)
        json.dump(dict(ok=ok, before=before, after=after), fh)
    os.replace(f + ".part", f)
    return ok


def steady_crop(project, s, dur, qsv=True):
    """A light steadying for a shaky shot (plan "stab"; reel.json "stabilise"), as the crop rather than a warp: the
    camera's own frame-to-frame motion (shake.py: corners tracked, a RANSAC fit, the dancers its outliers) summed to
    its path, the path smoothed over STAB_SIGMA, and the crop moved with what is left -- the jitter -- by at most
    STAB_CAP, the crop STAB_ZOOM smaller for room. A pan, a tilt or a follow stays as shot; nothing is warped, no
    frame is guessed, and a dancer filling the frame can't throw it (vidstab followed the dancers here: SSMC's
    animation shots came out 7-9x shakier, a frame lurching a third of the strip). None when unmeasured or rotated."""
    u = s["unit"]
    if u.get("rotate"):
        return None
    a, b = s["src_in"], s["src_in"] + dur
    try:
        t, mv = shot_motion(project, u["source"], a, dur, qsv)
    except subprocess.CalledProcessError:                    # (the iGPU out of sessions: the CPU reads it)
        if not qsv:
            raise
        t, mv = shot_motion(project, u["source"], a, dur, False)
    if len(t) < 4:
        return None
    aspect = 16 / 9
    pos = np.cumsum(mv * np.array([1.0, aspect]), 0)          # (x in frame widths, y in frame heights)
    import fnmatch
    cfg_ = (project.config if project is not None else {})
    strong = any(fnmatch.fnmatch(u.get("clip", ""), q) or u.get("category") == q            # reel.json "stab_strong":
                 for q in (cfg_.get("stab_strong") or []))   # steadied harder, the crop tighter
    hard = {"sigma": STAB_SIGMA, "cap": STAB_CAP * 2.5, "zoom": STAB_ZOOM * 2.3, **(cfg_.get("stab_strong_settings") or {})}
    sig = (hard["sigma"] if strong else STAB_SIGMA) * FPS      # (reel.json "stab_strong_settings": {"sigma": s,
                                                               # "cap": share, "zoom": share} -- a camera that sways)
    r = int(round(3 * sig))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()   # (odd: centred)
    half = len(k) // 2
    smooth = np.stack([np.convolve(np.pad(pos[:, j], half, mode="edge"), k, mode="valid") for j in range(2)], 1)
    cap, zoom = (hard["cap"], hard["zoom"]) if strong else (STAB_CAP, STAB_ZOOM)
    jit = np.clip(STAB_SIGN * (pos - smooth), -cap * np.array([1.0, aspect]), cap * np.array([1.0, aspect]))
    # (the crop moves against the motion tracked: measured, this is the way that steadies -- the other doubles it)
    keep = (t >= a) & (t <= b)
    try:
        ok = steadies(project, u["source"], a, dur, qsv, jit)
    except subprocess.CalledProcessError:                    # (the iGPU out of sessions beside other shots: the CPU)
        ok = steadies(project, u["source"], a, dur, False, jit) if qsv else False
    if not ok:
        return None
    c = s.get("crop") or dict(w=1.0, h=1.0, z=1.0, path=[[0.0, 0.5, 0.5]])
    w, h = c["w"] * (1 - zoom), c["h"] * (1 - zoom)
    pts = np.array(c["path"], float)
    tt = t[keep] - a
    bx = np.interp(tt, pts[:, 0], pts[:, 1])
    by = np.interp(tt, pts[:, 0], pts[:, 2])
    x = np.clip(bx + jit[keep, 0], w / 2, 1 - w / 2)
    y = np.clip(by + jit[keep, 1], h / 2, 1 - h / 2)
    return dict(c, w=round(w, 5), h=round(h, 5), steadied=True,
                path=[[round(float(q), 3), round(float(px), 5), round(float(py), 5)] for q, px, py in zip(tt, x, y)])


def lift_window(project, s, fmt, dur):
    """The night lift's tracked window for a shot (window.prepare), or None: (mask file, a, k, report)."""
    if not isinstance(s.get("lift"), dict) or project is None:
        return None
    from . import window
    return window.prepare(project, s, fmt, dur, frame_part(s))


_SEGMENTS: dict = {}


def lut_cube(project, u, at=None):
    """The unit's camera look as a .cube path (project-relative or absolute), or None: reel.json "lut_segments" (a
    JSON file of {clip: [[t0, t1, cube, cut], ...]}, cubes relative to its folder -- a clip whose camera changed
    part way, an ISO step or the fade drifting) by the stretch holding `at` (the shot's middle, source seconds),
    else reel.json "lut": {"<clip pattern>": "<.cube>"}."""
    if project is None or not u.get("source"):
        return None
    import fnmatch
    seg_f = project.config.get("lut_segments")
    if seg_f and at is not None:
        path = seg_f if os.path.isabs(seg_f) else project.path(seg_f)
        if path not in _SEGMENTS:
            _SEGMENTS[path] = json.load(open(path)) if os.path.exists(path) else {}
        rows = _SEGMENTS[path].get(u.get("clip", ""))
        if rows:
            row = next((r for r in rows if r[0] <= at < r[1]), rows[0] if at < rows[0][0] else rows[-1])
            return os.path.join(os.path.dirname(path), row[2])
    return next((c for pat, c in (project.config.get("lut") or {}).items()
                 if fnmatch.fnmatch(u.get("clip", ""), pat)), None)


def lut_filter(project, u, P=None, at=None):
    """The unit's camera look (lut_cube) as an ffmpeg lut3d, its path as the decoding ffmpeg sees it; None for
    footage without one."""
    cube = lut_cube(project, u, at)
    if not cube:
        return None
    path = cube if os.path.isabs(cube) else project.path(cube)
    path = (P or (lambda x: x))(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"lut3d=file='{path}':interp=tetrahedral"


def mid_of(s, length=None):
    """A shot's (or a panel's) middle in source seconds: where its look is read."""
    n = length if length is not None else s.get("length", 0.0) * (s.get("speed") or 1.0)
    return float(s.get("src_in", 0.0)) + n / 2


def shot_filter(s, fmt, project=None, dur=None, win=None, mask_in="1:v", qsv=True, P=None):
    """The shot's filter chain: rotation, stabilisation (a shaky stretch: the plan's "stab"), the crop that follows
    the subject, the project's size at FPS. win (lift_window's): the night lift through a window read from the input
    mask_in, which makes the chain a filter graph for -filter_complex."""
    W, H = size(project, fmt)
    u = s["unit"]
    f = [u["rotate"]] if u.get("rotate") else []
    lut = lut_filter(project, u, P, at=mid_of(s))
    if lut:                                                 # the camera's look: matched to the hero camera's, and
        f.append(lut)                                       # the reference film's colour over all of them
    if s.get("gain"):                                       # the exposure matched to its scene's (exposure.match)
        f.append(exposure.gain_filter(s["gain"], s.get("gain_protect")))
    c = s.get("crop")
    if s.get("stab") and project is not None:               # a light steadying: the crop rides the camera's jitter
        c = steady_crop(project, s, dur or s["length"] + 0.1, qsv) or c
    if c:
        f.append(f"crop=w=iw*{c['w']}:h=ih*{c['h']}:x=({crop_expr(c['path'], 1)})*iw-ow/2"
                 f":y=({crop_expr(c['path'], 2)})*ih-oh/2")
    if s.get("pull") or (s.get("lift") and not isinstance(s["lift"], dict)):   # a blown-out subject: highlights
        f.append(exposure.filt(s.get("pull"), s.get("lift")))   # pulled back; (the old curve) a dark one lifted
    if win:                                                 # a dark subject at night: lifted through its window
        from . import window
        f.append(window.graph(win[1], win[2], mask_in, kn=win[3].get("knee")).rstrip(","))
    p = s.get("speed") or 1.0                               # the shot's pace (plan "pace"): source read p times
    if p != 1.0 and not s.get("fx"):                        # as fast, the nearest frame each (a speed effect
        f.append(f"setpts=PTS/{p:.4f}")                     # applies it inside its curve instead)
    rate, after = finish_filters(cam_finish(project, u))    # the camera matched beyond colour (camera_finish)
    if s.get("reverse"):                                    # played backwards (reel.json "reverse"): reversed at
        f += [f"scale={W}:{H}:flags=lanczos", "format=yuv420p", *after, "reverse", rate, "setsar=1"]   # the source's
        return ",".join(f)                                  # own rate, 8-bit, so the frames held stay few
    f += [f"scale={W}:{H}:flags=lanczos", rate, "setsar=1", "format=yuv420p", *after, *shot_finish(project, s)]
    return ",".join(f)


def cam_finish(project, u):
    """reel.json "camera_finish": {"<clip pattern>": {"sharpen": -0.35, "rolloff": 0.8, "smooth": true}} -- what
    makes a camera sit beside the hero camera beyond colour: its sharpness matched ("sharpen", unsharp's amount at
    the output size: below 0 softens an over-sharpened camera, above 0 crisps a soft one), the brightest tones
    eased ("rolloff": the level, 0..1 of legal range, above which highlights are compressed a little -- a hard
    clip sits under white, as the hero's gentle shoulder does), and a low-frame-rate camera's motion blended to the
    film's rate ("smooth": each frame between two of its own, not held -- 25p in a 50p film stops stepping). {}."""
    import fnmatch
    if project is None:
        return {}
    for pat, v in (project.config.get("camera_finish") or {}).items():
        if fnmatch.fnmatch(u.get("clip", ""), pat):
            return v
    return {}


def shot_finish(project, s):
    """reel.json "shot_finish": [{"clip": "C2211", "at": 161.8, "black": 0.04, "saturation": 1.45}] -- one shot
    matched to its neighbours by hand (an overexposed take the exposure step brought down but left milky: its black
    point pulled down by `black` of the range, its colour back by `saturation`), the shot at `at` s of the clip
    (within 3 s of its in-point). [] for every other shot."""
    if project is None:
        return []
    u = s.get("unit") or {}
    for e in project.config.get("shot_finish") or []:
        if e.get("clip") in (u.get("clip"), u.get("id")) and abs(float(e.get("at", s.get("src_in", 0))) -
                                                                 float(s.get("src_in", 0))) <= 3.0:
            out = []
            if e.get("black"):
                b = float(e["black"])
                out.append(f"colorlevels=rimin={b:.3f}:gimin={b:.3f}:bimin={b:.3f}")
            if e.get("saturation"):
                out.append(f"eq=saturation={float(e['saturation']):.3f}")
            return out
    return []


def finish_filters(cf):
    """(the camera's frame-rate step, its finishing filters after the scale) for cam_finish's settings."""
    rate = (f"framerate=fps={FPS}:interp_start=0:interp_end=255:scene=100" if cf.get("smooth") else f"fps={FPS}")
    after = []
    if cf.get("saturation"):                                 # (a camera that reads paler than the hero, after its
        after.append(f"eq=saturation={float(cf['saturation']):.3f}")   # look: brought to the hero's richness)
    if cf.get("sharpen"):
        after.append(f"unsharp=lx=5:ly=5:la={float(cf['sharpen']):.3f}:cx=5:cy=5:ca=0")
    if cf.get("rolloff"):
        k = 16 + 219 * float(cf["rolloff"])                    # 8-bit legal range: above k, a gentle shoulder
        c = float(cf.get("rolloff_amount", 0.25))
        after.append(f"lutyuv=y=if(gt(val\\,{k:.1f})\\,val-{c:.3f}*pow(val-{k:.1f}\\,2)/{235 - k:.1f}\\,val)")
    return rate, after


PROJECT = [None]          # the project being rendered (its LUTs are part of every shot's cache key)


def _dust_version():
    from . import dust
    return dust.version(PROJECT[0])


def _dusty(s):
    from . import dust
    return dust.applies(PROJECT[0], s["unit"])


_file_ids = {}


def file_id(path) -> str | None:
    """A file's content as a short hash (None when it isn't there), read once per size and time: a cache key made
    of a LUT's or a dust map's content changes when the look does, and not when the file was only written again."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    tag = (path, st.st_size, st.st_mtime_ns)
    if tag not in _file_ids:
        with open(path, "rb") as f:
            _file_ids[tag] = hashlib.md5(f.read()).hexdigest()[:12]
    return _file_ids[tag]


def _lut_id(project, u, at=None):
    """The unit's LUT for a cache key: the filter and the cube's content (a cube rebuilt under the same name)."""
    cube = lut_cube(project, u, at)
    if not cube:
        return None
    return [lut_filter(project, u, at=at), file_id(cube if os.path.isabs(cube) else project.path(cube))]


def crop_key(c):
    """The part of a plan's crop that draws the picture: its size and its path. (The rest are the planner's scores
    of the framing -- action, face, fill, churn...: a re-plan that only re-scores must not re-render.)"""
    return [c["w"], c["h"], c["path"]] if c else None


def dunk_ramp(s, project=None):
    """A dunk's shot as it is rendered: with fx.speed's "dunk" ramp where the plan gave it the planner's old default
    ("hero", a dance move's curve: 0.2 s into 0.3x, half a second there) and reel.json names no "reference_ramp" of
    its own -- a named one, "hero" included, plays as named."""
    project = project or PROJECT[0]
    if s.get("dunk") and s.get("fx") == ["hero"] and project is not None and "reference_ramp" not in project.config:
        return dict(s, fx=["dunk"], effect="dunk")
    return s


def key(s, beat=None, size=None):
    """Cache name for a shot: changes whenever anything that shapes its pixels does, and only then (size: the frame
    it is rendered at, with the encode's settings; without one, the shot's identity alone -- the effects' random
    seed). A plain shot is its source stretch, crop and whatever is laid on it; an effect shot also the things its
    effects read: the move's moment and peak, the beat it plays at (its own "beat" where the plan gives one: what
    fx_shot renders with), a face to keep sharp, its role and what caps its speed. The shot's place in the plan is
    no part of it: a shot added or dropped earlier in the reel leaves the others' files as they are."""
    s = dunk_ramp(s)
    u = s["unit"]
    fx = s.get("fx") or None
    k = [u["id"], u["source"], u.get("rotate"), s["src_in"], s["frames"], fx, crop_key(s.get("crop")),
         s.get("logo"), s.get("stab"), 4, hw.enc_tag()]
    if fx:
        k.append(["fx", s.get("t"), s["peak"], s.get("beat", beat), u.get("t0"), u.get("t1"), u.get("role"),
                  s.get("kind") == "closeup",
                  (s.get("crop") or {}).get("face", 0) >= 0.5, bool(s.get("group")), bool(s.get("add"))])
    k = json.dumps(k
                   + (["half"] if s.get("stab") and STAB_ANALYSIS == "half" else [])
                   + ([s["span"]] if s.get("span") else []) + ([s["logo_image"]] if s.get("logo_image") else [])
                   + ([["pull", s["pull"]]] if s.get("pull") else [])
                   + ([["lift", s["lift"]]] if s.get("lift") else [])
                   + (["speed3"] if any(n in SPEED_FX for n in fx or ()) else [])   # (1x stretches on whole frames)
                   + ([["ramps", PROJECT[0].config.get("ramps")]] if PROJECT[0] is not None
                      and PROJECT[0].config.get("ramps") and any(n in RAMP_FX for n in fx or ()) else [])
                   + ([["pace", s["speed"]]] if (s.get("speed") or 1.0) != 1.0 else [])
                   + ([["reverse", True]] if s.get("reverse") else [])
                   + ([["stab_strong", True, PROJECT[0].config.get("stab_strong_settings")]]
                      if s.get("stab") and PROJECT[0] is not None and any(
                       __import__("fnmatch").fnmatch(u.get("clip", ""), q)
                       for q in (PROJECT[0].config.get("stab_strong") or [])) else [])
                   + ([["size", *size, hw.ENC_VERSION]] if size else [])
                   + ([[k, s[k]] for k in ("title", "split", "stutter", "card", "glide", "gain", "gain_protect")
                       if s.get(k)])
                   + ([["split_v", SPLIT_VERSION, split_style(PROJECT[0])]] if s.get("split") else [])
                   + ([["photo", s["photo"], _spatial_version()]] if s.get("photo") else [])
                   + ([["photos", s["photos"], _spatial_version()]] if s.get("photos") else [])
                   + ([["fit", s.get("span"), bool(s.get("fit_even")), _speed_version()]] if s.get("fit") or
                      "fit" in (s.get("fx") or []) else [])
                   + ([["shot_finish", shot_finish(PROJECT[0], s)]] if shot_finish(PROJECT[0], s) else [])
                   + ([["endcard", (PROJECT[0].config.get("endcard") if PROJECT[0] is not None else None)]]
                      if s.get("card") else [])
                   + ([["title_cfg", (PROJECT[0].config.get("title") if PROJECT[0] is not None else None)]]
                      if s.get("title") else [])
                   + ([["dust", _dust_version()]] if PROJECT[0] is not None and PROJECT[0].config.get("dust")
                      and s.get("unit", {}).get("source") and _dusty(s) else [])
                   + ([["finish", cam_finish(PROJECT[0], u)]] if cam_finish(PROJECT[0], u) else [])
                   + ([["panel_finish", [cam_finish(PROJECT[0], p["unit"]) for p in s["split"]["panels"]]]]
                      if s.get("split") and PROJECT[0] is not None else [])
                   + ([["lut", _lut_id(PROJECT[0], u, mid_of(s))]]
                      if PROJECT[0] is not None and lut_cube(PROJECT[0], u, mid_of(s)) else [])
                   + ([["panel_luts", [_lut_id(PROJECT[0], p["unit"], mid_of(p, s.get("length")))
                                       for p in s["split"]["panels"]]]]
                      if PROJECT[0] is not None and s.get("split") else []),
                   sort_keys=True)
    return hashlib.md5(k.encode()).hexdigest()[:10]


def stutter_graph(n, length, read):
    """A stutter's jump-cuts as a filter: n equal pieces of the shot's length, each from a point further on through
    the source read (a run of quick cuts on one moment, as an editor does on a clap or a cheer)."""
    piece = length / n
    step = max((read - piece) / max(n - 1, 1), piece)
    parts = "".join(f"[s{k}]" for k in range(n))
    g = f"split={n}{parts};" + ";".join(
        f"[s{k}]trim=start={k * step:.3f}:duration={piece:.3f},setpts=PTS-STARTPTS[p{k}]" for k in range(n))
    return g + ";" + "".join(f"[p{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0"


def panel_crop(rect, W, H, src_aspect=16 / 9, cx=0.5, cy=0.5, zoom=1.0):
    """The part of the source a split-screen panel shows: the panel's own shape, as big as the source allows (over
    zoom: closer in), centred on (cx, cy) (the subject) and kept inside the frame. (w, h, x, y) as fractions of the
    source."""
    pa = (rect[2] * W) / (rect[3] * H)
    w, h = (pa / src_aspect, 1.0) if pa < src_aspect else (1.0, src_aspect / pa)
    w, h = w / max(zoom, 1.0), h / max(zoom, 1.0)
    x = min(max(cx - w / 2, 0.0), 1 - w)
    y = min(max(cy - h / 2, 0.0), 1 - h)
    return w, h, x, y


PROXIES = [{}]           # the project's proxies, kept when the render reads the camera files (proxies_render)


def proxy_of(project, source, t):
    """(file, time) of a source's proxy (reel.json "proxies", even when the render itself reads the camera files),
    or None."""
    px = (PROXIES[0] or {}).get(source)
    if not px:
        return None
    f, shifts = px[0], px[1] if len(px) > 1 else []
    return os.path.join(project.folder, f), t + sum(d for at, d in shifts if t >= at)


BT709 = "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
# (a generated layer -- a split screen's canvas -- labelled as the footage is: unlabelled, the encoder's BT.709 tags
# make ffmpeg convert the whole picture's colour matrix, shifting every panel's colour)
SPLIT_VERSION = 3        # bumps split shots' cache keys when their drawing changes
SPLIT = dict(gutter=0.0062, colour="0xF2EFEA", enter=0.45, stagger=0.12)   # reel.json "split_style" overrides


def split_style(project):
    """The split screens' look (reel.json "split_style"): gutter (of the frame width, between and around panels:
    24 px at 4K), colour of the border, enter (s a panel takes to arrive), stagger (s between panels)."""
    return {**SPLIT, **((project.config.get("split_style") or {}) if project is not None else {})}


def enter_expr(style, k, n, x, y, pw, ph, W, H, st):
    """Overlay x, y expressions bringing panel k of n into its place (x, y) as the shot opens: "slide" -- each from
    its own side (left from the left, right from the right, middle up from below), "stagger" -- each up from below,
    one after the other, "push" -- the row in from the right together, then settled. An ease-out (cubic) over
    st["enter"] s, panel by panel st["stagger"] s apart; then held."""
    d, lag = st["enter"], st["stagger"] * k
    p = f"(1-pow(1-clip((t-{lag:.3f})/{d:.3f},0,1),3))"     # 0 -> 1, eased
    if style == "push":
        return f"{x}+({W}-{x}+{k * 40})*(1-{p})", f"{y}"
    if style == "cascade":                                  # a grid filling in cell by cell: a quick rise each
        d2, lag2 = d * 0.7, st["stagger"] * 0.5 * k
        p2 = f"(1-pow(1-clip((t-{lag2:.3f})/{d2:.3f},0,1),3))"
        return f"{x}", f"{y}+{int(ph * 0.35)}*(1-{p2})-{H * 2}*lt(t,{lag2:.3f})"
    if style == "stagger" or (style == "slide" and 0 < k < n - 1):
        return f"{x}", f"{y}+({H}-{y})*(1-{p})"
    if style == "slide":
        x0 = -pw if k == 0 else W
        return f"{x0}+({x}-({x0}))*{p}", f"{y}"
    return f"{x}", f"{y}"


def split_shot(project, s, fmt, out, qsv=False):
    """split_shot_on, decoded on Quick Sync when it can be -- and on the CPU when the iGPU runs out of sessions or
    memory (several panels at 4K, beside other shots rendering)."""
    try:
        return split_shot_on(project, s, fmt, out, qsv)
    except subprocess.CalledProcessError:
        if not qsv:
            raise
        return split_shot_on(project, s, fmt, out, False, cpu_exe=hw.win_ffmpeg())


def split_shot_on(project, s, fmt, out, qsv=False, cpu_exe=None):
    """A split screen: each panel its own footage (cropped to the panel's shape around its subject, the camera's look
    and the lens's dust fix on it), in panels set into a border (split_style) and arriving with the shot's entrance
    style (s["split"]["style"]: slide, stagger or push; none: in place)."""
    from . import dust
    W, H = size(project, fmt)
    st = dict(split_style(project))
    st["enter"] = min(st["enter"], 0.3 * s["length"])       # a short slot's panels settle well before its cut
    st["stagger"] = min(st["stagger"], 0.15 * s["length"] / max(len(s["split"]["panels"]), 1))
    g = int(round(W * st["gutter"] / 2)) * 2
    L, panels = s["length"], s["split"]["panels"]
    rects = [list(r) for r in s["split"]["rects"]]          # (a copy: filled with pixels below -- a retry on the CPU
                                                            # must read the layout, not the first try's pixels)
    style = s["split"].get("style")
    inputs, graph, exe0, extra = [], [], None, []
    many = len(panels) > 4                                   # a grid: each cell from its 1080p proxy (cells are
    for k, (p, r) in enumerate(zip(panels, rects)):          # small; 14 4K decodes at once are not)
        src, t = (proxy_of(project, p["unit"]["source"], p["src_in"]) if many else None) or \
            project.decode_from(p["unit"]["source"], p["src_in"])
        exe, dec, P, pre = hw.ffmpeg(src, qsv and not many)
        if cpu_exe and not qsv and hw.on_windows_drive(src):    # (the fallback: the Windows build, CPU decode --
            exe, dec, P, pre = cpu_exe, [], hw.winpath, ""      # the Linux build's lut2 at 4K 4:2:2 can crash)
        if exe0 is None:
            exe0, P0 = exe, P
        elif exe != exe0:                                    # one ffmpeg reads every panel: the CPU when mixed
            exe, dec, P, pre = "ffmpeg", [], (lambda x: x), ""
        inputs += [*dec, "-ss", f"{t:.3f}", "-t", f"{L + 0.1:.3f}", "-i", P(src)]
        c = p.get("crop") or {}
        path = c.get("path") or [[0, 0.5, 0.5]]
        mid = [0, *p["focus"]] if p.get("focus") else path[len(path) // 2]   # on its person (template's focus)
        x0, y0 = int(round(r[0] * W)) + g, int(round(r[1] * H)) + g          # the panel inside its border
        pw = int(round((r[2] * W - 2 * g) / 2) * 2) if r[0] + r[2] < 0.999 else int(round((W - x0 - g) / 2) * 2)
        ph = int(round((r[3] * H - 2 * g) / 2) * 2) if r[1] + r[3] < 0.999 else int(round((H - y0 - g) / 2) * 2)
        if r[0] > 0:
            x0 -= g // 2
            pw += g // 2
        w, h, x, y = panel_crop([r[0], r[1], pw / W, ph / H], W, H, cx=mid[1], cy=mid[2], zoom=float(p.get("zoom", 1.0)))
        lut = lut_filter(project, p["unit"], P, at=mid_of(p, L))
        prate, pa = finish_filters(cam_finish(project, p["unit"]))
        pafter = "".join("," + x for x in pa)
        head = f"[{k}:v]{pre}"
        img = dust.for_shot(project, dict(p, unit=p["unit"], src_in=p["src_in"]), L) if project is not None else None
        if img and not many:                                 # the lens's dust out first (dust.FILTER), per panel
            extra.append(img)                                # (a grid's small proxy cells don't need it)
            j = len(rects) + len(extra) - 1
            fmt = dust.pix_fmt(dust.source_layout(project, p["unit"])[2])
            graph.append(dust.FILTER.replace("[1:v]", f"[{j}:v]").replace("[0:v]", f"[{k}:v]{pre}")
                         .replace("[dm]", f"[dm{k}]").replace("[ds]", f"[ds{k}]").format(fmt=fmt) + f"[dz{k}]")
            head = f"[dz{k}]"
        graph.append(f"{head}{(lut + ',') if lut else ''}crop=iw*{w:.5f}:ih*{h:.5f}:iw*{x:.5f}:ih*{y:.5f},"
                     f"scale={pw}:{ph}:flags=lanczos,{prate},setsar=1,format=yuv420p{pafter}[p{k}]")
        rects[k] = [x0, y0, pw, ph]
    exe = exe0 or "ffmpeg"
    graph.append(f"color=c={st['colour']}:s={W}x{H}:r={FPS}:d={L + 0.1:.3f},format=yuv420p,{BT709}[b0]")
    for k, (x0, y0, pw, ph) in enumerate(rects):
        last = k == len(rects) - 1
        ex, ey = enter_expr(style, k, len(rects), x0, y0, pw, ph, W, H, st)
        graph.append(f"[b{k}][p{k}]overlay=x='{ex}':y='{ey}':eval=frame:shortest=1"
                     + (",format=yuv420p" if last else f"[b{k + 1}]"))
    logo = s["split"].get("logo")
    if logo and project is not None:                         # a spare grid cell: the logo on the border's colour
        lp = logo["image"] if os.path.isabs(logo["image"]) else os.path.join(project.folder, logo["image"])
        lr = logo["rect"]
        lw, lh = int(lr[2] * W * 0.62), int(lr[3] * H * 0.62)
        lx, ly = int((lr[0] + lr[2] / 2) * W - lw / 2), int((lr[1] + lr[3] / 2) * H - lh / 2)
        j = len(rects) + len(extra)
        extra.append(lp)
        graph[-1] = graph[-1].replace(",format=yuv420p", "[bl]") if graph[-1].endswith(",format=yuv420p") else graph[-1]
        graph.append(f"[{j}:v]scale={lw}:{lh}:force_original_aspect_ratio=decrease,format=rgba,loop=-1:1,fps={FPS},"
                     f"fade=t=in:st={st['stagger'] * 0.5 * len(rects):.2f}:d=0.4:alpha=1[lg];"
                     f"[bl][lg]overlay=x={lx}+({lw}-w)/2:y={ly}+({lh}-h)/2:shortest=1,format=yuv420p")
    extra_in = [a for img in extra for a in (["-loop", "1", "-i", P0(img)] if img.lower().endswith(".png")
                                             else [*dust.INPUT, "-i", P0(img)])]
    hw.run_enc(from_file([exe, "-v", "error", "-y", *inputs, *extra_in, "-an", "-filter_complex", ";".join(graph),
                          *hw.enc(), "-frames:v", str(s["frames"]), P0(out)], P0, out))


def blend_shot(project, s, fmt, out, qsv=False):
    """A double exposure (template.blend_shot): the base shot slowed (its 25p frames run together over four frames
    -- a soft, dreamy background, not a stutter) and over it the close-up of the person, dissolving in and out at
    the blend's opacity. Each layer is rendered as a plain shot first (its look, dust, steadying), then laid up."""
    import tempfile
    bl = s["blend"]
    L = s["length"]
    base = {k: v for k, v in s.items() if k != "blend"}
    over = dict(unit=bl["unit"], src_in=bl["src_in"], length=L, frames=s["frames"], crop=bl["crop"], fx=[],
                stab=True, t=bl["t"], peak=0.0)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out) or None) as d:
        a, b = os.path.join(d, "base.mp4"), os.path.join(d, "over.mp4")
        cut_shot(project, base, fmt, a, qsv)
        cut_shot(project, over, fmt, b, qsv)
        op, fin, fout = float(bl.get("opacity", 0.55)), min(1.2, L / 3), min(1.0, L / 3)
        graph = (f"[0:v]tmix=frames=4[a];[1:v]format=yuva420p,colorchannelmixer=aa={op:.3f},"
                 f"fade=t=in:st=0:d={fin:.2f}:alpha=1,fade=t=out:st={max(L - fout - 0.3, 0):.2f}:d={fout:.2f}:alpha=1[b];"
                 f"[a][b]overlay=0:0,format=yuv420p")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-i", b, "-filter_complex", graph,
                        *hw.enc(), "-frames:v", str(s["frames"]), out], check=True)


def _speed_version():
    """fx/speed.py's own code in a fitted shot's key (its curve changes: the shot renders again)."""
    import hashlib

    from ..fx import speed
    return hashlib.md5(open(speed.__file__, "rb").read()).hexdigest()[:8]


def _spatial_version():
    """The spatial effect's own code (and its layers' stamp) in a photo shot's key: change either, it renders again."""
    import hashlib

    from ..fx import spatial
    return hashlib.md5(open(spatial.__file__, "rb").read()).hexdigest()[:8]


def photo_shot(project, s, fmt, out):
    """A still photo as a spatial scene (fx/spatial.py): its layers made beforehand (`focalcut fx spatial`, or
    spatial.layers in the ML venv) into s["photo"]["layers"]; the camera's push and drift rendered here, the last
    s["photo"]["fade"] seconds faded to black when asked."""
    from ..fx import spatial
    W, H = size(project, fmt)
    ph = s["photo"]
    folder = ph["layers"] if os.path.isabs(ph["layers"]) else os.path.join(project.folder, ph["layers"])
    kw = {k: float(ph[k]) for k in ("push", "sway", "pop") if k in ph}
    spatial.render(folder, out, W, H, s["frames"] / FPS, FPS, encoder=hw.enc(),
                   direction=float(ph.get("direction", 1.0)), fade=float(ph.get("fade", 0.0)), **kw)


CAPTION_FADE = 0.6       # seconds each caption fades in and out over


def add_captions(project, path, caps, size):
    """Lay reel.json "captions" ([{"at": s, "to": s, "text": words, "ref": source line}], film time) over a joined
    film -- across its cuts, each fading in and out -- re-encoding the picture once, the sound copied."""
    from . import cards
    tmp = path + ".cap.mp4"
    inputs, graph, last = [], [], "[0:v]"
    for k, c in enumerate(caps):
        img = cards.caption(project, c, size)
        d = float(c["to"]) - float(c["at"])
        inputs += ["-loop", "1", "-t", f"{d:.3f}", "-i", img]
        f = min(CAPTION_FADE, d / 3)
        graph.append(f"[{k + 1}:v]format=rgba,fade=t=in:st=0:d={f:.2f}:alpha=1,"
                     f"fade=t=out:st={d - f:.3f}:d={f:.2f}:alpha=1,setpts=PTS+{float(c['at']):.3f}/TB[q{k}]")
        graph.append(f"{last}[q{k}]overlay=0:0:eof_action=pass[v{k}]")
        last = f"[v{k}]"
    graph[-1] = graph[-1].rsplit("[", 1)[0] + ",format=yuv420p[vout]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, *inputs, "-filter_complex", ";".join(graph),
                    "-map", "[vout]", "-map", "0:a?", *hw.enc(final=True), "-c:a", "copy", "-movflags", "+faststart",
                    tmp], check=True)
    os.replace(tmp, path)


def photo_grid_shot(project, s, fmt, out):
    """The editor's photos side by side (template's "photos" slot): each a spatial scene (fx/spatial: its layers made
    beforehand) at its cell's size, in the split screens' grid and border, arriving one after another (a quick
    fade in each, staggered) -- the beginning of the baptisms' grid of speakers, in stills."""
    import tempfile

    from ..fx import spatial
    W, H = size(project, fmt)
    st = split_style(project)
    layers = [x if os.path.isabs(x) else os.path.join(project.folder, x) for x in s["photos"]["layers"]]
    n = len(layers)
    cells, _logo = _grid_cells(n)
    g = max(2, int(round(st["gutter"] * W)) // 2 * 2)
    L = s["frames"] / FPS
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out) or None) as d:
        parts, graph = [], [f"color=c={st['colour']}:s={W}x{H}:r={FPS}:d={L + 0.1:.3f},format=yuv420p[c0]"]
        for k, (lay, r) in enumerate(zip(layers, cells, strict=True)):
            x0, y0 = int(round(r[0] * W)) + g, int(round(r[1] * H)) + g
            pw = int(round((r[2] * W - 2 * g) / 2) * 2) if r[0] + r[2] < 0.999 else int(round((W - x0 - g) / 2) * 2)
            ph = int(round((r[3] * H - 2 * g) / 2) * 2) if r[1] + r[3] < 0.999 else int(round((H - y0 - g) / 2) * 2)
            if r[0] > 0:
                x0 -= g // 2
                pw += g // 2
            f = os.path.join(d, f"p{k}.mp4")
            spatial.render(lay, f, pw, ph, L, FPS, encoder=hw.enc(), direction=1.0 if k % 2 == 0 else -1.0)
            parts += ["-i", f]
            t_in = 0.15 + k * float(st.get("stagger", 0.12)) * 1.5
            graph.append(f"[{k}:v]format=yuva420p,fade=t=in:st={t_in:.2f}:d=0.35:alpha=1[p{k}]")
            graph.append(f"[c{k}][p{k}]overlay={x0}:{y0}:eof_action=pass[c{k + 1}]")
        graph[-1] = graph[-1].rsplit("[", 1)[0] + ",format=yuv420p,setsar=1"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *parts, "-filter_complex", ";".join(graph),
                        *hw.enc(), "-frames:v", str(s["frames"]), out], check=True)


def _grid_cells(n):
    from .template import grid
    return grid(n, None)


def card_shot(project, s, fmt, out, qsv=False):
    """The end card: moving footage softened into a ground (a heavy blur, lifted towards the reference card's light
    sky) with the logo over it, fading in; the picture fades to black over its last second."""
    from . import cards
    W, H = size(project, fmt)
    L = s["length"]
    logo = cards.endcard(project, fmt, s["card"], (W, H))
    glow = float(((project.config.get("endcard") or {}) if project is not None else {}).get("glow", 0.0))
    if glow > 0 and logo:                                   # reel.json "endcard" "glow": a soft light behind the
        logo = cards.glow_under(logo, glow)                 # logo only, so it reads on a dark ground
    src, t = project.decode_from(s["unit"]["source"], s["src_in"])
    exe, dec, P, pre = hw.ffmpeg(src, qsv)
    lut = lut_filter(project, s["unit"], P, at=mid_of(s))
    a = (s["card"].get("logo_in") or 0.6)
    ec = (project.config.get("endcard") or {}) if project is not None else {}
    blur, sat = float(ec.get("blur", 1.0)), float(ec.get("saturation", 0.75))   # reel.json "endcard": how soft the
    bright = float(ec.get("brightness", 0.06))                                  # ground (1: the default blur) and
    sp = float(s.get("speed") or 1.0)                      # (a range slowed to fill the card)
    rate, after = finish_filters(cam_finish(project, s["unit"]))
    if sp < 0.999:
        rate = f"setpts=PTS/{sp:.4f}," + rate   # (the camera's own frame-rate blend and finish:
    soft = (f"scale={W // 4}:{H // 4},gblur=sigma={blur * W / 160:.1f},"   # a 25p drone ground steps otherwise) --
            if blur > 0 else "")                                         # softened only when asked (blur 0: sharp)
    graph = (f"[0:v]{pre}{(lut + ',') if lut else ''}{soft}"             # how much colour it keeps
             + f"eq=brightness={bright:.3f}:saturation={sat:.3f},scale={W}:{H}:flags=bicubic,{rate},"
             + "".join(a_ + "," for a_ in (after if blur <= 0 else []))
             + "setsar=1[g];"
             f"[1:v]format=rgba,loop=-1:1,fps={FPS},fade=t=in:st=0.3:d={a:.2f}:alpha=1[l];"
             f"[g][l]overlay=0:0:shortest=1,fade=t=out:st={max(L - 1.0, 0):.3f}:d=1.0,format=yuv420p")
    hw.run_enc(from_file([exe, "-v", "error", "-y", *dec, "-ss", f"{t:.3f}", "-t", f"{L * sp + 0.2:.3f}", "-i", P(src),
                          "-i", P(logo), "-an", "-filter_complex", graph, *hw.enc(), "-frames:v", str(s["frames"]),
                          P(out)], P, out))


def add_dust(project, s, vf, pre, P, src, length):
    """The lens's dust divided out of the shot (dust.py) as its first step, at the camera's own size: vf (the
    ffmpeg filter arguments built for the shot) with the shot's dust image as one more input. Unchanged without a
    dust map for this footage."""
    from . import dust
    img = dust.for_shot(project, s, length) if project is not None else None
    if not img:
        return vf
    if vf[0] == "-vf":
        vf = ["-filter_complex", "[0:v]" + vf[1]]
    k = vf.index("-filter_complex")
    n = sum(1 for a in vf[:k] if a == "-i") + 1                # its input index
    fmt = dust.pix_fmt(dust.source_layout(project, s["unit"])[2])
    fc = vf[k + 1]
    head = (dust.FILTER.replace("[1:v]", f"[{n}:v]").replace("[0:v]", f"[0:v]{pre}").format(fmt=fmt) + "[dz];")
    fc = head + fc.replace("[0:v]" + pre, "[dz]", 1)
    return [*vf[:k], *dust.INPUT, "-i", P(img), "-filter_complex", fc]


def cut_shot(project, s, fmt, out, qsv=False):
    """A plain shot straight from the camera file: decoded on the Intel iGPU when qsv (and hw.ffmpeg can; a
    stabilised shot reads its motion file on the Linux side, so it stays on the CPU), encoded by hw.enc."""
    if s.get("card"):
        return card_shot(project, s, fmt, out, qsv)
    if s.get("photo"):
        return photo_shot(project, s, fmt, out)
    if s.get("photos"):
        return photo_grid_shot(project, s, fmt, out)
    if s.get("split"):
        return split_shot(project, s, fmt, out, qsv)
    if s.get("blend"):
        return blend_shot(project, s, fmt, out, qsv)
    p = s.get("speed") or 1.0
    read = read_length(s)                                   # the source it plays (more of it when paced)
    if s.get("stutter"):
        read = s["length"] * 2.0 + 0.1
    if p != 1.0 and s.get("crop"):                          # follow the subject over all of it
        from .plan import reframe
        s = dict(s, crop=reframe(project, s["unit"], s["src_in"], s["length"] * p, fmt) or s["crop"])
    src, t = project.decode_from(s["unit"]["source"], s["src_in"])
    exe, dec, P, pre = hw.ffmpeg(src, qsv)
    win = lift_window(project, s, fmt, read)
    chain = pre + shot_filter(s, fmt, project, read, win, qsv=bool(dec), P=P)
    if s.get("stutter"):                                     # jump-cuts: n pieces, each skipping on through the read
        chain = pre + stutter_graph(s["stutter"], s["length"], read - 0.1) + "," + chain[len(pre):]
    mask = ["-i", P(win[0])] if win else []                  # input 1: the night lift's window
    vf = [*mask, "-filter_complex", "[0:v]" + chain] if win else ["-vf", chain]
    if s.get("title"):                                       # a title fading in over the shot (cards.title)
        from . import cards
        W, H = size(project, fmt)
        img = cards.title(project, fmt, s["title"], (W, H))
        if img:
            a, b = (s["title"].get("fade") or [0.4, 1.6])[:2]
            a = max(0.2, a - 0.8)                            # (the strokes register late: the fade starts sooner)
            k = len(mask) // 2 + 1
            vf = [*mask, "-i", P(img), "-filter_complex",
                  f"[0:v]{chain}[v];[{k}:v]format=rgba,loop=-1:1,fps={FPS},fade=t=in:st={a:.2f}:d={max(b - a, 0.3):.2f}"
                  f":alpha=1[t];[v][t]overlay=0:0:shortest=1,format=yuv420p"]
    logo = project.logo(s.get("logo_image"))
    if s.get("logo") and logo:                              # the event's logo over the opening, or an act's title
        W, H = size(project, fmt)
        fade = ",fade=t=in:st=0.15:d=0.6:alpha=1" if s["logo"] == "in" else ""
        vf = [*mask, "-i", P(logo), "-filter_complex",
              f"[0:v]{chain},vignette=angle=PI/5,eq=brightness=-0.05[v];"
              f"[{2 if win else 1}:v]scale={int(W * (0.5 if W > H else 0.8))}:-1:flags=lanczos,format=rgba,loop=-1:1,"
              f"fps={FPS}{fade}[l];[v][l]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p"]
    vf = add_dust(project, s, vf, pre, P, src, read)
    hw.run_enc(from_file([exe, "-v", "error", "-y", *dec, "-ss", f"{t:.3f}", "-t", f"{read:.3f}",
                          "-i", P(src), "-an", *vf, *hw.enc(), "-frames:v", str(s["frames"]), P(out)], P, out))


class Ahead:
    """An iterator run on a thread of its own, up to n of its items waiting to be taken: a decoder, a speed curve
    and an encoder each work while the others do, instead of taking turns around one loop. stop() lets the thread
    go when the reader has taken all it wants (or failed); what the iterator raised is raised to the reader."""

    def __init__(self, items, n=3):
        import queue
        self.q, self.end, self.halt, self.err = queue.Queue(n), object(), threading.Event(), None
        self.thread = threading.Thread(target=self._run, args=(items,), daemon=True)
        self.thread.start()

    def _put(self, x):
        import queue
        while not self.halt.is_set():
            try:
                return self.q.put(x, timeout=0.1)
            except queue.Full:
                pass

    def _run(self, items):
        try:
            for x in items:
                if self.halt.is_set():
                    return
                self._put(x)
        except BaseException as e:                 # (raised in the reader's thread, below)
            self.err = e
        self._put(self.end)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is self.end:
                if self.err is not None:
                    raise self.err
                return
            yield x

    def stop(self):
        self.halt.set()


FX_JOBS = 3              # speed-curve shots rendered at once (FOCALCUT_FX_JOBS): each a decode, the curve and an encode


def speed_only(s) -> bool:
    """Is the shot's effect chain one speed curve and nothing else? Those stream (fx_shot): no GPU model, a few
    frames held, so several render at once."""
    return len(s.get("fx") or ()) == 1 and s["fx"][0] in SPEED_FX


@functools.cache
def source_fps(path) -> float:
    """A file's frame rate (a speed ramp's slowest speed depends on it: fx.speed.slow_floor); FPS when unknown."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
        n, _, d = r.strip().strip(",").partition("/")
        return float(n) / float(d or 1)
    except (OSError, ValueError, ZeroDivisionError):
        return float(FPS)


def fx_shot(project, s, fmt, beat, out):
    """Render a shot through its effect chain (speed curves first change how much source is read). A speed curve
    on its own streams: the source is decoded to the 4:2:0 the encoder takes, the curve reads it as it goes
    (fx.speed.Speed.stream) and its output goes straight to the encoder -- two or three frames in memory, not the
    whole shot as RGB (8.6 GB for a 7 s 4K shot), and the frames it plays at 1x are the decoded frames themselves.
    Any other chain works on the whole shot as RGB frames, as its effects expect."""
    from .. import fx
    from ..fx.speed import Speed
    from .plan import reframe
    W, H = size(project, fmt)
    s = dunk_ramp(s, project)
    u, L = s["unit"], s["length"]
    ctx = fx.Context(fps=FPS, beat=beat, peak=s["peak"], seed=int(key(s)[:4], 16), params={"group": s.get("group"), "closeup": s["kind"] == "closeup",
                                     "face": s["kind"] == "closeup" or (s.get("crop") or {}).get("face", 0) >= 0.5,
                                     "cap": 1.0 if s.get("add") or s.get("span") else 3.5 if u["role"] == "transition" else 1.5,   # dancing never past 1.5x
                                     "pace": s.get("speed") or 1.0,
                                     "ramps": project.config.get("ramps"),     # fx.speed.Ramp: its shapes, and the
                                     "src_fps": source_fps(project.decode_from(u["source"], s["t"])[0]),  # slowest
                                     **({"glide": s["glide"]} if s.get("glide") else {})})
    if s.get("span"):                              # an editor's selection: all of it, from their in-point
        ctx.params["src_len"] = s["span"][1] - s["span"][0]
    if s.get("fit_even"):                          # (one even pace, no 1x tail: template's closing shot)
        ctx.params["even"] = True
    src_len, src_peak = L, s["peak"]
    for n in s["fx"]:
        e = fx.EFFECTS[n]()
        if isinstance(e, Speed):
            sp = e
            src_len, src_peak = e.source_length(L, ctx), e.source_peak(s["peak"], ctx, L)
    src_in = min(max(s["t"] - src_peak, u["t0"]), max(u["t1"] - src_len, u["t0"]))
    if "fit" in s["fx"]:
        src_in = min(max(s["span"][0], u["t0"]), max(u["t1"] - src_len, u["t0"]))
    elif src_len != L and abs(src_in - (s["t"] - src_peak)) > 1e-3:   # clamped at the unit's edge: slow down where
        ctx.peak = sp.out_time(s["t"] - src_in, L, ctx)               # the move now plays, not where it would have
        ctx.params["peak_fixed"] = True                               # (a ramp leaves it there: no bar line)
    shot = dict(s, src_in=src_in, length=src_len)
    if s.get("crop"):                              # the source read changed: follow the subject over it instead
        shot["crop"] = reframe(project, u, src_in, src_len, fmt) or s["crop"]
    src, t = project.decode_from(u["source"], src_in)
    exe, dec, P, pre = hw.ffmpeg(src, True)
    win = lift_window(project, shot, fmt, src_len)
    chain = pre + shot_filter(shot, fmt, project, src_len, win, qsv=bool(dec), P=P)
    vf = ["-i", P(win[0]), "-filter_complex", "[0:v]" + chain] if win else ["-vf", chain]
    vf = add_dust(project, shot, vf, pre, P, src, src_len)
    cmd = [exe, "-v", "error", *dec, "-ss", f"{t:.3f}", "-t", f"{src_len:.3f}", "-i", P(src), "-an", *vf,
           "-f", "rawvideo"]
    n_out = s["frames"]
    ctx.out_frames = n_out
    if speed_only(s):
        from ..fx.speed import Yuv
        rd = subprocess.Popen(from_file([*cmd, "-pix_fmt", "yuv420p", "-"], P, out), stdout=subprocess.PIPE)
        size_ = W * H * 3 // 2

        def source():
            while True:
                b = rd.stdout.read(size_)
                if len(b) < size_:
                    return
                yield Yuv(b, W, H)

        def played(frames):                        # n_out frames whatever the source gave: held on the last
            f, k = None, 0
            for f in sp.stream(frames, ctx, n_out):
                k += 1
                yield f.raw
            for _ in range(n_out - k if f is not None else 0):
                yield f.raw
        decoded = Ahead(source())                  # the decode, the curve and the encode each on their own thread
        curved = Ahead(played(decoded))
        try:
            write_frames(curved, out, "yuv420p", (W, H))
        finally:
            spare = rd.poll() is None              # still decoding: the frames past the curve's last (or a failure
            curved.stop(); decoded.stop()          # here) -- stopped, no error of its own
            if spare:
                rd.kill()
            rc = rd.wait()
            decoded.thread.join(5); curved.thread.join(5)
            rd.stdout.close()
        if rc and not spare:                       # the decoder itself failed
            if os.path.exists(out):
                os.remove(out)
            raise subprocess.CalledProcessError(rc, cmd[0])
        return
    raw = subprocess.run(from_file([*cmd, "-pix_fmt", "rgb24", "-"], P, out), capture_output=True, check=True).stdout
    frames = list(np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3))
    frames = fx.chain(frames, s["fx"], ctx)[:n_out]
    frames += [frames[-1]] * (n_out - len(frames))
    write_frames(frames, out)


def frame_count(path) -> int:
    return int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=nb_frames", "-of", "csv=p=0", path]).decode().strip().strip(","))


def frame_size(path) -> tuple:
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    return W, H


def read_frames(path, start=0, count=None):
    """A shot file's frames as RGB arrays: all of them, or count of them from frame start. Every decoded frame comes
    out once (ffmpeg's constant-rate output would repeat the first frame of a trimmed stretch to fill the time
    before it)."""
    W, H = frame_size(path)
    trim = [] if not start and count is None else \
        ["-vf", f"trim=start_frame={start}" + (f":end_frame={start + count}" if count is not None else "")
         + ",setpts=PTS-STARTPTS"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *trim, "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = list(np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3))
    if count is not None and len(frames) > count:
        raise RuntimeError(f"{path}: asked for {count} frames from {start}, decoded {len(frames)}")
    return frames


def write_frames(frames, out, pix_fmt="rgb24", size=None):
    """Frames (a list, or anything that yields arrays or raw bytes) encoded by hw.enc: RGB arrays, or with
    pix_fmt="yuv420p" and size=(W, H) raw 4:2:0 frames."""
    frames = iter(frames)
    first = next(frames)
    W, H = size or first.shape[1::-1]
    tags = hw.RAW_YUV if pix_fmt.startswith("yuv") else []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{W}x{H}",
                          "-r", str(FPS), *tags, "-i", "-", *hw.enc(), out], stdin=subprocess.PIPE)
    try:
        for f in itertools.chain([first], frames):
            p.stdin.write(f if isinstance(f, (bytes, bytearray, memoryview)) else np.ascontiguousarray(f).data)
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait():
        if os.path.exists(out):
            os.remove(out)
        raise subprocess.CalledProcessError(p.returncode, "ffmpeg (write_frames)")


def rgb_to_yuv420(f) -> bytes:
    """An RGB frame as raw yuv420p, BT.709 limited range (what the shots are): luma per pixel, chroma averaged over
    each 2 x 2."""
    import cv2
    x = f.astype(np.float32)
    H, W = x.shape[:2]
    y = cv2.transform(x, np.float32([[0.2126, 0.7152, 0.0722]]))
    cb = ((x[..., 2] - y) * (224 / 255 / 1.8556)).reshape(H // 2, 2, W // 2, 2).mean((1, 3)) + 128
    cr = ((x[..., 0] - y) * (224 / 255 / 1.5748)).reshape(H // 2, 2, W // 2, 2).mean((1, 3)) + 128
    planes = (y * (219 / 255) + 16, cb, cr)
    return b"".join(np.clip(p + 0.5, 16, 240 if k else 235).astype(np.uint8).tobytes() for k, p in enumerate(planes))


def splice(src, out, at, new):
    """src re-encoded to out with its frames from `at` on replaced by `new` (RGB arrays). Streamed in the shot's own
    4:2:0, so a long shot at 4K is never held in memory whole and the frames outside the blend are re-encoded from
    exactly what was decoded: no trip through RGB."""
    W, H = frame_size(src)
    size_ = W * H * 3 // 2
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-fps_mode", "passthrough", "-f", "rawvideo",
                           "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE)

    def frames():
        k = 0
        while True:
            b = rd.stdout.read(size_)
            if len(b) < size_:
                break
            yield rgb_to_yuv420(new[k - at]) if at <= k < at + len(new) else b
            k += 1
    try:
        write_frames(frames(), out, "yuv420p", (W, H))
    finally:
        rd.stdout.close()
        rd.wait()


def render_shots(project, plan, fmt, work, draft=False, timer=None):
    """Every shot's file (cached by key, and named by it): plain shots several at once -- DECODERS["qsv"] decoding
    on the Intel iGPU and DECODERS["cpu"] on the CPU, each encoding with NVENC where there is one. Shots whose only
    effect is a speed curve stream (fx_shot) FX_JOBS at once, each in one of the same slots; the other effect
    shots (which run their own GPU models) go one at a time beside them all."""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    timer = timer or Timer()
    dim = size(project, fmt)
    n, done, lock = len(plan["shots"]), [0], threading.Lock()
    slots = queue.Queue()
    for kind, k in hw.decoders(DECODERS).items():
        for _ in range(k if kind == "cpu" or hw.win_ffmpeg() else 0):
            slots.put(kind)

    def tick():
        with lock:
            done[0] += 1
            print(f"\r[reel] {fmt}: shot {done[0]}/{n}", end="", flush=True)

    def plain(s, out):
        if not os.path.exists(out) and not failed:
            kind = slots.get()
            try:
                cut_shot(project, s, fmt, out + ".part.mp4", qsv=kind == "qsv")
                os.replace(out + ".part.mp4", out)
            finally:
                slots.put(kind)
        tick()

    files, jobs, fxs, curves, seen = [], [], [], [], set()
    for s in plan["shots"]:
        fx = bool(s.get("fx")) and not draft
        out = os.path.join(work, f"shot_{key(s, plan['beat'], dim)}{'_fx' if fx else ''}.mp4")   # named by what it
        files.append(out)                                   # is, not where it plays: a shot added or dropped
        if out in seen:                                     # earlier in the plan re-renders nothing else (and
            tick()                                          # one stretch played twice is one file)
            continue
        seen.add(out)
        (curves if fx and speed_only(s) else fxs if fx else jobs).append((s, out))
    failed = []

    def curve(s, out):                                      # a speed curve alone: streamed, no GPU model. It holds
        if not os.path.exists(out) and not failed:          # a render slot like a plain shot (its encoder session)
            kind = slots.get()
            try:
                fx_shot(project, s, fmt, s.get("beat", plan["beat"]), out + ".part.mp4")
                os.replace(out + ".part.mp4", out)
                with lock:
                    print(f"\n  {'+'.join(s['fx'])}: {s['unit']['id']} @ {s['t']}s", flush=True)
            finally:
                slots.put(kind)
        tick()

    def pool(fn, todo, n, stage, t0):                       # todo's shots through fn, n at once
        def job(s, out):
            try:
                fn(s, out)
            except BaseException as e:                      # (raised by the main thread, below; the shots still
                failed.append(e)                            # waiting see it and are not started)
        with ThreadPoolExecutor(max(n, 1)) as ex:
            for s, out in todo:
                ex.submit(job, s, out)
        took[stage] = max(took.get(stage, 0.0), time.time() - t0)

    took, t0 = {}, time.time()
    fx_stage = "effect shots (beside the shots)"
    # every plain shot straight to a render slot (a shaky one is steadied in its own crop: steady_crop); the speed
    # curves FX_JOBS at once beside them
    side = [threading.Thread(target=pool, args=(plain, jobs, slots.qsize(), "shots", t0), daemon=True),
            threading.Thread(target=pool, args=(curve, curves, int(os.environ.get("FOCALCUT_FX_JOBS") or FX_JOBS),
                                                fx_stage, t0), daemon=True)]
    for th in side:
        th.start()
    for s, out in fxs:                                      # the other effect shots (they run their own GPU
        if failed:                                          # models) meanwhile, one at a time
            break
        if not os.path.exists(out):
            fx_shot(project, s, fmt, s.get("beat", plan["beat"]), out + ".part.mp4")
            os.replace(out + ".part.mp4", out)
            with lock:
                print(f"\n  {'+'.join(s['fx'])}: {s['unit']['id']} @ {s['t']}s", flush=True)
        tick()
    if fxs:
        took[fx_stage] = max(took.get(fx_stage, 0.0), time.time() - t0)
    for th in side:
        th.join()
    for stage, v in took.items():
        if stage != fx_stage or fxs or curves:
            timer.add(stage, v)
    if failed:
        raise failed[0]
    print()
    return files


def apply_transitions(plan, files):
    """Rewrite the tail of the outgoing shot and the head of the incoming one at every planned transition. Only the
    frames inside the blend are held; the rest of both shots streams through. Transitions that share no shot are
    done side by side (a run of neighbouring ones in order: each reads the file the one before wrote), and a
    transition's two shots are written at once."""
    from concurrent.futures import ThreadPoolExecutor

    from .. import fx
    at = [i for i, s in enumerate(plan["shots"]) if s.get("transition") and i]
    runs = []
    for i in at:                                   # neighbours (i, i + 1) share shot i: one run
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])

    def one(i):
        s = plan["shots"][i]
        name = s["transition"]
        tag = hashlib.md5((files[i - 1] + files[i] + name).encode()).hexdigest()[:6]
        a_out, b_out = files[i - 1].replace(".mp4", f"_o{tag}.mp4"), files[i].replace(".mp4", f"_i{tag}.mp4")
        if not (os.path.exists(a_out) and os.path.exists(b_out)):
            na, nb = frame_count(files[i - 1]), frame_count(files[i])
            ctx = fx.Context(fps=FPS, beat=s.get("beat", plan["beat"]), seed=i)
            tr = fx.TRANSITIONS[name]()
            n = min(tr.frames(ctx), na, nb) // 2 * 2
            tail, head = read_frames(files[i - 1], na - n // 2, n // 2), read_frames(files[i], 0, n // 2)
            if len(tail) != n // 2 or len(head) != n // 2:
                raise RuntimeError(f"transition {name} at shot {i}: read {len(tail)} + {len(head)} frames, not {n}")
            mid = tr.apply(tail, head, ctx)
            with ThreadPoolExecutor(2) as two:
                w = [two.submit(splice, files[i - 1], a_out + ".part.mp4", na - n // 2, mid[:n // 2]),
                     two.submit(splice, files[i], b_out + ".part.mp4", 0, mid[n // 2:])]
                for f in w:
                    f.result()
            os.replace(a_out + ".part.mp4", a_out); os.replace(b_out + ".part.mp4", b_out)
        files[i - 1], files[i] = a_out, b_out

    def run(idx):
        for i in idx:
            one(i)
    with ThreadPoolExecutor(3) as ex:              # (3 runs x 2 shots: six encoder sessions at most)
        for f in [ex.submit(run, idx) for idx in runs]:
            f.result()
    return files


NAT_LUFS = -24           # a cheer under the music, levelled to this (the song sits near -14)
NAT_DUCK = 0.65          # the song's level while a cheer sounds
CLEAN_FADE = 1.5         # a clean render's music fades out over its last this many seconds (no stinger to cover it)


def mix_audio(song, plan, stinger, out, clean=False):
    """The song from its start point, fading under the stinger, plus the stinger's own sound (clean: no stinger, the
    song fades out with the picture's last CLEAN_FADE seconds)."""
    ml, sa = plan["music_length"], plan["stinger_at"]
    if clean:
        ml, fade = sa, f"afade=t=out:st={max(sa - CLEAN_FADE, 0):.3f}:d={CLEAN_FADE}"
    else:
        fade = f"afade=t=out:st={sa:.3f}:d={max(ml - sa, 0.05):.3f}"
    ms = float(plan["music_start"])                          # (a song that starts a hair after the picture -- the
    pad = f"adelay={int(round(-ms * 1000))}:all=1," if ms < 0 else ""   # first downbeat led -- is delayed, not trimmed:
    graph = (f"[0:a]atrim=start={max(ms, 0.0)}:duration={ml + min(ms, 0.0)},asetpts=PTS-STARTPTS,{pad}"   # ffmpeg
             f"{fade},aresample=48000[m]")                   # reads a negative start as 0)
    inputs = ["-i", song]
    nat = [] if clean else (plan.get("nat") or [])
    if nat:                                                  # the crowd's own cheers under the music (template.nat_sound):
        k0 = 1                                               # each loudness-levelled well under the song, the song
        duck = "+".join(f"between(t,{n['at']:.3f},{n['at'] + n['length']:.3f})" for n in nat)   # dipped while it sounds
        graph = graph.replace("[m]", f",volume=volume='if({duck},{NAT_DUCK},1)':eval=frame[m]")
        parts = []
        for k, n in enumerate(nat):
            src = os.path.join(PROJECT[0].folder, n["source"]) if PROJECT[0] is not None else n["source"]
            inputs += ["-ss", f"{n['src_t']:.3f}", "-t", f"{n['length']:.3f}", "-i", src]
            d = int(n["at"] * 1000)
            parts.append(f"[{k0 + k}:a]aformat=channel_layouts=stereo,aresample=48000,loudnorm=I={NAT_LUFS}:TP=-6,"
                         f"aresample=48000,afade=t=in:d=0.15,afade=t=out:st={max(n['length'] - 1.2, 0):.2f}:d=1.2,"
                         f"adelay={d}|{d}[n{k}]")
        graph += ";" + ";".join(parts) + ";[m]" + "".join(f"[n{k}]" for k in range(len(nat))) \
            + f"amix=inputs={len(nat) + 1}:normalize=0:duration=first[m2]"
        graph = graph.replace("[m]anull", "[m2]anull")
        tail = "[m2]"
    else:
        tail = "[m]"
    amb = None if clean or PROJECT[0] is None else PROJECT[0].config.get("ambience")
    if amb:                                                  # reel.json "ambience": {"source": "C2213.MP4", "from": s,
        k = sum(1 for x in inputs if x == "-i")              # "to": s, "lufs": -32} -- the place's own sound (water,
                                                             # a far crowd) looped quietly under the whole film
        a0, a1 = float(amb["from"]), float(amb["to"])
        inputs += ["-ss", f"{a0:.3f}", "-t", f"{a1 - a0:.3f}", "-i", os.path.join(PROJECT[0].folder, amb["source"])]
        L = sa if not clean else ml
        graph += (f";[{k}:a]aformat=channel_layouts=stereo,aresample=48000,highpass=f=90,"
                  f"loudnorm=I={float(amb.get('lufs', -32))}:TP=-9,aresample=48000,"
                  f"aloop=loop=-1:size={int((a1 - a0) * 48000)},atrim=duration={L:.3f},"
                  f"afade=t=in:d=2,afade=t=out:st={max(L - 3, 0):.3f}:d=3[amb];"
                  f"{tail}[amb]amix=inputs=2:normalize=0:duration=first[m3]")
        tail = "[m3]"
    if stinger:
        k = sum(1 for x in inputs if x == "-i")
        inputs += ["-i", stinger]
        graph += (f";[{k}:a]aresample=48000,adelay={int(sa * 1000)}|{int(sa * 1000)}[s];"
                  f"{tail}[s]amix=inputs=2:normalize=0:duration=longest[a]")
    else:
        graph += f";{tail}anull[a]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[a]",
                    "-c:a", "pcm_s24le", out], check=True)


SHAPE = dict(open_db=-3.0,                     # reel.json "music_shape": the opening a little under full,
             hush=["briefing", "speakers"],   # the film's quiet scenes (categories)
             hush_db=-7.0, hush_cut=2500.0,     # held this far down and this muffled (a low-pass, Hz),
             build_db=-5.0, build_cut=3500.0,   # the event building from here at its first moment to full and open
             swell_db=1.5)                      # at its last; each moment's rise lifted by a bar-long swell


def shape_envelope(plan, cfg, t):
    """Gain (dB) and low-pass cutoff (Hz) at film times t (numpy array), from the plan: open_db and open at first; the
    hush scenes down and muffled (a bar's ease either side); from the event's first moment a build to full and open
    by its last, each moment's rise (the crowd's sound, plan["nat"]) lifted by a swell; full from there."""
    c = dict(SHAPE, **(cfg if isinstance(cfg, dict) else {}))
    bar = 4 * plan.get("beat", 0.5)
    shots = plan["shots"]
    hush = [s for s in shots if s["unit"]["category"] in c["hush"] and not s.get("dunk")]
    dunks = [s for s in shots if s.get("dunk")]
    rises = sorted(n["at"] for n in plan.get("nat") or [] if any(d["rec_in"] <= n["at"] < d["rec_in"] + d["length"]
                                                                   for d in dunks))
    db, cut = np.full_like(t, c["open_db"]), np.full_like(t, 20000.0)
    lcut = np.log(cut)

    def ease(x):
        return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
    if dunks:
        a, b = dunks[0]["rec_in"], dunks[-1]["rec_in"] + 0.5 * dunks[-1]["length"]
        k = ease((t - a) / max(b - a, 1e-6))
        on = t >= a - bar
        pre = ease((t - (a - bar)) / bar)                 # (the bar before the first moment: from the hush to the build)
        db = np.where(on, c["build_db"] * (1 - k), db)
        lcut = np.where(on, np.log(c["build_cut"]) + (np.log(20000.0) - np.log(c["build_cut"])) * k, lcut)
        if hush:
            hdb, hcut = c["hush_db"], np.log(c["hush_cut"])
            m = (t >= a - bar) & (t < a)
            db = np.where(m, hdb + (c["build_db"] - hdb) * pre, db)
            lcut = np.where(m, hcut + (np.log(c["build_cut"]) - hcut) * pre, lcut)
    if hush:
        h0 = hush[0]["rec_in"]
        h1 = dunks[0]["rec_in"] - bar if dunks else hush[-1]["rec_in"] + hush[-1]["length"]
        into = ease((t - (h0 - bar)) / bar)
        m = (t >= h0 - bar) & (t < h1)
        db = np.where(m, c["open_db"] + (c["hush_db"] - c["open_db"]) * into, db)
        lcut = np.where(m, np.log(20000.0) + (np.log(c["hush_cut"]) - np.log(20000.0)) * into, lcut)
    for r in rises:                                       # a swell over the bar after each rise
        x = (t - r) / bar
        db = db + c["swell_db"] * np.where((x >= 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)), 0.0)
    return db, np.exp(lcut)


def shape_song(song, plan, cfg, out, sr=48000):
    """The song with shape_envelope's gain and low-pass applied (in the STFT: a gentle 12 dB/octave roll-off above
    the cutoff, so the muffled stretches stay warm, not boxy). Film time t is song time t + plan["music_start"]."""
    import librosa
    import soundfile as sf
    y, _ = librosa.load(song, sr=sr, mono=False)
    y = y if y.ndim == 2 else np.stack([y, y])
    n_fft, hop = 4096, 1024
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    ft = librosa.frames_to_time(np.arange(S.shape[-1]), sr=sr, hop_length=hop) - float(plan.get("music_start", 0.0))
    db, fc = shape_envelope(plan, cfg, ft)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[:, None]
    H = 1.0 / np.sqrt(1.0 + (f / fc[None, :]) ** 4) * (10 ** (db / 20))[None, :]
    z = librosa.istft(S * H[None], hop_length=hop, length=y.shape[1])
    sf.write(out, z.T, sr, subtype="PCM_24")
    return out


CLEAN_DROP = ("logo", "logo_image", "title", "lift", "pull", "speed", "transition", "why")


def clean_plan(plan):
    """The same cut without what was added to the footage (Isaac, for a client: "a clean version ... keeping the same
    edit but without any filters or logos. Just the shots, so I can re use them"; no slow-mo or speed ramps, which
    tie footage to one track): every shot keeps its in-point, length and crop -- the same cut points, the same
    framing -- and loses its effects, speed changes, transition, lift, logo and title. No stinger."""
    for s in plan["shots"]:
        s["fx"], s["effect"] = [], None
        for k in CLEAN_DROP:
            s.pop(k, None)
    plan["stinger"] = 0.0
    return plan


def run(project, fmt, draft=False, cut=None, clean=False) -> str:
    """Render plan_<cut>.json (cut: a reel's format, or any other cut's name -- plan.place) to out/ (clean: the
    same cut clean_plan'd, to <name>_clean.mp4, with no overlay, exposure, stinger or logos)."""
    cut = cut or fmt
    PROJECT[0] = project
    if project.config.get("proxies_render") is False:      # the analysis read the proxies; the render reads the
        PROXIES[0] = dict(project.config.get("proxies") or {})   # camera files themselves (4K from 4K) -- a grid's
        project.config["proxies"] = {}                      # small cells excepted
    plan = project.load(f"plan_{cut}.json")
    fmt = plan.get("format", fmt)
    if clean:
        plan = clean_plan(plan)
    music = project.load(project.music_file) if os.path.exists(project.path(project.music_file)) else {}
    W, H = size(project, fmt)
    timer = Timer()
    global STAB_ANALYSIS
    STAB_ANALYSIS = "half" if str(project.config.get("stab_analysis") or "").lower() == "half" else "full"
    stab = project.config.get("stabilise")
    if not stab:                                             # off unless reel.json asks: shaky footage is avoided
        for s in plan["shots"]:                              # (plan.steady_shots), never warped (old plans too)
            s["stab"] = False
    elif isinstance(stab, list):                             # only some cameras ("stabilise": ["C0*"])
        import fnmatch
        for s in plan["shots"]:
            if not any(fnmatch.fnmatch(s["unit"].get("clip", ""), p) for p in stab):
                s["stab"] = False
    outdir, work = project.path("out"), project.path("work", cut)
    os.makedirs(outdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    marked = {} if clean else exposure.mark(project, plan)   # reel.json "exposure": blown-out / dark subjects
    for what, k in (("highlights pulled", 0), ("night subjects to lift (stops at most; faces decide)", 1)):
        done = {i: v[k] for i, v in marked.items() if v[k]}
        if done:
            print(f"[reel] {cut}: {what} on {len(done)} shots: "
                  + ", ".join(f"{i} {plan['shots'][i]['unit']['id']} {v:.2f}" for i, v in done.items()))
    from .velocity import speed_tails
    for line in speed_tails(plan):
        print(f"[reel] {cut}: WARNING speed: {line}")
    from ..fx import looks
    from . import overlay
    song = plan.get("song") or music.get("song")
    ov = None if clean else overlay.options(looks.overlay(project.config, fmt))   # reel.json "overlay"
    ahead, ahead_err = None, []
    if ov and not draft:                           # the MilkDrop layer needs the music and the acts, not the
        def layer():                               # pictures: it renders while the shots do
            t0 = time.time()
            try:
                overlay.prerender(work, plan, music, song, ov, W, H, FPS)
            except Exception as e:                 # (raised where the overlay would have rendered it)
                ahead_err.append(e)
            timer.add("milkdrop layer (beside the shots)", time.time() - t0)
        ahead = threading.Thread(target=layer, daemon=True)
        ahead.start()
    files = render_shots(project, plan, fmt, work, draft, timer)
    if not draft:
        with timer("transitions"):
            files = apply_transitions(plan, files)
    with open(os.path.join(outdir, f"{cut}_shots.json"), "w") as f:
        json.dump([os.path.relpath(p, project.reel_dir) for p in files], f, indent=1)
    if ov and not draft:                           # the visualiser layer over the cut (not the stinger)
        lst = os.path.join(work, "list_shots.txt")
        with open(lst, "w") as f:
            f.writelines("file '" + p.replace("'", "'\\''") + "'\n" for p in files)
        layered = os.path.join(work, "shots_overlay.mp4")
        with timer("milkdrop layer (waited for)"):
            ahead.join()
        if ahead_err:
            raise ahead_err[0]
        with timer("overlay"):
            overlay.apply(lst, layered, plan, music, song, ov, hw.enc(final=True), FPS)   # (the shots, read as
                                                                                           # one: no joined copy)
        files = [layered]
    stinger = project.stinger(fmt) if plan.get("stinger") and not clean else None
    with timer("stinger, join, mix and mux"):
        if stinger:                                    # the stinger as the last "shot", conformed to size and rate
            sting = os.path.join(work, f"stinger_{hw.enc_tag()}_{W}x{H}_{hw.ENC_VERSION}.mp4")
            if not os.path.exists(sting):
                hw.run_enc(["ffmpeg", "-v", "error", "-y", "-i", stinger, "-an",
                            "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS},setsar=1,format=yuv420p",
                            *hw.enc(final=True), sting + ".part.mp4"])
                os.replace(sting + ".part.mp4", sting)
            files = [*files, sting]
        lst = os.path.join(work, "list.txt")
        with open(lst, "w") as f:
            f.writelines("file '" + p.replace("'", "'\\''") + "'\n" for p in files)   # (concat's own quoting)
        mix = os.path.join(work, "mix.wav")
        shaped = song
        if project.config.get("music_shape") and not clean:   # the song's level and brightness built to the film
            shaped = shape_song(song, plan, project.config["music_shape"], os.path.join(work, "music_shaped.wav"))
        mix_audio(shaped, plan, stinger, mix, clean)
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", mix, "-af",
                              f"loudnorm=I={I}:TP={TP}:LRA=11:print_format=json", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
        gain = I - float(m["input_i"])                     # one gain for the whole mix (loudnorm's own linear mode
        level = (f"volume={gain:.2f}dB,alimiter=limit={10 ** (TP / 20):.4f}:attack=5:release=80:level=false"   # falls
                 f",aresample=48000")                    # back to squashing it when the mix has a real crescendo),
                                                         # the peaks limited
        final = os.path.join(outdir, f"{project.out_stem(cut, fmt)}{'_draft' if draft else ''}{'_clean' if clean else ''}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", mix,
                        "-map", "0:v", "-map", "1:a",      # (the picture joined straight into the delivered file)
                        "-af", level,
                        "-c:v", "copy", "-bsf:v", hw.VUI, "-c:a", "aac", "-b:a", "320k", "-shortest",
                        "-movflags", "+faststart",
                        final + ".part.mp4"], check=True)
        caps = [] if clean or project.config.get("burn_captions") is False else (project.config.get("captions") or [])
        # (reel.json "burn_captions": false -- the film clean, the captions only in the Premiere package's V2)
        if caps:                                                 # reel.json "captions": words over the film itself
            with timer("captions"):
                add_captions(project, final + ".part.mp4", caps, (W, H))
        for k in range(LOCK_WAIT):                               # a player holding the old file open (Windows locks
            try:                                                 # it): wait for it to let go rather than lose the render
                os.replace(final + ".part.mp4", final)
                break
            except PermissionError:
                if k == 0:
                    print(f"[reel] {os.path.basename(final)} is open in another program: waiting for it to close",
                          flush=True)
                if k == LOCK_WAIT - 1:
                    raise
                time.sleep(5)
    if not draft:                                            # the cut as rendered: what "keep": true re-cuts from
        project.dump(f"plan_{cut}{'_clean' if clean else ''}_rendered.json", plan)
    rep = timer.report()
    with open(os.path.join(outdir, f"{cut}{'_clean' if clean else ''}_timings.json"), "w") as f:
        json.dump(dict(rep, size=[W, H], frames=sum(s["frames"] for s in plan["shots"])), f, indent=1)
    print(f"[reel] {cut}: {W}x{H} in {rep['total']:.0f} s -- "
          + ", ".join(f"{k} {v:.0f} s" for k, v in rep.items() if k != "total" and v >= 0.5))
    print(f"[reel] -> {final}")
    for d in project.config.get("deliver") or []:            # reel.json "deliver": other versions of the film
        deliver(final, d)
    return final


def deliver(final, d, chunk=12.0, jobs=None):
    """A version of the finished film for where it will play (reel.json "deliver": [{"suffix": "_projector",
    "size": [1920, 1080], "fps": 30, "bitrate": "18M", "interp": "blend"}]): scaled, and its frame rate changed by
    making each new frame from its neighbours, not by dropping or repeating frames, so a 50p film plays evenly on a
    60 Hz projector laptop at 30 -- "blend" (default: a weighted mix of the two, fast) or "motion" (minterpolate,
    motion-compensated: best on quick motion, one core per stream) -- with the sound as is. The film goes through in
    `chunk`-second pieces side by side, joined after.
    Written beside the film."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    stem, ext = os.path.splitext(final)
    out = stem + d.get("suffix", f"_{d.get('fps', 30)}p") + ext
    w, h = d.get("size", [1920, 1080])
    fps, br = d.get("fps", 30), str(d.get("bitrate", "18M"))
    mbr = float(br.rstrip("M"))
    if d.get("interp") == "motion":                          # motion-compensated (best, slow: one core) or
        conv = f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"   # blended (default:
    else:                                                    # fast; a faint ghost only on quick motion)
        conv = f"framerate=fps={fps}:interp_start=0:interp_end=255:scene=100"
    vf = f"scale={w}:{h}:flags=lanczos,{conv},format=yuv420p"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                final], capture_output=True, text=True, check=True).stdout.strip())
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=os.path.dirname(final)) as tmp:
        pieces = [(k, k * chunk, min(chunk, dur - k * chunk)) for k in range(int(np.ceil(dur / chunk)))]
        pieces = [p for p in pieces if p[2] * fps >= 1.0]         # (not a sliver shorter than a frame)

        def one(p):
            k, a, n = p
            f = os.path.join(tmp, f"p{k:03d}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{n + 0.3:.3f}", "-i", final,
                            "-an",                         # (read a little past its end: its last frames need the next)
                            "-vf", vf, "-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-b:v", br,
                            "-maxrate", f"{mbr * 1.5:.0f}M", "-bufsize", f"{mbr * 2:.0f}M", "-r", str(fps),
                            "-frames:v", str(int(round(n * fps))), f], check=True)
            return f
        with ThreadPoolExecutor(jobs or max(2, min(len(pieces), (os.cpu_count() or 8) // 2))) as ex:
            files = list(ex.map(one, pieces))
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as fh:
            fh.write("".join(f"file '{f}'\n" for f in files))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", final,
                        "-map", "0:v", "-map", "1:a?", "-c", "copy", "-color_primaries", "bt709", "-color_trc",
                        "bt709", "-colorspace", "bt709", "-movflags", "+faststart", out + ".part.mp4"], check=True)
    os.replace(out + ".part.mp4", out)
    print(f"[reel] -> {out}  ({w}x{h} {fps}p, {d.get('interp', 'blend')}, {time.time() - t0:.0f} s)")
    return out
