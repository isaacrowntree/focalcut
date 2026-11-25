"""Exposure: find the daytime shots whose subject is blown out and pull their highlights back, shot by shot.

Judged on the subject only -- the subject track's box (frame.track) and the faces inside it (the people step's YuNet
scan) -- never the whole frame, so a bright sky or a white tent behind the dancer does not count. Measured on the
camera file's own 10-bit luma, where a sunlit face often sits above legal white (superwhite: Y' 940-1019), detail a
player clips away but a highlight shoulder brings back.

The fix is a smooth curve on luma only (lutyuv): untouched below KNEE; above it the brightest superwhite lands on
legal white (white stays white) and the upper midtones -- sunlit skin -- come down by up to pull. Shadows, midtones and colour stay as shot; it is a repair on the worst shots, not a
look. Night shots (a dark frame overall) are never pulled.

At night the opposite: faces that sit dark are lifted through a tracked window on them (window.py: the faces found
in the shot as it plays, an ellipse from head to chest feathered wide, a shadow lift with colour inside it, the room
outside as shot), toward "target" by at most "stops". This module only says which shots are night shots to try; the
window decides from the faces it finds, and leaves a shot without any (a crowd in silhouette, B-roll) alone. The old
midtone curve over the whole frame ("window": false) is kept for reels cut with it.

reel.json "exposure" (off unless set):
    {"auto": true,                      measure every shot and pull the blown-out ones
     "max": 0.12,                       the most any shot is pulled (of white, at the dip's deepest)
     "pull": {"C2148": 0.15,            by hand, per unit or per stretch of it ("<unit>@<in>-<out>", source time);
              "C2150@0:10-0:20": 0.1},  0 keeps a shot untouched
     "skip": ["C2153_crowd-dusk"],      never pulled or lifted
     "lift": true,                      night shots: lift dark faces through a tracked window (window.py), or
                                        {"target": 0.4, "stops": 0.6}; {"window": false, "target", "max"}: the
                                        old midtone curve over the whole frame
     "highlights": {"from": 0.78,       a bright daylight picture brought down by its top end and its overall
                    "frame": 0.45,      level, faces or none (highlight_amount); "only": clip patterns
                    "max": 0.12,
                    "only": ["C2*"]},
     "match": {"clips": ["baptisms"],  one scene shot at different exposures (the ISO raised for darker
               "level": "auto",         faces) brought to one level: each shot's background scaled to the
               "max": 1.2},             group's (match_gains), its faces then lifted back by the window
     "lift": {"day": true},             by day as well: faces that sit dark in a bright frame (dark skin in the
                                        shade of a cap, against noon sun on the water) are lifted through the same
                                        window -- the window still decides from the faces, and leaves lit ones be
     "lift_by": {"C2037": 0}}           by hand, like "pull": 0 keeps a shot as shot; with the window, the most
                                        it may brighten that shot's faces, in stops
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import frame as FR

BLACK, WHITE = 64, 940            # 10-bit video levels
TOP = 1.095                       # the brightest superwhite (Y' 1019), in legal-range units
KNEE = 0.45                       # the shoulder starts here: shadows and midtones stay as shot
MAX = 0.12                        # the most a shot is pulled (the curve stays monotonic below 0.19)
CLIP = 0.97                       # a pixel this bright (of legal white) has lost its detail on a player
DAY = 0.30                        # frame mean luma at or above this: a daytime shot
LIT = 0.55                        # faces darker than this (median) are in shade or backlit: nothing blown to fix
NIGHT_TARGET = 0.40              # a night subject's level (faces, else the median of the people) lifted toward this
LIFT_MAX = 0.08                   # the most a night shot is lifted (at mid-grey: a dark face ~2/3 of it)
SAMPLES = (0.2, 0.5, 0.8)         # where in the shot to look
GRAB_W = 960


def _matches(u, pats):
    """Whether a unit is one of the patterns -- its export category, or its clip name by pattern (None: every
    unit)."""
    import fnmatch
    return not pats or any(u.get("category") == p or fnmatch.fnmatch(u.get("clip", ""), p) for p in pats)


def config(project) -> dict:
    return project.config.get("exposure") or {}


def grab(path, t, rotate=None):
    """The 10-bit luma (legal range -> 0..1, superwhite above 1) of one frame, GRAB_W wide."""
    vf = (f"{rotate}," if rotate else "") + f"scale={GRAB_W}:-2:flags=area"
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-an", "-vf", vf,
                        "-pix_fmt", "gray16le", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(r, "<u2").astype(np.float32) / 64.0       # gray16le carries 10-bit luma shifted up
    h = len(y) // GRAB_W
    return ((y[:h * GRAB_W].reshape(h, GRAB_W) - BLACK) / (WHITE - BLACK)) if h else None


def regions(project, unit, times):
    """Per time: the subject box and the face boxes inside it, as frame fractions (x0, y0, x1, y1)."""
    if unit.get("rotate"):                                      # no subject track in the rotated frame
        return [((0.3, 0.15, 0.7, 0.85), []) for _ in times]
    stem = project.path("scores", unit["score"])
    union = np.full((len(times), 4), np.nan)
    if os.path.exists(stem + ".npz"):
        union, _, _ = FR.track(FR.load(stem + ".npz"), np.asarray(times, float))
    f = project.path("scores", f"faces_{unit['id']}.npz")
    yf = dict(np.load(f)) if os.path.exists(f) else None
    out = []
    for t, u in zip(times, union, strict=True):
        box = tuple(u) if not np.isnan(u).any() else (0.3, 0.15, 0.7, 0.85)
        faces = []
        if yf is not None:
            near = (np.abs(yf["t"] - t) <= 0.3) & (yf["score"] >= FR.FACE_SCORE) & (yf["h"] >= FR.FACE_H)
            for cx, cy, h in zip(yf["cx"][near], yf["cy"][near], yf["h"][near], strict=True):
                if box[0] <= cx <= box[2] and box[1] - 0.02 <= cy <= box[3]:
                    faces.append((cx - 0.3 * h * 9 / 16, cy - 0.3 * h, cx + 0.3 * h * 9 / 16, cy + 0.3 * h))
        out.append((box, faces))
    return out


def _cut(y, b):
    h, w = y.shape
    x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(b, (w, h, w, h), strict=True))
    return y[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]


def measure(project, unit, src_in, length):
    """The subject's exposure over a shot: frame (mean luma of the whole frame: day or night), subject (share of
    the subject's pixels past CLIP, from the top of the faces down), face (median face luma, None if no face is seen), face_clip, top (99.5th
    percentile of the subject), mid (median of the subject's pixels)."""
    times = [src_in + length * k for k in SAMPLES]
    rows = []
    for t, (box, faces) in zip(times, regions(project, unit, times), strict=True):
        path, dt = project.decode_from(unit["source"], t)
        y = grab(path, dt, unit.get("rotate"))
        if y is None:
            continue
        top = min(f[1] for f in faces) if faces else box[1] + 0.25 * (box[3] - box[1])
        s = _cut(y, (box[0], max(box[1], top), box[2], box[3]))   # the people from their heads down: not the sky
        fp = np.concatenate([_cut(y, f).ravel() for f in faces]) if faces else np.zeros(0)
        hh, ww = y.shape
        bgm = np.ones_like(y, bool)                              # the background: everything but the subject
        bx0, by0, bx1, by1 = (int(round(v * d)) for v, d in zip(box, (ww, hh, ww, hh), strict=True))
        bgm[max(by0, 0):max(by1, 0), max(bx0, 0):max(bx1, 0)] = False
        rows.append(dict(back=float(np.median(y[bgm])) if bgm.any() else float(np.median(y)),
                         frame=float(y.mean()), subject=float((s >= CLIP).mean()) if s.size else 0.0,
                         face=float(np.median(fp)) if fp.size else None,
                         face_clip=float((fp >= CLIP).mean()) if fp.size else None,
                         top=float(np.percentile(s, 99.5)) if s.size else 0.0,
                         mid=float(np.median(s)) if s.size else None))
    if not rows:
        return None
    faces = [r["face"] for r in rows if r["face"] is not None]
    fclip = [r["face_clip"] for r in rows if r["face_clip"] is not None]
    return dict(back=round(float(np.median([r["back"] for r in rows])), 3),
                frame=round(float(np.mean([r["frame"] for r in rows])), 3),
                subject=round(float(np.mean([r["subject"] for r in rows])), 4),
                face=round(float(np.mean(faces)), 3) if faces else None,
                face_clip=round(float(np.mean(fclip)), 3) if fclip else None,
                top=round(float(np.mean([r["top"] for r in rows])), 3),
                mid=round(float(np.mean(mids)), 3) if (mids := [r["mid"] for r in rows if r["mid"] is not None]) else None)


def amount(m, top=MAX):
    """How far to pull a measured shot: 0 for night and for subjects that hold their detail; more the more of the
    subject (and above all its face) is blown."""
    if not m or m["frame"] < DAY or m["face"] is None or m["face"] < LIT:
        return 0.0                                  # night, no face to judge by, or faces in shade / silhouette
    bad = max(m["subject"] / 0.10, m["face_clip"] / 0.20, (m["face"] - 0.80) / 0.10)
    if bad < 1.0:
        return 0.0
    return round(min(top, 0.03 + 0.04 * (bad - 1.0)), 3)


def highlight_amount(m, h):
    """A bright daylight picture brought down (reel.json "exposure" "highlights": {"from": 0.78, "frame": 0.45,
    "max": 0.12}): however its faces read, a shot whose top end (the subject's 99.5th percentile) sits above `from`
    of white, or whose whole frame averages above `frame`, is pulled -- the curve only moves the upper midtones
    and highlights (above KNEE), so faces in the midtones keep their level. Night shots are left alone."""
    if not m or m["frame"] < DAY:
        return 0.0
    top = float(h.get("max", MAX))
    a = max(0.0, (m["top"] - float(h.get("from", 0.78))) / 0.2) + max(0.0, (m["frame"] - float(h.get("frame", 0.45))) / 0.25)
    return round(min(top, top * a), 3) if a > 0.05 else 0.0


def match_gains(cfg, shots, cache):
    """reel.json "exposure" "match": {"clips": ["baptisms"], "level": "auto" | 0.6, "max": 1.2}: shots of one scene
    shot at different exposures (the ISO raised a stop to see darker faces) brought to one level -- each shot's
    background (everything but its subject) scaled to the level the group sits at ("auto": its lower third), never
    brighter, by at most "max" stops. Video is gamma-encoded, so a change of exposure is a scale of the signal: the
    gain is (level / background). {shot index: gain < 1}."""
    if not cfg:
        return {}
    pats = cfg.get("clips")
    backs = {i: cache.get(_key(s["unit"], s)) for i, s in enumerate(shots) if _matches(s["unit"], pats)}
    backs = {i: m["back"] for i, m in backs.items() if m and m.get("back")}
    if len(backs) < 2:
        return {}
    level = cfg.get("level", "auto")
    level = float(np.percentile(list(backs.values()), 33)) if level == "auto" else float(level)
    least = 2 ** (-float(cfg.get("max", 1.2)) / 2.2)
    return {i: round(max(least, level / b), 3) for i, b in backs.items() if b > level * 1.05}


PROTECT = (0.25, 0.60)    # match gains spare levels below the first, act in full above the second (gain_filter)


def protect_levels(cfg):
    """reel.json "exposure" "match" "protect": [from, to] (legal-range luma) -- the gain leaves levels below `from`
    (skin, dark skin above all, and shadows) as shot and takes its full effect above `to` (the water, sky, sand that
    the raised ISO blew bright); false scales the whole picture. Default PROTECT."""
    v = (cfg or {}).get("protect", PROTECT)
    return tuple(float(x) for x in v) if v else None


def protected(y, g, protect):
    """A level y (legal-range luma, 0..1) under gain g protected below protect[0] (the curve gain_filter renders)."""
    e = min(max((y - protect[0]) / (protect[1] - protect[0]), 0.0), 1.0)
    return y * (1 - (1 - g) * e * e * (3 - 2 * e))


def protect_span(g, protect, slope=0.25):
    """protect widened until the protected curve rises everywhere at least `slope` (a strong gain over a narrow span
    would make the water darker than the skin below it: a tone inversion) -- the top first, up to white, then the
    bottom down (a gain that strong spares a little less of the skin)."""
    ys = np.linspace(0.0, 1.0, 501)
    lo, hi = protect

    def rises(lo, hi):
        f = np.array([protected(y, g, (lo, hi)) for y in ys])
        return (np.diff(f) / np.diff(ys)).min() >= slope
    while hi < 1.0 and not rises(lo, hi):
        hi = min(hi + 0.02, 1.0)
    while lo > 0.0 and not rises(lo, hi):
        lo = max(lo - 0.02, 0.0)
    return round(lo, 3), round(hi, 3)


def gain_filter(g, protect=None, tag="g"):
    """A shot's exposure scaled by g (gamma-encoded: luma above black and chroma around neutral, alike). With
    protect (from, to): only the bright part of the picture -- the gained picture laid over the shot as shot, keyed
    by the shot's own luma (smoothstep from..to), so faces and shadows keep their level and their colour while the
    water and sky come down; no window, so no halo. Inline filter-graph text for a -vf / -filter_complex chain."""
    gain = (f"lutyuv=y='(val-minval)*{g:.4f}+minval':u='(val-(maxval+minval)/2)*{g:.4f}+(maxval+minval)/2'"
            f":v='(val-(maxval+minval)/2)*{g:.4f}+(maxval+minval)/2'")
    if not protect:
        return gain
    k0, k1 = 16 + 219 * protect[0], 219 * (protect[1] - protect[0])
    e = f"clip((val-{k0:.2f})/{k1:.2f}\\,0\\,1)"
    q = f"255*{e}*{e}*(3-2*{e})"
    return (f"format=yuv422p10le,split=3[{tag}b][{tag}x][{tag}q0];[{tag}x]{gain}[{tag}y];"
            f"[{tag}q0]extractplanes=y,format=gray,lut=c0={q}[{tag}q];[{tag}y][{tag}q]alphamerge[{tag}a];"
            f"[{tag}b][{tag}a]overlay=format=yuv422p10")


def lift_amount(m, target=NIGHT_TARGET, top=LIFT_MAX):
    """How far to lift a measured night shot: 0 by day and for subjects already at target; more the darker the
    subject (its faces, else the people's median), never past top."""
    if not m or m["frame"] >= DAY:
        return 0.0
    level = m["face"] if m.get("face") is not None else m.get("mid")
    if level is None or level >= target:
        return 0.0
    return round(min(top, 0.01 + 0.4 * (target - level)), 3)


def lift_curve(lift):
    """lutyuv luma expression (10-bit values): x + 4 * lift * x * (1 - x) over legal range -- lift at mid-grey, less
    toward black and white, which stay where they are (as does anything outside legal range)."""
    g = 4 * lift / (WHITE - BLACK)
    return f"if(between(val\\,{BLACK}\\,{WHITE})\\,val+{g:.8f}*(val-{BLACK})*({WHITE}-val)\\,val)"


def curve(pull):
    """lutyuv luma expression (10-bit values). Below KNEE: as shot. Above it, two smooth parts, both with zero slope
    change at the knee: a quadratic shoulder that lands the brightest superwhite (TOP) on legal white, so a player
    keeps the detail it would clip and white stays white (not grey); and a sin^2 dip, pull deep at its middle (~0.77
    of white: sunlit skin), zero at the knee and at TOP, that brings the blown skin and clothes back down."""
    k = BLACK + KNEE * (WHITE - BLACK)
    d = (TOP - KNEE) * (WHITE - BLACK)                       # the shoulder's width, 10-bit
    a = (TOP - 1.0) / (TOP - KNEE) ** 2 / (WHITE - BLACK)
    dip = pull * (WHITE - BLACK)
    return (f"if(lt(val\\,{k:.1f})\\,val\\,val-{a:.7f}*(val-{k:.1f})*(val-{k:.1f})"
            f"-{dip:.2f}*pow(sin(PI*(val-{k:.1f})/{d:.1f})\\,2))")


def filt(pull, lift=0.0):
    """The shot's highlight pull, or its night lift, for its filter chain ('' for neither)."""
    if pull and pull > 0:
        return f"format=yuv422p10le,lutyuv=y={curve(pull)}"
    return f"format=yuv422p10le,lutyuv=y={lift_curve(lift)}" if lift and lift > 0 else ""


def _key(unit, s):
    return f"{unit['id']}@{s['src_in']:.2f}+{s['length']:.2f}"


def _span(k):
    """'<unit>@<in>-<out>' (m:ss or seconds) -> (unit, in, out); a bare unit -> (unit, None, None)."""
    if "@" not in k:
        return k, None, None
    u, r = k.split("@", 1)
    sec = [sum(float(p) * 60 ** i for i, p in enumerate(reversed(x.split(":")))) for x in r.split("-")]
    return u, sec[0], sec[1]


def by_hand(cfg, unit, s, key="pull"):
    """A pull (or with key "lift_by", a lift) reel.json sets for this shot (its unit, or a stretch of it the shot
    overlaps), else None."""
    for k, v in (cfg.get(key) or {}).items():
        u, a, b = _span(k)
        if u in (unit["id"], unit.get("clip")) and (a is None or (s["src_in"] < b and s["src_in"] + s["length"] > a)):
            return float(v)
    return None


def lift_config(cfg):
    """reel.json "exposure" "lift" as (target, max), or None when night shots are left as shot."""
    v = cfg.get("lift")
    if not v:
        return None
    v = v if isinstance(v, dict) else {}
    return float(v.get("target", NIGHT_TARGET)), float(v.get("max", LIFT_MAX))


def mark(project, plan, workers=6):
    """Give each shot of the plan its "pull" and its night "lift" (removed where there is none), measuring what is
    not cached (focalcut/reel/exposure.json). Returns {shot index: (pull, lift)} for the shots changed."""
    cfg = config(project)
    shots = plan["shots"]
    for s in shots:
        s.pop("pull", None)
        s.pop("lift", None)
    if not cfg:
        return {}
    lc = lift_config(cfg)
    from .window import settings
    win = settings(cfg.get("lift"))                          # the lift through a window on the faces (window.py)
    day = isinstance(cfg.get("lift"), dict) and bool(cfg["lift"].get("day"))   # by day too: faces in hard shade
    cache_f = project.path("exposure.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    skip = set(cfg.get("skip") or [])

    def needs(s):                                            # measured (with the subject's median, for the lift)
        m = cache.get(_key(s["unit"], s), False)
        auto = cfg.get("auto") and by_hand(cfg, s["unit"], s) is None
        lift = lc and by_hand(cfg, s["unit"], s, "lift_by") is None
        match = _matches(s["unit"], (cfg.get("match") or {}).get("clips")) if cfg.get("match") else False
        return ((auto and m is False) or (lift and (m is False or (m and "mid" not in m)))
                or (match and (m is False or (m and "back" not in m))))
    want = [s for s in shots if s["unit"]["id"] not in skip and not s.get("photo") and not s.get("photos")
            and needs(s)]   # (a still photo:
                                                                                                    # graded already)
    if want:
        with ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(lambda s: measure(project, s["unit"], s["src_in"], s["length"]), want))
        for s, m in zip(want, got, strict=True):
            cache[_key(s["unit"], s)] = m
        with open(cache_f + ".part", "w") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
        os.replace(cache_f + ".part", cache_f)
    gains = match_gains(cfg.get("match"), shots, cache)
    out = {}
    for i, s in enumerate(shots):
        u = s["unit"]
        s.pop("gain", None)
        s.pop("gain_protect", None)
        if s.get("photo") or s.get("photos"):
            continue
        if i in gains:
            s["gain"] = gains[i]
            pl = protect_levels(cfg.get("match"))
            if pl:
                s["gain_protect"] = list(protect_span(gains[i], pl))
        if u["id"] in skip:
            continue
        m = cache.get(_key(u, s))
        p = by_hand(cfg, u, s)
        if p is None and cfg.get("auto"):
            p = amount(m, float(cfg.get("max", MAX)))
            if cfg.get("highlights") and _matches(u, cfg["highlights"].get("only")):
                p = max(p or 0.0, highlight_amount(m, cfg["highlights"]))
        if p and not s.get("gain"):                           # (a shot matched to its scene is already down:
            s["pull"] = min(round(p, 3), 0.18)                # no highlight pull on top)
        lf = by_hand(cfg, u, s, "lift_by")
        if s.get("gain_protect") and lf is None:             # matched with its faces spared: they are as shot
            lf = 0                                             # (the ISO raised for them), so no window on top
        if win and not s.get("pull") and not s.get("logo") and lf != 0 and m and (m["frame"] < DAY or day or s.get("gain")):
            s["lift"] = dict(win, stops=float(lf)) if lf else dict(win)   # (window.prepare decides from the faces)
            if s.get("gain") and not s.get("gain_protect"):   # a shot brought down a stop as a whole: its faces
                taken = -np.log2(s["gain"]) * 2.2              # lifted back by what it took (and a little more)
                s["lift"]["stops"] = round(min(1.6, max(s["lift"]["stops"], taken + 0.2)), 2)
        elif not win:
            if lf is None and lc and not s.get("pull"):
                lf = lift_amount(m, *lc)
            if lf and not s.get("pull"):
                s["lift"] = min(round(lf, 3), 0.15)
        if s.get("pull") or s.get("lift"):
            lv = s.get("lift", 0.0)
            out[i] = (s.get("pull", 0.0), lv["stops"] if isinstance(lv, dict) else lv)
    return out
