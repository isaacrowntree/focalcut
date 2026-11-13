"""Who a portrait strip is on, and keeping them in it.

A 9:16 strip cut from a 16:9 frame is a third of the frame wide: it shows one person (or two close together), and
it has to say which, then stay on them while they move. This module answers both from what the analysis already
measured -- the pose pass (boxes and keypoints, 5 a second), the face scan (2 a second), the focus pass and the
camera's own motion (shake.py: 25 a second):

tracks()   the people of a shot, each linked box to box through it: where their head and torso are at every pose
           frame, when their face shows, how big, sharp and busy they are.
choose()   the shot's subject -- one person for the whole shot, never handed over mid-shot: the one doing
           something (dancing, jumping), near the lens and in focus, the one the operator is following (the camera
           pans and they come to the middle), there when the shot opens. A clear face helps and is not required:
           a mask, a costume, a head down or a dancer side-on is still the subject.
aim()      where the strip's centre goes at every pose frame: the subject's head and body both in it; someone
           dancing right with them shares the strip when both fit (weighted to the nearer one) and is otherwise left
           out whole, never cut in half; someone lying down (a floor move) is framed on their torso; hands at work
           with no head in frame (a DJ's hands on the mixer) are framed where the fingers meet the gear.
glide()    that centre as a path the renderer follows: carried between pose frames by the camera's own motion (so
           a pan doesn't leave the subject behind) and smoothed only enough to take out the detector's jitter.

Everything here is derived; the editor's focal points and follow clicks are laid over it afterwards (plan.reframe).
"""
from __future__ import annotations

import numpy as np

from . import frame as FR

LINK_GAP = 0.9           # a person lost for longer than this (s) is a new track when they come back
FACE_SCORE, FACE_H = 0.65, 0.03                   # a face that counts (as frame.face_seen)
DET = 0.55               # a pose box under this confidence is no one (a bag, a shadow, half a person)
KP = 0.4                 # a keypoint this confident is used
HEAD = (0, 1, 2, 3, 4)   # COCO: nose, eyes, ears
ASPECT = 9 / 16          # frame height over width, for distances in frame widths

# choose(): what makes someone the shot's subject. Weights fitted on the editor's own corrections and on the shots
# they passed (bench/portrait_eval.py --fit), rounded; each feature is 0..1 across the people of the shot.
WEIGHTS = dict(size=1.7, act=1.0, bounce=0.45, face=0.35, sharp=1.0, seen=1.8, opens=0.0, central=1.4, centring=0.35,
               front=0.2, opener=0.85, held=0.0, alone=1.0, dom=1.0)
BEHIND = 0.8             # choose(): the person a shot opens on keeps it only if they're at least this big against the
SOFTER = 0.85            # best candidate (not further back) and at least this sharp (not out of the focus plane)
FOCUS = 0.6              # choose(): someone this much softer than the sharpest person in shot is never the subject
KEEP = 0.33              # choose(): the score someone needs over the person the shot opens on to take it from them
TOGETHER = 0.85          # a companion's head is within this share of the strip's width of the subject's
PEER = 0.6               # a companion is at least this share of the subject's size (head or body height)
LEAN = 0.75              # a spine this far off upright (0 standing .. 1 flat) is lying down
LYING = 1.25             # a body this many times wider than tall (in pixels) is lying down: frame the torso
HEAD_IN = 0.3            # the subject's head stays within this share of the strip's width of its centre
FINGERS = 0.6            # hands at work: the fingertips are this far past the wrist, in forearm lengths
SMOOTH = 0.12            # glide(): seconds of smoothing on the strip's path (the detector's jitter, not the move)
RATE = 0.1               # the path's sample spacing (s)


def _norm(d):
    return d["boxes"][..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]


def _head(kp, box, aw, ah):
    """A pose person's head: (x, y, size, sure) as frame fractions -- the mean of their nose, eyes and ears, else a
    head above the shoulders, else the top of their box (not sure)."""
    x0, y0, x1, y1 = box
    ok = kp[list(HEAD), 2] > KP
    sh = kp[[5, 6], 2] > KP
    size = 0.14 * (y1 - y0)
    if ok.sum() >= 2 or ok[0]:
        p = kp[list(HEAD)][ok]
        hx, hy = p[:, 0].mean() / aw, p[:, 1].mean() / ah
        if ok.sum() >= 2:
            size = max(size, 1.4 * np.ptp(p[:, 0]) / aw / ASPECT * 0.8)
        return hx, hy, float(min(size, 0.5)), True
    if sh.all():
        s = kp[[5, 6]]
        sw = abs(s[0, 0] - s[1, 0]) / aw
        return s[:, 0].mean() / aw, max(s[:, 1].mean() / ah - 0.6 * sw / ASPECT, y0), float(size), True
    return (x0 + x1) / 2, y0 + 0.12 * (y1 - y0), float(size), False


def _torso(kp, box, aw, ah):
    """The middle of a pose person's torso (x, y): shoulders and hips, else their box's."""
    ok = kp[[5, 6, 11, 12], 2] > KP
    if ok[:2].any() and ok[2:].any():
        p = kp[[5, 6, 11, 12]]
        top, low = p[:2][ok[:2]], p[2:][ok[2:]]
        return (top[:, 0].mean() + low[:, 0].mean()) / 2 / aw, (top[:, 1].mean() + low[:, 1].mean()) / 2 / ah
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def camera(shake, t0, t1):
    """The camera's own motion over a stretch: (times, cumulative dx, dy in frame widths / heights of the picture
    moving across the frame), from the shake pass (scores/<unit>_shake.npz: t, mv). None without it."""
    if shake is None or "mv" not in shake or len(shake["t"]) < 2:
        return None
    t, mv = np.asarray(shake["t"], float), np.asarray(shake["mv"], float)
    sel = (t >= t0 - 0.2) & (t <= t1 + 0.2)
    if sel.sum() < 2:
        return None
    return t[sel], np.cumsum(mv[sel, 0]), np.cumsum(mv[sel, 1]) / ASPECT


def tracks(d, yf, t0, t1):
    """The people of a stretch, one track each. A track: dict of arrays over its pose frames -- t, box (n, 4) frame
    fractions, head (n, 2), hs (head size, of the frame's height), sure (the head is from keypoints), torso (n, 2),
    act, foc, slot, face (a confident face of theirs within 0.3 s), fh (its height, else nan), and wrists/elbows
    (n, 2, 3) for hands at work. People are linked frame to frame by where their head and box were."""
    pt = d["pose_t"]
    ff = np.nonzero((pt >= t0) & (pt <= t1))[0]
    if not len(ff):
        return []
    nb, kp, aw, ah = _norm(d), d["kp"], float(d["aw"]), float(d["ah"])
    act = FR.activity(d)
    foc = d["focus"] if "focus" in d else np.full(nb.shape[:2], np.nan)
    conf = d["boxes"][..., 4] if d["boxes"].shape[-1] > 4 else np.ones(nb.shape[:2])
    live, done = [], []
    for f in ff:
        dets = []
        for k in range(nb.shape[1]):
            b = nb[f, k]
            if np.isnan(b[0]) or conf[f, k] < DET:
                continue
            hx, hy, hs, sure = _head(kp[f, k], b, aw, ah)
            twin = next((o for o in dets if abs(o["head"][0] - hx) < 0.05 and abs(o["head"][1] - hy) < 0.09), None)
            if twin is not None:                              # the pose pass found them twice: the surer box
                if twin["conf"] >= conf[f, k]:
                    continue
                dets = [o for o in dets if o is not twin]
            tx, ty = _torso(kp[f, k], b, aw, ah)
            q = kp[f, k][[5, 6, 11, 12]]
            lean = np.nan                                     # how far the spine is off upright: 0 standing, 1 flat
            if (q[:2, 2] > KP).any() and (q[2:, 2] > KP).any():
                top, low = q[:2][q[:2, 2] > KP], q[2:][q[2:, 2] > KP]
                dx, dy = abs(top[:, 0].mean() - low[:, 0].mean()), abs(top[:, 1].mean() - low[:, 1].mean())
                lean = float(np.arctan2(dx, dy) / (np.pi / 2))
            dets.append(dict(t=float(pt[f]), box=b, head=(hx, hy), hs=hs, sure=sure, torso=(tx, ty),
                             act=float(act[f, k]), foc=float(foc[f, k]), slot=k, conf=float(conf[f, k]), lean=lean,
                             arms=np.array([[kp[f, k, j, 0] / aw, kp[f, k, j, 1] / ah, kp[f, k, j, 2]]
                                            for j in (7, 8, 9, 10)])))
        live = [tr for tr in live if pt[f] - tr[-1]["t"] <= LINK_GAP or done.append(tr)]
        pairs = []
        for i, tr in enumerate(live):
            a = tr[-1]
            for j, b in enumerate(dets):
                ha, hb = a["box"][3] - a["box"][1], b["box"][3] - b["box"][1]
                wa = max(a["box"][2] - a["box"][0], b["box"][2] - b["box"][0], 0.06)
                dd = 0.5 * np.hypot(a["head"][0] - b["head"][0], (a["head"][1] - b["head"][1]) * ASPECT) + \
                    0.5 * abs((a["box"][0] + a["box"][2]) - (b["box"][0] + b["box"][2])) / 2
                if dd <= 0.6 * wa and 0.5 <= hb / max(ha, 1e-6) <= 2.0:
                    pairs.append((dd / wa, i, j))
        ti, dj = set(), set()
        for _, i, j in sorted(pairs):
            if i in ti or j in dj:
                continue
            live[i].append(dets[j]); ti.add(i); dj.add(j)
        live += [[b] for j, b in enumerate(dets) if j not in dj]
    out = []
    for tr in done + live:
        o = {k: np.array([x[k] for x in tr]) for k in ("t", "box", "head", "hs", "sure", "torso", "act", "foc", "slot",
                                                         "arms", "lean")}
        o["face"], o["fh"] = np.zeros(len(tr), bool), np.full(len(tr), np.nan)
        out.append(o)
    _faces(out, yf, t0, t1)
    for o in out:                                             # a head the keypoints missed some of the time (a
        ok = o["sure"]                                        # costume, a turn): where it sits in their box the
        if ok.any() and not ok.all():                         # rest of the time
            w = np.maximum(o["box"][:, 2] - o["box"][:, 0], 1e-3)
            h = np.maximum(o["box"][:, 3] - o["box"][:, 1], 1e-3)
            fx = float(np.median((o["head"][ok, 0] - o["box"][ok, 0]) / w[ok]))
            fy = float(np.median((o["head"][ok, 1] - o["box"][ok, 1]) / h[ok]))
            o["head"][~ok] = np.c_[o["box"][~ok, 0] + fx * w[~ok], o["box"][~ok, 1] + fy * h[~ok]]
    return out


def _faces(trs, yf, t0, t1):
    """Give each face of the scan to the person whose head it is (in the head of their box, their build's size, the
    nearest head): sets face / fh on their pose frames within 0.3 s, and moves the head there -- a face is a
    surer fix on a head than the keypoints."""
    if yf is None or not trs:
        return
    keep = (yf["t"] >= t0 - 0.3) & (yf["t"] <= t1 + 0.3) & (yf["score"] >= FACE_SCORE) & (yf["h"] >= FACE_H)
    for tr in trs:
        tr["scan"] = []
    for ts in np.unique(yf["t"][keep]):
        m = keep & (yf["t"] == ts)
        cand = []
        for a, (cx, cy, fh) in enumerate(zip(yf["cx"][m], yf["cy"][m], yf["h"][m])):
            for i, tr in enumerate(trs):
                j = int(np.abs(tr["t"] - ts).argmin())
                if abs(tr["t"][j] - ts) > 0.3:
                    continue
                x0, y0, x1, y1 = tr["box"][j]
                build = max(0.45 * (x1 - x0) / ASPECT, 0.12 * (y1 - y0))
                if x0 - 0.02 <= cx <= x1 + 0.02 and y0 - 0.03 <= cy <= y0 + 0.5 * (y1 - y0) and fh >= 0.3 * build:
                    dd = np.hypot(cx - tr["head"][j, 0], (cy - tr["head"][j, 1]) * ASPECT) if tr["sure"][j] else \
                        0.05 + abs(cx - (x0 + x1) / 2)
                    if tr["sure"][j] and dd > 1.2 * fh:        # their head is elsewhere: someone else's face, seen
                        continue                              # past them (a big box takes in half the frame)
                    cand.append((dd / max(fh, 1e-3), a, i, cx, cy, fh))
        fa, ta = set(), set()
        for _, a, i, cx, cy, fh in sorted(cand):
            if a in fa or i in ta:
                continue
            fa.add(a); ta.add(i)
            tr = trs[i]
            tr["scan"].append((float(ts), float(cx), float(cy), float(fh)))
            near = np.abs(tr["t"] - ts) <= 0.3
            tr["face"] |= near
            tr["fh"][near] = fh
            tr["hs"][near] = fh
            j = int(np.abs(tr["t"] - ts).argmin())
            if not tr["sure"][j] or np.hypot(cx - tr["head"][j, 0], (cy - tr["head"][j, 1]) * ASPECT) > 0.35 * fh:
                # the keypoints missed the head (a mask, a hat, a head tipped right back): the face is where it is,
                # and the frames around it keep the same correction
                off = np.array([cx, cy]) - tr["head"][j]
                tr["head"][near] = tr["head"][near] + off
                tr["sure"][near] = True


def features(trs, t0, length, cam=None, strip=None):
    """What choose() weighs, per track (N,) each 0..1: size (how near the lens: their box and head against the
    biggest), act (how much they dance, against the busiest), bounce (their head's own up-and-down: jumping), face
    (the share of the shot their face shows), sharp (focus against the sharpest), seen (the share of the shot they
    are in frame), opens (there when it opens), central (how near the middle they sit), centring (how far they come
    toward the middle over the shot -- with a moving camera, the one the operator follows), front (nobody bigger
    overlaps them: not half hidden behind someone), and against the strip the body tracker cut (strip: times from
    the shot's start, centres, width): opener (their head is in it as the shot opens) and held (the share of the
    shot it holds their head)."""
    t1 = t0 + length
    n = len(trs)
    f = {k: np.zeros(n) for k in (*WEIGHTS, "mid")}
    if not n:
        return f
    grid = np.arange(t0, t1 + 1e-6, 0.2)
    span = max(len(grid), 1)
    moved = 0.0
    if cam is not None:
        ct, cx, _ = cam
        moved = abs(float(np.interp(t1, ct, cx) - np.interp(t0, ct, cx)))
    raw = dict(size=np.zeros(n), act=np.zeros(n), sharp=np.full(n, np.nan))
    for i, tr in enumerate(trs):
        m = (tr["t"] >= t0 - 0.1) & (tr["t"] <= t1 + 0.1)
        if not m.any():
            continue
        b = np.clip(tr["box"][m], 0, 1)
        hgt, wid = b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]
        raw["size"][i] = float(np.median(np.sqrt(hgt * wid * ASPECT)) + 0.5 * np.nanmedian(np.where(
            np.isnan(tr["fh"][m]), tr["hs"][m], tr["fh"][m])))
        raw["act"][i] = float(np.mean(tr["act"][m]))
        if not np.isnan(tr["foc"][m]).all():
            raw["sharp"][i] = float(np.nanmean(tr["foc"][m]))
        f["seen"][i] = min(1.0, m.sum() / span)
        f["opens"][i] = float(tr["t"][m][0] <= t0 + 0.45)
        f["face"][i] = float(tr["face"][m].mean())
        hx, hy = tr["head"][m, 0], tr["head"][m, 1]
        f["central"][i] = float(np.clip(1 - 2 * np.mean(np.abs(tr["torso"][m, 0] * 0.5 + hx * 0.5 - 0.5)), 0, 1))
        if strip is not None:
            d = np.abs(hx - np.interp(tr["t"][m] - t0, strip[0], strip[1])) / (0.4 * strip[2])
            off = d <= 1
            f["held"][i] = float(off.mean())
            first = tr["t"][m] <= t0 + 0.5
            f["opener"][i] = float(off[first].mean()) if first.any() else 0.0
            f["mid"][i] = float(np.clip(1 - d[first], 0, 1).mean()) if first.any() else 0.0
        if m.sum() >= 3:
            k = max(1, m.sum() // 3)
            gain = abs(hx[:k].mean() - 0.5) - abs(hx[-k:].mean() - 0.5)
            f["centring"][i] = float(np.clip(gain / 0.2, 0, 1) * np.clip(moved / 0.15, 0, 1))
            wy = hy * 1.0
            if cam is not None:
                wy = hy - np.interp(tr["t"][m], cam[0], cam[2])
            wy = wy - np.polyval(np.polyfit(tr["t"][m], wy, 1), tr["t"][m])     # a rise and fall, not a drift
            f["bounce"][i] = float(np.clip(np.std(wy) / max(np.median(hgt), 1e-3) / 0.06, 0, 1))
    for k in ("size", "act"):
        f[k] = raw[k] / max(raw[k].max(), 1e-6)
    if not np.isnan(raw["sharp"]).all():
        f["sharp"] = np.nan_to_num(raw["sharp"] / np.nanmax(raw["sharp"]), nan=0.5)
    if n > 1:                                                 # far bigger than anyone else: right up front
        o = np.argsort(-raw["size"])
        f["alone"][o[0]] = float(np.clip((raw["size"][o[0]] / max(raw["size"][o[1]], 1e-6) - 1.3) / 0.7, 0, 1))
    else:
        f["alone"][:] = 1.0
    for i, tr in enumerate(trs):                              # half hidden behind someone bigger
        hid = 0.0
        for j, o in enumerate(trs):
            if i == j or raw["size"][j] <= 1.15 * raw["size"][i]:
                continue
            for a in range(len(tr["t"])):
                if not t0 - 0.1 <= tr["t"][a] <= t1 + 0.1:
                    continue
                b = int(np.abs(o["t"] - tr["t"][a]).argmin())
                if abs(o["t"][b] - tr["t"][a]) > 0.15:
                    continue
                x0, _, x1, _ = tr["box"][a]
                ox0, _, ox1, _ = o["box"][b]
                hid += max(0.0, min(x1, ox1) - max(x0, ox0)) / max(x1 - x0, 1e-3)
        m = (tr["t"] >= t0 - 0.1) & (tr["t"] <= t1 + 0.1)
        f["front"][i] = float(np.clip(1 - hid / max(m.sum(), 1), 0, 1))
    f["dom"] = f["size"] * f["act"] * f["front"]              # the biggest and the busiest at once
    return f


def choose(trs, t0, length, cam=None, weights=None, strip=None):
    """The shot's subject: (index, margin over the next, features) -- the best-scoring track (WEIGHTS over
    features), or the person the body tracker's strip opens on unless the best beats them by KEEP; -1 with nobody.
    One person for the whole shot."""
    f = features(trs, t0, length, cam, strip)
    if not trs:
        return -1, 0.0, f
    w = weights or WEIGHTS
    score = sum(w[k] * f[k] for k in w)
    score = np.where(f["seen"] > 0, score, -1.0)
    order = np.argsort(-score)
    total = sum(abs(v) for v in w.values())
    margin = float(score[order[0]] - score[order[1]]) / total if len(order) > 1 else 1.0
    k = int(order[0])
    sharp_ok = (f["seen"] >= 0.5) & (f["size"] >= 0.5)       # the camera's focus says who the shot is of: never a
    if sharp_ok.any():                                        # nearer, softer person over the one it focused on
        top = float(f["sharp"][sharp_ok].max())               # (Isaac: "a guy in foreground not in focus. not
        if f["sharp"][k] < FOCUS * top:                       #  locked in on the guy who's in focus")
            pool = np.where(sharp_ok & (f["sharp"] >= FOCUS * top) & (f["size"] * 1.15 <= f["size"][k])
                            & (f["act"] + 0.2 >= f["act"][k]), score, -9.0)   # (a dancer in fog stays the subject)
            if pool.max() > -9.0:                             # (only a nearer, softer one gives way: the foreground
                k = int(np.argmax(pool))                      #  out of the focus plane, the one behind it in focus)
    if strip is not None and f["opener"].max() >= 0.5:        # the shot keeps the person it opens on, unless
        op = int(np.argmax(f["opener"] + 0.5 * f["held"] + 0.5 * f["mid"] + 1e-3 * score))   # someone is clearly
        # more its subject; of two in the strip as it opens, the one at its middle (SSMC's opening: the instructor,
        # not the dancer behind him at the strip's edge)
        behind = (f["size"][op] < BEHIND * f["size"][k]                 # never someone further back and softer
                  and f["sharp"][op] < SOFTER * f["sharp"][k])         # than the best (Isaac: "she's on a 3d
        if op != k and f["opener"][k] < 0.5 and score[k] - score[op] < KEEP and not behind:   # plane further
            k = op                                                     # behind him, not even in full focus")
    return k, margin, f


def _at(tr, times, key):
    """A track's values (head, torso, box ...) at these times: interpolated between its pose frames, held past its
    ends and through gaps."""
    v = np.asarray(tr[key], float)
    if v.ndim == 1:
        return np.interp(times, tr["t"], v)
    return np.stack([np.interp(times, tr["t"], v[:, k]) for k in range(v.shape[1])], 1)


def lying(tr):
    """Is this person lying down or along the floor (a worm, a floor move): their spine (shoulders to hips) nearer
    flat than upright; with no torso keypoints, a box wider than tall (in pixels) that the frame doesn't cut top
    and bottom (someone right at the lens fills the frame's width standing up)."""
    lean = tr["lean"][~np.isnan(tr["lean"])]
    if len(lean) >= max(2, 0.4 * len(tr["t"])):
        return float(np.median(lean)) >= LEAN
    b = np.clip(tr["box"], 0, 1)
    whole = (b[:, 1] > 0.03) | (b[:, 3] < 0.97)
    wide = (b[:, 2] - b[:, 0]) / np.maximum((b[:, 3] - b[:, 1]) * ASPECT, 1e-3) >= LYING
    return bool((whole & wide).mean() >= 0.6)


def hands(tr):
    """Hands at work with no head in the picture (a DJ's hands on the mixer, a player's on the keys): where the
    fingers meet the gear, (n, 2) per pose frame -- each wrist carried on past the hand along the forearm, the two
    averaged -- or None when the head shows (no face, no sure head, the box cut by the frame's top) or the wrists
    don't."""
    top = (tr["box"][:, 1] <= 0.02) & ~tr["face"]
    arms = tr["arms"]                                         # (n, 4, 3): elbows l r, wrists l r
    seen = (arms[:, 2:, 2] > KP) & (arms[:, :2, 2] > 0.2)     # (n, 2): a wrist and its elbow
    headless = top & ((tr["head"][:, 1] < 0.04) | ~tr["sure"])
    if headless.mean() < 0.7 or seen.any(1).mean() < 0.7 or tr["face"].any():
        return None
    tips = arms[:, 2:, :2] + FINGERS * (arms[:, 2:, :2] - arms[:, :2, :2])
    w = seen.astype(float)
    pt = (tips * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-6)[:, None]
    ok = seen.any(1)
    if not ok.all():                                          # a frame with no wrist: where they were
        j = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))
        j[:np.argmax(ok)] = np.argmax(ok)
        pt = pt[j]
    return np.clip(pt, 0, 1)


def companion(trs, k, f, cw):
    """Who shares the strip with the subject: someone their size (PEER) right with them -- heads within TOGETHER of
    the strip's width for most of the frames both are seen, seen for most of the subject's time, not half hidden --
    the nearest such. -1: nobody (the subject alone)."""
    me = trs[k]
    best, bd = -1, 9.0
    for j, o in enumerate(trs):
        if j == k or f["size"][j] < PEER * f["size"][k] or f["seen"][j] < 0.6 * f["seen"][k] or f["front"][j] < 0.5:
            continue
        if f["size"][j] < BEHIND * f["size"][k] and f["sharp"][j] < SOFTER * f["sharp"][k]:   # smaller and softer:
            continue                                         # further back, out of the subject's focus plane (Isaac:
            #                                                  "on a 3d plane further behind him, not in full focus")
        lo, hi = max(me["t"][0], o["t"][0]), min(me["t"][-1], o["t"][-1])
        if hi - lo < 0.2 and len(me["t"]) > 2:
            continue
        ts = me["t"][(me["t"] >= lo) & (me["t"] <= hi)]
        if not len(ts):
            continue
        gap = np.abs(_at(me, ts, "head")[:, 0] - _at(o, ts, "head")[:, 0])
        if (gap <= TOGETHER * cw).mean() >= 0.7 and float(np.median(gap)) < bd:
            best, bd = j, float(np.median(gap))
    return best


def aim(trs, k, f, t0, length, cw, ch, solo=False):
    """Where the strip's centre goes, at RATE through the shot: (times from the shot's start, x, y, what) -- what is
    "person", "pair", "floor" or "hands" (see the module's notes). solo: this person alone, whoever is with them
    (the editor's own choice of person)."""
    me = trs[k]
    ts = np.arange(0, length + 1e-6, RATE)
    at = t0 + ts
    head, torso, box, hs = _at(me, at, "head"), _at(me, at, "torso"), _at(me, at, "box"), _at(me, at, "hs")
    hw = 0.5 * hs * ASPECT * 0.8                               # half a head's width, in frame widths
    tips = hands(me)
    if tips is not None:
        x = np.interp(at, me["t"], tips[:, 0])
        y = np.interp(at, me["t"], tips[:, 1])
        return ts, x, y, "hands"
    if lying(me):                                             # the move reads from the torso: the body's middle,
        x = 0.5 * torso[:, 0] + 0.5 * (box[:, 0] + box[:, 2]) / 2     # the head still inside the strip
        x = np.clip(x, head[:, 0] - (cw / 2 - hw - 0.01), head[:, 0] + (cw / 2 - hw - 0.01))
        return ts, x, torso[:, 1], "floor"
    what = "person"
    x = 0.5 * head[:, 0] + 0.5 * torso[:, 0]
    x = np.clip(x, head[:, 0] - HEAD_IN * cw, head[:, 0] + HEAD_IN * cw)
    j = -1 if solo else companion(trs, k, f, cw)
    if j >= 0:                                                # both in the strip, weighted to the nearer one
        o = trs[j]
        oh = _at(o, at, "head")
        wm, wo = f["size"][k] ** 2, f["size"][j] ** 2
        mid = (wm * head[:, 0] + wo * oh[:, 0]) / (wm + wo)
        lo = np.maximum(head[:, 0], oh[:, 0]) - (cw / 2 - 0.02)   # both heads' middles inside the strip, the
        hi = np.minimum(head[:, 0], oh[:, 0]) + (cw / 2 - 0.02)   # subject's in its middle part (HEAD_IN): the
        fits = lo <= hi                                           # companion gets what that leaves
        live = (at >= o["t"][0] - 0.3) & (at <= o["t"][-1] + 0.3)
        if (fits & live).mean() >= 0.7:
            both = np.clip(np.clip(mid, lo, hi), head[:, 0] - HEAD_IN * cw, head[:, 0] + HEAD_IN * cw)
            x = np.where(fits & live, both, x)
            hy = np.minimum(head[:, 1], oh[:, 1])
            return ts, x, hy + (0.5 - 0.32) * ch, "pair"
    # alone: slide (the head staying inside its part of the strip) off anyone whose head the strip's edge would cut
    others = [o for i, o in enumerate(trs) if i != k and f["size"][i] >= PEER * f["size"][k] and f["seen"][i] >= 0.3]
    if others:
        best, bc = 0.0, None
        for off in np.linspace(-0.5, 0.5, 11) * HEAD_IN * cw:
            cost = 0.6 * abs(off) / cw
            for o in others:
                live = (at >= o["t"][0] - 0.1) & (at <= o["t"][-1] + 0.1)
                oh, ow = _at(o, at, "head")[:, 0], 0.5 * _at(o, at, "hs") * ASPECT * 0.8
                edge = np.minimum(np.abs(oh - (x + off - cw / 2)), np.abs(oh - (x + off + cw / 2)))
                cost += float((live & (edge < ow)).mean())
            if bc is None or cost < bc - 1e-9:
                best, bc = off, cost
        x = x + best
    return ts, x, head[:, 1] + (0.5 - 0.32) * ch, what


def glide(ts, v, lo, hi):
    """A strip centre as a path: smoothed over SMOOTH seconds (the detector's jitter; the subject's own move is
    kept, with no lag -- the smoothing is centred) and kept inside the frame."""
    n = max(1, int(round(3 * SMOOTH / RATE)))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * RATE / SMOOTH) ** 2)
    k /= k.sum()
    return np.clip(np.convolve(np.pad(v, n, mode="edge"), k, mode="valid"), lo, hi)


WHY = dict(size="nearest the lens", act="the one dancing", bounce="jumping", face="face to the camera",
           sharp="in focus", seen="in shot throughout", central="in the middle", centring="the camera follows them",
           front="in front", opener="the shot opens on them", opens="there from the start", held="the tracker's own", alone="right up front", dom="the biggest and busiest")


def strip(d, yf, shake, t0, length, cw, ch, body=None):
    """The portrait strip of a shot, on its subject: dict(x, y on t (seconds from the shot's start, every RATE), on:
    "person" | "pair" | "floor" | "hands", why: a few words on why this person, lead: their track, mate: the
    companion's or None, cut: the share of the shot their head is cut off by the frame's own top, halved: the share
    someone their size has their head on the strip's edge, margin), or None
    when there is nobody to follow. body: the body tracker's strip (times, centres) -- who the shot opens on."""
    trs = tracks(d, yf, t0 - 0.3, t0 + length + 0.3)
    if not trs:
        return None
    cam = camera(shake, t0, t0 + length)
    k, margin, f = choose(trs, t0, length, cam, strip=None if body is None else (body[0], body[1], cw))
    if k < 0 or f["seen"][k] < 0.3:
        return None
    ts, x, y, what = aim(trs, k, f, t0, length, cw, ch)
    x, y = glide(ts, x, cw / 2, 1 - cw / 2), glide(ts, y, ch / 2, 1 - ch / 2)
    me = trs[k]
    m = (me["t"] >= t0) & (me["t"] <= t0 + length)
    cut = float(((me["box"][m, 1] <= 0.01) & (me["head"][m, 1] - 0.5 * me["hs"][m] < 0.0)).mean()) if m.any() else 0.0
    if what == "hands":                                       # (hands at work: there is no head to cut)
        cut = 0.0
    halved = 0.0                                              # someone their size with their head on the strip's edge
    at = t0 + ts
    for i, o in enumerate(trs):
        if i == k or f["size"][i] < PEER * f["size"][k] or f["seen"][i] < 0.3:
            continue
        live = (at >= o["t"][0] - 0.1) & (at <= o["t"][-1] + 0.1)
        oh, ow = _at(o, at, "head")[:, 0], 0.5 * _at(o, at, "hs") * ASPECT * 0.8
        edge = np.minimum(np.abs(oh - (x - cw / 2)), np.abs(oh - (x + cw / 2)))
        halved = max(halved, float((live & (edge < 0.6 * ow)).mean()))
    rest = [i for i in range(len(trs)) if i != k and f["seen"][i] > 0]
    if rest:
        o = max(rest, key=lambda i: sum(WEIGHTS[n] * f[n][i] for n in WEIGHTS))
        lead_by = sorted(WEIGHTS, key=lambda n: -WEIGHTS[n] * (f[n][k] - f[n][o]))[:2]
        why = ", ".join(WHY[n] for n in lead_by if f[n][k] > f[n][o])
    else:
        why = "the only one in shot"
    return dict(t=ts, x=x, y=y, on=what, why=why or "by a nose", lead=me, margin=margin, cut=cut, halved=halved,
                tracks=trs, k=k)


CLICK = 0.12             # clicked(): a head this near the click (of the frame's width) is the one clicked ...
CLICK_WHEN = 0.75        # ... seen within this many seconds of it


def clicked(trs, kt, kx, ky):
    """The person the editor clicked: the track whose head is nearest (kx, ky) at time kt (within CLICK), else the
    smallest box the click is inside (a click on a body, a costume, a back). -1: nobody there."""
    best, bd = -1, CLICK
    inside, area = -1, 9.0
    for i, tr in enumerate(trs):
        j = int(np.abs(tr["t"] - kt).argmin())
        if abs(tr["t"][j] - kt) > CLICK_WHEN:
            continue
        dd = float(np.hypot(tr["head"][j, 0] - kx, (tr["head"][j, 1] - ky) * ASPECT))
        if dd < bd:
            best, bd = i, dd
        x0, y0, x1, y1 = tr["box"][j]
        if x0 <= kx <= x1 and y0 <= ky <= y1 and (x1 - x0) * (y1 - y0) < area:
            inside, area = i, (x1 - x0) * (y1 - y0)
    return best if best >= 0 else inside


def follow(d, yf, t0, length, cw, ch, kt, kx, ky):
    """The strip on the person the editor clicked at (kt, kx, ky), for the whole shot: dict(t, x, y, on) as
    strip(), or None when no tracked person is at the click (the caller holds the point)."""
    trs = tracks(d, yf, min(t0, kt) - 0.5, max(t0 + length, kt) + 0.5)
    k = clicked(trs, kt, kx, ky)
    if k < 0:
        return None
    f = features(trs, t0, length)
    ts, x, y, what = aim(trs, k, f, t0, length, cw, ch, solo=True)
    me = trs[k]
    m = (me["t"] >= t0) & (me["t"] <= t0 + length)
    return dict(t=ts, x=glide(ts, x, cw / 2, 1 - cw / 2), y=glide(ts, y, ch / 2, 1 - ch / 2), on=what,
                face=float(me["face"][m].mean()) if m.any() else 0.0,
                seen=min(1.0, m.sum() / max(1.0, length / 0.2)))
