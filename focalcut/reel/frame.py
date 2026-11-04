"""The subject of a shot and how well it can be framed.

Subject: from the subject pass's track (SAM 3 masks scored by saliency and depth, ~1 a second) when it exists: the
union of the subject's masks and whether they touch the frame edges. Without it, the fallback is the sharpest
person box (the focus pass, else the largest) plus everyone close to it who is nearly as sharp. Either way, someone
well out of focus -- close to the lens in front of the shot -- is never the subject when the camera's focus is
clearly on someone else (in_focus).
Framing: 4K sources in a 1080p reel can be cropped up to ZMAX x without losing resolution. The crop follows the
subject; a shot frames well when the subject ends up centred, a good size, and not cut off by the camera's own
frame edge.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

ZMAX = 1.6                                        # 3840 / 1.6 = 2400 px across: still above 1080p
FILL = 0.62                                       # target subject height, fraction of the output frame
SHARP_REL = 0.55                                  # a person counts as in focus at this fraction of the sharpest
# the camera's focus (the focus pass): someone under BLUR of the sharpest person in frame and NEAR x their size or
# more -- closer to the lens, in front of the one the camera focused on (a dancer as near as the sharp one and soft
# is motion blur) -- is out of focus and never the subject; the sharp one takes over when at least MIN_H of the frame
# tall, in at least SHARE of the shot's frames with focus data (one soft frame is not the camera's choice).
# reel.json "focus_blur", "focus_near", "focus_min_h", "focus_share" (configure)
FOCUS = dict(blur=0.4, near=1.5, min_h=0.5, share=0.4, moving=1.0)
# a group too wide for a portrait strip: one dancer whole -- the one nearest the frame's centre, or one right up front
# -- never the floor between people (concepts §19); reel.json "group": "pan" asks for a slow pan through its middle
# instead (PAN_SPEED frame widths a second at most, from PAN_MIN s) (configure)
GROUP = dict(mode="centre")
PAN_SPEED, PAN_MIN = 0.08, 1.5
PAN, CENTRE = "pan", "centre"                     # follow()'s third value for a group shot
FOLLOW_SHARP = 1.5                                # following one dancer: the weight of being the one in focus
FRONT, FRONT_W = (1.8, 3.5), 6.0                  # in a group, a dancer this many times a member's usual size
                                                  # (ramping) is up front, and taken over whoever is central


def configure(cfg):
    """Set the focus rule's thresholds from a project's config (reel.json "focus_blur" / "focus_min_h" /
    "focus_share") and how a group is framed ("group"); a profile can set them."""
    for k in FOCUS:
        FOCUS[k] = float(cfg.get(f"focus_{k}", _FOCUS0[k]))
    GROUP["mode"] = cfg.get("group", CENTRE)


_FOCUS0 = dict(FOCUS)


def load(path):
    """scores/<name>.npz plus its _focus and _subject companions when they exist."""
    d = dict(np.load(path))
    stem = path[:-4]
    f = stem + "_focus.npz"
    d["focus"] = np.load(f)["focus"] if os.path.exists(f) else np.full(d["boxes"].shape[:2], np.nan, np.float32)
    w, h = d["size"]
    d["aw"], d["ah"] = (640, 640 * h / w) if w >= h else (640 * w / h, 640)
    f = stem + "_subject.npz"
    if os.path.exists(f):
        s = np.load(f)
        d["sub_t"], d["sub_box"], d["sub_touch"] = s["t"], s["box"], s["touch"]
    return d


def track(d, times):
    """Subject union box and edge contact at these times: the subject track (boxes interpolated between keyframes,
    contact from the nearest), or the focus fallback. Returns union (F, 4), touch (F, 4) or None, source name."""
    if "sub_t" in d and len(d["sub_t"]) > 1:
        st, sb = d["sub_t"], d["sub_box"]
        ok = ~np.isnan(sb[:, 0])
        union = np.full((len(times), 4), np.nan)
        if ok.sum() >= 2:
            for k in range(4):
                union[:, k] = np.interp(times, st[ok], sb[ok, k])
        near = np.abs(st[None, :] - np.asarray(times)[:, None])
        j = near.argmin(1)
        far = (near[np.arange(len(times)), j] > 1.2) | ~ok[j]    # no subject seen near this time
        union[far] = np.nan
        union, touch = in_focus(d, times, refine(d, times, union), d["sub_touch"][j])
        return union, touch, "sam"
    pt = d["pose_t"]
    idx = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    union, _, _ = subject(d["boxes"][idx], d["focus"][idx], d["aw"], d["ah"])
    return in_focus(d, times, union, None)[0], None, "focus"


def blurry(d, idx):
    """(F, P): who is out of focus in each pose frame: under FOCUS["blur"] of the sharpest person there and at least
    FOCUS["near"] x their area (in front of them, nearer the lens), and dancing less than the sharpest person (a
    close dancer softened by their own motion is the shot, not a bystander in front of it). Needs two people with
    focus data."""
    b, foc = d["boxes"][idx], d["focus"][idx]
    act = activity(d)[idx]
    have = ~np.isnan(b[..., 0]) & ~np.isnan(foc)
    out = np.zeros(have.shape, bool)
    area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    for f in np.nonzero(have.sum(1) >= 2)[0]:
        k = int(np.nanargmax(np.where(have[f], foc[f], np.nan)))
        out[f] = have[f] & (foc[f] < FOCUS["blur"] * foc[f, k]) & (area[f] >= FOCUS["near"] * area[f, k]) & \
            (act[f] < FOCUS["moving"] * max(act[f, k], 1e-6))
    return out


def in_focus(d, times, union, touch):
    """The subject held to the camera's focus. Where the focus pass shows a person well out of focus (blurry) inside
    the subject -- typically someone close to the lens in front of the shot -- they leave it: the subject is the
    members still in focus, or, with none, the sharpest person in frame, and those beside them nearly as sharp (either
    way, people at least MIN_H tall: with nobody that size in focus, the subject stays). Only when that holds for SHARE of the frames with focus data; frames without data between two
    changed ones follow them. Returns (union, touch) -- touch recomputed where the union changed."""
    if "pose_t" not in d or not len(d["pose_t"]) or np.isnan(d["focus"]).all():
        return union, touch
    pt = d["pose_t"]
    times = np.asarray(times, float)
    idx = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    close = np.abs(pt[idx] - times) <= 0.15
    aw, ah = d["aw"], d["ah"]
    boxes, foc = d["boxes"][idx], d["focus"][idx]
    nb = boxes[..., :4] / [aw, ah, aw, ah]
    w, h = union[:, 2] - union[:, 0], union[:, 3] - union[:, 1]
    grown = np.c_[union[:, 0] - 0.15 * w, union[:, 1] - 0.1 * h, union[:, 2] + 0.15 * w, union[:, 3] + 0.1 * h]
    m = members(boxes, grown, aw, ah) & close[:, None]
    soft = blurry(d, idx) & close[:, None]
    have = ~np.isnan(nb[..., 0]) & ~np.isnan(foc) & close[:, None]
    seen = have.sum(1) >= 2                                   # frames the focus pass can judge
    out, hit = union.copy(), np.zeros(len(times), bool)
    for f in np.nonzero(seen & soft.any(1))[0]:
        if not (m[f] & soft[f]).any():
            continue                                          # nobody out of focus in the subject
        top = np.nanmax(np.where(have[f], foc[f], np.nan))
        rel = np.where(have[f], foc[f] / max(top, 1e-6), 0.0)
        tall = (nb[f, :, 3] - nb[f, :, 1]) >= FOCUS["min_h"]
        keep = m[f] & have[f] & ~soft[f] & tall              # (never shrunk to a small figure far behind)
        if not keep.any():                                    # the subject was all blur: the sharp person instead
            sharp = have[f] & tall & (rel >= SHARP_REL)
            if not sharp.any():
                continue
            k0 = int(np.argmax(np.where(sharp, rel, -1)))
            keep = np.zeros_like(sharp); keep[k0] = True
            for k in np.nonzero(sharp)[0]:                    # with whoever stands beside them, nearly as sharp
                if _gap(nb[f, k], nb[f, k0]) < 0.6 * (nb[f, k0, 2] - nb[f, k0, 0]):
                    keep[k] = True
        b = nb[f, keep]
        out[f] = [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
        hit[f] = True
    if hit.sum() < FOCUS["share"] * max(seen.sum(), 1):
        return union, touch
    ev = np.nonzero(seen)[0]                                  # frames without focus data: between two changed ones,
    for f in np.nonzero(~seen)[0]:                            # follow them
        a, b_ = ev[ev < f], ev[ev > f]
        if len(a) and len(b_) and hit[a[-1]] and hit[b_[0]] and times[b_[0]] - times[a[-1]] <= 1.5:
            u = (times[f] - times[a[-1]]) / max(times[b_[0]] - times[a[-1]], 1e-6)
            out[f] = (1 - u) * out[a[-1]] + u * out[b_[0]]
            hit[f] = True
    if touch is not None:
        touch = np.array(touch, copy=True)
        touch[hit] = np.c_[out[hit, 0] < 0.01, out[hit, 1] < 0.01, out[hit, 2] > 0.99, out[hit, 3] > 0.99]
    return out, touch


def refine(d, times, union):
    """SAM 3 keyframes are ~1 s apart, and a dancer can cross a narrow portrait strip in less: between them, follow
    the subject's own pose boxes (5 fps). The keyframe track says who the subject is (the boxes mostly inside a
    slightly grown union); the pose boxes say where they are now. The union keeps its size, re-centred on them."""
    if "pose_t" not in d or not len(d["pose_t"]):
        return union
    pt = d["pose_t"]
    idx = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    close = np.abs(pt[idx] - np.asarray(times)) <= 0.15
    w, h = union[:, 2] - union[:, 0], union[:, 3] - union[:, 1]
    grown = np.c_[union[:, 0] - 0.15 * w, union[:, 1] - 0.1 * h, union[:, 2] + 0.15 * w, union[:, 3] + 0.1 * h]
    boxes = d["boxes"][idx]
    m = members(boxes, grown, d["aw"], d["ah"]) & close[:, None]
    out = union.copy()
    for f in np.nonzero(m.any(1))[0]:
        b = boxes[f, m[f], :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
        cx, cy = (b[:, 0].min() + b[:, 2].max()) / 2, (b[:, 1].min() + b[:, 3].max()) / 2
        ux, uy = (union[f, 0] + union[f, 2]) / 2, (union[f, 1] + union[f, 3]) / 2
        out[f] = union[f] + [cx - ux, cy - uy, cx - ux, cy - uy]
    return out


def members(boxes, union, aw, ah):
    """Which pose boxes belong to the subject: mostly inside its union."""
    F, P = boxes.shape[:2]
    m = np.zeros((F, P), bool)
    for f in range(F):
        if np.isnan(union[f, 0]):
            continue
        ux0, uy0, ux1, uy1 = union[f] * [aw, ah, aw, ah]
        for k in range(P):
            b = boxes[f, k]
            if np.isnan(b[0]):
                continue
            iw, ih = max(0, min(b[2], ux1) - max(b[0], ux0)), max(0, min(b[3], uy1) - max(b[1], uy0))
            m[f, k] = iw * ih >= 0.6 * (b[2] - b[0]) * (b[3] - b[1])
    return m


def subject(boxes, focus, aw, ah):
    """Per pose frame: subject union box as fractions (x0, y0, x1, y1; NaN if nobody), membership per box, and the
    subject's sharpness."""
    F, P = boxes.shape[:2]
    union = np.full((F, 4), np.nan)
    member = np.zeros((F, P), bool)
    sharp = np.full(F, np.nan)
    for f in range(F):
        ok = ~np.isnan(boxes[f, :, 0])
        if not ok.any():
            continue
        b = boxes[f, :, :4]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        fo = np.where(ok, focus[f], np.nan)
        if np.isnan(fo[ok]).all():
            score = np.where(ok, area, -1)
            near = ok & (area >= 0.45 * area[ok].max())
        else:
            fn = np.nan_to_num(fo / np.nanmax(fo))
            score = np.where(ok, fn * np.sqrt(area / area[ok].max()), -1)
            near = ok & (fn >= SHARP_REL)
            sharp[f] = np.nanmax(fo)
        best = int(np.argmax(score))
        grp = {best}
        changed = True
        while changed:                              # grow the group through people standing close to it
            changed = False
            for k in np.nonzero(near)[0]:
                if k in grp:
                    continue
                if any(_gap(b[k], b[j]) < 0.6 * (b[j, 2] - b[j, 0]) for j in grp):
                    grp.add(int(k)); changed = True
        g = sorted(grp)
        member[f, g] = True
        union[f] = [b[g, 0].min() / aw, b[g, 1].min() / ah, b[g, 2].max() / aw, b[g, 3].max() / ah]
    return union, member, sharp


def faces(kp):
    """Which pose people show their face to the camera, from the pose alone: nose and both eyes confidently seen.
    ViTPose often puts confident eyes on the back of a head, so this is only the fallback for face_seen."""
    return (kp[..., 0, 2] > 0.6) & (kp[..., 1, 2] > 0.6) & (kp[..., 2, 2] > 0.6) & (kp[..., 1, 0] > kp[..., 2, 0])


FACE_SCORE, FACE_H = 0.65, 0.03                   # a YuNet face that counts: detector score, height (of the frame)
FACE_OWN = 0.3                                    # ... and a person's own: at least this x their build (frame heights)


def face_seen(d, times, boxes, kp):
    """Which pose people (F, P) show their face to the camera. With a face scan of the unit (d["yf"]: the people
    step's YuNet detections, which a back of a head never sets off), a person shows their face when a confident face
    sits in the head of their box within 0.3 s; otherwise from the pose keypoints (faces)."""
    if "yf" not in d:
        return faces(kp)
    y = d["yf"]
    keep = (y["score"] >= FACE_SCORE) & (y["h"] >= FACE_H)
    yt, cx, cy, fh = y["t"][keep], y["cx"][keep], y["cy"][keep], y["h"][keep]
    nb = boxes[..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    aspect = d["ah"] / d["aw"]
    out = np.zeros(boxes.shape[:2], bool)
    for f, t in enumerate(times):
        near = np.abs(yt - t) <= 0.3
        if not near.any():
            continue
        x0, y0, x1, y1 = (nb[f, :, k][:, None] for k in range(4))
        head = (cx[near] >= x0) & (cx[near] <= x1) & (cy[near] >= y0 - 0.02) & (cy[near] <= y0 + 0.4 * (y1 - y0))
        # and theirs: a face their build's size, not a smaller one further back showing past their head (their
        # own face points aren't asked for: the pose is 5 a second and a spinning head moves between scans)
        build = np.maximum.reduce([0.45 * (x1 - x0) / aspect, 0.12 * (y1 - y0), np.zeros_like(x0)])
        sized = fh[near][None, :] >= FACE_OWN * build
        out[f] = (np.nan_to_num(head).astype(bool) & sized).any(1)
    return out


LIMBS = [0, 7, 8, 9, 10]                          # COCO keypoints: nose, elbows, wrists
ACTIVE = 0.35                                     # in the frame: moving at least this share of the busiest dancer
GROUP_W = 0.6                                     # a subject wider than this (of the frame) is a group routine


def activity(d):
    """How much each pose person dances (F, P), over the whole clip: how far their head and arms move about their
    shoulders since the previous pose frame (the same person: the nearest box) -- the two busiest of those points,
    so the body's own travel and stride drop out, so walking across the floor scores low and styling, turns and arm
    work score high -- in body heights a second, smoothed over ~0.6 s along the person. Cached in d["act"]."""
    if "act" in d:
        return d["act"]
    bx, kp, pt = d["boxes"], d["kp"], d["pose_t"]
    F, P = bx.shape[:2]
    cx, h = (bx[..., 0] + bx[..., 2]) / 2, np.maximum(bx[..., 3] - bx[..., 1], 1.0)
    prev = np.full((F, P), -1)
    raw = np.zeros((F, P))
    for f in range(1, F):
        for k in range(P):
            if np.isnan(cx[f, k]):
                continue
            dist = np.abs(cx[f - 1] - cx[f, k])
            j = int(np.nanargmin(dist)) if not np.isnan(dist).all() else -1
            if j < 0 or dist[j] > 0.5 * (bx[f, k, 2] - bx[f, k, 0]):
                continue
            prev[f, k] = j
            a, b = kp[f, k], kp[f - 1, j]
            if min(a[5, 2], a[6, 2], b[5, 2], b[6, 2]) < 0.3:
                continue                                       # no shoulders to measure against
            ok = (a[LIMBS, 2] > 0.3) & (b[LIMBS, 2] > 0.3)
            if ok.sum() >= 2:
                here = a[LIMBS, :2] - (a[5, :2] + a[6, :2]) / 2   # about the shoulders: the stride and the
                then = b[LIMBS, :2] - (b[5, :2] + b[6, :2]) / 2   # travel drop out
                mv = np.linalg.norm(here[ok] - then[ok], axis=1)
                raw[f, k] = np.mean(np.sort(mv)[-2:]) / h[f, k] / max(pt[f] - pt[f - 1], 1e-3)
    act = raw.copy()
    for _ in range(2):                                        # smooth back and forth along each person's chain
        for f in range(1, F):
            for k in range(P):
                if prev[f, k] >= 0:
                    act[f, k] = 0.6 * act[f, k] + 0.4 * act[f - 1, prev[f, k]]
        for f in range(F - 1, 0, -1):
            for k in range(P):
                if prev[f, k] >= 0:
                    j = prev[f, k]
                    act[f - 1, j] = 0.6 * act[f - 1, j] + 0.4 * act[f, k]
    d["act"] = np.where(np.isnan(cx), 0.0, act)
    return d["act"]


def follow(d, times, union, cw, split=0.8):
    """What a portrait crop (cw: its width at zoom 1, as a fraction of the frame) should hold, frame by frame.
    The crop is about the action, not whoever stands together: the people moving most (activity). A couple dancing
    shares the frame (couple: both dancers, the follow featured); a group is cut down to the people in the move.
    Otherwise, in a 9:16 strip out of a 16:9 frame, when the subject group is wider than the strip for a good part of the shot, centring on the group
    frames the empty floor between them: follow one dancer instead -- the one who shows their face most, then the
    sharpest and biggest -- tracked box to box. A group routine (GROUP) pans slowly across the group's middle or
    takes whoever dances nearest the frame's centre. Returns (boxes (F, 4), face (F,) bool, own boxes? -- the group's
    edge contact no longer applies; PAN or CENTRE for a group, which is truthy)."""
    with warnings.catch_warnings():                         # (all-NaN stretches: a dancer lost for a moment)
        warnings.simplefilter("ignore", RuntimeWarning)
        return _follow(d, times, union, cw, split)


def _follow(d, times, union, cw, split):
    pt = d["pose_t"]
    idx = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    close = np.abs(pt[idx] - np.asarray(times)) <= 0.15
    w, h = union[:, 2] - union[:, 0], union[:, 3] - union[:, 1]
    grown = np.c_[union[:, 0] - 0.15 * w, union[:, 1] - 0.1 * h, union[:, 2] + 0.15 * w, union[:, 3] + 0.1 * h]
    boxes, kp = d["boxes"][idx], d["kp"][idx]
    foc = d["focus"][idx] if "focus" in d else np.zeros(boxes.shape[:2])
    m = members(boxes, grown, d["aw"], d["ah"]) & close[:, None]
    fv = face_seen(d, times, boxes, kp) & m
    act = activity(d)[idx]
    hgt = np.nan_to_num(boxes[..., 3] - boxes[..., 1])
    act = act * np.sqrt(hgt / np.maximum(hgt.max(1, keepdims=True), 1e-6))   # small, far people jitter most
    pair = couple(d, times, boxes, m & close[:, None], close, cw, face_seen(d, times, boxes, kp), act, union)
    if pair is not None:
        return pair[0], pair[1], True
    # the group, but only the people in the action: a bystander inside the subject's box doesn't widen the crop
    nb0 = boxes[..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    mm = m & close[:, None]
    busy = np.where(mm, act, -1.0)
    top = busy.max(1, keepdims=True)
    keep = mm & (busy >= ACTIVE * np.maximum(top, 1e-6))
    # (a routine danced by a whole group is the group's action -- and the pose pass sees only a few of them)
    group = (keep.sum(1) >= 3) | (np.nan_to_num(union[:, 2] - union[:, 0]) > GROUP_W)
    trim = ~np.isnan(union[:, 0]) & keep.any(1) & (top[:, 0] > 0.05) & ~group
    if trim.any():
        union = union.copy()
        for f in np.nonzero(trim)[0]:
            b = nb0[f, keep[f]]
            union[f] = [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
        w = union[:, 2] - union[:, 0]
    wide = (np.nan_to_num(w) > split * cw) & (cw < 0.5)       # (a strip a third of the frame wide)
    if wide.mean() < 0.3:
        # the face that counts is the main person's (the one in the move, else the biggest): a partner's face over
        # a back is still a shot of a back
        area = np.where(m, (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]), -1)
        area = np.nan_to_num(area, nan=-1)
        main = np.argmax(np.where(m, act + 1e-6 * area, -1), 1)
        return union, fv[np.arange(len(idx)), main] & m.any(1), False
    nb = boxes[..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    crowd = (m.sum(1) >= 3).mean() >= 0.5                     # (a couple far apart is not a group)
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    if crowd and GROUP["mode"] == PAN and span >= PAN_MIN:
        return _pan(union, nb, m, fv, times, cw)
    f0 = int(np.argmax(np.where(wide, m.sum(1), -1)))
    pa = (nb[..., 2] - nb[..., 0]) * (nb[..., 3] - nb[..., 1])
    typical = float(np.nanmedian(np.where(m, pa, np.nan))) if m.any() else 0.0   # a member's usual size
    best, best_sc = None, -1
    for k0 in np.nonzero(m[f0])[0]:
        seq, face = np.full((len(idx), 4), np.nan), np.zeros(len(idx), bool)
        slot = np.full(len(idx), -1)                          # which pose slot is this dancer, frame by frame
        for order in (range(f0, len(idx)), range(f0, -1, -1)):
            prev = nb[f0, k0]
            for f in order:
                c = list(np.nonzero(m[f])[0])
                if f == f0:
                    k = k0
                elif c:
                    dist = [abs((nb[f, k, 0] + nb[f, k, 2]) - (prev[0] + prev[2])) / 2 for k in c]
                    k = c[int(np.argmin(dist))] if min(dist) < 0.12 else None
                else:
                    k = None
                if k is None:
                    continue
                prev = nb[f, k]; seq[f] = prev; face[f] = fv[f, k]; slot[f] = k
        ok = ~np.isnan(seq[:, 0])
        if not ok.any():
            continue
        area = np.nanmean((seq[ok, 2] - seq[ok, 0]) * (seq[ok, 3] - seq[ok, 1]))
        sharp = np.nanmean([foc[f, slot[f]] for f in np.nonzero(ok)[0]])
        moving = np.mean([act[f, slot[f]] for f in np.nonzero(ok)[0]]) / (act[m].max() + 1e-6) if m.any() else 0
        sharp_rel = np.nan_to_num(sharp / (np.nanmax(foc) + 1e-6))
        sc = 2.0 * face.mean() + 1.5 * moving + ok.mean() + np.sqrt(area) + FOLLOW_SHARP * sharp_rel
        if crowd:                                             # a group: whoever is in the middle of it -- unless
            cx = np.nanmean((seq[ok, 0] + seq[ok, 2]) / 2)    # one dancer is right up front, much bigger than the
            sc = 4.0 * (1 - 2 * abs(cx - 0.5)) + ok.mean() + 0.5 * face.mean()   # rest: they are the shot (in
        sc += FRONT_W * np.clip((area / max(typical, 1e-9) - FRONT[0]) / (FRONT[1] - FRONT[0]), 0, 1)   # a couple
        #                                                     too far apart, a class floor, too)
        if sc > best_sc:
            best, best_sc = (seq, face), sc
    if best is None:
        return union, fv.any(1), False
    seq, face = best
    gap = np.isnan(seq[:, 0])                                 # lost them for a moment: hold the last place they were
    if gap.any() and (~gap).any():
        j = np.maximum.accumulate(np.where(~gap, np.arange(len(seq)), 0))
        first = np.argmax(~gap)
        j[:first] = first
        seq = seq[j]
    return seq, face, CENTRE if crowd else True


def _pan(union, nb, m, fv, times, cw):
    """A strip-wide box sliding slowly across a group: through the middle of the group's span over the shot, at
    PAN_SPEED at most, the way the group drifts (else left to right). face: someone in it faces the camera."""
    ok = ~np.isnan(union[:, 0])
    if not ok.any():
        return union, fv.any(1), True
    gx0, gx1 = np.nanmedian(union[ok, 0]), np.nanmedian(union[ok, 2])
    t = np.asarray(times, float)
    span = max(t[-1] - t[0], 1e-3)
    travel = min(max(gx1 - gx0 - cw, 0.0), PAN_SPEED * span)
    cx = (union[ok, 0] + union[ok, 2]) / 2
    sign = -1.0 if len(cx) > 1 and cx[-1] - cx[0] < -0.02 else 1.0
    c = (gx0 + gx1) / 2 + sign * travel * ((t - t[0]) / span - 0.5)
    half = 0.45 * cw
    y0 = np.where(ok, union[:, 1], np.nanmedian(union[ok, 1]))
    y1 = np.where(ok, union[:, 3], np.nanmedian(union[ok, 3]))
    pcx = (nb[..., 0] + nb[..., 2]) / 2
    inside = m & (np.abs(np.nan_to_num(pcx, nan=9) - c[:, None]) <= half)
    return np.c_[c - half, y0, c + half, y1], (fv & inside).any(1), PAN


PAIR_GAP, PAIR_H = 1.0, (0.6, 1.65)              # partners: closer than this many widths (arm's length in open
                                                  # position), heights within this ratio
FOLLOW_ROOM = 0.25                                # extra room on the follow's side, in follow widths (arms, spins)
FOLLOW_W = 0.62                                   # when the couple won't fit: the crop centre's weight on the follow
SAME_H = 0.92                                     # partners within this height ratio are "of a size"
PAIR_DEPTH = 2.5                                  # a "partner" this many times smaller (box area) is further back


def _partners(a, b):
    """Two boxes (fractions) that dance together: side by side or overlapping, about the same height."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    return (PAIR_H[0] <= hb / max(ha, 1e-6) <= PAIR_H[1]
            and _gap(a, b) < PAIR_GAP * max(a[2] - a[0], b[2] - b[0]))


def couple(d, times, boxes, m, close, cw, fv, act=None, union=None):
    """A couple dancing together shares the frame: the subject and their partner (the nearest person about their
    size, whether or not the subject pass caught them), both tracked box to box. The follow -- the one the role scan
    reads as a woman, else the shorter -- gets the emphasis: the target box takes in both with extra room on the
    follow's side, so the crop centres a little towards her and zooms in only as far as both fit; when they are
    wider than the crop, the crop keeps the follow whole and as much of the lead as fits, weighted to the follow.
    Returns (target boxes (F, 4), face (F,) bool) or None when the shot isn't of a couple."""
    from .role import woman
    F = len(times)
    nb = boxes[..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    ok = ~np.isnan(nb[..., 0]) & close[:, None]
    pt = d["pose_t"]                              # out of focus: never one of the pair
    soft = blurry(d, np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)) if "focus" in d else np.zeros_like(ok)
    ok &= ~soft
    found = {}
    edge = (np.nan_to_num(nb[..., 0], nan=0.5) <= 0.01) | (np.nan_to_num(nb[..., 2], nan=0.5) >= 0.99)
    ok &= ~edge                                   # half out of the camera's own frame: never one of the pair
    for f in range(F):
        mem = [k for k in np.nonzero(m[f])[0] if not edge[f, k]] or [k for k in np.nonzero(m[f])[0]]
        if not mem:
            continue
        size = lambda k: (nb[f, k, 2] - nb[f, k, 0]) * (nb[f, k, 3] - nb[f, k, 1])      # noqa: B023
        mem.sort(key=lambda k: -(act[f, k] + 0.02 * size(k) ** 0.5) if act is not None else -size(k))
        a = mem[0]                                # the one doing the move (the biggest, all else equal)
        cand = [k for k in np.nonzero(ok[f] & ~soft[f])[0] if k != a and _partners(nb[f, a], nb[f, k])]
        if act is not None:                       # a partner dances too: not someone standing by
            cand = [k for k in cand if act[f, k] >= 0.2 * act[f, a] or act[f, a] < 0.05]   # (a lead posing too)
        if cand:                                  # a fellow member first, else the nearest person beside them
            inm = [k for k in cand if k in mem]
            b = (inm or sorted(cand, key=lambda k: _gap(nb[f, a], nb[f, k])))[0]
            found[f] = (a, b)
    seen = int(m.any(1).sum())
    if not seen or len(found) < 0.4 * seen:
        return None
    if act is not None:                           # a routine danced by a group is not a couple
        busy = np.where(m, act, -1.0)
        many = (m & (busy >= ACTIVE * np.maximum(busy.max(1, keepdims=True), 1e-6))).sum(1) >= 3
        wide = np.nan_to_num(union[:, 2] - union[:, 0]) > GROUP_W if union is not None and cw < 0.5 else many & False
        if (many | wide)[m.any(1)].mean() >= 0.5:
            return None
    # who may be one of the pair: the subject's own people and anyone dancing right beside them -- never someone
    # across the floor the tracker wandered onto
    allow = m.copy()
    for f in range(F):
        for k in np.nonzero(ok[f] & ~m[f])[0]:
            allow[f, k] = any(_partners(nb[f, j], nb[f, k]) for j in np.nonzero(m[f])[0])
    f0 = max(found, key=lambda f: sum((nb[f, k, 2] - nb[f, k, 0]) * (nb[f, k, 3] - nb[f, k, 1]) for k in found[f]))
    slots = np.full((F, 2), -1)
    for order in (range(f0, F), range(f0, -1, -1)):
        prev = [nb[f0, found[f0][0]], nb[f0, found[f0][1]]]
        for f in order:
            if f == f0:
                slots[f] = found[f0]
                continue
            c = list(np.nonzero(ok[f] & allow[f])[0])
            gap = np.abs((nb[f, :, 0] + nb[f, :, 2])[:, None] - [p_[0] + p_[2] for p_ in prev]) / 2   # (P, 2)
            best, cost = (-1, -1), 9.0
            for i in [*c, -1]:                    # each dancer to a box (or lost), never both to the same one
                for j in [*c, -1]:
                    if i == j != -1:
                        continue
                    di = gap[i, 0] if i >= 0 else 0.12
                    dj = gap[j, 1] if j >= 0 else 0.12
                    if di <= 0.12 and dj <= 0.12 and di + dj < cost:
                        best, cost = (i, j), di + dj
            slots[f] = best
            for j in (0, 1):
                if best[j] >= 0:
                    prev[j] = nb[f, best[j]]
    seq = np.full((2, F, 4), np.nan)
    face = np.zeros((2, F), bool)
    for j in (0, 1):
        got = slots[:, j] >= 0
        seq[j, got] = nb[np.nonzero(got)[0], slots[got, j]]
        face[j, got] = fv[np.nonzero(got)[0], slots[got, j]]
        if not got.any():
            return None
        idx = np.maximum.accumulate(np.where(got, np.arange(F), 0))   # lost a moment: hold where they were
        idx[:np.argmax(got)] = np.argmax(got)
        seq[j] = seq[j, idx]
    area = [np.nanmedian((seq[j, :, 2] - seq[j, :, 0]) * (seq[j, :, 3] - seq[j, :, 1])) for j in (0, 1)]
    if max(area) > PAIR_DEPTH * max(min(area), 1e-9):       # one right up at the lens, the other well behind:
        return None                                          # not dancing together (follow() takes the near one)
    wo = woman(d, times, boxes)
    p = [np.nanmean([wo[f, slots[f, j]] for f in range(F) if slots[f, j] >= 0] or [np.nan]) for j in (0, 1)]
    if np.isnan(p).any() or abs(p[0] - p[1]) < 0.15:         # the role scan can't tell: the shorter one -- but
        hgt = [np.nanmean(seq[j, :, 3] - seq[j, :, 1]) for j in (0, 1)]   # two of a size (a duo of women) are
        if min(hgt) < SAME_H * max(hgt):                      # told apart by who faces the camera, in focus
            fo = int(np.argmin(hgt))
        else:
            foc = d["focus"][np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)]
            top = max(np.nanmax(foc), 1e-6) if not np.isnan(foc).all() else 1.0
            sc = [face[j].mean() + np.nan_to_num(np.nanmean([foc[f, slots[f, j]] for f in range(F)
                                                             if slots[f, j] >= 0] or [np.nan]) / top) for j in (0, 1)]
            fo = int(np.argmax(sc))
    else:
        fo = int(np.argmax(p))
    fb, lb = seq[fo], seq[1 - fo]
    fw = fb[:, 2] - fb[:, 0]
    out = np.c_[np.minimum(fb[:, 0], lb[:, 0]), np.minimum(fb[:, 1], lb[:, 1]),
                np.maximum(fb[:, 2], lb[:, 2]), np.maximum(fb[:, 3], lb[:, 3])]
    fc, lc = (fb[:, 0] + fb[:, 2]) / 2, (lb[:, 0] + lb[:, 2]) / 2
    left = fc <= lc                                            # the follow is on the left
    out[:, 0] = np.where(left, out[:, 0] - FOLLOW_ROOM * fw, out[:, 0])
    out[:, 2] = np.where(left, out[:, 2], out[:, 2] + FOLLOW_ROOM * fw)
    win = 0.9 * cw
    wide = (out[:, 2] - out[:, 0]) > 0.95 * cw
    if wide.any():                                             # too wide: the follow whole, the lead as fits
        c = FOLLOW_W * fc + (1 - FOLLOW_W) * lc
        lo, hi = fb[:, 2] - win / 2, fb[:, 0] + win / 2
        c = np.where(lo <= hi, np.clip(c, lo, hi), fc)
        out[wide, 0], out[wide, 2] = c[wide] - win / 2, c[wide] + win / 2
    if union is not None:                         # drifted off the subject (the two overlap by under half of the
                                                  # narrower -- a group wider than the strip holds the pair whole
                                                  # inside it): the subject's own box that frame
        iw = np.clip(np.minimum(out[:, 2], union[:, 2]) - np.maximum(out[:, 0], union[:, 0]), 0, None)
        nw = np.maximum(np.minimum(out[:, 2] - out[:, 0], union[:, 2] - union[:, 0]), 1e-6)
        off = ~np.isnan(union[:, 0]) & (np.nan_to_num(iw / nw) < 0.5)
        out[off] = union[off]
    return out, face[fo] | face[1 - fo]


def _gap(a, b):
    """Horizontal gap between two boxes (0 if they overlap)."""
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]))


def crop_size(src_aspect, out_aspect):
    """Crop size at zoom 1 as fractions of the source (width, height)."""
    if out_aspect < src_aspect:                     # e.g. 9:16 out of 16:9: full height, narrow strip
        return out_aspect / src_aspect, 1.0
    return 1.0, src_aspect / out_aspect


def zmax(src_w, src_h, out_w, out_h):
    """The zoom ceiling for this pairing: the crop keeps at least the output's pixels, up to ZMAX."""
    cw, ch = crop_size(src_w / src_h, out_w / out_h)
    return float(np.clip(min(cw * src_w / out_w, ch * src_h / out_h), 1.0, ZMAX))


def framing(union, src_w, src_h, out_w, out_h, touch=None):
    """Per frame: (score 0..1, zoom, crop centre x, crop centre y) for the best crop around the subject. With the
    subject's edge contact (touch: left, top, right, bottom), a head cut off by the camera scores near zero."""
    F = len(union)
    cw0, ch0 = crop_size(src_w / src_h, out_w / out_h)
    zm = zmax(src_w, src_h, out_w, out_h)
    score, z, cx, cy = np.zeros(F), np.ones(F), np.full(F, 0.5), np.full(F, 0.5)
    for f in range(F):
        x0, y0, x1, y1 = union[f]
        if np.isnan(x0):
            continue
        uh, uw = y1 - y0, x1 - x0
        sx, sy = (x0 + x1) / 2, y0 + 0.45 * uh                  # a little headroom
        if touch is not None:                                   # cut off by the camera itself
            l, t_, r, b_ = touch[f]
            edge = (0.2 if t_ else 1.0) * (0.75 if b_ else 1.0) * (0.55 if (l or r) else 1.0)
        else:
            edge = 0.55 if (x0 < 0.01 or x1 > 0.99) else 1.0
        best = (-1, 1.0, 0.5, 0.5)
        for zz in np.linspace(1.0, zm, 9):                      # the zoom that frames the subject best: a tighter
            cw, ch = cw0 / zz, ch0 / zz                         # crop can also move further to centre them
            bx, by = np.clip(sx, cw / 2, 1 - cw / 2), np.clip(sy, ch / 2, 1 - ch / 2)
            cent = np.clip(1 - 2.2 * np.hypot((sx - bx) / cw, (sy - by) / ch), 0, 1)
            fill = uh / ch
            size = min(fill / FILL, 1.0) ** 0.8 * np.clip(1 - (fill - 0.85) / 0.12, 0, 1)   # small, or overflowing
            fit = np.clip(0.95 * cw / max(uw, 1e-3), 0, 1) ** 2  # too wide for the crop: someone gets cut off
            sc = cent * (0.35 + 0.65 * size) * fit * (1 - 0.04 * (zz - 1))   # all else equal, keep it wider
            if sc > best[0]:
                best = (sc, zz, bx, by)
        score[f], z[f], cx[f], cy[f] = best[0] * edge, best[1], best[2], best[3]
    return score, z, cx, cy
