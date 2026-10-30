"""The subject of a shot and how well it can be framed.

Subject: from the subject pass's track (SAM 3 masks scored by saliency and depth, ~1 a second) when it exists: the
union of the subject's masks and whether they touch the frame edges. Without it, the fallback is the sharpest
person box (the focus pass, else the largest) plus everyone close to it who is nearly as sharp.
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
        return refine(d, times, union), d["sub_touch"][j], "sam"
    pt = d["pose_t"]
    idx = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    union, _, _ = subject(d["boxes"][idx], d["focus"][idx], d["aw"], d["ah"])
    return union, None, "focus"


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
    """Which pose people show their face to the camera: nose and both eyes confidently seen (a back has none)."""
    return (kp[..., 0, 2] > 0.6) & (kp[..., 1, 2] > 0.6) & (kp[..., 2, 2] > 0.6) & (kp[..., 1, 0] > kp[..., 2, 0])


def follow(d, times, union, cw, split=0.8):
    """For a 9:16 strip out of a 16:9 frame. When the subject group is wider than the strip for a good part of the shot
    (a couple dancing apart), centring on the group frames the empty floor between them: follow one dancer instead --
    the one who shows their face most, then the sharpest and biggest -- tracked box to box. Returns (boxes (F, 4),
    face (F,) bool, followed one?)."""
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
    fv = faces(kp) & m
    wide = np.nan_to_num(w) > split * cw
    if wide.mean() < 0.3:
        # the face that counts is the main person's (the biggest in the group): a partner's face over a back
        # is still a shot of a back
        area = np.where(m, (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]), -1)
        main = np.argmax(np.nan_to_num(area, nan=-1), 1)
        return union, fv[np.arange(len(idx)), main] & m.any(1), False
    nb = boxes[..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    f0 = int(np.argmax(np.where(wide, m.sum(1), -1)))
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
        sc = 2.0 * face.mean() + ok.mean() + np.sqrt(area) + 0.5 * np.nan_to_num(sharp / (np.nanmax(foc) + 1e-6))
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
    return seq, face, True


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
