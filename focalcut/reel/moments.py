"""Turn per-clip scores into ranked dance moments.

Pose tracks (IoU-linked boxes, top 3 people) give the moves -- only the subject's (frame.track: the subject pass's
SAM 3 masks, else the in-focus people), so background dancers spinning don't make a moment:
  spin   the face is seen, then the back of the head, then the face again, within ~1.5 s
  dip    the torso tilts far from vertical (hip-centre -> shoulder-centre more than ~30 degrees)
  trick  kicks and lifts: an ankle up at hip height, or one dancer's hips above the other's shoulders
  style  both wrists above the head (arm styling, shines)
  shake  hips working: the semantic pass's hips score on the subject's crop, backed by the hips wobbling against a
         steady torso in the poses
Flow gives energy (subject motion with the camera's motion removed). Quality gates every moment: subject sharpness,
exposure, camera shake and subject size. Each moment also carries how well it frames after reframing the 4K source
(per output format), and an expressive face lifts it. The opening of each unit is marked down: routines rarely
show their best in the walk-on. Writes moments.json: the top non-overlapping moments per clip, with a clip-level
"interest" score used to share out screen time.

Further kinds: pattern (several turns in ~4 s with arms overhead), closeup (shoulders span much of the frame, face
forward), travel (the camera moves steadily while the subject stays framed).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from . import frame as FR

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices where nobody is tracked

HZ = 5                                           # pose rate
W = dict(spin=1.0, dip=1.1, trick=0.9, pattern=1.0, shake=1.0, closeup=0.95, travel=0.6, style=0.45, energy=0.55)  # kind weights
SHAKE0, SHAKE1 = 0.2, 0.15                       # hip wobble (torso lengths) where a shake starts, and its range
PER_CLIP, GAP = 20, 1.6                          # moments kept per clip; minimum spacing (s)

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
NOSE, L_EYE, R_EYE = 0, 1, 2
L_WRI, R_WRI, L_ANK, R_ANK = 9, 10, 15, 16



def clarity(luma, contrast, ring):
    """How clearly the subject reads, 0.1..1: lit (a well-exposed subject with detail in it) or silhouetted (a dark
    figure cut out against something bright: stage lights, a lit crowd), not murky (dim on dim, no contrast) and not
    blown out. Night footage lives or dies on this."""
    lit = np.clip((luma - 35) / 60, 0, 1) * np.clip((235 - luma) / 30, 0, 1)
    sil = np.clip((ring - luma - 35) / 50, 0, 1) * np.clip((ring - 70) / 50, 0, 1)
    detail = np.clip(contrast / 45, 0, 1)
    return np.clip(np.maximum(lit * (0.45 + 0.55 * detail), sil * (0.6 + 0.4 * detail)), 0.1, 1)

def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def tracks(boxes, kp, n=3):
    """Link the n largest people frame to frame by box overlap -> kp array (frames, n, 17, 3) with stable slots, and
    which box each slot holds per frame (-1: empty)."""
    F = len(boxes)
    out = np.full((F, n, 17, 3), np.nan, np.float32)
    which = np.full((F, n), -1)
    last = [None] * n
    for f in range(F):
        cand = [i for i in range(boxes.shape[1]) if not np.isnan(boxes[f, i, 0])][:n + 2]
        used = set()
        for s in range(n):
            if last[s] is None:
                continue
            best = max(cand, key=lambda i: iou(last[s], boxes[f, i]) if i not in used else -1, default=None)
            if best is not None and best not in used and iou(last[s], boxes[f, best]) > 0.2:
                out[f, s] = kp[f, best]; last[s] = boxes[f, best]; used.add(best); which[f, s] = best
            elif f > 0 and np.isnan(out[f - 1, s, 0, 0]):
                last[s] = None                    # lost for two frames: free the slot
        for s in range(n):
            if last[s] is None:
                free = [i for i in cand if i not in used]
                if free:
                    i = free[0]; out[f, s] = kp[f, i]; last[s] = boxes[f, i]; used.add(i); which[f, s] = i
    return out, which


def pt(k, i, conf=0.3):
    return np.where(k[..., i, 2:3] >= conf, k[..., i, :2], np.nan)


def moves(K):
    """Per frame, per tracked person: spin, dip, trick, style evidence in [0, 1]."""
    ls, rs = pt(K, L_SHO), pt(K, R_SHO)
    lh, rh = pt(K, L_HIP, 0.35), pt(K, R_HIP, 0.35)
    sho, hip = (ls + rs) / 2, (lh + rh) / 2
    torso = np.linalg.norm(sho - hip, axis=-1)                     # body scale
    present = (K[..., L_SHO, 2] >= 0.3) & (K[..., R_SHO, 2] >= 0.3)
    nose = np.nan_to_num(K[..., NOSE, 2])
    # spin: face seen -> back of the head -> face again, with the turned-away stretch brief (<= 0.8 s)
    front, back = present & (nose >= 0.6), present & (nose < 0.25)
    spin = np.zeros(nose.shape)
    W2, MAXBACK = int(0.9 * HZ), int(0.8 * HZ)
    F, P = nose.shape
    for p in range(P):
        f = 0
        while f < F:
            if back[f, p]:
                e = f
                while e + 1 < F and back[e + 1, p]:
                    e += 1
                before = front[max(0, f - W2):f, p].any()
                after = front[e + 1:e + 1 + W2, p].any()
                if before and after and (e - f + 1) <= MAXBACK:
                    spin[max(0, f - 2):e + 3, p] = 1.0
                f = e + 1
            else:
                f += 1
    up = sho - hip
    tilt = np.degrees(np.arctan2(np.abs(up[..., 0]), -up[..., 1]))   # 0 = upright, 90 = horizontal
    dip = np.nan_to_num(np.clip((tilt - 30) / 35, 0, 1))
    la, ra = pt(K, L_ANK, 0.5), pt(K, R_ANK, 0.5)
    ank_up = np.fmin(la[..., 1], ra[..., 1])                        # higher ankle (smaller y)
    kick = np.clip((hip[..., 1] - ank_up) / (0.4 * torso + 1e-6), 0, 1)   # a confidently seen ankle above the hips
    # lift: someone's hips above someone else's shoulders
    lift = np.zeros_like(kick)
    for p in range(K.shape[1]):
        for q in range(K.shape[1]):
            if p != q:
                lift[:, p] = np.fmax(lift[:, p], np.clip((sho[:, q, 1] - hip[:, p, 1]) / (0.5 * torso[:, q] + 1e-6), 0, 1))
    trick = np.fmax(kick, lift)
    lw, rw, nz = pt(K, L_WRI), pt(K, R_WRI), pt(K, NOSE)
    style = ((lw[..., 1] < nz[..., 1]) & (rw[..., 1] < nz[..., 1])).astype(float)
    # shake: hips working under a steady torso -- the hip centre swinging against the shoulders, the hips twisting
    # (their width pumping) or bouncing; rolling spread over ~1 s after removing the slow drift
    def wobble(x):
        x = np.where(np.isnan(x), np.nanmean(x, 0, keepdims=True), x)
        x = np.nan_to_num(x)
        k = np.ones(HZ) / HZ
        hp = x - np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)
        return np.sqrt(np.apply_along_axis(lambda c: np.convolve(c ** 2, k, mode="same"), 0, hp))
    sway = (hip[..., 0] - sho[..., 0]) / (torso + 1e-6)
    twist = np.abs(lh[..., 0] - rh[..., 0]) / (torso + 1e-6)
    bounce = (hip[..., 1] - sho[..., 1]) / (torso + 1e-6)
    hips_seen = ~np.isnan(hip[..., 0]) & present & (np.nan_to_num(torso) >= 35)   # small people: keypoint jitter
    shake = np.clip((wobble(sway) + wobble(twist) + 0.6 * wobble(bounce) - SHAKE0) / SHAKE1, 0, 1) * hips_seen
    return {k: np.nan_to_num(np.nanmax(v, axis=1)) for k, v in
            dict(spin=spin, dip=dip, trick=trick, style=style, shake=shake).items()}


def smooth(x, k=3):
    return np.convolve(x, np.ones(k) / k, mode="same")


def clip_series(path):
    """Per pose-frame series for a whole clip: move evidence, energy, quality."""
    d = FR.load(path)
    pt_ = d["pose_t"]
    if len(pt_) < 4:
        return None
    K, which = tracks(d["boxes"], d["kp"])
    # only the subject's moves count: people in the background spinning don't make a moment
    union, touch, _ = FR.track(d, pt_)
    member = FR.members(d["boxes"], union, d["aw"], d["ah"])
    if not member.any():                          # nothing matched (e.g. rotated clips): everyone counts
        member = ~np.isnan(d["boxes"][..., 0])
    sub_sharp = np.nanmax(np.where(member, d["focus"], np.nan), axis=1)
    is_sub = np.take_along_axis(np.c_[member, np.zeros(len(member), bool)], np.where(which < 0, member.shape[1], which), 1)
    K = np.where(is_sub[..., None, None], K, np.nan)
    mv = moves(K)
    t10 = d["t"]
    cam, subj = np.interp(pt_, t10, d["cam"]), np.interp(pt_, t10, d["subj"])
    idx = np.clip(np.searchsorted(t10, pt_), 0, len(t10) - 1)
    sharp, luma, area = d["sharp"][idx], d["luma"][idx], d["area"][idx]
    # energy: subject motion relative to this clip, but also absolute (a still clip shouldn't look energetic)
    energy = np.clip(subj / (np.percentile(subj, 90) + 1e-6), 0, 1) * np.clip(subj / 6, 0, 1)
    steady = np.clip(1.4 - cam / 12, 0.2, 1)
    if "contrast" in d:
        light = clarity(luma, d["contrast"][idx], d["ring"][idx])
    else:                                          # scored before contrast and ring were measured: exposure alone
        light = np.clip(1 - np.abs(luma - 115) / 110, 0.15, 1)
    quality = (np.clip(np.log1p(sharp) / np.log1p(150), 0, 1) ** 0.7
               * light * steady * np.clip(area / 0.04, 0.25, 1))
    # the subject in focus, relative to the sharpest this clip gets (soft or missed focus marks a moment down)
    if not np.isnan(sub_sharp).all():
        rel = np.nan_to_num(sub_sharp / (np.nanpercentile(sub_sharp, 90) + 1e-6), nan=0.5)
        quality = quality * (0.45 + 0.55 * np.clip(rel / 0.7, 0, 1))
    # framing after reframing the 4K source, per output format, averaged over ~2 s (a shot has to hold it)
    w, h = d["size"]
    k2 = np.ones(2 * HZ) / (2 * HZ)
    frm = {fmt: np.convolve(FR.framing(union, w, h, *size, touch=touch)[0], k2, mode="same")
           for fmt, size in (("landscape", (1920, 1080)), ("portrait", (1080, 1920)))}
    parts = {k: smooth(mv[k]) for k in ("spin", "dip", "trick", "style", "shake")}
    parts["shake"] *= np.clip(1 - 1.5 * parts["spin"], 0, 1)      # a turn swings the hips too: that's a spin
    # what the subject's crop shows (the semantic pass): hips/butt working is the real shake signal -- poses at 5 fps can't
    # see a 3-5 Hz shake, so their wobble only backs it up -- and an expressive face lifts any moment
    face_sem = np.zeros(len(pt_))
    sem = path[:-4] + "_semantic.npz"
    if os.path.exists(sem):
        z = np.load(sem)
        if len(z["t"]) > 1:
            near = np.abs(z["t"][None, :] - pt_[:, None]).min(1) <= 1.2
            hips = np.interp(pt_, z["t"], z["hips"]) * near
            face_sem = np.interp(pt_, z["t"], z["face"]) * near
            parts["shake"] = np.fmax(0.5 * parts["shake"], np.clip((hips - 0.3) / 0.4, 0, 1))
    parts["energy"] = smooth(energy)
    # turn pattern: several turns inside ~4 s with arms passing overhead (hand-overs)
    k4 = np.ones(int(4 * HZ)) / (4 * HZ)
    turns = np.convolve((mv["spin"] > 0.5).astype(float), k4, mode="same") * 4 * HZ / 6   # ~one turn = 6 flagged frames
    arms = np.convolve(mv["style"], k4, mode="same")
    parts["pattern"] = np.clip((turns - 0.5) / 1.5, 0, 1) * (0.6 + 0.4 * np.clip(arms * 3, 0, 1))
    # close-up: a face-forward person filling a big part of the frame (expression decides the rank later)
    w, h = d["size"]
    aw = 640 if w >= h else 640 * w / h                            # analysis-frame width
    sw = np.abs(K[:, :, L_SHO, 0] - K[:, :, R_SHO, 0])
    sw = np.where((K[:, :, L_SHO, 2] > 0.4) & (K[:, :, R_SHO, 2] > 0.4) & (K[:, :, NOSE, 2] > 0.6), sw, 0)
    parts["closeup"] = smooth(np.clip((np.nanmax(sw, axis=1) / aw - 0.28) / 0.25, 0, 1))   # shoulders span the frame
    # travelling shot: the camera moves steadily (smooth, sustained motion) while the subject stays framed and sharp
    k1 = np.ones(HZ) / HZ
    cm = np.convolve(cam, k1, mode="same")
    cv = np.sqrt(np.maximum(np.convolve(cam ** 2, k1, mode="same") - cm ** 2, 0))
    cxs = np.interp(pt_, t10, d["cx"])
    cxv = np.sqrt(np.maximum(np.convolve(cxs ** 2, k1, mode="same") - np.convolve(cxs, k1, mode="same") ** 2, 0))
    parts["travel"] = (np.clip((cm - 2) / 5, 0, 1) * np.clip(1 - cv / (cm + 1e-6), 0, 1)
                       * np.clip(1 - cxv / 0.12, 0, 1) * (area > 0.03))
    # a travelling shot shouldn't be marked down as camera shake
    quality = np.where(parts["travel"] > 0.4, quality / steady * 0.95, quality)
    act = activity(pt_, t10, d["subj"], ~np.isnan(union[:, 0]))
    return dict(t=pt_, parts=parts, quality=quality, steady=steady, framing=frm, activity=act,
                face=np.clip((face_sem - 0.5) / 0.4, 0, 1))


def activity(pt_, t10, subj, present):
    """Whole-shot activity, per pose frame: over the stretch a shot actually plays (about -0.8..+1.2 s around the
    peak), is the subject dancing the whole time (subject motion, 10 fps) and in frame the whole time (the subject
    track present)? Someone waiting, walking or standing scores low. 0..1."""
    win = (np.arange(-8, 13) / 10)[None, :]
    sw = np.interp(pt_[:, None] + win, t10, subj)
    there = np.interp(pt_[:, None] + win, pt_, np.asarray(present, float))
    return np.clip((sw.mean(1) - 0.6) / 1.6, 0, 1) * np.clip(there.mean(1) / 0.8, 0, 1)


def unit_moments(u, S):
    """Top non-overlapping moments inside one unit's time range. Transitions rank on steadiness and image quality."""
    res = dict(id=u["id"], category=u["category"], role=u["role"], orient=u["orient"], interest=0.0, moments=[])
    if S is None:
        return res
    t = S["t"]
    m = (t >= u["t0"] + 0.6) & (t <= u["t1"] - 0.6)          # keep clear of the unit's edges
    if not m.any():
        return res
    # the openings of routines (walking on, finding the frame) rarely hold the best of it: the middle does
    pos = (t - u["t0"]) / max(u["t1"] - u["t0"], 1e-6)
    prior = np.clip(0.3 + pos / 0.25 * 0.7, 0.3, 1) * np.clip((1 - pos) / 0.06, 0.5, 1)
    if u["role"] == "transition":
        total = S["quality"] * (0.6 + 0.4 * S["steady"]) * (0.7 + 0.3 * S["parts"]["energy"])
        per = {"landscape": total, "portrait": total}
    else:
        w = dict(W, shake=0.6) if u["role"] == "dance" else W   # socials: shakes, but spins and faces too
        # dancing the whole shot long; a close-up may hold still -- the face carries it
        act = np.fmax(S["activity"], 0.5 * S["parts"]["closeup"]) if "activity" in S else 1.0
        base = (sum(w[k] * S["parts"][k] for k in W) * S["quality"] * prior * (1 + 0.3 * S["face"])
                * (0.12 + 0.88 * act))
        per = {f: base * (0.15 + 0.85 * S["framing"][f]) for f in S["framing"]}   # a centred subject first
        total = np.fmax(per["landscape"], per["portrait"])
    total = np.where(m, total, -1)
    order, picked = np.argsort(-total), []
    for i in order:
        if total[i] <= 0.02 or len(picked) >= PER_CLIP:
            break
        if all(abs(t[i] - t[j]) >= GAP for j in picked):
            picked.append(i)
    for i in sorted(picked):
        wk = dict(W, shake=0.6) if u["role"] == "dance" else W
        kind = "broll" if u["role"] == "transition" else max(W, key=lambda k: wk[k] * S["parts"][k][i])
        res["moments"].append(dict(t=round(float(t[i]), 2), score=round(float(total[i]), 3), kind=kind,
                                   score_l=round(float(per["landscape"][i]), 3),
                                   score_p=round(float(per["portrait"][i]), 3),
                                   frame_l=round(float(S["framing"]["landscape"][i]), 2) if "framing" in S and u["role"] != "transition" else 1.0,
                                   frame_p=round(float(S["framing"]["portrait"][i]), 2) if "framing" in S and u["role"] != "transition" else 1.0,
                                   quality=round(float(S["quality"][i]), 2),
                                   activity=round(float(S["activity"][i]), 2) if "activity" in S else 1.0,
                                   parts={k: round(float(S["parts"][k][i]), 2) for k in W}))
    top = sorted((x["score"] for x in res["moments"]), reverse=True)[:5]
    res["interest"] = round(float(np.sum(top)), 3)
    return res


def run(project) -> list[dict]:
    units = project.load("units.json")
    cache, res, missing = {}, [], set()
    for u in units:
        p = project.path("scores", u["score"] + ".npz")
        if not os.path.exists(p):
            missing.add(u["score"]); continue
        if p not in cache:
            cache[p] = clip_series(p)
        r = unit_moments(u, cache[p]); r["unit"] = u; res.append(r)
    project.dump("moments.json", res)
    kinds = {}
    for r in res:
        for m in r["moments"]:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(f"[reel] {sum(len(r['moments']) for r in res)} moments from {len(res)} units: {kinds}")
    if missing:
        print("[reel] not scored yet:", " ".join(sorted(missing)))
    return res
