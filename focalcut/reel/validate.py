"""The editor's rules, checked on every shot the planner chose -- and the shot replaced when it breaks one.

The passes before this one each judge a moment their own way (the scrub's windows, assess, recast, the steadier),
and a shot one of them moves is not judged again by the others. This is the last word: after everything has placed
its shots, each one is measured over exactly the stretch it plays, as the strip shows it (the pose tracks of
lead.py: who is in the strip, how sharp, how big, whether their face shows, whether the frame cuts them), and
checked against the rules below. A shot that fails is replaced the same way every time:

  1. its own clip, the nearest stretch that passes everything (so the unit stays in the reel);
  2. else the best passing moment of another unit of the same kind -- for couples the featured ones first (picks
     "more", then units the editor picked from), then units not in the reel yet, then by rank;
  3. else it stays and is flagged in the plan log: "rule fail: <rule> at <time>".

The editor's own shots (picks, the ending, the opening they set) are never touched: what they break is reported,
as their choice. Every choice here is ordered by rule with explicit tie-breaks (time, then unit id): the same
inputs give the same plan.

The rules (RULES; reel.json plan.<fmt>.validate overrides, false turns the pass off):

  soft      the subject is in focus: their sharpness (focus pass: Laplacian energy inside their box at 1280 px) at
            least "sharp", absolute -- a foreground stand, speaker or phone being sharp counts for nothing
  focus     nobody else in the strip is far sharper than the subject (the camera focused on them instead)
  nobody    footage of people shows someone: a tracked person in the strip for "seen" of the shot
  out       the subject is in the strip for "seen" of the shot (not walked out of it, the camera not swung away)
  back      the subject's face shows for "face" of the shot, or their partner's does and the subject is not the
            nearer of the two turned away throughout (performers too: a player from behind fails)
  partner   a couple's partner is not out of the strip or cut by the frame's edge for "partner" of the shot
  halved    nobody the subject's size has their head on the strip's edge for "halved" of the shot (groups aside)
  headcut   the subject's head is not cut by the camera's own frame top for "headcut" of the shot
  far       a couple fills the strip: the taller of them at least "close" of its height
  shake     the camera's shake over the strip's width under the reel's high mark
  banned    none of the editor's banned stretches; day footage only by day
  overused  a B-roll clip plays at most "broll_max" times in a reel ("player_max" for performers), its shots
            "apart" seconds from each other
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import frame as FR
from . import lead as LD

RULES = dict(
    sharp=9.0,               # a performer's absolute sharpness floor (focus pass units: a lit player in focus
                             # reads 15 to 90, one the camera focused past 5 to 8)
    sharp_floor=5.5,         # ... and a dancer's (a dark floor and a moving body read low in focus)
    sharper=2.0,             # someone in the strip this many times sharper than the subject is who the camera is on
    seen=0.5,                # share of the shot the subject's head is in the strip
    face=0.4,                # share of the shot a face must show (the subject's, or their partner's)
    hidden=0.7,              # a partner less in the clear than this (lead.features "front") is hidden behind the other
    partner=0.5,             # share of the shot a couple's partner may be out of the strip or cut by the frame
    edge=0.4,                # share of the shot the subject's own head may sit on the strip's edge
    halved=0.5,              # share of the shot someone the subject's size may have their head on the strip's edge
    headcut=0.4,             # share of the shot the frame's top may cut the subject's head
    close=0.63,              # a couple: the taller partner's height in the strip, at least
    broll_max=2,             # shots of one B-roll clip in a reel (the crowd, the room) ...
    player_max=4,            # ... and of one clip of performers (a band's set, a DJ: the reel is partly about them)
    apart=20.0,              # ... and two shots of one B-roll clip at least this far apart in the reel (s)
    slide=60,                # in-points tried along a failing shot's own clip (nearest first)
    others=40,               # moments of other units tried
)
ORDER = ("banned", "day", "overused", "nobody", "out", "soft", "focus", "back", "partner", "halved", "headcut", "far",
         "shake")
WHAT = dict(banned="plays a banned stretch", day="day footage in a night act", overused="its B-roll clip is over-used",
            nobody="nobody in the strip", out="the subject is out of the strip", soft="the subject is soft",
            focus="the camera is focused on someone else", back="a back, no face", partner="the partner is out of frame",
            halved="someone halved at the strip's edge", headcut="head cut by the frame's top",
            far="the couple is far off", shake="shakes")


def rules(cfg):
    """The rules in force, or None when the pass is off (plan.<fmt>.validate false)."""
    v = (cfg or {}).get("validate", True)
    if v is False:
        return None
    return {**RULES, **(v if isinstance(v, dict) else {})}


MARGIN = dict(sharp=1.15, sharp_floor=1.15, sharper=0.85, seen=1.3, edge=0.6, face=1.5, hidden=1.15, partner=0.7,
              halved=0.7, headcut=0.6, close=1.05)


def strict(R):
    """The rules a replacement must pass: each threshold moved by MARGIN, so a shot that fails is not swapped for a
    stretch that only just scrapes through (the same back a second later, the same partner half a step further in)."""
    return {k: (v * MARGIN[k] if k in MARGIN else v) for k, v in R.items()}


def pose(project, unit):
    """A unit's pose scores and face scan as the reel sees the picture: (d, yf). A clip shot sideways (unit
    "rotate": transpose=1 / 2) has its boxes and keypoints turned with it; None, None without scores."""
    from . import plan as PL
    stem = project.path("scores", unit["score"])
    d = PL._cached(stem + ".npz", FR.load, (stem + "_focus.npz", stem + "_subject.npz"))
    if d is None:
        return None, None
    _, yf = PL._scores(project, unit)
    rot = unit.get("rotate")
    if not rot:
        return dict(d), yf
    key = ("rot", stem, rot)
    hit = PL._LOADED.get(key)
    if hit is None:
        if rot not in ("transpose=1", "transpose=2"):
            return None, None
        r = dict(d)
        aw, ah = float(d["aw"]), float(d["ah"])
        b, kp = np.array(d["boxes"], float), np.array(d["kp"], float)
        nb, nk = b.copy(), kp.copy()
        if rot == "transpose=2":                              # 90 degrees anticlockwise: (x, y) -> (y, w - x)
            nb[..., 0], nb[..., 1], nb[..., 2], nb[..., 3] = b[..., 1], aw - b[..., 2], b[..., 3], aw - b[..., 0]
            nk[..., 0], nk[..., 1] = kp[..., 1], aw - kp[..., 0]
        else:                                                 # clockwise: (x, y) -> (h - y, x)
            nb[..., 0], nb[..., 1], nb[..., 2], nb[..., 3] = ah - b[..., 3], b[..., 0], ah - b[..., 1], b[..., 2]
            nk[..., 0], nk[..., 1] = ah - kp[..., 1], kp[..., 0]
        r.update(boxes=nb, kp=nk, aw=ah, ah=aw)
        r.pop("act", None)
        for k in ("sub_t", "sub_box", "sub_touch"):           # (the subject track is not used here)
            r.pop(k, None)
        hit = PL._LOADED[key] = r
    return dict(hit), None                                    # (the face scan of a turned clip is not placed)


EDGE = 0.12              # a body whose middle is within this (of the strip's width) of its edge is cut by it
SMALL_HEAD = 0.05        # a head under this (of the frame's height) is too small for the face scan: keypoints say


def _own_faces(d, yf, trs):
    """Each track's face, frame by frame, as the planner's own crops count one (frame.face_seen: the face scan, in
    the frame's real shape -- lead.py's own linking assumes a 16:9 frame, and misses the faces of a clip shot
    upright); a head too small for the scan (a player on a far stage) shows a face when the pose found its nose and
    both eyes. Sets tr["face"] in place."""
    pt = d["pose_t"]
    dd = dict(d)
    if yf is not None:
        dd["yf"] = yf
    kpf = (d["kp"][..., 0, 2] > 0.5) & (d["kp"][..., 1, 2] > 0.5) & (d["kp"][..., 2, 2] > 0.5)
    for tr in trs:
        ii = np.clip(np.searchsorted(pt, tr["t"] - 1e-6), 0, len(pt) - 1)
        seen = FR.face_seen(dd, pt[ii], d["boxes"][ii], d["kp"][ii]) if yf is not None else kpf[ii]
        k = tr["slot"].astype(int)
        own = seen[np.arange(len(ii)), k]
        small = tr["hs"] < SMALL_HEAD
        tr["face"] = tr["face"] | own | (small & kpf[ii, k])


def kind_of(unit):
    """What a unit's footage is of: "couple" (the social floor), "show" (a stage), "player" (performers' B-roll:
    the band, a DJ), "scene" (B-roll with nobody it is about)."""
    from . import plan as PL
    if unit["role"] == "dance":
        return "couple"
    if unit["role"] != "transition":
        return "show"
    return "player" if PL.people_broll(unit) else "scene"


def measure(project, unit, a, L, crop, fmt, every=False):
    """What the rules need of the stretch a..a + L of a unit as the crop shows it: dict, or None when the unit has
    no pose scores (nothing to judge). Keys: n (people tracked), sub (the subject's numbers: seen, face, sharp,
    size, h, cut, leaving), mate (their partner's, or None), halved, sharper (the sharpest other person in the
    strip against the subject), group."""
    d, yf = pose(project, unit)
    if d is None:
        return None
    trs = LD.tracks(d, yf, a - 0.3, a + L + 0.3)
    f = LD.features(trs, a, L)
    _own_faces(d, yf, trs)
    c = crop or {}
    cw = float(c.get("w", 1.0))
    pth = np.array(c["path"], float) if c.get("path") else np.array([[0.0, 0.5, 0.5]])
    n_exp = max(1.0, np.floor(L / 0.2))
    rows = []
    for i, tr in enumerate(trs):
        m = (tr["t"] >= a - 1e-6) & (tr["t"] <= a + L + 1e-6)
        if not m.any():
            rows.append(None)
            continue
        cx = np.interp(tr["t"][m] - a, pth[:, 0], pth[:, 1])
        hx, hy = tr["head"][m, 0], tr["head"][m, 1]
        inn = (np.abs(hx - cx) <= cw / 2) & (hx >= 0) & (hx <= 1)
        b = tr["box"][m]
        hw = 0.5 * tr["hs"][m] * LD.ASPECT * 0.8
        edge = np.minimum(np.abs(hx - (cx - cw / 2)), np.abs(hx - (cx + cw / 2)))
        foc = tr["foc"][m]
        tx = tr["torso"][m, 0]                                # their body's middle at the strip's edge, or past
        lo_, hi_ = np.maximum(cx - cw / 2, 0.0), np.minimum(cx + cw / 2, 1.0)   # it: cut through, leaving
        side = (tx < lo_ + EDGE * cw) | (tx > hi_ - EDGE * cw)
        rows.append(dict(
            i=i, seen=min(1.0, float(inn.sum()) / n_exp), off=float(np.mean(np.abs(hx - cx)) / max(cw, 1e-3)),
            face=float((tr["face"][m] & inn).sum() / max(inn.sum(), 1)) if inn.any() else 0.0,
            sharp=float(np.nanmean(foc)) if not np.isnan(foc).all() else None,
            size=float(f["size"][i]), h=float(np.median(np.clip(b[:, 3], 0, 1) - np.clip(b[:, 1], 0, 1))),
            cut=float(((b[:, 1] <= 0.01) & (hy - 0.5 * tr["hs"][m] < 0.0)).mean()),
            leaving=float(side.mean()), tx=float(np.median(tx)), edge=float((edge < 0.6 * hw).sum()) / n_exp,
            hands=LD.hands(tr) is not None, act=float(f["act"][i]), front=float(f["front"][i])))
    live = [r for r in rows if r]
    cand = [r for r in live if r["seen"] >= 0.3]
    out = dict(n=len(live), sub=None, mate=None, halved=0.0, sharper=None, group=False, cw=cw, z=float(c.get("z", 1.0)))
    if every:
        out["rows"] = rows
    if not cand:
        return out
    if cw < 0.9:                                             # a strip: whoever it holds, nearest its middle
        sub = max(cand, key=lambda r: (round(r["seen"] * (1 - min(r["off"], 0.5)) * (0.5 + 0.5 * r["size"]), 4), -r["i"]))
    else:                                                    # the whole frame: the shot's own subject (lead.choose)
        k = LD.choose(trs, a, L)[0]
        sub = rows[k] if k >= 0 and rows[k] and rows[k]["seen"] >= 0.3 else \
            max(cand, key=lambda r: (round(r["seen"] * r["size"], 4), -r["i"]))
    out["sub"] = sub
    peers = [r for r in live if r is not sub and r["size"] >= LD.PEER * sub["size"]]
    out["group"] = sum(r["seen"] >= 0.5 or r["edge"] >= 0.5 for r in peers) >= 2
    out["halved"] = max((r["edge"] for r in peers), default=0.0) if cw < 0.9 else 0.0
    # their partner: the biggest other person their size in the frame for a good part of the shot (in the strip or
    # not: a partner left out of it is what the "partner" rule is about)
    pres = {r["i"]: min(1.0, float(((trs[r["i"]]["t"] >= a - 1e-6) & (trs[r["i"]]["t"] <= a + L + 1e-6)).sum()) / n_exp)
            for r in live}
    mates = [r for r in peers if pres[r["i"]] >= 0.4 and sub["size"] >= LD.PEER * r["size"]]
    if mates:
        out["mate"] = max(mates, key=lambda r: (round(r["size"], 4), -r["i"]))
    sh = [r["sharp"] for r in peers if r["sharp"] is not None and r["seen"] >= 0.5]
    if sh and sub["sharp"] is not None:
        out["sharper"] = max(sh) / max(sub["sharp"], 1e-6)
    return out


def check(m, unit, R, fmt="portrait"):
    """The rules a measured stretch breaks: [(rule, value)], in ORDER. m: measure()'s (None: nothing to judge)."""
    if m is None:
        return []
    kind = kind_of(unit)
    if kind == "scene":
        return []
    why = []
    sub = m["sub"]
    if sub is None:
        return [("nobody", 0.0)] if kind in ("couple", "player") else []
    if sub["seen"] < R["seen"] or sub["edge"] >= R["edge"]:   # gone from the strip, or their head on its edge
        why.append(("out", min(sub["seen"], 1 - sub["edge"])))
    if sub["sharp"] is not None and sub["sharp"] < (R["sharp"] if kind == "player" else R["sharp_floor"]):
        why.append(("soft", sub["sharp"]))
    elif m["sharper"] is not None and m["sharper"] >= R["sharper"] and kind != "couple":
        why.append(("focus", m["sharper"]))                  # (a couple: whoever leads the eye, both are the shot)
    mate = m["mate"]
    if not sub["hands"]:
        best = max(sub["face"], mate["face"] if mate else 0.0)
        # the nearer of a pair turned away with the partner half hidden behind them: a back, whoever's face shows
        wall = mate is not None and sub["face"] < R["face"] and sub["size"] >= mate["size"] and mate["front"] < R["hidden"]
        if best < R["face"] or wall:
            why.append(("back", sub["face"] if wall else best))
    if kind == "couple" and mate is not None and not m["group"]:
        gone = 1 - mate["seen"] * (1 - mate["leaving"])     # not in the strip, or their body on its edge
        if gone > R["partner"]:
            why.append(("partner", gone))
    if m["halved"] > R["halved"] and not m["group"]:
        why.append(("halved", m["halved"]))
    if sub["cut"] > R["headcut"] and not sub["hands"]:
        why.append(("headcut", sub["cut"]))
    if kind == "couple" and fmt == "portrait":
        tall = max(sub["h"], mate["h"] if mate else 0.0) * m["z"]
        if tall < R["close"]:
            why.append(("far", tall))
    return sorted(why, key=lambda x: ORDER.index(x[0]))


def judge(project, s, fmt, R, cfg, crop=None, at=None):
    """Every rule the shot s breaks as it plays ([(rule, value)]): the measured ones (check), its shake, the editor's
    bans, day footage out of the day. crop / at: judge it with this crop from this source in-point instead."""
    from . import plan as PL
    from . import shake
    u = s["unit"]
    a, L = PL.played(s) if at is None else (at, PL.played(s)[1])
    c = s.get("crop") if crop is None and at is None else crop
    why = check(measure(project, u, a, L, c, fmt), u, R, fmt)
    j = shake.jitter(project, u, a, a + L)
    if j is not None and j / max((c or {}).get("w", 1.0), 0.2) > cfg["shake"][1]:
        why.append(("shake", j / max((c or {}).get("w", 1.0), 0.2)))
    if PL.in_bans(a, L, u):
        why.append(("banned", 1.0))
    if u.get("day") and s["act"] not in PL.DAY_ACTS:
        why.append(("day", 1.0))
    return sorted(why, key=lambda x: ORDER.index(x[0]))


def featured(recs):
    """Units the editor asked for more of (picks "more"), then those they picked from: {unit id: 2 | 1}."""
    out = {}
    for r in recs:
        if r.get("more", 0) > 0:
            out[r["id"]] = 2
        elif any(m.get("add") or m.get("pick") for m in r["moments"]):
            out[r["id"]] = 1
    return out


def cap(R, unit):
    """How many shots of a B-roll unit's clip a reel may play."""
    return R["player_max"] if kind_of(unit) == "player" else R["broll_max"]


def overused(shots, R):
    """B-roll shots past a clip's share: indices of the planner's shots of a B-roll clip beyond "broll_max" in the
    reel, or nearer than "apart" to another of it (in reel time, or the same source moment) -- the editor's shots
    of the clip count and stay; of the planner's, the best-ranked (then the earliest) stay."""
    from . import plan as PL
    by = {}
    for i, s in enumerate(shots):
        if s["unit"]["role"] == "transition" and not s.get("logo_image") and not s.get("title"):
            by.setdefault(s["unit"]["clip"], []).append(i)
    out = []
    for clip in sorted(by):
        idx = by[clip]
        keep = [i for i in idx if PL.editors(shots[i])]
        mine = sorted((i for i in idx if not PL.editors(shots[i])),
                      key=lambda i: (not shots[i].get("logo"), not shots[i].get("band"), -shots[i].get("rank", 0), i))
        first = {}                                           # each of the clip's units keeps its best shot first
        for i in mine:                                       # (a clip cut into two units stays two units in the reel)
            first.setdefault(shots[i]["unit"]["id"], i)
        mine = sorted(mine, key=lambda i: (first[shots[i]["unit"]["id"]] != i, mine.index(i)))
        for i in mine:
            s = shots[i]
            near = any((abs(shots[k].get("rec_in", 0) - s.get("rec_in", 0)) < R["apart"] and first[s["unit"]["id"]] != i)
                       or (shots[k]["unit"]["id"] == s["unit"]["id"] and abs(shots[k]["src_in"] - s["src_in"]) < 2.0)
                       for k in keep)                        # (a unit's one shot stays, whatever it is near)
            if len(keep) >= cap(R, s["unit"]) or near:
                out.append(i)
            else:
                keep.append(i)
    return sorted(out)


def run(project, fmt, shots, moments, cfg, W=None, fix=True):
    """Check every shot and replace the planner's that break a rule (see the module's notes). Returns the report:
    dict(checked, replaced: [line], flagged: [line], picks: [line], dropped: [unit id], rules: {rule: n}).
    fix=False: nothing is replaced, every failing shot of the planner's is flagged."""
    from . import plan as PL
    from . import shake
    R = rules(cfg)
    rep = dict(checked=0, replaced=[], flagged=[], picks=[], dropped=[], rules={})
    if R is None:
        return rep
    W = PL.load_windows(project, fmt) if W is None and project is not None else (W or {})
    W = W or {}
    lo = cfg["shake"][0]
    RS = strict(R)
    recs = {r["id"]: r for r in moments}
    feat = featured(moments)
    day_cats = [*(cfg.get("day") or ()), *(cfg.get("day_shows") or ())]
    before = Counter(s["unit"]["id"] for s in shots)
    picks = PL.picks_mode(cfg)
    ob = cfg.get("open_broll")
    opening = (ob if isinstance(ob, list) else ("teaser", "arrival")) if ob else ()

    def fixed(s):                                            # the editor's, or a title: reported, never replaced
        return PL.editors(s) or bool(s.get("title") or s.get("logo_image"))

    def free(uid, a, b, me):
        for x in shots:
            if x is not me and x["unit"]["id"] == uid:
                xa, xl = PL.played(x)
                if min(b, xa + xl) - max(a, xa) > -PL.SNAP_GAP:
                    return False
        return True

    def broll_room(u, me):                                   # a B-roll clip has room for one more shot
        if u["role"] != "transition":
            return True
        mine = [x for x in shots if x is not me and x["unit"]["clip"] == u["clip"]]
        return len(mine) < cap(R, u) and all(abs(x.get("rec_in", 0) - me.get("rec_in", 0)) >= R["apart"] for x in mine)

    def settle(s, src, c):                                   # the shot as it now plays: crop, shake
        u = s["unit"]
        s["crop"] = c or None if fmt == "portrait" or c else s.get("crop")
        a, L = PL.played(s)
        j = shake.jitter(project, u, a, a + L)
        j = j / max((c or {}).get("w", 1.0), 0.2) if j is not None else None
        s["jitter"] = round(j, 5) if j is not None else None
        s["stab"] = bool(j is not None and j > lo and "_stabilized" not in u["source"])
        s.pop("land", None); s.pop("land_fits", None)

    def slide(s, why):
        """The nearest stretch of the shot's own clip that passes every rule: (dt, crop) or None."""
        u = s["unit"]
        if any(k in ("overused", "day") for k, _ in why):
            return None
        a, L = PL.played(s)
        move = bool(s.get("velocity") or s.get("lead_beats") or any(f in PL.SPEED and f != "fit" for f in s["fx"]))
        r = recs.get(u["id"])
        if move:                                             # built round a move: the unit's other moves of the kind
            dts = [m["t"] - s["t"] for m in (r["moments"] if r else ()) if m.get("kind") == s.get("kind")]
        else:
            dts = list(np.arange(u["t0"], u["t1"] - L + 1e-6, 0.25) - a)
        uc = dict(u, bans=[[x - PL.BAN_CLEAR, y + PL.BAN_CLEAR] for x, y in u.get("bans") or []]) \
            if PL.in_bans(a, L, u) else u
        # another stretch, not the same one nudged: clear of what failed (a B-roll clip too close to its other
        # shot in the reel only has to play another moment)
        clear = 0.25 if why[0][0] == "shake" else max(0.75 * L, 1.0)
        tried = 0
        for dt in sorted({round(float(d), 3) for d in dts}, key=lambda d: (abs(d), d)):
            na = round(a + dt, 3)
            if abs(dt) < clear or na < u["t0"] - 1e-6 or na + L > u["t1"] + 1e-6 or PL.in_bans(na, L, uc):
                continue
            if PL.span_ok(W.get(u["id"]), na, na + L) is False or not free(u["id"], na, na + L, s):
                continue
            if tried >= R["slide"]:
                break
            tried += 1
            src = round(s["src_in"] + dt, 3)
            c = PL.reframe(project, u, src, s["length"], fmt) if s.get("crop") or fmt == "portrait" else None
            x = dict(s, t=round(s["t"] + dt, 3), src_in=src, crop=c)
            if not judge(project, x, fmt, RS, cfg):
                return dt, c
        return None

    def other(i, s, why):
        """The best passing moment of another unit of the same kind: (record, moment, src, crop) or None."""
        u = s["unit"]
        kind = kind_of(u)
        a, L = PL.played(s)
        nb = [shots[k] for k in (i - 1, i + 1) if 0 <= k < len(shots)]
        have = Counter(x["unit"]["id"] for x in shots)
        pool = []
        kinds = (kind,) if kind != "scene" else ("scene", "player")
        if any(k == "overused" for k, _ in why):             # other B-roll first, then people
            kinds = ("scene", "player", "couple") if kind == "scene" else ("player", "scene", "couple")
        if picks:                                            # cut from the picks: what they leave is B-roll the
            kinds = tuple(k for k in kinds if k != "couple" and (kind == "couple" or k != "show"))   # editor left alone
        for r in moments:
            ru = r["unit"]
            if r["id"] == u["id"] or kind_of(ru) not in kinds or (fmt != "portrait" and r["orient"] != "landscape"):
                continue
            if any(n["unit"]["clip"] == ru["clip"] for n in nb):
                continue
            if kind_of(ru) != "couple" and any(n["unit"]["role"] == "transition" for n in nb) and kind == "couple":
                continue
            if s["act"] in opening and r["role"] != "transition":
                continue
            d = bool(ru.get("day")) or (bool(day_cats) and PL.fits(ru, day_cats))
            if d != (s["act"] in PL.DAY_ACTS) and (d or s["act"] == "arrival"):
                continue
            if cfg.get("sets") and ru.get("category") != u.get("category"):
                continue
            if s.get("band") and not PL.people_broll(ru):
                continue
            if not broll_room(ru, s):
                continue
            for m in r["moments"]:
                if (m.get("fmt_ok") or {}).get(fmt) is False or m.get("unusable") or m.get("pick") or m.get("add"):
                    continue
                # the least used first, a featured couple counting as two shots fewer per step (so they take the
                # first free slots and no unit takes them all); then its own category, then rank
                load = have[r["id"]] - (2 * feat.get(r["id"], 0) if kind_of(ru) == "couple" else 0)
                pool.append((kinds.index(kind_of(ru)), load, ru.get("category") != u.get("category"),
                             -round(PL.rk(m, fmt), 4), r["id"], m["t"], 0, r, m))
        tried = 0
        for *_, r, m in sorted(pool, key=lambda p: p[:7]):
            ru = r["unit"]
            x = PL.shot(r, m, fmt, s["beats"], act=s["act"])
            src = PL.clear_of_bans(min(max(PL.in_point(x, s["beat"]), ru["t0"]), max(ru["t1"] - L, ru["t0"])), L, ru)
            src = round(src, 3)
            if PL.in_bans(src, L, ru) or PL.span_ok(W.get(ru["id"]), src, src + L) is False \
                    or not free(ru["id"], src, src + L, s):
                continue
            if tried >= R["others"]:
                break
            tried += 1
            c = PL.reframe(project, ru, src, s["length"], fmt) if s.get("crop") or fmt == "portrait" else None
            y = dict(s, unit=ru, t=m["t"], src_in=src, crop=c, fx=[f for f in s["fx"] if f not in PL.SPEED],
                     speed=None, velocity=None)
            if not judge(project, y, fmt, RS, cfg):
                return r, m, src, c, x
        return None

    def name(i, s):
        a, L = PL.played(s)
        return f"#{i} {s['unit']['id']} {a:.2f}-{a + L:.2f} (reel {s.get('rec_in', 0):.1f} s)"

    def said(why):
        return ", ".join(f"{k} {v:.2f}" if k not in ("banned", "day", "overused") else k for k, v in why)

    for rnd in range(2):                                     # (a replacement can over-use a clip: once more)
        over = set(overused(shots, R))
        todo = []
        for i, s in enumerate(shots):
            why = judge(project, s, fmt, R, cfg)
            if i in over:
                why = sorted([*why, ("overused", 1.0)], key=lambda x: ORDER.index(x[0]))
            if rnd == 0:
                rep["checked"] += 1
            if not why:
                s.pop("rule_fail", None)
                continue
            if fixed(s):                                     # the editor's: theirs to choose (a picked clip is
                why = [x for x in why if x[0] not in ("banned", "overused", "day", "shake")]   # banned around its
                # picks, never in them; and it plays as shot, where they put it)
                if rnd == 0 and why:
                    rep["picks"].append(f"{name(i, s)}: {said(why)}")
                continue
            if not fix:                                      # (a refill keeps every shot where it was: reported)
                s["rule_fail"] = [k for k, _ in why]
                continue
            todo.append((i, s, why))
        if not todo:
            break
        for i, s, why in todo:
            first = why[0][0]
            was = name(i, s)
            got = slide(s, why)
            if got:
                dt, c = got
                s.update(t=round(s["t"] + dt, 3), src_in=round(s["src_in"] + dt, 3))
                settle(s, s["src_in"], c)
                s["validated"] = dict(rule=first, moved=round(dt, 2))
                s.pop("rule_fail", None)
                rep["replaced"].append(f"{was}: {said(why)} -> its own clip {dt:+.2f} s")
                rep["rules"][first] = rep["rules"].get(first, 0) + 1
                continue
            got = other(i, s, why)
            if got:
                r, m, src, c, x = got
                s.update({k: x[k] for k in ("unit", "t", "kind", "rank", "face", "framing", "thumb") if k in x},
                         src_in=src, fx=[f for f in s["fx"] if f not in PL.SPEED])
                for k in ("speed", "velocity", "lead_beats", "person", "add", "reuse", "half"):
                    s.pop(k, None)
                s["peak"] = round(s["t"] - s["src_in"], 3)
                settle(s, src, c)
                s["validated"] = dict(rule=first, was=was)
                s.pop("rule_fail", None)
                rep["replaced"].append(f"{was}: {said(why)} -> {r['id']} {src:.2f}"
                                       + (" (featured)" if feat.get(r["id"]) and kind_of(r["unit"]) == "couple" else ""))
                rep["rules"][first] = rep["rules"].get(first, 0) + 1
                continue
            s["rule_fail"] = [k for k, _ in why]
    for i, s in enumerate(shots):
        if s.get("rule_fail"):
            a, L = PL.played(s)
            rep["flagged"].append(f"rule fail: {', '.join(s['rule_fail'])} at {s.get('rec_in', 0):.1f} s "
                                  f"(#{i} {s['unit']['id']} {a:.2f}-{a + L:.2f}"
                                  + ("; nothing passing to replace it)" if fix else ")"))
    after = Counter(s["unit"]["id"] for s in shots)
    rep["dropped"] = sorted(u for u in before if not after.get(u))
    return rep


def report(rep, fmt):
    """The plan log's lines for a validation report."""
    if not rep or not rep.get("checked"):
        return
    n = len(rep["replaced"])
    print(f"[reel] {fmt}: rules: {rep['checked']} shots checked, {n} of the planner's replaced"
          + (f" ({', '.join(f'{v} {k}' for k, v in sorted(rep['rules'].items()))})" if n else "")
          + f", {len(rep['flagged'])} flagged, {len(rep['picks'])} of the editor's own break a rule (kept, as chosen)")
    for line in rep["replaced"]:
        print(f"[reel]   rule: {line}")
    for line in rep["flagged"]:
        print(f"[reel]   WARNING {line}")
    for line in rep["picks"]:
        print(f"[reel]   pick (kept): {line}")
    if rep["dropped"]:
        print(f"[reel]   rules: no longer in the reel (no stretch of theirs passes): {rep['dropped']}")
