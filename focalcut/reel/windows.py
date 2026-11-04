"""Scrub every unit for windows that pass every rule, then make the moments from those.

Moment-first selection picks a peak and then finds out whether a shot around it can be used; scrubbing goes the other
way: every unit with people is stepped STEP seconds at a time in windows of LENGTHS seconds, and each window is checked
the way the planner will cut it -- plan.window_ok (the subject's crop: in frame, face, partner, not low, steady, ...),
plus what this module adds: nothing in focus (moments.soft) and a disengaged face (the expression track: the worse
of the subject's and partner's faces, under reel.json "expression_floor"). Windows are cached per format in
windows_<fmt>.json ({unit: [{t0, t1, ok, score, rank, expr, reasons}]}: score is how well it frames, times the
expression factor; rank what the editor picks for -- RANK, reel.json "scrub_rank": the taste score learnt from their
picks (taste.py), an engaged face (laughing, eye contact), a face to camera through the window, the subject in sharp
focus, and only then the move evidence (spins, tricks, energy); the frame's overall sharpness is left out: a busy
background reads sharp).

The moments then come from the passing windows only: as many non-overlapping ones per unit as pass, each at its
window's peak, ranked by rank x score, with the window ("win") and which formats it passes in ("fmt_ok"). A unit with
no passing window keeps no moments and says why ("fails": reason -> windows). Reads only cached data: no models.
"""
from __future__ import annotations

import os

import numpy as np

STEP, LENGTHS = 0.5, (1.2, 2.4, 3.6)
EXPR_FLOOR, EXPR_W = -0.15, 0.6                   # disengaged under this fails; the factor below 0 (reel.json keys)
SOFT_SHARE = 0.25                                 # a window this much soft (nothing in focus) fails
# what ranks a passing window (reel.json "scrub_rank"): the learned taste first (taste.py), then an engaged face,
# a face to camera, the subject in focus, the move; without a taste model, the others share its weight
RANK = dict(taste=0.35, engaged=0.25, face=0.15, focus=0.15, move=0.10)
FOCUS_FULL = 30.0                                 # subject sharpness (focus pass) that counts as fully sharp
GAP, PER_UNIT = 1.6, 40                           # moments: minimum spacing (s), most kept per unit
FORMATS = ("landscape", "portrait")


# the pick rules on the people a shot is about (persons.py), reel.json "person_rules" overrides: someone the shot is
# about shows their own face for at least "face" of it, and the subject (the tallest) for at least "subject" unless
# their partner does for "face" (else "back": a couple mid-turn passes, a back with nobody's face doesn't); the
# subject's face turned no more than "yaw" eye-distances on the median (else "profile"; off by default, None: an
# engaged face side-on is not a back -- concepts §19 -- and the landmark measure is rough); and the subject's face at least "rel" of the frame's sharpest detail on the median
# (else "soft")
PERSON = dict(face=0.4, subject=0.2, yaw=None, rel=0.12, present=0.5)


def person_reasons(pr, t0, t1, rules=None):
    """The pick rules a window fails on its main people (persons track pr), and the measured values:
    (reasons, {face: per-slot share, yaw, rel})."""
    R = {**PERSON, **(rules or {})}
    if pr is None:
        return [], {}
    m = (pr["t"] >= t0 - 0.1) & (pr["t"] <= t1 + 0.1)
    if not m.any():
        return [], {}
    present = ~np.isnan(pr["box"][m][..., 0])                 # (k, 3)
    face = pr["face"][m]
    why, vals = [], {}
    shares = []
    for s in range(present.shape[1]):
        if present[:, s].mean() >= R["present"]:
            shares.append(float(face[present[:, s], s].mean()))
    vals["face"] = [round(x, 2) for x in shares]
    if shares and (max(shares) < R["face"] or (shares[0] < R["subject"] and max(shares[1:], default=0) < R["face"])):
        why.append("back")
    seen = face[:, 0]
    if seen.any():
        yaw = float(np.nanmedian(np.abs(pr["yaw"][m][seen, 0])))
        rel = float(np.nanmedian(pr["rel"][m][seen, 0])) if not np.isnan(pr["rel"][m][seen, 0]).all() else np.nan
        vals.update(yaw=round(yaw, 2), rel=None if np.isnan(rel) else round(rel, 3))
        if R["yaw"] is not None and yaw > R["yaw"]:
            why.append("profile")
        if not np.isnan(rel) and rel < R["rel"]:
            why.append("soft")
    return why, vals


def grid(u):
    """(t0, t1) windows over a unit, clear of its edges."""
    a, b = u["t0"] + 0.3, u["t1"] - 0.3
    return [(round(float(t), 2), round(float(t) + L, 2)) for L in LENGTHS for t in np.arange(a, b - L + 1e-6, STEP)]


def scrub(project, rec, S, tr, fmt, cfg, taste=None, pr=None):
    """Every window of one unit in one format: [{t0, t1, ok, score, rank, expr, taste, reasons, person}]. taste:
    (track, scores) of the unit's frames, or None; pr: its persons track (persons.py), for the pick rules on the
    people the shot is about (person_reasons) -- for people footage only, never scenery."""
    from . import plan as PL0
    people = rec["unit"]["role"] != "transition" or PL0.people_broll(rec["unit"])
    from . import expression as EX
    from . import moments as MO
    from . import plan as PL
    from . import taste as TA
    u = rec["unit"]
    per, _ = MO.curve(u, S) if S is not None else ({}, None)
    mv = per.get(fmt)
    top = float(np.percentile(mv, 95)) if mv is not None and len(mv) else 0.0
    W = cfg["rank"]
    out = []
    for t0, t1 in grid(u):
        r = PL.window_ok(project, rec, t0, t1, fmt)
        reasons, score = list(r["reasons"]), float(r["score"])
        e, best = EX.window(tr, t0, t1)
        move = focus = 0.0
        if S is not None:
            m = (S["t"] >= t0) & (S["t"] <= t1)
            if m.any():
                if S["soft"][m].mean() >= SOFT_SHARE:
                    reasons.append("soft")
                move = float(np.clip(mv[m].max() / max(top, 1e-6), 0, 1))
                sf = S["sub_focus"][m] if "sub_focus" in S else np.array([np.nan])
                focus = float(np.clip(np.nanmedian(sf) / FOCUS_FULL, 0, 1)) if not np.isnan(sf).all() else 0.0
        face = float((r.get("crop") or {}).get("face", 0.0))
        tv = TA.window(taste[0], taste[1], t0, t1) if taste else np.nan
        terms = dict(engaged=best, face=face, focus=focus, move=move, taste=tv)
        use = {k: v for k, v in W.items() if not np.isnan(terms.get(k, np.nan))}
        rank = sum(v * terms[k] for k, v in use.items()) / max(sum(use.values()), 1e-6)
        pw, pv = person_reasons(pr, t0, t1, cfg.get("person")) if people else ([], {})
        reasons += [w for w in pw if w not in reasons]
        if not np.isnan(e):
            if e < cfg["floor"]:
                reasons.append("expr")
            if e < 0:
                score *= max(0.1, 1 + cfg["w"] * e)
        out.append(dict(t0=t0, t1=t1, ok=not reasons, score=round(score, 3), rank=round(rank, 3),
                        expr=None if np.isnan(e) else round(e, 2), taste=None if np.isnan(tv) else round(tv, 2),
                        reasons=reasons, person=pv))
    return out


def moments_from(rec, S, tr, wins):
    """A unit's moments from its passing windows (wins: {fmt: [window, ...]}): non-overlapping, best first, each at
    its window's peak; and the fail reasons of a unit with none."""
    from . import expression as EX
    from . import moments as MO
    u = rec["unit"]
    per, total = MO.curve(u, S) if S is not None else ({f: None for f in FORMATS}, None)
    cand = []                                         # (value, t, window, fmt)
    for f, ws in wins.items():
        for w in ws:
            if not w["ok"]:
                continue
            if S is not None:
                m = (S["t"] >= w["t0"]) & (S["t"] <= w["t1"])
                i = np.nonzero(m)[0][int(np.argmax(per[f][m]))] if m.any() else None
                t = float(S["t"][i]) if i is not None else (w["t0"] + w["t1"]) / 2
            else:
                i, t = None, (w["t0"] + w["t1"]) / 2
            cand.append((max(w["rank"], 0.02) * w["score"], t, w, f, i))
    cand.sort(key=lambda c: -c[0])
    picked = []
    for c in cand:
        if len(picked) >= PER_UNIT:
            break
        if all(abs(c[1] - p[1]) >= GAP for p in picked):
            picked.append(c)
    ms = []
    for _, t, w, _f, i in sorted(picked, key=lambda c: c[1]):
        ok = {g: any(x["ok"] and x["t0"] <= t <= x["t1"] for x in wins.get(g, [])) for g in FORMATS}
        val = {g: max((max(x["rank"], 0.02) * x["score"] for x in wins.get(g, []) if x["ok"] and x["t0"] <= t <= x["t1"]),
                      default=0.0) for g in FORMATS}
        _, face = EX.window(tr, w["t0"], w["t1"])
        m = dict(t=round(t, 2), score=round(max(val.values()), 3), score_l=round(val["landscape"], 3),
                 score_p=round(val["portrait"], 3), win=[w["t0"], w["t1"]], fmt_ok=ok, face=round(face, 3),
                 expr=w["expr"])
        if S is not None and i is not None:
            wk = dict(MO.W, shake=0.6) if u["role"] == "dance" else MO.W
            m.update(kind="broll" if u["role"] == "transition" else max(MO.W, key=lambda k: wk[k] * S["parts"][k][i]),
                     frame_l=round(float(S["framing"]["landscape"][i]), 2), frame_p=round(float(S["framing"]["portrait"][i]), 2),
                     quality=round(float(S["quality"][i]), 2),
                     activity=round(float(S["activity"][i]), 2) if "activity" in S else 1.0,
                     parts={k: round(float(S["parts"][k][i]), 2) for k in MO.W})
        else:
            m.update(kind="broll" if u["role"] == "transition" else "energy")
        ms.append(m)
    fails = {}
    for ws in wins.values():
        for w in ws:
            for r in w["reasons"]:
                fails[r] = fails.get(r, 0) + 1
    return ms, fails


def unit_job(args):
    """One unit, in a worker process: its windows per format, its moments and fail reasons."""
    folder, cut, rec, formats, cfg = args
    from . import expression as EX
    from . import moments as MO
    from .project import Project
    project = _projects.get((folder, cut)) or _projects.setdefault((folder, cut), Project(folder, cut))
    u = rec["unit"]
    p = project.path("scores", u["score"] + ".npz")
    S = MO.clip_series(p) if os.path.exists(p) else None
    from . import taste as TA
    tr = EX.load(project, u)
    tt = TA.track(project, u) if cfg.get("taste") else None
    taste = (tt, TA.series(tt, *cfg["taste"])) if tt is not None else None
    from . import persons as PS
    pr = PS.load(project, u)
    formats = [f for f in formats if f == "portrait" or u["orient"] == "landscape"]   # (as the planner: a clip shot
    wins = {f: scrub(project, rec, S, tr, f, cfg, taste, pr) for f in formats}            # upright plays in portrait only)
    ms, fails = moments_from(rec, S, tr, wins)
    return u["id"], wins, ms, fails


_projects = {}


def run(project, formats=FORMATS, workers=12) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    cfg = dict(floor=float(project.config.get("expression_floor", EXPR_FLOOR)),
               w=float(project.config.get("expression_w", EXPR_W)),
               rank={**RANK, **(project.config.get("scrub_rank") or {})},
               person={**PERSON, **(project.config.get("person_rules") or {})})
    from . import taste as TA
    tm = TA.model(project)
    if tm is None:
        print("[reel] taste: no labelled frames yet (picks with adds and bans) -- ranking without it")
    else:
        cfg["taste"] = (tm[0], tm[1])
        print(f"[reel] taste: trained on {tm[3]} labelled frames (pooled); cross-validated AUC on this project's picks "
              + (f"{tm[2]:.2f}" if tm[2] is not None else "n/a (too few)"))
    recs = project.load("moments.json")
    if any("fails" in r for r in recs):              # scrubbed already: start again from the moments step's list
        recs = project.load("moments_raw.json")
    else:
        project.dump("moments_raw.json", recs)       # the moment-first list (kept for comparison, and re-runs)
    out = {f: {} for f in formats}
    by = {r["unit"]["id"]: r for r in recs}
    jobs = [(project.folder, project.cut, r, formats, cfg) for r in recs]
    with ProcessPoolExecutor(max(1, workers)) as ex:   # (forked: the focus and soft settings carry over)
        for k, (uid, wins, ms, fails) in enumerate(ex.map(unit_job, jobs)):
            rec = by[uid]
            for f in wins:
                out[f][uid] = wins[f]
            rec["moments"], rec["fails"] = ms, fails
            top = sorted((m["score"] for m in ms), reverse=True)[:5]
            rec["interest"] = round(float(np.sum(top)), 3)
            n = {f: sum(w["ok"] for w in wins[f]) for f in wins}
            print(f"[reel] windows {k + 1}/{len(recs)} {uid}: " + ", ".join(f"{f} {n[f]}/{len(wins[f])}" for f in wins)
                  + f", {len(ms)} moments" + ("" if ms else f" -- none: {fails}"), flush=True)
    for f in formats:
        project.dump(f"windows_{f}.json", out[f])
    project.dump("moments.json", recs)
    none = [r["unit"]["id"] for r in recs if not r["moments"]]
    print(f"[reel] windows: {sum(len(r['moments']) for r in recs)} moments from passing windows; "
          f"{len(none)} units with none" + (f": {none}" if none else ""))
    return out
