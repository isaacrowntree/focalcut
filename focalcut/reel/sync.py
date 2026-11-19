"""Picture motion follows the song's movement: how well a stretch of footage fits the music it would play under.

The music side is music.feel(): the song's movement on each beat (a stop, a close, an entry: music.movement) and its
texture (soft and flowing .. hard and struck). The picture side is motion.series(): the speed of the people in the
shot and the impacts in their movement (Davis & Agrawala's visual beats). fit() puts the two together for one shot
in one place in the song; the planner uses it to choose which pick plays where (plan.motion_order), which part of
a trimmed pick plays and where one of its own shots starts (plan.follow_motion); score() measures a whole cut.

The rules, in the order they weigh:

  rest on a stop    where the music pauses or closes down, the picture comes to rest: a movement lands as it stops
                    (a burst, then stillness: a pose hit, arms coming down, a turn finishing) and stays at rest
                    until the music sounds again. A shot that starts inside the stop starts at rest
  take off on entry where the music comes in, a movement takes off (a jump, arms going up, a spin starting); a shot
                    that starts on the entry starts moving
  texture           soft, flowing music takes soft, flowing movement (a sway, a gliding arm, hair: little of it
                    ends in an impact); hard, struck music takes movement that punches (hits, pops, jumps)
  energy            the picture moves about as much as the music drives

Only structure and texture are matched -- never every beat: motion rhythm and music rhythm are not one to one
(M2Beats, IJCAI 2024), a gesture finishes before the cut, and a cut synced to every beat reads as a gimmick.
Where Bellini et al. ("Dance to the beat", 2018) warp time to bring motion beats onto music beats, the planner
moves in-points and chooses which shot plays where: the cuts stay on the beat grid and nothing is sped up.
"""
from __future__ import annotations

import numpy as np

from . import motion as MO

W_EVENT, W_TEX, W_ENERGY = 3.0, 1.0, 0.6     # fit(): what an event, the texture and the energy weigh
EDGE = 0.25               # an event this close (s) to the shot's start is met by how the shot starts
ENERGY = (0.05, 0.30)     # speed (frame widths a second) that reads as barely moving / all go, on a log scale
LANDED = 0.5              # score(): a stop counts as met from this fit


def context(F, r0, n, T):
    """What the music does under record beats r0..r0 + n (F: music.feel(), T: plan.clock): dict(length, events:
    [(seconds into the shot, "rest" | "go", weight, seconds it lasts)], texture, level)."""
    t0, t1 = T(r0), T(r0 + n)
    ev = []
    for a, e, w in F.get("stops") or ():
        if a < r0 + n and e > r0:
            at = max(T(a), t0)
            ev.append((at - t0, "rest", w, min(T(e), t1) - at))
    for b in F.get("bursts") or ():
        if r0 <= b < r0 + n:
            ev.append((T(b) - t0, "go", 1.0, min(T(b + 1), t1) - T(b)))
    sl = slice(min(r0, len(F["texture"]) - 1), max(min(r0 + n, len(F["texture"])), min(r0, len(F["texture"]) - 1) + 1))
    return dict(length=t1 - t0, events=ev, texture=float(np.mean(F["texture"][sl])),
                level=float(np.clip(np.mean((F["level"] * np.clip(F["drive"] + 0.3, 0, 1))[sl]), 0, 1)))


def energy(v):
    """0..1 from a speed in frame widths a second (log scale between ENERGY)."""
    return float(np.clip(np.log(max(v, 1e-6) / ENERGY[0]) / np.log(ENERGY[1] / ENERGY[0]), 0, 1))


def _rest(M, a, b, ref):
    v = MO.speed(M, a, max(b, a + 0.2))
    return None if v is None else float(np.clip(1.0 - v / (ref + 1e-9), 0, 1))


def fit(M, a, p, ctx):
    """How well source time a onwards, played at speed p, fits the music in ctx (context()): dict(score 0..1,
    events: [0..1 per event], texture 0..1, energy 0..1). M: motion.series(); None (unmeasured): a neutral 0.5."""
    if M is None:
        return dict(score=0.5, events=[0.5] * len(ctx["events"]), texture=0.5, energy=0.5)
    L = ctx["length"] * p
    ref = float(np.median(M["v"])) * 1.5 + MO.STILL          # what "moving" means in this clip
    ev = []
    for at, kind, _w, dur in ctx["events"]:
        x = a + at * p
        if kind == "rest":
            r = _rest(M, x, x + dur * p, ref)
            if at >= EDGE:                                   # a movement lands as the music stops, then rests
                s = MO.settle(M, x, min(0.6, at * p), min(0.6, max(dur * p, 0.3)))
                q = None if s is None or r is None else 0.6 * float(np.clip((s + 0.1) / 0.6, 0, 1)) + 0.4 * r
            else:                                            # the shot starts in the stop: at rest
                q = r
        else:
            if at >= EDGE:
                b = MO.burst(M, x, min(0.5, at * p), 0.5)
                q = None if b is None else float(np.clip((b + 0.1) / 0.6, 0, 1))
            else:                                            # the shot starts on the entry: moving
                v = MO.speed(M, x, x + min(0.5, L))
                q = None if v is None else float(np.clip(v / (ref + 1e-9), 0, 1))
        ev.append(0.5 if q is None else q)
    pu, v = MO.punch(M, a, a + L), MO.speed(M, a, a + L)
    tex = 0.5 if pu is None else 1.0 - abs(pu - ctx["texture"])
    en = 0.5 if v is None else 1.0 - abs(energy(v) - ctx["level"])
    w = [W_EVENT * e[2] for e in ctx["events"]]
    score = (sum(wi * q for wi, q in zip(w, ev)) + W_TEX * tex + W_ENERGY * en) / (sum(w) + W_TEX + W_ENERGY)
    return dict(score=float(score), events=ev, texture=float(tex), energy=float(en))


def _corr(x, y, w):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if len(x) < 3 or w.sum() <= 0:
        return 0.0
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sx, sy = np.sqrt(np.average((x - mx) ** 2, weights=w)), np.sqrt(np.average((y - my) ** 2, weights=w))
    return float(np.average((x - mx) * (y - my), weights=w) / (sx * sy)) if sx > 1e-9 and sy > 1e-9 else 0.0


def score(shots, F, T, series, played):
    """How well a cut's picture follows its music: dict(
      match     0..1, the mean fit() of its shots, by length;
      stops     [met, of]: the music's stops and closes (by weight) where the picture is at rest (fit >= LANDED);
      entries   [met, of]: the entries where it takes off;
      texture   correlation, shot by shot, of how much the movement punches with how hard the music is;
      energy    correlation, shot by shot, of how much the picture moves with how hard the music drives;
      shots     the per-shot fits, for the log).
    shots: placed shots in order (beats, unit); series(unit) -> motion.series; played(shot) -> (source in, source
    length). Unmeasured shots count as neutral."""
    r0, rows, wsum, total = 0, [], 0.0, 0.0
    met = {"rest": [0.0, 0.0], "go": [0.0, 0.0]}
    tx, tm, ex, em, wt = [], [], [], [], []
    for s in shots:
        n = s["beats"]
        ctx = context(F, r0, n, T)
        r0 += n
        a, src = played(s)
        p = src / ctx["length"] if ctx["length"] > 0 else 1.0
        M = series(s["unit"])
        f = fit(M, a, p, ctx)
        rows.append(f)
        total += f["score"] * n
        wsum += n
        for (_, kind, w, _d), q in zip(ctx["events"], f["events"]):
            met[kind][1] += w
            met[kind][0] += w * (q >= LANDED)
        if M is not None:
            pu, v = MO.punch(M, a, a + src), MO.speed(M, a, a + src)
            if pu is not None and v is not None:
                tx.append(pu); tm.append(ctx["texture"]); ex.append(energy(v)); em.append(ctx["level"]); wt.append(n)
    return dict(match=round(total / max(wsum, 1), 3),
                stops=[round(met["rest"][0], 1), round(met["rest"][1], 1)],
                entries=[round(met["go"][0], 1), round(met["go"][1], 1)],
                texture=round(_corr(tx, tm, wt), 3), energy=round(_corr(ex, em, wt), 3), shots=rows)


def line(sc):
    """A score() as one line for the plan log."""
    return (f"match {sc['match']:.2f}; stops met {sc['stops'][0]:g}/{sc['stops'][1]:g}, entries "
            f"{sc['entries'][0]:g}/{sc['entries'][1]:g}; texture r {sc['texture']:+.2f}, energy r {sc['energy']:+.2f}")
