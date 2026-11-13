"""Velocity edits: where a speed ramp slows down, and on which beat.

A velocity edit (CapCut's custom speed curve, a time-remap ramp in Premiere / After Effects) runs fast through the
move, eases down to arrive slow exactly as the dancer lands -- the end of a turn, the hit of a pose -- holds there so
the hair and the face play out, then eases back up (focalcut.fx.speed draws the curve; dancing never past its CAP,
1.5x). Slowing mid-move gets it backwards: the fast part looks slow and the hold has nothing to show. So:

  settle()   finds the landing from the subject's own motion: a burst, then a rest
  landing()  puts it on the hardest-hitting beat of the shot that still leaves room for the hold after it

Every other dance shot shows its landing too (landed(), plan.place): a shot that cuts away mid-trick, or the instant
the dancer comes out of it, is all trick and no finish. When a move comes to rest inside or just after a shot, the shot
keeps the end of the move (PRE) and about a beat of the dancer at rest (plan "land_hold") before it cuts.

Holds on sustains: when the music lands on a beat and holds (a note, a chord or a voice held over the next beats,
music.py "sustains"), the picture holds with it. A velocity edit lands on that beat when its shot has one (landing():
the slow hold plays through the held note); a plain shot's landing is slid onto it (on_sustain()); and no cut falls
on a held beat (plan.hold_sustains). Before this was written down it happened by chance of two rules: landing() took
the shot's hardest accent, and an accent is a beat's onset against its neighbours' -- so the beat before a held note
(loud, then nothing) scored highest; and rhythm() gave its long "hold" phrases to the phrases with the lowest mean
accent, which are the ones where notes are held.
"""
from __future__ import annotations

import os

import numpy as np

from . import frame as FR

RAMP = ("hero", "bullet", "montage")                     # the speed curves that hold on a landing
HOLDS = dict(hero=1.0, bullet=2.0, montage=0.75)         # their holds, in beats (fx.speed); close-ups hold 0.5 more
PRE = 0.5                                                # seconds of the move kept before its landing
SUSTAIN = 100.0                                          # what a sustain's first beat adds to its accent (landing)
TAIL = 0.5                                               # every shot's last seconds play at 1x (fx.speed.TAIL)


def _motion(project, unit):
    """(t, subject motion) for a unit, gaps filled and lightly smoothed, or None."""
    f = project.path("scores", unit["score"] + ".npz")
    if not os.path.exists(f):
        return None
    d = FR.load(f)
    v = np.maximum(d["subj"], np.roll(d["subj"], 1))     # measured on alternate samples: fill the gaps
    return d["t"], np.convolve(v, np.ones(3) / 3, mode="same")


def _rest(t10, v, lo, hi):
    """The strongest burst-then-rest point between lo and hi: (time, strength), or (None, 0)."""
    best, best_sc = None, 0.0
    for tau in np.arange(lo, hi, 0.1):
        burst = v[(t10 >= tau - 0.6) & (t10 < tau)]
        rest = v[(t10 > tau + 0.1) & (t10 <= tau + 0.6)]
        if not len(burst) or not len(rest):
            continue
        b, r = burst.mean(), rest.mean()
        sc = b - 1.5 * r
        if b > 1.4 * r and b > 1.5 and sc > best_sc:
            best, best_sc = round(float(tau), 2), sc
    return best, best_sc


def settle(project, unit, t):
    """Where a move lands: the moment near t (-1.0..+0.8 s) where the subject's motion drops from a burst to a rest.
    Falls back to t."""
    mo = _motion(project, unit)
    if mo is None:
        return t
    best, _ = _rest(*mo, max(unit["t0"] + 0.7, t - 1.0), min(unit["t1"] - 0.7, t + 0.8))
    return t if best is None else best


def landed(project, unit, t, after=1.6):
    """Where the move around t comes to rest (the dancer lands): a burst of motion then a rest, from just before the
    peak to after seconds past it. None when nothing lands there (a groove with no finish, or a still)."""
    mo = _motion(project, unit)
    if mo is None:
        return None
    best, _ = _rest(*mo, max(unit["t0"] + 0.6, t - 0.3), min(unit["t1"] - 0.6, t + after))
    return best


def hold_in(s_in, length, land, hold):
    """The in-point that shows the landing: the end of the move (PRE) and hold seconds of rest after it, moved as
    little as possible from s_in. Returns (in-point, fits: the shot is long enough for both). A shot too short for
    both (a one-beat hit) lands 40% in: the end of the move, then the start of the rest."""
    lo, hi = land + hold - length, land - PRE               # in-points that show the rest / still show the move
    if lo <= hi:
        return min(max(s_in, lo), hi), True
    return land - 0.4 * length, False


def on_sustain(length, land, hold, at):
    """The in-point that puts the landing (source time land) `at` seconds into a shot of `length` seconds -- on the
    beat where the music lands and holds -- or None when the shot would then lose the end of the move (PRE) or the
    hold seconds of rest after it (hold_in's range)."""
    x = land - at
    return x if land + hold - length <= x <= land - PRE else None


def landing(beats, ramp, closeup, accents, tail_beats=0.0):
    """Beats from the shot's start to the landing: the strongest accent (accents[k]: beat k of the shot) among the
    beats that leave the hold, half a beat to ease out and tail_beats at 1x (the speed curve's TAIL) inside the shot.
    plan.place passes the music's hits (music.feel) with SUSTAIN added on a beat the music lands and holds on, so
    the slow hold lines up with the held note when the shot has one."""
    hold = HOLDS[ramp] + (0.5 if closeup else 0) + 0.5 + tail_beats
    ks = [k for k in range(1, beats) if beats - k >= hold] or [max(1, beats // 2)]
    return max(ks, key=lambda k: accents[k] if k < len(accents) else 0)


def end_speed(s, fps=50):
    """(lowest, highest) speed over a planned shot's last TAIL seconds, the way render.fx_shot times its curve; None
    for a shot with no speed effect (it plays at 1x)."""
    from .. import fx
    from ..fx.speed import Speed
    sp = [fx.EFFECTS[n]() for n in s.get("fx", []) if n in fx.EFFECTS and issubclass(fx.EFFECTS[n], Speed)]
    if not sp:
        return None
    u, L = s["unit"], s["length"]
    ctx = fx.Context(fps=fps, beat=s.get("beat") or 0.5, peak=s.get("peak"), seed=0,
                     params={"group": s.get("group"), "closeup": s.get("kind") == "closeup",
                             "cap": 3.5 if u.get("role") == "transition" else 1.5})
    if s.get("span"):
        ctx.params["src_len"] = s["span"][1] - s["span"][0]
    t = np.arange(int(round(L * fps)) + 1) / fps
    v = sp[-1].speed(t, L, ctx)
    v = v[t >= L - TAIL]
    return float(v.min()), float(v.max())


def speed_tails(plan, tol=0.02):
    """Shots whose last TAIL seconds are not at 1x, as report lines (an empty list: all good). A velocity edit's
    out-half (flashout: it rushes into its swipe on purpose) is left out."""
    from .. import fx
    out = []
    for i, s in enumerate(plan["shots"]):
        if s.get("velocity") and any(getattr(fx.EFFECTS.get(n), "velocity", False) for n in s.get("fx", [])):
            continue
        r = end_speed(s)
        if r and (r[0] < 1 - tol or r[1] > 1 + tol):
            out.append(f"shot {i} {s['unit']['id']} {'+'.join(s['fx'])}: last {TAIL}s at {r[0]:.2f}-{r[1]:.2f}x")
    return out
