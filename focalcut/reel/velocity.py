"""Velocity edits: where a speed ramp slows down, and on which beat.

A velocity edit (CapCut's custom speed curve, a time-remap ramp in Premiere / After Effects) runs fast through the
move, eases down to arrive slow exactly as the dancer lands -- the end of a turn, the hit of a pose -- holds there so
the hair and the face play out, then eases back up (focalcut.fx.speed draws the curve; dancing never past its CAP,
1.5x). Slowing mid-move gets it backwards: the fast part looks slow and the hold has nothing to show. So:

  settle()   finds the landing from the subject's own motion: a burst, then a rest
  landing()  puts it on the hardest-hitting beat of the shot that still leaves room for the hold after it
"""
from __future__ import annotations

import os

import numpy as np

from . import frame as FR

RAMP = ("hero", "bullet", "montage")                     # the speed curves that hold on a landing
HOLDS = dict(hero=1.0, bullet=2.0, montage=0.75)         # their holds, in beats (fx.speed); close-ups hold 0.5 more


def settle(project, unit, t):
    """Where a move lands: the moment near t (-1.0..+0.8 s) where the subject's motion drops from a burst to a rest.
    Falls back to t."""
    f = project.path("scores", unit["score"] + ".npz")
    if not os.path.exists(f):
        return t
    d = FR.load(f)
    t10, v = d["t"], d["subj"]
    v = np.maximum(v, np.roll(v, 1))                     # measured on alternate samples: fill the gaps
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    best, best_sc = t, 0.0
    for tau in np.arange(max(unit["t0"] + 0.7, t - 1.0), min(unit["t1"] - 0.7, t + 0.8), 0.1):
        burst = v[(t10 >= tau - 0.6) & (t10 < tau)].mean()
        rest = v[(t10 > tau + 0.1) & (t10 <= tau + 0.6)].mean()
        sc = burst - 1.5 * rest
        if burst > 1.4 * rest and burst > 1.5 and sc > best_sc:
            best, best_sc = round(float(tau), 2), sc
    return best


def landing(beats, ramp, closeup, accents):
    """Beats from the shot's start to the landing: the strongest accent (accents[k]: beat k of the shot) among the
    beats that leave the hold (and half a beat to ease out) inside the shot."""
    hold = HOLDS[ramp] + (0.5 if closeup else 0) + 0.5
    ks = [k for k in range(1, beats) if beats - k >= hold] or [max(1, beats // 2)]
    return max(ks, key=lambda k: accents[k] if k < len(accents) else 0)
