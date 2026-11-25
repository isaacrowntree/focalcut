"""What the camera is aimed at: a frame of something (people, the scene) or a frame of nothing -- the ground while the
operator walks, feet, the sky as the camera comes up, a hand over the lens, the inside of a bag, a blur as it swings.

Every unit's reference frames (reference_frames.npz: a SigLIP 2 embedding every couple of seconds, from `focalcut reel
reference`) are scored zero-shot against two prompt sets, "subject" and "nothing", each frame's best match in either
set against the other: aim.npz holds per frame (unit, t, nothing) with nothing in 0..1. template.Steady never plays a
stretch of a non-aerial clip where it is over "max".

reel.json "aim": {"subject": [...], "nothing": [...], "max": 0.6} replaces the prompts or the limit for a shoot (the
defaults are general: any event, indoors or out); "prefer": [...] (against "plain": [...]) scores what the drone
footage should show -- a person on a boat, say, over empty sand -- for the planner to favour.
"""
from __future__ import annotations

import os

import numpy as np

SUBJECT = ["a photo of people", "a photo of a person's face", "a photo of people talking to each other",
           "a photo of a crowd at an event", "a landscape photo of the sea and the coast", "a photo of a group of people",
           "a scenic photo of a park or a beach", "a photo of a speaker addressing people", "a photo of a building"]
NOTHING = ["a blurry photo of the ground", "a photo of grass taken pointing straight down",
           "a photo of sand taken pointing straight down", "a photo of someone's feet and legs from above",
           "an accidental photo of the sky", "a motion-blurred photo taken while the camera swings",
           "a dark photo of the inside of a bag", "a finger covering the camera lens",
           "a photo of the back of someone's shirt from very close up", "a photo of the floor"]
MAX = 0.6
VERSION = 1


def settings(cfg):
    c = cfg.get("aim") or {}
    return list(c.get("subject") or SUBJECT), list(c.get("nothing") or NOTHING), float(c.get("max", MAX))


def nothing(emb, subj, none, scale=25.0):
    """Per frame (0..1): how much it reads as a frame of nothing rather than of something."""
    d = (emb @ none.T).max(1) - (emb @ subj.T).max(1)
    return 1 / (1 + np.exp(-scale * d))


def run(project) -> str | None:
    """Score every reference frame; writes <reel>/aim.npz. Needs reference_frames.npz (and torch: SigLIP 2's text
    tower). Re-scored only when the frames or the prompts change."""
    import hashlib
    import json
    src = project.path("reference_frames.npz")
    if not os.path.exists(src):
        print("[reel] aim: no reference_frames.npz (run `focalcut reel reference` first) -- skipped", flush=True)
        return None
    subj, none, _ = settings(project.config)
    sig = hashlib.md5(json.dumps([VERSION, subj, none, os.path.getmtime(src),
                                  (project.config.get("aim") or {}).get("prefer"),
                                  (project.config.get("aim") or {}).get("plain")]).encode()).hexdigest()[:12]
    out = project.path("aim.npz")
    if os.path.exists(out) and str(np.load(out)["sig"]) == sig:
        return out
    from .reference import Embedder
    z = np.load(src)
    E = Embedder()
    emb = z["emb"].astype(np.float32)
    n = nothing(emb, E.texts(subj), E.texts(none))
    pref = np.zeros(len(n), np.float32)
    want = (project.config.get("aim") or {}).get("prefer") or []   # what the drone footage should show (a boat
    if want:                                                       # with someone in it, over plain sand)
        plain = (project.config.get("aim") or {}).get("plain") or ["an aerial photo of plain sand", "an aerial photo of empty water"]
        d = (emb @ E.texts(want).T).max(1) - (emb @ E.texts(plain).T).max(1)
        pref = (1 / (1 + np.exp(-25.0 * d))).astype(np.float32)
    np.savez(out, unit=z["unit"], t=z["t"], nothing=n.astype(np.float16), prefer=pref.astype(np.float16), sig=sig)
    print(f"[reel] aim: {len(n)} frames, {int((n > settings(project.config)[2]).sum())} of nothing", flush=True)
    return out


class Aim:
    """aim.npz loaded: nothing_in(unit id, a, b) -> the worst frame of nothing near a..b (0 when unscored)."""
    def __init__(self, project):
        p = project.path("aim.npz")
        self.ok = os.path.exists(p)
        if self.ok:
            z = np.load(p)
            u, t, n = z["unit"].astype(str), z["t"], z["nothing"].astype(np.float32)
            self.by = {k: (t[u == k], n[u == k]) for k in np.unique(u)}
            pf = z["prefer"].astype(np.float32) if "prefer" in z.files else np.zeros(len(n), np.float32)
            self.pref = {k: (t[u == k], pf[u == k]) for k in np.unique(u)} if pf.any() else {}
        else:
            self.pref = {}

    def nothing_in(self, uid, a, b, pad=1.0):
        if not self.ok or uid not in self.by:
            return 0.0
        t, n = self.by[uid]
        m = (t >= a - pad) & (t <= b + pad)
        return float(n[m].max()) if m.any() else 0.0
