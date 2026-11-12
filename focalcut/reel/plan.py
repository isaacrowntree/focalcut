"""Edit decision list for a recap reel: a story in acts, locked to the music's beat grid.

Acts follow the song (by default its bars split in fixed proportions; set your song's section changes in reel.json)
and tell the day:
  teaser   the festival itself -- band / crowd B-roll (tags in CFG "opening") with the event's logo front and centre
  arrival  daytime B-roll (classes, set-up, panels: tags matching CFG "day"), each held long, woven with the daytime
           show units (categories matching CFG "day_shows", e.g. demos)
  shows    the stage units (role "show"), their routines shuffled through each other
  night    the night's dancing (categories matching CFG "night_shows", e.g. animations, and the social floor, role
           "dance"), broken up by the other B-roll -- band, crowd, DJ
  climax   the music's peak: the strongest moves, short cuts, speed ramps, velocity edits, crowd reactions
  finale   faces, the packed floor, one last slow-motion move into the stinger

How it is cut (the editing rules, and what music videos do -- see docs/recipes/recap-reel.md):
  - every clip appears, with several different scenes from each: a clip that already has shots has to be that much
    better (SHARE), and a unit's shots sit APART seconds apart
  - card shuffle: a clip's shots spread through the act, never two from one clip side by side, never two B-roll shots
    side by side (weave)
  - cut on the action: a turn cuts to a turn, travel keeps its direction across the cut (flow.py), and a movement
    carries two or three cuts, then changes
  - shot lengths come in 8-beat phrases of varied patterns -- quick where the music is busy, long holds between, a
    build into the climax -- never one length over and over (rhythm); moves that need time get the long slots
  - each move's peak lands on a beat, and every cut comes ANTICIPATE seconds ahead of its beat
Every shot is reframed around its subject (frame.py). Effects and transitions are chosen for a reason: speed ramps
on a move's peak, spotlights on stage, a spin cut between two turns, a whip that carries on the camera's movement,
a zoom into a tighter shot, a flash on a musical hit, a signature transition at each change of act. A cut that asks
for nothing stays a hard cut on the beat, and no one transition carries an act.

Cut points sit on the frame nearest each beat counted from the reel's start, so per-shot rounding never accumulates
(rounding each shot on its own drifts the cuts by most of a second over a 3-minute reel).
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
from collections import Counter

import numpy as np

from . import frame as FR
from . import grammar as G
from . import velocity as V

FPS = 50
STINGER_OVERLAP = 1.2                             # the stinger starts this long before the music ends
FACE_W = 0.5                                      # weight of facial expression in a moment's rank
ANTICIPATE = 0.04                                 # cuts land this far ahead of the beat (2 frames), as editors cut
SHARE = 0.3                                       # rank given up per shot a clip already has: spread over every clip
APART = 6.0                                       # a unit's shots at least this far apart: different scenes
# Shot lengths come in 8-beat phrases (one salsa 8-count, two bars), never one length over and over: editors move
# between quick cuts, bar-length shots and long holds, faster where the music is busier and building into a peak
PHRASES = {
    "fast": [[2, 1, 1, 2, 2], [1, 1, 2, 2, 2], [2, 2, 1, 1, 2], [1, 1, 1, 1, 2, 2], [1, 1, 2, 1, 1, 2]],
    "medium": [[3, 1, 2, 2], [2, 2, 4], [2, 1, 1, 4], [4, 2, 2], [2, 3, 3], [1, 1, 2, 4], [2, 2, 1, 3]],
    "hold": [[8], [4, 4], [6, 2]],
    "build": [[2, 2, 1, 1, 1, 1]],
}
LONG = ("pattern", "travel", "spin", "trick", "dip", "style")    # moves that need time to play out
SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920)}
MOVES = ("spin", "dip", "trick", "pattern", "shake")
SPEED = ("hero", "bullet", "montage", "flashin", "flashout", "jumpcut", "fit")
FAST = ("flashin", "flashout", "jumpcut", "fit")       # the ones that speed footage up (an editor's pick never is)
ACT_NAMES = ("teaser", "arrival", "shows", "night", "climax", "finale")
DAY_ACTS = ("arrival", "shows")
SIGNATURE = G.SIGNATURE                           # the change of act gets its own transition (grammar.py)
_COMMON = dict(
    acts=None,                                    # [[name, first bar, end bar], ...] (bar = 4 beats), None: the split
    split=dict(arrival=10, shows=24, night=17, climax=13),
    grammar=G.GRAMMAR, caps=G.CAPS, flash_at=G.FLASH_AT,       # each act's transitions and how often (grammar.py)
    day=["class", "setup", "panel", "workshop", "venue"],      # B-roll tags (substrings) for the arrival
    opening=["band-wide", "band-crowd", "band", "crowd"],      # B-roll tags the reel opens on, best first
    day_shows=["demo"],                                        # show categories (substrings) seen in the daytime
    night_shows=["animation"],                                 # show categories danced at night
    # the band's players, one at a time, cut in on the music's hits: B-roll tags (substrings) and whole clips
    members=["singer", "violin", "horn", "trumpet", "trombone", "sax", "piano", "bass", "conga", "timbal", "bongo",
             "guitar", "drum", "percussion", "player"],
    member_clips=[],
    band_in=dict(shows=2, night=3, climax=3),                  # band-member inserts per act
    # the reel's last shot: [export file name, from, to(, crop centre x)] in that file's own seconds (e.g. the whole
    # room's hands in the middle, bursting up as the music ends); None: one last slow-motion move
    ending=None,
    # shake.py's hf over the crop's width: above the second a stretch is unusable (another part of the clip is used),
    # above the first it plays stabilised
    shake=[0.0012, 0.006],
    # categories or B-roll tags (substrings) kept for the climax and finale, e.g. the headliner's set: they stay out of
    # the earlier acts, and the climax and finale cut from them first
    late=[],
    # show categories (substrings) the climax cuts from after the late footage, e.g. ["night"] so a night headliner's
    # set isn't cut with daytime shows; None: every show (portrait: every dance unit)
    climax_from=None,
    land_hold=1.0,                               # beats of the dancer at rest a shot keeps after a move lands
)
CFG = {
    "landscape": dict(_COMMON, heroes=4, night_pace="medium"),
    "portrait": dict(_COMMON, heroes=3, night_pace="fast"),
}


def config(project, fmt):
    cfg = copy.deepcopy(CFG[fmt])
    cfg.update((project.config.get("plan") or {}).get(fmt, {}))
    cfg.setdefault("look", project.config.get("look"))      # a glitch look for the film (fx/glitch.py LOOKS)
    return cfg


def look_of(cfg):
    """The film's glitch look (reel.json "look": a key of fx/glitch.py LOOKS), or None."""
    if not cfg.get("look"):
        return None
    from ..fx.looks import LOOKS
    return LOOKS.get(cfg["look"])


def default_acts(cfg, bars):
    """Teaser 2 bars and finale 6; the bars between shared out in the split's proportions."""
    mid = max(bars - 8, 4)
    w = cfg["split"]
    tot = sum(w.values())
    acts, a, acc = [("teaser", 0, 2)], 2, 0
    for name in ("arrival", "shows", "night", "climax"):
        acc += w[name]
        b = 2 + round(mid * acc / tot)
        acts.append((name, a, b)); a = b
    return [*acts, ("finale", a, max(bars, a + 1))]


def rk(m, fmt):
    return m.get("score_l" if fmt == "landscape" else "score_p", m["score"]) + FACE_W * m.get("face", 0.0)


def fr(m, fmt):
    return m.get("frame_l" if fmt == "landscape" else "frame_p", 1.0)


class Picker:
    """Moments not used yet (a unit's shots must be APART seconds apart, so each is a different scene), ranked with a
    fair-share discount per shot their clip already has."""
    def __init__(self, fmt, records):
        self.fmt, self.recs, self.used, self.count = fmt, records, {}, {}
        self.hold_adds = False                    # (picks cut to pieces: include_adds places them, the acts don't)

    def free(self, r, m):
        return not (self.hold_adds and m.get("add")) and \
            all(abs(m["t"] - t) >= APART for t in self.used.get(r["id"], []))

    def share(self, r, m):
        """Rank after the fair-share discount: a clip that already has shots has to be that much better (less so for
        a clip the editor asked for more of: picks "more")."""
        return rk(m, self.fmt) - SHARE * self.count.get(r["unit"]["clip"], 0) / (1 + max(r.get("more", 0), 0))

    def load(self, clip):
        """How many shots a clip has, for choosing which clip gives one up (picks "more" counts against it)."""
        more = max((r.get("more", 0) for r in self.recs if r["unit"]["clip"] == clip), default=0)
        return self.count.get(clip, 0) / (1 + more)

    def take(self, r, m, **kw):
        self.used.setdefault(r["id"], []).append(m["t"])
        self.count[r["unit"]["clip"]] = self.count.get(r["unit"]["clip"], 0) + 1
        out = dict(unit=r["unit"], t=m["t"], kind=m["kind"], rank=round(rk(m, self.fmt), 3), face=m.get("face", 0),
                   framing=fr(m, self.fmt), thumb=m.get("thumb"), fx=[], effect=None, **kw)
        if m.get("span"):                                    # the editor's own in-out (picks "add" range)
            out["span"] = m["span"]
            if m.get("whole"):                               # (the best part of a longer one: pick_pieces)
                out["whole"] = m["whole"]
            if m.get("close"):                               # (the reel's close: no quick cut replays it)
                out["close"] = True
        return out

    def pool(self, recs, kinds=None, min_frame=0.0, usable=False):
        out = [(r, m) for r in recs for m in r["moments"] if self.free(r, m) and (kinds is None or m["kind"] in kinds)
               and ((r["role"] == "transition" and not people_broll(r["unit"])) or fr(m, self.fmt) >= min_frame)
               and not (usable and (m.get("unusable") or (m.get("fmt_ok") or {}).get(self.fmt) is False))]
        return sorted(out, key=lambda x: -self.share(*x))

    def best(self, recs, kinds=None, min_frame=0.35, strict=False):
        """The top moment: well framed if possible, of these kinds if possible (strict: of these kinds or none)."""
        p = self.pool(recs, kinds, min_frame, True) or self.pool(recs, kinds, 0.0, True) or self.pool(recs, kinds, min_frame) \
            or self.pool(recs, kinds) or (None if strict else self.pool(recs))     # a shaky / unframeable moment last
        if not p and not strict:                             # every moment used (the editor's picks leave few):
            p = sorted(((r, m) for r in (recs if any(r["moments"] for r in recs) else self.recs)  # the least used
                        for m in r["moments"]), key=lambda x: (len(self.used.get(x[0]["id"], [])), -self.share(*x)))
        return p[0] if p else None


UNFRAMED = 0.15          # portrait: a clip none of whose moments plays above this (assess) need not appear
PERFORMERS = ("band", "singer", "dj", "mc", "violin", "drum", "piano", "keys", "player", "member", "headline")
MEMBERS = []             # reel.json plan.<fmt>.members (portrait_rules() sets it): more performer tags


CLOSE_H = 0.5            # B-roll with someone this tall (of the frame) in most of it is of dancers, not the room


def close_dancers(d, times):
    """B-roll shot among the dancers -- a class floor with people up close, the camera moving with them -- framed
    like a subject in portrait (one dancer whole, followed), not cropped at the room's middle through whoever is
    there."""
    ii = np.clip(np.searchsorted(d["pose_t"], times), 0, len(d["pose_t"]) - 1)
    h = np.nan_to_num((d["boxes"][ii, :, 3] - d["boxes"][ii, :, 1]) / d["ah"]).max(1)
    return bool(len(h)) and float((h >= CLOSE_H).mean()) >= 0.6


def people_broll(u):
    """B-roll whose subject is a person -- a band member, singer, DJ, MC (its tag or id names one) -- not scenery: in
    portrait it is framed like any other subject (who the strip is on, faces, backs, a head cut off, sitting low),
    where venue, lights and crowd wides keep their free pass."""
    if u.get("role") != "transition":
        return False
    words = f"{u.get('tag', '')} {u.get('id', '')}".lower()
    return any(k in words for k in PERFORMERS + tuple(MEMBERS))


def framable(r, fmt):
    """Must this record appear in the reel? Always in landscape; in portrait not when no moment of it frames: the
    subject lost from the strip, a back, a couple cut in half or only a head throughout (assess's plays_p), unless the
    editor picked from it. A clip that won't crop to 9:16 is better left out than shown badly."""
    if any(m.get("pick") or m.get("add") for m in r["moments"]):
        return True
    if any("fmt_ok" in m for m in r["moments"]):              # the scrub: windows that pass every rule (windows.py)
        return any((m.get("fmt_ok") or {}).get(fmt, True) for m in r["moments"])
    if fmt != "portrait" or (r["role"] == "transition" and not people_broll(r.get("unit") or r)):
        return True
    return max((m.get("plays_p", 1.0) for m in r["moments"]), default=1.0) >= UNFRAMED


def cover(P, recs, n, beats, act, cap=4, kinds_pref=None):
    """n shots from recs: every unit once (clips with no shot yet first, then by their best moment as it plays in this
    format -- framing, face and shake are already in its rank; in portrait, not a clip that won't frame: framable),
    then extra shots by fair share, up to cap each."""
    shots, count = [], {}
    order = sorted([r for r in recs if framable(r, P.fmt)], key=lambda r: (P.count.get(r["unit"]["clip"], 0),
                                        -max((rk(m, P.fmt) for m in r["moments"]), default=0)))
    for r in order:
        if len(shots) >= n:
            break
        x = P.best([r], kinds_pref) or P.best([r])
        if x:
            shots.append(P.take(*x, beats=beats, act=act)); count[r["id"]] = 1
    while len(shots) < n:
        added = False
        have = {}
        for s in shots:
            have.setdefault(s["unit"]["id"], set()).add(s["kind"])
        cands = P.pool(recs, None, 0.35) or P.pool(recs)
        cands.sort(key=lambda x: -(P.share(*x) - (0.25 if x[1]["kind"] in have.get(x[0]["id"], ()) else 0)))
        for r, m in cands:
            if count.get(r["id"], 0) < cap and P.free(r, m):
                shots.append(P.take(r, m, beats=beats, act=act)); count[r["id"]] = count.get(r["id"], 0) + 1
                added = True
                break
        if not added:
            break
    return shots


def fit(shots, budget, base):
    """Make the act's shots add up to its beats: drop/add 1-beat changes on the weakest plain shots."""
    used = sum(s["beats"] for s in shots)
    plain = sorted([s for s in shots if not s["fx"]], key=lambda s: s["rank"])
    i = 0
    while used > budget and plain:
        s = plain[i % len(plain)]
        if s["beats"] > 1 and (s["beats"] > base or all(p["beats"] <= base for p in plain)):
            s["beats"] -= 1; used -= 1
        i += 1
        if i > 10 * len(plain):
            break
    best = sorted([s for s in shots if not s["fx"]], key=lambda s: -s["rank"]) or shots
    i = 0
    while used < budget and best:
        best[i % len(best)]["beats"] += 1; used += 1; i += 1
    return shots


def flow_link(a, unit, t, kind, direction):
    """How well a cut from shot a to (unit, t, kind) carries the movement: a turn into a turn, a move into the same
    move, travel continuing the same way across the cut. Roughly -1..2."""
    ka, sc = a["kind"], 0.0
    if ka == kind and ka in ("spin", "pattern"):
        sc += 1.2
    elif ka == kind and ka in MOVES:
        sc += 0.4
    elif ka in MOVES and kind in MOVES:
        sc += 0.2
    da, db = direction(a["unit"], a["t"])[1], direction(unit, t)[0]
    if abs(da) > 0.025 and abs(db) > 0.025:
        sc += 0.8 if (da > 0) == (db > 0) else -0.8
    return sc


def rhythm(budget, pace, energy, build=False, hold_every=4):
    """Shot lengths for an act: one pattern per 8-beat phrase -- busier phrases of the music (energy, per phrase,
    relative to the act) get the quicker patterns, every hold_every-th quieter phrase is held long, the phrase into
    the next act can build -- never the same pattern twice running. A remainder shorter than a phrase closes it."""
    n, rest = divmod(budget, 8)
    med = float(np.median(energy)) if len(energy) else 0
    out, last = [], None
    for k in range(n):
        e = energy[k] if k < len(energy) else med
        if build and k == n - 1:
            pool = PHRASES["build"]
        elif hold_every and k % hold_every == hold_every - 1 and e <= med:
            pool = PHRASES["hold"]
        else:
            pool = PHRASES["fast" if (pace == "fast" or e > 1.1 * med) and pace != "slow" else "medium"]
        options = [p for p in pool if p != last] or pool
        last = options[(k * 7 + int(10 * e)) % len(options)]  # varied, but the same plan every run
        out += last
    out += {0: [], 1: [1], 2: [2], 3: [2, 1], 4: [2, 2], 5: [2, 1, 2], 6: [2, 2, 2], 7: [2, 1, 2, 2]}[rest]
    return out


def phrase_energy(music, a, n):
    """Mean beat accent over each 8-beat phrase from beat a (the act's start); 1.0 without accents."""
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    acc = np.array(music.get("beat_accent") or np.ones(len(bt)))[i0:]
    return [float(acc[a + 8 * k: a + 8 * k + 8].mean()) if a + 8 * k < len(acc) else 1.0 for k in range(n)]


def squeeze(lengths, n):
    """Too few distinct moments for the slots: hold some shots longer, merging the shortest neighbouring pairs --
    not into a run of equal lengths, and sparing the act's closing phrase (it may be a build)."""
    lengths = list(lengths)
    keep = len(lengths) - 6

    def cost(i):
        v = lengths[i] + lengths[i + 1]
        same = (i > 0 and lengths[i - 1] == v) + (i + 2 < len(lengths) and lengths[i + 2] == v)
        return v + 1.5 * same + (3 if i + 1 >= keep else 0)
    while len(lengths) > n > 0:
        i = min(range(len(lengths) - 1), key=cost)
        lengths[i:i + 2] = [lengths[i] + lengths[i + 1]]
    return lengths


def fit_slots(shots, lengths):
    """Give the woven shots the act's lengths, then swap neighbours so moves that need time get the long slots and
    the quick slots go to hits and shakes -- without putting one clip twice in a row."""
    lengths = squeeze(lengths, len(shots))
    for s, n in zip(shots, lengths):
        s["beats"] = n

    def ok(i):
        return all(shots[j]["unit"]["clip"] != shots[i]["unit"]["clip"] for j in (i - 1, i + 1) if 0 <= j < len(shots))
    for i in range(len(shots)):
        if lengths[i] >= 3 and shots[i]["kind"] not in LONG:
            for j in sorted(range(max(0, i - 3), min(len(shots), i + 4)), key=lambda j: abs(j - i)):
                if lengths[j] <= 2 and shots[j]["kind"] in LONG and shots[j]["unit"]["role"] == shots[i]["unit"]["role"]:
                    shots[i], shots[j] = shots[j], shots[i]
                    if ok(i) and ok(j):
                        break
                    shots[i], shots[j] = shots[j], shots[i]
    for s, n in zip(shots, lengths):
        s["beats"] = n
    return shots


def give_slot_fx(shots, name, kinds, n, min_beats=4, max_beats=99):
    """The effect on the n strongest plain shots of these kinds that already sit in a slot of the right length."""
    done = 0
    for i in sorted([i for i, s in enumerate(shots) if s["kind"] in kinds and not s["fx"]
                     and min_beats <= s["beats"] <= max_beats and s["unit"]["role"] != "transition"],
                    key=lambda i: -shots[i]["rank"]):
        if done >= n:
            break
        if any(shots[j]["fx"] for j in (i - 1, i + 1) if 0 <= j < len(shots)):
            continue                                        # one effect at a time: never two in a row
        shots[i]["fx"], shots[i]["effect"] = [name], name
        done += 1


def weave(shots, prev=None, slots=None, direction=None):
    """Order an act's shots like a card shuffle: a clip's shots spread out through the others, never two from one
    clip or two B-roll shots side by side (B-roll on evenly spaced slots, or the given ones), each cut carrying the
    movement on (flow_link), a movement held two or three cuts and then changed."""
    from .flow import still
    direction = direction or still
    rest, out = list(shots), []

    def broll(s):
        return s["unit"]["role"] == "transition"

    def family(s):
        return s["unit"].get("tag", "").split("-")[0]

    def link(c):
        last, sc = (out[-1] if out else prev), 0.0
        if last is not None:
            if last["unit"]["clip"] == c["unit"]["clip"]:
                sc -= 10
            if broll(last) and broll(c):
                sc -= 3
            if not (broll(last) or broll(c)):
                sc += flow_link(last, c["unit"], c["t"], c["kind"], direction)
        if len(out) >= 2 and all(r["kind"] == c["kind"] for r in out[-2:]):
            sc -= 2.0                                        # a movement carries 2-3 cuts, then it changes
        for r in out[-3:]:
            if r["unit"]["clip"] == c["unit"]["clip"]:
                sc -= 1.5
            if broll(r) and broll(c) and family(r) == family(c):
                sc -= 1.5
        # a clip with many shots still to place goes sooner, so its shots spread through the act
        sc += 3 * sum(r["unit"]["clip"] == c["unit"]["clip"] for r in rest) / len(rest)
        return sc
    nb, N = sum(map(broll, shots)), len(shots)
    if slots is None:                                        # B-roll spaced evenly
        slots = {int((k + 0.5) * N / nb) for k in range(nb)} if 0 < nb < N else set()
    while rest:
        want = [c for c in rest if broll(c) == (len(out) in slots)] if slots else rest
        c = max(want or rest, key=link)
        rest.remove(c); out.append(c)
    return out


def broll_slots(lengths, n_shots, nb):
    """Where nb B-roll shots go among n_shots: evenly spaced, on slots of 2+ beats where possible, never touching
    (and never first or last)."""
    longs = [i for i in range(1, n_shots - 1) if lengths[i] >= 2] or list(range(n_shots))
    slots = set()
    for k in range(nb):
        aim = (k + 0.5) * n_shots / nb
        free = [i for i in range(1, n_shots - 1) if not {i - 1, i, i + 1} & slots]
        if free:
            slots.add(min(free, key=lambda i: abs(i - aim) + (0 if i in longs else 1.5)))
    return slots


PLAIN = ("reaction", "ending", "band", "logo", "featured")              # shots a coverage pass never replaces or splits


def cover_all(P, seq, recs, home, confined=False):
    """Every export file in the reel -- a unit, not just its camera clip: a clip the editor split into five social
    segments is five files, and each one appears (in portrait, each one that frames: framable). A unit with no shot takes the place of the weakest plain shot of a
    unit that has several, in the act where it belongs (home: record id -> act; the social floor may also go in the
    climax, unless confined: song sets keep footage to its own songs); with no such shot, a long plain shot there is
    split to make room (more cuts, the grid unchanged). Never beside its own clip, never B-roll beside B-roll."""
    have = {s["unit"]["id"] for s in seq}
    for r in sorted([r for r in recs if r["unit"]["id"] not in have and framable(r, P.fmt)],
                    key=lambda r: -max((rk(m, P.fmt) for m in r["moments"]), default=0)):
        x = P.best([r])
        if not x:
            continue
        act = home.get(r["id"], "night" if r["role"] == "transition" else "climax")
        acts = [act] + (["climax", "night"] if r["role"] == "dance" and not confined else [])
        acts = list(dict.fromkeys(acts))
        clip, units = r["unit"]["clip"], Counter(s["unit"]["id"] for s in seq)

        def fits(i, j, r=r, clip=clip):                      # the new shot at i, between seq[i - 1] and seq[j]
            nb = [seq[k] for k in (i - 1, j) if 0 <= k < len(seq)]
            return all(n["unit"]["clip"] != clip for n in nb) and \
                not (r["role"] == "transition" and any(n["unit"]["role"] == "transition" for n in nb))

        def plain(x):
            return not x["fx"] and not any(x.get(f) for f in PLAIN) and not x.get("add")
        idx = [i for i, x in enumerate(seq) if x["act"] in acts and plain(x) and 0 < i < len(seq) - 1
               and units[x["unit"]["id"]] > 1 and fits(i, i + 1)]
        if idx:
            i = max(idx, key=lambda i: (-acts.index(seq[i]["act"]), units[seq[i]["unit"]["id"]],
                                        P.load(seq[i]["unit"]["clip"]), -seq[i]["rank"]))
            old = seq[i]
            P.count[old["unit"]["clip"]] -= 1
            seq[i] = P.take(*x, beats=old["beats"], act=old["act"])
            continue
        idx = [i for i, x in enumerate(seq) if x["act"] in acts and plain(x) and x["beats"] >= 2
               and 0 < i < len(seq) - 1 and x["unit"]["clip"] != clip and fits(i + 1, i + 1)
               and not (r["role"] == "transition" and x["unit"]["role"] == "transition")]
        if not idx:
            print(f"[reel] {P.fmt}: no room for {r['unit']['id']}")
            continue
        i = max(idx, key=lambda i: (seq[i]["beats"], -acts.index(seq[i]["act"]), -seq[i]["rank"]))
        old = seq[i]
        give = old["beats"] // 2
        old["beats"] -= give
        seq.insert(i + 1, P.take(*x, beats=give, act=old["act"]))


PICK_BEATS = 32          # an editor's selection is one shot, however long (this only guards against a runaway)
PICK_CURVE = 1e9         # (no speed curve: a selection plays at real speed, however long; it was 4 s, at PICK_PACE)
PICK_PACE = {"dance": 1.05, "transition": 1.8}   # this fast: dancing peaks near 1.5x at its ends, B-roll quicker


def plain(x):
    return not x["fx"] and not x.get("reaction") and not x.get("ending") and not x.get("band") \
        and not x.get("logo") and not x.get("add") and not x.get("featured")


def picks_once(seq, beat=None):
    """The editor's picks take precedence: each plays once (a repeat's beats go to the shot before it, or after it at
    an act's start; with picks cut to pieces, absorb keeps them off any one shot), never sped up (slow motion is fine: FAST effects come off). A pick that is also the configured
    ending plays as the ending: its other copies go. Changes seq in place."""
    seen = {(s["unit"]["id"], tuple(s["span"])) for s in seq if s.get("ending") and s.get("span")}
    i = 0
    while i < len(seq):
        s = seq[i]
        if s.get("span"):
            if any(f in FAST for f in s["fx"]):
                s["fx"] = [f for f in s["fx"] if f not in FAST]
                s["effect"] = s["fx"][0] if s["fx"] else None
            k = (s["unit"]["id"], tuple(s["span"]))
            if k in seen and not s.get("ending") and len(seq) > 1:
                j = i - 1 if i > 0 and seq[i - 1]["act"] == s["act"] else i + 1 if i + 1 < len(seq) else i - 1
                if beat:                                     # picks cut to pieces: kept off any one shot
                    seq.pop(i)
                    absorb(seq, s, s["beats"], beat, at=min(i, len(seq) - 1))
                    continue
                seq[j]["beats"] += s["beats"]
                seq.pop(i)
                continue
            seen.add(k)
        i += 1


SPREAD_MAX = 8           # a plain shot takes a pick's spare beats up to this; past it they go to quick cuts


def spread(seq, near, n, cap=None, top=None):
    """n beats given out one at a time to the shots near (nearest first) still under cap beats (PLAIN_HOLD); when all
    are at cap, the cap goes up one, to top at most -- a pick's spare beats never pile onto one shot. Returns the
    beats left over (top reached: none if top is None). Changes seq in place."""
    cap = cap or PLAIN_HOLD
    while n > 0 and near:
        j = next((j for j in near if seq[j]["beats"] < cap), None)
        if j is None:
            if top is not None and cap >= top:
                break
            cap += 1
            continue
        seq[j]["beats"] += 1; n -= 1
    return n


def absorb(seq, s, n, beat, at=None):
    """n spare beats freed beside shot s (an editor's pick, with picks cut to pieces), kept off any one
    shot: spread over the act's plain shots up to SPREAD_MAX, then quick cuts of the picks' unseen parts, then more
    of the act's cut-down picks themselves (back towards the editor's whole in-out, nearest first), and only then
    evenly over the plain shots. at: where s is (or was, once taken out). Changes seq in place."""
    act = s["act"]
    if at is None:
        at = next(k for k, x in enumerate(seq) if x is s)
    order = lambda js: sorted(js, key=lambda j: abs(j - at))
    left = spread(seq, order(j for j in range(len(seq)) if seq[j]["act"] == act and plain(seq[j])), n, top=SPREAD_MAX)
    had = sum(x["beats"] for x in seq)
    for size in (2, 1):
        short = left - (sum(x["beats"] for x in seq) - had)
        if short:
            quick_cuts(seq, short, act, beat, min(at + 1, len(seq)), size=size, hold=False)
    short = left - (sum(x["beats"] for x in seq) - had)
    for j in order(j for j in range(len(seq)) if seq[j]["act"] == act and seq[j].get("whole") and not seq[j].get("reuse")):
        if short <= 0:
            break
        x = seq[j]
        grow = max(0, min(short, int((x["whole"][1] - x["whole"][0]) / beat) - x["beats"]))
        if grow:
            x["beats"] += grow; short -= grow
            a = max(x["whole"][0], min(x["span"][0], x["whole"][1] - x["beats"] * beat))
            x["span"] = [round(a, 3), round(min(x["whole"][1], a + x["beats"] * beat), 3)]
    if short > 0:
        near = order(j for j in range(len(seq)) if seq[j]["act"] == act and plain(seq[j])) or \
            order(j for j in range(len(seq)) if seq[j]["act"] == act)
        spread(seq, near, short)


def fit_pick(seq, i, beat, floor=1):
    """The editor's selection plays as long as they chose (to the nearest beat): its shot gives spare beats to the
    plain shot nearest it, and takes the beats it needs from the weakest plain shots in its act -- condensing them to
    2 beats, then to 1, then dropping the weakest whose unit is in the reel elsewhere; the act's best shots (top
    quarter by rank) are never dropped. The act's grid stays exact. floor 2 (picks cut to pieces): no plain shot goes
    under 2 beats -- the weakest go instead, a clip's only shot too when there is no other."""
    s = seq[i]
    if not s.get("span") or not beat:
        return
    n = s["span"][1] - s["span"][0]
    if n > PICK_CURVE:                                       # a long one: all of it, on a speed curve
        n /= PICK_PACE.get(s["unit"]["role"], PICK_PACE["dance"])
        s["fx"], s["effect"] = ["fit"], "fit"
    want = max(1, min(PICK_BEATS, round(n / beat)))
    while s["beats"] != want:
        i = next(k for k, x in enumerate(seq) if x is s)
        near = sorted((j for j in range(len(seq)) if j != i and seq[j]["act"] == s["act"] and plain(seq[j])),
                      key=lambda j: abs(j - i))
        d = want - s["beats"]
        if not near:
            return
        if d < 0 and floor < 2:
            seq[near[0]]["beats"] -= d; s["beats"] += d
            return
        if d < 0:                                            # picks cut to pieces: its spare beats kept off any
            s["beats"] += d                                  # one shot (absorb)
            absorb(seq, s, -d, beat)
            return
        weak = sorted(near, key=lambda j: seq[j].get("rank", 0))
        j = next((j for j in weak if seq[j]["beats"] > 2), None)
        if j is None and floor < 2:
            j = next((j for j in weak if seq[j]["beats"] == 2), None)
        if j is not None:
            g = min(seq[j]["beats"] - (2 if seq[j]["beats"] > 2 else 1), d)
            seq[j]["beats"] -= g; s["beats"] += g
            continue
        units = Counter(x["unit"]["id"] for x in seq)
        top = np.quantile([seq[j].get("rank", 0) for j in near], 0.75)
        j = next((j for j in weak if units[seq[j]["unit"]["id"]] > 1 and 0 < j < len(seq) - 1 and seq[j].get("rank", 0) <= top),
                 None)
        if j is None and floor >= 2:
            j = next((j for j in weak if 0 < j < len(seq) - 1 and seq[j].get("rank", 0) <= top), None)
        if j is None:
            return
        s["beats"] += seq.pop(j)["beats"]


def pick_beat(recs, beat, room):
    """The beat the editor's selections are measured in: the real one when they fit the room between the teaser and
    the finale; when together they are longer, a longer one, so each plays the same share of its length (from its
    in-point) and all of them get in."""
    need = sum(max(1, round((m["span"][1] - m["span"][0]) / beat)) for r in recs for m in r["moments"]
               if m.get("add") and m.get("span"))
    if not need or need <= room:
        return beat
    k = max(room, 1) / need
    print(f"[reel] picks: {need} beats of selections for {room} beats of room: each plays {k:.0%} of its length")
    return beat / k


def settle_picks(seq, beat, budget, real):
    """When the editor's picks fill the reel (pick_beat), each act gives every pick its length in beat (the scaled
    beat) and the act's other shots make up the difference: plain shots shorten (weakest first), then go, then the
    longest picks give a beat each; spare beats go to plain shots, else to the picks cut shortest. Keeps each act
    on its budget, so the acts stay on the song's sections."""
    for act, room in budget.items():
        mine = [s for s in seq if s["act"] == act]
        picks = [s for s in mine if s.get("span")]
        if not picks:
            continue
        for s in picks:
            s["beats"] = max(1, min(PICK_BEATS, round((s["span"][1] - s["span"][0]) / beat)))
        over = sum(s["beats"] for s in mine) - room
        spare = sorted((s for s in mine if plain(s) and not s.get("span")), key=lambda s: s.get("rank", 0))
        for s in spare:
            while over > 0 and s["beats"] > 1:
                s["beats"] -= 1; over -= 1
        for s in spare:
            if over <= 0:
                break
            k = seq.index(s)
            if 0 < k < len(seq) - 1 and not s.get("open"):
                over -= s["beats"]
                seq.remove(s)
        while over > 0:
            s = max(picks, key=lambda s: s["beats"])
            if s["beats"] <= 1:
                break
            s["beats"] -= 1; over -= 1
        keep = [s for s in spare if s in seq]
        for s in keep:                                       # spare beats: the act's own shots first,
            if over >= 0:
                break
            s["beats"] += 1; over += 1
        while over < 0:                                      # then its picks, up to their full length,
            s = min(picks, key=lambda s: s["beats"] * real / (s["span"][1] - s["span"][0]))
            if s["beats"] >= max(1, math.floor((s["span"][1] - s["span"][0]) / real)):
                break
            s["beats"] += 1; over += 1
        if over < 0:                                         # then quick cuts from the picks
            quick_cuts(seq, -over, act, real)


def unclump_clips(seq):
    """No clip twice in a row: a shot that follows its own clip swaps with the nearest shot of the same act that
    fits there (and leaves no new pair where it came from), or else moves to the nearest gap in its act between two
    other clips. The first and last shots stay. Changes seq in place."""
    clip = lambda k: seq[k]["unit"]["clip"] if 0 <= k < len(seq) else None
    for i in range(1, len(seq) - 1):
        if clip(i) != clip(i - 1):
            continue
        for j in sorted(range(1, len(seq) - 1), key=lambda j: abs(j - i)):
            if j == i or seq[j]["act"] != seq[i]["act"] or abs(j - i) == 1:
                continue
            if clip(j) not in (clip(i - 1), clip(i + 1)) and clip(i) not in (clip(j - 1), clip(j + 1)):
                seq[i], seq[j] = seq[j], seq[i]
                break
        else:                                                # no swap: move it to a gap in its act instead
            s = seq.pop(i)
            gaps = [g for g in range(1, len(seq)) if seq[g - 1]["act"] == s["act"] == seq[g]["act"]
                    and s["unit"]["clip"] not in (clip(g - 1), clip(g))]
            seq.insert(min(gaps, key=lambda g: abs(g - i)) if gaps else i, s)


FINALE_HOLD = 4          # a plain shot in the finale holds at most this long when there are picks to cut in
CLOSE_BEATS = 4          # the closing shot holds at least this long (when its footage allows)


def close_last(seq, beat, close):
    """The pick named by close ([clip, source time], reel.json plan.<fmt> "close") ends the reel when the picks don't
    fill it (finale_reuse does it when they do): moved to the end, its old place quick cuts, held at real speed for
    all its footage up to 2 x CLOSE_BEATS. The beats it adds come off the plain shots afterwards (the grid's
    evening out). Returns whether it was found."""
    mine = [s for s in seq if s.get("span") and not s.get("reuse") and s["unit"]["id"].startswith(close[0])
            and s["span"][0] - 0.5 <= close[1] <= s["span"][1]]
    want = max(mine, key=lambda s: s["span"][1] - s["span"][0], default=None)   # the pick itself, not a cut of it
    if want is None:
        return False
    want["close"] = True                                     # (quick cuts never take its footage)
    k = seq.index(want)
    if k != len(seq) - 1:
        seq.remove(want)
        want["reuse"] = True                                 # (none of the cuts from its own footage)
        quick_cuts(seq, want["beats"], want["act"], beat, k, avoid=[want])
        want["reuse"] = None
        seq.append(want)
    want["act"] = "finale"
    want["beats"] = min(max(1, math.floor((want["span"][1] - want["span"][0]) / beat + 0.25)), 2 * CLOSE_BEATS)
    want["fx"], want["effect"] = [], None
    return True


def finale_reuse(seq, beat, close=None):
    """When the editor's picks fill the reel, the finale is rebuilt on its budget: its picks in their order, then
    quick cuts from the picks (quick_cuts, 2 beats each: the song's outro), and a closing shot held as long as its
    footage lasts, up to 2 x CLOSE_BEATS -- the pick named by close ([clip, source time], reel.json plan.<fmt>
    "close") wherever the planner put it, else the longest crowd or DJ pick in the finale with CLOSE_BEATS of footage
    or more (else the longest pick)."""
    fin = [s for s in seq if s["act"] == "finale"]
    if not fin:
        return
    full = lambda s: max(1, math.floor((s["span"][1] - s["span"][0]) / beat + 0.25))

    def to_end(mv):                                          # (its old place becomes quick cuts)
        k, gap, act = seq.index(mv), mv["beats"], mv["act"]
        seq.remove(mv)
        mv["reuse"] = True                                   # (none of the cuts from its own footage)
        quick_cuts(seq, gap, act, beat, k, avoid=[mv])
        mv["reuse"], mv["beats"], mv["act"] = None, 0, "finale"
        last = [s for s in seq if s["act"] == "finale"][-1]
        seq.insert(seq.index(last) + 1, mv)

    want = next((s for s in seq if close and s.get("span") and not s.get("reuse")
                 and s["unit"]["id"].startswith(close[0]) and s["span"][0] - 0.5 <= close[1] <= s["span"][1]), None)
    if want is not None:
        if want["act"] != "finale":
            to_end(want)
        want["reuse"] = True                                 # the last shot is seen once: no quick cut shows it
        for r in [r for r in seq if r.get("reuse") and r is not want and r["unit"]["id"] == want["unit"]["id"]
                  and r["span"][0] < want["span"][1] and r["span"][1] > want["span"][0]]:
            k = seq.index(r)
            seq.remove(r)
            quick_cuts(seq, r["beats"], r["act"], beat, k)
        want["reuse"] = None
    # no pick in the finale long enough to end on: the longest of the climax's moves to the end
    elif not any(s.get("span") and not s.get("reuse") and full(s) >= CLOSE_BEATS for s in fin):
        late = [s for s in seq if s["act"] == "climax" and s.get("span") and not s.get("reuse") and not s["fx"]
                and full(s) >= CLOSE_BEATS]
        if late:
            to_end(max(late, key=lambda s: (s["unit"]["role"] == "transition", full(s))))
    fin = [s for s in seq if s["act"] == "finale"]
    room, at = sum(s["beats"] for s in fin), seq.index(fin[0])
    picks = [s for s in fin if s.get("span") and not s.get("reuse")]
    for s in fin:
        seq.remove(s)
    close = want if want is not None else \
        (sorted((s for s in picks if full(s) >= CLOSE_BEATS), key=lambda s: (s["unit"]["role"] != "transition", -full(s)))
         or sorted(picks, key=lambda s: -full(s)) or [None])[0]
    keep = [s for s in picks if s is not close]
    if close is not None:
        close["beats"] = min(full(close), 2 * CLOSE_BEATS, room)   # all of it, up to twice the least
        close["fx"], close["effect"] = [], None             # (held to the end, at real speed)
    left = room - (close["beats"] if close else 0) - sum(s["beats"] for s in keep)
    while left < 0 and keep:                                 # (the picks first, as long as they can stay)
        s = max(keep, key=lambda s: s["beats"])
        if s["beats"] > 1:
            s["beats"] -= 1; left += 1
        else:
            keep.remove(s); left += 1
    seq[at:at] = keep + ([close] if close else [])
    if left > 0:
        quick_cuts(seq, left, "finale", beat, at + min(1, len(keep)), size=2)


QUICK_DROP = ("beats", "act", "fx", "effect", "logo", "logo_image", "band", "reaction", "ending", "lead_beats",
              "featured", "share", "add", "open", "person", "span", "whole", "reuse")   # a quick cut is footage, not a role


ENDING_GAP = 4.0         # the closing shot's footage, this far either side, is never cut in earlier (reserve_ending)


def quick_cuts(seq, n, act, beat, at=None, size=1, hold=True, avoid=()):
    """n beats of quick cuts, size beats each, from the editor's picks, put into act at at (default: spread through
    it, never at its ends): first the ends that pick_beat trimmed off them (footage not seen yet), then the openings
    of picks two acts or more away (never footage seen moments ago); daytime footage (unit "day") never into a night
    act, and the day's own picks first in the day (DAY_ACTS); crowd and B-roll picks first, never the same clip twice
    in a row; never the closing shot's footage (the ending or close, and avoid: one taken out of seq to move it,
    ENDING_GAP either side: it is seen once, at the end, even where another pick holds the same moment). A cut is
    plain footage: no logo, title, effect or ending from the pick it comes from."""
    picks = [s for s in seq if s.get("span") and not s.get("reuse") and not s.get("close") and not s.get("ending")]
    if not n or not picks:
        return
    used = {(s["unit"]["id"], round(s["span"][0], 2)) for s in seq if s.get("reuse")}
    last_shots = [(s["unit"]["id"], s["span"]) for s in [*seq, *avoid]
                  if (s.get("ending") or s.get("close") or any(s is x for x in avoid)) and s.get("span")]
    names = [a for a in ACT_NAMES if any(x["act"] == a for x in seq)]
    near = set(names[max(0, names.index(act) - 1):names.index(act) + 1]) if act in names else {act}
    is_day = lambda s: s["unit"].get("day", s["act"] in DAY_ACTS)
    if act not in DAY_ACTS:                                  # daytime footage never cuts into the night
        picks = [s for s in picks if not is_day(s)]
    order = sorted(picks, key=lambda s: (is_day(s) != (act in DAY_ACTS),              # day into the day first
                                         s["unit"]["role"] != "transition", s["act"] != act,
                                         -(s["span"][1] - s["span"][0])))
    tails = [(s, [s["span"][0] + s["beats"] * beat, s["span"][1]]) for s in order]
    shown = [(x["unit"]["id"], x["span"]) for x in seq if x.get("span")]
    rest = [(s, sp) for s in order if s.get("whole")                 # a pick cut to its best part: the rest of it,
            for sp in ([s["whole"][0], s["span"][0]], [s["span"][1], s["whole"][1]])   # what isn't in the reel yet
            if sp[1] - sp[0] >= 0.8 * beat * size
            and not any(u == s["unit"]["id"] and min(b, sp[1]) - max(a, sp[0]) > 0.1 for u, (a, b) in shown)]
    srcs = rest + [(s, sp) for s, sp in tails if sp[1] - sp[0] >= 0.8 * beat * size] + \
        [(s, [s["span"][0], min(s["span"][1], s["span"][0] + beat * size)]) for s in order
         if s["act"] not in near and s["span"][1] - s["span"][0] >= 0.8 * beat * size]
    srcs = [(s, sp) for s, sp in srcs if (s["unit"]["id"], round(sp[0], 2)) not in used
            and not any(u == s["unit"]["id"] and sp[0] < b + ENDING_GAP and sp[1] > a - ENDING_GAP for u, (a, b) in last_shots)]
    if at is None:
        idx = [k for k, s in enumerate(seq) if s["act"] == act]
        if not idx:
            return
        at = idx[len(idx) // 2] + 1 if len(idx) > 1 else idx[0]
    out, last = [], seq[at - 1]["unit"]["clip"] if at else None
    while sum(c["beats"] for c in out) < n and srcs:
        k = next((k for k, (s, _) in enumerate(srcs) if s["unit"]["clip"] != last), 0)
        s, sp = srcs.pop(k)
        c = {key: v for key, v in s.items() if key not in QUICK_DROP}
        c.update(span=[round(sp[0], 3), round(sp[1], 3)], t=round((sp[0] + sp[1]) / 2, 2),
                 beats=min(size, n - sum(c["beats"] for c in out)), act=act, fx=[], effect=None, reuse=True)
        out.append(c)
        last = s["unit"]["clip"]
    seq[at:at] = out
    left = n - sum(c["beats"] for c in out)                   # fewer sources than beats: the cuts and the plain
    if out and left and hold:                                 # shots round them share the rest, none piling up
        mine = {id(c) for c in out}
        near = sorted((j for j in range(len(seq)) if seq[j]["act"] == act and (id(seq[j]) in mine or plain(seq[j]))),
                      key=lambda j: abs(j - at))
        spread(seq, near, spread(seq, near, left, top=SPREAD_MAX))   # (past the top: still evenly)


def spread_picks(seq, recs, home, beat, budget, full=0.9):
    """Where the editor's selections would overfill their act (more of their beats than the act has, so each would
    be cut to a beat) and a later act has room -- above all an act the planner left empty, its footage all picks --
    the act's last selections (by source time) move there: {(record id, moment time): act}. B-roll stays home,
    and a day act's picks (DAY_ACTS) never move into the night.
    full: the share of an act's beats its picks may fill (less, for picks cut to pieces: room for other footage)."""
    if not beat or not budget:
        return {}
    shown = {(x["unit"]["id"], tuple(x["span"])) for x in seq if x.get("span")}
    want = lambda m: max(1, min(PICK_BEATS, round((m["span"][1] - m["span"][0]) / beat)))
    todo = [(r, m) for r in recs for m in r["moments"] if m.get("add") and m.get("span")
            and (r["unit"]["id"], tuple(m["span"])) not in shown]
    act_of = lambda r: home.get(r["id"], "night" if r["role"] == "transition" else "climax")
    load = Counter()
    for r, m in todo:
        load[act_of(r)] += want(m)
    for x in seq:
        if x.get("span"):
            load[x["act"]] += x["beats"]
    order = [a for a in ACT_NAMES if a not in ("teaser", "finale")]
    out = {}
    for a in order:
        mine = sorted([(r, m) for r, m in todo if act_of(r) == a and r["role"] != "transition"],
                      key=lambda rm: -rm[1]["t"])
        for r, m in mine:
            if load[a] <= budget.get(a, 0) * full:
                break
            later = [b for b in order[order.index(a) + 1:] if load[b] + want(m) <= budget.get(b, 0) * full
                     and not (a in DAY_ACTS and b not in DAY_ACTS)]      # day footage never into the night
            if not later:
                break
            b = min(later, key=lambda b: load[b] / max(budget[b], 1))
            out[(r["id"], m["t"])] = b
            load[a] -= want(m); load[b] += want(m)
    return out


def home_picks(seq, recs, home):
    """An editor's pick in an act its footage may not play in (daytime footage in a night act, night footage in the
    arrival) -- whichever pass put it there (the night's B-roll takes from every pick) -- moves to the end of its home
    act (home: record id -> act); balance_picks then fits the acts to their beats. Changes seq in place."""
    if not any(r["unit"].get("day") for r in recs):        # no daytime footage: the arrival plays what there is
        return
    hm = {r["unit"]["id"]: home.get(r["id"]) for r in recs}
    for s in [s for s in seq if s.get("span") and s.get("add") and not s.get("reuse")]:
        day, a = bool(s["unit"].get("day")), s["act"]
        if a == "teaser" or day == (a in DAY_ACTS) or not (day or a == "arrival"):
            continue
        h = hm.get(s["unit"]["id"])
        if h not in ACT_NAMES or h == a:
            continue
        seq.remove(s)
        k = max((i for i, x in enumerate(seq) if ACT_NAMES.index(x["act"]) <= ACT_NAMES.index(h)), default=-1)
        s["act"] = h
        seq.insert(k + 1, s)


def balance_picks(seq, beat, budget):
    """The editor's picks take precedence: every selection gets its full length (to the nearest beat), and an act
    they overfill passes its last shot on into the next act while the acts after it have beats to spare (the order
    stays; the act lines move), and otherwise makes room: its other shots shorten to 2 beats, then 1, then go
    (weakest first; never the opening, the ending or the logo). The finale's ending is never pushed past; the grid is
    evened out after; an act still short gets quick cuts from the picks. Changes seq in place."""
    if not beat or not budget:
        return
    for x in seq:
        if x.get("span") and not x.get("ending") and not x.get("open"):
            x["beats"] = max(1, min(PICK_BEATS, round((x["span"][1] - x["span"][0]) / beat)))
    keep = ("ending", "open", "logo", "span")
    order = [a for a in ACT_NAMES if a != "teaser"]
    used = lambda a: sum(x["beats"] for x in seq if x["act"] == a)
    for a, b in zip(order, order[1:] + [None]):
        while True:
            mine = [x for x in seq if x["act"] == a]
            over = sum(x["beats"] for x in mine) - budget.get(a, 0)
            if over <= 0:
                break
            later = order[order.index(a) + 1:]
            spare = sum(budget.get(c, 0) - used(c) for c in later)
            stays = lambda x, a=a, b=b: x["unit"].get("day") and a in DAY_ACTS and b not in DAY_ACTS   # daytime footage
            last = next((x for x in reversed(mine) if not stays(x)), mine[-1])                   # never into the night
            if stays(last):
                b = None
            if b and spare > 0 and not any(last.get(k) for k in ("ending", "open", "logo")) \
                    and (spare >= last["beats"] or over >= last["beats"]):
                if last is not mine[-1]:                     # (a daytime shot after it stays: it moves to the
                    seq.remove(last)                          # act's end first, so the acts stay in one piece)
                    seq.insert(seq.index(mine[-1]) + 1, last)
                last["act"] = b
                continue
            other = sorted((x for x in mine if not any(x.get(k) for k in keep)), key=lambda x: x.get("rank", 0))
            x = next((x for x in other if x["beats"] > 2), None) or next((x for x in other if x["beats"] == 2), None)
            if x is not None:
                x["beats"] -= 1
                continue
            if other:
                seq.remove(other[0])
                continue
            if b is None or (b == "finale" and any(y.get("ending") for y in mine[-1:])):
                break
            mine[-1]["act"] = b
    for a in order:                                          # an act left short (its picks were trimmed back to
        n = budget.get(a, 0) - used(a)                        # their length, or passed on): the picks again, cut
        if n > 0 and any(x["act"] == a for x in seq):        # down, 2 beats each
            quick_cuts(seq, n, a, beat, size=2)


def include_adds(P, seq, recs, home, beat=None, budget=None, insert=False):
    """The editor's extra shots (picks "add") are in the reel: each one not already shown takes the place of the
    weakest plain shot in its act (home: record id -> act; spread_picks moves some where their act is full), never
    beside its own clip -- or, with no such shot, splits the longest plain one; in an act with no shots at all it
    goes on at that act's end, at its full length. A selection (an in-out) runs its own length (fit_pick).
    insert (picks cut to pieces, pick_pieces): a selection goes in between two shots of its act instead, as far from
    the act's other picks as it can (so picks and the rest of the footage take turns), and its beats come from the
    act's weakest plain shots."""
    moved = spread_picks(seq, recs, home, beat, budget, PIECE_FULL if insert else 0.9)
    has_day = any(r["unit"].get("day") for r in recs)
    for r in recs:
        for m in [m for m in r["moments"] if m.get("add")]:
            near = [k for k, x in enumerate(seq) if x["unit"]["id"] == r["unit"]["id"] and abs(x["t"] - m["t"]) < 1.5
                    and not (m.get("span") and x.get("span") and x["span"] != m["span"])]
            if near:
                if m.get("span"):
                    seq[near[0]].update(span=m["span"], add=True)
                    fit_pick(seq, near[0], beat, floor=2 if insert else 1)
                continue
            act = moved.get((r["id"], m["t"])) or home.get(r["id"], "night" if r["role"] == "transition" else "climax")
            if not any(x["act"] == act and plain(x) for x in seq) and act in ACT_NAMES:
                k = ACT_NAMES.index(act)                     # an act of only picks (or none): on at its end
                at = next((i for i, x in enumerate(seq) if x["act"] in ACT_NAMES[k + 1:]), len(seq))
                n = (m["span"][1] - m["span"][0]) / beat if m.get("span") and beat else 2
                seq.insert(at, P.take(r, m, beats=max(1, min(PICK_BEATS, round(n))), act=act))
                seq[at]["add"] = True
                continue
            clip, units = r["unit"]["clip"], Counter(x["unit"]["id"] for x in seq)
            if insert and m.get("span"):
                gaps = [j for j in range(len(seq) - 1) if seq[j]["act"] == act == seq[j + 1]["act"]
                        and clip not in (seq[j]["unit"]["clip"], seq[j + 1]["unit"]["clip"])
                        and not (r["role"] == "transition" and "transition" in (seq[j]["unit"]["role"],
                                                                                seq[j + 1]["unit"]["role"]))]
                if gaps:
                    at = [k for k, x in enumerate(seq) if x.get("span")]
                    gap = lambda j, at=at: min((abs(j + 0.5 - k) for k in at), default=99)
                    j = max(gaps, key=lambda j: (min(gap(j), 3), plain(seq[j]) + plain(seq[j + 1]), gap(j)))
                    seq.insert(j + 1, P.take(r, m, beats=1, act=act))
                    seq[j + 1]["add"] = True
                    fit_pick(seq, j + 1, beat, floor=2)
                    continue

            def fits(i, act=act, clip=clip, units=units, r=r, lone=False):
                return (plain(seq[i]) and 0 < i < len(seq) - 1 and (lone or units[seq[i]["unit"]["id"]] > 1)
                        and clip not in (seq[i - 1]["unit"]["clip"], seq[i + 1]["unit"]["clip"])
                        and not (r["role"] == "transition" and "transition" in (seq[i - 1]["unit"]["role"],
                                                                                seq[i + 1]["unit"]["role"])))
            day = bool(r["unit"].get("day"))                 # elsewhere only where its footage may play: day
            may = lambda a, day=day, act=act: a != "teaser" and (   # footage in the day, night never in the
                not has_day or day == (a in DAY_ACTS) or not (day or a == "arrival")) and (   # arrival, and never
                a not in ACT_NAMES or act not in ACT_NAMES or ACT_NAMES.index(a) >= ACT_NAMES.index(act))   # before its act
            # a pick comes before "use everything": with no other place free, it takes a clip's only shot
            idx = next((ix for ix in (                       # (its own act first, even a clip's only shot)
                [i for i, x in enumerate(seq) if x["act"] == act and fits(i)],
                [i for i, x in enumerate(seq) if x["act"] == act and fits(i, lone=True)],
                [i for i, x in enumerate(seq) if may(x["act"]) and fits(i)],
                [i for i, x in enumerate(seq) if may(x["act"]) and fits(i, lone=True)]) if ix), [])
            if idx:
                i = max(idx, key=lambda i: (seq[i]["act"] == act, P.load(seq[i]["unit"]["clip"]), -seq[i]["rank"]))
                old = seq[i]
                P.count[old["unit"]["clip"]] -= 1
                seq[i] = P.take(r, m, beats=old["beats"], act=old["act"])   # (the act's grid stays exact)
            else:                                            # split the longest plain shot that is not beside its clip
                long = [i for i, x in enumerate(seq) if plain(x) and x["beats"] >= 3 and 0 < i < len(seq) - 1
                        and x["act"] != "finale" and may(x["act"]) and clip not in (x["unit"]["clip"],
                                                                                     seq[i + 1]["unit"]["clip"])]
                if long:
                    j = max(long, key=lambda i: seq[i]["beats"])
                    seq[j]["beats"] -= 1
                else:                                        # no plain shot left to give way: in after any shot
                    gaps = [j for j in range(len(seq) - 1) if may(seq[j]["act"])   # of its act not beside
                            and clip not in (seq[j]["unit"]["clip"], seq[j + 1]["unit"]["clip"])]   # its clip;
                    gaps = [j for j in gaps if seq[j]["act"] == act] or gaps   # the grid is evened out at the end
                    gaps = gaps or [j for j in range(len(seq) - 1) if may(seq[j]["act"])]   # (beside its clip
                    if not gaps:                                                           # before left out)
                        print(f"[reel] no room for the pick {r['unit']['id']} {m['t']:.1f}")
                        continue
                    j = gaps[len(gaps) // 2]
                i = j + 1
                seq.insert(i, P.take(r, m, beats=1, act=seq[j]["act"]))
            seq[i]["add"] = True
            fit_pick(seq, i, beat, floor=2 if insert else 1)       # (picks cut to pieces: absorb their spare)


SPOT_MIN = 0.35          # cover_people never takes a spot that plays worse than this: any one of assess's penalties
KISS_Q = 0.92            # SigLIP kiss confidence that makes a kiss a must-have
LINGER = 2.5             # seconds the camera stays on someone (people.linger): filmed on purpose, placed first
SEEN = 0.045             # a face this big (fraction of frame height) counts as the person being in the shot


def split_for(seq, cover, count, shows, items, act, clip, role="dance", skip=()):
    """A shot to split for an extra one after it: long enough (4+ beats, so both keep 2; 3+ in the climax),
    plain, and still showing everyone it is the only shot of once it is shorter. The longest in the act first, then
    any act but the teaser, the finale and the acts in skip."""
    def ok(i):
        x = seq[i]
        need = 3 if x["act"] == "climax" else 4                    # both halves keep 2 beats (climax: 1 for the insert)
        if (x["beats"] < need or x["fx"] or x.get("ending") or x.get("logo") or x.get("band") or x.get("reaction")
                or x.get("span") or x.get("featured")
                or not 0 < i < len(seq) - 1 or clip in (x["unit"]["clip"], seq[i + 1]["unit"]["clip"])
                or (role == "transition" and "transition" in (x["unit"]["role"], seq[i + 1]["unit"]["role"]))):
            return False                                         # (no B-roll beside B-roll)
        short = dict(x, beats=x["beats"] - (2 if x["beats"] >= 4 else 1))
        return all(count[c] > 1 or shows(short, items[c]) for c in cover[i])
    idx = [i for i in range(len(seq)) if seq[i]["act"] == act and ok(i)] or \
        [i for i in range(len(seq)) if seq[i]["act"] not in ("teaser", "finale", *skip) and ok(i)]
    return max(idx, key=lambda i: (seq[i]["act"] == act, seq[i]["beats"])) if idx else None


def cover_people(P, seq, recs, home, people, fmt, beat, climax_from=None, allowed=None, rate=None):
    """Everyone in it (people.json: the people step): every kiss, then every person not yet seen in a shot. The
    moment that shows the most people still missing goes in first (a group beats a single face), in place of the
    weakest plain shot in its act -- never a shot that is someone's only appearance, never one that would take a
    unit out of the reel, never beside its own clip -- or, when there is no such shot, by splitting a long one
    (split_for). A person is in a shot when their face is big enough inside the shot's stretch (and, in portrait,
    inside the 9:16 strip). climax_from (plan.CFG): dancing from other categories stays out of the climax unless it
    belongs there; allowed(rec, act) (song sets) says which acts a record may go in. Returns (people shown, people,
    kisses shown, kisses)."""
    if not people:
        return None
    by = {r["unit"]["id"]: r for r in recs if r["moments"]}
    half_w = 0.28 if fmt == "portrait" else 0.5

    def window(x):
        h = x["beats"] * beat / 2 + 0.25
        return x["unit"]["id"], x["t"] - h, x["t"] + h

    def shows(x, item):
        uid, a, b = window(x)
        if item[0] == "kiss":
            return item[1] == uid and a <= item[2] <= b
        return any(ap["unit"] == uid and a <= ap["t"] <= b and ap["h"] >= SEEN and abs(ap["cx"] - 0.5) <= half_w
                   for ap in item[1]["appearances"])

    def featured(it):
        return it[0] == "person" and it[1].get("linger", 0) >= LINGER

    kisses = [("kiss", u, t) for u, v in (people.get("kisses") or {}).items() for t, q in v if q >= KISS_Q]
    persons = [("person", p) for p in people.get("people", [])]
    items = kisses + persons
    cover = [[j for j, it in enumerate(items) if shows(x, it)] for x in seq]
    count = np.zeros(len(items), int)
    for c in cover:
        count[c] += 1
    # every moment a shot could be made of: kisses, and each person's clear, framed appearances (half-second grid)
    spots = {(u, t) for _, u, t in kisses if u in by}          # (a portrait-only segment isn't in landscape)
    for _, p in persons:
        spots |= {(a["unit"], round(a["t"] * 2) / 2) for a in p["appearances"]
                  if a["unit"] in by and a["h"] >= SEEN * 1.3 and abs(a["cx"] - 0.5) <= half_w}

    def band_ok(sp):                              # in band footage, a shot of someone reads as a band shot: only
        r = by.get(sp[0])                         # where a player is on screen and in focus (plan.assess)
        if not r or "band_ok" not in r:
            return True
        t, ok = r["band_ok"]
        w = (t >= sp[1] - 0.4) & (t <= sp[1] + 1.2)
        return bool(w.any() and ok[w].mean() >= 0.6)
    spots = {sp for sp in spots if band_ok(sp)        # nor the editor's banned stretches -- the rest of a picked
             and not in_bans(sp[1] - beat, 2 * beat, by[sp[0]]["unit"])}     # clip is banned (apply_picks)

    probe = {sp: [k for k, it in enumerate(items) if shows(dict(unit=dict(id=sp[0]), t=sp[1], beats=2), it)]
             for sp in spots}
    tried = set()
    while True:
        gain = {sp: [k for k in ks if not count[k]] for sp, ks in probe.items()
                if sp not in tried and not any(x["unit"]["id"] == sp[0] and abs(x["t"] - sp[1]) < 0.6
                                               for x in seq)}     # (not a moment a shot already plays)
        gain = {sp: ks for sp, ks in gain.items() if ks}
        if not gain:
            break
        sp = max(gain, key=lambda sp: (any(items[k][0] == "kiss" for k in gain[sp]),       # kisses, then the people
                                       sum(featured(items[k]) for k in gain[sp]),           # the camera stayed on,
                                       len(gain[sp])))                                      # then anyone
        tried.add(sp)
        r, t = by[sp[0]], sp[1]
        if rate is not None and rate(r, t) < SPOT_MIN:       # someone walking, a crowd crossing the crop: their
            continue                                         # next-best spot instead
        near = min(r["moments"], key=lambda m: abs(m["t"] - t))
        m = dict(near, t=t, pick=True)
        act = home.get(r["id"], "night" if r["role"] == "transition" else "climax")
        clip = r["unit"]["clip"]
        units = {}
        for x in seq:
            units[x["unit"]["id"]] = units.get(x["unit"]["id"], 0) + 1

        skip = ("climax",) if (climax_from and act != "climax" and r["role"] != "transition"
                               and not any(k in r.get("category", "") for k in climax_from)) else ()
        skip = (*skip, *(a for a in ACT_NAMES if (a in DAY_ACTS) != (act in DAY_ACTS)))   # day stays in the day
        if allowed:
            skip = (*skip, *(a for a in ACT_NAMES if not allowed(r, a)))
            if act in skip:
                act = next((a for a in ACT_NAMES if a not in skip), act)
        trade = any(featured(items[k]) for k in gain[sp])        # someone filmed on purpose may displace passers-by

        def ok(i, r=r, clip=clip, units=units, trade=trade):
            x = seq[i]
            return (not x["fx"] and not x.get("reaction") and not x.get("ending") and not x.get("band")
                    and not x.get("featured")
                    and not x.get("logo") and not (x.get("add") and not x.get("person")) and 0 < i < len(seq) - 1
                    and units[x["unit"]["id"]] > 1
                    and all(count[c] > 1 or (trade and items[c][0] == "person" and not featured(items[c]))
                            for c in cover[i])
                    and clip not in (seq[i - 1]["unit"]["clip"], seq[i + 1]["unit"]["clip"])
                    and not (r["role"] == "transition" and "transition" in (seq[i - 1]["unit"]["role"],
                                                                            seq[i + 1]["unit"]["role"])))
        idx = [i for i in range(len(seq)) if seq[i]["act"] == act and ok(i)] or \
            [i for i in range(len(seq)) if seq[i]["act"] not in ("teaser", "finale", *skip) and ok(i)]
        if idx:                                                  # take a plain shot's place
            i = max(idx, key=lambda i: (seq[i]["act"] == act, P.load(seq[i]["unit"]["clip"]), -seq[i]["rank"]))
            old = seq[i]
            P.count[old["unit"]["clip"]] -= 1
            count[cover[i]] -= 1
            seq[i] = P.take(r, m, beats=old["beats"], act=old["act"])   # (the grid stays exact)
        else:                                                    # or split one: it keeps most of its beats
            i = split_for(seq, cover, count, shows, items, act, clip, r["role"], skip)
            if i is None:
                continue
            old = seq[i]
            give = 2 if old["beats"] >= 4 else 1
            count[cover[i]] -= 1
            old["beats"] -= give
            cover[i] = [k for k, it in enumerate(items) if shows(old, it)]
            count[cover[i]] += 1
            i += 1
            seq.insert(i, P.take(r, m, beats=give, act=old["act"]))
            cover.insert(i, [])
        seq[i]["add"] = True
        seq[i]["person"] = True
        cover[i] = [k for k, it in enumerate(items) if shows(seq[i], it)]
        count[cover[i]] += 1
    nk = len(kisses)
    feat = [k for k, it in enumerate(items) if featured(it)]
    return (int((count[nk:] > 0).sum()), len(persons), int((count[:nk] > 0).sum()), nk,
            int((count[feat] > 0).sum()), len(feat))


def _secs(x):
    """'1:23.5' or 83.5 -> seconds."""
    if isinstance(x, (int, float)):
        return float(x)
    parts = [float(v) for v in str(x).strip().split(":")]
    return sum(v * 60 ** k for k, v in enumerate(reversed(parts)))


def _norm(v):
    """An in-out as (from, to) seconds, for comparing picks written differently ("1:18.6-1:22.8" / "1:18.60-...")."""
    if not isinstance(v, str) or "-" not in v.strip()[1:]:
        return v
    return tuple(round(_secs(t), 2) for t in v.split("-", 1))


def pick_ending(picks):
    """The editor's ending pick (a clip's "end": its in-out): (export file, from, to) in that file's seconds, or None."""
    for uid, e in (picks or {}).items():
        if not uid.startswith("_") and isinstance(e, dict) and isinstance(e.get("end"), str) \
                and "all" not in e.get("ban", []) and "-" in e["end"].strip()[1:]:
            a, b = _norm(e["end"])
            return uid, a, b
    return None


def apply_picks(recs, picks, only=True):
    """The editor's own calls, by export file name, with times in that file's own timeline (m:ss or seconds):
        {"C2025": {"ban": ["0:00-0:20"], "prefer": ["0:41-0:46"], "add": ["1:12.5"], "more": 2},
         "C1984-05_couple": {"ban": ["all"]}}
    Banned stretches lose their moments ("all": the whole clip); preferred ones jump the queue (+1.0) -- and if no
    detected moment falls inside, one is made at the middle of the range (kind and framing from the nearest), so a
    suggestion always lands. "add": extra shots the reel must include, each at that time (include_adds); a range
    ("1:12.4-1:15.1", from the source browser) is the editor's in-out: one shot, from its in-point, its length. "more":
    ask for more shots from this clip -- its fair-share discount is divided by 1 + more and its moments rank up.
    "open": the in-out ("1:18.6-1:22.8") the reel opens on, under the event's logo, in place of the opening B-roll;
    "end": the in-out it ends on (pick_ending: it becomes the plan's "ending"). Either may also be in "add".
    A clip with picks gives the reel those and nothing else -- the editor chose its best part, so the planner takes
    nothing more from it: its other moments go, and the rest of it is banned (single-time adds keep 1.5 s either
    side). The planner may still cut a pick down (pick_pieces). not only (reel.json plan.<fmt> "only_picks": false):
    the rest of the clip stays in play, between the picks. Keys starting with "_" are notes. Changes recs in place."""
    picks = {k: v for k, v in (picks or {}).items() if not k.startswith("_")}
    for r in recs:
        p = picks.get(r["id"]) or picks.get(r["unit"]["id"])
        if not p:
            continue
        u = r["unit"]
        if "all" in p.get("ban", []):                         # a banned clip is out, whatever else it was given
            r["moments"], u["bans"] = [], [*u.get("bans", []), [u["t0"] - 1, u["t1"] + 1]]
            continue

        def span(txt, u=u):
            if txt == "all":
                return u["t0"] - 1, u["t1"] + 1
            x0 = u.get("export_t0", 0.0)
            if isinstance(txt, str):
                txt = txt.split("-", 1) if "-" in txt.strip()[1:] else [txt, txt]   # one time: a moment, not a range
            return tuple(x0 + _secs(v) for v in txt)
        for a, b in map(span, p.get("ban", [])):
            r["moments"] = [m for m in r["moments"] if not a <= m["t"] <= b]
            u["bans"] = [*u.get("bans", []), [a, b]]           # in_point keeps shots out of them too
        for a, b in map(span, p.get("prefer", [])):
            inside = [m for m in r["moments"] if a <= m["t"] <= b]
            if not inside and r["moments"]:
                near = min(r["moments"], key=lambda m: abs(m["t"] - (a + b) / 2))
                inside = [dict(near, t=round((a + b) / 2, 2))]
                r["moments"].append(inside[0])
            for m in inside:
                for k in [k for k in ("score_l", "score_p") if k in m] or ["score"]:
                    m[k] += 1.0
                m["pick"] = True
        roles = {_norm(p.get(k)) for k in ("open", "end") if p.get(k)}
        stars = {_norm(k): n for k, n in (p.get("stars") or {}).items() if isinstance(n, (int, float))}
        for v in [v for v in p.get("add", []) if _norm(v) not in roles] + ([p["open"]] if p.get("open") else []):
            opening = v is p.get("open")
            a, b = span(v)
            t = (a + b) / 2
            if not r["moments"]:
                continue
            near = min(r["moments"], key=lambda m: abs(m["t"] - t))
            m = near if abs(near["t"] - t) < 0.3 else dict(near, t=round(t, 2))
            if m is not near:
                r["moments"].append(m)
            for k in [k for k in ("score_l", "score_p") if k in m] or ["score"]:
                m[k] += 1.0
            m["pick"] = True
            if stars.get(_norm(v)):
                m["stars"] = stars[_norm(v)]                     # (how much the editor wants it: pick_pieces)
            m["open" if opening else "add"] = True                # (the opening is placed by story, not include_adds)
            if b - a > 0.05:
                m["span"] = [round(a, 3), round(b, 3)]
                # the pick's footage is the pick's alone: no other moment in it, and no other shot over it
                r["moments"] = [x for x in r["moments"] if x is m or not a <= x["t"] <= b]
                u["bans"] = [*u.get("bans", []), [a, b]]
        keep = [span(v) for v in p.get("add", []) + ([p["open"]] if p.get("open") else [])]
        if only and any(b - a > 0.05 for a, b in keep):
            r["moments"] = [m for m in r["moments"] if m.get("add") or m.get("open")]
            keep = sorted((a, b) if b - a > 0.05 else (a - 1.5, a + 1.5) for a, b in keep)
            edge = u["t0"] - 1
            for a, b in keep + [(u["t1"] + 1, u["t1"] + 1)]:
                if a > edge:
                    u["bans"] = [*u.get("bans", []), [edge, a]]
                edge = max(edge, b)
        if p.get("more"):
            r["more"] = max(int(p["more"]), 0)
            for m in r["moments"]:
                for k in [k for k in ("score_l", "score_p") if k in m] or ["score"]:
                    m[k] += 0.15 * r["more"]


PLAIN_HOLD = 4           # with picks cut to pieces, a plain shot holds at most this many beats (split_long)
PIECE_FULL = 0.55        # picks cut to pieces fill at most this share of an act (spread_picks): the rest is other footage
PIECE_KEEP = 0.7         # a pick's second piece goes in only if it plays at least this well against its best one


def pick_like(project, cfg):
    """The picks another format kept (cfg "pick_like": "landscape"), as [unit id, from, to] source seconds of the
    parts its plan plays, or None (no such key, or that format not planned yet)."""
    other = cfg.get("pick_like")
    f = other and project.path(f"plan_{project.cut_name(other)}.json")
    if not f or not os.path.exists(f):
        return None
    return [[s["unit"]["id"], s["src_in"], s["src_in"] + s["length"]] for s in json.load(open(f))["shots"] if s.get("add")]


def pick_pieces(project, recs, fmt, beat, hold, budget, share=None, ending=None, close=None, clip_share=None,
                grow=None, whole=(), by=None, like=None):
    """Long picks cut down to their best part (reel.json plan.<fmt> "pick_hold": beats): each in-out longer than
    hold beats becomes its best hold-beat window, scored on the moment curve (moments.curve: moves, quality, framing
    in this format, activity, face) -- the editor chose the stretch, the planner the best of it. Then the picks are
    condensed to a share of the reel's beats (budget, default half), shared out by subject: a clip with many picks
    keeps its best ones up to its share (at least one; clip_share -- cfg "pick_clip_share", default 1.5 -- times
    the even share, at least two pieces' worth; raise it when the editor's picks bunch in a few clips), and a long
    pick earns a second piece of itself when there is room and it plays nearly as well (PIECE_KEEP); with grow (cfg
    "pick_grow": beats, above hold) the room left lengthens the best picks to their best grow-beat window instead, so
    a short reel can take every pick at hold beats and the strongest at grow. A clip in whole (cfg "pick_whole": its
    picks are the reel's focus) keeps its picks whole and uncounted, as the ending does. by="clarity" (cfg "pick_by")
    scores a part on how clearly its subject shows (the scrub's framing score, the pick rules passed, their faces
    seen) and drops the least clear picks first, whichever clip they're from, in place of a share per clip -- B-roll
    (no dancer to be clear) ranked among the B-roll on its own score, the people shots among themselves, each by its
    place in its own group, so neither crowds the other out. Stars (picks "stars": {range: 1-5}, unrated 3) come
    first either way: the most starred are lengthened first and dropped last. like (cfg "pick_like": another format,
    planned first): that format's choice -- the picks it kept, each at the part it plays -- in place of choosing
    again, so both formats show the same picks. Each piece is a
    shot of its own (include_adds spreads them through the act, between the rest of the footage). The ending (cfg
    "ending": [file, from, to, ...]) and the close (cfg "close": [unit id, time]) are left whole and aren't counted.
    With a scrub (windows_<fmt>.json), the parts are scored on its windows instead (rank x score; a window that fails
    a rule at half weight): what the editor picks for, not the move curve. Returns the picks
    left out ([export file, from, to] in source seconds, for pick_report). Changes recs in place."""
    from . import moments as MO
    series, cands = {}, []
    w = hold * beat
    wpath = project.path(f"windows_{fmt}.json")
    scrub = json.load(open(wpath)) if os.path.exists(wpath) else None

    def scrubbed(uid, x, y):                                 # best scrub window mostly inside [x, y]
        best = 0.0
        for z in scrub.get(uid) or ():
            if min(y, z["t1"]) - max(x, z["t0"]) >= 0.6 * min(y - x, z["t1"] - z["t0"]):
                best = max(best, clarity(z) if by == "clarity" else z["rank"] * z["score"] * (1.0 if z["ok"] else 0.5))
        return best

    def clarity(z):                                          # the subject clearly seen: framed, every rule passed,
        q = z["score"] * (1.0 if z["ok"] else 0.4)           # their face showing (B-roll: its framing alone)
        face = (z.get("person") or {}).get("face")
        return q * (0.5 + 0.5 * max(face)) if face else q
    for r in recs:
        for m in [m for m in r["moments"] if m.get("add") and m.get("span") and not m.get("open")]:
            a, b = m["span"]
            u = r["unit"]
            if ending and ending[0] in (r["id"], u["id"]) and abs(u.get("export_t0", 0.0) + ending[1] - a) < 0.05:
                continue
            if close and u["id"].startswith(close[0]) and a - 0.5 <= close[1] <= b:
                continue
            if any(u["id"].startswith(c) or r["id"].startswith(c) for c in whole or ()):
                continue
            path = project.path("scores", u.get("score", u["id"]) + ".npz")
            if path not in series:
                series[path] = MO.clip_series(path) if os.path.exists(path) else None
            S = series[path]
            t, c = (S["t"], MO.curve(u, S)[0][fmt]) if S is not None else (None, None)
            win = lambda x, y, t=t, c=c, S=S: (0.5 * c[k].mean() + 0.5 * c[k].max()) if S is not None and \
                (k := (t >= x) & (t <= y)).any() else 0.0
            if scrub is not None:
                win = lambda x, y, uid=u["id"]: scrubbed(uid, x, y)
            if b - a <= w + 0.5 * beat:
                wins = [(win(a, b), a, b)]
            else:                                            # every hold-beat window, best first, none overlapping
                xs = sorted(((win(x, x + w), x, x + w) for x in np.arange(a, b - w + 1e-6, 0.1)), key=lambda z: -z[0])
                wins = []
                for z in xs:
                    if all(z[1] >= y[2] or z[2] <= y[1] for y in wins):
                        wins.append(z)
            cands.append(dict(r=r, m=m, wins=wins, subject=u["id"], win=win))
    if not cands:
        return []
    if by == "clarity":                                      # each pick's place in its own group (B-roll, people)
        for grp in (True, False):
            g = sorted((c for c in cands if (c["r"]["unit"]["role"] == "transition") == grp), key=lambda c: c["wins"][0][0])
            for k, c in enumerate(g):
                c["q"] = (k + 1) / len(g)
    star = lambda c: c["m"].get("stars") or 3
    room = budget * (share or 0.5)
    n = lambda z: max(1, round((z[2] - z[1]) / beat))
    subj = Counter(c["subject"] for c in cands)
    cap = max(2 * hold, (clip_share or 1.5) * room / len(subj))
    keep = {id(c): [c["wins"][0]] for c in cands}
    load = Counter()
    for c in cands:
        load[c["subject"]] += n(c["wins"][0])
    dropped = []
    if like is not None:                                     # the other format chose: the same picks, the same parts
        for c in cands:
            a, b = c["m"]["span"]
            uid = c["r"]["unit"]["id"]
            z = [(1.0, max(a, x), min(b, y)) for u, x, y in like if u == uid and min(b, y) - max(a, x) > 0.1]
            if z:
                keep[id(c)] = [max(z, key=lambda q: q[2] - q[1])]
            else:
                keep.pop(id(c))
                dropped.append(c)
        load = Counter()
        for c in cands:
            if id(c) in keep:
                load[c["subject"]] += n(keep[id(c)][0])
    while like is None:                                      # over its share, or the picks over theirs: the subject
        total = sum(load.values())                           # with the most drops its weakest pick
        over = [s for s in subj if load[s] > cap and subj[s] > 1] or \
            ([s for s in subj if subj[s] > 1] if total > room else [])
        if by == "clarity" and total > room:                 # the least clear pick goes, from whichever clip
            c = min((c for c in cands if id(c) in keep), key=lambda c: (star(c), c["q"], keep[id(c)][0][0]))
            load[c["subject"]] -= n(keep.pop(id(c))[0]); subj[c["subject"]] -= 1
            dropped.append(c)
            continue
        if not over:
            break
        s = max(over, key=lambda s: load[s])
        c = min((c for c in cands if c["subject"] == s and id(c) in keep), key=lambda c: (star(c), keep[id(c)][0][0]))
        load[s] -= n(keep.pop(id(c))[0]); subj[s] -= 1
        dropped.append(c)
    if grow and grow > hold and like is None:                # room left: the best picks play grow beats
        g = grow * beat
        for c in sorted((c for c in cands if id(c) in keep), key=lambda c: (-star(c), -c.get("q", 0), -keep[id(c)][0][0])):
            a, b = c["m"]["span"]
            z0 = keep[id(c)][0]
            if b - a <= z0[2] - z0[1] + 0.5 * beat:
                continue                                     # (already the whole pick)
            L = min(g, b - a)
            xs = [(c["win"](x, x + L), x, x + L) for x in np.arange(a, b - L + 1e-6, 0.1)] or [(z0[0], a, b)]
            z = max(xs, key=lambda q: q[0])
            more = n(z) - n(z0)
            if more > 0 and sum(load.values()) + more <= room:
                keep[id(c)] = [z]
                load[c["subject"]] += more
    extra = [] if grow or like is not None else sorted(((z, c) for c in cands if id(c) in keep for z in c["wins"][1:]
                    if z[0] >= PIECE_KEEP * c["wins"][0][0]), key=lambda zc: -zc[0][0])
    for z, c in extra:                                       # room left: a long pick's next best part
        if load[c["subject"]] + n(z) <= cap and sum(load.values()) + n(z) <= room:
            keep[id(c)].append(z); load[c["subject"]] += n(z)
    for c in cands:
        r, m = c["r"], c["m"]
        r["moments"] = [x for x in r["moments"] if x is not m]
        for k, (_, x, y) in enumerate(sorted(keep.get(id(c), []), key=lambda z: z[1])):
            r["moments"].append(dict(m, t=round(float(x + y) / 2, 2), span=[round(float(x), 3), round(float(y), 3)],
                                     whole=m["span"],
                                     part=k))
    x0 = lambda c: c["r"]["unit"].get("export_t0", 0.0)
    out = [f"{c['r']['unit']['id']} {c['m']['span'][0] - x0(c):.2f}-{c['m']['span'][1] - x0(c):.2f}" for c in dropped]
    print(f"[reel] {fmt}: picks cut to their best {hold} beats; {len(cands) - len(dropped)} of {len(cands)} kept "
          f"({sum(load.values())} beats of {room:.0f}), {sum(len(v) - 1 for v in keep.values())} second pieces"
          + (f"; condensed: {', '.join(out)}" if out else ""))
    return [[c["r"]["unit"]["id"], *c["m"]["span"]] for c in dropped]


def split_long(P, seq, recs, home, hold=PLAIN_HOLD):
    """A plain shot longer than hold beats (outside the teaser and finale) becomes two: it keeps the first half, and
    the second is the best well-framed moment not used yet from footage that belongs in that act (home: record id ->
    act; else of the same role, else any dancing), not of
    a clip beside it, never B-roll beside B-roll -- more of the rest of the footage between the editor's picks.
    Changes seq in place."""
    i = 0
    while i < len(seq):
        x = seq[i]
        if plain(x) and not x.get("span") and x["beats"] > hold and x["act"] not in ("teaser", "finale"):
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            near = {x["unit"]["clip"], nxt["unit"]["clip"] if nxt else None}
            broll = "transition" in (x["unit"]["role"], nxt["unit"]["role"] if nxt else None)
            ok = [r for r in recs if r["unit"]["clip"] not in near and not (broll and r["role"] == "transition")]
            got = next((g for pool in ([r for r in ok if home.get(r["id"]) == x["act"]],
                                       [r for r in ok if r["role"] == x["unit"]["role"]],
                                       [r for r in ok if r["role"] != "transition"])
                        if (g := P.pool(pool, None, 0.35))), None)
            if got:
                give = x["beats"] // 2
                x["beats"] -= give
                seq.insert(i + 1, P.take(*got[0], beats=give, act=x["act"]))
                continue
        i += 1


def merge_flashes(P, seq, keep=("teaser", "climax", "finale")):
    """No 1-beat shot outside the climax (keep: acts whose quick cuts stay; band inserts on the hits stay too): a
    plain or people shot of 1 beat takes a beat from a plain shot of 3+ beside it in its act, or else goes, its beat
    to the shorter plain shot beside it (a clip seen elsewhere first). Between the editor's picks, every other shot
    holds long enough to read. Changes seq in place."""
    units = Counter(x["unit"]["id"] for x in seq)
    soft = lambda x: (plain(x) or (x.get("person") and not x["fx"])) and not x.get("span")
    flash = lambda k: soft(seq[k]) and seq[k]["beats"] == 1 and seq[k]["act"] not in keep
    nbs = lambda k: [j for j in (k - 1, k + 1) if 0 <= j < len(seq) and plain(seq[j]) and not seq[j].get("span")
                     and seq[j]["act"] == seq[k]["act"]]
    while True:
        ks = [k for k in range(1, len(seq) - 1) if flash(k) and nbs(k)]
        if not ks:
            break
        k = min(ks, key=lambda k: units[seq[k]["unit"]["id"]] == 1)
        big = [j for j in nbs(k) if seq[j]["beats"] >= 3]
        if big:
            seq[max(big, key=lambda j: seq[j]["beats"])]["beats"] -= 1
            seq[k]["beats"] += 1
            continue
        j = min(nbs(k), key=lambda j: seq[j]["beats"])
        seq[j]["beats"] += 1
        units[seq[k]["unit"]["id"]] -= 1
        P.count[seq[k]["unit"]["clip"]] -= 1
        seq.pop(k)


def picks_only(recs, categories):
    """Export categories the editor picks from by hand (reel.json "picks_only"): their files are in the source browser
    like any other, but the reel uses only the stretches the editor added; a file with no add picks stays out (it is
    not part of "use everything"). Run after apply_picks. Changes recs in place."""
    for r in recs:
        if r["unit"].get("category") in categories:
            r["moments"] = [m for m in r["moments"] if m.get("add")]


def keep_previous(recs, shots, picks):
    """Clips the editor left alone keep the shots an earlier cut took from them (reel.json "keep": that cut's plan
    file, "plan_{fmt}_v8.json", or true: the last one rendered, plan_{fmt}_rendered.json): only the moments those shots played stay, raised by 0.5, so the re-cut uses the
    same footage there and re-times it to the beat. A clip with any add or ban is the editor's and is untouched
    here, as is one the earlier cut didn't use. Run after apply_picks. Changes recs in place."""
    mine = {k for k, e in (picks or {}).items() if isinstance(e, dict) and (e.get("add") or e.get("ban"))}
    used = {}
    for s in shots:
        used.setdefault(s["unit"]["id"], []).append(s["t"])
    for r in recs:
        ts = used.get(r["unit"]["id"])
        if not ts or not r["moments"] or {r["id"], r["unit"]["id"]} & mine:
            continue
        keep = []
        for t in sorted(set(ts)):
            near = min(r["moments"], key=lambda m: abs(m["t"] - t))
            m = dict(near) if abs(near["t"] - t) < 0.05 else dict(near, t=t)
            for k in [k for k in ("score_l", "score_p") if k in m] or ["score"]:
                m[k] += 0.5
            keep.append(m)
        r["moments"] = keep


def act_logos(seq, logos):
    """Title an act (reel.json "act_logos": {"<act>": "<image>"}, e.g. a headliner's name as their song comes in):
    the act's first plain shot of 4 beats or more (else its first plain shot) gets that image front and centre, faded
    in, the way the opening carries the event's logo. {"<act>": {"image": ..., "match": ["<category or tag>", ...]}}
    puts it on the act's first plain shot from matching footage instead (e.g. the headliner on stage), lengthened to
    4 beats from the plain shots after it so the title can be read (or held over them); with no matching shot, as for
    a plain image. An act
    with no plain shot is left as it is. Changes seq in place."""
    for act, spec in (logos or {}).items():
        image, match = (spec, []) if isinstance(spec, str) else (spec["image"], spec.get("match") or [])
        cand = [s for s in seq if s["act"] == act and plain(s) and not s.get("span")]
        hits = [s for s in cand if any(k in (s["unit"].get("category") or "") or k in (s["unit"].get("tag") or "")
                                       for k in match)]
        s = hits[0] if hits else next((s for s in cand if s["beats"] >= 4), cand[0] if cand else None)
        if not s:
            continue
        s["logo"], s["logo_image"] = "in", image
        k = seq.index(s) + 1                                 # long enough to read, from the plain shots after it
        while s["beats"] < 4 and k < len(seq) and seq[k]["act"] == act and plain(seq[k]) and not seq[k].get("span"):
            give = min(4 - s["beats"], seq[k]["beats"] - 1)   # (each keeps a beat; the grid stays exact)
            if give <= 0:
                break
            s["beats"] += give; seq[k]["beats"] -= give; k += 1
        k, shown = seq.index(s) + 1, s["beats"]              # still short: the title stays up over the next
        while shown < 4 and k < len(seq) and seq[k]["act"] == act and plain(seq[k]):     # shots, as the opening's does
            seq[k]["logo"], seq[k]["logo_image"] = "hold", image
            shown += seq[k]["beats"]; k += 1


def close_only(project, recs, cfg):
    """Footage that only works close up (reel.json "close_only"): {"match": ["day"], "face": 0.2}. Export files whose
    category or tag contains a match keep just the moments with a face at least `face` of the frame height within
    0.75 s (the people step's face scan) -- e.g. day footage where the venue still looked empty, so only shots
    filled by people get in. The editor's own adds always stay. A file with no close moment drops out. Changes recs
    in place."""
    if not cfg:
        return
    match, need = cfg.get("match") or [], float(cfg.get("face", 0.2))
    for r in recs:
        u = r["unit"]
        if not any(m in (u.get("category") or "") or m in (u.get("tag") or "") for m in match):
            continue
        path = project.path("scores", f"faces_{u['id']}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"close_only needs the people step's face scan ({path}); run --steps people")
        f = np.load(path)
        big = f["t"][(f["h"] >= need) & (f["score"] >= 0.8)]
        r["moments"] = [m for m in r["moments"] if m.get("add") or (np.abs(big - m["t"]) <= 0.75).any()]


def pick_report(shots, picks, condensed=(), only=True):
    """What went wrong with the editor's picks in a plan, one line each (empty: all well). An in-out pick ("add"
    range) must be in it exactly once, at least as long as marked (slow motion may show less footage; that is
    fine), never sped up, with no other shot over its footage; a ban must keep every shot out (a pick over a ban
    wins, and is reported as the conflict it is). A pick cut to its best part (pick_pieces) need only be that part;
    one condensed out (condensed: pick_pieces' list) must be out; and with only false (picks cut to pieces) the rest
    of a picked clip may be used too."""
    out = []
    for uid, e in (picks or {}).items():
        if uid.startswith("_"):
            continue
        mine = [s for s in shots if s["unit"]["id"] == uid]
        x0 = mine[0]["unit"].get("export_t0", 0.0) if mine else 0.0
        spans = []
        for v in ([] if "all" in e.get("ban", []) else e.get("add", [])):   # (a banned clip's picks don't count)
            if not isinstance(v, str) or "-" not in v.strip()[1:]:
                continue                                     # a single time: a moment, placed or not by the planner
            a, b = (x0 + _secs(t) for t in v.split("-", 1))
            spans.append((a, b))
            hit = [s for s in mine if s.get("span") and not s.get("reuse") and abs((s.get("whole") or s["span"])[0] - a) < 0.01
                   and (only or s.get("whole") or abs(s["span"][1] - b) < 0.01)]   # (pieces: this pick's own)
            if any(c[0] == uid and abs(c[1] - a) < 0.01 for c in condensed):
                if hit:
                    out.append(f"{uid} {v}: condensed out, but in the reel")
                continue
            over = [s for s in mine if not s.get("span") and min(s["src_in"] + s["length"], b) - max(s["src_in"], a) > 0.1]
            if not hit:
                out.append(f"{uid} {v}: missing")
                continue
            if len({tuple(s["span"]) for s in hit}) < len(hit):
                out.append(f"{uid} {v}: in the reel {len(hit)} times")
            s = hit[0]
            if any(f in FAST for f in s["fx"]):
                out.append(f"{uid} {v}: sped up ({', '.join(s['fx'])})")
            elif not any(f in SPEED for f in s["fx"]) and not s.get("whole") and only and \
                    s["length"] < (b - a) * s.get("share", 1.0) - 0.3:
                out.append(f"{uid} {v}: plays {s['length']:.2f} s of {b - a:.2f}")
            if over:
                out.append(f"{uid} {v}: {len(over)} other shot(s) over its footage")
        if spans and only:
            other = [s for s in mine if not s.get("span") and not s.get("ending") and
                     not any(min(s["src_in"] + s["length"], b) - max(s["src_in"], a) > 0.1 for a, b in spans)]
            if other:
                out.append(f"{uid}: {len(other)} shot(s) outside your selections (only those should be used)")
        for f, k in (("open", 0), ("end", -1)):
            if isinstance(e.get(f), str) and "all" not in e.get("ban", []) and shots:
                a = x0 + _norm(e[f])[0] if isinstance(_norm(e[f]), tuple) else None
                s = shots[k]
                if a is None or s["unit"]["id"] != uid or abs(s["src_in"] - a) > 0.3:
                    out.append(f"{uid} {e[f]}: marked as the {'opening' if f == 'open' else 'ending'}, but it isn't "
                               f"the {'first' if f == 'open' else 'last'} shot")
        for v in e.get("ban", []):
            if v == "all":
                if mine:
                    out.append(f"{uid} banned, but in the reel {len(mine)} times")
                continue
            a, b = (x0 + _secs(t) for t in (v.split("-", 1) if "-" in v.strip()[1:] else [v, v]))
            bad = [s for s in mine if not s.get("span") and min(s["src_in"] + s["length"], b) - max(s["src_in"], a) > 0.1]
            if bad:
                out.append(f"{uid} ban {v}: still used ({len(bad)} shot(s))")
            elif any(min(sb, b) - max(sa, a) > 0.25 for sa, sb in spans):   # (touching at an edge is not a clash)
                out.append(f"{uid} ban {v}: one of your picks overlaps it (the pick wins)")
    return out


def ending_record(recs, cfg):
    """(record, source time) of the configured ending shot (cfg "ending": [export file, from, to]), or None."""
    e = cfg.get("ending")
    r = next((r for r in recs if e and e[0] in (r["id"], r["unit"]["id"]) and r["moments"]), None)
    return (r, r["unit"].get("export_t0", 0.0) + (e[1] + e[2]) / 2) if r else None


def reserve_ending(moments, cfg):
    """Keep the ending's stretch (+-4 s) for the ending: no other shot uses it."""
    e = cfg.get("ending")
    for r in moments if e else ():
        if e[0] in (r["id"], r["unit"]["id"]):
            x0 = r["unit"].get("export_t0", 0.0)
            r["moments"] = [m for m in r["moments"] if not x0 + e[1] - 4 <= m["t"] <= x0 + e[2] + 4] or r["moments"]


def band_in(P, seq, recs, music, cfg):
    """The band's players cut in through the reel, one at a time: short shots replacing plain dance shots on the
    music's hardest hits, spread out through each act, never next to other B-roll, an effect or another player."""
    def member(r):
        return r["role"] == "transition" and (any(k in r["unit"].get("tag", "") for k in cfg["members"])
                                              or r["unit"]["clip"] in cfg["member_clips"])
    players = [r for r in recs if member(r)]
    if not players:
        return
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    acc = np.array(music.get("beat_accent") or np.ones(len(bt)))[i0:]
    pos, rec = [], 0
    for x in seq:
        pos.append(rec); rec += x["beats"]
    for act, n in cfg["band_in"].items():
        units = Counter(x["unit"]["id"] for x in seq)
        idx = [i for i, x in enumerate(seq) if x["act"] == act]
        if not idx:
            continue
        lo, hi = idx[0], idx[-1]
        for k in range(n):
            aim = lo + (k + 0.5) * (hi - lo + 1) / n
            cand = [i for i in range(lo + 1, hi) if seq[i]["unit"]["role"] != "transition" and not seq[i]["fx"]
                    and not any(seq[i].get(f) for f in ("ending", "add", "reaction", "logo")) and seq[i]["beats"] <= 2
                    and units[seq[i]["unit"]["id"]] > 1
                    and not any(seq[j]["unit"]["role"] == "transition" for j in (i - 1, i + 1))
                    and not any(seq[j].get("band") or seq[j]["fx"] for j in (i - 1, i + 1))]
            if not cand:
                break
            i = max(cand, key=lambda i: (acc[pos[i]] if pos[i] < len(acc) else 1) - 0.15 * abs(i - aim))
            used = {x["unit"]["clip"] for x in seq if x.get("band")}
            order = sorted(players, key=lambda r: (r["unit"]["clip"] in used, P.count.get(r["unit"]["clip"], 0)))
            x = next((P.best([r]) for r in order if P.best([r])), None)
            if not x:
                break
            old = seq[i]
            P.count[old["unit"]["clip"]] -= 1
            units[old["unit"]["id"]] -= 1; units[x[0]["unit"]["id"]] += 1
            seq[i] = P.take(*x, beats=old["beats"], act=act); seq[i]["band"] = True


def fits(unit, match):
    """Is this footage one of these (substrings of its category or tag)? "!x" rules out footage matching x;
    "@portrait" / "@landscape" match the way it was shot. Empty match: anything."""
    has = lambda k: unit.get("orient") == k[1:] if k.startswith("@") else \
        k in (unit.get("category") or "") or k in (unit.get("tag") or "")
    yes, no = [k for k in match if not k.startswith("!")], [k[1:] for k in match if k.startswith("!")]
    return (not yes or any(map(has, yes))) and not any(map(has, no))


def song_sets(project, music):
    """Song sets (reel.json songs[k] "footage", "title" + "title_match" or "titles": [{"image", "match"}, ...] for a
    song two acts share, in order): the beats each song plays for in the reel and
    the footage it may use, e.g. a festival's acts in the day's order -- each singer in their own song, the day's crowd
    before the first night set. None when no song names its footage."""
    songs, secs = project.config.get("songs") or [], music.get("sections") or []
    if not any(sp.get("footage") or sp.get("title") or sp.get("titles") for sp in songs) or len(songs) != len(secs):
        return None
    out, b = [], 0
    for sp, sec in zip(songs, secs):
        n = 4 * sec["bars"]
        titles = sp.get("titles") or ([dict(image=sp["title"], match=sp.get("title_match"))] if sp.get("title") else [])
        titles = [dict(image=t["image"], match=t.get("match") or sp.get("footage") or []) for t in titles]
        out.append(dict(b0=b, b1=b + n, footage=sp.get("footage") or [], titles=titles,
                        title_match=titles[0]["match"] if titles else [],
                        feature=float(sp.get("feature") or 0),
                        feature_match=sp.get("feature_match") or sp.get("title_match") or []))
        b += n
    return out


def starts(seq):
    out, b = [], 0
    for s in seq:
        out.append(b); b += s["beats"]
    return out


def set_at(sets, beat):
    return next((st for st in sets if st["b0"] <= beat < st["b1"]), sets[-1])


SLOT = ("beats", "act", "fx", "effect", "logo", "logo_image", "band", "reaction", "ending", "lead_beats", "featured")


def slot(content, place):
    """A shot's footage in another shot's place (its beats, act, effects and title stay with the place)."""
    return {**{k: v for k, v in content.items() if k not in SLOT}, **{k: v for k, v in place.items() if k in SLOT}}


def unclump(seq, sets):
    """No clip cuts to itself: one of the pair trades places with a plain shot elsewhere in its song whose footage
    fits there too, where neither lands beside its own clip or puts B-roll beside B-roll. Changes seq in place."""
    def clash(k):
        return 0 < k < len(seq) and seq[k - 1]["unit"]["clip"] == seq[k]["unit"]["clip"]

    def good(k):
        u, nb = seq[k]["unit"], [seq[x]["unit"] for x in (k - 1, k + 1) if 0 <= x < len(seq)]
        return all(n["clip"] != u["clip"] for n in nb) and not (
            u["role"] == "transition" and any(n["role"] == "transition" for n in nb))
    for _ in range(3):
        at = starts(seq)
        for i in [k for k in range(1, len(seq)) if clash(k)]:
            if not clash(i):
                continue
            for a in (i, i - 1):
                st = set_at(sets, at[a])
                if not plain(seq[a]) or seq[a].get("span"):
                    continue
                done = False
                for j in range(len(seq)):
                    if abs(j - a) < 2 or not plain(seq[j]) or seq[j].get("span") or seq[j]["act"] == "teaser" \
                            or not (st["b0"] <= at[j] < st["b1"]):
                        continue
                    seq[a], seq[j] = slot(seq[j], seq[a]), slot(seq[a], seq[j])
                    if good(a) and good(j):
                        done = True
                        break
                    seq[a], seq[j] = slot(seq[j], seq[a]), slot(seq[a], seq[j])    # (undo)
                if done:
                    break


def confine(P, seq, recs, sets):
    """Each song set uses only its own footage. A shot that starts inside a song's beats on footage the song doesn't
    name first trades places with a plain shot in a song it does belong to (whose footage fits here), so a person or
    clip it carries stays in the reel; failing that it is swapped for the best free moment of the song's own footage
    (B-roll for B-roll where it can), keeping the shot's beats and effects. The editor's picks and the opening stay.
    Changes seq in place."""
    at = starts(seq)

    def placeable(x, j, skip=()):                            # content x at slot j: not beside its clip, no B-roll pair
        nb = [seq[k]["unit"] for k in (j - 1, j + 1) if 0 <= k < len(seq) and k not in skip]
        return all(n["clip"] != x["clip"] for n in nb) and not (
            x["role"] == "transition" and any(n["role"] == "transition" for n in nb))

    def free(s):
        return not s.get("span") and not (s.get("add") and not s.get("person")) and s["act"] != "teaser" \
            and not s["fx"] and not s.get("logo") and not s.get("ending")

    def wrong(i):
        return not fits(seq[i]["unit"], set_at(sets, at[i])["footage"]) and free(seq[i])
    for i in range(len(seq)):
        if not wrong(i):
            continue
        here = set_at(sets, at[i])
        for j in range(len(seq)):                            # trade places
            if j == i or not free(seq[j]) or not fits(seq[j]["unit"], here["footage"]) \
                    or not fits(seq[i]["unit"], set_at(sets, at[j])["footage"]):
                continue
            if placeable(seq[j]["unit"], i, (j,)) and placeable(seq[i]["unit"], j, (i,)):
                seq[i], seq[j] = slot(seq[j], seq[i]), slot(seq[i], seq[j])
                break
    n = Counter(x["unit"]["id"] for x in seq)
    for i, s in enumerate(seq):                              # what's still out of place: the song's own footage
        st = set_at(sets, at[i])
        if fits(s["unit"], st["footage"]) or s.get("span") or (s.get("add") and not s.get("person")) \
                or s["act"] == "teaser" or s.get("logo") or n[s["unit"]["id"]] < 2:   # (never a file's only shot)
            continue
        own = [r for r in recs if fits(r["unit"], st["footage"])]
        same = [r for r in own if r["role"] == s["unit"]["role"]]
        x = next((x for x in P.pool(same) if placeable(x[0]["unit"], i)), None) or \
            next((x for x in P.pool(own) if placeable(x[0]["unit"], i)), None)
        if not x:                                            # every moment used: the stretch furthest from what's shown
            gap = lambda r, m: min((abs(m["t"] - t) for t in P.used.get(r["id"], [])), default=99.0)
            far = sorted(((r, m) for r in own for m in r["moments"] if placeable(r["unit"], i)),
                         key=lambda x: (x[0]["role"] != s["unit"]["role"], -min(gap(*x), 3.0), -rk(x[1], P.fmt)))
            x = far[0] if far and gap(*far[0]) >= 1.0 else None
        if not x:
            continue
        P.count[s["unit"]["clip"]] -= 1
        n[s["unit"]["id"]] -= 1; n[x[0]["unit"]["id"]] += 1
        new = P.take(*x, beats=s["beats"], act=s["act"])
        for k in SLOT[2:]:
            if k in s:
                new[k] = s[k]
        seq[i] = new
    relocate(seq, sets)


def relocate(seq, sets):
    """A file's only shot that still sits in a song it doesn't belong to moves to the first song it does: into the
    longest plain shot there (split, the stray taking its beats from it), its old place's beats going to the plain
    shot beside it, so the grid stays exact. Changes seq in place."""
    n = Counter(x["unit"]["id"] for x in seq)
    for _ in range(len(seq)):
        at = starts(seq)
        stray = next((i for i, s in enumerate(seq) if n[s["unit"]["id"]] == 1 and s["act"] != "teaser"
                      and not fits(s["unit"], set_at(sets, at[i])["footage"]) and not s["fx"] and not s.get("logo")
                      and not s.get("span") and not s.get("ending")), None)
        if stray is None:
            return
        s = seq[stray]
        home = next((st for st in sets if fits(s["unit"], st["footage"])), None)
        nb = [k for k in (stray - 1, stray + 1) if 0 <= k < len(seq) and plain(seq[k]) and seq[k]["act"] == s["act"]]
        if not home or not nb:
            return

        def room(j, s=s, home=home, at=at, stray=stray):
            x = seq[j]
            if not (home["b0"] <= at[j] < home["b1"]) or not plain(x) or x.get("span") or x["beats"] < 3 \
                    or j in (stray - 1, stray, stray + 1):
                return False
            after = seq[j + 1]["unit"] if j + 1 < len(seq) else None
            return x["unit"]["clip"] != s["unit"]["clip"] and (after is None or after["clip"] != s["unit"]["clip"]) \
                and not (s["unit"]["role"] == "transition" and "transition" in (x["unit"]["role"], after and after["role"]))
        js = [j for j in range(len(seq)) if room(j)]
        if not js:
            return
        j = max(js, key=lambda j: seq[j]["beats"])
        give = min(s["beats"], seq[j]["beats"] - 2) or 1
        seq[nb[0]]["beats"] += s["beats"]                    # its old place: the neighbour plays on
        moved = dict(s, beats=give, act=seq[j]["act"])
        seq[j]["beats"] -= give
        seq.insert(j + 1, moved)
        seq.pop(stray if stray < j + 1 else stray + 1)


def feature(P, seq, recs, sets):
    """The artist gets the screen (reel.json songs[k] "feature": share, "feature_match": [...], default title_match):
    plain dancing in the song is swapped for the artist's own footage, the weakest shots first, until that share of
    the song's shots is theirs -- never beside its own clip or B-roll beside B-roll, never a file's or a person's
    only shot, never an effect or a title. Changes seq in place."""
    n = Counter(x["unit"]["id"] for x in seq)
    for st in sets:
        if not st["feature"] or not st["feature_match"]:
            continue
        own = [r for r in recs if fits(r["unit"], st["feature_match"])]
        at = starts(seq)
        idx = [i for i in range(len(seq)) if st["b0"] <= at[i] < st["b1"]]
        have = sum(fits(seq[i]["unit"], st["feature_match"]) for i in idx)
        cands = sorted((i for i in idx if plain(seq[i]) and not seq[i].get("span") and not seq[i].get("person")
                        and seq[i]["act"] != "teaser" and n[seq[i]["unit"]["id"]] > 1
                        and not fits(seq[i]["unit"], st["feature_match"])), key=lambda i: seq[i]["rank"])
        for i in cands:
            if have >= st["feature"] * len(idx):
                break
            nb = [seq[k]["unit"] for k in (i - 1, i + 1) if 0 <= k < len(seq)]

            def ok(x, nb=nb):
                return all(u["clip"] != x[0]["unit"]["clip"] for u in nb) and not (
                    x[0]["role"] == "transition" and any(u["role"] == "transition" for u in nb))
            x = next((x for x in P.pool(own) if ok(x)), None)
            if not x:
                continue
            old = seq[i]
            P.count[old["unit"]["clip"]] -= 1
            n[old["unit"]["id"]] -= 1; n[x[0]["unit"]["id"]] += 1
            seq[i] = P.take(*x, beats=old["beats"], act=old["act"]); seq[i]["featured"] = True   # (coverage keeps off)
            have += 1


def set_titles(P, seq, recs, sets):
    """Each song set's title (an artist's name) as the song comes in, on the artist's own footage (title_match): the
    first of its shots in the song's first 3 bars, or else one of those shots swapped for the artist's best free
    moment (never beside its own clip, never B-roll beside B-roll); lengthened to 4 beats from the plain shots after
    it, or the title held over the next shots, so it can be read. Changes seq in place."""
    n = Counter(x["unit"]["id"] for x in seq)
    for st, k_t, title in [(x, k, t) for x in sets for k, t in enumerate(x["titles"])]:
        nt = len(st["titles"])                               # the k-th of nt titles: from k/nt of the way in
        at = starts(seq)
        b0 = max([st["b0"]] + [at[i] + s["beats"] for i, s in enumerate(seq) if s["act"] == "teaser"])   # after the opening
        b0 += k_t * (st["b1"] - b0) // nt
        first = [i for i, s in enumerate(seq) if b0 <= at[i] < min(b0 + 12, st["b1"]) and not s.get("logo")]
        i = next((i for i in first if fits(seq[i]["unit"], title["match"])), None)
        if i is None:                                        # (none early on: theirs a little later, or a swap)
            first += [i for i, s in enumerate(seq) if b0 + 12 <= at[i] < min(b0 + 24, st["b1"]) and not s.get("logo")]
            i = next((i for i in first if fits(seq[i]["unit"], title["match"])), None)
        own = [r for r in recs if fits(r["unit"], title["match"])]
        for j in first if i is None else ():
            if seq[j].get("span") or seq[j].get("person") or n[seq[j]["unit"]["id"]] < 2:   # nor a file's only shot
                continue
            nb = [seq[k]["unit"] for k in (j - 1, j + 1) if 0 <= k < len(seq)]
            x = next((x for x in P.pool(own) if all(n["clip"] != x[0]["unit"]["clip"] for n in nb) and not (
                x[0]["role"] == "transition" and any(n["role"] == "transition" for n in nb))), None)
            if x:
                old = seq[j]
                P.count[old["unit"]["clip"]] -= 1
                n[old["unit"]["id"]] -= 1; n[x[0]["unit"]["id"]] += 1
                seq[j] = dict(P.take(*x, beats=old["beats"], act=old["act"]), fx=old["fx"], effect=old["effect"])
                i = j
                break
        if i is None:
            continue
        s = seq[i]
        s["logo"], s["logo_image"] = "in", title["image"]
        k = i + 1
        while s["beats"] < 4 and k < len(seq) and seq[k]["act"] == s["act"] and plain(seq[k]) and not seq[k].get("span"):
            give = min(4 - s["beats"], seq[k]["beats"] - 1)
            if give <= 0:
                break
            s["beats"] += give; seq[k]["beats"] -= give; k += 1
        k, shown = i + 1, s["beats"]
        while shown < 4 and k < len(seq) and seq[k]["act"] == s["act"] and not seq[k].get("logo"):
            seq[k]["logo"], seq[k]["logo_image"] = "hold", title["image"]
            shown += seq[k]["beats"]; k += 1


def story(fmt, cfg, music, moments, video_beats, total_beats, direction=None, rate=None):
    """The shots, act by act, with their beats summing to video_beats. direction(unit, t) -> (dx_in, dx_out) says
    which way a moment travels (flow.Flow); without it, cuts match on the kind of move only."""
    from .flow import still
    direction = direction or still
    recs = [r for r in moments if r["moments"] and (fmt == "portrait" or r["orient"] == "landscape")]
    P = Picker(fmt, recs)
    P.hold_adds = bool(cfg.get("pick_hold"))
    tag = lambda r: r["unit"].get("tag", "")
    cat = lambda r: r.get("category", "")
    late = [r for r in recs if any(k in cat(r) or k in tag(r) for k in cfg["late"])]
    early = [r for r in recs if r not in late]
    late_dance = [r for r in late if r["role"] != "transition"]
    late_crowd = [r for r in late if r["role"] == "transition"]
    shows = [r for r in early if r["role"] == "show"]
    socials = [r for r in early if r["role"] == "dance"]
    broll = [r for r in early if r["role"] == "transition"]
    day = [r for r in broll if any(k in tag(r) for k in cfg["day"])]
    night = [r for r in broll if r not in day] or broll
    crowd = [r for r in night if "crowd" in tag(r)]
    dance = shows + socials or late_dance
    if not dance:
        raise SystemExit("no dance or show moments to cut a reel from (run the score and moments steps)")
    demos = [r for r in shows if any(k in cat(r) for k in cfg["day_shows"])]
    anim = [r for r in shows if any(k in cat(r) for k in cfg["night_shows"]) and r not in demos]
    stage = [r for r in shows if r not in demos and r not in anim] or shows or socials
    day_cats = [*(cfg.get("day") or ()), *(cfg.get("day_shows") or ())]
    for r in recs:                                           # daytime footage, for every later pass: it never
        r["unit"]["day"] = r in demos or r in day or bool(day_cats and fits(r["unit"], day_cats))   # plays at night
    after_dark = lambda rs: [r for r in rs if not r["unit"]["day"]] or rs   # the night acts' footage
    home = {**{r["id"]: "arrival" for r in demos + day}, **{r["id"]: "shows" for r in stage},
            **{r["id"]: "night" for r in anim + socials + night},
            **{r["id"]: "finale" if r["role"] == "transition" else "climax" for r in late}}
    bars = total_beats // 4
    acts = {name: (a * 4, b * 4) for name, a, b in (cfg["acts"] or default_acts(cfg, bars))}
    budget = {k: max(min(b, video_beats) - a, 0) for k, (a, b) in acts.items()}
    pick = lambda recs, kinds, mf: P.best(recs, kinds, mf) or P.best(dance)
    seq = []

    # teaser: the festival itself -- the band and the packed room -- with the event's logo front and centre; with no
    # such B-roll, the two best moves of the day, slowed through their peak
    rank_tag = lambda r: next((k for k, t in enumerate(cfg["opening"]) if t in tag(r)), None)
    opener = next(((r, m) for r in recs for m in r["moments"] if m.get("open")), None)
    if opener:                                               # the editor's own opening shot (picks "open"), at its
        r, m = opener                                        # full length; a longer one takes beats from the arrival
        need = math.ceil(round((m["span"][1] - m["span"][0]) / music["beat"], 3)) if m.get("span") else 0
        if need > budget["teaser"]:
            budget["arrival"] -= need - budget["teaser"]; budget["teaser"] = need
        s = P.take(r, m, beats=budget["teaser"], act="teaser"); s["logo"] = "in"; s["open"] = True
        seq.append(s)
    else:
        wide = sorted([r for r in night if rank_tag(r) is not None], key=lambda r: (r["orient"] != fmt, rank_tag(r)))
        wide = [r for k, r in enumerate(wide) if all(r["unit"]["clip"] != q["unit"]["clip"] for q in wide[:k])]
        if len(wide) >= 2:
            for k, r in enumerate(wide[:2]):                 # two different clips
                s = P.take(*P.best([r]), beats=budget["teaser"] // 2, act="teaser")
                s["logo"] = "in" if k == 0 else "hold"
                seq.append(s)
        else:
            for _ in range(2):
                s = P.take(*pick(dance, ("spin", "dip", "trick"), 0.55), beats=budget["teaser"] // 2, act="teaser")
                s["fx"], s["effect"] = ["hero"], "hero"
                seq.append(s)
        seq[-1]["beats"] += budget["teaser"] % 2

    # arrival: the daytime -- classes, set-up, panels -- each held as one long shot (they barely move), woven with
    # the daytime show units so two of them never sit side by side, in an uneven rhythm
    if day or demos:
        ar = cover(P, day, len(day), 4, "arrival", cap=1)
        left = budget["arrival"] - 4 * len(ar)
        ar += cover(P, demos or stage, max(len(demos), left // 2), 2, "arrival", cap=3,
                    kinds_pref=("spin", "dip", "trick", "pattern", "shake"))
        ar = weave(ar, seq[-1], direction=direction)
        lb, ld = itertools.cycle([4, 3, 5, 4, 3, 5, 4]), itertools.cycle([2, 3, 1, 2, 1, 3, 2, 2, 1, 3, 2, 1])
        for s in ar:                                         # held B-roll, the dancing in an uneven rhythm between
            s["beats"] = next(lb) if s["unit"]["role"] == "transition" else next(ld)
    else:
        ar = cover(P, broll or dance, max(budget["arrival"] // 4, 1), 4, "arrival", cap=3)
    seq += fit(ar, budget["arrival"], 2)

    # shows: the stage, its routines shuffled through each other, in phrases of varied length; the peaks of the best
    # routines slowed down in the held slots, stage spotlights, zoom kicks on the hips in the quick ones
    lengths = rhythm(budget["shows"], cfg.get("shows_pace", "medium"), phrase_energy(music, acts["shows"][0], budget["shows"] // 8))
    sh = weave(cover(P, stage, len(lengths), 2, "shows", cap=6), seq[-1], direction=direction)
    fit_slots(sh, lengths)
    # the stage's own language: a few of the best moves held in slow motion, and once, the stage's spotlight on its
    # strongest solo -- no zoom kicks or overlays on top of choreography that already says it
    give_slot_fx(sh, "hero", ("spin", "dip", "trick", "pattern", "style", "closeup"), cfg["heroes"], min_beats=3)
    give_slot_fx(sh, "spotlight", ("pattern", "style", "spin", "energy", "shake"), 1)
    seq += fit(sh, budget["shows"], 2)

    # night: the band, the DJ and the crowd, broken up by the night's dancing -- the night show units and the social
    # floor. The lengths come first, the dancing is chosen next, and the B-roll takes evenly spaced long slots: never
    # more B-roll than dancing to break it up
    lengths = rhythm(budget["night"], cfg["night_pace"], phrase_energy(music, acts["night"][0], budget["night"] // 8),
                     build=True)
    N = len(lengths)
    if socials:
        ni = cover(P, anim, max(len(anim), round(0.3 * N)), 2, "night", cap=5, kinds_pref=("shake", "closeup"))
        ni += cover(P, socials, round(0.65 * N) - len(ni), 2, "night", cap=2,
                    kinds_pref=("spin", "dip", "trick", "pattern", "closeup"))
    else:
        ni = cover(P, anim or stage, round(0.6 * N), 2, "night", cap=12, kinds_pref=("shake", "closeup"))
    nb = min(len(night), len(ni), N - len(ni))
    ni += cover(P, night, nb, 2, "night", cap=1)
    if ni:
        lengths = squeeze(lengths, len(ni))
        nb = sum(s["unit"]["role"] == "transition" for s in ni)
        bslots = broll_slots(lengths, len(ni), nb)
        ni = weave(ni, seq[-1], slots=bslots if len(bslots) == nb else None, direction=direction)
        fit_slots(ni, lengths)
        if socials:                                          # the night's one glow: neon belongs to the club
            give_slot_fx(ni, "neon", ("spin", "pattern", "style"), 1)
        give_slot_fx(ni, "hero", ("spin", "dip", "trick", "closeup"), 2, min_beats=3)
        ni[-1]["fx"], ni[-1]["effect"] = [], None            # the velocity edit into the climax needs it plain
    seq += fit(ni, budget["night"], 2)

    # climax: phrases of short cuts that each land on a slowed-down move; crowd reactions after the biggest ones
    patterns = [[2, 2, 1, 1, 2], [1, 1, 2, "H"], [2, 1, 1, 2, 2], [1, 1, 1, 1, "H"], [2, 2, 2, 2], [1, 1, 2, "H"]]
    slots = []
    left = budget["climax"] - 6                              # the last 6 beats: the peak of the whole reel
    for p in patterns * 3:
        need = sum(4 if x == "H" else x for x in p)
        if need > left:
            continue
        slots += p; left -= need
    k = 0
    while left > 0:                                          # what's left: 2, 1, 1, ... not a run of single beats
        x = min((2, 1, 1)[k % 3], left)
        slots.append(x); left -= x; k += 1
    slots.append("B")
    pool = after_dark(dance if fmt == "portrait" else shows or dance)
    if cfg.get("climax_from"):
        pool = [r for r in pool if any(k in cat(r) for k in cfg["climax_from"])] or pool
    if late_dance:                                          # the headliner's set first, the rest of the day after
        pool = late_dance if len(late_dance) >= 4 else after_dark(late_dance + pool)
    crowd = [r for r in late_crowd if "crowd" in tag(r)] + crowd
    heroes = P.pool(pool, ("spin", "dip", "trick"), 0.45)
    if not heroes and cfg.get("climax_from"):               # the climax's own footage has no spins or dips: its
        heroes = (P.pool(pool, (*MOVES, "closeup"), 0.45)     # best dancing, not footage the climax was kept from
                  or P.pool(pool, (*MOVES, "closeup")))
    n_h = slots.count("H")
    hero_moments, seen = [], set()
    for r, m in heroes:                                      # the best moves, one per unit
        if r["id"] not in seen:
            hero_moments.append((r, m)); seen.add(r["id"])
        if len(hero_moments) == n_h + 1:
            break
    peak = hero_moments.pop(0) if hero_moments else None     # the very best: the climax's last beat
    hero_moments.sort(key=lambda x: rk(x[1], fmt))           # build: weaker heroes first
    reserved = [(r["id"], m["t"]) for r, m in hero_moments + ([peak] if peak else [])]

    def clash(x):                                            # the slow-motion moments are spoken for
        return any(x[0]["id"] == rid and abs(x[1]["t"] - t) < APART for rid, t in reserved)
    cl, h_i, last_clip = [], 0, None
    for k, sl in enumerate(slots):
        if sl == "B" or (sl == "H" and h_i < len(hero_moments)):
            if sl == "B":
                s = P.take(*(peak or P.best(pool, MOVES, 0.4) or P.best(after_dark(dance))), beats=6, act="climax"); s["fx"] = ["bullet"]
            else:
                s = P.take(*hero_moments[h_i], beats=4, act="climax"); h_i += 1; s["fx"] = ["hero"]
            s["effect"] = s["fx"][0]
        else:
            nxt = slots[k + 1] if k + 1 < len(slots) else None    # nor the clip of the slow-motion shot coming next
            upcoming = (hero_moments[h_i][0]["unit"]["clip"] if nxt == "H" and h_i < len(hero_moments)
                        else peak[0]["unit"]["clip"] if nxt == "B" and peak else None)
            after_hero = cl and cl[-1]["effect"] == "hero" and h_i in (2, 3)
            src = crowd if after_hero and crowd else pool
            two_shakes = len(cl) >= 2 and all(c["kind"] == "shake" for c in cl[-2:])
            kinds = tuple(k for k in (*MOVES, "energy", "closeup") if not (two_shakes and k == "shake"))
            cand = [x for x in P.pool(src, kinds if src is pool else None, 0.4)
                    if x[0]["unit"]["clip"] not in (last_clip, upcoming) and not clash(x)] or \
                [x for x in P.pool(src) if not clash(x)] or P.pool(src) or \
                (cfg.get("climax_from") and (cl or seq)[-1]["unit"]["role"] != "transition"     # the night's own
                 and [x for x in P.pool(late_crowd + night) if x[0]["unit"]["clip"] != last_clip]) or \
                P.pool(after_dark(dance))                    # B-roll before footage the climax was kept from
            prev = cl[-1] if cl else seq[-1]                 # cut on the action: carry the last shot's movement on
            if not cand:                                     # nothing left to fill it with (the editor's picks
                break                                        # will, include_adds); the grid is evened at the end
            cand = cand[:12]
            cand.sort(key=lambda x: -(P.share(*x) + (0.6 if x[0]["id"] not in P.used else 0)
                                      + 0.5 * flow_link(prev, x[0]["unit"], x[1]["t"], x[1]["kind"], direction)))
            s = P.take(*cand[0], beats=4 if sl == "H" else sl, act="climax")
            if after_hero and src is crowd:
                s["reaction"] = True
        last_clip = s["unit"]["clip"]
        cl.append(s)
    # the biggest visual on the drop: the reel's best move, held in bullet time, burns with fire -- the one subject
    # effect of the climax, built up to by the slow-motion holds before it
    if cl:                                                   # (empty when the picks will fill it, include_adds)
        cl[-1]["fx"], cl[-1]["effect"] = (["bullet", "aura"] if cfg.get("drop_slowmo", True) else ["aura"]), "aura"
        # (reel.json plan.<fmt> "drop_slowmo": false -- a look whose drop effect plays at real speed, e.g. a
        # sandstorm that settles round sharp dancers)
    seq += cl

    # finale: faces, the packed floor, and one last slow-motion move
    fi = []
    dance = after_dark(late_dance + dance if late_dance else dance)
    closeups = P.pool(dance, ("closeup",), 0.3) or P.pool(dance, ("closeup",))
    faces = sorted(closeups, key=lambda x: -(x[1].get("face", 0) + 0.5 * rk(x[1], fmt)))

    def face(beats):
        prev = (fi or seq)[-1]["unit"]["clip"]               # never the clip just before it
        x = next(((r, m) for r, m in faces if P.free(r, m) and r["unit"]["clip"] != prev
                  and all(r["id"] != s["unit"]["id"] for s in fi)), None)
        return P.take(*(x or P.best(dance)), beats=beats, act="finale")
    fi.append(face(2))
    def move(kinds, mf):                                     # never the clip just before it
        prev = (fi or seq)[-1]["unit"]["clip"]
        for ks, f in ((kinds, mf), (kinds, 0.0), (MOVES, 0.0), (None, 0.0)):
            x = next((c for c in P.pool(dance, ks, f) if c[0]["unit"]["clip"] != prev), None)
            if x:
                return x
        return P.best(dance)
    s = P.take(*move(("spin", "dip"), 0.45), beats=4, act="finale"); s["fx"], s["effect"] = ["hero"], "hero"
    fi.append(s)
    cw = [r for r in crowd if r["orient"] == fmt] or crowd   # the packed floor; any night B-roll once it's used up
    fi.append(P.take(*((P.best(cw) if cw else None) or P.best(night) or P.best(dance)), beats=4, act="finale"))
    fi.append(face(3))
    rest = max(budget["finale"] - 13, 2)
    end = ending_record(recs, cfg)
    if end:                                                  # a face and its hair in deep slow motion before the
        fi[-1]["fx"], fi[-1]["effect"] = ["hero"], "hero"    # ending (else the last move is the slow one)
    if end:                                                  # the reel's chosen last shot, to the music's end
        r, t = end
        e = cfg["ending"]                                    # as long as its stretch, not longer: a short one
        eb = max(2, math.ceil(round((e[2] - e[1]) / music["beat"], 3)))   # leaves its beats to one more move
        if rest - eb >= 2:
            x = P.take(*move(MOVES, 0.4), beats=rest - eb, act="finale")
            fi.append(x)
            rest = eb
        near = min(r["moments"], key=lambda m: abs(m["t"] - t))
        s = P.take(r, dict(near, t=round(t, 2), kind="crowd"), beats=rest, act="finale"); s["ending"] = True
        x0 = r["unit"].get("export_t0", 0.0)                 # the editor's own in-out (a pick of it is this shot)
        s["span"] = [round(x0 + e[1], 3), round(x0 + e[2], 3)]
        s.pop("whole", None)
    else:                                                    # the last move to the end, at real speed (a long shot
        s = P.take(*move(("dip", "spin", "trick"), 0.4), beats=rest, act="finale")   # sped up to its last beat
    fi.append(s)                                             # looks wrong; the move before it is the slow one)
    seq += fit(fi, budget["finale"], 2)
    sets, allowed = cfg.get("sets"), None
    if sets:                                                 # song sets: each song's own footage, in the day's order
        confine(P, seq, recs, sets)
        allowed = lambda r, a: any(fits(r["unit"], st["footage"]) and st["b0"] < acts[a][1] and acts[a][0] < st["b1"]
                                   for st in sets) if a in acts else False
        for r in recs:
            first = next((st for st in sets if fits(r["unit"], st["footage"])), None)
            if first:
                home[r["id"]] = next((a for a, (x, y) in acts.items() if x <= first["b0"] + 8 < y), home.get(r["id"]))
    cover_all(P, seq, recs, home, confined=bool(sets))
    pbeat = pick_beat(recs, music["beat"], video_beats - budget["teaser"] - min(budget["finale"], 8))
    P.hold_adds = False
    if cfg.get("close"):                                     # the close: no quick cut ever replays its footage
        c = cfg["close"]
        for r in recs:
            for m in r["moments"] if r["unit"]["id"].startswith(c[0]) else ():
                if m.get("span") and m["span"][0] - 0.5 <= c[1] <= m["span"][1]:
                    m["close"] = True
    include_adds(P, seq, recs, home, pbeat, budget, insert=bool(cfg.get("pick_hold")))
    cov = cover_people(P, seq, recs, home, cfg.get("people"), fmt, music["beat"], cfg.get("climax_from"), allowed,
                       rate)
    if cov:
        print(f"[reel] {fmt}: people the camera stayed on {cov[4]}/{cov[5]}, everyone {cov[0]}/{cov[1]}, "
              f"kisses {cov[2]}/{cov[3]}")
    band_in(P, seq, recs, music, cfg)
    if sets:
        confine(P, seq, recs, sets)
        set_titles(P, seq, recs, sets)
        confine(P, seq, recs, sets)                          # (a title may have given a lone stray a second shot)
        feature(P, seq, recs, sets)
        unclump(seq, sets)
    # an act's first and last shots carry its signature transition: give them at least 2 beats, from a neighbour
    for k, s in enumerate(seq):
        edge = (k > 0 and seq[k - 1]["act"] != s["act"]) or (k + 1 < len(seq) and seq[k + 1]["act"] != s["act"])
        if edge and s["beats"] < 2:
            near = sorted(seq, key=lambda d: abs(seq.index(d) - k))
            donor = next((d for d in near if d["act"] == s["act"] and d is not s and d["beats"] >= 3 and not d["fx"]
                          and not d.get("span")),
                         None) or \
                next((d for d in near if d["act"] == s["act"] and d is not s and d["beats"] == 2 and not d["fx"]
                      and not d.get("span")), None)
            if donor:
                donor["beats"] -= 1; s["beats"] += 1
    for k in range(1, len(seq)):                             # one effect at a time across act changes too
        a, b = seq[k - 1], seq[k]
        if a["act"] != b["act"] and a["fx"] and b["fx"] and b["effect"] in ("hero", "spotlight", "neon"):
            b["fx"], b["effect"] = [], None
    if not cfg.get("pick_hold"):                             # (picks cut to pieces are spread on purpose)
        home_picks(seq, recs, home)
    picks_once(seq, music["beat"] if cfg.get("pick_hold") else None)
    for s in [s for s in seq if s.get("span")]:            # later passes may have moved beats: the editor's picks last
        if not any(x is s for x in seq):                     # (a quick cut a fit before this one took out)
            continue
        fit_pick(seq, next(k for k, x in enumerate(seq) if x is s), pbeat, floor=2 if cfg.get("pick_hold") else 1)
        if pbeat != music["beat"]:
            s["share"] = round(music["beat"] / pbeat, 3)     # (pick_beat: all of them in, each this share)
    balance_picks(seq, pbeat, budget)
    for x in [x for x in seq if x["act"] == "finale" and plain(x) and not x.get("span") and x["beats"] > FINALE_HOLD]:
        if any(y.get("span") for y in seq):                  # a long filler at the end: the editor's picks again,
            n = x["beats"] - FINALE_HOLD                      # cut down, 2 beats each (more cuts at the end)
            x["beats"] = FINALE_HOLD
            quick_cuts(seq, n, "finale", music["beat"], seq.index(x) + 1, size=2)
    if pbeat != music["beat"]:
        settle_picks(seq, pbeat, budget, music["beat"])
        finale_reuse(seq, music["beat"], cfg.get("close"))
    elif cfg.get("close") and not close_last(seq, music["beat"], cfg["close"]):
        print(f"[reel] {fmt}: the close {cfg['close']} isn't among the picks placed")
    unclump_clips(seq)
    for s in seq:                                            # slow motion needs room: none on a shot under 3 beats
        if s["beats"] < 3 and any(f in ("hero", "bullet", "montage", "aura") for f in s["fx"]):
            s["fx"] = [f for f in s["fx"] if f not in ("hero", "bullet", "montage", "aura")]
            s["effect"] = s["fx"][0] if s["fx"] else None
    # whatever the acts could not fill (or overfilled) comes off / goes on the plain shots, so the grid stays exact
    over = sum(s["beats"] for s in seq) - video_beats
    spare = sorted([s for s in seq if not s["fx"] and not s.get("span")], key=lambda s: s["rank"]) or seq
    i = 0
    while over != 0 and i < 50 * len(spare):
        s = spare[i % len(spare)]
        if over > 0 and s["beats"] > 1:
            s["beats"] -= 1; over -= 1
        elif over < 0:
            s["beats"] += 1; over += 1
        i += 1
    # still over (every other shot is down to a beat: the editor's picks fill the reel): the plain shots go, weakest
    # first, then the longest picks give a beat each
    for s in sorted([s for s in seq if plain(s) and not s.get("open")], key=lambda s: s["rank"]):
        if over <= 0:
            break
        if 0 < seq.index(s) < len(seq) - 1:
            over -= s["beats"]
            seq.remove(s)
    while over > 0:
        s = max((s for s in seq if s.get("span")), key=lambda s: s["beats"], default=None)
        if s is None or s["beats"] <= 1:
            break
        s["beats"] -= 1; over -= 1
    if over < 0:                                             # (a dropped shot took more than was over)
        max((s for s in seq if s.get("span")), key=lambda s: s["beats"], default=seq[-1])["beats"] -= over
    if cfg.get("pick_hold"):                                 # picks cut to pieces: no long filler between them, and
        split_long(P, seq, recs, home)                       # none too short to read (the grid stays as it is)
        merge_flashes(P, seq)
        unclump_clips(seq)
    if cfg.get("reuse") is False:
        no_reuse(seq, music["beat"])
    if cfg.get("open_broll"):
        open_broll(P, seq, recs, cfg["open_broll"] if isinstance(cfg["open_broll"], list) else ("teaser", "arrival"))
    if cfg.get("open_with"):
        open_with(seq, cfg["open_with"], beat=music["beat"])
    if cfg.get("end_with"):
        open_with(seq, cfg["end_with"], end=True, beat=music["beat"])
    return seq, acts


def no_reuse(seq, beat):
    """No pick seen twice (cfg "reuse": false): the quick cuts made from picks already in the reel (their trimmed
    ends, their openings again) go, and their beats go to the picks beside them, as far as each pick's own in-out
    has footage for, the nearest first (else to the B-roll beside them, else the shot before). The grid stays."""
    def room(x):                                             # beats of the pick's own footage it doesn't play yet
        w = x.get("whole") or x.get("span")
        return max(0, int((w[1] - w[0]) / beat + 0.25) - x["beats"]) if w else 0
    for r in [x for x in seq if x.get("reuse")]:
        k = next(i for i, x in enumerate(seq) if x is r)
        seq.pop(k)
        n = r["beats"]
        near = sorted(range(len(seq)), key=lambda i: (abs(i - k + 0.5), i))
        for i in near:                                       # picks with footage to spare, nearest first
            x = seq[i]
            if n and x.get("span") and not any(x.get(f) for f in ("ending", "logo", "open")):
                g = min(n, room(x))
                x["beats"] += g; n -= g
        for i in near:                                       # then B-roll beside it (it can hold)
            x = seq[i]
            if n and not x.get("span") and x["unit"]["role"] == "transition" and not x.get("ending"):
                x["beats"] += n; n = 0
        if n:
            seq[max(0, k - 1)]["beats"] += n


def open_with(seq, specs, end=False, beat=None):
    """The editor's picks that follow the logo, in order (cfg "open_with": [[export file, "m:ss.ss"], ...], each a
    pick's in-point): moved from wherever they were placed to just after the teaser, into the arrival; end=True (cfg
    "end_with"): to just before the ending shot, into the finale. The grid stays as it is (an act gives up the beats
    another gains). With beat, each plays whole: the beats it needs come from the shots of its act, before it first."""
    at = next((k for k, x in enumerate(seq) if x.get("ending")), len(seq)) if end else \
        max((k + 1 for k, x in enumerate(seq) if x["act"] == "teaser"), default=0)
    for uid, t in specs:
        t = _secs(t) if isinstance(t, str) else float(t)
        k = next((k for k, x in enumerate(seq) if x.get("span") and x["unit"]["id"] == uid
                  and abs(x["span"][0] - x["unit"].get("export_t0", 0.0) - t) < 0.05), None)
        if k is None:
            print(f"[reel] {'end' if end else 'open'}_with: no placed pick of {uid} at {t:.2f}")
            continue
        x = seq.pop(k)
        x["act"] = "finale" if end else "arrival"
        at = at if k >= at else at - 1
        seq.insert(at, x)
        if beat:                                             # whole: beats from the shots before it in its act
            need = max(1, round((x["span"][1] - x["span"][0]) / beat)) - x["beats"]
            for step in (-1, 1):                             # (the shots before it first, then after)
                j = at + step
                while need > 0 and 0 <= j < len(seq) and seq[j]["act"] == x["act"]:
                    d = seq[j]
                    if d["beats"] > 1 and not any(d.get(f) for f in ("logo", "ending", "open")):
                        d["beats"] -= 1; x["beats"] += 1; need -= 1
                    else:
                        j += step
        at += 1


def open_broll(P, seq, recs, acts=("teaser", "arrival")):
    """B-roll alone in the opening acts (cfg "open_broll": true, or the acts): a shot of people there swaps places
    with a B-roll shot of a later act (not the finale), the nearest in length -- an editor's pick moves on, it isn't
    lost -- or, with none to swap, a planner's shot becomes a B-roll moment; the logo shot leads the teaser. The
    grid stays as it is."""
    br = lambda s: s["unit"]["role"] == "transition"
    for s in [s for s in seq if s["act"] in acts and not br(s) and not s.get("logo")]:
        k = next(i for i, x in enumerate(seq) if x is s)
        later = [j for j, x in enumerate(seq) if j > k and x["act"] not in acts and x["act"] != "finale" and br(x)
                 and not any(x.get(f) for f in ("logo", "ending", "open", "close"))]
        if later:
            j = min(later, key=lambda j: (abs(seq[j]["beats"] - s["beats"]), j))
            x = seq[j]
            x["act"], s["act"] = s["act"], x["act"]
            seq[k], seq[j] = x, s
        elif not s.get("span"):
            got = P.best([r for r in recs if r["role"] == "transition"])
            if got:
                seq[k] = P.take(*got, beats=s["beats"], act=s["act"])
    lg = next((i for i, x in enumerate(seq) if x["act"] == "teaser" and x.get("logo") == "in"), None)
    if lg:                                                   # the logo opens the reel
        seq.insert(0, seq.pop(lg))


def transitions(seq, music, cfg, direction=None):
    """Mostly hard cuts on the beat. Each act has its own short list of transitions and a cap on each (grammar.py),
    the change of act its signature one, and a transition goes in only where the cut has its reason."""
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    acc = np.array(music.get("beat_accent") or np.ones(len(bt)))[i0:]
    rec, pos = 0, []
    for s in seq:
        pos.append(rec); rec += s["beats"]
    room = [s["beats"] for s in seq]                        # beats left in each shot for transition halves
    look = look_of(cfg)

    def put(i, name, why):
        velocity = name == "velocity"                       # a velocity edit: out fast, in fast, a swipe between
        name = "swipe" if velocity else name
        name = (look or {}).get("reasons", {}).get(name, name)   # the same reason, in the film's look
        h = G.HALF[name]
        if room[i - 1] < h + 0.5 or room[i] < h + 0.5:
            return False
        if why != "act change" and (seq[i - 1].get("transition") or (i + 1 < len(seq) and seq[i + 1].get("transition"))):
            return False                                    # never back to back
        A, B = seq[i - 1], seq[i]
        if velocity and not (any(f in SPEED for f in A["fx"] + B["fx"]) or A["beats"] < 2 or B["beats"] < 2
                             or A.get("span") or B.get("span")          # (an editor's pick plays at real speed,
                             or i == len(seq) - 1 or B.get("ending")):  #  and so does the last shot)
            A["fx"].append("flashout"); B["fx"].insert(0, "flashin"); A["velocity"] = B["velocity"] = True
        seq[i]["transition"] = name; seq[i]["why"] = why
        room[i - 1] -= h; room[i] -= h
        return True

    def travel(A, B):
        """A's movement at its cut and B's at its start, same way and both clearly moving: a whip carries it on."""
        if direction is None:
            return False
        a_out, b_in = direction(A["unit"], A["t"])[1], direction(B["unit"], B["t"])[0]
        return a_out * b_in > 0 and min(abs(a_out), abs(b_in)) > G.TRAVEL

    used = {}
    for i in range(1, len(seq)):
        A, B = seq[i - 1], seq[i]
        act = B["act"]
        if A["act"] != B["act"]:
            sig = SIGNATURE.get((A["act"], B["act"]), "dipwhite")
            put(i, sig, "act change")
            continue
        if act == "teaser" or B.get("reaction") or B.get("band") or A.get("band"):
            continue                                        # the logo holds; reactions and the band cut straight in
        caps = cfg["caps"].get(act, {})
        vocab = [n for n in cfg["grammar"].get(act, ()) if used.get((act, n), 0) < caps.get(n, 0)]
        accent = acc[pos[i]] if pos[i] < len(acc) else 1
        for name, why in G.reasons(A, B, vocab, accent, cfg["flash_at"].get(act, 9), travel):
            if put(i, name, why):
                used[(act, name)] = used.get((act, name), 0) + 1
                break


def glitch_look(seq, music, look):
    """The film's effects said in its glitch look (fx/glitch.py LOOKS; transitions() already speaks it): the stock
    subject effects become the look's (look["fx"]); each act gets one more subject effect on a move it suits
    (look["extra"]); and the act's hardest hits get an accent (look["hits"]: a freeze, a punch, a JPEG crunch). Still
    mostly hard cuts: never on the teaser, the ending, B-roll, the band, a reaction or an editor's selection, never
    beside another effect or a transition, one effect a shot."""
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    acc = np.array(music.get("beat_accent") or np.ones(len(bt)))[i0:]
    pos, rec = [], 0
    for s in seq:
        pos.append(rec); rec += s["beats"]
    every = look.get("pulse")                     # a glitch cut on the cut nearest each phrase (look["pulse"] beats),
    if every:                                     # where none is near: the look's transitions in turn
        cuts = {pos[i]: i for i in range(1, len(seq))}
        names = look["transitions"]
        k = 0

        def cuttable(i):
            A, B = seq[i - 1], seq[i]
            return (A["act"] != "teaser" and not any(x.get(f) for x in (A, B) for f in ("ending", "logo", "span"))
                    and min(A["beats"], B["beats"]) >= 1.5 and not B.get("transition")
                    and not (i + 1 < len(seq) and seq[i + 1].get("transition")) and not seq[i - 1].get("transition"))
        for b in range(every, rec, every):
            near = [p for p in cuts if abs(p - b) <= every // 4]
            if any(seq[cuts[p]].get("transition") for p in near):
                continue
            ok = sorted((p for p in near if cuttable(cuts[p])), key=lambda p: abs(p - b))
            if ok:
                i = cuts[ok[0]]
                seq[i]["transition"], seq[i]["why"] = names[k % len(names)], "phrase"
                k += 1
    swap = look.get("fx", {})
    for s in seq:
        s["fx"] = [swap.get(f, f) for f in s["fx"]]
        s["effect"] = swap.get(s.get("effect"), s.get("effect"))

    def free(i):
        s = seq[i]
        return (0 < i < len(seq) - 1 and not s["fx"] and not s.get("transition") and not seq[i + 1].get("transition")
                and s["unit"]["role"] != "transition" and s["act"] != "teaser"
                and not any(s.get(k) for k in ("ending", "band", "reaction", "logo", "span"))
                and not any(seq[j]["fx"] for j in (i - 1, i + 1)))

    def hit(i):                                   # the music's accent where the shot's move peaks
        b = pos[i] + int(round(seq[i].get("peak", 0) / max(seq[i].get("beat", 0.6), 1e-3)))
        return acc[b] if 0 <= b < len(acc) else 1.0
    for act, name, kinds in look.get("extra", []):
        cand = [i for i, s in enumerate(seq) if s["act"] == act and s["kind"] in kinds and s["beats"] >= 2 and free(i)]
        if cand:
            i = max(cand, key=lambda i: (seq[i]["rank"], hit(i)))
            seq[i]["fx"], seq[i]["effect"] = [name], name
    for act, name, n in look.get("hits", []):
        for _ in range(n):
            cand = [i for i, s in enumerate(seq) if s["act"] == act and free(i)]
            if not cand:
                break
            i = max(cand, key=hit)
            seq[i]["fx"] = [name]


def reframe(project, unit, t0, length, fmt, focus=True):
    """A crop that keeps the subject centred: one zoom for the shot, the centre gliding with the dancers. None when
    the source fills the frame as it is and there is nothing to follow. The editor's focal points for this format
    (picks "<fmt>", set in the source browser) take over the centre where there are any (focus=False: as derived)."""
    c = _reframe(project, unit, t0, length, fmt)
    picks = project.picks()
    if fmt == "portrait" and (picks.get(unit["id"]) or {}).get("follow"):   # the editor's chosen person
        dense = project.path("scores", f"faces_dense_{unit['id']}.npz")
        yf = _cached(dense, lambda f: dict(np.load(f))) if os.path.exists(dense) else None
        c = follow_override(c, yf, follow_keys(picks, unit), t0, length)
    return focus_override(c, focus_keys(picks, unit, fmt), t0, length) if focus else c


def follow_keys(picks, unit):
    """The people the editor chose to follow in a unit: [(source time, x, y)] where they clicked them, x and y
    fractions of the frame. picks.json: {"<export file>": {"follow": [["0:16.10", 0.52, 0.30], ...]}}."""
    p = picks.get(unit["id"]) or {}
    x0 = unit.get("export_t0", 0.0)
    return sorted((x0 + _secs(t), float(x), float(y)) for t, x, y in p.get("follow") or [])


FOLLOW_NEAR = 0.12         # a face this near the click (of the frame), within FOLLOW_WHEN s of it, is theirs
FOLLOW_WHEN = 0.75


def follow_override(crop, yf, keys, t0, length):
    """The portrait crop on the person the editor clicked (picks "follow", set in the source browser), for the
    whole shot: their face's track (face_tracks, anyone's -- not only the people the planner took the shot to be
    about) steers the strip, the face in the upper third, the strip widened for a close face. A click with no face
    near it (a back, a face too small to find) holds the strip on the click. Unchanged when no click is near the shot."""
    keys = [k for k in keys if t0 - FOCUS_REACH <= k[0] <= t0 + length + FOCUS_REACH]
    if not crop or not keys:
        return crop
    kt, kx, ky = min(keys, key=lambda k: abs(k[0] - (t0 + length / 2)))
    w, h, z = crop["w"], crop["h"], crop.get("z", 1.0)
    grid = np.arange(0, length + 0.01, 0.2)
    best, bd = None, FOLLOW_NEAR
    for tr in (face_tracks(yf, t0 - 1.0, t0 + length + 1.0) if yf is not None else []):
        j = int(np.argmin(np.abs(tr[0] - kt)))
        dd = float(np.hypot(tr[1][j] - kx, tr[2][j] - ky))
        if abs(tr[0][j] - kt) <= FOLLOW_WHEN and dd < bd:
            best, bd = tr, dd
    if best is None:
        x = np.full_like(grid, np.clip(kx, w / 2, 1 - w / 2))
        y = np.full_like(grid, np.clip(ky, h / 2, 1 - h / 2))
        out = dict(crop, follow="point")
    else:
        t, cx, cy, fh, _ = best
        tt = t - t0
        zoom = max(min(1.0, FACE_FIT * h / max(float(np.median(fh)), 1e-3)), 1 / max(z, 1.0))
        w, h = w / zoom, h / zoom
        k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2); k /= k.sum()   # ~0.3 s
        x = np.convolve(np.pad(np.interp(grid, tt, cx), 4, mode="edge"), k, mode="valid")
        y = np.convolve(np.pad(np.interp(grid, tt, cy + (0.5 - FACE_TOP) * h), 4, mode="edge"), k, mode="valid")
        x, y = np.clip(x, w / 2, 1 - w / 2), np.clip(y, h / 2, 1 - h / 2)
        seen = min(1.0, float(((tt >= 0) & (tt <= length)).sum()) / max(1.0, length / 0.5))
        out = dict(crop, w=round(w, 4), h=round(h, 4), z=round(z * zoom, 3), follow="face", facetrack=True,
                   **{f: round(max(float(crop.get(f, 0.0)), seen), 2) for f in ("face", "present", "inside", "head")})
    out["path"] = [[round(float(a), 2), round(float(b), 4), round(float(c), 4)] for a, b, c in zip(grid, x, y)]
    return out


def focus_keys(picks, unit, fmt):
    """The editor's focal points for a unit in one format: [(source time, x, y)], x and y fractions of the frame.
    picks.json: {"<export file>": {"portrait": [["0:16.10", 0.52, 0.45], ...]}}, times in the export's timeline."""
    p = picks.get(unit["id"]) or {}
    x0 = unit.get("export_t0", 0.0)
    return sorted((x0 + _secs(t), float(x), float(y)) for t, x, y in p.get(fmt) or [])


FOCUS_REACH = 0.5          # a focal point steers a shot that plays within this many seconds of it


def focus_override(crop, keys, t0, length):
    """The crop's centre moved onto the editor's focal points over the shot (held before the first and after the
    last, gliding between them); its size stays as derived. Unchanged when no point is near the shot."""
    keys = [k for k in keys if t0 - FOCUS_REACH <= k[0] <= t0 + length + FOCUS_REACH]
    if not crop or not keys:
        return crop
    kt, kx, ky = (np.array(v) for v in zip(*keys))
    w, h = crop["w"], crop["h"]
    grid = np.arange(0, length + 0.01, 0.2)
    x = np.clip(np.interp(grid + t0, kt, kx), w / 2, 1 - w / 2)
    y = np.clip(np.interp(grid + t0, kt, ky), h / 2, 1 - h / 2)
    return dict(crop, focus=True, path=[[round(float(a), 2), round(float(b), 4), round(float(c), 4)]
                                        for a, b, c in zip(grid, x, y)])


_LOADED = {}


def _cached(path, load, also=()):
    """A score file loaded once while it (and the companions in also) is unchanged: a plan crops thousands of spots
    from the same few files, and the review server's crops come from here too."""
    try:
        key = (os.stat(path).st_mtime_ns, *(os.stat(f).st_mtime_ns if os.path.exists(f) else 0 for f in also))
    except FileNotFoundError:
        return None
    hit = _LOADED.get(path)
    if not hit or hit[0] != key:
        hit = _LOADED[path] = (key, load(path))
    return hit[1]


def _reframe(project, unit, t0, length, fmt):
    W, H = SIZE[fmt]
    stem = project.path("scores", unit["score"])
    d = _cached(stem + ".npz", FR.load, (stem + "_focus.npz", stem + "_subject.npz"))
    d = dict(d) if d is not None else None                 # (the extras below are per call)
    dense = project.path("scores", f"faces_dense_{unit['id']}.npz")    # the persons step's scan (every 0.5 s,
    yf = _cached(dense if os.path.exists(dense) else project.path("scores", f"faces_{unit['id']}.npz"),   # multi-scale)
                 lambda f: dict(np.load(f)))                            # over the people step's ~1 s one
    if d is not None and yf is not None and not unit.get("rotate"):   # the unit's face scan (people step)
        d["yf"] = yf
    rf = _cached(project.path("scores", f"role_{unit['id']}.npz"), lambda f: dict(np.load(f)))
    if d is not None and rf is not None and not unit.get("rotate"):   # who reads as the follow (role step)
        d["role"] = rf
    src_w, src_h = sorted((int(d["size"][0]), int(d["size"][1])) if d is not None else (2160, 3840))[::-1]
    if unit["orient"] == "portrait":                        # as the reel sees it: after any rotation
        src_w, src_h = src_h, src_w
    cw0, ch0 = FR.crop_size(src_w / src_h, W / H)
    centre = None if cw0 >= 0.999 and ch0 >= 0.999 else dict(w=round(cw0, 4), h=round(ch0, 4), z=1.0, path=[[0, 0.5, 0.5]])
    if d is None or unit.get("rotate") or (unit["role"] == "transition" and centre is None):
        return centre                                        # B-roll in its own shape stays as shot
    pt = d["pose_t"]
    sel = (pt >= t0 - 0.5) & (pt <= t0 + length + 0.5)
    if sel.sum() < 2:
        return centre
    union, touch, _ = FR.track(d, pt[sel])
    narrow = cw0 < 0.5                                      # a 9:16 strip out of a 16:9 frame: little room for lag
    face = np.zeros(len(union), bool)
    person = unit["role"] != "transition" or people_broll(unit) or close_dancers(d, pt[sel])   # (scenery B-roll:
    #                                                         no one to follow; a class floor up close has dancers)
    one = False
    if person:                                              # a couple shares the frame; a group too wide for a
        union, face, one = FR.follow(d, pt[sel], union, cw0)  # strip: follow one of them
        if one:                                             # the group's edge contact no longer applies: the
            touch = None                                    # dancer's own box is checked against the frame below
    if narrow and any(k in (unit.get("tag") or "") for k in SPEAKER_TAGS):
        sp = speaker(d, pt[sel])                            # a panel: the one speaking, never the gap between
        if sp is not None:
            union, one, touch, person = sp, True, None, True
    _, z, _, _ = FR.framing(union, src_w, src_h, W, H, touch=touch)
    ok = ~np.isnan(union[:, 0])
    if not ok.any():
        return centre
    zz = float(np.median(z[ok])) if person else 1.0         # one zoom for the whole shot
    cw, ch = cw0 / zz, ch0 / zz
    sx = (union[ok, 0] + union[ok, 2]) / 2
    sy = union[ok, 1] + 0.45 * (union[ok, 3] - union[ok, 1])
    ts = pt[sel][ok] - t0
    grid = np.arange(0, length + 0.01, 0.1)

    def smooth(v, sig):
        x = np.interp(grid, ts, v)
        k = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2); k /= k.sum()
        return np.convolve(np.pad(x, 3 * sig, mode="edge"), k, mode="valid")
    pan = one == FR.PAN                                     # a group panned (reel.json "group": "pan")
    if narrow and person and not pan:                       # never through someone the subject's size beside them
        zz, sx = clean_centre(d, pt[sel][ok], union[ok], sx, zz, cw0, ch0)
        cw, ch = cw0 / zz, ch0 / zz
    if narrow:                                              # a subject sitting low in the strip (a band under a
        zz = lift(union[ok], zz, cw0, ch0, FR.zmax(src_w, src_h, W, H))   # tall rig): zoom so the crop moves down
        cw, ch = cw0 / zz, ch0 / zz
    if narrow and not pan:                                  # look room: a profile gets space ahead of their gaze
        room = np.clip((cw - (union[ok, 2] - union[ok, 0])) / 2, 0, None)   # never so far it cuts them (a
        sx = sx + np.clip(cw * gaze(d, pt[sel][ok], union[ok], R["look_player"] if unit["role"] != "dance" else R["look"]),
                          -room, room)                      # couple filling the strip gets none)
    uh_ = union[ok, 3] - union[ok, 1]                       # a subject taller than the crop keeps their head in it
    sy = union[ok, 1] + 0.45 * np.minimum(uh_, 0.95 * ch)   # (not a torso, hands and a mic stand)
    x = np.clip(smooth(sx, 3 if narrow else 6), cw / 2, 1 - cw / 2)   # ~0.3 / 0.6 s: like a camera operator
    # (the gaze shift is smoothed again on its own over ~1 s inside gaze(): a head turn doesn't whip the strip)
    y = np.clip(smooth(sy, 10), ch / 2, 1 - ch / 2)
    ft = None
    if narrow and person and not pan and "yf" in d:          # the strip off the face it should be on: follow the face
        main = _cached(project.path("scores", f'persons_{unit["id"].replace("#", "-")}.npz'), lambda f: dict(np.load(f)))
        ft = face_track(d["yf"], t0, length, grid, x, y, cw, ch, zz, main)
        if ft is not None:
            x, y, zz = ft["x"], ft["y"], zz * ft["zoom"]
            cw, ch = cw0 / zz, ch0 / zz
    # how well the crop holds the subject over the shot itself: present (mostly in the crop), inside (wholly in it,
    # not cut by the crop or the camera's own edge) and face (a face to the camera, in the crop), as fractions
    inn = (pt[sel] >= t0) & (pt[sel] <= t0 + length)
    stats = dict(present=0.0, inside=0.0, face=0.0, action=0.0, churn=0.0, blocked=0.0, split=0.0, body=1.0, low=0.5,
                 hide=0.0, head=1.0, back=0.0, torn=0.0, fill=0.0)
    if inn.any():
        tt = pt[sel][inn] - t0
        bx, by = np.interp(tt, grid, x), np.interp(tt, grid, y)
        b = union[inn]
        ovx = np.clip(np.minimum(b[:, 2], bx + cw / 2) - np.maximum(b[:, 0], bx - cw / 2), 0, None) / \
            np.maximum(b[:, 2] - b[:, 0], 1e-3)
        ovy = np.clip(np.minimum(b[:, 3], by + ch / 2) - np.maximum(b[:, 1], by - ch / 2), 0, None) / \
            np.maximum(b[:, 3] - b[:, 1], 1e-3)
        ov = np.nan_to_num(ovx * ovy)
        edge = (np.nan_to_num(b[:, 0]) < 0.01) | (np.nan_to_num(b[:, 2], nan=1) > 0.99)
        if touch is not None:
            edge |= np.asarray(touch)[inn][:, [0, 2]].any(1)
        # how much the one the crop is on is dancing (frame.activity) -- the biggest person near its centre -- not
        # someone behind them: a man walking across the floor in front of a class is a man walking
        ii = np.clip(np.searchsorted(pt, pt[sel][inn]), 0, len(pt) - 1)
        pb, act = d["boxes"][ii], FR.activity(d)[ii]
        pcx = (pb[..., 0] + pb[..., 2]) / 2 / d["aw"]
        ph = np.nan_to_num((pb[..., 3] - pb[..., 1]) / d["ah"])
        off = np.abs(np.nan_to_num(pcx, nan=9) - bx[:, None]) / (cw / 2)
        who = np.argmax(np.where(off <= 1, ph * (1.2 - off), -1), 1)
        seen = (np.arange(pb.shape[1])[None, :] == who[:, None]) & (off <= 1)
        # the one at the crop's centre keeps changing: people crossing through it, not a dancer the shot is on
        sx_ = np.take_along_axis(np.nan_to_num(pcx, nan=9), who[:, None], 1)[:, 0]
        churn = float((np.abs(np.diff(sx_)) > 0.35 * cw).mean()) if len(sx_) > 1 else 0.0
        # someone right up against the camera across the crop: head cut off by the frame's top, most of the crop
        # covered, no face of theirs in shot (a back, a torso) -- whoever the shot was meant to be about
        y0n, y1n = pb[..., 1] / d["ah"], pb[..., 3] / d["ah"]
        cov = np.clip(np.minimum(np.nan_to_num(pb[..., 2] / d["aw"]), bx[:, None] + cw / 2) -
                      np.maximum(np.nan_to_num(pb[..., 0] / d["aw"], nan=9), bx[:, None] - cw / 2), 0, None) / cw
        fseen = FR.face_seen(d, pt[sel][inn], pb, d["kp"][ii])
        wall = (np.nan_to_num(y0n, nan=1) <= 0.03) & (np.nan_to_num(y1n) - np.nan_to_num(y0n) >= 0.75) & \
            (cov >= 0.5) & ~fseen & (fseen.mean(0) < WALL_FACE)[None, :]   # (a dancer mid-turn shows a face
        # most of the shot: only someone with no face for nearly all of it is a wall)
        cut, body = sliced(d, ii, union[inn], bx, by, cw, ch)
        hide, head, back, torn = shown(pb / [d["aw"], d["ah"], d["aw"], d["ah"], 1], d["kp"][ii], fseen, who, seen.any(1),
                                 bx, by, cw, ch, ah=d["ah"])
        stats = dict(present=float((ov >= 0.6).mean()), inside=float(((ov >= 0.92) & ~edge).mean()),
                     face=float((face[inn] & (ov >= 0.6)).mean()),
                     action=float(np.where(seen, act, 0).max(1).mean()), churn=churn,
                     blocked=float(wall.any(1).mean()), split=cut, body=body, hide=hide, head=head, back=back,
                     torn=torn, fill=float(np.nanmedian((b[:, 3] - b[:, 1]) / ch)) if not np.isnan(b[:, 1]).all()
                     else 0.0,
                     low=float(np.nanmedian(((b[:, 1] + b[:, 3]) / 2 - (by - ch / 2)) / ch))
                     if not np.isnan(b[:, 1]).all() else 0.5)
        if pan:                                              # a pan passes people through its centre (cut-offs still
            stats.update(churn=0.0)                          # count: one person whole, never someone sliced)
    if ft is not None:                                       # on a face track: the face is the subject's, whatever
        held = ft["held"]                                    # the body tracker made of the shot
        stats.update(face=max(stats["face"], held), present=max(stats["present"], held),
                     inside=max(stats["inside"], held), head=max(stats["head"], held))
    return dict(w=round(cw, 4), h=round(ch, 4), z=round(zz, 3), **({"facetrack": True} if ft is not None else {}),
                path=[[round(float(a), 2), round(float(b), 4), round(float(c), 4)]
                      for a, b, c in zip(grid[::2], x[::2], y[::2])],
                **{k: round(v, 2) for k, v in stats.items()})


FACE_SCORE, FACE_MIN_H = 0.75, 0.04   # face_track: a face this confident and this tall (of the frame) counts
FACE_HELD = 0.8          # the strip already on the shot's main face this share of the scan: left as it is
FACE_TOP = 0.32          # where the face sits down the strip (of its height): eyes in the upper third
FACE_FIT = 0.4           # a face taller than this (of the strip) zooms out to fit, head and shoulders
FACE_GAIN = 0.25         # the face track replaces the strip only if it holds a face this much more of the shot


def face_tracks(yf, t0, t1, main=None):
    """The faces of a stretch linked into tracks, one per person: [(times, cx, cy, h, score)], biggest first. A face
    joins the track whose last face is near it (an eighth of the frame, a similar size), else starts its own. With
    main (the persons step: t, box (K, 3, 4) frame fractions), only the faces of the people the shot is about count
    -- in the head of one of their boxes -- never a bystander's face beside a subject whose own face is hidden."""
    sel = (yf["t"] >= t0) & (yf["t"] <= t1) & (yf["score"] >= FACE_SCORE) & (yf["h"] >= FACE_MIN_H)
    rows = sorted(zip(*(np.asarray(yf[k])[sel] for k in ("t", "cx", "cy", "h", "score"))))
    if main is not None and len(main["t"]):
        def theirs(t, cx, cy):
            j = int(np.argmin(np.abs(main["t"] - t)))
            if abs(main["t"][j] - t) > 0.3:
                return False
            for x0, y0, x1, y1 in main["box"][j]:
                if not np.isnan(x0) and x0 <= cx <= x1 and y0 - 0.05 <= cy <= y0 + 0.4 * (y1 - y0):
                    return True
            return False
        rows = [r for r in rows if theirs(r[0], r[1], r[2])]
    tracks = []
    for t, cx, cy, h, s in rows:
        best, bd = None, 0.125
        for tr in tracks:
            lt, lx, ly, lh, _ = tr[-1]
            dd = np.hypot(cx - lx, cy - ly)
            if t > lt and t - lt <= 1.5 and dd < bd and 0.6 <= h / lh <= 1.6:
                best, bd = tr, dd
        (best.append if best is not None else lambda r: tracks.append([r]))((t, cx, cy, h, s))
    tracks = [tuple(np.array(v) for v in zip(*tr)) for tr in tracks]
    return sorted(tracks, key=lambda tr: -float((tr[3] * tr[4]).sum()))


def face_track(yf, t0, length, grid, x, y, cw, ch, zz=1.0, main=None):
    """The portrait strip onto the shot's main face, when the body tracker's strip misses it (concepts §12): a crowd
    shot up close, where the pose boxes lose the person or settle on the one beside them. Only the faces of the
    shot's main people count (the persons step, when it has run). The main face is the
    biggest, steadiest one over the shot (size x confidence x time seen); a second face that fits beside it shares
    the strip (a couple, two friends), the main one weighted. The face sits in the upper third; one too big for the
    strip zooms out. Only a face near the strip already (the planner's subject: three quarters of its width) is
    followed; a bigger face elsewhere is someone else. None when the strip already holds the main face (FACE_HELD of the scan), no face is seen, or the
    new strip doesn't hold a face for clearly more of the shot (FACE_GAIN) than the old one.
    Returns x, y on grid, the zoom factor (<= 1: out), and held: the share of the shot the face is in the new strip."""
    tr = face_tracks(yf, t0 - 0.25, t0 + length + 0.25, main)
    near = [k for k in tr if (np.abs(k[1] - np.interp(k[0] - t0, grid, x)) <= 0.75 * cw).mean() >= 0.5]
    if not near:                                             # the planner's subject keeps the shot: only their own
        return None                                          # face re-centres it, never someone else's beside them
    t, cx, cy, h, _ = near[0]
    if len(t) < max(1, 0.4 * (length / 0.5)):                 # seen for under ~40% of the shot: not the shot's face
        return None
    tt = t - t0
    held = (np.abs(cx - np.interp(tt, grid, x)) <= cw / 2 - h * 0.35) & (np.abs(cy - np.interp(tt, grid, y)) <= ch / 2 - h * 0.6)
    if held.mean() >= FACE_HELD:
        return None
    zoom = max(min(1.0, FACE_FIT * ch / max(float(np.median(h)), 1e-3)), 1 / zz)   # a big close face: the strip
    #                                                         widens to fit it, as far as the frame allows
    cw2, ch2 = cw / zoom, ch / zoom
    mx = cx.copy()
    for t2, cx2, _cy2, h2, _ in near[1:2]:                    # a partner beside them, both in the strip
        near = np.interp(tt, t2 - t0, cx2, left=np.nan, right=np.nan)
        ok = ~np.isnan(near) & (np.abs(near - cx) + (h + np.interp(tt, t2 - t0, h2)) * 0.5 < 0.85 * cw2)
        if ok.mean() >= 0.5 and float(np.median(h2)) >= 0.6 * float(np.median(h)):
            mx = np.where(ok, 0.6 * cx + 0.4 * np.nan_to_num(near), cx)
    k = np.exp(-0.5 * (np.arange(-9, 10) / 3) ** 2); k /= k.sum()   # ~0.3 s, as the body track is smoothed
    xs = np.convolve(np.pad(np.interp(grid, tt, mx), 9, mode="edge"), k, mode="valid")
    ys = np.convolve(np.pad(np.interp(grid, tt, cy + (0.5 - FACE_TOP) * ch2), 9, mode="edge"), k, mode="valid")
    xs, ys = np.clip(xs, cw2 / 2, 1 - cw2 / 2), np.clip(ys, ch2 / 2, 1 - ch2 / 2)
    shot = max(1, int(round(length / 0.5)))

    def cover(xp, yp, cwp, chp):                             # share of the shot a confident face is in the strip
        seen = set()
        for t2, cx2, cy2, h2, _ in tr:
            tt2 = t2 - t0
            on = (tt2 >= 0) & (tt2 <= length) & (np.abs(cx2 - np.interp(tt2, grid, xp)) <= cwp / 2 - h2 * 0.35) & \
                (np.abs(cy2 - np.interp(tt2, grid, yp)) <= chp / 2 - h2 * 0.6)
            seen.update(np.round(tt2[on] * 2).astype(int).tolist())
        return min(1.0, len(seen) / shot)
    was, now = cover(x, y, cw, ch), cover(xs, ys, cw2, ch2)
    if now < was + FACE_GAIN:                                # no clear gain: the body tracker's strip stays
        return None
    return dict(x=xs, y=ys, zoom=zoom, held=round(now, 2))


FILL_GOOD = 0.55         # window_ok: a subject this tall in the crop (of its height) or more scores in full

RULES = dict(            # portrait rules: reel.json plan.<fmt>.portrait_rules (a profile can set them)
    low=0.66,            # the subject's centre this far down the crop (0 top, 1 bottom): sits low
    low_zmax=1.5,        # lift() may zoom this far to move the crop down (past the lossless ~1.12x of a 4K strip)
    stay=0.85,           # the subject in the strip in at least this share of the shot's frames
    hide=0.4,            # share of frames a faceless back covers most of a partner, neither face seen
    head=0.5,            # a performer's head in the crop in at least this share of frames
    back=0.6,            # share of frames the subject is seen from behind, no face of theirs or their partner's
    look=1 / 6,          # look room: a profile dancer's strip moves this far (of its width) toward their gaze
    look_player=0.25,    # ... a performer's further: their drums, keys, the crowd they sing to
    torn=0.3,            # share of the shot the dance partner is half in the strip, or in and out of it
    sliver=0.4,          # someone the frame edge slices, this tall (of the subject) or more, half in the strip: split
    rank=0.15)           # how far an unusable moment ranks down (another part of the clip, or recast, wins)
R = dict(RULES)          # the rules in force (build() sets them from the config)


def portrait_rules(cfg):
    """Set the portrait rules from plan.<fmt>.portrait_rules over the defaults (RULES)."""
    R.clear(); R.update(RULES)
    MEMBERS[:] = list((cfg or {}).get("members") or [])
    R.update({k: float(v) for k, v in ((cfg or {}).get("portrait_rules") or {}).items() if k in RULES})


LOW_AIM = 0.56           # where lift() aims to bring a low subject (R["low"]: what sits low, 0 top .. 1 bottom)


def facing(kp, box_w):
    """Which way a head faces from its COCO keypoints (F,): -1 (to the frame's left) .. 1 (right), 0 to the camera
    or unknown -- the nose's offset from the visible ears' (else eyes') midpoint, over a quarter of the box width."""
    nose_ok = kp[:, 0, 2] > 0.3
    ears = kp[:, 3:5]
    eyes = kp[:, 1:3]
    def mid(pts):
        w = (pts[..., 2] > 0.3).astype(float)
        return (pts[..., 0] * w).sum(1) / np.maximum(w.sum(1), 1e-6), w.sum(1) > 0
    ex, eok = mid(ears)
    yx, yok = mid(eyes)
    ref = np.where(eok, ex, yx)
    f = np.clip((kp[:, 0, 0] - ref) / np.maximum(0.25 * box_w, 1e-3), -1, 1)
    return np.where(nose_ok & (eok | yok), f, 0.0)


SPEAKER_TAGS = ("panel", "talk", "interview", "speech")   # footage of people talking: frame the speaker


def speaker(d, times):
    """Of the people on a panel, the one speaking, held for the shot: the one most face-on to the room (whoever they
    talk to turns to them), a hand at the mouth (the mic) breaking a tie. Their box per time as frame fractions
    (x0, y0, x1, y1), or None with fewer than two people the size of the largest."""
    pt = d["pose_t"]
    ii = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    bx, kp = d["boxes"][ii, :, :4].astype(float), d["kp"][ii]
    h = np.nan_to_num(bx[..., 3] - bx[..., 1])
    big = h >= 0.75 * h.max(1, keepdims=True)
    if (big.sum(1) < 2).mean() > 0.5:
        return None
    bw = np.nan_to_num(bx[..., 2] - bx[..., 0])
    turn = np.stack([np.abs(facing(kp[:, p], bw[:, p])) for p in range(kp.shape[1])], 1)
    nose = kp[..., 0, :2]
    hand = np.min([np.hypot(*(kp[..., w, :2] - nose).transpose(2, 0, 1)) / np.maximum(h, 1) for w in (9, 10)], 0)
    hand = np.where((kp[..., [9, 10], 2].max(-1) > 0.3) & (kp[..., 0, 2] > 0.3), hand, 9)
    score = np.where(big, turn + 0.3 * (hand > 0.35), 9).mean(0)
    who = int(score.argmin())
    cx = np.nanmedian((bx[:, who, 0] + bx[:, who, 2]) / 2)
    out = np.full((len(ii), 4), np.nan)
    for f in range(len(ii)):                                # the same person each frame: nearest to them
        c = (bx[f, :, 0] + bx[f, :, 2]) / 2
        p = int(np.nanargmin(np.where(big[f], np.abs(c - cx), np.inf))) if big[f].any() else -1
        if p >= 0:
            out[f] = bx[f, p]
            cx = c[p]
    ah = float(d["ah"]) if np.ndim(d["ah"]) == 0 else float(d["ah"][0])
    out[:, [0, 2]] /= float(d["aw"]); out[:, [1, 3]] /= ah
    return out


def gaze(d, times, union, k):
    """Look room, per time: the strip-centre shift (in strip widths) toward where the subject faces -- k of a strip
    for a full profile (1/6 puts the face a third of the way in from the far side), eased by how side-on the head
    is, smoothed over ~1 s so a head turn doesn't whip the strip."""
    if k <= 0 or len(times) == 0:
        return np.zeros(len(times))
    pt = d["pose_t"]
    ii = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    _, main, _ = _people(d, ii, union)
    f = np.arange(len(ii))
    kp = d["kp"][ii, np.maximum(main, 0)]
    bw = np.nan_to_num(d["boxes"][ii, np.maximum(main, 0), 2] - d["boxes"][ii, np.maximum(main, 0), 0])
    look = np.where(main >= 0, facing(kp, bw), 0.0)
    look = np.where(np.abs(look) >= 0.35, look, 0.0)        # a three-quarter view or less: no shift
    n = max(1, min(len(f), 5))                              # pose at 5 fps: ~1 s
    look = np.convolve(np.pad(look, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), mode="same")[n:-n]
    return k * look


def lift(union, zz, cw0, ch0, zm):
    """The zoom for a subject sitting low in a portrait strip: the least zoom (from zz, up to max(zm, R["low_zmax"]))
    at which the crop, moved down, holds the subject's centre at LOW_AIM of its height (R["low"] at worst) -- while the subject
    still fits (its height within 90% of the crop's, its width within the strip). zz unchanged when it doesn't sit
    low, or no zoom up to the ceiling lifts it clear (a player in the bottom quarter: ranked down instead)."""
    cy = float(np.median((union[:, 1] + union[:, 3]) / 2))
    uh = float(np.median(union[:, 3] - union[:, 1]))
    uw = float(np.median(union[:, 2] - union[:, 0]))

    def pos(z):
        ch = ch0 / z
        by = float(np.clip(cy, ch / 2, 1 - ch / 2))
        return (cy - (by - ch / 2)) / ch
    if pos(zz) <= R["low"]:
        return zz
    best = zz
    for z in np.linspace(zz, max(zm, R["low_zmax"], zz), 13)[1:]:
        if uh > 0.9 * ch0 / z or uw > 0.95 * cw0 / z:          # tighter would cut the subject
            break
        best = float(z)
        if pos(z) <= LOW_AIM:
            break
    return round(best, 3) if pos(best) <= R["low"] else zz      # (a zoom that still leaves it low: not worth it)


PEER_H = 0.75            # someone at least this tall (of the subject) beside them counts as the subject's size
SLICE_COST = 2.5         # a crop centre's cost per such person cut through (vs. a crop width off the subject)
MOVE_COST = 6.0          # ... and per strip width it moves between pose frames: a framing that's good holds, a
                         # moment's slice (an arm thrown out) doesn't jerk it away; a lasting one (2+ frames) does


def steady(cost, cand, cw, move=MOVE_COST):
    """The cheapest path of crop centres through a shot (Viterbi over cost (C, F), centres cand (C,)): each frame's
    own cost plus move per strip width moved from the frame before. Returns the candidate index per frame."""
    C, F = cost.shape
    step = move * np.abs(cand[:, None] - cand[None, :]) / cw          # (to, from)
    acc, back = cost[:, 0].copy(), np.zeros((F, C), int)
    for f in range(1, F):
        tot = acc[None, :] + step
        back[f] = tot.argmin(1)
        acc = cost[:, f] + tot[np.arange(C), back[f]]
    k = np.zeros(F, int)
    k[-1] = int(acc.argmin())
    for f in range(F - 1, 0, -1):
        k[f - 1] = back[f, k[f]]
    return k


def _people(d, ii, union):
    """At pose frames ii: the people as fractions of the frame (F, P, 4), the subject (the tallest pose box mostly
    inside the subject's union, -1 if none) and who is the subject's size (PEER_H) -- the ones a crop must not cut."""
    nb = d["boxes"][ii, :, :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    ok = ~np.isnan(nb[..., 0])
    hgt = np.where(ok, nb[..., 3] - nb[..., 1], 0)
    m = FR.members(d["boxes"][ii], union, d["aw"], d["ah"])
    main = np.where(m.any(1), np.argmax(np.where(m, hgt, -1), 1), -1)
    mh = np.where(main >= 0, hgt[np.arange(len(ii)), np.maximum(main, 0)], 9)
    return nb, main, ok & (hgt >= PEER_H * mh[:, None])


def _cut(nb, x0, x1):
    """How badly each box is cut by a crop's sides (F, P) or (C, F, P): 0 wholly in or out, 1 cut down the middle."""
    vis = np.clip(np.minimum(nb[..., 2], x1) - np.maximum(nb[..., 0], x0), 0, None) / \
        np.maximum(nb[..., 2] - nb[..., 0], 1e-3)
    return np.clip(np.minimum(vis, 1 - vis) / 0.3, 0, 1), vis


def clean_centre(d, times, union, sx, zz, cw0, ch0):
    """A portrait strip's centre, frame by frame, moved off anyone the subject's size it would cut through: the
    partner of a couple too far apart for the strip is left out whole rather than cut in half, someone half in at the
    edge goes out, while the subject stays whole and near the centre. The shot's zoom widens to 1 when that cuts
    through fewer people. Returns (zoom, centres)."""
    pt = d["pose_t"]
    ii = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
    nb, main, peer = _people(d, ii, union)
    if (main < 0).all():
        return zz, sx
    F = len(ii)
    sub = nb[np.arange(F), np.maximum(main, 0)]

    def best(z):
        cw = cw0 / z
        cand = np.linspace(cw / 2, 1 - cw / 2, 41)[:, None]                  # (C, 1)
        cut, _ = _cut(nb[None], (cand - cw / 2)[..., None], (cand + cw / 2)[..., None])   # (C, F, P)
        others = np.where(peer[None] & (np.arange(nb.shape[1]) != main[:, None])[None], cut, 0).sum(2)
        _, sv = _cut(sub[None], cand - cw / 2, cand + cw / 2)
        cost = np.abs(cand - sx[None]) / cw + SLICE_COST * others + 6 * np.clip(0.95 - sv, 0, None)
        cost = np.where((main >= 0)[None], cost, np.abs(cand - sx[None]) / cw)
        k = steady(cost, cand[:, 0], cw)
        return cand[k, 0], cost[k, np.arange(F)].sum(), others[k, np.arange(F)].sum()
    c, cost, n = best(zz)
    if zz > 1.0:
        c1, cost1, n1 = best(1.0)
        if n1 < 0.7 * n:
            return 1.0, c1
    return zz, c


def sliced(d, ii, union, bx, by, cw, ch):
    """Over a shot: the share of frames where someone the subject's size is cut through by the crop's sides, and the
    share where the subject's body (hips) is in the crop -- not only a head and shoulders that hide the dancing."""
    nb, main, peer = _people(d, ii, union)
    cut, vis = _cut(nb, (bx - cw / 2)[:, None], (bx + cw / 2)[:, None])
    me = np.arange(nb.shape[1]) == main[:, None]
    others = np.where(peer & ~me, cut, 0)
    # someone the camera's own frame edge slices (close to the camera, half out of the source) and at least half in
    # the strip: a sliver of a partner -- an arm, part of a dress -- or half a near-camera dancer at the strip's edge.
    # Their box is clipped, so they can fall under PEER_H: counted here, down to R["sliver"] of the subject's height.
    ok = ~np.isnan(nb[..., 0])
    mh = np.where(main >= 0, np.nan_to_num(nb[np.arange(len(ii)), np.maximum(main, 0), 3] -
                                           nb[np.arange(len(ii)), np.maximum(main, 0), 1]), 9)[:, None]
    hh = np.nan_to_num(nb[..., 3] - nb[..., 1])
    edge = (np.nan_to_num(nb[..., 0], nan=0.5) <= 0.01) | (np.nan_to_num(nb[..., 2], nan=0.5) >= 0.99)
    sliver = ok & ~me & edge & (hh >= R["sliver"] * mh) & (vis > 0.5)
    others = np.where(sliver, 1.0, others)
    kp = d["kp"][ii, np.maximum(main, 0)]                                    # (F, 17, 3)
    kx, ky = kp[..., 0] / d["aw"], kp[..., 1] / d["ah"]
    inside = (kp[..., 2] > 0.3) & (np.abs(kx - bx[:, None]) <= cw / 2) & (np.abs(ky - by[:, None]) <= ch / 2)
    has = main >= 0
    if not has.any():
        return 0.0, 1.0
    return float((others.max(1) > 0.5)[has].mean()), float(inside[:, [11, 12]].any(1)[has].mean())


WALL_FACE = 0.25         # a person showing a face in this share of a shot's frames or more is never a wall


def shown(nb, kp, fseen, who, has, bx, by, cw, ch, ah=None):
    """Over a shot, as shares of the frames the crop has a subject: hide -- someone with no face to the camera (a back)
    covering most of a partner's box, and no face of either seen (a back-turned follow hiding the lead, whatever her
    hair is doing); head -- the subject's head (box top) inside the crop, not cut off by its top; back -- the subject
    seen from behind (no face seen, no nose in the pose: a performer's back of the head, not a player looking down).
    nb: pose boxes as frame fractions (F, P, 5); who: the subject per frame; has: a subject in the crop."""
    F, P = nb.shape[:2]
    if not has.any():
        return 0.0, 1.0, 0.0, 0.0
    ok = ~np.isnan(nb[..., 0])
    x0, y0, x1, y1 = (np.nan_to_num(nb[..., k]) for k in range(4))
    incrop = ok & (np.minimum(x1, bx[:, None] + cw / 2) - np.maximum(x0, bx[:, None] - cw / 2) > 0.3 * (x1 - x0))
    area = np.maximum((x1 - x0) * (y1 - y0), 1e-6)
    hide = np.zeros(F, bool)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            ov = np.clip(np.minimum(x1[:, i], x1[:, j]) - np.maximum(x0[:, i], x0[:, j]), 0, None) * \
                np.clip(np.minimum(y1[:, i], y1[:, j]) - np.maximum(y0[:, i], y0[:, j]), 0, None) / area[:, j]
            hide |= incrop[:, i] & incrop[:, j] & (ov >= 0.6) & ((y1[:, j] - y0[:, j]) >= 0.6 * (y1[:, i] - y0[:, i])) \
                & ~fseen[:, i] & ~fseen[:, j]
    f = np.arange(F)
    top = y0[f, who]
    if ah:                                                  # the head from its own points (nose, eyes, ears), less
        hk = kp[f, who, :5]                                 # a forehead: a close dancer's raised arms put her box
        hy = np.where(hk[..., 2] > 0.3, hk[..., 1] / float(ah), np.nan)   # top at the frame's edge, not her head
        kt = np.nanmin(np.where(np.isnan(hy).all(1, keepdims=True), 9, hy), 1) - 0.04
        top = np.where(kt < 8, kt, top)
    head = (top >= by - ch / 2 - 0.02) & (top > 0.01)       # (cut by the strip's top, or the camera's own)
    # the subject's own partner: anyone whose box overlaps the subject's by a third of its own -- their faces count,
    # bystanders' faces elsewhere in the strip don't
    sx0, sy0, sx1, sy1 = x0[f, who][:, None], y0[f, who][:, None], x1[f, who][:, None], y1[f, who][:, None]
    pov = np.clip(np.minimum(x1, sx1) - np.maximum(x0, sx0), 0, None) * \
        np.clip(np.minimum(y1, sy1) - np.maximum(y0, sy0), 0, None) / area
    partner = ok & (pov >= 0.33) & (np.arange(P)[None, :] != who[:, None])
    back = ~fseen[f, who] & (kp[f, who, 0, 2] < 0.3) & ~(partner & fseen).any(1)
    # the dance partner beside them (close, nearly as tall, level with them), over the whole shot: cleanly in the strip
    # (70%+) or cleanly out of it is framing; half in, or in and out from one moment to the next, is not
    sw, shh = (sx1 - sx0), (sy1 - sy0)
    gap = np.maximum(x0 - sx1, sx0 - x1)
    level = (np.minimum(y1, sy1) - np.maximum(y0, sy0)) > 0.5 * shh
    near = ok & (np.arange(P)[None, :] != who[:, None]) & ((y1 - y0) >= 0.6 * shh) & (gap < 0.5 * sw) & level
    vis = np.clip(np.minimum(x1, bx[:, None] + cw / 2) - np.maximum(x0, bx[:, None] - cw / 2), 0, None) / \
        np.maximum(x1 - x0, 1e-3)
    pv = np.where(near, vis, -1).max(1)                     # the partner's share in the strip (-1: no partner)
    sel = has & (pv >= 0)
    if sel.sum() >= 2:
        inn_, out_ = (pv[sel] >= 0.7), (pv[sel] <= 0.1)
        torn = float(((~inn_) & (~out_)).mean() + min(inn_.mean(), out_.mean()))
    else:
        torn = 0.0
    return float(hide[has].mean()), float(head[has].mean()), float(back[has].mean()), torn


IDLE = 0.55              # frame.activity of the crop's subject below this: walking or standing, not dancing
SPLIT = 0.5              # portrait: share of frames with someone the subject's size cut through, at the full mark-down
BODY = 0.5               # portrait: share of frames with the subject's hips in the crop, below which it marks down
BLOCKED = 0.3            # share of frames someone right up against the camera covers the crop (head cut, no face)
CHURN = 0.45             # share of frames where the one at the crop's centre is someone else: a crowd passing


def plays(c, role, portrait=None):
    """How a stretch plays from its reframe stats (0..1): the subject lost from the crop, walking or standing rather
    than dancing, a crowd crossing the crop, a performer never wholly in it, or someone blocking the camera, each mark
    it down."""
    g = 0.04 if c.get("present", 1.0) < 0.6 else 1.0
    if c.get("action") is not None and c["action"] < IDLE:
        g *= 0.15
    if c.get("churn", 0.0) > CHURN:
        g *= 0.2
    if c.get("inside", 1.0) < 0.25 and role != "dance":
        g *= 0.3
    if c.get("blocked", 0.0) > BLOCKED:
        g *= 0.15
    portrait = c.get("w", 1.0) < 0.5 if portrait is None else portrait
    if portrait:                                             # someone the subject's size cut in half beside them, or
        g *= 1 - 0.7 * min(c.get("split", 0.0) / SPLIT, 1)  # a partner sliced to a sliver by the frame edge
    if c.get("w", 1.0) < 0.5:                                # a portrait strip: only a head and shoulders
        g *= 0.35 + 0.65 * min(c.get("body", 1.0) / BODY, 1)
    if portrait:
        g *= R["rank"] if unusable(c, role) else 1.0
    return g


def unusable(c, role):
    """A portrait stretch no subject plays in, whatever its kind (a hair flip, a hit): the subject down at the bottom
    under the rig (low), the subject walking out of the strip (exit), a back-turned dancer hiding the partner (hide), a
    head cut off by the crop's or the camera's top (head), or the subject seen from behind with no face of theirs or their partner's (back: a bystander's face
    elsewhere in the strip doesn't rescue it). Returns the reason, or None."""
    if c.get("w", 1.0) < 0.5 and c.get("low", 0.5) > R["low"]:
        return "low"
    if c.get("hide", 0.0) > R["hide"]:
        return "hide"
    if role == "dance" and c.get("torn", 0.0) > R["torn"]:   # the follow half out of the strip, half the time
        return "torn"
    if c.get("present", 1.0) < R["stay"]:                   # walks out of the strip during the shot
        return "exit"
    if c.get("head", 1.0) < R["head"]:                       # a head cut off at the top: no face, dancer or player
        return "head"
    if c.get("back", 0.0) > R["back"]:                       # seen from behind, no face of theirs or their partner's
        return "back"
    return None


def spot_rater(project, fmt):
    """rate(record, t): how a shot around source time t would play (plays), for moments the plan makes up itself
    (cover_people); transitions and unframed footage pass."""
    seen = {}

    def rate(r, t):
        k = (r["id"], round(t, 1))
        if k not in seen:
            c = reframe(project, r["unit"], t - 0.8, 2.4, fmt) if r["role"] != "transition" or people_broll(r["unit"]) \
                else None
            seen[k] = plays(c, r["role"], fmt == "portrait") if c else 1.0
        return seen[k]
    return rate


def assess(project, recs, fmt, cfg):
    """How each moment would actually play in this format, over the stretch a shot shows (-0.8..+1.6 s): the subject
    held in the crop and a face to the camera (reframe's stats), and camera shake (shake.py) as the viewer sees it --
    a crop a third of the frame wide shows the camera's jitter three times over. A moment too shaky to use, or whose
    subject keeps leaving the crop, drops out (another part of the clip takes its place); a whole subject ranks up;
    in portrait a face ranks up a lot -- most of all on the social floor, where it's otherwise people's backs. A
    moment where the one in the crop is walking or standing rather than dancing (IDLE), where people keep crossing
    through the crop (CHURN), where someone right up against the camera blocks it (BLOCKED), or where a performer is
    never wholly in it, ranks well down, so another part of the clip takes its place.
    Editor's picks are never dropped. Reframes are cached in hold_cache.json."""
    from . import role, shake
    lo, hi = cfg["shake"]
    key = "score_p" if fmt == "portrait" else "score_l"
    path = project.path("hold_cache.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for r in recs:
        u = r["unit"]
        bf = project.path("scores", f"band_{u['id']}.npz")
        if r["role"] == "transition" and role.band_footage(project, u) and os.path.exists(bf):
            z = np.load(bf)                       # where a player is on screen and in focus (cover_people keeps to it)
            r["band_ok"] = (z["t"], (z["p"] >= role.BAND_P) & (z["sharp"] >= role.BAND_SHARP))
        yf = "|yf" if os.path.exists(project.path("scores", f"faces_{u['id']}.npz")) else ""   # (face scan: re-assess)
        for m in r["moments"]:
            k = f"{fmt}|{u['id']}|{m['t']}{yf}|c4"
            if k not in cache:
                c = reframe(project, u, m["t"] - 0.8, 2.4, fmt) or {}
                cache[k] = [c.get("present", 1.0), c.get("inside", 1.0), c.get("face", 0.0), c.get("w", 1.0),
                            c.get("action"), c.get("churn", 0.0), c.get("blocked", 0.0), c.get("split", 0.0),
                            c.get("body", 1.0), c.get("low", 0.5), c.get("hide", 0.0), c.get("head", 1.0),
                            c.get("back", 0.0), c.get("torn", 0.0)]
            row = cache[k] + [0.5, 0.0, 1.0, 0.0, 0.0][len(cache[k]) - 9:] if len(cache[k]) < 14 else cache[k]
            present, inside, face, cw, action, churn, blocked, split, body, low, hide, head, back, torn = row[:14]
            st = dict(present=present, inside=inside, action=action, churn=churn, blocked=blocked, w=cw, split=split,
                      body=body, low=low, hide=hide, head=head, back=back, torn=torn)
            j = shake.jitter(project, u, m["t"] - 0.8, m["t"] + 1.6)
            j = j / max(cw, 0.2) if j is not None else None
            m["jitter"] = round(j, 5) if j is not None else None
            sc = m.get(key, m["score"])
            why = unusable(st, r["role"]) if fmt == "portrait" and not m.get("pick") else None
            if why:                                          # any footage, band and stage too: no subject plays here
                sc *= R["rank"]
                m["unusable"] = why
            if j is not None and j > hi and not m.get("pick"):
                sc *= 0.03
                m["unusable"] = m.get("unusable") or "shake"      # (band_in and the rest pass it over too)
            elif j is not None and j > lo:
                sc *= 1 - 0.35 * min((j - lo) / (hi - lo), 1)
            if r["role"] == "transition" and role.band_footage(project, u):     # the players in shot, sharp -- not the
                b = role.musician(project, u, m["t"] - 0.4, m["t"] + 1.2)  # crowd, a back or a blur between them
                if b is not None:
                    p, sh = b
                    m["band"] = round(p, 2)
                    if (p < role.BAND_P or sh < role.BAND_SHARP) and not m.get("pick"):
                        sc *= 0.03
                    else:
                        sc *= 0.3 + p * min(sh / role.SHARP_FULL, 1.0)
            perf = people_broll(u)                           # a performer: framed like a subject (not the dance
            if r["role"] != "transition" or (perf and fmt == "portrait"):   # floor's idle / crossing rules)
                m["present"], m["inside"], m["face_vis"] = present, inside, face
                m["action"] = action
                social = r["role"] == "dance"
                rs = dict(st, low=0.5, hide=0.0, head=1.0, back=0.0, torn=0.0)
                if perf:
                    rs.update(action=None, churn=0.0)
                g = 1.0 if m.get("pick") else plays(rs, r["role"], fmt == "portrait")   # (the rest: ranked above)
                g *= (0.3 + 0.7 * inside) if social else (0.45 + 0.55 * inside)
                if fmt == "portrait":
                    g *= ((0.3 + 1.4 * face) * (0.25 if face < 0.3 else 1)) if social else (0.5 + 1.0 * face)
                else:
                    g *= 0.85 + 0.3 * face
                m["plays_" + fmt[0]] = round(g, 3)
                sc *= g
            m[key] = sc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(cache, open(path, "w"))


def clock(music):
    """Seconds from the reel's first beat to its beat k. One song: k beats of its fitted period. Several songs (music.join):
    the joined grid's real beat times, since the tempo may change where one song hands over to the next."""
    beat = music["beat"]
    if not music.get("sections"):
        return lambda k: k * beat
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    g = bt[i0:] - bt[i0]
    per = float(g[-1] - g[-2])
    return lambda k: float(g[k]) if k < len(g) else float(g[-1] + (k - len(g) + 1) * per)


def in_point(s, beat):
    """Source in-point: the move's peak lands on a beat inside the shot -- mid-shot for a speed effect, about 40% in
    for a plain shot, on the cut itself for a one-beat hit."""
    u, L = s["unit"], s["beats"] * beat
    if s.get("span"):                                        # the editor's selection: it starts at their in-point
        return min(max(s["span"][0], u["t0"]), max(u["t1"] - L, u["t0"]))
    if "lead_beats" in s:                                    # a velocity edit: the landing on its chosen beat
        lead = s["lead_beats"] * beat
    elif any(f in SPEED for f in s["fx"]):
        lead = (s["beats"] // 2) * beat
    else:
        lead = round(0.4 * s["beats"]) * beat
    t = min(max(s["t"] - lead, u["t0"]), max(u["t1"] - L, u["t0"]))
    return clear_of_bans(t, L, u)


def in_bans(t, L, u):
    """A shot at [t, t + L] plays some of the editor's banned stretches (more than 0.1 s of one)."""
    return any(min(t + L, b) - max(t, a) > 0.1 for a, b in u.get("bans") or [])


def clear_of_bans(t, L, u):
    """A shot's in-point moved, as little as it can, so its window [t, t + L] stays out of the editor's banned
    stretches (picks "ban": a moment just outside a ban would otherwise bring the banned shot back). Unchanged when
    nothing fits."""
    bans = u.get("bans") or []
    hit = lambda x: any(min(x + L, b) - max(x, a) > 0.1 for a, b in bans)
    if not hit(t):
        return t
    cands = [x for a, b in bans for x in (b, a - L) if u["t0"] <= x <= u["t1"] - L and not hit(x)]
    return min(cands, key=lambda x: abs(x - t)) if cands else t


def build(project, fmt, music, moments, stinger_len=0.0, direction=None, picks=None, assess_shots=True):
    """The plan for one format: shots with beat-locked record positions and source in-points. picks: the editor's
    overrides (apply_picks); None reads them from the project. assess_shots: rank moments on how they play in this
    format (framing, face, shake) first."""
    picks = project.picks() if picks is None else picks
    cfg = config(project, fmt)
    portrait_rules(cfg)
    end = pick_ending(picks)
    if end:                                                  # the editor's ending pick over reel.json's
        cfg["ending"] = [*end, (list(cfg.get("ending") or []) + [None] * 4)[3] or 0.5]
    if os.path.exists(project.path("people.json")):          # everyone in it (the people step)
        cfg["people"] = project.load("people.json")
    moments = [r for r in copy.deepcopy(moments) if project.in_cut(r["unit"])]   # a cut: its own footage
    if picks:
        apply_picks(moments, picks, only=cfg.get("only_picks", True))   # a picked clip: the editor's picks alone
    picks_only(moments, project.config.get("picks_only", ()))
    condensed = []
    if cfg.get("pick_hold") and picks:                       # long picks: their best part; many: condensed
        b = music.get("beat") or float(np.median(np.diff(music["beats"])))
        condensed = pick_pieces(project, moments, fmt, b, cfg["pick_hold"], music["length"] / b, cfg.get("pick_share"),
                                 cfg.get("ending"), cfg.get("close"), cfg.get("pick_clip_share"), cfg.get("pick_grow"),
                                 cfg.get("pick_whole") or (), cfg.get("pick_by"), pick_like(project, cfg))
    kp = project.config.get("keep")
    kp = "plan_{fmt}_rendered.json" if kp is True else kp    # true: the last cut rendered (render.run saves it)
    kp = kp and kp.format(fmt=project.cut_name(fmt))
    if kp and os.path.exists(project.path(kp)):
        keep_previous(moments, project.load(kp)["shots"], picks)
    close_only(project, moments, project.config.get("close_only"))
    reserve_ending(moments, cfg)
    if assess_shots:
        assess(project, [r for r in moments if r["moments"] and (fmt == "portrait" or r["orient"] == "landscape")],
               fmt, cfg)
    music = dict(music, beat=music.get("beat") or float(np.median(np.diff(music["beats"]))))   # fitted period
    beat = music["beat"]
    total_beats = len(music["beats"]) if music.get("sections") else round(music["length"] / beat)
    if cfg["acts"] is None and music.get("acts"):            # several songs: the acts sit on the songs
        cfg["acts"] = music["acts"]
    cfg["sets"] = song_sets(project, music)
    video_beats = total_beats - (round(STINGER_OVERLAP / beat) if stinger_len else 0)   # the stinger covers the end
    seq, acts = story(fmt, cfg, music, moments, video_beats, total_beats, direction, spot_rater(project, fmt))
    if fmt == "portrait" and assess_shots:
        recast(project, fmt, seq, moments, beat)
    act_logos(seq, project.config.get("act_logos"))
    plan = place(project, fmt, seq, music, cfg, direction, stinger_len, acts)
    if fmt == "portrait" and assess_shots:
        recast_placed(project, fmt, plan["shots"], moments, cfg)
    fm = [r for r in moments if fmt == "portrait" or r["orient"] == "landscape"]
    pace_shots(plan["shots"], cfg)                            # first: a paced shot plays more source, judged so
    ensure_units(project, fmt, plan["shots"], fm, cfg)
    snap_windows(project, fmt, plan["shots"], fm, cfg)        # last: whichever pass placed it, on passing windows
    ensure_units(project, fmt, plan["shots"], fm, cfg)        # (a unit whose only shot that recast moves away)
    if pace_shots(plan["shots"], cfg):                        # a slot that changed footage takes its own pace:
        snap_windows(project, fmt, plan["shots"], fm, cfg)    # on passing windows at that length too
    end_clean(project, fmt, plan["shots"], fm, cfg)
    mo = [r for r in moments if r["moments"] and (fmt == "portrait" or r["orient"] == "landscape")]
    plan["units_total"] = len({r["id"] for r in mo})
    plan["music_length"] = music["length"]
    plan["picks_condensed"], plan["pick_hold"] = condensed, cfg.get("pick_hold")
    return plan


RECAST = 0.25            # portrait: a planner's shot framing worse than this over the stretch it plays is recast
RECAST_TRY = 20          # candidates reframed per recast shot (the best-ranked first)


def framed(c, role):
    """How a crop plays in portrait (0..~1.7), as assess weighs a moment: plays(), the subject whole, a face."""
    if not c:
        return 1.0
    social, face, inside = role == "dance", c.get("face", 0.0), c.get("inside", 1.0)
    g = plays(c, role, True) * ((0.3 + 0.7 * inside) if social else (0.45 + 0.55 * inside))
    return g * (((0.3 + 1.4 * face) * (0.25 if face < 0.3 else 1)) if social else (0.5 + 1.0 * face))


def window_ok(project, rec, t0, t1, fmt, rules=None):
    """Does the stretch t0..t1 (source seconds) of rec's unit pass every framing rule in this format? A pure check,
    for scrubbing a clip densely and choosing only from windows that already work. Returns dict(ok, score, crop,
    reasons, jitter): score is how well it frames -- framed() / plays() with the subject's own face term, times its
    size in the crop (an editor picks one clear subject, large, face to the lens: not a busy, "sharp" crowd), times
    the shake factor; no energy or kind in it. crop is reframe's for exactly this span, reasons every rule it fails
    (names below). rules: overrides of R (portrait_rules) and "shake": [lo, hi]. Scenery B-roll always passes."""
    from . import shake
    u, role = rec["unit"], rec["role"]
    saved = dict(R)
    try:
        R.update({k: v for k, v in (rules or {}).items() if k in RULES})
        lo, hi = (rules or {}).get("shake") or CFG[fmt]["shake"]
        if role == "transition" and not people_broll(u):
            return dict(ok=True, score=1.0, crop=None, reasons=[], jitter=None)
        c = reframe(project, u, t0, t1 - t0, fmt) or {}
        portrait = fmt == "portrait"
        why = []
        if c.get("present", 1.0) < 0.6:
            why.append("lost")
        if portrait:
            for name, bad in (("exit", c.get("present", 1.0) < R["stay"]),
                              ("low", c.get("w", 1.0) < 0.5 and c.get("low", 0.5) > R["low"]),
                              ("hide", c.get("hide", 0.0) > R["hide"]),
                              ("torn", role == "dance" and c.get("torn", 0.0) > R["torn"]),
                              ("head", c.get("head", 1.0) < R["head"]),
                              ("back", c.get("back", 0.0) > R["back"]),
                              ("split", c.get("split", 0.0) > SPLIT),
                              ("body", c.get("w", 1.0) < 0.5 and c.get("body", 1.0) < BODY),
                              ("noface", role == "dance" and c.get("face", 1.0) < 0.3)):
                if bad:
                    why.append(name)
        perf = people_broll(u)
        if not perf and c.get("action") is not None and c["action"] < IDLE:
            why.append("idle")
        if not perf and c.get("churn", 0.0) > CHURN:
            why.append("churn")
        if c.get("blocked", 0.0) > BLOCKED:
            why.append("blocked")
        j = shake.jitter(project, u, t0, t1)
        j = j / max(c.get("w", 1.0), 0.2) if j is not None else None
        if j is not None and j > hi:
            why.append("shake")
        g = framed(c, role) if portrait else plays(c, role, False) * (0.85 + 0.3 * c.get("face", 0.0))
        if j is not None and j > lo:
            g *= 1 - 0.35 * min((j - lo) / (hi - lo), 1)
        g *= 0.4 + 0.6 * min(c.get("fill", FILL_GOOD) / FILL_GOOD, 1.0)   # the subject large in frame
        return dict(ok=not why, score=round(float(g), 3), crop=c or None, reasons=why,
                    jitter=round(j, 5) if j is not None else None)
    finally:
        R.clear(); R.update(saved)


def recast(project, fmt, seq, moments, beat):
    """assess judges a moment over a fixed stretch around it; the shot plays its own length from its own in-point.
    A planner's portrait shot that frames badly over what it actually shows (the subject lost from the strip,
    a back, someone cut in half -- framed() under RECAST) takes the best-framing unused moment of the same kind of
    footage (its category, else its role) that isn't beside its own clip, keeping any effect. The editor's picks,
    featured people, quick cuts, band and ending shots and B-roll stay."""
    n = 0
    for i, s in enumerate(seq):
        u = s["unit"]
        if not _recastable(s):
            continue
        L = s["beats"] * beat
        g = framed(reframe(project, u, in_point(s, beat), L, fmt), u["role"])
        if g >= RECAST:
            continue
        best = None
        for r, m in _alternatives(seq, i, moments, fmt):
            x = shot(r, m, fmt, s["beats"], act=s["act"], fx=s["fx"])      # (an effect shot keeps its effect)
            gx = framed(reframe(project, r["unit"], in_point(x, beat), L, fmt), r["role"])
            if gx > max(2 * g, RECAST) and (best is None or gx > best[0]):
                best = (gx, x)
        if best:
            seq[i] = best[1]
            n += 1
    if n:
        print(f"[reel] {fmt}: {n} shots recast that framed badly over what they show")


def _recastable(s):
    return (s["unit"]["role"] != "transition" or people_broll(s["unit"])) and not s.get("span") and not s.get("person") and not s.get("reuse") \
        and not any(s.get(k) for k in ("reaction", "ending", "band", "logo", "add", "featured"))


def _alternatives(seq, i, moments, fmt):
    """Unused moments of the same kind of footage as seq[i] (its category first, then by rank) not beside its clip;
    day footage (unit "day") only in the day, and the day's arrival only day footage."""
    u, act = seq[i]["unit"], seq[i]["act"]
    used = [(x["unit"]["id"], x["t"]) for x in seq]
    near = {seq[k]["unit"]["clip"] for k in (i - 1, i + 1) if 0 <= k < len(seq)}
    fits_act = lambda ru: bool(ru.get("day")) == (act in DAY_ACTS) or not (ru.get("day") or act == "arrival")
    cands = [(r, m) for r in moments if r["role"] == u["role"] and r["unit"]["clip"] not in near and fits_act(r["unit"])
             for m in r["moments"] if not m.get("pick") and m.get("plays_p", 1.0) >= RECAST
             and all(not (uid == r["id"] and abs(t - m["t"]) < 4.0) for uid, t in used)]
    cands.sort(key=lambda rm: (rm[0]["unit"].get("category") != u.get("category"), -rk(rm[1], fmt)))
    return cands[:RECAST_TRY]


SLIDE = (-0.5, 0.5, -1.0, 1.0, -0.25, 0.25)   # recast_placed: in-point moves tried first (of the shot's length)


def recast_placed(project, fmt, shots, moments, cfg):
    """recast once more on the placed shots, over exactly what each plays (place() moves in-points and lengthens
    shots for their landings): a shot still framing badly takes the best-framing alternative at the same length,
    reframed and re-measured for shake. Speed-effect shots keep theirs (their landing is timed to the moment)."""
    from . import shake
    lo, n, n_slid = cfg["shake"][0], 0, 0
    for i, s in enumerate(shots):
        if not _recastable(s) or not s.get("crop") or any(f in SPEED for f in s["fx"]):
            continue
        g = framed(s["crop"], s["unit"]["role"])
        if g >= RECAST:
            continue
        u, L = s["unit"], s["length"]                        # a good shot gone wrong (she walks out halfway, the
        slid = None                                          # partner turns away): first the same scene, a little
        for dt in SLIDE:                                     # earlier or later, before another clip
            src = clear_of_bans(min(max(s["src_in"] + dt * L, u["t0"]), max(u["t1"] - L, u["t0"])), L, u)
            if abs(src - s["src_in"]) < 0.1:
                continue
            c = reframe(project, u, src, L, fmt)
            gx = framed(c, u["role"])
            if gx >= RECAST and (slid is None or gx > slid[0]):
                slid = (gx, src, c)
        if slid:
            _, src, c = slid
            s.update(src_in=round(src, 3), peak=round(s["t"] - src, 3), crop=c, slid=True)
            s.pop("land", None); s.pop("land_fits", None)
            j = shake.jitter(project, u, src, src + L)
            j = j / max(c.get("w", 1.0) if c else 1.0, 0.2) if j is not None else None
            s["jitter"] = round(j, 5) if j is not None else None
            s["stab"] = bool(j is not None and j > lo and "_stabilized" not in u["source"])
            n_slid += 1
            continue
        best = None
        for r, m in _alternatives(shots, i, moments, fmt):
            x = shot(r, m, fmt, s["beats"], act=s["act"])
            u, L = r["unit"], s["length"]
            src = clear_of_bans(min(max(in_point(x, s["beat"]), u["t0"]), max(u["t1"] - L, u["t0"])), L, u)
            c = reframe(project, u, src, L, fmt)
            gx = framed(c, u["role"])
            if gx > max(2 * g, RECAST) and (best is None or gx > best[0]):
                best = (gx, x, src, c)
        if best:
            _, x, src, c = best
            s.update({k: x[k] for k in ("unit", "t", "kind", "rank", "face", "framing", "thumb") if k in x}, src_in=src, crop=c)
            s.pop("land", None); s.pop("land_fits", None)
            j = shake.jitter(project, s["unit"], src, src + s["length"])
            j = j / max(c.get("w", 1.0) if c else 1.0, 0.2) if j is not None else None
            s["jitter"] = round(j, 5) if j is not None else None
            s["stab"] = bool(j is not None and j > lo and "_stabilized" not in s["unit"]["source"])
            n += 1
    if n or n_slid:
        print(f"[reel] {fmt}: {n} placed shots recast over what they play, {n_slid} slid along their own clip")


def _swappable(s):
    """A placed shot that can give its slot to a unit with none: plain (no effect timed to it), not an editor's pick,
    an ending, a logo, a band insert, a title, a featured person's or a reaction shot."""
    return not s["fx"] and not s.get("span") and not s.get("reuse") and not s.get("velocity") and \
        not any(s.get(k) for k in ("ending", "logo", "logo_image", "band", "add", "featured", "person", "reaction",
                                   "title", "open"))


COVER_FLOOR = 0.15       # a unit coverage would force in needs a passing window framing at least this well


def ensure_units(project, fmt, shots, moments, cfg, W=None):
    """Every unit with a moment that passes in this format (the scrub's fmt_ok, nothing marked unusable) is in the
    reel at least once: after everything else has placed, recast and covered people, a unit still missing takes the
    slot of a plain repeat -- a shot of a unit with two or more, never beside its own clip, B-roll never beside
    B-roll; with song sets, only a slot of its own category -- reframed and measured for shake over what it plays.
    More distinct dancers beat a second shot of the same couple. With the scrub (W: windows_<fmt>.json) it starts on
    the unit's passing windows, so only a slot short enough for one of them will do. Units that can't be placed are
    logged."""
    from . import shake
    lo = cfg["shake"][0]
    W = load_windows(project, fmt) if W is None and project is not None else W
    have = Counter(s["unit"]["id"] for s in shots)
    day_cats = [*(cfg.get("day") or ()), *(cfg.get("day_shows") or ())]

    def day(r):
        return bool(r["unit"].get("day")) or (bool(day_cats) and fits(r["unit"], day_cats))

    def passing(r):                                          # the scrub's verdict where there is one (it judged
        return [m for m in r["moments"] if (m.get("fmt_ok") or {}).get(fmt) is True or (   # the windows themselves)
                "fmt_ok" not in m and not m.get("unusable") and not (fmt == "portrait" and m.get("plays_p", 1.0) < UNFRAMED))]
    floor = cfg.get("cover_floor", COVER_FLOOR)

    def best(r):                                             # how well its best passing window frames
        ws = [w["score"] for w in (W or {}).get(r["id"]) or () if w["ok"]]
        return max(ws) if ws else None
    miss = [r for r in moments if r["id"] not in have and passing(r)
            and (r["role"] != "transition" or people_broll(r["unit"]))]
    weak = [r["id"] for r in miss if (b := best(r)) is not None and b < floor]
    if weak:                                                 # coverage never forces in a shot not worth seeing
        print(f"[reel] {fmt}: not forced in, their best window too weak (< {floor}): {weak}")
        miss = [r for r in miss if r["id"] not in weak]
    miss.sort(key=lambda r: -max(rk(m, fmt) for m in passing(r)))
    placed, stuck = [], []

    def options(r):                                          # its best moment, and the slots it fits: {slot: in}
        m = max(passing(r), key=lambda m: rk(m, fmt))
        clip = r["unit"]["clip"]

        def ok(i, r=r, clip=clip):
            x = shots[i]
            if have[x["unit"]["id"]] < 2 or not _swappable(x):
                return False
            if cfg.get("sets") and x["unit"].get("category") != r["unit"].get("category"):
                return False
            if day(r) != (x["act"] in DAY_ACTS) and (day(r) or x["act"] == "arrival"):
                return False                                 # day footage stays in the day, and the day is day
            nb = [shots[k] for k in (i - 1, i + 1) if 0 <= k < len(shots)]
            return all(n["unit"]["clip"] != clip for n in nb) and \
                not (r["role"] == "transition" and any(n["unit"]["role"] == "transition" for n in nb))
        idx = [i for i in range(1, len(shots) - 1) if ok(i)]
        u = r["unit"]

        def start(i, r=r, m=m, u=u):                         # its in-point for slot i: on passing windows when
            s = shots[i]                                     # the scrub has them (a slot too long for every
            L = s["length"]                                  # passing stretch of the unit doesn't take it)
            src = clear_of_bans(min(max(in_point(shot(r, m, fmt, s["beats"], act=s["act"]), s["beat"]), u["t0"]),
                                    max(u["t1"] - L, u["t0"])), L, u)
            wins = (W or {}).get(u["id"])
            if not wins or any(s["fx"]) or span_ok(wins, src, src + L) is not False:
                return src
            cands = [x for w in wins if w["ok"] for x in (w["t0"], w["t1"] - L, w["t0"] - SNAP_TOL,
                                                           w["t1"] - L + SNAP_TOL)]
            cands = [round(x, 3) for x in cands if u["t0"] <= x <= u["t1"] - L]
            cands = [x for x in cands if not in_bans(x, L, u) and span_ok(wins, x, x + L) is True]
            return min(cands, key=lambda x: abs(x - src)) if cands else None
        at = {i: start(i) for i in idx}
        return m, {i: x for i, x in at.items() if x is not None}
    while miss:                                              # the unit with the fewest slots it fits goes first
        opts = {r["id"]: options(r) for r in miss}           # (one that passes only over a short stretch isn't
        for r in [r for r in miss if not opts[r["id"]][1]]:  # left without one by a unit any slot would take)
            stuck.append(r["id"])
        miss = [r for r in miss if opts[r["id"]][1]]
        if not miss:
            break
        r = min(miss, key=lambda r: len(opts[r["id"]][1]))   # (ties: the best first, as miss is sorted)
        miss.remove(r)
        (m, at), u = opts[r["id"]], r["unit"]
        idx = list(at)
        wanted = Counter(i for q in miss for i in opts[q["id"]][1])   # a slot fewer of the others fit, first
        cat = u.get("category")
        i = max(idx, key=lambda i: (shots[i]["unit"].get("category") == cat, -wanted[i],
                                    have[shots[i]["unit"]["id"]], -shots[i]["rank"]))
        s = shots[i]
        x = shot(r, m, fmt, s["beats"], act=s["act"])
        L, src = s["length"], at[i]
        c = reframe(project, u, src, L, fmt)
        have[s["unit"]["id"]] -= 1
        have[u["id"]] += 1
        s.update({k: x[k] for k in ("unit", "t", "kind", "rank", "face", "framing", "thumb") if k in x},
                 src_in=round(src, 3), peak=round(x["t"] - src, 3), crop=c, covered=True)
        s.pop("land", None); s.pop("land_fits", None)
        j = shake.jitter(project, u, src, src + L)
        j = j / max(c.get("w", 1.0) if c else 1.0, 0.2) if j is not None else None
        s["jitter"] = round(j, 5) if j is not None else None
        s["stab"] = bool(j is not None and j > lo and "_stabilized" not in u["source"])
        placed.append(r["id"])
    if placed or stuck:
        print(f"[reel] {fmt}: {len(placed)} units with passing windows given a shot (a repeat's slot)"
              + (f"; no slot for {stuck}" if stuck else ""))
    return placed


SNAP_TOL = 0.3           # a window "covers" a shot that starts or ends this close inside it (windows.grid's edge)
SNAP_GAP = 0.5           # a shot moved onto passing windows keeps this far (s) from the unit's other shots


def load_windows(project, fmt):
    """The scrub's windows for this format ({unit: [{t0, t1, ok, ...}]}), None without a scrub."""
    p = project.path(f"windows_{fmt}.json")
    return json.load(open(p)) if os.path.exists(p) else None


INSIDE = 0.8             # span_ok: a shortest window this much inside a shot is a second the shot plays


def span_ok(wins, a, b, tol=SNAP_TOL):
    """Does the source stretch a..b sit on passing windows? True / False, None when the scrub didn't judge it (no
    windows there: scenery, an unscrubbed unit). The shortest windows it plays decide first: any of them lying
    mostly inside it (INSIDE of the window) that fails fails it -- a back filling the frame for a second, the camera
    panning off the subject, the subject leaving, however well a longer window over it averages them out. Then the
    windows that cover it (start within tol after a, end within tol before b) decide: any passing one passes it. A
    shot longer than every window passes when passing windows inside it tile it end to end."""
    if not wins:
        return None
    tol += 1e-3                                              # (in-points are kept to the millisecond)
    short = min(w["t1"] - w["t0"] for w in wins)
    inner = [w for w in wins if w["t1"] - w["t0"] <= short + 1e-3
             and min(b, w["t1"]) - max(a, w["t0"]) >= INSIDE * (w["t1"] - w["t0"])]
    if any(not w["ok"] for w in inner):
        return False                                         # a second of it fails: a long window passing over
    cover = [w for w in wins if w["t0"] <= a + tol and w["t1"] >= b - tol]   # it averages that away
    if cover:
        return any(w["ok"] for w in cover)
    if not any(w["t1"] > a and w["t0"] < b for w in wins):
        return None
    reach = a + tol
    for w in sorted((w for w in wins if w["ok"] and w["t0"] >= a - tol and w["t1"] <= b + tol), key=lambda w: w["t0"]):
        if w["t0"] > reach:
            break
        reach = max(reach, w["t1"])
    return reach >= b - tol


def window_exempt(s):
    """Shots chosen by hand or by config, which play where they were put: the editor's picks (and the close among
    them), the ending, the opening, logos and titles. (cover_people's shots are marked add too, with person: those
    the planner chose; so are quick cuts replaying a pick, reuse: the planner put them there, framed its own way.)"""
    return bool((s.get("span") and not s.get("reuse")) or (s.get("add") and not s.get("person")) or s.get("ending")
                or s.get("open") or s.get("logo")
                or s.get("logo_image") or s.get("title"))


END_FILL = ("band", "singer", "stage", "dj", "crowd")   # what follows a last shot cut short, in that order


def end_clean(project, fmt, shots, moments, cfg, W=None):
    """The reel's last shot ends before its subject leaves (the camera panning off onto nothing ends nothing): when
    its end fails the scrub, it is cut back to the longest stretch that ends well (two beats or more; a configured
    ending too, its moment kept -- an editor's pick, the close among them, always plays as chosen), and the beats it
    gives up go to a shot after it -- the band, the stage, the
    DJ or the crowd (END_FILL), else any footage passing there. Changes shots in place; returns the filler or None."""
    W = load_windows(project, fmt) if W is None and project is not None else W
    if not shots or not W:
        return None
    s = shots[-1]
    u, p = s["unit"], s.get("speed") or 1.0
    wins = W.get(u["id"])
    if s.get("span") and not s.get("reuse"):                 # an editor's pick (the close) plays as chosen
        return None
    if any(f in SPEED and f != "fit" for f in s["fx"]) or span_ok(wins, s["src_in"], s["src_in"] + s["length"] * p) is not False:
        return None
    beat = s["beat"]
    n = next((n for n in range(s["beats"] - 1, 1, -1)
              if span_ok(wins, s["src_in"], s["src_in"] + n * beat * p) is not False), None)
    if n is None:
        print(f"[reel] {fmt}: WARNING the last shot {u['id']} leaves its subject and no 2 beats of it end well")
        return None
    k, rec0 = s["beats"] - n, s["rec_in"]
    edge = lambda t: round(t * FPS)
    L2, at = k * beat, rec0 + n * beat
    rank_tag = lambda r: next((i for i, t in enumerate(END_FILL) if t in (r["unit"].get("tag") or "")), len(END_FILL))
    cands = sorted((r for r in moments if r["moments"] and r["unit"]["clip"] != u["clip"]
                    and (fmt == "portrait" or r["orient"] == "landscape")),
                   key=lambda r: (r["role"] != "transition", rank_tag(r), -max(rk(m, fmt) for m in r["moments"])))
    fill = None
    for r in cands:
        ru = r["unit"]
        for m in sorted(r["moments"], key=lambda m: -rk(m, fmt)):
            a = round(min(max(m["t"] - L2 / 2, ru["t0"]), max(ru["t1"] - L2, ru["t0"])), 3)
            if span_ok(W.get(r["id"]), a, a + L2) is False:
                continue
            fill = {key: v for key, v in s.items() if key not in (
                "span", "ending", "close", "open", "logo", "logo_image", "title", "speed", "pull", "lift", "snapped",
                "recast_window", "transition", "land", "land_fits", "reaction", "add", "person", "reuse")}
            fill.update(unit=ru, t=m["t"], kind=m.get("kind", "crowd"), rank=rk(m, fmt), src_in=a, peak=round(m["t"] - a, 3),
                        beats=k, length=round(L2, 4), rec_in=round(at, 3), frames=edge(rec0 + s["length"]) - edge(at),
                        fx=[], effect=None, face=m.get("face", 0.0), end_fill=True)
            fill["crop"] = reframe(project, ru, a, L2, fmt) if project is not None and fmt == "portrait" else None
            break
        if fill:
            break
    if fill is None:
        print(f"[reel] {fmt}: WARNING the last shot {u['id']} leaves its subject and nothing passes to follow it")
        return None
    s["beats"], s["length"], s["frames"] = n, round(n * beat, 4), edge(at) - edge(rec0)
    if s.get("crop") and project is not None:
        s["crop"] = reframe(project, u, s["src_in"], s["length"] * p, fmt) or s["crop"]
    shots.append(fill)
    print(f"[reel] {fmt}: the last shot {u['id']} cut to {n} beats before its subject leaves; "
          f"{fill['unit']['id']} ends the reel ({k} beats)")
    return fill


def named(unit, k):
    """Is k this unit's id or its clip's (an export cut from clip C1991 is "C1991-02_couple", "C1991_matinee")?"""
    i = unit.get("id") or ""
    return k in (i, unit.get("clip")) or (i.startswith(k) and not i[len(k):len(k) + 1].isalnum())


def pace_shots(shots, cfg):
    """A base speed under each shot (reel.json plan.<fmt> "pace": {"<footage match>": 1.15, ...}: a unit or clip
    id ("C1991", every unit cut from it) or a fits() match, the first that matches): footage danced slower than the track plays that much quicker throughout, its speed ramps
    on top of it, still never past the 1.5x cap. A plain shot reads its peak's moment as before (the source shifts
    so the move lands where it did); an editor's whole selection ("fit") plays as chosen."""
    rules = cfg.get("pace") or {}
    changed = 0
    for s in shots:
        old = s.pop("speed", None) or 1.0
        p = next((float(v) for k, v in rules.items() if named(s["unit"], k) or fits(s["unit"], [k])), 1.0)
        p = 1.0 if "fit" in s["fx"] else round(min(p, 1.5), 3)
        if p != 1.0:
            s["speed"] = p
        if p == old:
            continue
        changed += 1
        if not s.get("span") and not any(f in SPEED for f in s["fx"]):
            u = s["unit"]                                           # the peak's moment stays on the beat
            src = s["t"] - s["peak"] * p
            s["src_in"] = round(min(max(src, u["t0"]), max(u["t1"] - s["length"] * p, u["t0"])), 3)
    return changed


def played(s):
    """(source in, source length) render reads for a shot: a speed effect reads more or less source than the shot's
    length, around its peak (render.fx_shot)."""
    sp = [f for f in s["fx"] if f in SPEED and f != "fit"]
    if not sp or s.get("span"):
        return s["src_in"], s["length"] * (1.0 if "fit" in s["fx"] else s.get("speed") or 1.0)
    from .. import fx
    from ..fx.speed import Speed
    u, L = s["unit"], s["length"]
    ctx = fx.Context(fps=FPS, beat=s["beat"], peak=s["peak"], seed=0,
                     params={"group": s.get("group"), "closeup": s.get("kind") == "closeup",
                             "cap": 3.5 if u["role"] == "transition" else 1.5, "pace": s.get("speed") or 1.0})
    src_len, src_peak = L, s["peak"]
    for n in s["fx"]:
        e = fx.EFFECTS[n]()
        if isinstance(e, Speed):
            src_len, src_peak = e.source_length(L, ctx), e.source_peak(s["peak"], ctx, L)
    return min(max(s["t"] - src_peak, u["t0"]), max(u["t1"] - src_len, u["t0"])), src_len


def window_fails(shots, W):
    """The shots that play a stretch every covering window fails: [(index, shot, why)], why the exempt kind
    (window_exempt) or None for one the planner chose."""
    out = []
    for i, s in enumerate(shots):
        a, L = played(s)
        if span_ok(W.get(s["unit"]["id"]), a, a + L) is False:
            out.append((i, s, "pick" if (s.get("span") and not s.get("reuse")) or (s.get("add") and not s.get("person"))
                        else "ending" if s.get("ending")
                        else "opening" if s.get("open") or s.get("logo") else "title" if window_exempt(s) else None))
    return out


BAN_CLEAR = 2.0          # snap_windows: a shot moved off banned footage along its own clip lands this far clear of
                         # the ban (just past its edge is the same rejected scene)


def snap_windows(project, fmt, shots, moments, cfg, W=None, split=True):
    """Every shot the planner chose (not the editor's picks, the ending, the opening) plays a stretch that passes
    the scrub in this format, whichever pass put it there (band inserts, people, quick cuts, landings, slides): one
    on failing windows moves along its own clip, as little as it can, onto passing ones (the move and its beat move
    with it); else it takes a passing moment of another unit of the same kind (a unit not in the reel yet first,
    never beside its own clip, day footage in the day; with song sets its own set's first), keeping its effect;
    else a plain shot of SPLIT_BEATS or more is cut in two (the first half on its own clip, the second recast onto
    another's), so a long slot no stretch passes over whole still plays footage that passes; else it is logged.
    Returns (slid, recast, stuck)."""
    from . import shake
    W = load_windows(project, fmt) if W is None else W
    if not W:
        return 0, 0, []
    lo = cfg["shake"][0]
    have = Counter(s["unit"]["id"] for s in shots)
    day_cats = [*(cfg.get("day") or ()), *(cfg.get("day_shows") or ())]

    def free(uid, a, b, me):                                 # no other shot of the unit plays this stretch, or
        for x in shots:                                      # one just beside it (the same moment twice)
            if x is not me and x["unit"]["id"] == uid:
                xa, xl = played(x)
                if min(b, xa + xl) - max(a, xa) > -SNAP_GAP:
                    return False
        return True

    def fits_at(s, u, dt, a, L):                             # the shot moved by dt: inside the unit, clear, passing
        na = round(a + dt, 3)                                # as it will be stored
        return (u["t0"] - 1e-6 <= na and na + L <= u["t1"] + 1e-6 and not in_bans(na, L, u)
                and span_ok(W.get(u["id"]), na, na + L) is True and free(u["id"], na, na + L, s))

    def passing(r):                                          # (the windows judge the exact span: assess's
        return [m for m in r["moments"] if (m.get("fmt_ok") or {}).get(fmt) is True]   # "unusable" is a guess at it)

    def measure(s):
        u = s["unit"]
        a, L = played(s)
        c = reframe(project, u, s["src_in"], s["length"], fmt) if s.get("crop") or fmt == "portrait" else None
        s["crop"] = c or s.get("crop")
        j = shake.jitter(project, u, a, a + L)
        j = j / max((c or {}).get("w", 1.0), 0.2) if j is not None else None
        s["jitter"] = round(j, 5) if j is not None else None
        s["stab"] = bool(j is not None and j > lo and "_stabilized" not in u["source"])
        s.pop("land", None); s.pop("land_fits", None)

    def offsets(s, r, uid, a, L, speedy):                    # moves to try, smallest first: onto a passing
        dts = [m["t"] - s["t"] for m in ((passing(r) or r["moments"]) if r else [])]   # moment (its move on the
        # shot's beat; unassessed moments -- a refill -- are judged by fits_at on the windows instead)
        if not speedy:                                       # a plain shot can start anywhere: on a window's edges
            for w in W.get(uid) or ():
                if w["ok"]:
                    dts += [w["t0"] - a, w["t1"] - L - a, w["t0"] - SNAP_TOL - a, w["t1"] - L + SNAP_TOL - a]
        return sorted({round(d, 3) for d in dts}, key=abs)

    slid = recast_n = 0
    ob = cfg.get("open_broll")
    opening = (ob if isinstance(ob, list) else ("teaser", "arrival")) if ob else ()
    stuck, stuck_at = [], []
    todo = [(i, s) for i, s, why in window_fails(shots, W) if not why]
    late = [(i, s) for i, s in enumerate(shots) if s["unit"].get("day") and s["act"] not in DAY_ACTS
            and not window_exempt(s) and all(i != j for j, _ in todo)]     # daytime footage in a night act,
    late += [(i, s) for i, s in enumerate(shots) if not window_exempt(s) and in_bans(*played(s), s["unit"])
             and all(i != j for j, _ in todo + late)]                      # or footage banned since it was placed
    for i, s in sorted(todo + late, key=lambda x: x[0]):                    # whichever pass put it there
        u = s["unit"]
        a, L = played(s)
        speedy = any(f in SPEED for f in s["fx"])
        rec = next((r for r in moments if r["id"] == u["id"]), None)
        uc = dict(u, bans=[[x - BAN_CLEAR, y + BAN_CLEAR] for x, y in u["bans"]]) if in_bans(a, L, u) else u
        dt = None if (u.get("day") and s["act"] not in DAY_ACTS) or s.get("half") == 2 else \
            next((d for d in offsets(s, rec, u["id"], a, L, speedy) if fits_at(s, uc, d, a, L)), None)
        if dt is not None:
            s.update(t=round(s["t"] + dt, 3), src_in=round(s["src_in"] + dt, 3), snapped=round(dt, 2))
            measure(s)
            slid += 1
            continue
        nb = [shots[k] for k in (i - 1, i + 1) if 0 <= k < len(shots)]
        perf = people_broll(u)
        best = None
        for loose in ((False, True) if in_bans(a, L, u) else (False,)):   # banned footage must go: footage of
            if best:                                                       # another kind before none
                break
            for r in moments:
                ru = r["unit"]
                if r["id"] == u["id"] or (not loose and (r["role"] != u["role"] if not perf else not people_broll(ru))):
                    continue
                if any(n["unit"]["clip"] == ru["clip"] for n in nb):
                    continue
                if s["act"] in opening and r["role"] != "transition":   # (cfg "open_broll": B-roll there alone)
                    continue
                own_set = not cfg.get("sets") or ru.get("category") == u.get("category")   # (another set's only
                d = bool(ru.get("day")) or (bool(day_cats) and fits(ru, day_cats))         # when its own has none)
                if d != (s["act"] in DAY_ACTS) and (d or s["act"] == "arrival"):
                    continue
                ms = passing(r) or r["moments"]                  # none left: footage the config keeps out
                for dt in offsets(s, r, r["id"], a, L, speedy):
                    if not ms or not fits_at(s, ru, dt, a, L):
                        continue
                    mid = a + dt + L / 2                         # the shot's moment: the passing one nearest it
                    m = min(ms, key=lambda m, mid=mid: abs(m["t"] - mid))
                    key = (own_set, have[r["id"]] == 0, ru.get("category") == u.get("category"), rk(m, fmt))
                    if best is None or key > best[0]:
                        best = (key, r, m, dt)
        if best:
            _, r, m, dt = best
            x = shot(r, m, fmt, s["beats"], act=s["act"])
            have[u["id"]] -= 1
            have[r["id"]] += 1
            s.update({k: x[k] for k in ("unit", "t", "kind", "rank", "face", "framing", "thumb") if k in x},
                     src_in=round(s["src_in"] + dt, 3), recast_window=True)
            s["peak"] = round(s["t"] - s["src_in"], 3)
            measure(s)
            recast_n += 1
            continue
        stuck.append(f"#{i} {u['id']}@{a:.1f}")
        stuck_at.append(s)
    halves = [s for s in stuck_at if split and s["beats"] >= SPLIT_BEATS and not s["fx"] and not s.get("velocity")
              and "rec_in" in s and "frames" in s]                 # (placed shots only)
    if halves:
        for s in halves:
            k = next(k for k, x in enumerate(shots) if x is s)
            shots[k + 1:k + 1] = [split_shot(s)]
        print(f"[reel] {fmt}: {len(halves)} long shots no stretch passes over whole cut in two")
        d_slid, d_recast, stuck = snap_windows(project, fmt, shots, moments, cfg, W, split=False)
        for k in range(len(shots) - 1, 0, -1):               # a second half nothing took: back into its first
            x, prev = shots[k], shots[k - 1]
            if x.get("half") == 2 and prev.get("half") == 1 and prev["unit"]["clip"] == x["unit"]["clip"]:
                prev.update(beats=prev["beats"] + x["beats"], length=round(prev["length"] + x["length"], 4),
                            frames=prev["frames"] + x["frames"])
                del shots[k]
        return slid + d_slid, recast_n + d_recast, stuck
    if slid or recast_n or stuck:
        print(f"[reel] {fmt}: {slid} shots slid onto passing windows, {recast_n} recast to another unit's"
              + (f"; none passes for {stuck}" if stuck else ""))
    return slid, recast_n, stuck


SPLIT_BEATS = 4          # snap_windows: a stuck plain shot this long is cut in two


def split_shot(s):
    """Cut shot s in two on a beat, in place: s keeps the first half, the second (returned, half 2) plays on from
    where it ends -- to be recast onto another clip (two halves of one clip side by side would cut to itself)."""
    b1 = s["beats"] // 2
    beat = s["beat"]
    f1 = round((s["rec_in"] + b1 * beat) * FPS) - round(s["rec_in"] * FPS)
    L1 = round(b1 * beat, 4)
    t = dict(s, beats=s["beats"] - b1, length=round(s["length"] - L1, 4), frames=s["frames"] - f1,
             rec_in=round(s["rec_in"] + L1, 3), src_in=round(s["src_in"] + L1, 3), half=2, fx=list(s["fx"]))
    t["t"] = s["t"] if t["src_in"] <= s["t"] <= t["src_in"] + t["length"] else round(t["src_in"] + t["length"] / 2, 3)
    t["peak"] = round(t["t"] - t["src_in"], 3)
    s.update(beats=b1, length=L1, frames=f1, half=1)
    if not s["src_in"] <= s["t"] <= s["src_in"] + L1:
        s["t"] = round(s["src_in"] + L1 / 2, 3)
    s["peak"] = round(s["t"] - s["src_in"], 3)
    for x in (s, t):
        x.pop("land", None); x.pop("land_fits", None)
    return t


def window_report(project, fmt, plan):
    """The plan log's check: every shot the planner chose plays on passing windows (WARNING lines for any that
    doesn't); the editor's picks and the ending on failing ones are counted, as their choice."""
    W = load_windows(project, fmt)
    if not W:
        return []
    bad = window_fails(plan["shots"], W)
    mine = [(i, s) for i, s, why in bad if not why]
    chosen = Counter(why for _, _, why in bad if why)
    print(f"[reel]   windows: {len(plan['shots']) - len(bad)}/{len(plan['shots'])} shots on passing windows"
          + (f" ({', '.join(f'{n} {k}' for k, n in sorted(chosen.items()))} on failing ones, as chosen)" if chosen else ""))
    lines = []
    for i, s in mine:
        a, L = played(s)
        lines.append(f"shot {i} {s['unit']['id']} {a:.2f}-{a + L:.2f}: on failing windows (its end, or all that cover it)")
        print(f"[reel]   WARNING windows: {lines[-1]}")
    return lines


def shot(r, m, fmt, beats, act="shows", fx=(), **kw):
    """One shot of a moment (a moments.json record r and one of its moments m), for a hand-built sequence: place()
    times it, frames it and gives it its transition."""
    s = Picker(fmt, [r]).take(r, m, beats=beats, act=act, **kw)
    s["fx"], s["effect"] = list(fx), (fx[-1] if fx else None)
    return s


def lands(project, seq, beat, hold):
    """Where each plain dance shot's move lands (velocity.landed), and the beats it needs to show it: the end of the
    move and hold seconds of rest. A shot too short for that takes a beat from a plain neighbour in its act that can
    spare one (an act's edge shots keep 2 for their transition; a neighbour with its own landing keeps what it needs)."""
    need = math.ceil((V.PRE + hold) / beat - 1e-6)
    for i, s in enumerate(seq):
        if (s["unit"]["role"] == "transition" or any(s.get(f) for f in ("ending", "logo", "reaction", "span"))
                or any(f in SPEED for f in s["fx"])):
            continue
        s["land"] = V.landed(project, s["unit"], s["t"])
        if s["land"] is None:
            continue

        def spare(j, s=s):
            d = seq[j]
            edge = (j > 0 and seq[j - 1]["act"] != d["act"]) or (j + 1 < len(seq) and seq[j + 1]["act"] != d["act"])
            keep = max(need if d.get("land") is not None else 1, 2 if edge else 1)
            return (d["act"] == s["act"] and not d["fx"] and not any(d.get(f) for f in ("ending", "logo", "span"))
                    and d["beats"] - 1 >= keep)
        while s["beats"] < need:
            donors = [j for j in (i + 1, i - 1) if 0 <= j < len(seq) and spare(j)]
            if not donors:
                break
            j = max(donors, key=lambda j: seq[j]["beats"])
            seq[j]["beats"] -= 1; s["beats"] += 1


def real_speed_close(seq):
    """The reel's closing shot (the last, the configured ending, finale_reuse's close) plays at 1x: no speed effect."""
    for k, s in enumerate(seq):
        if (k == len(seq) - 1 or s.get("ending")) and any(f in SPEED for f in s["fx"]):
            s["fx"] = [f for f in s["fx"] if f not in SPEED]
            s["effect"] = s["fx"][-1] if s["fx"] else None


def place(project, fmt, seq, music, cfg=None, direction=None, stinger_len=0.0, acts=None, song=None):
    """Turn a sequence of shots (beats, act, fx; from story() or built by hand with shot()) into a plan that render,
    xml_export and review can use: record positions locked to the beat grid, velocity edits slowed on the dancer's
    landing on a strong beat, the crop that follows the subject, shake (stabilised where needed) and the
    transitions the cuts have a reason for. music: music.json or music.analyse() of any song."""
    cfg = cfg or config(project, fmt)
    music = dict(music, beat=music.get("beat") or float(np.median(np.diff(music["beats"]))))
    beat = music["beat"]
    T = clock(music)
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    acc = list(np.array(music.get("beat_accent") or np.ones(len(bt)))[i0:])
    from . import shake
    lo = cfg["shake"][0]

    def frame(s):
        """The shot's crop (following its subject over the stretch it plays) and shake, from its in-point."""
        u, L = s["unit"], s["length"]
        s["crop"] = reframe(project, u, s["src_in"], L, fmt)
        if s.get("ending") and s["crop"]:                    # the whole scene, centred: no following one person
            s["crop"] = dict(w=s["crop"]["w"] * s["crop"]["z"], h=s["crop"]["h"] * s["crop"]["z"], z=1.0,
                             path=[[0, (list(cfg["ending"]) + [0.5])[3], 0.5]])
        j = shake.jitter(project, u, s["src_in"], s["src_in"] + L)
        j = j / max((s["crop"] or {}).get("w", 1.0), 0.2) if j is not None else None
        s["jitter"] = round(j, 5) if j is not None else None
        s["stab"] = bool(j is not None and j > lo and "_stabilized" not in u["source"])   # (not twice)
    hold = max(cfg.get("land_hold", 1.0) * beat, 0.45)
    real_speed_close(seq)
    lands(project, seq, beat, hold)
    rec = 0
    for s in seq:
        L = T(rec + s["beats"]) - T(rec)
        s["beat"] = round(L / s["beats"], 5)                  # this shot's own beat (the tempo may change)
        u = s["unit"]
        ramp = next((f for f in s["fx"] if f in V.RAMP), None)
        if ramp:                                             # velocity edit: slow on the landing, on a strong beat
            s["t"] = V.settle(project, u, s["t"])
            s["lead_beats"] = V.landing(s["beats"], ramp, s["kind"] == "closeup", acc[rec:], V.TAIL / s["beat"])
        src_in = in_point(s, s["beat"])
        if s.get("land") is not None:                        # the move's finish and the rest after it
            src_in, s["land_fits"] = V.hold_in(src_in, L, s["land"], hold)
            src_in = clear_of_bans(min(max(src_in, u["t0"]), max(u["t1"] - L, u["t0"])), L, u)
        if s.get("span"):                                    # the editor's selection starts at their in-point
            src_in = in_point(s, s["beat"])
        if s.get("ending"):                                  # the configured ending runs out at its "to" time --
            x0, (a, b) = u.get("export_t0", 0.0), cfg["ending"][1:3]   # or, longer than its stretch, plays all of
            src_in = max(x0 + (a if b - a <= L else b - L), u["t0"])    # it from its "from", fading past the end
        f0, f1 = round(T(rec) * FPS), round(T(rec + s["beats"]) * FPS)
        s.update(rec_in=round(T(rec), 3), length=round(L, 3), frames=f1 - f0, src_in=round(src_in, 3),
                 peak=round(s["t"] - src_in, 3), group=u["role"] == "show")
        frame(s)
        rec += s["beats"]
    transitions(seq, music, cfg, direction)
    if look_of(cfg):
        glitch_look(seq, music, look_of(cfg))
        real_speed_close(seq)                                # (a look's own effects for these: none either)
    for s in seq:                                            # velocity edits changed some shots' speed effects
        if any(f in SPEED for f in s["fx"]) and not s.get("ending"):
            src_in = round(in_point(s, s["beat"]), 3)
            s["land"] = s["land_fits"] = None                 # (a speed effect times itself on the peak)
            if src_in != s["src_in"]:
                s.update(src_in=src_in, peak=round(s["t"] - src_in, 3))
                frame(s)
    video = round(T(rec), 3)
    plan = dict(format=fmt, beat=beat, music_start=round(music["start"] - ANTICIPATE, 3),
                music_length=round(video + (STINGER_OVERLAP if stinger_len else 0), 3), video_length=video,
                stinger_at=video, stinger=stinger_len, shots=seq,
                acts={k: [T(a), T(min(b, rec))] for k, (a, b) in (acts or {}).items()})
    if song or music.get("song"):
        plan["song"] = song or music["song"]
    return plan


def coverage_report(project, fmt, plan):
    """The plan log's coverage: subject units (dance, show, performer B-roll; an editor's picks-only footage with no
    moments aside) and dance units in the cut / all of them, and
    each unit left out with why (the scrub's fail counts on its record, else 'never frames' / 'no moments')."""
    mo = [r for r in project.load("moments.json") if project.in_cut(r["unit"])]
    seq = plan["shots"]
    used = {s["unit"]["id"] for s in seq}
    subj = [r for r in mo if r["role"] != "transition" or people_broll(r["unit"])]
    only = set(project.config.get("picks_only") or ())
    subj = [r for r in subj if r["unit"].get("category") not in only or any(m.get("add") for m in r["moments"])]
    dance = [r for r in subj if r["role"] == "dance"]
    print(f"[reel]   coverage: subject units {sum(r['id'] in used for r in subj)}/{len(subj)}"
          + (f", dance units {sum(r['id'] in used for r in dance)}/{len(dance)}" if dance else ""))
    out = [r for r in subj if r["id"] not in used]
    for r in out:
        fails = r.get("fails") or {}
        why = ", ".join(f"{k} {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])[:4]) if fails else (
            "no moments" if not r["moments"] else "never frames" if not framable(r, fmt) else "no room")
        print(f"[reel]     left out {r['id']}: {why}")


def refill(project, fmt, base):
    """Re-cut by refilling an earlier plan (reel.json "refill"): every shot stays where it was, except those whose
    footage is banned now (the editor's bans and "reject" stretches added since), which take passing footage --
    their own clip a little earlier or later first, else another clip's -- the way snap_windows moves any shot. So
    a review round replaces only what was rejected; a full re-plan reshuffles half the cut and brings in shots
    nobody has looked at."""
    picks = project.picks()
    cfg = config(project, fmt)
    portrait_rules(cfg)
    moments = [r for r in copy.deepcopy(project.load("moments.json")) if project.in_cut(r["unit"])]
    if picks:
        apply_picks(moments, picks, only=cfg.get("only_picks", True))
    picks_only(moments, project.config.get("picks_only", ()))
    plan = copy.deepcopy(base)
    units = {r["unit"]["id"]: r["unit"] for r in moments}
    for s in plan["shots"]:
        u = units.get(s["unit"]["id"])
        if u is not None:
            s["unit"] = dict(s["unit"], bans=u.get("bans") or [])
    banned = sum(1 for s in plan["shots"] if not window_exempt(s) and in_bans(*played(s), s["unit"]))
    fm = [r for r in moments if fmt == "portrait" or r["orient"] == "landscape"]
    slid, recast_n, stuck = snap_windows(project, fmt, plan["shots"], fm, cfg)
    print(f"[reel] {project.cut_name(fmt)}: refilled from the earlier plan: {banned} banned shots"
          f" ({slid} slid, {recast_n} recast, {len(stuck)} stuck {stuck})")
    return plan


def run(project, fmt, stinger_len=0.0) -> dict:
    from .flow import Flow
    fl = Flow(project)
    rf = project.config.get("refill")
    rf = "plan_{fmt}_rendered.json" if rf is True else rf      # true: the last cut rendered
    rf = rf and rf.format(fmt=project.cut_name(fmt))
    if rf and os.path.exists(project.path(rf)):
        plan = refill(project, fmt, project.load(rf))
    else:
        plan = build(project, fmt, project.load(project.music_file), project.load("moments.json"), stinger_len,
                     fl.direction)
    fl.save()
    plan["format"] = fmt
    project.dump(f"plan_{project.cut_name(fmt)}.json", plan)
    seq = plan["shots"]
    used = len({s["unit"]["id"] for s in seq})
    trs = [s.get("transition") for s in seq[1:] if s.get("transition")]
    fxs = [f for s in seq for f in s["fx"]]
    print(f"[reel] {project.cut_name(fmt)}: {len(seq)} shots, {used}/{plan['units_total']} units, video {plan['video_length']:.1f}s"
          f" + stinger {stinger_len:.1f}s")
    for name in ACT_NAMES:
        ss = [s for s in seq if s["act"] == name]
        print(f"[reel]   {name:8s} {len(ss):3d} shots, {sum(bool(s.get('transition')) for s in ss)} transitions, "
              f"fx {[f for s in ss for f in s['fx']]}")
    print("[reel]   transitions:", {t: trs.count(t) for t in sorted(set(trs))})
    picks = {k: v for k, v in project.picks().items() if not k.startswith("_")}
    if picks:                                                # the editor's picks, checked against what was planned
        bad = pick_report(seq, picks, plan.get("picks_condensed", ()), not plan.get("pick_hold"))
        n = sum(1 for e in picks.values() for v in e.get("add", []))
        print(f"[reel]   picks: {n} added, {sum(len(e.get('ban', [])) for e in picks.values())} bans"
              + (f"; {len(bad)} to look at:" if bad else "; all in once, " + (
                  f"at their best part, {len(plan['picks_condensed'])} condensed out" if plan.get("picks_condensed")
                  else "at full length") + ", bans held"))
        for line in bad:
            print(f"[reel]     {line}")
    print("[reel]   effects:", {f: fxs.count(f) for f in sorted(set(fxs))})
    coverage_report(project, fmt, plan)
    window_report(project, fmt, plan)
    for line in V.speed_tails(plan):                         # every shot ends at real speed
        print(f"[reel]   WARNING speed: {line}")
    return plan
