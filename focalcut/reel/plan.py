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
    land_hold=1.0,                                # beats of the dancer at rest a shot keeps after a move lands
)
CFG = {
    "landscape": dict(_COMMON, heroes=4, night_pace="medium"),
    "portrait": dict(_COMMON, heroes=3, night_pace="fast"),
}


def config(project, fmt):
    cfg = copy.deepcopy(CFG[fmt])
    cfg.update((project.config.get("plan") or {}).get(fmt, {}))
    return cfg


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

    def free(self, r, m):
        return all(abs(m["t"] - t) >= APART for t in self.used.get(r["id"], []))

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
        return out

    def pool(self, recs, kinds=None, min_frame=0.0):
        out = [(r, m) for r in recs for m in r["moments"] if self.free(r, m) and (kinds is None or m["kind"] in kinds)
               and (r["role"] == "transition" or fr(m, self.fmt) >= min_frame)]
        return sorted(out, key=lambda x: -self.share(*x))

    def best(self, recs, kinds=None, min_frame=0.35, strict=False):
        """The top moment: well framed if possible, of these kinds if possible (strict: of these kinds or none)."""
        p = self.pool(recs, kinds, min_frame) or self.pool(recs, kinds) or (None if strict else self.pool(recs))
        return p[0] if p else None


def cover(P, recs, n, beats, act, cap=4, kinds_pref=None):
    """n shots from recs: every unit once (clips with no shot yet first, then by their best moment as it plays in this
    format -- framing, face and shake are already in its rank), then extra shots by fair share, up to cap each."""
    shots, count = [], {}
    order = sorted(recs, key=lambda r: (P.count.get(r["unit"]["clip"], 0),
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


PLAIN = ("reaction", "ending", "band", "logo")               # shots a coverage pass never replaces or splits


def cover_all(P, seq, recs, home):
    """Every export file in the reel -- a unit, not just its camera clip: a clip the editor split into five social
    segments is five files, and each one appears. A unit with no shot takes the place of the weakest plain shot of a
    unit that has several, in the act where it belongs (home: record id -> act; the social floor may also go in the
    climax); with no such shot, a long plain shot there is split to make room (more cuts, the grid unchanged). Never
    beside its own clip, never B-roll beside B-roll."""
    have = {s["unit"]["id"] for s in seq}
    for r in sorted([r for r in recs if r["unit"]["id"] not in have],
                    key=lambda r: -max((rk(m, P.fmt) for m in r["moments"]), default=0)):
        x = P.best([r])
        if not x:
            continue
        act = home.get(r["id"], "night" if r["role"] == "transition" else "climax")
        acts = [act] + (["climax", "night"] if r["role"] == "dance" else [])
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
        and not x.get("logo") and not x.get("add")


def picks_once(seq):
    """The editor's picks take precedence: each plays once (a repeat's beats go to the shot before it, or after it at
    an act's start), never sped up (slow motion is fine: FAST effects come off). Changes seq in place."""
    seen = set()
    i = 0
    while i < len(seq):
        s = seq[i]
        if s.get("span"):
            if any(f in FAST for f in s["fx"]):
                s["fx"] = [f for f in s["fx"] if f not in FAST]
                s["effect"] = s["fx"][0] if s["fx"] else None
            k = (s["unit"]["id"], tuple(s["span"]))
            if k in seen and len(seq) > 1:
                j = i - 1 if i > 0 and seq[i - 1]["act"] == s["act"] else i + 1 if i + 1 < len(seq) else i - 1
                seq[j]["beats"] += s["beats"]
                seq.pop(i)
                continue
            seen.add(k)
        i += 1


def fit_pick(seq, i, beat):
    """The editor's selection plays as long as they chose (to the nearest beat): its shot gives spare beats to the
    plain shot nearest it, and takes the beats it needs from the weakest plain shots in its act -- condensing them to
    2 beats, then to 1, then dropping the weakest whose unit is in the reel elsewhere; the act's best shots (top
    quarter by rank) are never dropped. The act's grid stays exact."""
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
        if d < 0:
            seq[near[0]]["beats"] -= d; s["beats"] += d
            return
        weak = sorted(near, key=lambda j: seq[j].get("rank", 0))
        j = next((j for j in weak if seq[j]["beats"] > 2), None) or next((j for j in weak if seq[j]["beats"] == 2), None)
        if j is not None:
            g = min(seq[j]["beats"] - (2 if seq[j]["beats"] > 2 else 1), d)
            seq[j]["beats"] -= g; s["beats"] += g
            continue
        units = Counter(x["unit"]["id"] for x in seq)
        top = np.quantile([seq[j].get("rank", 0) for j in near], 0.75)
        j = next((j for j in weak if units[seq[j]["unit"]["id"]] > 1 and 0 < j < len(seq) - 1 and seq[j].get("rank", 0) <= top),
                 None)
        if j is None:
            return
        s["beats"] += seq.pop(j)["beats"]


def include_adds(P, seq, recs, home, beat=None):
    """The editor's extra shots (picks "add") are in the reel: each one not already shown takes the place of the
    weakest plain shot in its act (home: record id -> act), never beside its own clip -- or, with no such shot, splits
    the longest plain one. A selection (an in-out) runs its own length (fit_pick)."""
    for r in recs:
        for m in [m for m in r["moments"] if m.get("add")]:
            near = [k for k, x in enumerate(seq) if x["unit"]["id"] == r["unit"]["id"] and abs(x["t"] - m["t"]) < 1.5
                    and not (m.get("span") and x.get("span") and x["span"] != m["span"])]
            if near:
                if m.get("span"):
                    seq[near[0]].update(span=m["span"], add=True)
                    fit_pick(seq, near[0], beat)
                continue
            act = home.get(r["id"], "night" if r["role"] == "transition" else "climax")
            clip, units = r["unit"]["clip"], Counter(x["unit"]["id"] for x in seq)

            def fits(i, act=act, clip=clip, units=units, r=r):
                return (plain(seq[i]) and 0 < i < len(seq) - 1 and units[seq[i]["unit"]["id"]] > 1
                        and clip not in (seq[i - 1]["unit"]["clip"], seq[i + 1]["unit"]["clip"])
                        and not (r["role"] == "transition" and "transition" in (seq[i - 1]["unit"]["role"],
                                                                                seq[i + 1]["unit"]["role"])))
            idx = [i for i, x in enumerate(seq) if x["act"] == act and fits(i)] or \
                [i for i, x in enumerate(seq) if x["act"] not in ("teaser", "finale") and fits(i)]
            if idx:
                i = max(idx, key=lambda i: (seq[i]["act"] == act, P.load(seq[i]["unit"]["clip"]), -seq[i]["rank"]))
                old = seq[i]
                P.count[old["unit"]["clip"]] -= 1
                seq[i] = P.take(r, m, beats=old["beats"], act=old["act"])   # (the act's grid stays exact)
            else:                                            # split the longest plain shot that is not beside its clip
                long = [i for i, x in enumerate(seq) if plain(x) and x["beats"] >= 3 and 0 < i < len(seq) - 1
                        and x["act"] not in ("teaser", "finale") and clip not in (x["unit"]["clip"],
                                                                                   seq[i + 1]["unit"]["clip"])]
                if not long:
                    print(f"[reel] no room for the pick {r['unit']['id']} {m['t']:.1f}")
                    continue
                j = max(long, key=lambda i: seq[i]["beats"])
                seq[j]["beats"] -= 1
                i = j + 1
                seq.insert(i, P.take(r, m, beats=1, act=seq[j]["act"]))
            seq[i]["add"] = True
            fit_pick(seq, i, beat)


KISS_Q = 0.92            # SigLIP kiss confidence that makes a kiss a must-have
LINGER = 2.5             # seconds the camera stays on someone (people.linger): filmed on purpose, placed first
SEEN = 0.045             # a face this big (fraction of frame height) counts as the person being in the shot


def split_for(seq, cover, count, shows, items, act, clip, role="dance"):
    """A shot to split for an extra one after it: long enough (4+ beats, so both keep 2; 3+ in the climax),
    plain, and still showing everyone it is the only shot of once it is shorter. The longest in the act first."""
    def ok(i):
        x = seq[i]
        need = 3 if x["act"] == "climax" else 4                    # both halves keep 2 beats (climax: 1 for the insert)
        if (x["beats"] < need or x["fx"] or x.get("ending") or x.get("logo") or x.get("band") or x.get("reaction") or x.get("span")
                or not 0 < i < len(seq) - 1 or clip in (x["unit"]["clip"], seq[i + 1]["unit"]["clip"])
                or (role == "transition" and "transition" in (x["unit"]["role"], seq[i + 1]["unit"]["role"]))):
            return False                                         # (no B-roll beside B-roll)
        short = dict(x, beats=x["beats"] - (2 if x["beats"] >= 4 else 1))
        return all(count[c] > 1 or shows(short, items[c]) for c in cover[i])
    idx = [i for i in range(len(seq)) if seq[i]["act"] == act and ok(i)] or \
        [i for i in range(len(seq)) if seq[i]["act"] not in ("teaser", "finale") and ok(i)]
    return max(idx, key=lambda i: (seq[i]["act"] == act, seq[i]["beats"])) if idx else None


def cover_people(P, seq, recs, home, people, fmt, beat):
    """Everyone in it (people.json: the people step): every kiss, then every person not yet seen in a shot. The
    moment that shows the most people still missing goes in first (a group beats a single face), in place of the
    weakest plain shot in its act -- never a shot that is someone's only appearance, never one that would take a
    unit out of the reel, never beside its own clip -- or, when there is no such shot, by splitting a long one
    (split_for). A person is in a shot when their face is big enough inside the shot's stretch (and, in portrait,
    inside the 9:16 strip). Returns (people shown, people, kisses shown, kisses)."""
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
    spots = {(u, t) for _, u, t in kisses}
    for _, p in persons:
        spots |= {(a["unit"], round(a["t"] * 2) / 2) for a in p["appearances"]
                  if a["unit"] in by and a["h"] >= SEEN * 1.3 and abs(a["cx"] - 0.5) <= half_w}
    probe = {sp: [k for k, it in enumerate(items) if shows(dict(unit=dict(id=sp[0]), t=sp[1], beats=2), it)]
             for sp in spots}
    tried = set()
    while True:
        gain = {sp: [k for k in ks if not count[k]] for sp, ks in probe.items() if sp not in tried}
        gain = {sp: ks for sp, ks in gain.items() if ks}
        if not gain:
            break
        sp = max(gain, key=lambda sp: (any(items[k][0] == "kiss" for k in gain[sp]),       # kisses, then the people
                                       sum(featured(items[k]) for k in gain[sp]),           # the camera stayed on,
                                       len(gain[sp])))                                      # then anyone
        tried.add(sp)
        r, t = by[sp[0]], sp[1]
        near = min(r["moments"], key=lambda m: abs(m["t"] - t))
        m = dict(near, t=t, pick=True)
        act = home.get(r["id"], "night" if r["role"] == "transition" else "climax")
        clip = r["unit"]["clip"]
        units = {}
        for x in seq:
            units[x["unit"]["id"]] = units.get(x["unit"]["id"], 0) + 1

        trade = any(featured(items[k]) for k in gain[sp])        # someone filmed on purpose may displace passers-by

        def ok(i, r=r, clip=clip, units=units, trade=trade):
            x = seq[i]
            return (not x["fx"] and not x.get("reaction") and not x.get("ending") and not x.get("band")
                    and not x.get("logo") and not (x.get("add") and not x.get("person")) and 0 < i < len(seq) - 1
                    and units[x["unit"]["id"]] > 1
                    and all(count[c] > 1 or (trade and items[c][0] == "person" and not featured(items[c]))
                            for c in cover[i])
                    and clip not in (seq[i - 1]["unit"]["clip"], seq[i + 1]["unit"]["clip"])
                    and not (r["role"] == "transition" and "transition" in (seq[i - 1]["unit"]["role"],
                                                                            seq[i + 1]["unit"]["role"])))
        idx = [i for i in range(len(seq)) if seq[i]["act"] == act and ok(i)] or \
            [i for i in range(len(seq)) if seq[i]["act"] not in ("teaser", "finale") and ok(i)]
        if idx:                                                  # take a plain shot's place
            i = max(idx, key=lambda i: (seq[i]["act"] == act, P.load(seq[i]["unit"]["clip"]), -seq[i]["rank"]))
            old = seq[i]
            P.count[old["unit"]["clip"]] -= 1
            count[cover[i]] -= 1
            seq[i] = P.take(r, m, beats=old["beats"], act=old["act"])   # (the grid stays exact)
        else:                                                    # or split one: it keeps most of its beats
            i = split_for(seq, cover, count, shows, items, act, clip, r["role"])
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


def apply_picks(recs, picks):
    """The editor's own calls, by export file name, with times in that file's own timeline (m:ss or seconds):
        {"C2025": {"ban": ["0:00-0:20"], "prefer": ["0:41-0:46"], "add": ["1:12.5"], "more": 2},
         "C1984-05_couple": {"ban": ["all"]}}
    Banned stretches lose their moments ("all": the whole clip); preferred ones jump the queue (+1.0) -- and if no
    detected moment falls inside, one is made at the middle of the range (kind and framing from the nearest), so a
    suggestion always lands. "add": extra shots the reel must include, each at that time (include_adds); a range
    ("1:12.4-1:15.1", from the source browser) is the editor's in-out: one shot, from its in-point, its length. "more":
    ask for more shots from this clip -- its fair-share discount is divided by 1 + more and its moments rank up.
    Keys starting with "_" are notes. Changes recs in place."""
    picks = {k: v for k, v in (picks or {}).items() if not k.startswith("_")}
    for r in recs:
        p = picks.get(r["id"]) or picks.get(r["unit"]["id"])
        if not p:
            continue
        u = r["unit"]

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
        for v in p.get("add", []):
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
            m["pick"] = m["add"] = True
            if b - a > 0.05:
                m["span"] = [round(a, 3), round(b, 3)]
                # the pick's footage is the pick's alone: no other moment in it, and no other shot over it
                r["moments"] = [x for x in r["moments"] if x is m or not a <= x["t"] <= b]
                u["bans"] = [*u.get("bans", []), [a, b]]
        if p.get("more"):
            r["more"] = max(int(p["more"]), 0)
            for m in r["moments"]:
                for k in [k for k in ("score_l", "score_p") if k in m] or ["score"]:
                    m[k] += 0.15 * r["more"]


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


def story(fmt, cfg, music, moments, video_beats, total_beats, direction=None):
    """The shots, act by act, with their beats summing to video_beats. direction(unit, t) -> (dx_in, dx_out) says
    which way a moment travels (flow.Flow); without it, cuts match on the kind of move only."""
    from .flow import still
    direction = direction or still
    recs = [r for r in moments if r["moments"] and (fmt == "portrait" or r["orient"] == "landscape")]
    P = Picker(fmt, recs)
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
    wide = sorted([r for r in night if rank_tag(r) is not None], key=lambda r: (r["orient"] != fmt, rank_tag(r)))
    wide = [r for k, r in enumerate(wide) if all(r["unit"]["clip"] != q["unit"]["clip"] for q in wide[:k])]
    if len(wide) >= 2:
        for k, r in enumerate(wide[:2]):                     # two different clips
            s = P.take(*P.best([r]), beats=budget["teaser"] // 2, act="teaser"); s["logo"] = "in" if k == 0 else "hold"
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
    lengths = rhythm(budget["shows"], "medium", phrase_energy(music, acts["shows"][0], budget["shows"] // 8))
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
    pool = dance if fmt == "portrait" else shows or dance
    if late_dance:                                           # the headliner's set first, the rest of the day after
        pool = late_dance if len(late_dance) >= 4 else late_dance + pool
    crowd = [r for r in late_crowd if "crowd" in tag(r)] + crowd
    heroes = P.pool(pool, ("spin", "dip", "trick"), 0.45)
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
                s = P.take(*(peak or pick(pool, MOVES, 0.4)), beats=6, act="climax"); s["fx"] = ["bullet"]
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
                [x for x in P.pool(src) if not clash(x)] or P.pool(src) or P.pool(dance)
            prev = cl[-1] if cl else seq[-1]                 # cut on the action: carry the last shot's movement on
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
    cl[-1]["fx"], cl[-1]["effect"] = ["bullet", "aura"], "aura"
    seq += cl

    # finale: faces, the packed floor, and one last slow-motion move
    fi = []
    dance = late_dance + dance if late_dance else dance
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
        near = min(r["moments"], key=lambda m: abs(m["t"] - t))
        s = P.take(r, dict(near, t=round(t, 2), kind="crowd"), beats=rest, act="finale"); s["ending"] = True
    else:
        s = P.take(*move(("dip", "spin", "trick"), 0.4), beats=rest, act="finale")
        s["fx"], s["effect"] = ["hero"], "hero"
    fi.append(s)
    seq += fit(fi, budget["finale"], 2)
    cover_all(P, seq, recs, home)
    include_adds(P, seq, recs, home, music["beat"])
    cov = cover_people(P, seq, recs, home, cfg.get("people"), fmt, music["beat"])
    if cov:
        print(f"[reel] {fmt}: people the camera stayed on {cov[4]}/{cov[5]}, everyone {cov[0]}/{cov[1]}, "
              f"kisses {cov[2]}/{cov[3]}")
    band_in(P, seq, recs, music, cfg)
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
    picks_once(seq)
    for s in [s for s in seq if s.get("span")]:            # later passes may have moved beats: the editor's picks last
        fit_pick(seq, seq.index(s), music["beat"])
    # whatever the acts could not fill (or overfilled) comes off / goes on the plain shots, so the grid stays exact
    over = sum(s["beats"] for s in seq) - video_beats
    plain = sorted([s for s in seq if not s["fx"] and not s.get("span")], key=lambda s: s["rank"]) or seq
    i = 0
    while over != 0 and i < 50 * len(plain):
        s = plain[i % len(plain)]
        if over > 0 and s["beats"] > 1:
            s["beats"] -= 1; over -= 1
        elif over < 0:
            s["beats"] += 1; over += 1
        i += 1
    return seq, acts


def transitions(seq, music, cfg, direction=None):
    """Mostly hard cuts on the beat. Each act has its own short list of transitions and a cap on each (grammar.py),
    the change of act its signature one, and a transition goes in only where the cut has its reason."""
    bt = np.array(music["beats"]); i0 = int(np.argmin(np.abs(bt - music["start"])))
    acc = np.array(music.get("beat_accent") or np.ones(len(bt)))[i0:]
    rec, pos = 0, []
    for s in seq:
        pos.append(rec); rec += s["beats"]
    room = [s["beats"] for s in seq]                        # beats left in each shot for transition halves

    def put(i, name, why):
        velocity = name == "velocity"                       # a velocity edit: out fast, in fast, a swipe between
        name = "swipe" if velocity else name
        h = G.HALF[name]
        if room[i - 1] < h + 0.5 or room[i] < h + 0.5:
            return False
        if why != "act change" and (seq[i - 1].get("transition") or (i + 1 < len(seq) and seq[i + 1].get("transition"))):
            return False                                    # never back to back
        A, B = seq[i - 1], seq[i]
        if velocity and not (any(f in SPEED for f in A["fx"] + B["fx"]) or A["beats"] < 2 or B["beats"] < 2
                             or A.get("span") or B.get("span")):        # (an editor's pick plays at real speed)
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
            put(i, SIGNATURE.get((A["act"], B["act"]), "dipwhite"), "act change")
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


def reframe(project, unit, t0, length, fmt, focus=True):
    """A crop that keeps the subject centred: one zoom for the shot, the centre gliding with the dancers. None when
    the source fills the frame as it is and there is nothing to follow. The editor's focal points for this format
    (picks "<fmt>", set in the source browser) take over the centre where there are any (focus=False: as derived)."""
    c = _reframe(project, unit, t0, length, fmt)
    return focus_override(c, focus_keys(project.picks(), unit, fmt), t0, length) if focus else c


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


def _reframe(project, unit, t0, length, fmt):
    W, H = SIZE[fmt]
    path = project.path("scores", unit["score"] + ".npz")
    d = FR.load(path) if os.path.exists(path) else None
    yf = project.path("scores", f"faces_{unit['id']}.npz")   # the unit's face scan (people step), when there is one
    if d is not None and os.path.exists(yf) and not unit.get("rotate"):
        d["yf"] = dict(np.load(yf))
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
    if unit["role"] != "transition":                        # a couple dancing apart in a strip: follow one of them
        union, face, one = FR.follow(d, pt[sel], union, cw0 if narrow else 9.0)
        if one:                                             # the group's edge contact no longer applies: the
            touch = None                                    # dancer's own box is checked against the frame below
    _, z, _, _ = FR.framing(union, src_w, src_h, W, H, touch=touch)
    ok = ~np.isnan(union[:, 0])
    if not ok.any():
        return centre
    zz = 1.0 if unit["role"] == "transition" else float(np.median(z[ok]))   # one zoom for the whole shot
    cw, ch = cw0 / zz, ch0 / zz
    sx = (union[ok, 0] + union[ok, 2]) / 2
    sy = union[ok, 1] + 0.45 * (union[ok, 3] - union[ok, 1])
    ts = pt[sel][ok] - t0
    grid = np.arange(0, length + 0.01, 0.1)

    def smooth(v, sig):
        x = np.interp(grid, ts, v)
        k = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2); k /= k.sum()
        return np.convolve(np.pad(x, 3 * sig, mode="edge"), k, mode="valid")
    x = np.clip(smooth(sx, 3 if narrow else 6), cw / 2, 1 - cw / 2)   # ~0.3 / 0.6 s: like a camera operator
    y = np.clip(smooth(sy, 10), ch / 2, 1 - ch / 2)
    # how well the crop holds the subject over the shot itself: present (mostly in the crop), inside (wholly in it,
    # not cut by the crop or the camera's own edge) and face (a face to the camera, in the crop), as fractions
    inn = (pt[sel] >= t0) & (pt[sel] <= t0 + length)
    stats = dict(present=0.0, inside=0.0, face=0.0)
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
        stats = dict(present=float((ov >= 0.6).mean()), inside=float(((ov >= 0.92) & ~edge).mean()),
                     face=float((face[inn] & (ov >= 0.6)).mean()))
    return dict(w=round(cw, 4), h=round(ch, 4), z=round(zz, 3),
                path=[[round(float(a), 2), round(float(b), 4), round(float(c), 4)]
                      for a, b, c in zip(grid[::2], x[::2], y[::2])],
                **{k: round(v, 2) for k, v in stats.items()})


def assess(project, recs, fmt, cfg):
    """How each moment would actually play in this format, over the stretch a shot shows (-0.8..+1.6 s): the subject
    held in the crop and a face to the camera (reframe's stats), and camera shake (shake.py) as the viewer sees it --
    a crop a third of the frame wide shows the camera's jitter three times over. A moment too shaky to use, or whose
    subject keeps leaving the crop, drops out (another part of the clip takes its place); a whole subject ranks up;
    in portrait a face ranks up a lot -- most of all on the social floor, where it's otherwise people's backs.
    Editor's picks are never dropped. Reframes are cached in hold_cache.json."""
    from . import shake
    lo, hi = cfg["shake"]
    key = "score_p" if fmt == "portrait" else "score_l"
    path = project.path("hold_cache.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for r in recs:
        u = r["unit"]
        yf = "|yf" if os.path.exists(project.path("scores", f"faces_{u['id']}.npz")) else ""   # (face scan: re-assess)
        for m in r["moments"]:
            k = f"{fmt}|{u['id']}|{m['t']}{yf}"
            if k not in cache:
                c = reframe(project, u, m["t"] - 0.8, 2.4, fmt) or {}
                cache[k] = [c.get("present", 1.0), c.get("inside", 1.0), c.get("face", 0.0), c.get("w", 1.0)]
            present, inside, face, cw = cache[k]
            j = shake.jitter(project, u, m["t"] - 0.8, m["t"] + 1.6)
            j = j / max(cw, 0.2) if j is not None else None
            m["jitter"] = round(j, 5) if j is not None else None
            sc = m.get(key, m["score"])
            if j is not None and j > hi and not m.get("pick"):
                sc *= 0.03
            elif j is not None and j > lo:
                sc *= 1 - 0.35 * min((j - lo) / (hi - lo), 1)
            if r["role"] != "transition":
                m["present"], m["inside"], m["face_vis"] = present, inside, face
                g = 0.04 if present < 0.6 and not m.get("pick") else 1.0
                social = r["role"] == "dance"
                g *= (0.3 + 0.7 * inside) if social else (0.45 + 0.55 * inside)
                if fmt == "portrait":
                    g *= ((0.3 + 1.4 * face) * (0.25 if face < 0.3 else 1)) if social else (0.5 + 1.0 * face)
                else:
                    g *= 0.85 + 0.3 * face
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
    if os.path.exists(project.path("people.json")):          # everyone in it (the people step)
        cfg["people"] = project.load("people.json")
    moments = copy.deepcopy(moments)
    if picks:
        apply_picks(moments, picks)
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
    video_beats = total_beats - (round(STINGER_OVERLAP / beat) if stinger_len else 0)   # the stinger covers the end
    seq, acts = story(fmt, cfg, music, moments, video_beats, total_beats, direction)
    plan = place(project, fmt, seq, music, cfg, direction, stinger_len, acts)
    mo = [r for r in moments if r["moments"] and (fmt == "portrait" or r["orient"] == "landscape")]
    plan["units_total"] = len({r["id"] for r in mo})
    plan["music_length"] = music["length"]
    return plan


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
    lands(project, seq, beat, hold)
    rec = 0
    for s in seq:
        L = T(rec + s["beats"]) - T(rec)
        s["beat"] = round(L / s["beats"], 5)                  # this shot's own beat (the tempo may change)
        u = s["unit"]
        ramp = next((f for f in s["fx"] if f in V.RAMP), None)
        if ramp:                                             # velocity edit: slow on the landing, on a strong beat
            s["t"] = V.settle(project, u, s["t"])
            s["lead_beats"] = V.landing(s["beats"], ramp, s["kind"] == "closeup", acc[rec:])
        src_in = in_point(s, s["beat"])
        if s.get("land") is not None:                        # the move's finish and the rest after it
            src_in, s["land_fits"] = V.hold_in(src_in, L, s["land"], hold)
            src_in = clear_of_bans(min(max(src_in, u["t0"]), max(u["t1"] - L, u["t0"])), L, u)
        if s.get("span"):                                    # the editor's selection starts at their in-point
            src_in = in_point(s, s["beat"])
        if s.get("ending"):                                  # the configured ending runs out at its "to" time
            src_in = max(u.get("export_t0", 0.0) + cfg["ending"][2] - L, u["t0"])
        f0, f1 = round(T(rec) * FPS), round(T(rec + s["beats"]) * FPS)
        s.update(rec_in=round(T(rec), 3), length=round(L, 3), frames=f1 - f0, src_in=round(src_in, 3),
                 peak=round(s["t"] - src_in, 3), group=u["role"] == "show")
        frame(s)
        rec += s["beats"]
    transitions(seq, music, cfg, direction)
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


def run(project, fmt, stinger_len=0.0) -> dict:
    from .flow import Flow
    fl = Flow(project)
    plan = build(project, fmt, project.load("music.json"), project.load("moments.json"), stinger_len, fl.direction)
    fl.save()
    project.dump(f"plan_{fmt}.json", plan)
    seq = plan["shots"]
    used = len({s["unit"]["id"] for s in seq})
    trs = [s.get("transition") for s in seq[1:] if s.get("transition")]
    fxs = [f for s in seq for f in s["fx"]]
    print(f"[reel] {fmt}: {len(seq)} shots, {used}/{plan['units_total']} units, video {plan['video_length']:.1f}s"
          f" + stinger {stinger_len:.1f}s")
    for name in ACT_NAMES:
        ss = [s for s in seq if s["act"] == name]
        print(f"[reel]   {name:8s} {len(ss):3d} shots, {sum(bool(s.get('transition')) for s in ss)} transitions, "
              f"fx {[f for s in ss for f in s['fx']]}")
    print("[reel]   transitions:", {t: trs.count(t) for t in sorted(set(trs))})
    print("[reel]   effects:", {f: fxs.count(f) for f in sorted(set(fxs))})
    return plan
