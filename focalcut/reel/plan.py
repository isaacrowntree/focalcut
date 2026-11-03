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


def picks_once(seq):
    """The editor's picks take precedence: each plays once (a repeat's beats go to the shot before it, or after it at
    an act's start), never sped up (slow motion is fine: FAST effects come off). A pick that is also the configured
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
                seq[j]["beats"] += s["beats"]
                seq.pop(i)
                continue
            seen.add(k)
        i += 1


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
        if d < 0:
            seq[near[0]]["beats"] -= d; s["beats"] += d
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
        quick_cuts(seq, gap, act, beat, k)
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


def quick_cuts(seq, n, act, beat, at=None, size=1):
    """n beats of quick cuts, size beats each, from the editor's picks, put into act at at (default: spread through
    it, never at its ends): first the ends that pick_beat trimmed off them (footage not seen yet), then the openings
    of picks two acts or more away (never footage seen moments ago); crowd and B-roll picks first, never the same clip
    twice in a row. A cut is plain footage: no logo, title, effect or ending from the pick it comes from."""
    picks = [s for s in seq if s.get("span") and not s.get("reuse")]
    if not n or not picks:
        return
    used = {(s["unit"]["id"], round(s["span"][0], 2)) for s in seq if s.get("reuse")}
    names = [a for a in ACT_NAMES if any(x["act"] == a for x in seq)]
    near = set(names[max(0, names.index(act) - 1):names.index(act) + 1]) if act in names else {act}
    order = sorted(picks, key=lambda s: (s["unit"]["role"] != "transition", s["act"] != act,
                                         -(s["span"][1] - s["span"][0])))
    tails = [(s, [s["span"][0] + s["beats"] * beat, s["span"][1]]) for s in order]
    srcs = [(s, sp) for s, sp in tails if sp[1] - sp[0] >= 0.8 * beat * size] + \
        [(s, [s["span"][0], min(s["span"][1], s["span"][0] + beat * size)]) for s in order
         if s["act"] not in near and s["span"][1] - s["span"][0] >= 0.8 * beat * size]
    srcs = [(s, sp) for s, sp in srcs if (s["unit"]["id"], round(sp[0], 2)) not in used]
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
    if out:
        out[-1]["beats"] += n - sum(c["beats"] for c in out)   # (fewer sources than beats: the last cut holds)
    seq[at:at] = out


def spread_picks(seq, recs, home, beat, budget, full=0.9):
    """Where the editor's selections would overfill their act (more of their beats than the act has, so each would
    be cut to a beat) and a later act has room -- above all an act the planner left empty, its footage all picks --
    the act's last selections (by source time) move there: {(record id, moment time): act}. B-roll stays home.
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
            later = [b for b in order[order.index(a) + 1:] if load[b] + want(m) <= budget.get(b, 0) * full]
            if not later:
                break
            b = min(later, key=lambda b: load[b] / max(budget[b], 1))
            out[(r["id"], m["t"])] = b
            load[a] -= want(m); load[b] += want(m)
    return out


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
            last = mine[-1]
            if b and spare > 0 and not any(last.get(k) for k in ("ending", "open", "logo")) \
                    and (spare >= last["beats"] or over >= last["beats"]):
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
    for r in recs:
        for m in [m for m in r["moments"] if m.get("add")]:
            near = [k for k, x in enumerate(seq) if x["unit"]["id"] == r["unit"]["id"] and abs(x["t"] - m["t"]) < 1.5
                    and not (m.get("span") and x.get("span") and x["span"] != m["span"])]
            if near:
                if m.get("span"):
                    seq[near[0]].update(span=m["span"], add=True)
                    fit_pick(seq, near[0], beat)
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
            # a pick comes before "use everything": with no other place free, it takes a clip's only shot
            idx = next((ix for ix in (                       # (its own act first, even a clip's only shot)
                [i for i, x in enumerate(seq) if x["act"] == act and fits(i)],
                [i for i, x in enumerate(seq) if x["act"] == act and fits(i, lone=True)],
                [i for i, x in enumerate(seq) if x["act"] != "teaser" and fits(i)],
                [i for i, x in enumerate(seq) if x["act"] != "teaser" and fits(i, lone=True)]) if ix), [])
            if idx:
                i = max(idx, key=lambda i: (seq[i]["act"] == act, P.load(seq[i]["unit"]["clip"]), -seq[i]["rank"]))
                old = seq[i]
                P.count[old["unit"]["clip"]] -= 1
                seq[i] = P.take(r, m, beats=old["beats"], act=old["act"])   # (the act's grid stays exact)
            else:                                            # split the longest plain shot that is not beside its clip
                long = [i for i, x in enumerate(seq) if plain(x) and x["beats"] >= 3 and 0 < i < len(seq) - 1
                        and x["act"] not in ("teaser", "finale") and clip not in (x["unit"]["clip"],
                                                                                   seq[i + 1]["unit"]["clip"])]
                if long:
                    j = max(long, key=lambda i: seq[i]["beats"])
                    seq[j]["beats"] -= 1
                else:                                        # no plain shot left to give way: in after any shot
                    gaps = [j for j in range(len(seq) - 1) if seq[j]["act"] != "teaser"   # of its act not beside
                            and clip not in (seq[j]["unit"]["clip"], seq[j + 1]["unit"]["clip"])]   # its clip;
                    gaps = [j for j in gaps if seq[j]["act"] == act] or gaps   # the grid is evened out at the end
                    if not gaps:
                        print(f"[reel] no room for the pick {r['unit']['id']} {m['t']:.1f}")
                        continue
                    j = gaps[len(gaps) // 2]
                i = j + 1
                seq.insert(i, P.take(r, m, beats=1, act=seq[j]["act"]))
            seq[i]["add"] = True
            fit_pick(seq, i, beat)


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
    spots = {sp for sp in spots if band_ok(sp)}

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
    A clip with in-out picks gives the reel those and nothing else: its other moments go, and the rest of it is
    banned (single-time adds keep 1.5 s either side) -- unless not only (picks cut to pieces, pick_pieces: the rest of
    the clip stays in play, between the picks). Keys starting with "_" are notes. Changes recs in place."""
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


def pick_pieces(project, recs, fmt, beat, hold, budget, share=None, ending=None):
    """Long picks cut down to their best part (reel.json plan.<fmt> "pick_hold": beats): each in-out longer than
    hold beats becomes its best hold-beat window, scored on the moment curve (moments.curve: moves, quality, framing
    in this format, activity, face) -- the editor chose the stretch, the planner the best of it. Then the picks are
    condensed to a share of the reel's beats (budget, default half), shared out by subject: a clip with many picks
    keeps its best ones up to its share (at least one; 1.5x the even share, at least two pieces' worth), and a long
    pick earns a second piece of itself when there is room and it plays nearly as well (PIECE_KEEP). Each piece is a
    shot of its own (include_adds spreads them through the act, between the rest of the footage). The ending (cfg
    "ending": [file, from, to, ...]) is left whole and isn't counted. Returns the picks
    left out ([export file, from, to] in source seconds, for pick_report). Changes recs in place."""
    from . import moments as MO
    series, cands = {}, []
    w = hold * beat
    for r in recs:
        for m in [m for m in r["moments"] if m.get("add") and m.get("span") and not m.get("open")]:
            a, b = m["span"]
            u = r["unit"]
            if ending and ending[0] in (r["id"], u["id"]) and abs(u.get("export_t0", 0.0) + ending[1] - a) < 0.05:
                continue
            path = project.path("scores", u.get("score", u["id"]) + ".npz")
            if path not in series:
                series[path] = MO.clip_series(path) if os.path.exists(path) else None
            S = series[path]
            t, c = (S["t"], MO.curve(u, S)[0][fmt]) if S is not None else (None, None)
            win = lambda x, y, t=t, c=c, S=S: (0.5 * c[k].mean() + 0.5 * c[k].max()) if S is not None and \
                (k := (t >= x) & (t <= y)).any() else 0.0
            if b - a <= w + 0.5 * beat:
                wins = [(win(a, b), a, b)]
            else:                                            # every hold-beat window, best first, none overlapping
                xs = sorted(((win(x, x + w), x, x + w) for x in np.arange(a, b - w + 1e-6, 0.1)), key=lambda z: -z[0])
                wins = []
                for z in xs:
                    if all(z[1] >= y[2] or z[2] <= y[1] for y in wins):
                        wins.append(z)
            cands.append(dict(r=r, m=m, wins=wins, subject=u["id"]))
    if not cands:
        return []
    room = budget * (share or 0.5)
    n = lambda z: max(1, round((z[2] - z[1]) / beat))
    subj = Counter(c["subject"] for c in cands)
    cap = max(2 * hold, 1.5 * room / len(subj))
    keep = {id(c): [c["wins"][0]] for c in cands}
    load = Counter()
    for c in cands:
        load[c["subject"]] += n(c["wins"][0])
    dropped = []
    while True:                                              # over its share, or the picks over theirs: the subject
        total = sum(load.values())                           # with the most drops its weakest pick
        over = [s for s in subj if load[s] > cap and subj[s] > 1] or \
            ([s for s in subj if subj[s] > 1] if total > room else [])
        if not over:
            break
        s = max(over, key=lambda s: load[s])
        c = min((c for c in cands if c["subject"] == s and id(c) in keep), key=lambda c: keep[id(c)][0][0])
        load[s] -= n(keep.pop(id(c))[0]); subj[s] -= 1
        dropped.append(c)
    extra = sorted(((z, c) for c in cands if id(c) in keep for z in c["wins"][1:]
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
    pool = dance if fmt == "portrait" else shows or dance
    if cfg.get("climax_from"):
        pool = [r for r in pool if any(k in cat(r) for k in cfg["climax_from"])] or pool
    if late_dance:                                          # the headliner's set first, the rest of the day after
        pool = late_dance if len(late_dance) >= 4 else late_dance + pool
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
                [x for x in P.pool(src) if not clash(x)] or P.pool(src) or \
                (cfg.get("climax_from") and (cl or seq)[-1]["unit"]["role"] != "transition"     # the night's own
                 and [x for x in P.pool(late_crowd + night) if x[0]["unit"]["clip"] != last_clip]) or \
                P.pool(dance)                                # B-roll before footage the climax was kept from
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
    picks_once(seq)
    for s in [s for s in seq if s.get("span")]:            # later passes may have moved beats: the editor's picks last
        fit_pick(seq, seq.index(s), pbeat, floor=2 if cfg.get("pick_hold") else 1)
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
    yf = _cached(project.path("scores", f"faces_{unit['id']}.npz"), lambda f: dict(np.load(f)))
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
    person = unit["role"] != "transition" or people_broll(unit)   # (scenery B-roll: no one to follow)
    if person:                                              # a couple shares the frame; a group too wide for a
        union, face, one = FR.follow(d, pt[sel], union, cw0)  # strip: follow one of them
        if one:                                             # the group's edge contact no longer applies: the
            touch = None                                    # dancer's own box is checked against the frame below
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
    if narrow and person:                                   # never through someone the subject's size beside them
        zz, sx = clean_centre(d, pt[sel][ok], union[ok], sx, zz, cw0, ch0)
        cw, ch = cw0 / zz, ch0 / zz
    if narrow:                                              # a subject sitting low in the strip (a band under a
        zz = lift(union[ok], zz, cw0, ch0, FR.zmax(src_w, src_h, W, H))   # tall rig): zoom so the crop moves down
        cw, ch = cw0 / zz, ch0 / zz
    if narrow:                                              # look room: a profile gets space ahead of their gaze
        sx = sx + cw * gaze(d, pt[sel][ok], union[ok], R["look_player"] if unit["role"] != "dance" else R["look"])
    uh_ = union[ok, 3] - union[ok, 1]                       # a subject taller than the crop keeps their head in it
    sy = union[ok, 1] + 0.45 * np.minimum(uh_, 0.95 * ch)   # (not a torso, hands and a mic stand)
    x = np.clip(smooth(sx, 3 if narrow else 6), cw / 2, 1 - cw / 2)   # ~0.3 / 0.6 s: like a camera operator
    # (the gaze shift is smoothed again on its own over ~1 s inside gaze(): a head turn doesn't whip the strip)
    y = np.clip(smooth(sy, 10), ch / 2, 1 - ch / 2)
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
            (cov >= 0.5) & ~fseen
        cut, body = sliced(d, ii, union[inn], bx, by, cw, ch)
        hide, head, back, torn = shown(pb / [d["aw"], d["ah"], d["aw"], d["ah"], 1], d["kp"][ii], fseen, who, seen.any(1),
                                 bx, by, cw, ch)
        stats = dict(present=float((ov >= 0.6).mean()), inside=float(((ov >= 0.92) & ~edge).mean()),
                     face=float((face[inn] & (ov >= 0.6)).mean()),
                     action=float(np.where(seen, act, 0).max(1).mean()), churn=churn,
                     blocked=float(wall.any(1).mean()), split=cut, body=body, hide=hide, head=head, back=back,
                     torn=torn, fill=float(np.nanmedian((b[:, 3] - b[:, 1]) / ch)) if not np.isnan(b[:, 1]).all()
                     else 0.0,
                     low=float(np.nanmedian(((b[:, 1] + b[:, 3]) / 2 - (by - ch / 2)) / ch))
                     if not np.isnan(b[:, 1]).all() else 0.5)
    return dict(w=round(cw, 4), h=round(ch, 4), z=round(zz, 3),
                path=[[round(float(a), 2), round(float(b), 4), round(float(c), 4)]
                      for a, b, c in zip(grid[::2], x[::2], y[::2])],
                **{k: round(v, 2) for k, v in stats.items()})


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
        k = cost.argmin(0)
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


def shown(nb, kp, fseen, who, has, bx, by, cw, ch):
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
        apply_picks(moments, picks, only=not cfg.get("pick_hold"))
    picks_only(moments, project.config.get("picks_only", ()))
    condensed = []
    if cfg.get("pick_hold") and picks:                       # long picks: their best part; many: condensed
        b = music.get("beat") or float(np.median(np.diff(music["beats"])))
        condensed = pick_pieces(project, moments, fmt, b, cfg["pick_hold"], music["length"] / b, cfg.get("pick_share"),
                                 cfg.get("ending"))
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
    """Unused moments of the same kind of footage as seq[i] (its category first, then by rank) not beside its clip."""
    u = seq[i]["unit"]
    used = [(x["unit"]["id"], x["t"]) for x in seq]
    near = {seq[k]["unit"]["clip"] for k in (i - 1, i + 1) if 0 <= k < len(seq)}
    cands = [(r, m) for r in moments if r["role"] == u["role"] and r["unit"]["clip"] not in near
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


def run(project, fmt, stinger_len=0.0) -> dict:
    from .flow import Flow
    fl = Flow(project)
    plan = build(project, fmt, project.load(project.music_file), project.load("moments.json"), stinger_len, fl.direction)
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
    for line in V.speed_tails(plan):                         # every shot ends at real speed
        print(f"[reel]   WARNING speed: {line}")
    return plan
