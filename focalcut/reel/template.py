"""The reel cut on a reference film's own edit (reel.json "mode": "reference"; reference.py learns it).

Every shot of the reference becomes a slot: its place on the music, its length, the category of this project's
footage it looks like, its scale, and what it is made of (a split screen, a stutter, a title over it, an end card).
Each slot takes this project's moment that best fits it:

  - looks like the reference's shot (SigLIP 2: the reference frame against frames of the footage around the moment)
  - is good footage (moments.py's rank: sharp, lit, steady, framed)
  - keeps to the order things happened: each slot aims at the time of day its place in the reference's run of that
    category stands for (a run of baptisms in the reference walks through the baptisms here from first to last)
  - every one of the shoot's event (events.py, e.g. a baptism's dunk and rise) once: the reference's long shots in the
    water are the dunks, in order; more dunks than such slots and the shorter shots of the water take them too

The music is the reference's when it is the same song (the cuts land where its editor put them), else its timeline
scaled to this song; either way each cut lands on the beat nearest. Hard cuts only, as the reference has them; a
dunk plays slowed (reel.json "reference_slow", default 0.6x: the reference holds its rises), the title fades in as
it did over the opening shot (reel.json "title": the lines, font, colour), and the end card is drawn from the
reference's (its ground colours and the logo's place) with this project's logo (reel.json "endcard" / "logo").
Shots the reference plays after its music ends (a tag carried by its own sound) are left out unless reel.json
"tag": true.
"""
from __future__ import annotations

import copy
import fnmatch
import os

import numpy as np

from . import plan as P
from .render import FPS

SLOW = 0.6               # a dunk's slow motion (50p footage at 0.6x: real frames, no in-betweens to speak of)
DUNK_LEN = 3.5           # a reference shot in the water this long or longer is a dunk slot
DUNK_AT = 0.62           # where in a dunk shot the rise lands (of its length)
DUNK_SLACK = -6.0        # seconds past the next dunk's rise that ground footage from the event may be from (in_order):
                         # under zero -- before it, so a shot from there ends before they go under
CARD_MAX = 22.0         # a documentary's end card runs to the song's last note unless that is longer than this
AIR_PREFER = 0.5         # a documentary's drone b-roll away from the event: at least this "prefer" score (aim.py)
AIR_WINDOW = 900.0
AIR_BEHIND = 120.0       # nor from more than this before where the day has got to       # a documentary's drone shots: from within this many seconds of where the cut is in the day
DUNK_SLACK_AIR = -3.0    # the same for the drone's wides of the water (they show the people too: the next to go
                         # under, before they do)
CHRONO = 0.35            # weight of keeping to the order things happened
LOOKS = 0.55             # weight of looking like the reference's shot (the rest: the moment's own rank)


# -- the reference's slots on this song --------------------------------------------------------------------------
def slots(ref, tag=False):
    """The reference's shots as slots: stutter runs joined into one, tags after the music dropped (unless tag)."""
    out = []
    for s in ref["shots"]:
        if s.get("after_music") and not tag:
            continue
        if "stutter" in s and out and out[-1].get("stutter") == s["stutter"]:
            o = out[-1]
            o["t1"], o["length"] = s["t1"], round(s["t1"] - o["t0"], 3)
            o["pieces"] += 1
            continue
        o = copy.deepcopy(s)
        if "stutter" in s:
            o["pieces"] = 1
        out.append(o)
    return out


def beat_grid(music, ref, sl, same=None):
    """Each slot's beats on this song: the reference's cut times (scaled to this song unless it is the reference's
    own) on the nearest beat, at least one beat each. Returns the beats per slot and the start beat index."""
    beats = np.array(music["beats"], float)
    start = int(np.argmin(np.abs(beats - music["start"])))
    end_ref = ref.get("music_end") or ref["duration"]
    same = same if same is not None else True
    scale = 1.0 if same else (music["length"] / end_ref)
    t0 = beats[start]
    cuts = [t0 + s["t0"] * scale - (sl[0]["t0"] * scale) for s in sl] + [t0 + sl[-1]["t1"] * scale - sl[0]["t0"] * scale]
    idx = [int(np.argmin(np.abs(beats - c))) for c in cuts]
    for k in range(1, len(idx)):                             # strictly increasing: each slot at least a beat
        idx[k] = max(idx[k], idx[k - 1] + 1)
    return [b - a for a, b in zip(idx, idx[1:])], start


# -- the footage -----------------------------------------------------------------------------------------------
class Looks:
    """How much the footage around a moment looks like a reference shot (reference_frames.npz)."""
    def __init__(self, project):
        z = np.load(project.path("reference_frames.npz"))
        self.ref = z["ref"].astype(np.float32)
        self.emb = z["emb"].astype(np.float32)
        self.t = z["t"]
        self.by = {}
        for k, u in enumerate(z["unit"]):
            self.by.setdefault(str(u), []).append(k)
        self.by = {u: np.array(v) for u, v in self.by.items()}
        self.high = z["high"].astype(np.float32) if "high" in z.files else None   # aerial: flying high (0..1)
        self.cat_of = {}
        self.cats = []

    def categories(self, units):
        """Which category each unit's frames belong to (for cat_like)."""
        for u in units:
            self.cat_of[u["id"]] = u["category"]
        self.cats = sorted(set(self.cat_of.values()))
        self.cat_rows = {c: np.concatenate([self.by[u] for u, cc in self.cat_of.items() if cc == c and u in self.by])
                         for c in self.cats if any(cc == c and u in self.by for u, cc in self.cat_of.items())}

    def cat_like(self, shot_i, cat, top=5):
        k = self.cat_rows.get(cat)
        if k is None or not len(k):
            return -1.0
        v = np.sort(self.emb[k] @ self.ref[shot_i])
        return float(v[-top:].mean())

    def like(self, shot_i, uid, t, window=2.5):
        k = self.by.get(uid)
        if k is None or shot_i is None:                       # (a documentary slot copies no film's shot)
            return 0.0
        near = k[np.abs(self.t[k] - t) <= window]
        if not len(near):
            near = k[[int(np.argmin(np.abs(self.t[k] - t)))]]
        return float((self.emb[near] @ self.ref[shot_i]).max())


def clocks(project, units):
    """Wall clock per unit id (reference.wall_clock), cached in reference.json's folder."""
    from .reference import wall_clock
    path = project.path("wall_clock.json")
    have = project.load("wall_clock.json") if os.path.exists(path) else {}
    for u in units:
        if u["id"] not in have:
            have[u["id"]] = wall_clock(project, u)
    project.dump("wall_clock.json", have)
    return have


def runs(sl):
    """For each slot: its category run's (first, last) slot indices, by category -- the reference's sections in
    which a category recurs (its baptisms interleaved with the drone). A category's slots all count as one run from
    its first to its last appearance."""
    first, last = {}, {}
    for i, s in enumerate(sl):
        c = s.get("category")
        first.setdefault(c, i)
        last[c] = i
    return first, last


def ensure(body, need, looks, span, spare=(), floating=()):
    """reel.json "reference_min": {"<category>": n}: at least n slots of each named category (and one of every
    category, so no export folder is left out). Short of that, the slots that look most like the category's footage
    become its own -- only where the order things happened allows it: before the first slot of any category that
    began after it ended, and after the last of any that ended before it began. `spare`: slots never taken (the
    opening under the title)."""
    cats = set(looks.cats)
    want = {c: 1 for c in cats}
    want.update(need or {})
    for c, n in sorted(want.items(), key=lambda kv: (span.get(kv[0]) or [0])[0]):   # in the order of the day
        if c not in cats:
            continue
        have = sum(1 for s in body if s.get("category") == c)
        if have >= n:
            continue
        sp = span.get(c)
        lo, hi = 0, len(body)
        if sp:
            for i, s in enumerate(body):
                if s.get("category") in floating:              # B-roll floats: it doesn't pin the order of the day
                    continue
                o = span.get(s.get("category"))
                if o and o[0] > sp[1] + 60:
                    hi = min(hi, i)
                if o and o[1] < sp[0] - 60:
                    lo = max(lo, i + 1)
        count = {}
        for s in body:
            count[s.get("category")] = count.get(s.get("category"), 0) + 1
        def free(i, c=c, count=count):
            return (i not in spare and body[i].get("category") != c and not body[i].get("card")
                    and "stutter" not in body[i]
                    and count.get(body[i].get("category"), 0) > want.get(body[i].get("category"), 1)
                    and (not body[i].get("split") or c not in floating))
        cand = sorted((i for i in range(lo, hi) if free(i)), key=lambda i: -looks.cat_like(body[i]["i"], c))
        take = cand[: n - have]
        k = hi                                                 # short of room: the first slots of what comes next,
        while len(take) < n - have and k < len(body):          # in a row from its start (the order still holds)
            if free(k):
                take.append(k)
            elif body[k].get("category") not in floating:
                break
            k += 1
        for i in take:
            count[body[i]["category"]] -= 1
            body[i]["category"], body[i]["forced"] = c, True
    return body


def dunks(recs, name=None):
    """Every moment of the reel's event (reel.json "reference_event", default the shoot's first "events" entry)."""
    return [(r, m) for r in recs for m in r["moments"] if m.get("event") and (name is None or m["event"] == name)]


# -- the plan --------------------------------------------------------------------------------------------------
class Cut:
    """One reel cut on the reference's edit: the reference's slots on this song, this project's footage and what is
    known of it (when each clip was shot, what it looks like, how it moves, how steady it is), and the picker that
    keeps a moment from being used twice. Setting it up decides what every slot is for (which take a dunk); plan()
    then fills them -- the editor's own picks and the closing shot first, so nothing else takes them."""

    def __init__(self, project, fmt, music, moments):
        self.project, self.fmt, self.music = project, fmt, music
        self.cfg = P.config(project, fmt)
        self.ref = project.load("reference.json") if os.path.exists(project.path("reference.json")) else {"shots": []}
        self.documentary = project.config.get("mode") == "documentary"
        # the footage and what is known of it
        self.recs = self._footage(moments)
        self.looks = Looks(project)
        self._flown_moments()
        self.wall = clocks(project, [r["unit"] for r in self.recs])
        self.span = self._spans()
        self.looks.categories([r["unit"] for r in self.recs])
        self.floating = {r["unit"]["category"] for r in self.recs if r["unit"]["role"] == "transition"}
        self.beat = music.get("beat") or float(np.median(np.diff(music["beats"])))
        self.ramp = project.config.get("reference_ramp", "dunk")   # a dunk's velocity edit (fx.speed): fast in, slow
                                                                   # rise
        self.pk = P.Picker(fmt, self.recs)
        self.cams = Cameras(project, self.recs)
        self.steady = Steady(project, self.cams)
        self._clean_moments()
        self.want_moves = project.config.get("moves_for") or {}
        self.lastcat = project.config.get("reference_last")
        self.water = {c for c in (project.config.get("reference_dunk_categories") or ["baptisms"])}
        self.dk = self._dunks()
        self.shake_max = project.config.get("shake_max") or {}
        self.speeds = project.config.get("drone_speed", [4.0, 1.25])
        self.pen = project.config.get("camera_penalty") or {}  # a weaker camera wins only by a margin
        self.bad_match = self._bad_match()
        self.cap = project.config.get("camera_max") or {}      # a camera's most shots in the reel
        # the structure: the slots and their beats -- the reference film's edit, or the documentary engine's
        if self.documentary:
            from . import documentary as D
            self.body, self.card, self.beats_per, self.grid_info = D.structure(project, music, self._story())
        else:
            sl = slots(self.ref, bool(project.config.get("tag")))
            same = os.path.abspath(project.config.get("song") or "") != "" and (
                project.config.get("reference_same_song", True))
            self.body = [s for s in sl if not s.get("card")]   # the slots that take footage
            self.card = next((s for s in sl if s.get("card")), None)
            self.beats_per, _ = beat_grid(music, self.ref, self.body + ([self.card] if self.card else []), same)
        # what each slot is for
        self.first, self.last = self._categories()
        if self.lastcat and self.body:                         # the last shot before the card (the group, arms up)
            self.body[-1]["category"] = self.lastcat
        self.dunk_at = self._assign_dunks()
        self.near_dunk = self._nearest_dunks()
        self.dunk_times = sorted((i, w) for i, (r, m) in self.dunk_at.items()   # (slot, when): the event's
                                 if (w := self.when(r, m)) is not None)           # clock, as the reel plays it
        over = self.cams.over(self.water)
        self.air_units = [r for r in self.recs if r["unit"]["category"] in over]
        self.over_water = [s["i"] for s in self.body if s.get("category") in over and s.get("scale") == "aerial"] or \
            [s["i"] for s in self.body if s.get("category") in over]
        self.lineups = self._lineups()                         # categories where everyone is shown, side by side
        self.cursor = None                                     # where in the day the cut has got to
        self.relaxed, self.skipped = [], []                    # slots filled past the soft rules; slots left empty

    def _story(self):
        """What the documentary engine needs to know of the footage: the ground categories in the order of the day,
        the aerial ones and when they flew, the event and how many moments of it, how much usable footage each
        category has (seconds clear of the camera's faults), the every_unit lineups, the reference's title and card,
        and the last shot's category."""
        when = {}
        for r in self.recs:
            w = self.wall.get(r["unit"]["id"])
            if w is not None:
                when.setdefault(r["unit"]["category"], []).append(w + (r["unit"]["t1"] - r["unit"]["t0"]) / 2)
        cats = sorted(when, key=lambda c: float(np.median(when[c])))
        aerial = [(c, min(when[c])) for c in cats if self.cams.aerial(c)]
        ground = [c for c in cats if not self.cams.aerial(c) and c not in self.floating]
        rich, runs_ = {}, {}
        for r in self.recs:
            u = r["unit"]
            f = None if self.cams.aerial(u["category"]) else self.steady.faults(u)
            clean = (u["t1"] - u["t0"]) * (1.0 - float(f[1].mean())) if f is not None else (u["t1"] - u["t0"])
            rich[u["category"]] = rich.get(u["category"], 0.0) + clean
            if any(y <= u["t0"] + 0.5 and z >= u["t1"] - 0.5 for y, z in u.get("bans") or []):
                continue                                       # (a clip banned whole: none of it)
            if f is not None:                                  # each clip's longest clean stretch (a clip gives
                t, bad, _ = f                                  # one shot to a section, mostly)
                m = (t >= u["t0"]) & (t <= u["t1"])
                best = run = 0.0
                for b in bad[m]:
                    run = 0.0 if b else run + 0.1
                    best = max(best, run)
                runs_.setdefault(u["category"], []).append(round(best, 1))
        ev = next((c for c in ground if c in self.water), None)
        lineups = {}
        for c, spec in (self.project.config.get("every_unit") or {}).items():
            spec = spec if isinstance(spec, dict) else {"columns": spec}
            units = {r["unit"]["id"] for r in self.recs if r["unit"]["category"] == c and self.steady.hold.ever(r["unit"])}
            if units:
                lineups[c] = (len(units), int(spec.get("columns", 3)), bool(spec.get("grid")))
        shots = self.ref.get("shots") or []
        card = next((s for s in shots if s.get("card")), None)
        blends = {}                                             # reel.json "blend": each base shot over the dunk
        for j, b in enumerate(self.project.config.get("blend") or []):   # nearest it in time
            r = next((x for x in self.recs if b["base"][0] in (x["unit"]["id"], x["unit"]["clip"])), None)
            w0 = self.wall.get(r["unit"]["id"]) if r else None
            if w0 is None or not self.dk:
                continue
            a, _b = P.span_of(b["base"][1])
            at = w0 + a + r["unit"].get("export_t0", 0.0) - r["unit"]["t0"]
            k = min(range(len(self.dk)), key=lambda k: abs((self.when(*self.dk[k]) or 0) - at))
            blends[k] = j
        bar = 4 * self.beat
        clips = {c: sum(1 for x in v if x >= 1.5 * bar) for c, v in runs_.items()}   # (that hold 1.5 bars)
        for r in self.recs:                                    # (the editor's picks hold a shot each too)
            if any(m.get("add") for m in r["moments"]):
                clips[r["unit"]["category"]] = clips.get(r["unit"]["category"], 0) + 1
        starts = {c: min(when[c]) for c in cats}
        ends = {}
        for r in self.recs:
            w = self.wall.get(r["unit"]["id"])
            if w is not None:
                c = r["unit"]["category"]
                ends[c] = max(ends.get(c, w), w + r["unit"]["t1"] - r["unit"]["t0"])
        op = (self.project.config.get("documentary") or {}).get("open")
        slow = self.project.config.get("slow_picks") or []
        montage = (self.project.config.get("documentary") or {}).get("montage") or []   # sections cut as a montage
                                                                                      # of the editor's picks
        bar = 4 * self.beat
        bases = [P.span_of(b["base"][1]) + (b["base"][0],) for b in self.project.config.get("blend") or []]
        ev_picks = {}                                          # {dunk index: picks just after it}
        pick_slots = []                                        # the editor's picks outside the event: a slot each,
        for r in self.recs:                                    # sized for it, where it falls in the day (the event's
            u = r["unit"]                                      # own are placed by in_order)
            if u["category"] in self.water:
                continue
            for m in r["moments"]:
                w = self.when(r, m)
                if not m.get("add") or w is None or (op and op[0] in (u["id"], u["clip"]) and m.get("span")
                                                    and abs(m["span"][0] - P.span_of(op[1])[0]
                                                            - u.get("export_t0", 0.0)) < 0.2):
                    continue
                in_event = bool(self.dk) and (self.when(*self.dk[0]) or w) - 60 <= w <= (self.when(*self.dk[-1]) or w) + 8
                if in_event and (not self.documentary or any(
                        c in (u["id"], u["clip"]) and m.get("span") and abs(m["span"][0] - a - u.get("export_t0", 0.0)) < 0.2
                        for a, _b, c in bases)):
                    continue                                   # (the event's own: in_order places it; a blend's base)
                if (self.project.config.get("documentary") or {}).get("last_pick") in (u["id"], u["clip"]):
                    continue                                   # (the close's own)
                gr = (self.project.config.get("endcard") or {}).get("ground") or []
                if len(gr) >= 3 and gr[0] in (u["id"], u["clip"]) and m.get("span") and \
                        m["span"][0] < float(gr[2]) and m["span"][1] > float(gr[1]):
                    continue                                   # (the end card's own ground: kept for the close)
                n = (m["span"][1] - m["span"][0]) if m.get("span") else 4.0
                pb = next((v for c, v in (self.project.config.get("pick_bars") or {}).items() if c in (u["clip"], u["id"])),
                          None)
                bars = int(pb) if pb else max(1, round(n * 2.0 / bar)) if any(c in (u["clip"], u["id"]) for c in slow) \
                    else (3 if n >= 5.5 else 2)                # (a slowed pick: never under half speed)
                if any(x[3][0] == u["id"] and abs(x[3][1] - m["t"]) < 0.5 for x in pick_slots):
                    continue                                   # (one pick, one slot: a pick can sit in two records)
                if u["category"] in montage:                   # (a montage: each pick about its own length, in beats
                    bars = float(pb) if pb else min(5, max(2, round(n / self.beat))) / 4.0   # -- 2 to 5, so none
                                                                                           # flashes past)
                if in_event:                                   # a pick during the event (the drone's b-roll): its
                    k = max((j for j, (rr, mm) in enumerate(self.dk) if (self.when(rr, mm) or 0) <= w), default=-1)
                    if not any(x[2][0] == u["id"] and abs(x[2][1] - m["t"]) < 0.5 for v in ev_picks.values()
                               for x in v):                     # own slot right after the dunk it follows
                        ev_picks.setdefault(k, []).append((u["category"], bars, [u["id"], round(m["t"], 2)]))
                    continue
                pick_slots.append((u["category"], w, bars, [u["id"], round(m["t"], 2)]))
        for c in montage:                                      # the montage ends on a bar line: its last pick takes
            mine = sorted((x for x in pick_slots if x[0] == c), key=lambda x: x[1])   # the beats left over
            if mine:
                total = sum(x[2] for x in mine)
                pad = (-total) % 1.0
                k = pick_slots.index(mine[-1])
                pick_slots[k] = (*mine[-1][:2], mine[-1][2] + pad, mine[-1][3])
        over = [c for c in self.cams.over(self.water)]
        long_ = {}                                             # reel.json "documentary" "event_s_at": {"C2216@201":
        for key, sec in ((self.project.config.get("documentary") or {}).get("event_s_at") or {}).items():   # 8.0}
            clip, _, at = key.partition("@")                   # -- a moment that takes longer (a baptism by pouring:
            for k, (r_, m_) in enumerate(self.dk):             # three pours)
                if clip in (r_["unit"]["clip"], r_["unit"]["id"]) and (not at or abs(float(at) - m_["t"]) <= 6):
                    long_[k] = float(sec)
        return dict(order=ground, aerial=aerial, starts=starts, over=over, runs=runs_, blends=blends, clips=clips,
                    pick_slots=pick_slots, ends=ends, event_long=long_, event_picks=ev_picks,
                    event=(ev, len(self.dk)) if ev else None, richness=rich,
                    lineups=lineups, title=(shots[0].get("title") if shots else None), card=card,
                    last=self.lastcat)

    # -- setting up: the footage, the day, what each slot is for ----------------------------------------------------
    def _footage(self, moments):
        """The clips this cut may use (this format's, in the cut), the editor's picks applied to their moments."""
        recs = [r for r in moments if (r["moments"] or self.documentary)   # (a documentary also finds moments in
                and (self.fmt == "portrait" or r["orient"] == "landscape")  # a clip's steady stretches)
                and self.project.in_cut(r["unit"])]
        picks = self.project.picks()
        if picks:
            P.apply_picks(recs, picks, only=self.cfg.get("only_picks", True))
        for clip, span, *_why in self.project.config.get("include") or []:   # reel.json "include": a stretch the
            r = next((x for x in recs if clip in (x["unit"]["id"], x["unit"]["clip"])), None)   # cut must show,
            if r is None:                                                    # placed as a pick (its own slot,
                continue                                                     # by when it happened)
            x0 = r["unit"].get("export_t0", 0.0)
            a, b = P.span_of(span)
            r["moments"].append(dict(t=round(x0 + (a + b) / 2, 2), kind="closeup", score=1.0, score_l=1.0, score_p=1.0,
                                     frame_l=1.0, frame_p=1.0, add=True, pick=True,
                                     span=[round(x0 + a, 3), round(x0 + b, 3)]))
        return recs

    def aim_pref(self, u, t, pad=2.0):
        """How much the frames around t show what reel.json "aim" "prefer" asks of the footage (0..1), else 0."""
        a = self.steady.aim
        if not a.ok or u["id"] not in a.pref:
            return 0.0
        tt, v = a.pref[u["id"]]
        m = (tt >= t - pad) & (tt <= t + pad)
        return float(v[m].max()) if m.any() else 0.0

    def _flown_moments(self, every=6.0, floor=0.5):
        """A drone's moments are where it flies: its own few (the moment step looks for people) and, every `every`
        seconds, each reference frame that reads as high above the scene (Looks.high over `floor`) -- a 27-minute
        flight over the event is more than a handful of shots."""
        if self.looks.high is None:
            return
        pats = self.project.config.get("aerial") or ["DJI_*"]
        for r in self.recs:
            u = r["unit"]
            if not any(fnmatch.fnmatch(u["clip"], p) for p in pats) or u.get("picks_only"):
                continue
            k = self.looks.by.get(u["id"])
            if k is None:
                continue
            have = [m["t"] for m in r["moments"]]
            last = -1e9
            for j in k[np.argsort(self.looks.t[k])]:
                t = float(self.looks.t[j])
                if self.looks.high[j] < floor or t - last < every or any(abs(t - h) < every / 2 for h in have):
                    continue
                if not u["t0"] + 2 <= t <= u["t1"] - 2 or any(y - 2 <= t <= z + 2 for y, z in u.get("bans") or []):
                    continue
                r["moments"].append(dict(t=round(t, 2), kind="flight", score=0.5, score_l=0.5, score_p=0.5,
                                         frame_l=1.0, frame_p=1.0, flown=True))
                last = t

    def _clean_moments(self, least=2.0):
        """Every stretch of ground footage clear of camera faults for `least` seconds or more offers a moment at its
        middle, when none of the clip's own moments sits inside it -- the moment step found its best moments by what
        the people do, not by where the camera held, and a walk through a crowd may hold still only between them.
        Scored under the real ones, so they still win where they can play."""
        for r in self.recs:
            u = r["unit"]
            if self.cams.aerial(u["category"]) or u.get("picks_only"):
                continue
            f = self.steady.faults(u)
            if f is None:
                continue
            t, bad, _ = f
            m = (t >= u["t0"]) & (t <= u["t1"])
            tt, bb = t[m], bad[m]
            k = 0
            while k < len(bb):
                if bb[k]:
                    k += 1
                    continue
                j = k
                while j < len(bb) and not bb[j]:
                    j += 1
                a, b = float(tt[k]), float(tt[j - 1])
                banned = any(min(b, z) - max(a, y) > 0 for y, z in u.get("bans") or [])
                if b - a >= least and not banned and not any(a <= x["t"] <= b for x in r["moments"]):
                    r["moments"].append(dict(t=round((a + b) / 2, 2), kind="steady", score=0.35, score_l=0.35,
                                             score_p=0.35, frame_l=1.0, frame_p=1.0, clean=True))
                k = j

    def _spans(self):
        """Each category's stretch of the day: from its first clip's start to its last clip's end (wall clock)."""
        span = {}
        for r in self.recs:
            w = self.wall.get(r["unit"]["id"])
            if w is not None:
                c = span.setdefault(r["unit"]["category"], [w, w + r["unit"]["t1"] - r["unit"]["t0"]])
                c[0], c[1] = min(c[0], w), max(c[1], w + r["unit"]["t1"] - r["unit"]["t0"])
        return span

    def _categories(self):
        """Settle what footage each slot asks for (the title's own, every category reel.json "reference_min" wants
        seen), then each category's run of slots: its first and last."""
        title_on = self.project.config.get("title_on")         # the footage the title sits over (a calm wide)
        if title_on and self.body and any(s.get("title") for s in self.body[:1]):
            self.body[0]["category"] = title_on
        if not self.documentary:                               # (the documentary engine gives every category its
            ensure(self.body, self.project.config.get("reference_min"), self.looks, self.span,   # section)
                   floating=self.floating)
        return runs(self.body)

    def _lineups(self):
        """reel.json "every_unit": {"<category>": n}: every clip of that category on screen once (each speaker at
        the mic, say), side by side -- the category's slots become split screens of up to n columns, as many panels
        in all as there are clips, filled in the order they were shot. {category: [units still to show]}."""
        want = self.project.config.get("every_unit") or {}
        out = {}
        self.grids = {}
        for cat, n in want.items():
            spec = n if isinstance(n, dict) else {"columns": n}
            n = spec.get("columns", 3)
            units = sorted({r["unit"]["id"]: r["unit"] for r in self.recs if r["unit"]["category"] == cat
                            and self.steady.hold.ever(r["unit"])}.values(),     # (a speaker: someone with the mic)
                           key=lambda u: self.wall.get(u["id"]) or 0)
            slots_ = [i for i, b in enumerate(self.body) if b.get("category") == cat and "stutter" not in b
                      and not b.get("title") and i not in self.dunk_at]
            if not units or not slots_:
                continue
            per = [len(units) // len(slots_) + (1 if k < len(units) % len(slots_) else 0) for k in range(len(slots_))]
            per = [min(max(p, 1), int(n)) for p in per]
            if spec.get("grid") and len(units) > 3:              # everyone at once, Brady Bunch style: the
                gi = max(slots_, key=lambda i: self.body[i]["length"])   # longest slot, a grid of every clip
                slots_ = [i for i in slots_ if i != gi]
                self.body[gi]["grid"] = cat
                self.grids[cat] = list(units)
                per = [len(units) // max(len(slots_), 1) + (1 if k < len(units) % max(len(slots_), 1) else 0)
                       for k in range(len(slots_))]
                per = [min(max(p, 1), int(n)) for p in per]
            for i, k in zip(slots_, per):
                self.body[i]["split"] = columns(k) if k > 1 else None
                self.body[i]["lineup"] = cat
            out[cat] = list(units)
        return out

    def split_style(self):
        """The next split screen's entrance (reel.json "split_styles", default slide, stagger, push in turn): no two
        splits in a row arrive the same way."""
        styles = self.project.config.get("split_styles") or ["slide", "stagger", "push"]
        self.n_splits = getattr(self, "n_splits", 0) + 1
        return styles[(self.n_splits - 1) % len(styles)]

    def lineup(self, i, s):
        """The next clips of a lineup in the order they were shot, each at its own best moment for slot i."""
        cat = s["lineup"]
        n = len(s["split"]) if s.get("split") else 1
        queue, got = self.lineups.get(cat, []), []
        everyone = {r["unit"]["id"] for r in self.recs if r["unit"]["category"] == cat}
        while queue and len(got) < n:
            u = queue.pop(0)
            pick = self.portrait(i, s, cat, u, everyone)
            if pick:
                got.append(pick)
        return got

    def portrait(self, i, s, cat, u, everyone, tall=0.55, off=0.3):
        """One clip's panel in a lineup: its best moment where the person nearest the camera fills the frame's
        height (`tall`: a speaker on a long lens, not someone passing) near its middle (`off`) -- else its best
        moment at all."""
        L = s["length"]
        cands = []
        for strict in (True, False):
            cands = self.best(i, dict(s, category=cat), avoid=everyone - {u["id"]}, n=1, strict=strict)
            r = next((x for x in self.recs if x["unit"]["id"] == u["id"]), None)
            if r is None:
                break
            ranked = sorted((m for m in r["moments"] if self.pk.free(r, m) and not m.get("event")),
                            key=lambda m: -self.pk.share(r, m))
            for m in ranked:
                pr = self.steady.presence(u, m["t"] - 0.4 * L, L)
                if pr and pr[0] >= tall and pr[1] <= off and pr[2] >= 0.6 and pr[3] < 0.97 and (not strict or self.steady.ok(
                        u, m["t"] - 0.4 * L, L, m["t"], self.looks) is not None):
                    return r, m
            best_t = None                                      # no moment of it qualifies: the clip scanned every
            for t in np.arange(u["t0"] + 0.4 * L + 0.2, u["t1"] - 0.6 * L - 0.2, 0.5):   # half second instead
                pr = self.steady.presence(u, t - 0.4 * L, L)
                if not pr or pr[0] < tall * 0.8 or pr[1] > off or pr[2] < 0.6 or pr[3] >= 0.97:
                    continue
                if strict and self.steady.ok(u, t - 0.4 * L, L, t, self.looks) is None:
                    continue
                score = pr[0] - pr[1] + 0.5 * pr[2]
                if best_t is None or score > best_t[0]:
                    best_t = (score, t)
            if best_t:
                m = dict(t=round(float(best_t[1]), 2), kind="closeup", score=0.6, score_l=0.6, score_p=0.6,
                         frame_l=1.0, frame_p=1.0, scanned=True)
                return r, m
        return cands[0] if cands else None

    def _dunks(self):
        """Every moment of the reel's event, in the order they happened."""
        from . import events as EV
        EV.configure(self.project.config)
        event = self.project.config.get("reference_event") or (EV.names() or [None])[0]
        return sorted(dunks(self.recs, event), key=lambda x: (self.when(*x) or 0, x[1]["t"]))

    def _assign_dunks(self):
        """The dunks, in order, onto the long water slots (and more slots of the water when there are more dunks):
        {slot: (record, moment)}."""
        body, dk = self.body, self.dk
        marked = [i for i, s in enumerate(body) if s.get("dunk_slot")]   # the documentary engine's own
        if marked and dk:
            return {i: dk[k] for k, i in enumerate(marked[:len(dk)])}
        long_ = [i for i, s in enumerate(body) if s.get("category") in self.water and s["length"] >= DUNK_LEN
                 and not s.get("split") and "stutter" not in s]
        more = [i for i, s in enumerate(body) if s.get("category") in self.water and i not in long_
                and s["length"] >= 1.8 and not s.get("split") and "stutter" not in s]
        want = list(long_)
        first = min(long_ + more, default=None)               # the event opens on its first dunk: the water's first
        if first is not None and first not in want:            # slot that holds one (footage from before it is a
            want.append(first)                                 # minute or so; in_order keeps the rest after it)
        for i in sorted(more, key=lambda i: -body[i]["length"]):
            if len(want) >= len(dk):
                break
            if i not in want:
                want.append(i)
        want.sort()
        if not (dk and want):
            return {}
        if len(dk) <= len(want):                               # each dunk once, spread over the water slots
            pos = np.round(np.linspace(0, len(want) - 1, len(dk))).astype(int) if len(dk) > 1 else [0]
            return {want[p]: dk[k] for k, p in enumerate(pos)}
        pos = np.round(np.linspace(0, len(dk) - 1, len(want))).astype(int)   # more dunks than slots: the best spread
        return {want[k]: dk[p] for k, p in enumerate(pos)}                   # through it

    def _nearest_dunks(self):
        """Each slot's nearest placed dunk (by slot order): the time of day it happened."""
        near, placed = {}, sorted(self.dunk_at)
        for i in range(len(self.body)):
            if placed:
                j = min(placed, key=lambda k: abs(k - i))
                near[i] = self.when(*self.dunk_at[j])
        return near

    def _bad_match(self):
        """The clips of a matched camera whose colour won't match the hero closely enough (reel.json "match_max"):
        they stay out."""
        gate = self.project.config.get("match_max")
        return {v["match"][0] for k, v in (self.ref.get("luts") or {}).items()
                if ":" in k and gate is not None and (v.get("check") or {}).get("after", 0) > float(gate)}

    # -- the time of day --------------------------------------------------------------------------------------------
    def when(self, r, m):
        """The wall-clock time of a moment, or None when its clip's clock isn't known."""
        w = self.wall.get(r["unit"]["id"])
        return None if w is None else w + (m["t"] - r["unit"]["t0"])

    def in_order(self, i, r, m, slack=DUNK_SLACK):
        """Ground footage shot during the event (between the first and the last placed dunk, give or take a minute)
        may fill slot i only if it was shot before the next dunk placed after the slot: no one after their baptism
        before it plays, no later baptism before an earlier one, none of it in the opening. (Earlier footage later
        is a cutaway -- the crowd cheering all morning, at the close -- and free, as is footage from outside the
        event and any when no dunk is placed.)"""
        aerial = self.cams.aerial(r["unit"]["category"])
        if aerial:                                             # (a drone's wide of the water is of everyone at once:
            slack = DUNK_SLACK_AIR                             # it may run a little further ahead)
        w = self.when(r, m)
        if aerial and w is not None and self.dunk_times and \
                self.dunk_times[0][1] - 60.0 <= w <= self.dunk_times[-1][1] + 60.0:
            prev = max((t for k, t in self.dunk_times if k < i), default=None)   # a drone over the event: after the
            if prev is not None and w < prev + 8.0:                              # dunk just shown (never that one
                return False                                                     # again, nor before it)
        if w is None or not self.dunk_times:
            return True
        first, last = self.dunk_times[0][1], self.dunk_times[-1][1]
        if not (first - 60.0 <= w <= last + 60.0):
            return True
        hi = min((t for k, t in self.dunk_times if k > i), default=None)
        if not aerial and not self.documentary:
            return hi is None or w <= hi + slack
        lo = max((t for k, t in self.dunk_times if k < i), default=None)   # a documentary, from the ground too:
        if lo is not None and w < lo - 1.0:                                # never someone before the dunk just
            return False                                                   # shown (nor a recap after the last)
        return hi is None or w <= hi + slack

    def target(self, i, cat):
        """The time of day slot i stands for: its place in its category's run, over that category's span."""
        a, b = self.first.get(cat, i), self.last.get(cat, i)
        f = 0.5 if b == a else (i - a) / (b - a)
        sp = self.span.get(cat)
        return None if sp is None else sp[0] + f * (sp[1] - sp[0])

    # -- the cameras' rules -----------------------------------------------------------------------------------------
    def penalty(self, u):
        """What a weaker camera's shots give away in the scoring (reel.json "camera_penalty")."""
        return next((float(v) for p, v in self.pen.items() if fnmatch.fnmatch(u["clip"], p)), 0.0)

    def capped(self, u):
        """This clip's camera already has its most shots in the reel (reel.json "camera_max")."""
        for p, n in self.cap.items():
            if fnmatch.fnmatch(u["clip"], p):
                return sum(c for clip, c in self.pk.count.items() if fnmatch.fnmatch(clip, p)) >= n
        return False

    def shake_limits(self, u):
        """This clip's camera's shake limits (reel.json "shake_max"), or None."""
        return next((v for p, v in self.shake_max.items() if fnmatch.fnmatch(u["clip"], p)), None)

    def too_shaky(self, u, t, length):
        """A camera with a shake limit (reel.json "shake_max"): this stretch over its second number."""
        lim = self.shake_limits(u)
        if not lim:
            return False
        from . import shake
        j = shake.jitter(self.project, u, t - length / 2, t + length / 2)
        return j is not None and j > lim[1]

    # -- the drone, in step with the ground -------------------------------------------------------------------------
    def on_scene(self, r, t, keep=0.6):
        """Is the drone over the action at t: its frame looks as much like the reference's own aerials of the water
        as the better part (`keep`) of its flight does -- not flying off over the houses."""
        looks = self.looks
        if not self.over_water:
            return True
        k = looks.by.get(r["unit"]["id"])
        if k is None:
            return True
        target = looks.ref[self.over_water].mean(0)
        sims = looks.emb[k] @ target
        near = np.abs(looks.t[k] - t) <= 2.0
        return bool(near.any() and sims[near].max() >= np.quantile(sims, 1 - keep))

    def drone_at(self, w, lead=0.0, search=(0.0, -3.0, 3.0, -6.0, 6.0), length=3.0):
        """The aerial footage at wall-clock time w (+ lead s, or a few seconds either side while the drone was over
        the action): (record, a moment made there), or None."""
        for d in search:
            for r in self.air_units:
                w0 = self.wall.get(r["unit"]["id"])
                if w0 is None:
                    continue
                t = r["unit"]["t0"] + (w + lead + d - w0)
                banned = any(y - 3.0 <= t <= z + 3.0 for y, z in r["unit"].get("bans") or [])   # (the editor's
                if not banned and r["unit"]["t0"] + 1.0 <= t <= r["unit"]["t1"] - 1.0 and self.on_scene(r, t) and \
                        self.steady.aerial_ok(r["unit"], t - 0.4 * length, t + 0.6 * length, self.looks):
                    # (the editor's bans hold; flying for the whole shot, so placing it never slides it off)
                    m = dict(t=round(t, 2), kind="broll", score=0.5, score_l=0.5, score_p=0.5, frame_l=1.0,
                             frame_p=1.0, synced=True)
                    if self.pk.free(r, m):
                        return r, m
        return None

    def alt_dunk(self, i, reach=2):
        """For an aerial slot within `reach` slots of a dunk: the drone over that dunk (its descent), if it flew
        there. A documentary: the drone over the next person to go under, a few seconds before they do -- the lead
        into their dunk, never after it."""
        if self.documentary:
            nxt = min((k for k in self.dunk_at if k > i), default=None)
            if nxt is None:
                return None
            w = self.when(*self.dunk_at[nxt])
            L = self.body[i]["length"] if i < len(self.body) else 4.0
            return self.drone_at(w, -6.0, search=(0.0, -3.0, -6.0), length=L) if w is not None else None
        for j in sorted((k for k in self.dunk_at if abs(k - i) <= reach), key=lambda k: abs(k - i)):
            w = self.when(*self.dunk_at[j])
            got = self.drone_at(w, -1.5) if w is not None else None
            if got:
                return got
        return None

    # -- choosing a slot's footage ----------------------------------------------------------------------------------
    def pool(self, cat):
        """The clips a slot of this category chooses from: that category's; when there are none, any footage from
        around the point in the day the cut has got to."""
        if isinstance(cat, tuple) and cat[0] == "*":           # (the last resort: the slot's kind of camera, from
            now, wall, want = self.cursor, self.wall, cat[1]    # nearest where the cut has got to in the day:
            kind = [r for r in self.recs                        # within 20 minutes, then an hour, then any)
                    if self.cams.aerial(r["unit"]["category"]) == self.cams.aerial(want)] or self.recs
            for back in (1200, 3600):
                near = [r for r in kind if now is None or wall.get(r["unit"]["id"]) is None
                        or (now - back <= wall[r["unit"]["id"]] + r["unit"]["t1"] - r["unit"]["t0"]
                        and wall[r["unit"]["id"]] <= now + 900)]
                if near:
                    return near
            return kind
        pool = [r for r in self.recs if r["unit"]["category"] == cat]
        if not pool:
            now, wall = self.cursor, self.wall
            pool = [r for r in self.recs if now is None or wall.get(r["unit"]["id"]) is None
                    or (wall[r["unit"]["id"]] <= now + 900 and wall[r["unit"]["id"]] + r["unit"]["t1"] - r["unit"]["t0"]
                        >= now - 300)] or self.recs
        return pool

    def usable(self, s, r, m, kinds=None, strict=True, i=None):
        """May this moment fill slot s: still free, of the kind asked for, steady enough over the slot's length, and
        its camera within its rules. An editor's pick ("add") passes the steadiness and camera rules."""
        u = r["unit"]
        if not self.pk.free(r, m) or (kinds and m.get("kind") not in kinds):
            return False
        if m.get("event") and not kinds:
            return False                                       # the event's moments are for their slots
        gr = (self.project.config.get("endcard") or {}).get("ground") or []
        if len(gr) >= 3 and gr[0] in (u["id"], u["clip"]) and float(gr[1]) - s["length"] <= m["t"] <= float(gr[2]):
            return False                                       # (the end card's own ground: kept for the close)
        if self.documentary and not m.get("add") and s.get("category") in \
                ((self.project.config.get("documentary") or {}).get("montage") or []):
            return False                                       # (a montage section: the editor's picks only)
        need = ((self.project.config.get("documentary") or {}).get("faces") or {}).get(s.get("section") or s.get("category"))
        if need and self.documentary and not m.get("add") and \
                self.steady.frontal(u, m["t"] - 0.4 * s["length"], s["length"]) < float(need):
            return False                                       # (people arriving: faces turned to the lens)
        only = (self.project.config.get("section_footage") or {}).get(s.get("section") or s.get("category"))
        if only and not any(fnmatch.fnmatch(u["clip"], q) or u["id"] == q for q in only):
            return False                                       # (reel.json "section_footage": this section's own)
        if self.documentary and (m.get("add") or m.get("fit")):
            return False                                       # (the editor's picks have their own slots)
        if self.documentary and i is not None and self.dunk_times and i > self.dunk_times[-1][0] and \
                u["category"] in self.water:
            return False                                       # (after the last dunk: the celebration, not the water)
        if self.documentary and u["category"] in self.water and not m.get("event") and not m.get("add"):
            return False                                       # (the water from the ground: the dunks themselves)
        if self.documentary and self.cams.aerial(u["category"]) and not m.get("add") and \
                u["category"] not in self.cams.over(self.water) and self.aim_pref(u, m["t"]) < AIR_PREFER:
            return False                                       # (drone b-roll outside the event: the editor's picks,
                                                               # or what reel.json "aim" "prefer" asks for -- boats)
        if self.documentary and self.cams.aerial(u["category"]) and not m.get("add") and self.cursor is not None:
            w = self.when(r, m)                                # (a drone shot from about now in the day -- never
            if w is not None and (w - self.cursor > AIR_WINDOW or w < self.cursor - AIR_BEHIND):   # the dawn
                return False                                   # flight over the baptisms; never going back in time)
        if any(min(m["t"] + 1.0, z) - max(m["t"] - 1.0, y) > 0 for y, z in u.get("bans") or []) and not m.get("add"):
            return False                                       # (never a banned stretch, nor slid out of one)
        if i is not None and not self.in_order(i, r, m):
            return False                                       # (the event plays in the order it happened)
        L_ = s["length"] * (self.cams.pace(u, m["t"], self.speeds, s["length"]) if self.cams.by_move(u) else 1.0)
        L_ += min(0.5, 0.2 * L_)                               # (the slot lands on the beat: room for it to grow)
        hint = self.steady.ok(u, m["t"] - 0.4 * L_, L_, m["t"], self.looks)
        if hint is None and not m.get("add") and (strict or not self.steady.hard_ok(
                u, m["t"] - 0.4 * L_, L_, self.looks)):
            return False                                       # (relaxed: the soft rules; never the hard ones)
        return not (self.too_shaky(u, m["t"], s["length"]) or (
            (self.capped(u) or u["clip"] in self.bad_match) and not m.get("add")))

    def score(self, s, r, m, tgt, aim):
        """How well a moment fits slot s: like the reference's shot, good footage, near the time of day the slot
        stands for (`tgt`), less its camera's penalty; footage shot for its move also for the move the reference made
        (or the moves asked of it), for moving at all, and for being over the dunk at `aim`."""
        u, cat = r["unit"], s.get("category")
        sp = self.span.get(cat)
        v = self.looks.like(s.get("like", s["i"]) if not self.documentary else s.get("like"), u["id"], m["t"])
        w = self.when(r, m)
        ch = 0.0
        if tgt is not None and w is not None and sp and sp[1] > sp[0]:
            ch = -abs(w - tgt) / (sp[1] - sp[0])
        sc = LOOKS * v + (1 - LOOKS) * 0.5 * self.pk.share(r, m) + CHRONO * ch - self.penalty(u)
        if self.documentary and not self.cams.aerial(u["category"]):   # a clip again only when the others are
            sc -= 0.6 * self.pk.count.get(u["clip"], 0)                   # used (card shuffle)
            sc -= 0.3 if m.get("clean") else 0.0                          # (the moment step's own first)
        if self.documentary and not self.cams.aerial(u["category"]):   # people in the middle of the frame, whole
            wc = float((self.project.config.get("documentary") or {}).get("centre_weight", 1.0))
            pr = self.steady.presence(u, m["t"] - 0.4 * s["length"], s["length"]) if wc else None
            if pr:
                sc -= wc * (2.0 * pr[1] + 0.5 * (1.0 - pr[2]))   # off centre (0..0.5), the face unseen
        if self.cams.by_move(u):
            mv = self.cams.move(u, m["t"], s["length"])
            if mv and s.get("move") and mv == s["move"]:       # the move the reference made there
                sc += 0.2
            if mv and mv in (self.want_moves.get(cat) or ()):  # the moves asked of this footage
                sc += 0.3
            sc += 0.15 * self.cams.motion(u, m["t"])
            sc += 0.8 * self.aim_pref(u, m["t"])               # (what the drone should be looking at: aim.py)
            sc -= 0.25 * self.pk.count.get(u["clip"], 0)       # (another flight before the same one again)
            if aim is not None and w is not None:              # the drone over the same dunk
                sc += 0.6 * max(0.0, 1 - abs(w - aim) / 20.0)
        return sc

    def best(self, i, s, kinds=None, n=1, avoid=(), at=None, strict=True):
        """The moments that fit slot i best: like the reference's shot, good, in the order of the day; a drone shot
        of the water near a dunk shows that dunk (the drone at the same moment) and its moving, swooping parts.
        With n > 1 each comes from a different clip."""
        cat = s.get("category")
        pool = self.pool(cat)
        tgt = self.target(i, cat)
        drone = self.cams.aerial(cat)
        aim = at if at is not None else (
            self.near_dunk.get(i) if drone and cat in self.cams.over(self.water) else None)
        out = []
        for r in pool:
            if r["unit"]["id"] in avoid:
                continue
            for m in r["moments"]:
                if self.usable(s, r, m, kinds, strict, i):
                    out.append((self.score(s, r, m, tgt, aim), r, m))
        out.sort(key=lambda x: -x[0])
        picked, units = [], set()
        for _sc, r, m in out:
            if n > 1 and r["unit"]["id"] in units:
                continue
            picked.append((r, m)); units.add(r["unit"]["id"])
            if len(picked) >= n:
                break
        return picked

    def fallback(self, i, s):
        """A plain slot's footage when its category may be used up: the best of its category, else of any from about
        this time of day, else (the last resort) from the hour before the cut's place in the day, its kind of camera
        first; past the soft steadiness rules it is flagged (plan["relaxed"]). The hard rules, and the event's order
        (in_order), hold throughout."""
        s = dict(s, section=s.get("section") or s.get("category"))   # (a fallback stays the section's: its footage)
        air = [c for c in sorted(self.cams.air) if c != s.get("category")] if self.cams.aerial(s.get("category")) else []
        if s.get("category") in self.water:                    # the event's own slot: the drone over it next
            air = list(self.cams.over(self.water))

        got = self.best(i, s)
        for c in air:                                          # another flight (or the drone over the event)
            got = got or self.best(i, dict(s, category=c))
        if self.documentary and s.get("category") in self.cams.over(self.water):
            return got                                         # (a cutaway in the event is the drone, or nothing)
        if self.documentary:                                   # a scene shows its own footage (a drone slot
            return got or self.best(i, s, strict=False)        # the drone's), or nothing (the slot dropped:
                                                               # build()) -- never another scene's, out of its time
        got = (got or self.best(i, dict(s, category=None)) or self.best(i, s, strict=False)
               or self.best(i, dict(s, category=None), strict=False)
               or self.best(i, dict(s, category=("*", s.get("category"))), strict=False))   # (never empty)
        if not got and self.documentary:                       # nothing on the ground: the drone's b-roll
            for c in sorted(self.cams.air):
                got = got or self.best(i, dict(s, category=c), strict=False)
        if got and self.steady.ok(got[0][0]["unit"], got[0][1]["t"] - 0.4 * s["length"], s["length"],
                                  got[0][1]["t"], self.looks) is None:
            self.relaxed.append(s["i"])
        return got

    def angles(self, i, s, n):
        """A split screen in the water: one baptism from every camera that saw it -- the ground camera on the person
        (their dunk when it is still free, else a moment of theirs either side of it) and the drone over them at the
        same second -- then more of the water to fill the panels."""
        pk = self.pk
        if self.documentary:                                   # a documentary shows each dunk once: the angles are
            got = self.reaction_angles(i, s, n)                # on the moment after the one that just played
            if got:
                return got
        tgt = self.target(i, s.get("category"))
        cand = sorted(self.dk, key=lambda x: abs((self.when(*x) or 0) - (tgt or 0)))
        for r, m in cand:
            if not self.in_order(i, r, m):
                continue
            w = self.when(r, m)
            ground = (r, m) if pk.free(r, m) else next(
                ((r, x) for x in sorted(r["moments"], key=lambda x: abs(x["t"] - m["t"]))
                 if abs(x["t"] - m["t"]) <= 10 and pk.free(r, x) and not x.get("event")), None)
            air = self.drone_at(w, -1.0) if w is not None else None
            if ground and air:
                got = [ground, air]
                while len(got) < n:
                    extra = self.best(i, s, avoid={g[0]["unit"]["id"] for g in got})
                    if not extra:
                        break
                    got.append(extra[0])
                return got
        return self.best(i, s, n=n)

    def reaction_angles(self, i, s, n, after=(2.0, 12.0)):
        """The side-by-side after a dunk: the person just baptised -- the hug, the smile, the cheer -- from the ground
        a few seconds after their rise, and the drone over them at the same moment; none of it a dunk again. None
        when the footage isn't there."""
        prev = max((k for k in self.dunk_at if k < i), default=None)
        if prev is None:
            return None
        r, m = self.dunk_at[prev]
        u = r["unit"]
        lo, hi = m["t"] + after[0], m["t"] + after[1]
        cands = [x for x in r["moments"] if lo <= x["t"] <= hi and not x.get("event") and self.pk.free(r, x)]
        cands += [dict(t=round(t, 2), kind="reaction", score=0.5, score_l=0.5, score_p=0.5, frame_l=1.0,
                       frame_p=1.0) for t in np.arange(lo + 1.0, hi, 1.0)]
        L = s["length"]
        ground = next(((r, x) for x in cands if self.steady.camera_ok(u, x["t"] - 0.4 * L, x["t"] + 0.6 * L)
                       and self.pk.free(r, x) and not any(min(x["t"] + 0.6 * L, z) - max(x["t"] - 0.4 * L, y) > 0
                                                         for y, z in u.get("bans") or [])), None)
        if ground is None:
            return None
        w = self.when(*ground)
        air = self.drone_at(w, -1.0) if w is not None else None
        return [ground, air] if air else None

    def panels(self, i, s):
        """A split screen's footage, one moment a panel: the angles on one baptism in the water, else the best of the
        slot's category from different clips -- filled out from any footage when that runs short."""
        n = len(s["split"])
        got = self.angles(i, s, n) if s.get("category") in self.water else self.best(i, s, n=n)
        if len(got) < n:
            got += self.best(i, dict(s, category=None), n=n - len(got), avoid={g[0]["unit"]["id"] for g in got})
        return got

    # -- what is spoken for before the slots are filled -------------------------------------------------------------
    def reserve_picks(self):
        """The editor's picks ("add"): each its own slot, in its part of the day -- a plain slot of its category,
        else one whose category was shot around then, else any: {slot: (record, moment)}."""
        body, reserved = self.body, {}
        lp = (self.project.config.get("documentary") or {}).get("last_pick") if self.documentary else None
        adds = [(rr, mm) for rr in self.recs for mm in rr["moments"] if mm.get("add")
                and not (lp and lp in (rr["unit"]["id"], rr["unit"]["clip"]))]   # (the close's pick is the close's)
        gr = (self.project.config.get("endcard") or {}).get("ground") or []
        if len(gr) >= 3:                                        # (the end card's own range: the close's)
            adds = [(rr, mm) for rr, mm in adds if not (gr[0] in (rr["unit"]["id"], rr["unit"]["clip"]) and mm.get("span")
                                                        and mm["span"][0] < float(gr[2]) and mm["span"][1] > float(gr[1]))]
        bases = [P.span_of(b["base"][1]) + (b["base"][0],) for b in self.project.config.get("blend") or []]
        adds = [(rr, mm) for rr, mm in adds if not any(c in (rr["unit"]["id"], rr["unit"]["clip"]) and mm.get("span")
                                                       and abs(mm["span"][0] - a - rr["unit"].get("export_t0", 0.0)) < 0.2
                                                       for a, _b, c in bases)]   # (those are the blends' own)
        op = (self.project.config.get("documentary") or {}).get("open") if self.documentary else None
        if op and body:                                         # the opening title over the editor's pick, whole
            a, b = P.span_of(op[1])                             # (sped up to fit: reel.json "documentary" "open")
            hit = next(((rr, mm) for rr, mm in adds if op[0] in (rr["unit"]["id"], rr["unit"]["clip"])
                        and mm.get("span") and abs(mm["span"][0] - a - rr["unit"].get("export_t0", 0.0)) < 0.2), None)
            if hit:
                reserved[0] = hit
                hit[1]["fit"] = True
                adds = [x for x in adds if not (x[0]["unit"]["id"] == hit[0]["unit"]["id"]   # (each copy of it)
                                                and x[1].get("span") == hit[1].get("span"))]
        for k, b in enumerate(body):                            # each pick into the slot made for it
            if b.get("pick_for"):
                hit = next(((rr, mm) for rr, mm in adds if rr["unit"]["id"] == b["pick_for"][0]
                            and abs(mm["t"] - b["pick_for"][1]) < 0.5), None)
                if hit:
                    reserved[k] = hit
                    adds = [x for x in adds if not (x[0]["unit"]["id"] == hit[0]["unit"]["id"]
                                                    and abs(x[1]["t"] - hit[1]["t"]) < 0.5)]
        for r, m in sorted(adds, key=lambda x: self.when(*x) or 0):
            w = self.when(r, m)
            cat = r["unit"]["category"]
            free_slots = [i for i, b in enumerate(body) if i not in reserved and i not in self.dunk_at
                          and not b.get("split") and "stutter" not in b and not b.get("title") and i != len(body) - 1]
            same = [i for i in free_slots if body[i].get("category") == cat]
            pool_ = same or [i for i in free_slots if body[i].get("category") not in self.floating
                             and (self.span.get(body[i].get("category")) or [0, 0])[0] - 600 <= (w or 0)
                             <= (self.span.get(body[i].get("category")) or [0, 0])[1] + 600] or free_slots
            if pool_ and self.documentary and self.dunk_times and w is not None and \
                    self.dunk_times[0][1] - 60 <= w <= self.dunk_times[-1][1] + 60:
                lead = [i for i in free_slots if self.in_order(i, r, m, slack=4.0)   # (the event: just before the
                        and all(t >= w - 4.0 for k, t in self.dunk_times if k > i)]   # dunk it leads into)
                nxt = min((k for k, t in self.dunk_times if t >= w - 4.0), default=None)
                lead = [i for i in lead if nxt is None or i < nxt]
                if not lead:                                  # no slot between the dunks around it: the one
                    def viol(i, w=w):                         # least out of order
                        after = [t for k, t in self.dunk_times if k > i]
                        before = [t for k, t in self.dunk_times if k < i]
                        return max(0.0, w - min(after, default=w)) + max(0.0, max(before, default=w) - w)
                    ev = [i for i in free_slots if self.dunk_times[0][0] < i < self.dunk_times[-1][0]]
                    lead = [min(ev, key=viol)] if ev else []
                if lead:
                    reserved[max(lead)] = (r, m)
                    continue
            if pool_:
                i = min(pool_, key=lambda i: abs((self.target(i, body[i].get("category")) or 0) - (w or 0)))
                reserved[i] = (r, m)
        return reserved

    def reserve_last(self):
        """The closing shot (reel.json "reference_last") first, before others take it: (record, moment), or None."""
        if not (self.lastcat and self.body):
            return None
        lp = (self.project.config.get("documentary") or {}).get("last_pick") if self.documentary else None
        if lp:                                                 # the editor's pick closes the film (reel.json
            hit = next(((r, m) for r in self.recs if lp in (r["unit"]["id"], r["unit"]["clip"])   # "documentary"
                        for m in r["moments"] if m.get("add")), None)                          # "last_pick")
            if hit:
                self.pk.take(*hit, beats=self.beats_per[len(self.body) - 1], act=self.lastcat)
                return hit
        got = self.best(len(self.body) - 1, self.body[-1]) or \
            self.best(len(self.body) - 1, self.body[-1], strict=False)   # (the soft rules give before the close does)
        if not got:
            return None
        self.pk.take(*got[0], beats=self.beats_per[len(self.body) - 1], act=self.lastcat)
        return got[0]

    # -- filling the slots ------------------------------------------------------------------------------------------
    def shot(self, i, s, nb, reserved, last_pick):
        """Slot i's shot (nb beats long), or None when nothing fits it: the editor's pick from its own in-point, the
        closing shot, a dunk (with its velocity edit), a split screen, the drone over the dunk beside it, else the
        best footage there is for it."""
        pk, cams = self.pk, self.cams
        act = s.get("category") or "shots"
        if s.get("photos"):                                    # reel.json "documentary" "photo_grid": the editor's
            pg = (self.project.config.get("documentary") or {}).get("photo_grid") or {}   # photos, each a spatial
            return dict(unit=dict(id="photos", clip="photos", category="photos", source="", t0=0.0, t1=1e9,   # scene,
                                  orient="landscape"), t=0.0, kind="photo", rank=0.0, face=0, framing=1.0,      # in a
                        fx=[], effect=None, beats=nb, act=act,                                                  # grid
                        photos=dict(layers=list(pg.get("photos") or []), style=pg.get("style", "stagger")))
        if s.get("breath"):                                    # reel.json "documentary" "breath": {"clip", "from",
            br = (self.project.config.get("documentary") or {}).get("breath") or {}   # "to"} -- that stretch, whole
            r = next((x for x in self.recs if br.get("clip") in (x["unit"]["id"], x["unit"]["clip"])), None)
            if r is None:
                self.skipped.append(s["i"])
                return None
            a, b = float(br["from"]), float(br["to"])
            shot = pk.take(r, dict(t=round((a + b) / 2, 2), kind="broll", score=1.0, score_l=1.0, score_p=1.0,
                                   frame_l=1.0, frame_p=1.0, span=[a, b]), beats=nb, act=act)
            shot.update(span=[a, b], fit=True, breath=True, fx=["fit"], effect="fit")
            return shot
        if i in reserved:
            shot = pk.take(*reserved[i], beats=nb, act=act)
            shot["pick"] = True
            u_ = reserved[i][0]["unit"]
            if any(c in (u_["clip"], u_["id"]) for c in self.project.config.get("slow_picks") or []):
                shot["fx"], shot["effect"] = ["fit"], "fit"   # (reel.json "slow_picks": played whole, slowed to fit)
                sp = None if any(c in (u_["clip"], u_["id"]) for c in self.project.config.get("pick_bars") or {}) \
                    else reserved[i][1].get("span")           # (reel.json "pick_bars": the editor's length for it)
                if sp:                                         # (never slower than half speed: as many bars as
                    most = max(4, int((sp[1] - sp[0]) * 2.0 / self.beat / 4 + 0.5) * 4)   # that takes, the rest
                    if nb > most:                                                       # to the next shot)
                        shot["beats"], shot["_spare"] = most, nb - most
            if any(c in (u_["clip"], u_["id"]) for c in self.project.config.get("nat_picks") or []):
                shot["nat"] = True                             # (reel.json "nat_picks": its own sound under it)
            if reserved[i][1].get("fit"):
                shot["fit"] = True
        elif i == len(self.body) - 1 and last_pick:
            r, m = last_pick                                   # taken already (reserve_last): cut afresh for this slot
            shot = dict(P.Picker(self.fmt, [r]).take(r, m, beats=nb, act=act), last=True)
            u_ = r["unit"]
            if m.get("add"):
                shot["pick"] = True
                if any(c in (u_["clip"], u_["id"]) for c in self.project.config.get("slow_picks") or []):
                    shot["fx"], shot["effect"] = ["fit"], "fit"
                if any(c in (u_["clip"], u_["id"]) for c in self.project.config.get("nat_picks") or []):
                    shot["nat"] = True
        elif s.get("blend") is not None and (bs := self.blend_shot(i, s, nb, act)):
            shot = bs
        elif i in self.dunk_at:
            shot = pk.take(*self.dunk_at[i], beats=nb, act=act)
            shot["dunk"] = True
            r_, m_ = self.dunk_at[i]
            ramp = self.ramp
            for k, v in (self.project.config.get("event_ramp") or {}).items():   # reel.json "event_ramp":
                clip, _, at = k.partition("@")                                    # {"C2216@200": "none"} -- a
                if clip in (r_["unit"]["clip"], r_["unit"]["id"]) and (not at or abs(float(at) - m_["t"]) <= 6):
                    ramp = None if v in (None, "none", "") else v                 # baptism by pouring, at real speed
            w = self.when(r_, m_)                              # (the same second exactly: no lead, no search --
            air = self.drone_at(w, 0.0, search=(0.0,), length=nb * self.beat) \
                if s.get("angles") and w is not None else None  # else the dunk plays full frame)
            if air:                                            # this dunk from the ground and the drone at once
                ar, am = air
                pk.take(ar, am, beats=nb, act=act)
                shot["split"] = dict(rects=[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0]],
                                     panels=[dict(unit=r_["unit"], t=m_["t"]), dict(unit=ar["unit"], t=am["t"])],
                                     style=self.split_style())
            elif ramp:
                shot["fx"], shot["effect"] = [ramp], ramp
        elif s.get("grid"):                                   # everyone of a category on screen at once
            cat, units = s["grid"], self.grids.get(s["grid"], [])
            everyone = {u["id"] for u in units}
            got, own = [], self.pk                             # everyone, even those with a panel of their own:
            self.pk = P.Picker(self.fmt, self.recs)            # a picker of the grid's own
            try:
                for u in units:
                    pick = self.portrait(i, s, cat, u, everyone)
                    if pick:
                        pick = self.best_cell(pick, nb * self.beat)
                        got.append(pick)
                        self.pk.take(*pick)
                for clip, t, *at in (self.project.config.get("grid_extra") or {}).get(cat, []):   # reel.json
                    r = next((x for x in self.recs if clip in (x["unit"]["id"], x["unit"]["clip"])), None)   # someone
                    if r is not None:                          # else too: [clip, t] or [clip, t, x, y] -- the cell
                        got.append((r, dict(t=float(t), kind="closeup", score=0.6, score_l=0.6, score_p=0.6,   # framed
                                            frame_l=1.0, frame_p=1.0, extra=True,                        # on x, y
                                            **({"focus": [float(at[0]), float(at[1])]} if len(at) >= 2 else {}),
                                            **({"zoom": float(at[2])} if len(at) >= 3 else {}))))
            finally:
                self.pk = own
            if len(got) < 2:
                got = self.fallback(i, s)
                if not got:
                    self.skipped.append(s["i"])
                    return None
                shot = pk.take(*got[0], beats=nb, act=act)
            else:
                shot = dict(P.Picker(self.fmt, self.recs).take(*got[0], beats=nb, act=act))
                cells, logo = grid(len(got), self.project.config.get("logo"))
                shot["split"] = dict(rects=cells, panels=[dict(unit=r["unit"], t=m["t"], **({"focus_at": m["focus"]}
                                                                         if m.get("focus") else {}),
                                                               **({"zoom": m["zoom"]} if m.get("zoom") else {}))
                                                          for r, m in got], style="cascade", logo=logo)
        elif s.get("lineup"):                                 # everyone of a category, side by side
            got = self.lineup(i, s) or self.fallback(i, s)
            got = [self.best_cell(g, nb * self.beat) if g[0]["unit"]["category"] == s.get("category") else g
                   for g in (got or [])]                       # (each speaker's best stretch: no bloom, no passer-by)
            if not got:
                self.skipped.append(s["i"])
                return None
            shot = pk.take(*got[0], beats=nb, act=act)
            for r, m in got[1:]:
                pk.take(r, m, beats=nb, act=act)
            if len(got) > 1:                                   # (after another lineup or the grid: the panels
                prev_split = i > 0 and (self.body[i - 1].get("lineup") or self.body[i - 1].get("grid"))   # swap in
                shot["split"] = dict(rects=columns(len(got)), panels=[dict(unit=r["unit"], t=m["t"]) for r, m in got],
                                     style="none" if prev_split else self.split_style())   # place, no rebuild)
        elif s.get("split"):
            got = self.panels(i, s) or self.fallback(i, s)
            if not got:
                self.skipped.append(s["i"])
                return None
            shot = pk.take(*got[0], beats=nb, act=act)
            for r, m in got[1:]:
                pk.take(r, m, beats=nb, act=act)
            rects = s["split"]
            cams_ = {Cameras.camera(self.project, r["unit"]) for r, _m in got}
            if len(cams_) > 1:                                 # one moment from several cameras: side by side,
                rects = columns(len(got))                      # never stacked (reel.json "angles_layout": "as_reference")
                if self.project.config.get("angles_layout") == "as_reference":
                    rects = s["split"]
            shot["split"] = dict(rects=rects, panels=[dict(unit=r["unit"], t=m["t"]) for r, m in got],
                                 style=self.split_style())
        elif s.get("category") in cams.over(self.water) and (synced := self.alt_dunk(i)):
            shot = pk.take(*synced, beats=nb, act=act)
            shot["synced"] = True
        else:
            got = self.fallback(i, s)
            if not got:
                self.skipped.append(s["i"])
                return None
            shot = pk.take(*got[0], beats=nb, act=act)
            if "stutter" in s:
                shot["stutter"] = max(3, s.get("pieces", 3) + 1)
            elif cams.by_move(shot["unit"]):                   # footage shot for its move: a held horizon runs as a
                shot["speed"] = cams.pace(shot["unit"], shot["t"], self.speeds, nb * self.beat)   # time-lapse
                shot["move"] = cams.move(shot["unit"], shot["t"], nb * self.beat * shot["speed"])
        return shot

    def best_cell(self, pick, length, reach=6.0):
        """A grid cell's stretch: of the portrait's moment and its neighbours (within `reach` s), the one that plays
        best small -- the mic held, the camera clear of faults, the picture's level steady (no exposure stepping
        through it: an ISO change, the sun out), nobody crossing in front of the speaker (a passer-by's box over
        theirs, taller -- nearer the lens)."""
        r, m = pick
        u = r["unit"]
        st = self.steady
        d = st._score(u)

        def score(a):
            b = a + length
            q = -3.0 * (1.0 - st.hold.held(u, a, b)) - (0.0 if st.camera_ok(u, a, b) else 1.0)
            if d is not None:
                k = (d["t"] >= a) & (d["t"] <= b)
                if k.any():
                    lu = d["luma"][k].astype(float)
                    lu = lu[lu > 0]                                  # (samples the score step didn't measure)
                    if len(lu) > 2:
                        q -= 6.0 * float(lu.max() - lu.min()) / max(float(np.median(lu)), 1.0)   # exposure stepping
                pk_ = (d["pose_t"] >= a) & (d["pose_t"] <= b)
                hb = st.hold.at(u, a, b)
                if hb is not None and pk_.any():
                    cross = 0
                    for bx in d["boxes"][pk_]:
                        for x in bx[~np.isnan(bx[:, 0])]:
                            x0, x1 = x[0] / d["aw"], x[2] / d["aw"]
                            if (x[3] - x[1]) / d["ah"] > (hb[3] - hb[1]) * 1.15 and min(x1, hb[2]) - max(x0, hb[0]) > 0.3 * (hb[2] - hb[0]):
                                cross += 1                          # someone nearer, over the speaker
                                break
                    q -= 3.0 * cross / pk_.sum()
            for a0, b0 in used.get(u["id"], []):                     # (a speaker in the grid and again in a
                if min(b, b0) - max(a, a0) > -1.0:                     # lineup: a different stretch of them each
                    q -= 4.0                                           # time, not the same few seconds twice)
            return q
        used = self.__dict__.setdefault("cells_used", {})
        t0 = m["t"] - 0.4 * length
        cands = [t0 + dd for dd in np.arange(-reach, reach + 0.01, 0.5) if u["t0"] <= t0 + dd and t0 + dd + length <= u["t1"]]
        if not cands:
            return pick
        best = max(cands, key=score)
        used.setdefault(u["id"], []).append((best, best + length))
        return r, dict(m, t=round(best + 0.4 * length, 2))

    def blend_shot(self, i, s, nb, act):
        """A double exposure (reel.json "blend"): the base -- the editor's pick of the drone over a dunk -- slowed,
        and over it, dissolving in and out, the person about to go under in that dunk (the next dunk slot's), close.
        The shot, or None when its footage isn't there."""
        cfg = (self.project.config.get("blend") or [])[s["blend"]]
        r = next((x for x in self.recs if cfg["base"][0] in (x["unit"]["id"], x["unit"]["clip"])), None)
        nxt = next((k for k in sorted(self.dunk_at) if k > i), None)
        if r is None or nxt is None:
            return None
        a, b = P.span_of(cfg["base"][1])
        x0 = r["unit"].get("export_t0", 0.0)
        m = next((mm for mm in r["moments"] if mm.get("span") and abs(mm["span"][0] - a - x0) < 0.2), None) or \
            dict(t=round(x0 + (a + b) / 2, 2), kind="flight", score=0.5, span=[x0 + a, x0 + b])
        shot = P.Picker(self.fmt, [r]).take(r, m, beats=nb, act=act)
        dr, dm = self.dunk_at[nxt]
        shot["blend"] = dict(speed=float(cfg.get("speed", 0.5)), opacity=float(cfg.get("opacity", 0.55)),
                             unit=dr["unit"], t=dm["t"])
        shot["span"] = [x0 + a, x0 + b]
        return shot

    def fill(self, reserved, last_pick):
        """Every slot's shot, in the reel's order, each noting the reference shot it stands in for (and its title);
        the cut's place in the day moves on with each shot that has one."""
        seq, carry = [], 0
        for i, (s, nb) in enumerate(zip(self.body, self.beats_per)):
            shot = self.shot(i, s, nb, reserved, last_pick)
            if shot is None:
                if self.documentary:                           # (the documentary's bars are the song's: the shot
                    prev = seq[-1] if seq else None            # before holds through a slot nothing could fill --
                    if prev is not None and not (prev.get("pick") or prev.get("fx") or prev.get("dunk")
                                                 or prev.get("split") or prev.get("blend")):
                        prev["beats"] += nb                    # unless it is a pick, an effect, a dunk or a split:
                    else:                                      # then the next shot takes them)
                        carry += nb
                continue
            if carry:
                shot["beats"] += carry
                carry = 0
            carry += shot.pop("_spare", 0)
            w = self.wall.get(shot["unit"]["id"])
            if w is not None and shot["unit"]["category"] not in self.floating:
                self.cursor = w + shot["t"] - shot["unit"]["t0"]
            shot["ref"] = dict(i=s["i"], t0=s["t0"], length=s["length"], scale=s.get("scale"),
                               category=s.get("category"))
            if s.get("title") and self.project.config.get("title"):
                shot["title"] = dict(s["title"])
            seq.append(shot)
        if carry and seq:                                      # (bars still owed at the very end: the last shot)
            seq[-1]["beats"] += carry
        return seq

    def recap_in_order(self, seq):
        """After the last dunk, the event's ground footage (a look back at the people in the water) plays in the order
        it happened: those shots' footage dealt back onto their own slots by the time it was shot -- each slot keeps
        its length, beats, reference and title. Dunks, split screens, stutters, picks and the last shot stay put."""
        if not self.dunk_times:
            return
        first, last = self.dunk_times[0][1], self.dunk_times[-1][1]
        end = max((k for k, s in enumerate(seq) if s.get("dunk")), default=None)
        if end is None:
            return
        keep = ("beats", "ref", "title", "last")
        idx, items = [], []
        for k in range(end + 1, len(seq)):
            s = seq[k]
            if s.get("dunk") or s.get("split") or s.get("stutter") or s.get("pick") or s.get("last") \
                    or self.cams.aerial(s["unit"]["category"]):
                continue
            w = self.wall.get(s["unit"]["id"])
            if w is None:
                continue
            w += s["t"] - s["unit"]["t0"]
            if first - 60.0 <= w <= last + 60.0:
                idx.append(k)
                items.append((w, s))
        order = [s for _w, s in sorted(items, key=lambda x: x[0])]
        slots = [{f: seq[k].get(f) for f in keep} for k in idx]
        for k, body, slot in zip(idx, order, slots, strict=True):
            new = {f: v for f, v in body.items() if f not in keep}
            new.update({f: v for f, v in slot.items() if v is not None})
            seq[k] = new

    def sections_in_order(self, seq):
        """A documentary's sections each run in the order things happened: within every run of shots cast from one
        ground category (the arrival, the briefing), the footage dealt back onto the run's own slots by when it was
        shot -- each slot keeps its length, beats, reference and title. Dunks, split screens, picks, the event's own
        footage (its order is in_order's) and the last shot stay put."""
        keep = ("beats", "ref", "title", "last")
        k = 0
        def cat(x):                                            # (the section a shot was cast for)
            return (x.get("ref") or {}).get("category") or x["unit"]["category"]
        while k < len(seq):
            c = cat(seq[k])
            j = k
            while j < len(seq) and cat(seq[j]) == c:
                j += 1
            idx = [x for x in range(k, j) if not (seq[x].get("dunk") or seq[x].get("split") or seq[x].get("pick")
                                                  or seq[x].get("last")) and self.wall.get(seq[x]["unit"]["id"])
                   is not None and seq[x]["unit"]["category"] == c]
            if c not in self.water and not self.cams.aerial(c) and len(idx) > 1:
                when = {x: self.wall[seq[x]["unit"]["id"]] + seq[x]["t"] - seq[x]["unit"]["t0"] for x in idx}
                order = [seq[x] for x in sorted(idx, key=lambda x: when[x])]
                slots_ = [{f: seq[x].get(f) for f in keep} for x in idx]
                lengths = sorted((sl["beats"] for sl in slots_), reverse=True)   # (any order of a section's bar
                for x, body, slot in zip(idx, order, slots_, strict=True):          # lengths plays: each shot, in
                    L = next((b for b in lengths if self.steady.hard_ok(              # time order, takes the longest
                        body["unit"], body["t"] - 0.4 * b * self.beat, b * self.beat, self.looks)), lengths[-1])
                    lengths.remove(L)                                                # left that it holds cleanly)
                    new = {f: v for f, v in body.items() if f not in keep}
                    new.update({f: v for f, v in slot.items() if v is not None})
                    new["beats"] = L
                    seq[x] = new
            k = j

    def plan(self):
        """The reel's plan: the slots filled, laid on the music, the end card, the crowd's sound, and what to report."""
        project = self.project
        reserved = self.reserve_picks()
        seq = self.fill(reserved, self.reserve_last())
        self.recap_in_order(seq)
        if self.documentary:
            self.sections_in_order(seq)
        if self.documentary:
            from . import documentary as D
            d = D.settings(project.config)
            plan = place(project, self.fmt, seq, self.music, self.cfg, self.beat, self.steady, self.looks,
                         lead=d["lead"], downbeat=self.music["beats"][self.grid_info["start"]])
        else:
            plan = place(project, self.fmt, seq, self.music, self.cfg, self.beat, self.steady, self.looks)
        if self.card:
            length = self.beats_per[-1] * self.beat
            short = False
            if self.documentary:                               # (to the song's last note: the grid's beats run
                end = self.music["start"] + self.music["length"]   # a little off the median)
                rest = end - (plan["music_start"] + plan["video_length"])
                short = rest > CARD_MAX                    # (a film shortened: the card its own length, the
                length = length if short else max(rest, length * 0.5)   # song fading out over it)
            card_at = plan["video_length"]
            plan = add_card(project, self.fmt, plan, self.card, length, cams=self.cams)
            if short:
                plan["stinger_at"] = round(card_at, 3)
        plan["nat"] = nat_sound(project, plan) if project.config.get("nat", True) else []
        plan["mode"] = "documentary" if self.documentary else "reference"
        if self.documentary:
            from . import documentary as D
            plan["timing"] = D.check(plan, self.music, project.config)
        plan["relaxed"], plan["skipped"] = self.relaxed, self.skipped
        plan["units_total"] = len({r["unit"]["id"] for r in self.recs})
        plan["dunks"] = dict(found=len(self.dk), placed=len(self.dunk_at))
        return plan


def build(project, fmt, music, moments, rounds=4):
    """The plan for one format (see Cut). A documentary whose structure left slots nothing could fill (no footage
    that keeps the rules, from the right time) is built again with those bars given to the section that has footage
    to spare (reel.json "documentary" "grow", default the first brisk section: more arrivals), until none is left
    empty -- the film stays as long as its song, nothing held past its length."""
    import copy as _copy
    cut = Cut(project, fmt, music, _copy.deepcopy(moments))   # (a cut adds moments to the footage it is given)
    plan = cut.plan()
    if not cut.documentary:
        return plan
    d = project.config.setdefault("documentary", {})
    grow = d["grow"] if "grow" in d else next(iter(d.get("brisk") or []), None)   # (null: none -- the film shorter)
    for _ in range(rounds):
        empty = [b for b in cut.body if b["i"] in set(cut.skipped)]
        lost = sum(b.get("bars", 0) for b in empty)
        if not lost:
            break
        sizes = dict(d.get("section_bars") or {})
        drop = list(d.get("drop") or [])
        for b in empty:                                        # a section's empty shot: the section smaller; any
            kind, cat = (b.get("seqkey") or "::").split(":")[:2]   # other slot: left out of the structure
            if kind == "section":
                have = sum(x.get("bars", 0) for x in cut.body if x.get("seqkey", "").startswith(f"section:{cat}:"))
                sizes[cat] = max(int(sizes.get(cat, have)) - b.get("bars", 0), 0)
            elif b.get("seqkey"):
                drop.append(b["seqkey"])
        if grow:                                               # (no section to grow: the film is that much shorter)
            have = sum(b.get("bars", 0) for b in cut.body if b.get("seqkey", "").startswith(f"section:{grow}:"))
            sizes[grow] = int(sizes.get(grow, have)) + lost
        d["section_bars"], d["drop"] = sizes, drop
        print(f"[reel]   {len(cut.skipped)} slot(s), {lost} bar(s) with nothing to play: "
              f"{'to ' + grow if grow else 'left out, the film shorter'}", flush=True)
        cut = Cut(project, fmt, music, _copy.deepcopy(moments))
        plan = cut.plan()
    return plan


def grid(n, logo=None):
    """A grid for n panels on a 16:9 frame -- as many columns as keeps each cell nearest square-ish and rows full
    (14 -> 5 x 3): [x, y, w, h] per panel, in reading order, and the spare cell (the middle of the last row when
    there is one) for the logo: ([cells], {"rect": cell, "image": logo} or None)."""
    best = None
    for cols in range(2, 9):
        rows = -(-n // cols)
        spare = cols * rows - n
        cell = (16 / cols) / (9 / rows)                       # the cell's aspect
        cost = abs(np.log(cell / 1.1)) + 0.6 * spare
        if best is None or cost < best[0]:
            best = (cost, cols, rows)
    _, cols, rows = best
    cells = [[round(c / cols, 4), round(r / rows, 4), round(1 / cols, 4), round(1 / rows, 4)]
             for r in range(rows) for c in range(cols)]
    spare = cells[n:]
    keep = cells[:n]
    sp = None
    if spare and logo:                                         # the logo in the bottom row's middle: the panels
        mid = (rows - 1, cols // 2)                            # there shift round it
        m = [round(mid[1] / cols, 4), round(mid[0] / rows, 4), round(1 / cols, 4), round(1 / rows, 4)]
        keep = [c for c in cells if c != m][:n]
        sp = {"rect": m, "image": logo}
    return keep, sp


def columns(n):
    """A split screen of n panels side by side: [x, y, w, h] each, as fractions of the frame."""
    return [[round(c / n, 4), 0.0, round(1 / n, 4), 1.0] for c in range(n)]


class Cameras:
    """What each category of footage is: aerial (a drone: every clip DJI_* or reel.json "aerial" patterns), and how
    much its camera moves around a moment (the score's camera motion, against the clip's own range)."""
    def __init__(self, project, recs):
        self.project = project
        pats = project.config.get("aerial") or ["DJI_*"]
        self.air = {r["unit"]["category"] for r in recs if any(fnmatch.fnmatch(r["unit"]["clip"], p) for p in pats)}
        self.cats = {r["unit"]["category"] for r in recs}
        self.cache = {}

    def aerial(self, cat):
        return cat in self.air

    @staticmethod
    def camera(project, u):
        """Which camera shot a unit: its reel.json "cameras" name, else its clip name's letters."""
        import re
        for name, pats in (project.config.get("cameras") or {}).items():
            if any(fnmatch.fnmatch(u["clip"], p) for p in pats):
                return name
        m = re.match(r"[A-Za-z]+", u["clip"])
        return m.group(0) if m else u["clip"]

    def by_move(self, u):
        """Is this footage chosen for the camera's move (a drone's dive, a gimbal's orbit) or for what the people in
        it are doing? reel.json "shot_by": {"<clip pattern or category>": "move" | "subject"}; aerial footage by its
        move, the rest by its subject, unless said otherwise."""
        for pat, how in (self.project.config.get("shot_by") or {}).items():
            if u["category"] == pat or fnmatch.fnmatch(u["clip"], pat):
                return how == "move"
        return u["category"] in self.air

    def over(self, water):
        """The aerial categories that fly over the water categories' time (named like them: drone-baptisms)."""
        return [c for c in sorted(self.air) if any(w in c for w in water)]

    def _cam(self, u):
        if u["score"] not in self.cache:
            p = self.project.path("scores", u["score"] + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                self.cache[u["score"]] = (z["t"], z["cam"], float(np.percentile(z["cam"], 90)) or 1.0)
            else:
                self.cache[u["score"]] = None
        return self.cache[u["score"]]

    def motion(self, u, t, half=1.5):
        c = self._cam(u)
        if c is None:
            return 0.0
        tt, cam, top = c
        m = (tt >= t - half) & (tt <= t + half)
        return float(np.clip(cam[m].mean() / top, 0, 1)) if m.any() else 0.0

    def move(self, u, t, length=3.0):
        """The camera's move around a moment (moves.py), or None unmeasured."""
        from . import moves
        got = moves.at(self.project, u, t - length / 2, t + length / 2)
        return got[0] if got else None

    def pace(self, u, t, speeds, length=3.0):
        """A drone shot's speed: a held shot (a horizon, a hover) at the first -- a time-lapse -- a move at the
        second. Measured over the stretch the sped-up shot will read."""
        held, moving = speeds
        spans = (self.project.config.get("timelapse_only") or {}).get(u["clip"]) or \
            (self.project.config.get("timelapse_only") or {}).get(u["id"])
        x0 = u.get("export_t0", 0.0)
        if spans is not None and not any(a + x0 <= t <= b + x0 for a, b in (P.span_of(x) for x in spans)):
            return moving                                    # (reel.json "timelapse_only": sped up only there)
        mv = self.move(u, t, length * held)
        if mv is None:
            return held if self.motion(u, t, 2.5) < 0.25 else moving
        return held if mv in ("held", "drift") else moving


class Steady:
    """Whether a stretch of footage plays well start to finish -- the rules a picked shot must keep all the way
    through, not just at its moment:

      camera   a hard rule, never relaxed: every 0.1 s of the stretch clear of the clip's camera faults (faults(),
               each padded by `pad`): on a tripod (reel.json "tripod": clip patterns or categories) any pan, tilt or
               zoom (the operator finding focus and framing); handheld or gimbal a swing beyond `swing`, the camera
               lowered or raised (`drop`: the end of a take, the camera dropping to the ground), a twist (`twist`),
               shake (`jitter`: the path's wobble about its own 1 s average); for all, frames the tracker lost (the
               picture smeared by a whip), the picture going soft (`blur` of the clip's median sharpness), and frames
               of nothing (aim.py: the ground, feet, the sky, a hand on the lens; `aim` max)
      subject  the people stay in the frame's middle and inside it (frame.track's union of the subject over the
               stretch: its centre within `centre` of the middle, its edges off the frame's edges)
      aerial   the drone flying (reference_frames "high": a frame that reads as high above the beach and sea, not
               low over the sand while it takes off or is caught), clear of each flight's first and last seconds

    ok() also slides a stretch a little either way (never past its moment) to find one that keeps them."""
    def __init__(self, project, cams):
        self.project, self.cams = project, cams
        c = project.config.get("steady") or {}
        self.tripod = project.config.get("tripod") or []
        self.still = float(c.get("still", 0.05))             # frame widths a second, tripod
        self.zoom = float(c.get("zoom", 0.05))               # log zoom a second, tripod
        self.swing = float(c.get("swing", 0.3))              # frame widths a second, handheld
        self.drop = float(c.get("drop", 0.15))               # frame widths a second, straight up or down, handheld
        self.twist = float(c.get("twist", 0.12))             # radians a second, handheld
        self.jitter = float(c.get("jitter", 0.012))          # frame widths off the path's 1 s average, handheld
        self.jitter_steadied = float(c.get("jitter_steadied", 0.025))   # the same for footage always steadied
        self.blur = float(c.get("blur", 0.35))               # sharpness below this share of the clip's median
        self.pad = float(c.get("pad", 0.3))                  # seconds each fault is kept clear of
        self.air_turn = float(c.get("air_turn", 0.05))       # radians a second a drone shot may turn (yaw) at most
        self.walk = list(c.get("walking") or [])             # clips shot walking and steadied hard: their sway is
                                                             # no fault (reel.json "steady" "walking")
        self.centre = float(c.get("centre", 0.3))
        self.trim = float(c.get("air_trim", 25.0))
        self.high = float(c.get("air_high", 0.5))
        self._sc, self._hi, self._faults = {}, None, {}
        from .aim import Aim, settings
        from .holding import Holding
        self.aim, self.aim_max = Aim(project), settings(project.config)[2]
        self.hold = Holding(project)                         # who holds the mic (reel.json "holding")

    def is_tripod(self, u):
        return any(u["category"] == p or fnmatch.fnmatch(u["clip"], p) for p in self.tripod)

    def _score(self, u):
        if u["score"] not in self._sc:
            from . import frame as FR
            p = self.project.path("scores", u["score"] + ".npz")
            self._sc[u["score"]] = FR.load(p) if os.path.exists(p) else None
        return self._sc[u["score"]]

    def faults(self, u):
        """(t, bad, why) for a unit's clip at moves.RATE: where the camera misbehaves (see the class docstring),
        each fault widened by `pad` either side. None when the clip's moves weren't measured."""
        from scipy.ndimage import uniform_filter1d

        from . import moves
        key = u["score"]
        if key in self._faults:
            return self._faults[key]
        z = moves.series(self.project, u)
        if z is None or len(z["t"]) < 6:
            self._faults[key] = None
            return None
        t, dx, dy, rot, zm = z["t"], z["dx"], z["dy"], z["rot"], z["zoom"]
        R = moves.RATE
        k = max(int(R * 0.5), 1)

        def rate(x):                                          # per second, over the half second about each frame
            return uniform_filter1d(x, k, mode="nearest") * R
        why = {}
        lost = (dx == 0) & (dy == 0) & (rot == 0) & (zm == 0)      # the tracker lost the picture (a smear)
        lost[0] = False
        walking = any(fnmatch.fnmatch(u["clip"], q) or u["category"] == q for q in self.walk)
        why["lost"] = uniform_filter1d(lost.astype(float), k, mode="nearest") * k >= (3.0 if walking else 1.5)
        if self.is_tripod(u):
            why["moved"] = rate(np.hypot(dx, dy)) > self.still
            why["zoom"] = np.abs(rate(zm)) > self.zoom
        elif any(fnmatch.fnmatch(u["clip"], q) or u["category"] == q for q in self.walk):
            why["drop"] = np.abs(rate(dy)) > self.drop * 2.5     # (walking with the camera, steadied hard: the sway
        else:                                                   # it takes out isn't a fault -- a drop still is)
            why["swing"] = rate(np.hypot(dx, dy)) > self.swing
            why["drop"] = np.abs(rate(dy)) > self.drop
            why["twist"] = np.abs(rate(rot)) > self.twist
        if not self.is_tripod(u) and not any(fnmatch.fnmatch(u["clip"], q) or u["category"] == q for q in self.walk):
            px, py = np.cumsum(dx), np.cumsum(dy)
            steadied = any(fnmatch.fnmatch(u.get("clip", ""), q) or u["category"] == q
                           for q in self.project.config.get("stabilise_always") or [])
            lim = self.jitter_steadied if steadied else self.jitter  # (what the render's steadying crop takes out
            why["shake"] = np.hypot(px - uniform_filter1d(px, R, mode="nearest"),     # isn't a fault)
                                    py - uniform_filter1d(py, R, mode="nearest")) > lim
        d = self._score(u)
        if d is not None and len(d["t"]) > 8:                 # the picture going soft (against the clip's own)
            sh = uniform_filter1d(d["sharp"].astype(float), 4, mode="nearest")
            med = float(np.median(sh)) or 1.0
            why["blur"] = np.interp(t, d["t"], sh) < self.blur * med * (0.5 if walking else 1.0)   # (walking blurs)
        if self.aim.ok and u["id"] in self.aim.by:             # frames of nothing
            at, an = self.aim.by[u["id"]]
            if len(at):                                       # (sampled every couple of seconds: a frame of
                j = an > self.aim_max                         # nothing marks its own half second, or the whole gap
                k = np.clip(np.searchsorted(at, t), 1, max(len(at) - 1, 1))   # between two such frames; the motion
                lo, hi = np.maximum(k - 1, 0), np.minimum(k, len(at) - 1)  # marks where it starts and ends)
                near = np.where(np.abs(t - at[lo]) <= np.abs(t - at[hi]), lo, hi)
                why["aim"] = (j[near] & (np.abs(t - at[near]) <= 0.5)) | (j[lo] & j[hi])
        bad = np.zeros(len(t), bool)
        for v in why.values():
            bad |= v
        w = int(round(self.pad * R))
        if w:
            bad = uniform_filter1d(bad.astype(float), 2 * w + 1, mode="nearest") > 0
        self._faults[key] = (t, bad, why)
        return self._faults[key]

    def camera_ok(self, u, a, b):
        if a < u["t0"] - 0.05 or b > u["t1"] + 0.05:          # past the clip's ends: nothing there to play
            return False
        f = self.faults(u)
        if f is None:
            return True
        t, bad, _ = f
        m = (t >= a) & (t <= b)
        return not bool(bad[m].any())

    def camera_why(self, u, a, b):
        """The faults in a..b by name (for the plan's report)."""
        f = self.faults(u)
        if f is None:
            return []
        t, _, why = f
        m = (t >= a) & (t <= b)
        return sorted(k for k, v in why.items() if v[m].any())

    def subject_ok(self, u, a, b):
        from . import frame as FR
        d = self._score(u)
        if d is None:
            return True
        tt = np.arange(a, b, 0.25)
        union, _, _ = FR.track(d, tt)
        ok = ~np.isnan(union[:, 0])
        if ok.mean() < 0.5:
            return True                                      # nobody tracked: nothing to keep centred
        cx = (union[ok, 0] + union[ok, 2]) / 2
        inside = (union[ok, 0] > 0.01) & (union[ok, 2] < 0.99) & (union[ok, 1] > 0.005)
        stays = ok[0] <= ok[-1] and ok[len(ok) // 2:].mean() >= 0.9 if ok[0] else True   # whoever it opens on is
        return bool(np.all(np.abs(cx - 0.5) <= self.centre) and inside.all() and stays)  # still there at its end

    def aerial_ok(self, u, a, b, looks):
        if a < u["t0"] + self.trim or b > u["t1"] - self.trim:
            return False
        if looks.high is None:
            return True
        k = looks.by.get(u["id"])
        if k is None:
            return True
        near = k[(looks.t[k] >= a - 1.0) & (looks.t[k] <= b + 1.0)]
        if not (len(near) == 0 or looks.high[near].min() >= self.high):
            return False
        return self.one_move(u, a, b)

    def one_move(self, u, a, b, tail=0.4):
        """A drone shot plays one move: nothing new starting in its last `tail` of it (a hover that swings away to
        a boat, a push that turns into a pan) -- it is cut before the camera goes somewhere else."""
        from . import moves
        z = moves.series(self.project, u)
        if z is None:
            return True
        m = (z["t"] >= a) & (z["t"] <= b)
        if m.sum() < 8:
            return True
        turn = float(np.abs(np.convolve(z["rot"][m], np.ones(5) / 5, "same")).max()) * moves.RATE
        if turn > self.air_turn:                               # (the picture turning: the drone yawing -- its
            return False                                       # orientation should hold through a shot)
        sp = np.hypot(z["dx"][m], z["dy"][m]) * moves.RATE + np.abs(z["rot"][m]) * moves.RATE * 0.5
        k = int(len(sp) * (1 - tail))
        head, end = float(np.median(sp[:k])), float(np.percentile(sp[k:], 80))
        return not (end > 2.0 * head + 0.03)

    def focus(self, u, a, length):
        """Where a panel cut from this footage should centre over a stretch: the subject's middle across, and down
        it a little above its middle -- faces sit in the upper part of a person (frame.track's union, its median
        over the stretch); None when nobody is tracked."""
        if self.cams.aerial(u["category"]):                   # (the drone's own framing: over the action)
            return None
        if self.hold.applies(u) and self.hold.measured(u):     # the speaker: whoever holds the mic
            b = self.hold.at(u, a, a + length)
            if b is not None:
                return [round(float((b[0] + b[2]) / 2), 4), round(float(b[1] + 0.28 * (b[3] - b[1])), 4)]
        d = self._score(u)
        if d is None:
            return None
        m = (d["pose_t"] >= a) & (d["pose_t"] <= a + length)
        if not m.any():
            return None
        rule = next((v for k, v in (self.project.config.get("subject") or {}).items()
                     if u["category"] == k or fnmatch.fnmatch(u["clip"], k)), None)
        if rule:                                               # people together (a baptism: the three in the
            pts = []                                           # water): the middle of the group, every whole person
            for boxes in d["boxes"][m]:
                ok = boxes[~np.isnan(boxes[:, 0])]
                ok = ok[(ok[:, 3] - ok[:, 1]) / d["ah"] >= float(rule.get("tall", 0.15))]
                if len(ok):
                    x0, x1 = ok[:, 0].min() / d["aw"], ok[:, 2].max() / d["aw"]
                    y0, y1 = ok[:, 1].min() / d["ah"], ok[:, 3].max() / d["ah"]
                    pts.append(((x0 + x1) / 2, y0 + 0.35 * (y1 - y0)))
            if pts:
                cx, cy = np.median(np.array(pts), axis=0)
                return [round(float(cx), 4), round(float(cy), 4)]
        pts = []
        for boxes, kp in zip(d["boxes"][m], d["kp"][m], strict=True):   # the nearest person whose face shows
            ok = np.nonzero(~np.isnan(boxes[:, 0]))[0]                 # (the one at the mic, not a body passing in
            if not len(ok):                                            # front or the crowd behind)
                continue
            seen = [j for j in ok if np.nan_to_num(kp[j, 0, 2]) >= 0.5]
            pool = seen or list(ok)
            b = boxes[pool[int(np.argmax([(boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]) for j in pool]))]]
            pts.append(((b[0] + b[2]) / 2 / d["aw"], (b[1] + 0.28 * (b[3] - b[1])) / d["ah"]))
        if not pts:
            return None
        cx, cy = np.median(np.array(pts)[:, :2], axis=0)
        return [round(float(cx), 4), round(float(cy), 4)]

    def presence(self, u, a, length):
        """How the nearest person sits over a stretch: (median height as a share of the frame, median distance of
        their middle from the frame's middle, share of frames their face is seen -- the pose's nose and an eye,
        their tallest -- a body right up against the lens fills the whole height), or None when nobody is found."""
        if self.hold.applies(u) and self.hold.measured(u):     # the speaker (the mic's holder), not the biggest face
            z = self.hold._load(u)
            m = (z["t"] >= a) & (z["t"] <= a + length)
            hb = z["holder"][m]
            if m.sum() < 3:
                return None
            ok = ~np.isnan(hb[:, 0])
            if not ok.any():
                return (0.0, 0.5, 0.0, 0.0)
            hh, cx = hb[ok, 3] - hb[ok, 1], (hb[ok, 0] + hb[ok, 2]) / 2
            return (float(np.median(hh)), float(np.median(np.abs(cx - 0.5))), float(ok.mean()), 0.0)   # (a close
            # speaker fills the height: no body-against-the-lens test once we know who it is)
        d = self._score(u)
        if d is None:
            return None
        m = (d["pose_t"] >= a) & (d["pose_t"] <= a + length)
        rows = []
        for boxes, kp in zip(d["boxes"][m], d["kp"][m], strict=True):
            ok = np.nonzero(~np.isnan(boxes[:, 0]))[0]
            if not len(ok):
                continue
            seen = [j for j in ok if np.nan_to_num(kp[j, 0, 2]) >= 0.5 and np.nan_to_num(kp[j, 1:3, 2]).max() >= 0.5]
            pool = seen or list(ok)                            # the largest person whose face shows (else largest)
            k = pool[int(np.argmax([(boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]) for j in pool]))]
            b = boxes[k]
            face = float(bool(seen))
            rows.append(((b[3] - b[1]) / d["ah"], abs((b[0] + b[2]) / 2 / d["aw"] - 0.5), face))
        if len(rows) < 3:
            return None
        rows = np.array(rows)
        return (float(np.median(rows[:, 0])), float(np.median(rows[:, 1])), float(np.mean(rows[:, 2])),
                float(rows[:, 0].max()))

    def frontal(self, u, a, length, tall=0.25):
        """Share of a stretch's pose samples with someone near the lens (`tall` of the frame or more) facing it: the
        nose and both eyes seen, the nose between the eyes -- not a profile or a back."""
        d = self._score(u)
        if d is None:
            return 0.0
        m = (d["pose_t"] >= a) & (d["pose_t"] <= a + length)
        if not m.any():
            return 0.0
        n = 0
        for boxes, kp in zip(d["boxes"][m], d["kp"][m], strict=True):
            for b, k in zip(boxes, kp, strict=True):
                if np.isnan(b[0]) or (b[3] - b[1]) / d["ah"] < tall:
                    continue
                nose, le, re = k[0], k[1], k[2]
                if min(np.nan_to_num(nose[2]), np.nan_to_num(le[2]), np.nan_to_num(re[2])) >= 0.5 and \
                        min(le[0], re[0]) <= nose[0] <= max(le[0], re[0]):
                    n += 1
                    break
        return n / m.sum()

    def hard_ok(self, u, src_in, length, looks=None, slide=(0.0, -2.0, 2.0)):
        """The rules never relaxed: an aerial shot is of the drone flying (not taking off, landing or being
        caught); any other, the camera clear of faults (shake, the camera dropped, a smear, a frame of nothing)
        and, where a category's subject is whoever holds the mic, the mic held; where it is people together, they
        are there, whole (people_ok) -- at src_in, or slid by `slide`."""
        if self.cams.aerial(u["category"]):
            return any(self.aerial_ok(u, src_in + d, src_in + d + length, looks) for d in slide)
        return any(self.camera_ok(u, src_in + d, src_in + d + length) and self.held_ok(u, src_in + d, length)
                   and self.people_ok(u, src_in + d, length) for d in slide)

    def people_ok(self, u, a, length):
        """reel.json "subject": {"<category or clip pattern>": {"people": 2, "share": 0.9}}: footage whose moment is
        people together (a baptism: the candidate and the two in the water with them) shows at least that many whole
        people -- inside the frame's sides, at least `tall` of its height -- in `share` of the stretch: never empty
        water, one person waiting alone, or someone cut by the frame's edge. Footage without a rule: always."""
        rule = next((v for k, v in (self.project.config.get("subject") or {}).items()
                     if u["category"] == k or fnmatch.fnmatch(u["clip"], k)), None)
        if not rule:
            return True
        d = self._score(u)
        if d is None:
            return True
        m = (d["pose_t"] >= a) & (d["pose_t"] <= a + length)
        if m.sum() < 2:
            return False
        need, share = int(rule.get("people", 1)), float(rule.get("share", 0.9))
        edge, tall = float(rule.get("edge", 0.02)), float(rule.get("tall", 0.15))
        n = 0
        for bx in d["boxes"][m]:
            ok = bx[~np.isnan(bx[:, 0])]
            whole = sum(1 for x in ok if x[0] / d["aw"] > edge and x[2] / d["aw"] < 1 - edge
                        and (x[3] - x[1]) / d["ah"] >= tall)
            n += whole >= need
        return n / m.sum() >= share

    def held_ok(self, u, a, length):
        """A category whose subject is whoever holds something (reel.json "holding": the speaker with the mic):
        it is held through the stretch (holding.SHARE of it). Other footage: always."""
        from .holding import SHARE
        return not self.hold.applies(u) or self.hold.held(u, a, a + length) >= SHARE

    def hard_near(self, u, src_in, length, looks=None, reach=None, step=0.25):
        """The nearest in-point (within `reach` s; the whole unit by default) whose stretch keeps the hard rules, or
        None."""
        reach = reach if reach is not None else max(u["t1"] - u["t0"], step)
        for d in sorted(np.arange(-reach, reach + step, step), key=abs):
            a = src_in + float(d)
            if u["t0"] <= a and a + length <= u["t1"] and self.hard_ok(u, a, length, looks, slide=(0.0,)):
                return round(a, 3)
        return None

    def ok(self, u, src_in, length, t=None, looks=None, slide=(0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0)):
        """The in-point to use (src_in, or one slid a little), or None when no stretch near it keeps the rules."""
        for d in slide:
            a = src_in + d
            b = a + length
            if a < u["t0"] or b > u["t1"] or (t is not None and not (a <= t <= b)):
                continue
            if self.cams.by_move(u):
                if self.aerial_ok(u, a, b, looks) if self.cams.aerial(u["category"]) else True:
                    return round(a, 3)
                continue
            if self.camera_ok(u, a, b) and self.subject_ok(u, a, b):
                return round(a, 3)
        return None


def nat_sound(project, plan, length=3.2):
    """The crowd's cheer under each dunk's rise -- where the rise plays in the reel (with a ramp, where the ramp puts
    it) -- from the camera file's own sound (moments.cheers' onset nearest the rise), and under the last shot (the
    celebration at the close), and nowhere else. reel.json "nat": false leaves the music alone."""
    from .events import cheer, level
    out, heard = [], {}
    for s in plan["shots"]:
        if not s.get("dunk") and not s.get("last") and not s.get("nat"):
            continue
        u = s["unit"]
        src = os.path.join(project.folder, u["source"])
        if src not in heard:
            heard[src] = cheer(level(src))
        if s.get("last") or s.get("nat"):                     # the close (and a pick asked for): its own sound,
            a0 = s["span"][0] if ("fit" in (s.get("fx") or []) or s.get("fit")) and s.get("span") \
                else s["src_in"]                               # from where its picture starts
            out.append(dict(source=u["source"], src_t=round(a0, 2), at=round(s["rec_in"], 3),
                            length=round(s["length"], 3)))
            continue
        near = [c for c, _ in heard[src] if -1.0 <= c - s["t"] <= 3.0]
        if not near:                                           # a cheer too quiet to find: the camera's own sound
            near = [s["t"] + 0.5]                              # from the rise all the same (every dunk is cheered)
        rise = s["peak"]                                       # where the rise plays in the shot
        if s.get("fx"):
            try:
                from ..fx.speed import ramp_window
                rise = ramp_window(s["fx"][0], s["length"], s["peak"], s.get("beat") or plan["beat"],
                                   params={"ramps": project.config.get("ramps")})[2]
            except (KeyError, ValueError, TypeError):
                pass
        out.append(dict(source=u["source"], src_t=round(near[0] - 0.2, 2), at=round(s["rec_in"] + rise, 3),
                        length=min(length, s["length"] - rise + 1.0)))
    return out


def best_in_pick(project, steady, u, a, b, L, step=0.1):
    """Where in an editor's pick (a..b) a shorter slot (L) plays: the stretch with the fewest camera faults (a blur,
    a roll as the operator settles) and, for the drone, the most of what reel.json "aim" "prefer" asks for (people,
    a boat -- not the empty sand the flight starts on) and the least of nothing; else the earliest."""
    from .aim import Aim
    aim = getattr(steady, "_aim_obj", None) or Aim(project)
    steady._aim_obj = aim
    air = "DJI" in u.get("clip", "") or u.get("category", "").startswith("drone")
    best, key = a, None
    for x in np.arange(a, b - L + 1e-6, step):
        faults = len(steady.camera_why(u, x, x + L) or [])
        pref = 0.0
        if air and u["id"] in aim.pref:
            t, v = aim.pref[u["id"]]
            m = (t >= x) & (t <= x + L)
            pref = float(v[m].mean()) if m.any() else 0.0
        k = (faults, -round(pref, 1), aim.nothing_in(u["id"], x, x + L, pad=0.0) > 0.6, x)
        if key is None or k < key:
            best, key = float(x), k
    return best


def place(project, fmt, seq, music, cfg, beat, steady=None, looks=None, lead=None, downbeat=None):
    """The slots on the music: each at its beats, its in-point (a dunk's rise at DUNK_AT of the shot, slowed; a plain
    shot's moment 40% in, as plan.in_point; the steadiest stretch near it when the clip shakes), its crop and shake.
    Hard cuts: no transitions, holds or glides -- the reference's timing is the point. downbeat: the beat time the
    first slot starts on (the documentary's first bar line), else the music's start; lead: how far ahead of the beat
    the cuts land (else plan.ANTICIPATE)."""
    from . import shake
    if downbeat is not None:
        music = dict(music, start=float(downbeat))
    lead = P.ANTICIPATE if lead is None else float(lead)
    T = P.clock(music)
    rec = 0
    lo = cfg["shake"][0]
    for s in seq:
        L = T(rec + s["beats"]) - T(rec)
        s["beat"] = round(L / s["beats"], 5)
        u = s["unit"]
        if s.get("photos") or s.get("photo"):                 # a still: nothing to find in it, just its place
            f0, f1 = round(T(rec) * FPS), round(T(rec + s["beats"]) * FPS)
            s.update(rec_in=round(T(rec), 3), length=round(L, 3), frames=f1 - f0, src_in=0.0, peak=0.0, crop=None,
                     stab=False, jitter=None, group=False, transition=None)
            rec += s["beats"]
            continue
        if s.get("dunk") and s.get("split"):                  # a dunk side by side: real time, the rise at
            src_len = L                                          # DUNK_AT of the shot in each panel
            src_in = s["t"] - DUNK_AT * L
            peak = DUNK_AT * L
            for p_ in s["split"]["panels"]:
                p_["t"] = round(p_["t"] - DUNK_AT * L + 0.4 * L, 3)   # (the panels' in-point is t - 0.4 L)
        elif s.get("dunk"):                                   # (the speed curve reads its own source:
            src_len = L                                          # render.fx_shot; the rise lands at the peak)
            at = DUNK_AT
            ramp = (s.get("fx") or [None])[0]

            def window(x, ramp=ramp, L=L, s=s):                  # the source the ramp reads around the rise
                if ramp:
                    try:
                        from ..fx.speed import ramp_window
                        pre, post, _ = ramp_window(ramp, L, x * L, s["beat"],
                                                   params={"ramps": project.config.get("ramps")})
                        return s["t"] - pre, s["t"] + post
                    except (KeyError, ValueError, TypeError):
                        pass
                return s["t"] - x * L, s["t"] + (1 - x) * L
            if steady is not None:                               # the rise where the camera holds still around it
                at = next((x for x in sorted(np.arange(0.2, 0.81, 0.02), key=lambda x: abs(x - DUNK_AT))   # (the
                           if steady.camera_ok(u, *window(x))), None)   # nearest the usual that is clear of a
                # focus hunt or a nudge just before the dunk)
                if at is None:                                   # (the camera rule is hard: flagged loudly)
                    at, s["hard_fail"] = DUNK_AT, steady.camera_why(u, s["t"] - DUNK_AT * L, s["t"] + (1 - DUNK_AT) * L)
            src_in = s["t"] - at * L
            peak = at * L
        elif s.get("stutter"):
            src_len = L * 2.0                                    # jump-cuts skip forward through twice the time
            src_in = s["t"] - 0.5 * src_len
            if steady is not None and not steady.hard_ok(u, src_in, src_len, looks, slide=(0.0,)):
                src_in = steady.hard_near(u, src_in, src_len, looks) or src_in
            peak = 0.5 * L
        elif s.get("blend"):                                  # a double exposure: the base slowed from its in-
            bl = s["blend"]                                    # point; over it, the person about to go under
            s["speed"] = bl["speed"]
            src_len = L * bl["speed"]
            src_in = s["span"][0]
            peak = 0.5 * L
            ou = bl["unit"]
            o_in = max(ou["t0"], bl["t"] - 3.7 - L)            # (ending before the dunk shot that follows begins)
            fc = steady.focus(ou, o_in, L) if steady is not None else None
            cx, cy = (fc or [0.5, 0.4])
            z = 0.5                                             # close: half the frame, on them
            cx, cy = min(max(cx, z / 2), 1 - z / 2), min(max(cy, z / 2), 1 - z / 2)
            bl.update(src_in=round(o_in, 3), crop=dict(w=z, h=z, z=1 / z, path=[[0.0, round(cx, 4), round(cy, 4)]]))
        elif s.get("span") and (s.get("fit") or "fit" in (s.get("fx") or [])):   # the whole pick, fitted to the slot
            a, b = s["span"]
            pe = next((v for c, v in (project.config.get("pick_end") or {}).items() if c in (u["clip"], u["id"])), None)
            if pe is not None:                                 # reel.json "pick_end": {"C2220": 2.0} -- where a
                b = min(b, float(pe) + u.get("export_t0", 0.0))   # fitted pick stops short (a lurch at its end)
                s["span"] = [a, b]                             # (the renderer reads the span: trimmed there too)
            if s.get("last"):                                  # the closing shot: one even pace across it (a pan
                s["fit_even"] = True                           # along everyone, not eased in and hurried out)
            src_in, src_len = a, b - a
            s["speed"] = round(src_len / L, 4)
            peak = 0.5 * L
        elif s.get("span"):                                   # an editor's pick: the cleanest stretch of it
            src_len = L
            a, b = s["span"]
            src_in = a if b - a >= L else a - (L - (b - a)) / 2
            if b - a > L + 0.15 and steady is not None:
                src_in = best_in_pick(project, steady, u, a, b, L)
            peak = min(max(s["t"] - src_in, 0.0), L)
        else:
            src_len = L * (s.get("speed") or 1.0)
            src_in = s["t"] - 0.4 * src_len
            if steady is not None:                               # slid to a stretch that keeps the rules throughout
                h = steady.ok(u, src_in, src_len, s["t"], looks)
                if h is not None:
                    src_in = h
                else:
                    s["unsteady"] = True
                    if not steady.hard_ok(u, src_in, src_len, looks, slide=(0.0,)):   # never a hard-rule break
                        alt = steady.hard_near(u, src_in, src_len, looks)            # (a drone landing): the
                        if alt is not None:                                          # nearest stretch that keeps
                            src_in = alt                                             # them instead
                        else:
                            s["hard_fail"] = True
            peak = 0.4 * L
        src_in = min(max(src_in, u["t0"]), max(u["t1"] - src_len, u["t0"]))
        if not s.get("span"):                                  # (a pick's own range is banned to the rest -- not
            src_in = P.clear_of_bans(src_in, src_len, u)       # to the pick itself)
        if steady is not None and s.get("dunk"):              # a dunk is its own subject (the candidate under
            if not s.get("hard_fail") and not steady.camera_ok(u, src_in, src_in + src_len):   # water): the camera
                s["hard_fail"] = steady.camera_why(u, src_in, src_in + src_len)                # rule alone
        elif steady is not None and not s.get("span") and not s.get("hard_fail") and \
                not steady.hard_ok(u, src_in, src_len, looks, slide=(0.0,)):       # (clamped or moved off a ban
            alt = steady.hard_near(u, src_in, src_len, looks)                     # onto a fault: the nearest
            if alt is not None and not s.get("dunk"):                             # clean stretch, else flagged)
                src_in = P.clear_of_bans(alt, src_len, u)
            if not steady.hard_ok(u, src_in, src_len, looks, slide=(0.0,)):
                s["hard_fail"] = steady.camera_why(u, src_in, src_in + src_len) or ["air"]
        rv = (project.config.get("reverse") or {})
        rv = rv.get(u["clip"]) or rv.get(u["id"]) or []
        x0 = u.get("export_t0", 0.0)                           # (times in the export file's own timeline)
        if any(a + x0 - 0.5 <= src_in and src_in + src_len <= b + x0 + 0.5 for a, b in (P.span_of(x) for x in rv)):
            s["reverse"] = True                                # (reel.json "reverse": this stretch played backwards)
        f0, f1 = round(T(rec) * FPS), round(T(rec + s["beats"]) * FPS)
        s.update(rec_in=round(T(rec), 3), length=round(L, 3), frames=f1 - f0, src_in=round(src_in, 3),
                 peak=round(min(max(s["t"] - src_in, 0.0), L) if not s.get("dunk") else peak, 3), group=False)
        s["crop"] = P.reframe(project, u, s["src_in"], src_len, fmt)
        if s.get("last") and s.get("pick") and fmt == "landscape":   # the editor's closing shot of everyone: their
            s["crop"] = None                                           # frame, whole -- never zoomed onto a few
        j = shake.jitter(project, u, s["src_in"], s["src_in"] + src_len)
        j = j / max((s["crop"] or {}).get("w", 1.0), 0.2) if j is not None else None
        s["jitter"] = round(j, 5) if j is not None else None
        lim = next((v for p, v in (project.config.get("shake_max") or {}).items()
                    if fnmatch.fnmatch(u.get("clip", ""), p)), None)
        s["stab"] = bool(j is not None and j > (lim[0] if lim else lo))
        if any(fnmatch.fnmatch(u.get("clip", ""), q) or u["category"] == q            # reel.json
               for q in project.config.get("stabilise_always") or []):              # "stabilise_always": every
            s["stab"] = True                                   # shot of these cameras steadied (render.steady_crop
                                                               # keeps it only where it steadies)
        if s.get("split"):
            for p in s["split"]["panels"]:
                pu = p["unit"]
                if s.get("dunk"):                              # (one dunk from two cameras: the same second in
                    p["src_in"] = round(min(max(p["t"] - 0.4 * L, pu["t0"]), max(pu["t1"] - L, pu["t0"])), 3)
                    p["crop"] = P.reframe(project, pu, p["src_in"], L, fmt)   # both -- never slid apart)
                    if steady is not None:
                        p["focus"] = steady.focus(pu, p["src_in"], L)
                    continue
                a0 = min(max(p["t"] - 0.4 * L, pu["t0"]), max(pu["t1"] - L, pu["t0"]))
                small = len(s["split"]["panels"]) > 4            # a grid's cells: a sixteenth of the frame each,
                if steady is not None and small:                 # where a little shake doesn't show -- the mic
                    if not steady.held_ok(pu, a0, L):              # still held throughout
                        h = next((a for a in np.arange(pu["t0"], pu["t1"] - L, 0.25) if steady.held_ok(pu, a, L)), None)
                        if h is None:
                            s.setdefault("hard_fail_panels", []).append(pu["id"])
                        a0 = h if h is not None else a0
                elif steady is not None:                         # a panel keeps the hard rules as a shot does
                    h = steady.ok(pu, a0, L, p["t"], looks)
                    if h is None and not steady.hard_ok(pu, a0, L, looks, slide=(0.0,)):
                        h = steady.hard_near(pu, a0, L, looks)
                        if h is None:
                            s.setdefault("hard_fail_panels", []).append(pu["id"])
                    a0 = h if h is not None else a0
                p["src_in"] = round(a0, 3)
                p["crop"] = P.reframe(project, pu, p["src_in"], L, fmt)
                if p.get("focus_at"):                          # (framed where the editor's setting says)
                    p["focus"] = p["focus_at"]
                elif steady is not None:                         # the panel framed on its person, not the frame's
                    p["focus"] = steady.focus(pu, p["src_in"], L)  # middle
        s["transition"] = "dissolve" if s.get("blend") and rec else None   # (into a double exposure: dissolved)
        rec += s["beats"]
    video = round(T(rec), 3)
    return dict(format=fmt, beat=beat, music_start=round(music["start"] - lead, 3), music_length=video,
                video_length=video, stinger_at=video, stinger=0.0, shots=seq, acts={},
                song=music.get("song") or project.config.get("song"))


def add_card(project, fmt, plan, card, length, ground=None, cams=None):
    """The end card as the last shot: this project's logo over moving footage softened into a ground (a calm aerial
    of the sky and the sea: reel.json "endcard" "ground": ["<unit id>", seconds] -- or ["<unit id>", from, to], that
    range slowed to fill the card, and kept out of the film before it -- else the steadiest aerial moment
    not used), fading out with the music."""
    rec = plan["video_length"]
    frames = round((rec + length) * FPS) - round(rec * FPS)
    moments = project.load("moments.json")
    used = {(s["unit"]["id"], round(s["t"])) for s in plan["shots"]}
    pick = (project.config.get("endcard") or {}).get("ground")
    unit, t = None, None
    if pick:
        r = next((r for r in moments if r["unit"]["id"] == pick[0]), None)
        unit, t = (r["unit"], float(pick[1])) if r else (None, None)
    if unit is None:
        air = [(r, m) for r in moments if cams and cams.aerial(r["unit"]["category"]) and r["orient"] == "landscape"
               for m in r["moments"] if (r["unit"]["id"], round(m["t"])) not in used
               and m["t"] + length * 0.5 <= r["unit"]["t1"] and m["t"] - length * 0.5 >= r["unit"]["t0"]]
        if air:
            r, m = min(air, key=lambda x: cams.motion(x[0]["unit"], x[1]["t"], 3.0))
            unit, t = r["unit"], m["t"]
    if unit is None:
        return plan
    speed = 1.0
    if pick and len(pick) >= 3 and unit is not None:         # ["<unit>", from, to]: the editor's exact range, slowed
        src_in, speed = float(pick[1]), min(1.0, (float(pick[2]) - float(pick[1])) / length)   # to fill the card
        t = src_in
    else:
        src_in = max(unit["t0"], min(t - length / 2, unit["t1"] - length))
    s = dict(card=dict(t0=card["t0"], t1=card["t1"], card=card.get("card")), length=round(length, 3),
             frames=frames, rec_in=rec, beats=0, act="card", fx=[], effect=None, kind="card", t=t, src_in=round(src_in, 3),
             peak=round(t - src_in, 3), unit=unit, crop=None, stab=False, speed=round(speed, 4))
    plan["shots"].append(s)
    plan["video_length"] = round(rec + length, 3)
    plan["music_length"] = plan["stinger_at"] = plan["video_length"]
    return plan


def run(project, fmt):
    plan = build(project, fmt, project.load(project.music_file), project.load("moments.json"))
    report(plan)
    return plan


def report(plan):
    seq = plan["shots"]
    print(f"[reel] {plan['format']}: {plan.get('mode', 'reference')} cut, {len(seq)} shots, {len({s['unit']['id'] for s in seq})}/"
          f"{plan['units_total']} units, {plan['video_length']:.1f} s; dunks {plan['dunks']['placed']}/"
          f"{plan['dunks']['found']}")
    cats = {}
    for s in seq:
        cats[s["act"]] = cats.get(s["act"], 0) + 1
    print("[reel]   by category:", cats)
    if plan.get("timing"):
        tm = plan["timing"]
        print(f"[reel]   timing: {tm['pace']}")
        for k, kind, detail in tm["issues"]:
            print(f"[reel]   timing {kind}{'' if k is None else ' shot ' + str(k)}: {detail}")
    for s in seq:
        tags = [k for k in ("title", "split", "stutter", "dunk", "card") if s.get(k)]
        if tags:
            print(f"[reel]     {s['rec_in']:6.1f}s {s['unit']['id']:28s} {', '.join(tags)}")
    broken = [s for s in seq if s.get("hard_fail") or s.get("hard_fail_panels")]
    if broken:
        print(f"[reel]   WARNING hard rule broken (no stretch nearby keeps it): "
              f"{', '.join(s['unit']['id'] + '@' + str(s['src_in']) + str(s.get('hard_fail') or '') + str(s.get('hard_fail_panels') or '') for s in broken)}")
    loose = [s for s in seq if s.get("unsteady")]
    if loose:
        print(f"[reel]   no steady stretch found (kept): {', '.join(s['unit']['id'] + '@' + str(s['src_in']) for s in loose)}")
    shaky = [s for s in seq if s.get("jitter") and s["jitter"] > 0.006]
    if shaky:
        print(f"[reel]   shaky: {', '.join(s['unit']['id'] + '@' + str(s['src_in']) for s in shaky)}")


def lut_for(config, clip):
    """The look LUT for a clip (reel.json "lut": {"<pattern>": "<cube>"}), or None."""
    for pat, cube in (config.get("lut") or {}).items():
        if fnmatch.fnmatch(clip, pat):
            return cube
    return None
