"""The documentary engine: the structure of a slow, observational film -- where its cuts fall and what each shot is
for -- worked out from the music, the footage and the order of the day, not copied from another film's edit.

A music video cuts on the beat, short and busy; an event highlight copies that. A contemplative documentary holds
(docs/research/documentary-editing.md, measured on 17 films in docs/research/references.json): its shots are whole
bars (2 by default, 3-4 for a held moment, 1 for a cutaway, never less), its cuts land on bar lines a couple of frames
before the downbeat, its pace stays level when the song swells (the loud chorus is no reason to cut faster), the
first shot is long and the last longer, the key moments (a baptism's immersion) are single unbroken shots with room
before and after, and aerials are punctuation between scenes. romp's timing ideas (docs/research/romp-timing.md) give
the rest: a shot's length starts from what it needs -- time to read the frame, for its action to finish, for a camera
move to settle -- and the music only snaps that to the grid; one accent per phrase, not one per beat.

structure() turns that into slots -- the same shape a reference film's shots take (template.slots), so the casting
in template.Cut (which footage fills each slot: the camera and subject rules, the speaker at the mic, the event in
order, the drone over the dunk) is shared by both modes:

    1. the story: the ground categories in the order they happened (their clips' wall clocks); aerial categories
       are punctuation, not chapters
    2. fixed parts: the opening (an aerial under the title, `open_bars`), the event (a lead-in, every event moment
       its own `event_bars` shot -- every `angles_every`-th side by side, the ground and the drone at that second --
       a cutaway after every `cutaway_every`), every_unit categories (the speakers: their lineups and grid), the last held shot (`last_bars`)
       and the end card
    3. the rest of the song's bars shared between the other sections by how much usable footage each has
    4. section starts snapped to 4-bar phrases where a flexible section can give or take a bar
    5. each section's shots: whole bars, by its kind (`bars`), varied by a seeded hash (the same every run), an
       aerial between sections

check() is the timing lint for a placed plan (romp's motion check, applied to the edit): pace (median, 10th
percentile, cuts per minute), whole bars, shots shorter than they need. reel.json "documentary" sets any of DEFAULTS.
"""
from __future__ import annotations

import hashlib

import numpy as np

DEFAULTS = {
    "lead": 0.09,             # cuts this far (s) ahead of the beat (the films measured: 0.08-0.10 ahead of the tracker)
    "open_bars": 4,           # the opening shot (an aerial under the title): 1.5-2x the median
    "last_bars": 3,           # the last shot before the card: at least 1.5x the median
    "card_bars": 3,           # the end card over moving footage
    "event_bars": 3,          # each event moment (an immersion) held: 2-3 bars, 5-8 s
    "lead_in": [],            # the event's lead-in shots (walking in, the group in the water), in bars
    "cutaway_every": 2,       # a cutaway (the drone over the water, at that moment) after every n event moments
    "cutaway_bars": 2,
    "angles_every": 3,        # every n-th event moment played side by side: the ground and the drone at that second
                              # (each moment once -- never a side-by-side slot repeating a dunk or out of order)
    "angles_bars": 3,
    "lineup_bars": 2,         # each side-by-side of speakers (their clean, mic-held stretches are short)
    "grid_bars": 3,           # everyone at once
    "aerial_bars": 2,         # an aerial between sections
    "aerial_share": 0.12,     # aerials' share of the film at most (the measured films: 3-7%; a drone-led shoot more)
    "bars": {"default": [2, 3, 3], "brisk": [1, 2, 2, 3], "event": [2]},   # a section's shots, by its kind (repeats
                                                                         # weigh a length)
    "brisk": [],              # categories cut briskly (arrivals, the celebration): 1-2 bars a shot
    "centre_weight": 1.0,     # how much a moment of people loses for its subject off the frame's middle
    "pace": {"median": [4.0, 7.0], "p10": 2.3, "per_min": [6, 14]},   # check(): the film's pace inside these
}


def settings(cfg) -> dict:
    d = dict(DEFAULTS)
    d.update(cfg.get("documentary") or {})
    return d


def hash01(*key) -> float:
    """A number in [0, 1) from the key, the same every run (romp's seeded variation: no random state)."""
    return int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16) / 2 ** 32


def bar_grid(music):
    """(beats per bar, the start beat index -- the first downbeat at or after the music's start, the bar count from
    there to the end)."""
    beats = np.array(music["beats"], float)
    bars = np.array(music.get("bars") or beats[::4], float)
    bpb = max(int(round(float(np.median(np.diff(bars))) / float(np.median(np.diff(beats))))), 1) if len(bars) > 2 else 4
    b0 = bars[bars >= music["start"] - 0.05]
    b0 = float(b0[0]) if len(b0) else float(bars[0])
    start = int(np.argmin(np.abs(beats - b0)))
    end = music["start"] + music["length"]
    n = int((end - beats[start]) / (bpb * float(np.median(np.diff(beats)))))
    return bpb, start, n


def tail_beats(music, bpb, start, n):
    """The beats left after n whole bars from the start beat to the song's end (the end card runs through them)."""
    beats = np.array(music["beats"], float)
    end = music["start"] + music["length"]
    return max(int(round((end - beats[start]) / float(np.median(np.diff(beats))))) - n * bpb, 0)


def snap_phrases(sizes, flexible, phrase=4):
    """Section sizes (bars) nudged so as many section starts as possible fall on phrase boundaries: a flexible
    section gives or takes a bar or two from the next flexible one; the total is kept."""
    sizes = list(sizes)
    for k in range(len(sizes) - 1):
        start = sum(sizes[:k + 1])
        off = start % phrase
        if not off or not flexible[k]:
            continue
        nxt = next((j for j in range(k + 1, len(sizes)) if flexible[j]), None)
        if nxt is None:
            continue
        d = -off if off <= phrase // 2 else phrase - off               # to the nearer boundary
        if sizes[k] + d >= 2 and sizes[nxt] - d >= 2:
            sizes[k] += d
            sizes[nxt] -= d
    return sizes


def split(n_bars, choices, seed, runs=None, bar_s=None):
    """n bars as shots of the allowed lengths (bars), varied by a seeded hash, the last taking what is left. runs
    (each clip's longest clean stretch, s) with bar_s: a length is used only while clips that hold it are left --
    footage whose steady stretches are short (a walk through the crowd) is cut in short shots."""
    out, left, k = [], n_bars, 0
    choices = sorted(int(c) for c in choices) or [2]
    pool = sorted(runs or [], reverse=True)

    def holds(c):                                                  # a clip left whose stretch holds c bars
        return runs is None or bar_s is None or len([r for r in pool if r >= c * bar_s - 0.3]) > \
            len([o for o in out if o >= c])
    while left > 0:
        fit = [c for c in choices if c <= left and holds(c)] or [c for c in choices if c <= left and c == min(choices)]
        if not fit:
            out[-1:] = [out[-1] + left] if out else [left]
            break
        c = fit[int(hash01(seed, k) * len(fit))]
        if 0 < left - c < min(choices):                               # never leave a stub shorter than allowed
            c = left
        out.append(c)
        left -= c
        k += 1
    return out


def structure(project, music, ctx):
    """The slots of a documentary cut, the beats each takes, and the card slot. ctx: what the casting knows -- a dict
    with "order" (ground categories in the order of the day), "aerial" (aerial categories, by when they flew:
    [(category, wall start)]), "event" (the event's category and its number of moments), "richness" ({category:
    usable seconds}), "lineups" ({category: (units, columns, grid)}), "title" (the reference's title, for the first
    slot), "card" (the reference's card slot), "last" (the category of the last shot)."""
    d = settings(project.config)
    bpb, start, total = bar_grid(music)
    bar_s = bpb * float(np.median(np.diff(np.array(music["beats"], float))))
    slots = []

    def slot(cat, bars, **kw):
        slots.append(dict(category=cat, bars=bars if bars != int(bars) else int(bars), length=round(bars * bar_s, 3),
                          **kw))

    ev_cat, ev_n = ctx.get("event") or (None, 0)
    air = [c for c, _w in sorted(ctx.get("aerial") or [], key=lambda x: x[1])]
    air_dict = dict(ctx.get("aerial") or [])
    order = [c for c in ctx["order"] if c != ev_cat] if ev_cat else list(ctx["order"])
    # where the event sits in the day: after the sections that happened before it
    ev_at = len([c for c in ctx["order"][: ctx["order"].index(ev_cat)] if c != ev_cat]) if ev_cat in ctx["order"] \
        else len(order)

    if d.get("event_s"):                                              # a moment's length in seconds (a dunk and its
        d = dict(d, event_bars=max(int(d["event_bars"]), int(np.ceil(float(d["event_s"]) / bar_s - 0.05))))   # rise,
                                                                      # whatever the song's tempo)
    # -- fixed parts -------------------------------------------------------------------------------------------------
    fixed = d["open_bars"] + d["last_bars"] + d["card_bars"]
    ev_shape = []
    if ev_cat and ev_n:
        ev_shape += [("lead", b) for b in d["lead_in"]]
        if d.get("breath"):                                            # one long, uncut, quiet shot before the first
            ev_shape.append(("breath", int(d["breath"].get("bars", 4)), "breath"))   # moment (reel.json "breath")
        blends = ctx.get("blends") or {}                               # a double exposure before a moment: the
        for k in range(ev_n):                                          # drone over it, slowed, and the person
            if k in blends:                                            # about to go under dissolving over it
                ev_shape.append(("blend", d.get("blend_bars", 3), blends[k]))
            both = bool(d["angles_every"]) and (k + 1) % d["angles_every"] == 0   # this one from the ground and
            eb = max(d["event_bars"], int(np.ceil(ctx["event_long"][k] / bar_s - 0.05))) \
                if k in (ctx.get("event_long") or {}) else d["event_bars"]   # (a longer moment: event_s_at)
            if k == 0:
                for pc, pb_, pf in (ctx.get("event_picks") or {}).get(-1, []):   # (an editor's pick before the first)
                    ev_shape.append(("pick", pb_, pc, pf))
            ev_shape.append(("event", eb, "angles") if both else ("event", eb))   # the drone side by side
            mine = (ctx.get("event_picks") or {}).get(k, [])  # the editor's b-roll shot after this moment: its
            for pc, pb_, pf in mine:                           # own slot, here (before the next moment), in place
                ev_shape.append(("pick", pb_, pc, pf))         # of the drone cutaway
            if d["cutaway_every"] and (k + 1) % d["cutaway_every"] == 0 and k + 1 < ev_n and not mine:
                ev_shape.append(("cutaway", d["cutaway_bars"]))
    lineup_shape = {}
    for c, (units, cols, grid) in (ctx.get("lineups") or {}).items():
        n_lin = int(np.ceil(units / max(cols, 1)))
        lineup_shape[c] = ([("grid", d["grid_bars"])] if grid and units > 3 else []) + \
            [("lineup", d["lineup_bars"])] * n_lin
    fixed += sum(x[1] for x in ev_shape) + sum(b for v in lineup_shape.values() for _k, b in v)
    n_air = max(len(order) - 1, 0) if air else 0
    air_cap = int(d["aerial_share"] * total) - d["open_bars"]
    n_air = max(0, min(n_air, air_cap // max(d["aerial_bars"], 1)))
    fixed += n_air * d["aerial_bars"]
    picks = sorted(ctx.get("pick_slots") or [], key=lambda x: x[1])      # the editor's picks: a slot each, where
    fixed += sum(x[2] for x in picks)                                     # they fall in the day (not a section's
                                                                          # shot taken)
    flex = [c for c in order if c not in lineup_shape]
    short = 2 * len(flex) - (total - fixed)                           # a short song: the event's moments give a
    for k, (kind, b, *_rest) in enumerate(ev_shape):                  # bar each first (never under two)
        if short <= 0:
            break
        if kind == "event" and b > max(2, int(d.get("event_floor", 2))) and not d.get("event_s"):   # (never under
                                                                          # its set length: a dunk with its rise)
            ev_shape[k] = (kind, b - 1)
            fixed -= 1
            short -= 1
    free = max(total - fixed, 2 * len(flex))
    # -- the rest by richness ---------------------------------------------------------------------------------------
    rich = {c: max(float((ctx.get("richness") or {}).get(c, 1.0)), 1.0) for c in flex}
    wsum = sum(np.sqrt(v) for v in rich.values()) or 1.0
    sizes = {c: max(2, int(round(free * np.sqrt(rich[c]) / wsum))) for c in flex}
    diff = free - sum(sizes.values())
    for c in sorted(flex, key=lambda c: -rich[c]):                     # the rounding to the richest
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        if sizes[c] + step >= 2:
            sizes[c] += step
            diff -= step

    clips = ctx.get("clips") or {}                                     # a section no longer than its footage: about
    cap = {c: max(2, int(d.get("bars_per_clip", 2) * clips[c])) for c in flex if c in clips}   # 2 bars a clip
    spare = sum(max(0, sizes[c] - cap[c]) for c in cap) + max(diff, 0)   # that holds a shot -- the rest to the
    for c in cap:                                                      # sections that have more
        sizes[c] = min(sizes[c], cap[c])
    roomy = [c for c in flex if c not in cap or sizes[c] < cap[c]]
    while spare > 0 and roomy:
        for c in sorted(roomy, key=lambda c: -rich[c]):
            if spare <= 0:
                break
            if c in cap and sizes[c] >= cap[c]:
                continue
            sizes[c] += 1
            spare -= 1
        roomy = [c for c in roomy if c not in cap or sizes[c] < cap[c]]
    for c, n in (d.get("section_bars") or {}).items():               # a project's own size for a section: the
        if c in sizes:                                                 # bars it takes from (or gives to) the rest
            spare += sizes[c] - int(n)
            sizes[c] = int(n)
    for c in sorted((x for x in sizes if x not in (d.get("section_bars") or {})), key=lambda x: -sizes[x]):
        while spare < 0 and sizes[c] > 2:
            sizes[c] -= 1
            spare += 1
    spare = max(spare, 0)
    # -- the running order -------------------------------------------------------------------------------------------
    seq = []                                                          # (kind, category, bars)
    seq.append(("open", air[0] if air else (order[0] if order else ev_cat), d["open_bars"]))
    sections = order[:ev_at] + ([ev_cat] if ev_cat else []) + order[ev_at:]
    used_air = 0
    ends_ = ctx.get("ends") or {}
    for k, c in enumerate(sections):
        while picks and c != ev_cat and ends_.get(c) is not None and picks[0][1] < ends_[c]:
            pc, _w, pb, pf = picks.pop(0)                              # an editor's pick before the first scene still
            seq.append(("pick", pc, pb, pf))                           # going on when it was shot
        if k and used_air < n_air and air:                             # an aerial between scenes: the flight
            starts = ctx.get("starts") or {}                            # nearest in time to the scene it opens
            a = min(air_dict, key=lambda x: abs(air_dict[x] - starts.get(c, air_dict[x]))) if starts else air[-1]
            seq.append(("aerial", a, d["aerial_bars"]))
            used_air += 1
        if c == ev_cat:
            over = ctx.get("over") or ([air[-1]] if air else [])         # the drone over the event
            for kind, b, *extra in ev_shape:
                if kind == "pick":                                     # (the editor's own, of its own category)
                    seq.append(("pick", extra[0], b, extra[1]))
                    continue
                if kind == "breath":
                    seq.append(("breath", over[0] if over else ev_cat, b))
                    continue
                seq.append((kind, over[0] if kind in ("cutaway", "blend") and over else ev_cat, b, *extra))
        elif c in lineup_shape:
            for kind, b in lineup_shape[c]:
                seq.append((kind, c, b))
        elif sizes.get(c, 2) > 0:                                      # (a section set to 0 bars: none at all --
            seq.append(("section", c, sizes.get(c, 2)))                # its picks only, a phrase snap won't grow it)
    for pc, _w, pb, pf in picks:                                       # (any left: before the close)
        seq.append(("pick", pc, pb, pf))
    if d.get("photo_grid"):                                            # the editor's photos as a grid of spatial
        seq.append(("photos", "photos", int(d["photo_grid"].get("bars", 3))))   # scenes, before the close
    seq.append(("last", ctx.get("last") or sections[-1], d["last_bars"]))
    if spare > 0:                                                      # bars no section has footage for: the drone
        limits = (("aerial", 3), ("cutaway", 3), ("open", 6), ("event", d["event_bars"] + 1), ("section", 99))
        for kind, most in limits:                                      # between scenes first, then the cutaways,
            for k, x in enumerate(seq):                                # the title, the event's moments
                if spare <= 0:
                    break
                if x[0] == kind and x[2] < most:
                    add = min(most - x[2], spare)
                    seq[k] = (x[0], x[1], x[2] + add, *x[3:])
                    spare -= add
    # section starts on phrases: the flexible sections give or take
    sizes_list = [x[2] for x in seq]
    flexible = [x[0] == "section" for x in seq]
    sizes_list = snap_phrases(sizes_list, flexible)
    seq = [(x[0], x[1], b, *x[3:]) for x, b in zip(seq, sizes_list, strict=True)]

    # slots a build before found nothing for (build()) leave before anything is pinned to the music
    drop, seen, kept, keys = set(d.get("drop") or []), {}, [], []
    for x in seq:
        key = f"{x[0]}:{x[1]}:{seen.get((x[0], x[1]), 0)}"
        seen[(x[0], x[1])] = seen.get((x[0], x[1]), 0) + 1
        if key not in drop:
            kept.append(x)
            keys.append(key)
    seq = kept
    # sections that start where the music changes (reel.json "documentary" "section_starts": {"<category>": s}):
    # the flexible entries before it give or take the bars, the ones after it take or give them back
    for cat, sec in sorted((d.get("section_starts") or {}).items(), key=lambda kv: kv[1]):
        want = round(float(sec) / bar_s)
        at = next((k for k, x in enumerate(seq) if x[1] == cat and x[0] in ("section", "event", "grid", "lineup",
                                                                             "lead", "aerial")), None)
        if at is None:
            continue
        diff = want - sum(x[2] for x in seq[:at])
        flex_ = [k for k in range(at) if seq[k][0] in ("section", "aerial")]
        after = [k for k in range(at, len(seq)) if seq[k][0] in ("section", "aerial", "cutaway")]
        for ks, sign in ((flex_[::-1], 1), (after if d.get("pin_balance") else [], -1)):   # (by default what
                                                                      # follows just moves: the card gives or takes)
            left = diff
            for k in ks:
                if left == 0:
                    break
                x = seq[k]
                step = max(left, 1 - x[2]) if left < 0 else left       # (never under a bar)
                if sign < 0:
                    step = max(-left, 1 - x[2]) if left > 0 else -left
                seq[k] = (x[0], x[1], x[2] + step, *x[3:])
                left -= step if sign > 0 else -step
    # -- the shots ---------------------------------------------------------------------------------------------------
    seq = [(*x[:3], key, *x[3:]) for x, key in zip(seq, keys, strict=True)]
    for k, c, b, key, *extra in seq:
        n0 = len(slots)
        if k == "blend":
            slot(c, b, blend=extra[0], scale="aerial")
            continue
        if k == "pick":
            slot(c, b, pick_for=extra[0])
            continue
        if k == "section":
            kinds = d["bars"].get(c) or (d["bars"]["brisk"] if c in d["brisk"] else d["bars"]["default"])
            for n in split(b, kinds, (c, b), (ctx.get("runs") or {}).get(c), bar_s):
                slot(c, n)
        elif k == "event":
            slot(c, b, dunk_slot=True, **({"angles": True} if extra and extra[0] == "angles" else {}))
        elif k == "angles":
            slot(c, b, split=[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0]])
        elif k == "open":
            slot(c, b, title=ctx.get("title"), scale="aerial")
        elif k in ("aerial", "cutaway"):
            slot(c, b, scale="aerial")
        elif k == "photos":
            slot(c, b, photos=True)
        elif k == "breath":
            slot(c, b, breath=True, scale="aerial")
        else:                                                          # lead, cutaway, grid, lineup, last
            slot(c, b)
        for x in slots[n0:]:
            x["seqkey"] = key
    t = 0.0
    for i, s in enumerate(slots):
        s.update(i=i, t0=round(t, 3), t1=round(t + s["length"], 3))
        if not s.get("title"):
            s.pop("title", None)
        t += s["length"]
    card = None
    if ctx.get("card") is not None:
        card = dict(ctx["card"], category=None, bars=d["card_bars"], length=round(d["card_bars"] * bar_s, 3),
                    i=len(slots), t0=round(t, 3), t1=round(t + d["card_bars"] * bar_s, 3))
    beats_per = [int(round(s["bars"] * bpb)) for s in slots] + (   # (a montage's picks: whole beats, not bars)
        [card["bars"] * bpb + tail_beats(music, bpb, start, total)] if card else [])   # (the card to the last note)
    return slots, card, beats_per, dict(bpb=bpb, start=start, bars=total, bar_s=round(bar_s, 4))


def check(plan, music, cfg=None):
    """The timing lint for a placed plan: the film's pace against the documentary's (median shot, its 10th
    percentile, cuts a minute), cuts off the song's bar lines (each cut `lead` ahead of its downbeat), and shots shorter than they need (an event moment under two
    bars). Returns {"pace": {...}, "issues": [(shot, kind, detail)]}."""
    d = settings(cfg or {})
    shots = [s for s in plan["shots"] if not s.get("card")]
    L = np.array([s["length"] for s in shots]) if shots else np.zeros(1)
    beat = float(music.get("beat") or np.median(np.diff(music["beats"])))
    bpb = bar_grid(music)[0]
    bar = beat * bpb
    pace = dict(median=round(float(np.median(L)), 2), p10=round(float(np.percentile(L, 10)), 2),
                per_min=round(60 * len(L) / max(float(L.sum()), 1e-6), 1))
    issues = []
    lo, hi = d["pace"]["median"]
    if not lo <= pace["median"] <= hi:
        issues.append((None, "pace", f"median shot {pace['median']} s outside {lo}-{hi}"))
    if pace["p10"] < d["pace"]["p10"]:
        issues.append((None, "pace", f"10th percentile {pace['p10']} s under {d['pace']['p10']}"))
    a, b = d["pace"]["per_min"]
    if not a <= pace["per_min"] <= b:
        issues.append((None, "pace", f"{pace['per_min']} cuts a minute outside {a}-{b}"))
    lines = np.array(music.get("bars") or [], float)
    t0 = plan.get("music_start", 0.0)                                # (song time = reel time + music_start)
    lead = float(d["lead"])
    on = []
    for k, s in enumerate(shots):
        if len(lines):
            at = s["rec_in"] + t0 + lead                               # the bar line the cut stands for
            off = float(np.min(np.abs(lines - at)))
            on.append(off < 0.06)
            if off >= 0.06:
                issues.append((k, "bars", f"cut at {s['rec_in']:.2f} s is {off:.2f} s off a bar line"))
        if s.get("dunk") and s["length"] < 2 * bar - 0.05:
            issues.append((k, "unread", f"an event moment held {s['length']:.2f} s, under two bars"))
        if s.get("hard_fail"):
            issues.append((k, "rule", f"{s['unit']['id']} {s['hard_fail']}"))
    if on:
        pace["on_bars"] = round(float(np.mean(on)), 2)
    return dict(pace=pace, issues=issues)
