"""The picture follows the music: what the music is doing on each beat (music.py), hits that are real, pacing by
section, and holds on sustains."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from focalcut.reel import grammar as G
from focalcut.reel import music as M
from focalcut.reel import plan as P
from focalcut.reel import velocity as V

BEAT = 0.5


def song(sec, accent=None, held=()):
    """music.json for a song whose beats carry these sections: [(name, beats), ...]."""
    names = [n for n, k in sec for _ in range(k)]
    n = len(names)
    drive = [0.1 if s in ("intro", "break", "build") else 0.9 for s in names]
    return dict(beats=list(np.arange(n) * BEAT), bars=list(np.arange(0, n, 4) * BEAT), start=0.0, beat=BEAT,
                length=n * BEAT, end=n * BEAT, beat_energy=[1.0] * (n - 1),
                beat_accent=list(accent if accent is not None else np.ones(n)), beat_kick=drive, beat_drive=drive,
                beat_section=names, beat_sustain=[int(i in held) for i in range(n)])


def shot(act="night", beats=2, kind="shake", role="show", clip=None, **kw):
    shot.n = getattr(shot, "n", 0) + 1
    c = clip or f"C{shot.n}"
    return dict(kind=kind, act=act, beats=beats, fx=[], rank=0.5, t=10.0, beat=BEAT,
                unit=dict(id=c, clip=c, role=role, tag="", t0=0.0, t1=60.0), **kw)


# --- the analysis --------------------------------------------------------------------------------------------------
def test_sections_name_the_runs_of_driving_and_calm_beats():
    d = [0.1] * 16 + [0.8] * 16 + [0.15] * 16 + [0.95] * 32 + [0.6] * 12
    s = M.sections(d)
    assert [(v, b - a) for a, b, v in M._runs(s)] == [("intro", 16), ("groove", 16), ("break", 16), ("drop", 32),
                                                      ("outro", 12)]
    d[40] = 0.9                                              # one beat of kick in the break: still the break
    assert M.sections(d)[40] == "break"
    perc = [0.2] * 40 + [0.9] * 8 + [0.5] * 44               # a snare roll up to the drop: a build
    assert M.sections(d[:92], perc)[40:48] == ["build"] * 8
    assert set(M.sections([0.7] * 64)) == {"groove"}         # a band playing throughout (salsa): one groove


def test_sustains_are_held_beats_after_the_beat_that_sounded():
    held = [0, 0, 1, 1, 0, 0, 0, 1, 0] + [1] * 12
    assert M.sustains(held) == [[1, 2], [6, 1], [8, M.SUSTAIN_MAX]]
    assert M.sustains([1, 1, 0]) == []                       # (nothing sounded before it)


def test_feel_reads_the_reels_beats_and_an_old_music_json_drives_throughout():
    acc = np.ones(48); acc[[4, 20]] = 3.0
    m = song([("intro", 16), ("drop", 32)], acc, held=(21, 22))
    F = M.feel(m)
    assert F["known"] and F["hit"][4] == 0 and F["hit"][20] == 3.0       # an accent in the intro is not a hit
    assert F["blink"][:16].max() == 0 and F["blink"][16:].min() == 1     # no blink without a rhythm section
    assert F["sustain"] == [(20, 23)] and F["entries"] == [16]
    F = M.feel(dict(m, start=8 * BEAT))                                  # the reel starts 8 beats in
    assert F["hit"][12] == 3.0 and F["sustain"] == [(12, 15)] and F["entries"] == [8]
    old = {k: v for k, v in m.items() if k not in ("beat_kick", "beat_drive", "beat_section", "beat_sustain")}
    F = M.feel(old)
    assert not F["known"] and not F["calm"].any() and list(F["hit"]) == list(acc) and F["sustain"] == []
    assert F["blink"].min() == 1 and set(F["section"]) == {"groove"}


def test_features_hear_the_kick_the_break_and_a_held_note():
    pytest.importorskip("librosa")
    sr, n = 22050, 64
    bt = np.arange(n) * BEAT
    y = np.zeros(int((n * BEAT + 1) * sr), np.float32)
    t = np.arange(int(0.25 * sr)) / sr
    kick = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 18)).astype(np.float32)
    rng = np.random.default_rng(0)
    hat = (rng.normal(size=len(t)) * np.exp(-t * 90) * 0.15).astype(np.float32)
    tt = np.arange(len(y)) / sr
    y += (0.15 * np.sin(2 * np.pi * 440 * tt)).astype(np.float32)        # a pad throughout
    for i, b in enumerate(bt):
        a = int(b * sr)
        if 16 <= i < 32 or i >= 48:                                      # kick: beats 16-31 and from 48
            y[a:a + len(kick)] += kick
        if not 36 <= i < 40:                                             # hats on and between the beats, but for a
            for off in (0.0, 0.25):                                      # held bar at beats 36-39
                j = int((b + off) * sr)
                y[j:j + len(hat)] += hat
    f = M.features(y, sr, bt, 0.0, n * BEAT)
    runs = [(v, a, b) for a, b, v in M._runs(f["beat_section"])]
    assert [v for v, a, b in runs] == ["intro", "drop", "break", "drop"]   # (in as hard as the song gets: a drop)
    assert [a for v, a, b in runs] == [0, 16, 32, 48]
    assert max(f["beat_drive"][2:14]) < 0.3 and min(f["beat_drive"][18:30]) > 0.6
    assert [35, 4] in f["sustains"]                                      # beat 35 sounds, 36-39 are held
    assert not any(f["beat_sustain"][16:32])


# --- hits that are real --------------------------------------------------------------------------------------------
def test_no_flash_on_an_accent_in_a_breakdown_and_the_act_changes_on_a_dissolve():
    acc = np.ones(64); acc[[4, 52]] = 5.0                    # a sung syllable in the break, a hit in the drop
    m = song([("break", 32), ("drop", 32)], acc)
    seq = [shot(beats=4), shot(beats=4), shot(beats=28), shot(beats=4), shot(beats=12), shot(beats=12)]
    P.transitions(seq, m, P.CFG["landscape"])
    assert seq[1].get("transition") != "flash"               # at beat 4: an accent, no rhythm section
    assert seq[3].get("transition") is None and seq[4].get("transition") is None
    assert seq[5]["transition"] == "flash"                   # at beat 52: the same accent where the music drives
    seq = [shot("teaser", 8), shot("arrival", 8), shot("arrival", 24), shot("shows", 8)]
    P.transitions(seq, m, P.CFG["landscape"])
    assert seq[1]["transition"] == "fade" and "calm" in seq[1]["why"]    # teaser -> arrival: no dip to white
    assert seq[3]["transition"] == "zoom"                    # the signature, where the music drives
    old = {k: v for k, v in m.items() if not k.startswith("beat_") or k in ("beat_accent", "beat_energy")}
    seq = [shot("teaser", 8), shot("arrival", 8)]
    P.transitions(seq, old, P.CFG["landscape"])
    assert seq[1]["transition"] == "dipwhite"                # (a music.json without the description: as before)


def test_a_calm_passage_gets_a_dissolve_now_and_then_never_a_whip():
    m = song([("break", 48), ("drop", 16)])
    seq = [shot("shows", 4, kind="spin") for _ in range(16)]
    P.transitions(seq, m, P.CFG["landscape"], direction=lambda u, t: (0.3, 0.3))
    calm = [s.get("transition") for s in seq[1:12]]
    assert set(calm) == {None, "fade"} and 2 <= calm.count("fade") <= 3  # (spins both sides, all travelling right)
    at = [4 * (k + 1) for k, x in enumerate(calm) if x]
    assert all(b - a >= G.CALM_EVERY for a, b in pairwise(at))
    assert any(s.get("transition") in ("spin", "swipe") for s in seq[12:])   # the drop keeps its own


def test_a_looks_phrase_glitch_and_hit_accent_need_the_music_to_hit():
    acc = np.ones(64); acc[[8, 16, 40, 48]] = 3.0
    m = song([("break", 32), ("drop", 32)], acc)
    look = dict(pulse=8, transitions=["g_tear"], hits=[("night", "g_freeze", 4)])
    seq = [shot(beats=4, peak=0.0) for _ in range(16)]
    P.glitch_look(seq, m, look)
    cuts = [4 * i for i, s in enumerate(seq) if s.get("transition")]
    assert cuts and all(c >= 32 for c in cuts)               # none over the break, however hard its accents
    assert all(not s["fx"] for s in seq[:8])                 # and no freeze or punch there either


def test_flickers_and_blinks_need_a_beat_that_hits():
    pytest.importorskip("cv2")
    from focalcut.reel import overlay as O
    acc = np.ones(96); acc[[10, 70]] = 3.0
    m = song([("intro", 16), ("groove", 16), ("break", 28), ("build", 4), ("drop", 32)], acc)
    m["beat_energy"] = [1.0] * 95
    for i in (10, 70):                                       # two loud beats: one in the intro, one in the drop
        m["beat_energy"][i] = 4.0
    n = int(96 * BEAT * 25)
    Gd = O.grid(m, dict(music_start=0.0), n, 25)
    assert Gd["drops"] == [16, 64]                           # the rhythm section coming in
    assert Gd["drive"][:16 * 12].max() == 0 and Gd["drive"][17 * 13:31 * 12].min() == 1   # 12.5 frames a beat
    f = O.flickers(Gd, np.ones(n, np.float32), O.options(True), 25)
    assert not any(i in f for i in range(12, 16))            # into the first groove from a quiet intro: none
    assert [f.get(i) for i in range(60, 64)] == [O.WHITE, O.WHITE, O.WHITE, O.DARK]   # up the build
    assert (10 not in f and 70 not in f) or f.get(70)          # the loud beat in the intro: never
    assert 10 not in f
    m2 = dict(m, beat_section=[("break" if s == "build" else s) for s in m["beat_section"]])
    f = O.flickers(O.grid(m2, dict(music_start=0.0), n, 25), np.ones(n, np.float32), O.options(True), 25)
    assert not any(i in f for i in range(56, 64))            # a drop out of a vocal break: its own flash says it


# --- picture energy follows music energy ------------------------------------------------------------------------------
def test_rhythm_cuts_long_where_the_music_is_calm_and_quick_on_a_drop():
    e = [1.0] * 6
    lengths = P.rhythm(48, "fast", e, feel=["calm", "calm", None, "drop", "drop", "calm"])
    ph = [lengths[:0]]
    out, run = [], 0
    for x in lengths:                                        # back into phrases of 8
        ph[-1] = ph[-1] + [x]; run += x
        if run % 8 == 0:
            out.append(ph[-1]); ph.append([])
    assert all(p in P.PHRASES["calm"] for p in (out[0], out[1], out[5])) and out[0] != out[1]
    assert all(max(p) <= 2 for p in out[3:5])
    assert P.rhythm(48, "fast", e) == P.rhythm(48, "fast", e, feel=[None] * 6)   # nothing known: as before
    m = song([("intro", 16), ("groove", 16), ("drop", 16)])
    assert P.phrase_feel(m, 0, 6) == ["calm", "calm", None, None, "drop", "drop"]
    assert P.phrase_feel({k: v for k, v in m.items() if k != "beat_section" and k != "beat_drive"}, 0, 2) == [None] * 2


def test_pace_lengths_gives_calm_shots_the_beats_of_driving_ones():
    calm = [True] * 12 + [False] * 24
    L = [2] * 18
    out = P.pace_lengths(L, [2] * 18, [6] * 18, calm)        # everything at its floor: nothing to give
    assert out == L
    L = [2, 2, 2, 2, 2, 2] + [3] * 8
    out = P.pace_lengths(L, [2] * 14, [6] * 14, calm)
    assert sum(out) == sum(L)
    pos = P._starts(out)
    inside = [n for p, n in zip(pos, out, strict=True) if calm[p]]
    assert max(inside) == P.CALM_BEATS and sum(inside) > 6 and len(inside) < 6   # fewer, longer shots in the calm
    assert all(n >= 2 for n in out)
    b = [False] * 6 + [False, True] * 4                      # B-roll among the driving shots gives first
    out = P.pace_lengths(L, [2] * 14, [6] * 14, calm, give=lambda i: b[i])
    assert all(out[i] == 2 for i in range(6, 14) if b[i])
    L = [2] * 6 + [3, 2, 3, 3, 2, 3, 4, 3, 2, 3, 3, 2]       # rule 11: the driving shots' lengths still vary
    out = P.pace_lengths(L, [2] * 18, [6] * 18, calm)
    assert out != L and max(b - a for a, b, v in M._runs(out[6:])) < P.EVEN_RUN


# --- holds on sustains ----------------------------------------------------------------------------------------------
def test_no_cut_inside_a_sustain():
    # the music lands on beat 7 and holds until beat 10 sounds; cuts at 4, 8, 12
    L, stuck = P.hold_sustains([4, 4, 4, 4], [1] * 4, [9] * 4, [(7, 10)])
    assert L == [4, 3, 5, 4] and not stuck                   # the cut moves back onto the hit: the shot holds through
    L, stuck = P.hold_sustains([4, 5, 3, 4], [1] * 4, [9] * 4, [(7, 10)])
    assert L == [4, 6, 2, 4] and not stuck                   # (nearer the next beat that sounds: on to it)
    L, stuck = P.hold_sustains([4, 4, 4, 4], [1, 4, 4, 1], [9, 4, 4, 9], [(7, 10)])
    assert L == [4, 4, 4, 4] and stuck == [2]                # neither shot can change: left, and named
    L, stuck = P.hold_sustains([4, 4, 4, 4], [1] * 4, [9] * 4, [(7, 10)], locked=[2])
    assert stuck == [2]
    assert P.hold_sustains([4, 4, 4], [1] * 3, [9] * 3, [(4, 6), (8, 12)])[0] == [4, 4, 4]   # cuts on the hits stay
    # the shot before has no beat to give, the one before that does: the short shot moves whole
    L, stuck = P.hold_sustains([4, 4, 2, 4], [2] * 4, [4, 4, 2, 5], [(9, 11)])
    assert L == [4, 3, 2, 5] and not stuck


def test_the_drops_effect_waits_for_the_drop():
    m = song([("break", 8), ("drop", 24)])
    seq = [shot("night", 4)] + [shot("climax", 4, kind="spin") for _ in range(6)]
    assert P.drop_effect(seq, m) == 2                        # the climax opens over the break: its second shot
    assert seq[2]["fx"] == ["bullet"] and not seq[1]["fx"]


def test_hold_through_moves_only_what_may_move():
    seq = [shot(beats=4), shot(beats=4), shot(beats=4), shot(beats=4)]
    assert P.hold_through(seq, [(7, 10)], BEAT) == [] and [s["beats"] for s in seq] == [4, 3, 5, 4]
    # an editor's pick gives a beat and plays that much of its in-out, from its own in-point
    seq = [shot(beats=4), shot(beats=4, span=[10.0, 12.0], add=True), shot(beats=4), shot(beats=4)]
    assert P.hold_through(seq, [(7, 10)], BEAT) == [] and [s["beats"] for s in seq] == [4, 3, 5, 4]
    assert seq[1]["span"] == [10.0, 11.5] and seq[1]["whole"] == [10.0, 12.0]
    # ... and takes beats only up to its whole in-out: this one is whole already, so the shot before holds through
    seq = [shot(beats=4), shot(beats=5), shot(beats=3, span=[10.0, 11.5], add=True), shot(beats=4)]
    assert P.hold_through(seq, [(7, 10)], BEAT) == [] and [s["beats"] for s in seq] == [4, 6, 2, 4]
    # a shot with an effect keeps its length: the one before it gives, and it moves whole
    seq = [shot(beats=4), shot(beats=4), shot(beats=4), shot(beats=4)]
    seq[1]["fx"] = ["hero"]
    assert P.hold_through(seq, [(7, 10)], BEAT) == [] and [s["beats"] for s in seq] == [3, 4, 5, 4]
    seq = [shot(beats=4), shot(beats=4), shot(beats=4), shot(beats=4)]
    for s in seq[:3]:                                        # nothing can move: left, and named
        s["fx"] = ["hero"]
    assert P.hold_through(seq, [(7, 10)], BEAT) == [2] and [s["beats"] for s in seq] == [4, 4, 4, 4]
    seq = [shot(beats=4), shot("shows", 4), shot("night", 4), shot(beats=4)]
    assert P.hold_through(seq, [(7, 10)], BEAT) == [2]       # never across a change of act


def test_pace_through_keeps_each_acts_length_and_lets_broll_give_first():
    m = song([("break", 16), ("drop", 32)])
    seq = [shot("shows", 2) for _ in range(8)] + [shot("shows", 3) for _ in range(4)] + [shot("night", 3) for _ in range(4)]
    seq[9]["unit"]["role"] = "transition"
    seq[1]["fx"] = ["hero"]
    assert P.pace_through(seq, m, BEAT) > 0
    assert sum(s["beats"] for s in seq[:12]) == 28 and [s["beats"] for s in seq[12:]] == [3] * 4   # the acts keep theirs
    assert seq[9]["beats"] == 2 and seq[1]["beats"] == 2 and max(s["beats"] for s in seq[:6]) > 2
    old = {k: v for k, v in m.items() if k not in ("beat_drive", "beat_section")}
    assert P.pace_through(seq, old, BEAT) == 0


def test_a_velocity_edit_lands_on_the_sustain_and_a_landing_slides_onto_it():
    acc = [1.0, 1.2, 3.0, 1.0, 1.1, 1.0, 1.0, 1.0]
    assert V.landing(8, "hero", False, acc) == 2             # the hardest accent, as ever
    acc[4] += V.SUSTAIN                                      # the music lands on beat 4 and holds
    assert V.landing(8, "hero", False, acc) == 4
    # a 4 s shot, the dancer lands at 20.0 in the source, the music lands 1.5 s into the shot
    assert V.on_sustain(4.0, 20.0, 0.5, 1.5) == 18.5
    assert V.on_sustain(4.0, 20.0, 0.5, 0.2) is None         # (it would lose the end of the move)
    assert V.on_sustain(4.0, 20.0, 0.5, 3.8) is None         # (... or the rest after it)
