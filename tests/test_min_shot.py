"""reel.json plan.<fmt> "min_shot": no shot on screen shorter than it."""
import numpy as np

from focalcut.reel import plan as PL

BEAT = 0.458                                     # 131 BPM: one beat is under half a second


def shot(uid, beats, act="shows", span=None, whole=None, **kw):
    s = dict(unit=dict(id=uid, clip=uid.split("_")[0], role="dance", source=f"{uid}.mp4", t0=0.0, t1=60.0),
             beats=beats, act=act, fx=[], effect=None, rank=1.0, t=1.0, kind="pattern")
    if span:
        s.update(span=list(span), add=True)
    if whole:
        s["whole"] = list(whole)
    s.update(kw)
    return s


def test_least_beats_rounds_min_shot_up_to_whole_beats_and_is_off_by_default():
    assert PL.least_beats({}, BEAT) == 1
    assert PL.least_beats({"min_shot": 0}, BEAT) == 1
    assert PL.least_beats({"min_shot": 0.9}, BEAT) == 2
    assert PL.least_beats({"min_shot": 0.916}, BEAT) == 2       # exactly two beats is enough
    assert PL.least_beats({"min_shot": 1.0}, BEAT) == 3
    with PL.at_least({"min_shot": 0.9}, BEAT) as n:
        assert n == PL.LEAST == 2
        with PL.at_least({}, BEAT):                              # (nested: an inner pass keeps the floor)
            assert PL.LEAST == 2
    assert PL.LEAST == 1


def test_regroup_joins_short_lengths_into_a_neighbour_and_keeps_the_sum():
    assert PL.regroup([2, 1, 1, 2, 2], 2) == [2, 2, 2, 2]
    assert PL.regroup([1, 1, 1, 1, 2, 2], 2) == [2, 2, 2, 2]
    assert PL.regroup([3, 1, 2, 2], 2) == [3, 3, 2]
    assert PL.regroup([1, 1, 2, "H", 1, 3], 2) == [2, 2, "H", 4]     # nothing joins across a slow-motion slot
    assert PL.regroup([2, 1, 1, 2, 2], 1) == [2, 1, 1, 2, 2]
    for p in [x for v in PL.PHRASES.values() for x in v]:
        q = PL.regroup(p, 3)
        assert sum(q) == sum(p) and min(q) >= 3


def test_rhythm_uses_no_phrase_with_a_shot_under_min_shot():
    energy = list(np.linspace(0.6, 1.6, 8))
    feel = [None, "drop", None, None, "drop", None, None, None]
    off = PL.rhythm(68, "fast", energy, build=True, feel=feel)
    assert 1 in off and sum(off) == 68                          # (the default: quick 1-beat cuts)
    with PL.at_least({"min_shot": 0.9}, BEAT):
        on = PL.rhythm(68, "fast", energy, build=True, feel=feel)
        assert min(on) >= 2 and sum(on) == 68
        assert len(set(on)) > 1                                  # not one length over and over
    assert PL.rhythm(68, "fast", energy, build=True, feel=feel) == off   # deterministic, and off again outside


def test_quick_cuts_are_at_least_min_shot_each():
    def reel():
        return [shot("C1", 4, act="night", span=[0, 12]), shot("C2", 4, act="night", span=[0, 12]),
                shot("C3", 4, act="night"), shot("C4", 4, act="night", span=[20, 30]), shot("C5", 4, act="night")]
    seq = reel()
    PL.quick_cuts(seq, 5, "night", BEAT, at=3)
    assert 1 in [x["beats"] for x in seq if x.get("reuse")] and sum(x["beats"] for x in seq) == 25
    seq = reel()
    with PL.at_least({"min_shot": 0.9}, BEAT):
        PL.quick_cuts(seq, 5, "night", BEAT, at=3)
    cuts = [x["beats"] for x in seq if x.get("reuse")]
    assert min(cuts) >= 2 and sum(x["beats"] for x in seq) == 20 + 5    # fewer cuts for the same beats


def test_hold_min_joins_a_planners_short_shot_to_a_neighbour_on_the_grid():
    seq = [shot("A", 3), shot("B", 1), shot("C", 4, span=[10, 11.9]), shot("D", 1, act="night"), shot("E", 2, act="night")]
    total = sum(x["beats"] for x in seq)
    grown, merged = PL.hold_min(seq, BEAT, 2)
    assert min(x["beats"] for x in seq) >= 2 and sum(x["beats"] for x in seq) == total
    assert [x["unit"]["id"] for x in seq] == ["A", "C", "E"] and merged == 2 and grown == 0
    assert seq[0]["beats"] == 4                                  # B into the planner's A, not into the pick C
    assert seq[1]["span"] == [10, 11.9] and seq[1]["beats"] == 4  # the pick as it was: never past its out-point


def test_hold_min_holds_a_short_pick_longer_from_its_own_in_out_and_drops_one_too_short():
    seq = [shot("A", 5), shot("P", 1, span=[20, 20.5], whole=[19, 22]), shot("B", 3)]
    grown, merged = PL.hold_min(seq, BEAT, 2)
    p = seq[1]
    assert grown == 1 and merged == 0 and p["beats"] == 2 and [x["beats"] for x in seq] == [4, 2, 3]
    assert p["span"][0] >= 19 and p["span"][1] <= 22 and p["span"][0] + 2 * BEAT <= 22 + 1e-6   # inside the in-out
    assert not p["fx"]                                           # never sped up
    seq = [shot("A", 5), shot("Q", 1, span=[20, 20.5]), shot("B", 3)]   # half a second picked: can't fill 2 beats
    assert PL.hold_min(seq, BEAT, 2) == (0, 1)
    assert [x["unit"]["id"] for x in seq] == ["A", "B"] and sum(x["beats"] for x in seq) == 9


def test_hold_min_gives_a_flash_effect_shot_its_beats_too():
    seq = [shot("A", 4), shot("F", 1, fx=["flashin"], effect="flashin"), shot("B", 4)]
    PL.hold_min(seq, BEAT, 2)
    assert all(x["beats"] >= 2 for x in seq) and sum(x["beats"] for x in seq) == 9


def test_hold_min_is_off_without_min_shot():
    seq = [shot("A", 3), shot("B", 1), shot("C", 1, span=[0, 0.4])]
    assert PL.hold_min(seq, BEAT) == (0, 0) and [x["beats"] for x in seq] == [3, 1, 1]


def test_fit_pick_leaves_a_pick_too_short_for_min_shot_out_of_its_slot():
    seq = [shot("A", 4), shot("P", 1, span=[5, 5.4]), shot("B", 4)]
    with PL.at_least({"min_shot": 0.9}, BEAT):
        PL.fit_pick(seq, 1, BEAT)
    assert [x["unit"]["id"] for x in seq] == ["A", "B"] and sum(x["beats"] for x in seq) == 9
    seq = [shot("A", 4), shot("P", 1, span=[5, 6.0]), shot("B", 4)]   # one second: it plays 2 beats
    with PL.at_least({"min_shot": 0.9}, BEAT):
        PL.fit_pick(seq, 1, BEAT)
    assert [x["beats"] for x in seq] == [3, 2, 4]
