"""The last pass over a plan: the editor's rules on what the planner chose (reel/validate.py)."""
from focalcut.reel import validate as V

R = dict(V.RULES)
COUPLE = dict(id="C1-01_couple", role="dance", clip="C1", tag="segments", category="segments")
BAND = dict(id="C2_band", role="transition", clip="C2", tag="band", category="broll")
CROWD = dict(id="C3_crowd", role="transition", clip="C3", tag="crowd", category="broll")


def person(**kw):
    return dict(i=0, seen=1.0, off=0.0, face=1.0, sharp=30.0, size=1.0, h=0.8, cut=0.0, leaving=0.0, edge=0.0,
                hands=False, act=1.0, front=1.0) | kw


def shot(sub=None, mate=None, **kw):
    return dict(n=1, sub=sub, mate=mate, halved=0.0, sharper=None, group=False, cw=0.3164, z=1.0) | kw


def names(why):
    return [k for k, _ in why]


def test_a_good_shot_passes():
    assert V.check(shot(person(), person(i=1, size=0.9)), COUPLE, R) == []
    assert V.check(shot(person()), BAND, R) == []


def test_scenery_is_not_judged_on_people():
    assert V.check(shot(None), CROWD, R) == []
    assert V.check(shot(person(face=0.0, sharp=5.0)), CROWD, R) == []


def test_only_a_foreground_object_is_sharp():
    # the camera focused on a stand: the player is soft, whatever else in the frame is sharp
    assert names(V.check(shot(person(sharp=5.3)), BAND, R)) == ["soft"]
    # a dancer on a dark floor reads lower than a lit player and still passes
    assert V.check(shot(person(sharp=7.0), person(i=1)), COUPLE, R) == []


def test_the_camera_is_on_someone_else():
    assert names(V.check(shot(person(sharp=12.0), sharper=2.4), BAND, R)) == ["focus"]


def test_a_back_fails_performers_too():
    assert names(V.check(shot(person(face=0.0)), BAND, R)) == ["back"]
    # a couple mid-turn: the partner's face shows and they are in the clear
    assert V.check(shot(person(face=0.1), person(i=1, face=0.9, size=1.0)), COUPLE, R) == []
    # the nearer partner turned away with the other half hidden behind them
    assert names(V.check(shot(person(face=0.38), person(i=1, face=0.62, size=0.75, front=0.55)), COUPLE, R)) == ["back"]


def test_partner_out_of_frame():
    assert names(V.check(shot(person(), person(i=1, seen=0.7, leaving=0.33)), COUPLE, R)) == ["partner"]


def test_subject_out_of_the_strip():
    assert names(V.check(shot(person(seen=0.4)), BAND, R)) == ["out"]
    assert names(V.check(shot(person(seen=0.5, edge=0.5)), BAND, R)) == ["out"]
    assert names(V.check(shot(None), BAND, R)) == ["nobody"]


def test_halved_and_head_cut():
    assert names(V.check(shot(person(), halved=0.7), BAND, R)) == ["halved"]
    assert V.check(shot(person(), halved=0.7, group=True), BAND, R) == []
    assert names(V.check(shot(person(cut=1.0)), BAND, R)) == ["headcut"]
    assert V.check(shot(person(cut=1.0, hands=True, face=0.0)), BAND, R) == []     # hands at work: no head to cut


def test_a_far_off_couple_fails():
    assert names(V.check(shot(person(h=0.38), person(i=1, h=0.36, size=0.9), z=1.26), COUPLE, R)) == ["far"]
    assert V.check(shot(person(h=0.55), person(i=1, h=0.5, size=0.9), z=1.225), COUPLE, R) == []


def test_broll_clip_at_most_twice():
    def s(i, unit=CROWD, **kw):
        return dict(unit=unit, rec_in=30.0 * i, src_in=10.0 * i, rank=1.0 - 0.1 * i, fx=[], act="night") | kw
    shots = [s(0), s(1), s(2), s(3), s(4, unit=COUPLE)]
    assert V.overused(shots, R) == [2, 3]
    # the editor's own shots of the clip count and stay
    shots = [s(0), s(1, span=[1, 2]), s(2, span=[3, 4]), s(3)]
    assert V.overused(shots, R) == [0, 3]
    # two of one clip close together in the reel: one goes
    shots = [s(0), dict(s(1), rec_in=5.0)]
    assert V.overused(shots, R) == [1]


def test_rules_off_and_overridden():
    assert V.rules({"validate": False}) is None
    assert V.rules({"validate": {"close": 0.5}})["close"] == 0.5
    assert V.rules({})["broll_max"] == 2
    assert V.cap(R, CROWD) == 2 and V.cap(R, BAND) == 4          # performers' footage gets more room than scenery
