"""Shaky footage is avoided, never stabilised: the planner's own shots move to their clip's steadiest stretch, a
trimmed pick plays the steadiest part of what the editor picked, and the editor's picks stay where they are."""
import numpy as np

from focalcut.reel import plan, shake

T = np.arange(0, 30, 0.1)
J = np.where((T >= 10) & (T < 20), 0.0005, 0.01)            # steady 10-20 s, shaky either side


def _shot(src_in, length=2.0, **kw):
    u = dict(id="C1_a", score="C1", t0=0.0, t1=30.0, role="transition", bans=[], source="C1.MP4")
    return dict(unit=u, t=src_in + 1.0, src_in=src_in, peak=1.0, length=length, fx=[], kind="crowd", **kw)


def test_best_part_takes_the_steady_stretch_of_a_pick():
    a = plan.best_part(None, 4.0, 16.0, 2.0, (T, J))
    assert 10.0 - 1e-6 <= a <= 14.0
    assert plan.best_part(None, 4.0, 16.0, 2.0) == 4.0              # no shake measured: the editor's in-point
    assert plan.best_part(None, 11.0, 19.0, 2.0, (T, J)) == 11.0    # all steady: the editor's in-point


def test_planner_shots_move_to_steady_footage_and_picks_stay(monkeypatch):
    monkeypatch.setattr(shake, "series", lambda project, name: (T, J))
    mine, pick = _shot(2.0), _shot(24.0, span=[24.0, 26.0], add=True)
    cfg = dict(shake=[0.0012, 0.006])
    assert plan.steady_shots(None, "landscape", [mine, pick], [], cfg, W={}) == 1
    assert 10.0 - 1e-6 <= mine["src_in"] <= 18.0 and mine["t"] - mine["src_in"] == 1.0
    assert pick["src_in"] == 24.0
    assert not plan.editors(mine) and plan.editors(pick)


def test_steady_keeps_clear_of_bans_and_other_shots(monkeypatch):
    monkeypatch.setattr(shake, "series", lambda project, name: (T, J))
    mine, other = _shot(2.0), _shot(10.0, span=[10.0, 14.0], add=True, length=4.0)
    mine["unit"]["bans"] = [[15.0, 17.0]]
    other["unit"] = mine["unit"]
    plan.steady_shots(None, "landscape", [mine, other], [], dict(shake=[0.0012, 0.006]), W={})
    a = mine["src_in"]
    assert 17.0 - 1e-6 <= a <= 18.0                          # not on the pick (+ its gap), not on the ban


def test_stabilise_all_reaches_prestabilised_sources():
    import copy
    shots = [dict(jitter=0.004, stab=False, unit=dict(source="C2037_stabilized.mov")),
             dict(jitter=0.0005, stab=False, unit=dict(source="C2021.MP4")),
             dict(jitter=0.004, stab=True, unit=dict(source="C2021.MP4"))]
    for mode, want in ((None, [False, False, False]), (True, [False, False, True]), ("all", [True, False, True])):
        got = copy.deepcopy(shots)
        plan.stabilise(got, dict(stabilise=mode, shake=[0.0012, 0.006]))
        assert [s["stab"] for s in got] == want


def test_calm_shots_glide_in_landscape_only_where_the_music_is_calm():
    seq = [dict(act="teaser", beats=4, fx=[], unit=dict(role="transition")),
           dict(act="night", beats=2, fx=[], unit=dict(role="dance"), span=[1, 2], add=True),
           dict(act="night", beats=1, fx=[], unit=dict(role="dance")),
           dict(act="climax", beats=4, fx=[], unit=dict(role="dance")),
           dict(act="climax", beats=4, fx=["bullet"], unit=dict(role="dance")),
           dict(act="finale", beats=4, fx=[], unit=dict(role="dance"), ending=True)]
    beats = list(np.arange(40) * 0.5)
    sec = ["intro"] * 7 + ["drop"] * 33
    music = dict(beats=beats, start=0.0, beat_section=sec, beat_drive=[0.1] * 7 + [0.9] * 33)
    assert plan.glide_calm(seq, music) == 2
    assert [s["fx"] for s in seq] == [["glide"], ["glide"], [], [], ["bullet"], []]


def test_glide_is_a_steady_gentle_slow_motion():
    from focalcut import fx
    g = fx.EFFECTS["glide"]()
    ctx = fx.Context(fps=50, beat=0.5, peak=0.5, seed=0, params={})
    assert abs(g.source_length(2.0, ctx) - (1.6 + 2 / 50)) < 0.03


def test_clean_plan_keeps_the_cut_and_drops_what_was_added():
    from focalcut.reel import render
    plan = dict(stinger=3.0, shots=[
        dict(src_in=1.0, length=2.0, rec_in=0.0, crop=dict(w=0.3), fx=["glide"], effect="glide", logo="in",
             transition="flash", lift=dict(window=4), speed=1.15, unit=dict(id="a")),
        dict(src_in=5.0, length=1.0, rec_in=2.0, crop=None, fx=["bullet"], effect="bullet", title="x.png",
             unit=dict(id="b"))])
    c = render.clean_plan(plan)
    assert [(s["src_in"], s["length"], s["rec_in"]) for s in c["shots"]] == [(1.0, 2.0, 0.0), (5.0, 1.0, 2.0)]
    assert c["shots"][0]["crop"] == dict(w=0.3)
    assert all(not s["fx"] and not any(s.get(k) for k in render.CLEAN_DROP) for s in c["shots"])
    assert c["stinger"] == 0.0
