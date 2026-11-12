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
