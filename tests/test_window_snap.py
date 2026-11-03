"""Every shot the planner chose plays on windows the scrub passed (plan.snap_windows), whichever pass placed it;
the editor's picks and the ending stay where they were put; the plan log warns about any left (window_report)."""
import pytest

from focalcut.reel import plan


def win(t0, t1, ok):
    return dict(t0=t0, t1=t1, ok=ok, score=0.5, rank=0.5)


def test_span_ok_by_the_covering_windows():
    W = [win(1.0, 2.2, False), win(1.0, 3.4, False), win(5.0, 6.2, True)]
    assert plan.span_ok(W, 1.2, 2.0) is False                 # every covering window fails
    assert plan.span_ok(W, 5.2, 6.0) is True
    assert plan.span_ok(W, 20.0, 21.0) is None                # nothing scrubbed there
    assert plan.span_ok(None, 1.0, 2.0) is None               # scenery: no windows at all


def test_a_long_shot_passes_when_passing_windows_tile_it():
    W = [win(t, t + 3.6, True) for t in (10.0, 12.0, 14.0)]
    assert plan.span_ok(W, 10.0, 17.6) is True
    W[1]["ok"] = W[2]["ok"] = False
    assert plan.span_ok(W, 10.0, 17.6) is False


def unit(i, cat="night"):
    return {"id": i, "clip": i, "role": "dance", "category": cat, "t0": 0.0, "t1": 60.0, "source": i + ".MP4"}


def placed(i, src, **kw):
    return dict(unit=unit(i), fx=[], beats=4, length=1.8, beat=0.45, rank=0.5, t=src + 0.7, peak=0.7, kind="spin",
                act="night", src_in=src, crop={"w": 0.32}, **kw)


def grid(a, b, bad=()):
    return [win(round(t, 2), round(t + L, 2), not any(x <= t < y or x < t + L <= y for x, y in bad))
            for L in (1.2, 2.4) for t in [a + 0.5 * k for k in range(int((b - a) / 0.5))] if t + L <= b]


@pytest.fixture
def fakes(monkeypatch):
    from focalcut.reel import shake
    monkeypatch.setattr(plan, "reframe", lambda pr, u, t0, L, fmt: {"w": 0.32, "t0": t0})
    monkeypatch.setattr(shake, "jitter", lambda pr, u, a, b: 0.0001)


def rec(i, ts, ok=True, cat="night"):
    return {"id": i, "role": "dance", "orient": "landscape", "unit": unit(i, cat),
            "moments": [dict(t=t, kind="spin", score=0.4, score_p=0.4, fmt_ok={"portrait": ok}) for t in ts]}


CFG = dict(shake=[0.0012, 0.006])


def test_a_planner_shot_on_failing_windows_slides_onto_passing_ones(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(8.0, 14.0)]), "B": grid(0.3, 40.0)}
    shots = [placed("B", 2.0), placed("A", 10.0), placed("B", 20.0)]
    slid, recast, stuck = plan.snap_windows(None, "portrait", shots, [rec("A", [30.0]), rec("B", [5.0])], CFG, W)
    s = shots[1]
    assert (slid, recast, stuck) == (1, 0, [])
    assert s["unit"]["id"] == "A" and plan.span_ok(W["A"], s["src_in"], s["src_in"] + s["length"])
    assert abs(s["src_in"] - 10.0) < 6.0                      # the nearest passing stretch, not anywhere
    assert s["peak"] == 0.7 and s["t"] == pytest.approx(s["src_in"] + 0.7)   # the move moves with it
    assert s["crop"]["t0"] == s["src_in"]                     # reframed over what it now plays
    assert plan.window_fails(shots, W) == []


def test_picks_the_ending_and_the_opening_stay_where_they_were_put(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(0.0, 40.0)])}
    shots = [placed("A", 3.0, open=True), placed("A", 10.0, span=[10.0, 11.8]), placed("A", 20.0, ending=True)]
    assert plan.snap_windows(None, "portrait", shots, [rec("A", [30.0], ok=False)], CFG, W) == (0, 0, [])
    assert [s["src_in"] for s in shots] == [3.0, 10.0, 20.0]
    assert [w for *_, w in plan.window_fails(shots, W)] == ["opening", "pick", "ending"]


def test_a_people_shot_is_the_planners_not_a_pick(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(8.0, 14.0)])}
    shots = [placed("A", 10.0, add=True, person=True)]
    assert [w for *_, w in plan.window_fails(shots, W)] == [None]
    assert plan.snap_windows(None, "portrait", shots, [rec("A", [30.0])], CFG, W)[0] == 1


def test_a_unit_with_no_passing_window_gives_the_shot_to_another_unit(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(0.0, 40.0)]), "C": grid(0.3, 40.0), "D": grid(0.3, 40.0)}
    shots = [placed("C", 2.0), placed("A", 10.0), placed("C", 20.0)]
    moments = [rec("A", [10.7], ok=False), rec("C", [30.0]), rec("D", [15.0]), rec("E", [15.0], cat="day")]
    slid, recast, _ = plan.snap_windows(None, "portrait", shots, moments, dict(CFG, day=["day"]), W)
    assert (slid, recast) == (0, 1)
    s = shots[1]
    assert s["unit"]["id"] == "D" and s["recast_window"]      # not C (beside its own clip), not E (day footage)
    assert plan.span_ok(W["D"], s["src_in"], s["src_in"] + s["length"])


def test_a_slide_keeps_clear_of_the_units_other_shots(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(8.0, 40.0)]), "B": grid(0.3, 40.0)}
    shots = [placed("A", 4.0), placed("B", 2.0), placed("A", 10.0)]
    plan.snap_windows(None, "portrait", shots, [rec("A", []), rec("B", [5.0])], CFG, W)
    a, b = shots[0], shots[2]
    assert b["unit"]["id"] != "A" or abs(b["src_in"] - a["src_in"]) >= a["length"] + plan.SNAP_GAP


def test_the_plan_log_warns_about_a_planner_shot_left_on_failing_windows(fakes, monkeypatch, capsys):
    W = {"A": grid(0.3, 40.0, bad=[(8.0, 14.0), (29.0, 33.0)])}
    monkeypatch.setattr(plan, "load_windows", lambda pr, fmt: W)
    plan_ = dict(shots=[placed("A", 10.0), placed("A", 30.0, span=[30.0, 31.8]), placed("A", 20.0)])
    lines = plan.window_report(None, "portrait", plan_)
    assert len(lines) == 1 and "shot 0 A" in lines[0]
    out = capsys.readouterr().out
    assert "WARNING windows" in out and "1/3 shots on passing windows (1 pick on failing ones" in out


def test_an_in_point_rounded_to_the_millisecond_still_sits_on_its_window():
    W = [win(57.3, 60.9, True), win(57.8, 61.4, False)]
    assert plan.span_ok(W, 57.996, 57.996 + 3.204) is True    # b - tol is 60.900000000000006


def test_a_recast_trusts_the_windows_over_assess_s_guess(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(0.0, 40.0)]), "C": grid(0.3, 40.0), "D": grid(0.3, 40.0)}
    shots = [placed("C", 2.0), placed("A", 10.0), placed("C", 20.0)]
    d = rec("D", [15.0])
    d["moments"][0]["unusable"] = "shake"                     # judged around the moment, not over this span
    _, recast, _ = plan.snap_windows(None, "portrait", shots, [rec("A", [10.7], ok=False), d], CFG, W)
    assert recast == 1 and shots[1]["unit"]["id"] == "D"


def test_a_unit_whose_scrubbed_moment_passes_is_covered_whatever_assess_guessed(fakes):
    shots = [placed("A", 1.0), placed("A", 5.0), placed("B", 2.0), placed("A", 9.0), placed("B", 6.0)]
    d = rec("D", [20.0])
    d["moments"][0].update(unusable="shake", plays_p=0.05)
    assert plan.ensure_units(None, "portrait", shots, [d], CFG) == ["D"]


def test_a_missing_unit_takes_a_slot_short_enough_for_its_passing_windows(fakes):
    long_, short = placed("A", 9.0), placed("A", 30.0)
    long_.update(length=2.7, beats=6)
    short.update(length=0.9, beats=2)
    shots = [placed("A", 1.0), long_, placed("B", 2.0), short, placed("B", 6.0)]
    W = {"D": [win(20.0, 21.2, True), win(19.0, 22.6, False)]}   # it passes only over 1.2 s
    assert plan.ensure_units(None, "portrait", shots, [rec("D", [20.5])], CFG, W) == ["D"]
    d = next(s for s in shots if s["unit"]["id"] == "D")
    assert d is short and plan.span_ok(W["D"], d["src_in"], d["src_in"] + d["length"])
