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
    W = [win(57.3, 60.9, True), win(58.8, 62.4, False)]
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


def test_with_song_sets_a_recast_takes_its_own_set_first_and_another_only_when_it_must(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(0.0, 40.0)]), "C": grid(0.3, 40.0), "D": grid(0.3, 40.0), "E": grid(0.3, 40.0)}
    shots = [placed("C", 2.0), placed("A", 10.0), placed("C", 20.0)]
    for x in shots:
        x["unit"]["category"] = "headline"
    moments = [rec("A", [10.7], ok=False, cat="headline"), rec("D", [15.0], cat="headline"), rec("E", [15.0], cat="other")]
    plan.snap_windows(None, "portrait", shots, moments, dict(CFG, sets=True), W)
    assert shots[1]["unit"]["id"] == "D"
    shots[1] = placed("A", 10.0)
    shots[1]["unit"]["category"] = "headline"
    _, recast, _ = plan.snap_windows(None, "portrait", shots, moments[:1] + moments[2:], dict(CFG, sets=True), W)
    assert recast == 1 and shots[1]["unit"]["id"] == "E"


def test_coverage_never_forces_in_a_unit_whose_best_window_is_weak(fakes):
    shots = [placed("A", 1.0), placed("A", 5.0), placed("B", 2.0), placed("A", 9.0), placed("B", 6.0)]
    W = {"D": [dict(win(20.0, 21.2, True), score=0.1)], "E": [dict(win(20.0, 21.2, True), score=0.9)]}
    assert plan.ensure_units(None, "portrait", shots, [rec("D", [20.5]), rec("E", [20.5])], CFG, W) == ["E"]


def test_the_last_shot_ends_before_its_subject_leaves_and_the_band_takes_the_rest(fakes):
    W = {"A": grid(0.3, 40.0, bad=[(13.6, 40.0)]),                    # the camera pans off from 13.6 s
         "BAND": grid(0.3, 40.0)}
    last = placed("A", 10.0)
    last.update(beats=12, length=12 * 0.45, beat=0.45, rec_in=100.0, frames=270)
    shots = [placed("B", 2.0), last]
    band = rec("BAND", [8.0])
    band.update(role="transition")
    band["unit"].update(role="transition", tag="band")
    fill = plan.end_clean(None, "portrait", shots, [rec("C", [5.0]), band], CFG, W)
    assert fill is shots[-1] and fill["unit"]["id"] == "BAND" and fill["end_fill"]
    assert last["beats"] + fill["beats"] == 12 and last["beats"] >= 2
    assert plan.span_ok(W["A"], last["src_in"], last["src_in"] + last["length"]) is not False
    assert last["frames"] + fill["frames"] == 270 and fill["rec_in"] == pytest.approx(100.0 + last["beats"] * 0.45)


def test_a_steady_crop_rides_out_a_one_frame_slice_but_follows_a_lasting_one():
    import numpy as np
    cand = np.linspace(0.15, 0.85, 15)
    cost = np.abs(cand[:, None] - 0.5) / 0.3 + np.zeros((1, 10))
    blip = cost.copy()
    blip[7, 9] += 2.5                                      # the centre slices someone for one frame
    assert np.abs(cand[plan.steady(blip, cand, 0.3)] - 0.5).max() <= 0.05 + 1e-9   # a nudge at most, not a jump
    lasting = cost.copy()
    lasting[7, 4:] += 2.5                                  # from frame 4 on: it moves off
    k = plan.steady(lasting, cand, 0.3)
    assert cand[k[-1]] != 0.5 and len(set(k)) <= 2         # off it, in one move at most


def test_a_panel_frames_the_speaker_facing_the_room_not_the_gap_between_them():
    import numpy as np
    F = 6
    kp = np.zeros((F, 2, 17, 3))
    boxes = np.zeros((F, 2, 5))
    for p, (x0, nose_dx) in enumerate(((100, 0.0), (400, -40.0))):   # 0 faces the room; 1 turns to them
        boxes[:, p, :4] = (x0, 50, x0 + 200, 350)
        cx = x0 + 100
        kp[:, p, 3:5, :] = [[cx - 30, 100, 0.9], [cx + 30, 100, 0.9]]   # ears
        kp[:, p, 1:3, :] = [[cx - 15, 95, 0.9], [cx + 15, 95, 0.9]]     # eyes
        kp[:, p, 0, :] = [cx + nose_dx, 105, 0.9]
    d = dict(pose_t=np.arange(F) / 5.0, kp=kp, boxes=boxes, aw=640, ah=360)
    box = plan.speaker(d, d["pose_t"])
    assert np.allclose(box[:, 0], 100 / 640) and np.allclose(box[:, 2], 300 / 640)


def test_daytime_footage_in_a_night_act_is_recast_with_night_footage(fakes):
    W = {"A": grid(0.3, 40.0), "C": grid(0.3, 40.0), "D": grid(0.3, 40.0)}
    shots = [placed("C", 2.0), placed("A", 10.0), placed("C", 20.0)]
    shots[1]["unit"]["day"] = True                           # passing windows, wrong time of day
    shots[1]["act"] = "climax"
    _, recast, _ = plan.snap_windows(None, "portrait", shots, [rec("A", [10.7]), rec("C", [30.0]), rec("D", [15.0])],
                                     CFG, W)
    assert recast == 1 and shots[1]["unit"]["id"] == "D"


def test_a_close_dancer_with_arms_up_keeps_her_head_by_her_face_points():
    import numpy as np
    F = 5
    nb = np.zeros((F, 1, 5))
    nb[:, 0, :4] = (0.4, 0.0, 0.6, 1.0)                       # raised arms: the box touches the frame's top
    kp = np.zeros((F, 1, 17, 3))
    kp[:, 0, :5] = [500, 250, 0.9]                            # face points a quarter of the way down (ah = 1000)
    args = (nb, kp, np.ones((F, 1), bool), np.zeros(F, int), np.ones(F, bool), np.full(F, 0.5), np.full(F, 0.5), 0.3, 1.0)
    assert plan.shown(*args, ah=1000)[1] == 1.0
    assert plan.shown(*args)[1] == 0.0                        # by the box top alone: cut off


def test_a_window_mostly_past_the_shots_end_doesnt_judge_it():
    W = [win(92.7, 93.9, True), win(93.2, 94.4, True), win(93.7, 94.9, False), win(92.7, 95.1, True)]
    assert plan.span_ok(W, 93.167, 94.6) is True              # 93.7-94.9 fails, a quarter of it after the end
    assert plan.span_ok(W, 93.167, 94.769) is False           # nine tenths of it inside: the shot plays it
    W[1]["ok"] = False
    assert plan.span_ok(W, 93.167, 94.6) is False             # its own last stretch fails


def test_a_long_passing_window_never_hides_a_failing_second_inside_the_shot():
    # Fiesta C2139@28.0: a back fills 28.3-29.5 ("back"), a 2.4 s window over the shot averages it to a pass
    W = [win(27.8, 29.0, True), win(28.3, 29.5, False), win(27.3, 29.7, True), win(26.8, 30.4, True)]
    assert plan.span_ok(W, 28.04, 29.41) is False
    W[1]["ok"] = True
    assert plan.span_ok(W, 28.04, 29.41) is True


def test_a_recast_never_lands_a_long_slot_on_a_picked_clip_s_banned_footage(fakes):
    # EN: a 29 s slot recast onto C2103, whose footage outside Isaac's short picks only_picks bans; clear_of_bans
    # hands back the in-point unchanged when nothing fits, so "unchanged" must not read as "clear"
    W = {"A": grid(0.3, 40.0, bad=[(0.0, 40.0)]), "C": grid(0.3, 40.0), "P": grid(0.3, 60.0)}
    slot = placed("A", 1.0)
    slot.update(length=29.0, beats=60)
    shots = [placed("C", 2.0), slot, placed("C", 20.0)]
    p = rec("P", [1.5, 30.0])
    p["unit"]["bans"] = [[-1.0, 28.0], [31.0, 61.0]]          # only 28-31 s is the editor's
    plan.snap_windows(None, "portrait", shots, [rec("A", [10.7], ok=False), p], CFG, W)
    s = shots[1]
    assert s["unit"]["id"] != "P" or not plan.in_bans(s["src_in"], s["length"], s["unit"])
    assert plan.in_bans(1.0, 29.0, p["unit"]) and not plan.in_bans(28.0, 3.0, p["unit"])


def test_the_unit_with_the_fewest_slots_it_fits_is_covered_first(fakes):
    # Fiesta portrait: C2144 (any slot) took the one short slot before C2146 (passes only over ~1.4 s) was tried
    long_, short = placed("A", 9.0), placed("A", 30.0)
    long_.update(length=2.7, beats=6)
    short.update(length=0.9, beats=2)
    shots = [placed("A", 1.0), short, placed("B", 2.0), long_, placed("B", 6.0), placed("A", 40.0), placed("B", 9.0)]
    W = {"D": [win(20.0, 21.2, True), win(19.0, 22.6, False)], "E": grid(0.3, 40.0)}
    e = rec("E", [10.0])
    e["moments"][0].update(score=0.9, score_p=0.9)            # the better unit: tried first by rank alone
    assert sorted(plan.ensure_units(None, "portrait", shots, [e, rec("D", [20.5])], CFG, W)) == ["D", "E"]
    assert short["unit"]["id"] == "D"


def test_a_long_shot_no_stretch_passes_over_whole_is_cut_in_two_onto_passing_footage(fakes):
    # Fiesta portrait arrival: 8-beat slots (3.7 s) on clips whose every 3.7 s stretch has a failing second
    bad = [(t, t + 0.6) for t in range(2, 40, 3)]           # passing only in 2.4 s stretches
    W = {"A": grid(0.3, 40.0, bad=bad), "C": grid(0.3, 40.0), "D": grid(0.3, 40.0, bad=bad)}
    long_ = placed("A", 9.0)
    long_.update(length=3.6, beats=8, frames=180, rec_in=10.0)
    shots = [placed("C", 2.0), long_, placed("C", 20.0)]
    plan.snap_windows(None, "portrait", shots, [rec("A", [10.0, 20.0]), rec("C", [5.0]), rec("D", [16.0])], CFG, W)
    assert len(shots) == 4 and [s["beats"] for s in shots[1:3]] == [4, 4]
    assert shots[1]["frames"] + shots[2]["frames"] == 180 and shots[2]["rec_in"] == round(10.0 + shots[1]["length"], 3)
    assert shots[2]["unit"]["clip"] != shots[1]["unit"]["clip"]
    for s in shots[1:3]:
        assert plan.span_ok(W[s["unit"]["id"]], s["src_in"], s["src_in"] + s["length"]) is True
