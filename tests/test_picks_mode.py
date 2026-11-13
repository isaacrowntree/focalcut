"""Picks mode (reel.json "mode": "picks"): the reel is cut from the editor's picks, as they chose them."""
from __future__ import annotations

import json
import os
from collections import Counter
from itertools import groupby, pairwise

from focalcut.reel import plan as P
from focalcut.reel.project import Project

from .test_reel import fake_reel

BEAT = 0.6


def rng(a, b):
    return f"{a:.2f}-{b:.2f}"


def picks_reel(tmp_path, picks, n_units=16, bars=20, plan=None, **config):
    """A fake reel in picks mode: `bars` of music, picks.json as given."""
    pr, music, moments = fake_reel(tmp_path, n_units=n_units, beat=BEAT)
    music["length"] = round(bars * 4 * BEAT, 3)
    pr.config.update(mode="picks", **config)
    if plan:
        pr.config["plan"] = {f: dict(plan) for f in ("landscape", "portrait")}
    os.makedirs(pr.reel_dir, exist_ok=True)
    with open(pr.path("picks.json"), "w") as f:
        json.dump(picks, f)
    return pr, music, moments


def spans(plan):
    """(unit, in, beats) of every pick shot, in order."""
    return [(s["unit"]["id"], round((s.get("whole") or s["span"])[0], 2), s["beats"]) for s in plan["shots"]
            if s.get("add") and s.get("span")]


def overlaps(shots):
    """Pairs of shots that play the same source."""
    out = []
    for i, a in enumerate(shots):
        for b in shots[i + 1:]:
            if a["unit"]["id"] == b["unit"]["id"] and \
                    min(a["src_in"] + a["length"], b["src_in"] + b["length"]) - max(a["src_in"], b["src_in"]) > 0.05:
                out.append((a["unit"]["id"], a["src_in"], b["src_in"]))
    return out


def test_every_pick_plays_once_whole_and_only_unpicked_broll_fills_the_rest(tmp_path):
    picks = {"U00": {"add": [rng(10, 12.4), rng(40, 41.2)]},         # 4 and 2 beats
             "U01": {"add": [rng(20, 23)], "ban": [rng(50, 60)]},    # 5 beats
             "U02": {"add": [rng(30, 30.5)]},                        # under a beat: one beat
             "U07": {"add": [rng(12, 14.4)]},                        # crowd B-roll, picked
             "U04": {"ban": ["all"]}}
    pr, music, moments = picks_reel(tmp_path, picks)
    plan = P.build(pr, "landscape", music, moments)
    shots = plan["shots"]
    assert sum(s["beats"] for s in shots) == 80                  # the grid is exact
    # every pick once, at its own length
    assert sorted(spans(plan)) == [("U00", 10.0, 4), ("U00", 40.0, 2), ("U01", 20.0, 5), ("U02", 30.0, 1), ("U07", 12.0, 4)]
    assert all(s["src_in"] == s["span"][0] for s in shots if s.get("add"))
    assert not overlaps(shots)                                   # nothing is seen twice
    # what the picks leave is B-roll from clips the editor left alone: no picked clip's other footage, no people
    # footage they didn't pick, nothing banned
    rest = [s for s in shots if not s.get("add")]
    assert rest and all(s["unit"]["role"] == "transition" for s in rest)
    assert not {s["unit"]["id"] for s in rest} & {"U00", "U01", "U02", "U07", "U04"}
    assert "U04" not in {s["unit"]["id"] for s in shots}
    # no quick cuts, pieces or planner effects
    assert not any(s.get("reuse") or s.get("person") or s.get("band") for s in shots)
    assert not any(f in P.SPEED for s in shots if s.get("add") for f in s["fx"])   # a pick plays at real speed
    # it opens on B-roll under the logo
    assert shots[0]["logo"] == "in" and shots[0]["unit"]["role"] == "transition"
    assert not P.pick_report(shots, picks, plan["picks_condensed"])


def many_picks(n_clips=6, per=6, length=2.4):
    """per picks of `length` seconds in each of the first show / dance clips."""
    ids = [f"U{i:02d}" for i in range(16) if i % 4 != 3][:n_clips]
    return {u: {"add": [rng(5 + 8 * k, 5 + 8 * k + length) for k in range(per)]} for u in ids}


def test_picks_longer_than_the_reel_are_trimmed_the_least_starred_first_and_none_is_dropped(tmp_path):
    picks = many_picks()                                         # 36 picks x 4 beats = 144 beats for 80
    picks["U00"]["stars"] = {picks["U00"]["add"][0]: 5, picks["U00"]["add"][1]: 4}
    pr, music, moments = picks_reel(tmp_path, picks, bars=22)
    plan = P.build(pr, "landscape", music, moments)
    got = spans(plan)
    assert len(got) == 36 and plan["picks_condensed"] == []      # all of them in
    assert sum(s["beats"] for s in plan["shots"]) == 88
    by = {(u, a): n for u, a, n in got}
    assert by[("U00", 5.0)] == 4 and by[("U00", 13.0)] == 4      # the starred ones whole
    assert min(by.values()) == P.PICK_FLOOR                      # no slivers: none under the floor
    assert sorted(by.values()).count(3) == 4                     # the room left shared out, a beat each
    assert not overlaps(plan["shots"])
    assert all(s.get("whole") for s in plan["shots"] if s.get("add") and s["beats"] < 4)   # (marked as trimmed)
    assert not P.pick_report(plan["shots"], picks, plan["picks_condensed"])


def test_picks_that_cannot_fit_even_trimmed_go_the_least_starred_first_and_are_reported(tmp_path, capsys):
    picks = many_picks(n_clips=6, per=8)                         # 48 picks: 96 beats at the floor, room for ~76
    keep = picks["U01"]["add"][3]
    picks["U01"]["stars"] = {keep: 5}
    low = picks["U02"]["add"][2]
    picks["U02"]["stars"] = {low: 1}
    pr, music, moments = picks_reel(tmp_path, picks)
    plan = P.build(pr, "landscape", music, moments)
    out = {(u, round(a, 2)) for u, a, _ in plan["picks_condensed"]}
    assert out and ("U02", 21.0) in out and ("U01", 29.0) not in out     # the one-star first, never the five-star
    got = spans(plan)
    assert len(got) + len(out) == 48 and not {(u, a) for u, a, _ in got} & out
    assert sum(s["beats"] for s in plan["shots"]) == 80
    assert "left out for room" in capsys.readouterr().out        # and the log names them
    assert not P.pick_report(plan["shots"], picks, plan["picks_condensed"])


def test_both_formats_play_the_same_picks_in_the_same_order_at_the_same_lengths(tmp_path):
    picks = many_picks()
    picks["U07"] = {"add": [rng(12, 14.4), rng(30, 31.2)]}
    pr, music, moments = picks_reel(tmp_path, picks)
    a = P.build(pr, "landscape", music, moments)
    b = P.build(pr, "portrait", music, moments)
    assert spans(a) == spans(b) and len(a["shots"]) == len(b["shots"])


def test_no_clip_twice_in_a_row_and_broll_never_beside_broll(tmp_path):
    picks = many_picks(n_clips=6, per=4, length=1.2)
    picks["U07"] = {"add": [rng(10 + 9 * k, 11.2 + 9 * k) for k in range(5)]}
    picks["U11"] = {"add": [rng(10 + 9 * k, 11.2 + 9 * k) for k in range(2)]}
    pr, music, moments = picks_reel(tmp_path, picks, bars=16)
    shots = P.build(pr, "landscape", music, moments)["shots"]
    body = [s for s in shots if not s.get("logo")]
    assert all(a["unit"]["clip"] != b["unit"]["clip"] for a, b in pairwise(body))
    assert not any(a["unit"]["role"] == b["unit"]["role"] == "transition" for a, b in pairwise(body))
    # a clip's picks are spread out, not played as a block
    at = [k for k, s in enumerate(body) if s["unit"]["id"] == "U07"]
    assert max(at) - min(at) >= 2 * (len(at) - 1) and len(at) >= 3   # (one of its five went for room: B-roll first)


def test_card_shuffle_keeps_the_acts_in_order_and_mends_the_cut():
    items = [(0, "A", 1.0), (0, "A", 2.0), (0, "A", 3.0), (0, "B", 1.0), (1, "C", 1.0), (1, "C", 2.0), (1, "D", 5.0)]
    out = P.card_shuffle(items, lambda x: x)
    assert sorted(out) == sorted(items)
    assert all(a[1] != b[1] for a, b in pairwise(out))      # (a pick of the next act comes forward to do it)
    assert out[0][0] == 0 and out[-1][0] == 1
    # runs of one-beat shots are broken up when lengths are known
    items = [(0, f"C{k}", float(k)) for k in range(12)]
    beats = {x: (1 if k < 6 else 3) for k, x in enumerate(items)}
    out = P.card_shuffle(items, lambda x: x, beats=lambda x: beats[x])
    runs = max(len(list(g)) for one, g in groupby(out, key=lambda x: beats[x] == 1) if one)
    assert runs <= 2


def test_the_editors_opening_and_ending_are_honoured(tmp_path):
    picks = many_picks(n_clips=3, per=3)
    picks["U07"] = {"add": [rng(12, 14.4)], "open": rng(20, 24.8)}
    picks["U11"] = {"add": [rng(40, 41.8)]}
    plan_cfg = {"open_with": [["U01", "0:21.00"]], "end_with": [["U00", "0:13.00"]], "ending": ["U11", 50.0, 53.0, 0.5]}
    pr, music, moments = picks_reel(tmp_path, picks, plan=plan_cfg)
    for fmt in ("landscape", "portrait"):
        shots = P.build(pr, fmt, music, moments)["shots"]
        assert shots[0]["unit"]["id"] == "U07" and shots[0]["open"] and shots[0]["logo"] == "in" and shots[0]["src_in"] == 20.0
        assert (shots[1]["unit"]["id"], shots[1]["span"][0]) == ("U01", 21.0)          # open_with: after the logo
        assert (shots[-2]["unit"]["id"], shots[-2]["span"][0]) == ("U00", 13.0)        # end_with: into the ending
        assert shots[-1]["ending"] and shots[-1]["unit"]["id"] == "U11" and shots[-1]["beats"] == 5
        assert shots[1]["beats"] == 4 and shots[-2]["beats"] == 4                      # the named picks play whole
        assert Counter((u, a) for u, a, _ in spans({"shots": shots})).most_common(1)[0][1] == 1


def test_close_names_the_pick_that_ends_the_reel(tmp_path, capsys):
    picks = many_picks(n_clips=3, per=3)
    pr, music, moments = picks_reel(tmp_path, picks, plan={"close": ["U01", 22.0]})
    shots = P.build(pr, "landscape", music, moments)["shots"]
    assert (shots[-1]["unit"]["id"], shots[-1]["span"][0]) == ("U01", 21.0) and shots[-1]["close"]
    assert sum(1 for s in shots if s["unit"]["id"] == "U01" and s.get("span") and s["span"][0] == 21.0) == 1
    # a close that isn't one of this reel's picks is left out, and said
    pr.config["plan"] = {"landscape": {"close": ["U09", 3.0]}}
    shots = P.build(pr, "landscape", music, moments)["shots"]
    assert "isn't among this reel's picks" in capsys.readouterr().out
    assert len(spans({"shots": shots})) == 9


def test_a_cut_that_says_how_it_ends_does_not_take_the_projects_close(tmp_path):
    d = tmp_path / "focalcut" / "reel"
    d.mkdir(parents=True)
    main = {"close": ["C1", 66.0], "ending": ["C2", 1.0, 3.0], "pick_hold": 4}
    cfg = {"plan": {"landscape": dict(main), "portrait": dict(main)},
           "cuts": {"short": {"plan": {"landscape": {"ending": ["C3", 1.0, 3.0]}}},
                    "own": {"plan": {"landscape": {"ending": ["C3", 1.0, 3.0], "close": ["C4", 2.0]}}},
                    "same": {"length": 60}}}
    (d / "reel.json").write_text(json.dumps(cfg))
    short = Project(str(tmp_path), "short").config["plan"]
    assert "close" not in short["landscape"] and short["landscape"]["ending"][0] == "C3"
    assert short["landscape"]["pick_hold"] == 4                   # (everything else is still the project's)
    assert short["portrait"]["close"] == ["C1", 66.0]            # a format the cut says nothing about: as before
    assert Project(str(tmp_path), "own").config["plan"]["landscape"]["close"] == ["C4", 2.0]
    assert Project(str(tmp_path), "same").config["plan"]["landscape"]["close"] == ["C1", 66.0]
    assert Project(str(tmp_path)).config["plan"]["landscape"]["close"] == ["C1", 66.0]


def test_a_pick_in_a_clip_with_no_detected_moments_still_plays(tmp_path):
    picks = {"U00": {"add": [rng(10, 12.4)]}, "U01": {"add": [rng(9.2, 9.6)]}}
    for mode in ("picks", None):
        pr, music, moments = picks_reel(tmp_path, picks)
        pr.config["mode"] = mode
        moments[1]["moments"] = []                               # the moments step found nothing in U01
        plan = P.build(pr, "landscape", music, moments)
        assert ("U01", 9.2) in {(u, a) for u, a, _ in spans(plan)}
        assert not [x for x in P.pick_report(plan["shots"], picks, plan["picks_condensed"]) if "missing" in x]


def test_picks_for_another_reel_and_this_reels_bans_stay_out(tmp_path):
    d = tmp_path / "focalcut" / "reel"
    d.mkdir(parents=True)
    (d / "reel.json").write_text(json.dumps({"cuts": {"short": {"mode": "picks"}}}))
    a, b = rng(10, 12.4), rng(40, 42.4)
    picks = {"U00": {"add": [a, b], "only": {a: ["main"]}},              # one pick each reel
             "U01": {"add": [a], "only": {a: ["main"]}},                 # picked for the main reel only
             "U05": {"add": [a], "ban": ["all"], "only": {"ban:all": ["short"]}},   # banned for the short only
             "U07": {"add": [a], "only": {a: ["main"]}}}                 # B-roll picked for the main reel only
    (d / "picks.json").write_text(json.dumps(picks))
    _, music, moments = fake_reel(tmp_path, beat=BEAT)
    music["length"] = round(20 * 4 * BEAT, 3)
    pr = Project(str(tmp_path), "short")
    shots = P.build(pr, "landscape", music, moments)["shots"]
    ids = {s["unit"]["id"] for s in shots}
    assert spans({"shots": shots}) == [("U00", 40.0, 4)]
    assert not ids & {"U01", "U05", "U07"}                       # nothing of them, not as filler either
    assert all(s.get("add") or s["unit"]["role"] == "transition" for s in shots)


def test_the_old_pick_keys_are_accepted_and_change_nothing_in_picks_mode(tmp_path, capsys):
    picks = many_picks(n_clips=3, per=3)
    pr, music, moments = picks_reel(tmp_path, picks)
    want = spans(P.build(pr, "landscape", music, moments))
    old = {"pick_hold": 2, "pick_grow": 4, "pick_share": 0.9, "pick_clip_share": 10.0, "pick_by": "clarity",
           "pick_whole": ["U00"], "pick_like": "landscape", "reuse": False, "open_broll": True}
    pr.config["plan"] = {"landscape": dict(old)}
    pr.own_plan = pr.config["plan"]
    assert spans(P.build(pr, "landscape", music, moments)) == want
    assert "not needed" in capsys.readouterr().out


def test_without_the_mode_the_plan_is_the_planners_as_before(tmp_path):
    pr, music, moments = fake_reel(tmp_path)
    assert not P.picks_mode(P.config(pr, "landscape"))
    plan = P.build(pr, "landscape", music, moments)
    assert {s["unit"]["clip"] for s in plan["shots"]} == {r["unit"]["clip"] for r in moments}   # use everything


def broll_and_people(n_broll=4):
    """Picks that overrun the reel: 12 of people (U00-U02) and n_broll of B-roll (U03, U07), all 4 beats long."""
    picks = many_picks(n_clips=3, per=4)
    picks["U07"] = {"add": [rng(10 + 9 * k, 12.4 + 9 * k) for k in range(n_broll // 2)]}
    picks["U03"] = {"add": [rng(10 + 9 * k, 12.4 + 9 * k) for k in range(n_broll - n_broll // 2)]}
    return picks


def test_broll_picks_give_up_their_beats_before_any_pick_of_people(tmp_path):
    # 16 picks x 4 beats = 64, and room for 56 of them: the 4 B-roll picks can give those 8 beats on their own
    pr, music, moments = picks_reel(tmp_path, broll_and_people(), bars=16, open={"U11": rng(20, 22)})
    plan = P.build(pr, "landscape", music, moments)
    got = [s for s in plan["shots"] if s.get("add") and s.get("span")]
    assert len(got) == 16                                        # nobody dropped
    people = [s for s in got if s["unit"]["role"] != "transition"]
    broll = [s for s in got if s["unit"]["role"] == "transition"]
    assert all(s["beats"] == 4 and not s.get("whole") for s in people)   # every pick of people whole
    assert all(s["beats"] == 2 for s in broll)                   # the B-roll at its floor


def test_portrait_drops_broll_picks_before_it_trims_people_and_keeps_starred_ones(tmp_path):
    picks = broll_and_people()
    picks["U07"]["stars"] = {picks["U07"]["add"][0]: 5}          # a B-roll pick the editor starred: it stays
    pr, music, moments = picks_reel(tmp_path, picks, bars=14, open={"U11": rng(20, 22)})   # room for 48 of 64 beats
    land = P.build(pr, "landscape", music, moments)
    port = P.build(pr, "portrait", music, moments)
    def role(s):
        return s["unit"]["role"] == "transition"
    lp = [s for s in land["shots"] if s.get("add") and s.get("span")]
    pp = [s for s in port["shots"] if s.get("add") and s.get("span")]
    assert len(lp) == 16 and sum(s["beats"] for s in lp if not role(s)) <= 40   # landscape: all in, people trimmed too
    assert sum(s["beats"] for s in pp if not role(s)) == 47      # portrait: the people all but whole (12 x 4)
    kept = [s for s in pp if role(s)]
    assert [(s["unit"]["id"], s["beats"], (s.get("whole") or s["span"])[0]) for s in kept] == [("U07", 1, 10.0)]
    assert len(port["picks_condensed"]) == 3                     # and the three left out are named
