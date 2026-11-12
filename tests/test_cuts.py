"""reel.json "cuts": other reels from the same footage and analysis."""
import json

import pytest

from focalcut.reel.plan import fits
from focalcut.reel.project import Project


def unit(uid, category, tag, orient):
    return dict(id=uid, category=category, tag=tag, orient=orient)


def project(tmp_path, cuts):
    d = tmp_path / "focalcut" / "reel"
    d.mkdir(parents=True)
    recs = [unit("C1-01_couple", "segments", "couple", "portrait"), unit("C2_band", "broll", "band", "landscape"),
            unit("C3", "demos", "demos", "landscape")]
    json.dump([dict(id=u["id"], unit=u) for u in recs], open(d / "moments.json", "w"))
    json.dump({"C1-01_couple": {"add": ["0:01"]}, "C3": {"add": ["0:02"]}, "_note": "x"}, open(d / "picks.json", "w"))
    json.dump(dict(name="EV", look="clean", open={"C3": "0:00-0:03"},
                   plan=dict(portrait=dict(heroes=3, ending=["C3", 1, 2])), cuts=cuts), open(d / "reel.json", "w"))
    return str(tmp_path)


def test_fits_matches_the_way_footage_was_shot():
    night, day = ["@portrait", "band"], ["@landscape", "!band"]
    assert fits(unit("a", "segments", "couple", "portrait"), night)
    assert fits(unit("b", "broll", "band-singer", "landscape"), night)
    assert not fits(unit("c", "demos", "demos", "landscape"), night)
    assert fits(unit("c", "demos", "demos", "landscape"), day)
    assert not fits(unit("b", "broll", "band-singer", "landscape"), day)
    assert not fits(unit("a", "segments", "couple", "portrait"), day)


def test_a_cut_keeps_its_own_footage_settings_and_picks(tmp_path):
    folder = project(tmp_path, dict(night=dict(formats=["portrait"], footage=["@portrait", "band"], open={},
                                               plan=dict(portrait=dict(ending=None)))))
    p = Project(folder, "night")
    assert p.formats == ["portrait"] and p.cut_name("portrait") == "night_portrait" and p.music_file == "music.json"
    assert p.out_stem(p.cut_name("portrait"), "portrait") == "EV_night_portrait"
    assert p.config["plan"]["portrait"] == dict(heroes=3, ending=None)   # merged per format
    assert p.config["look"] == "clean" and "cuts" not in p.config
    assert set(p.picks()) == {"C1-01_couple", "_note"}                  # no opening, no picks of other footage
    assert p.in_cut(unit("C2_band", "broll", "band", "landscape")) and not p.in_cut(unit("C3", "demos", "demos", "landscape"))
    with pytest.raises(RuntimeError):
        p.save_config()
    main = Project(folder)
    assert main.cut_name("portrait") == "portrait" and set(main.picks()) == {"C1-01_couple", "C3", "_note"}
    assert Project(project(tmp_path / "b", dict(night=dict(song="b.mp3"))), "night").music_file == "music_night.json"
    q = Project(project(tmp_path / "c", dict(day=dict(footage=["@landscape"], ban=["C2_band"]))), "day")
    assert q.in_cut(unit("C3", "demos", "demos", "landscape")) and not q.in_cut(unit("C2_band", "broll", "band", "landscape"))
    assert not Project(q.folder, None).ban and Project(project(tmp_path / "d", dict(day=dict(ban=["C9"]))), "day") \
        .in_cut(dict(id="C9-01_couple", clip="C9", category="segments", tag="couple", orient="portrait")) is False
    with pytest.raises(KeyError):
        Project(folder, "nope")


def test_rejects_ban_for_the_planner_but_leave_the_editors_picks_and_taste_alone(tmp_path):
    folder = project(tmp_path, {})
    d = tmp_path / "focalcut" / "reel"
    json.dump({"C1": {"add": ["0:05-0:06"], "ban": ["0:01-0:02"]}}, open(d / "picks.json", "w"))
    raw = json.load(open(d / "reel.json"))
    raw["profiles"] = {"qc": {"reject": {"C1": ["0:10-0:12"], "C2": ["0:03-0:04"]}}}
    raw["use"] = ["qc"]
    json.dump(raw, open(d / "reel.json", "w"))
    pr = Project(folder)
    assert pr.picks()["C1"] == {"add": ["0:05-0:06"], "ban": ["0:01-0:02", "0:10-0:12"]}
    assert pr.picks()["C2"] == {"ban": ["0:03-0:04"]}
    assert pr.picks(rejects=False)["C1"] == {"add": ["0:05-0:06"], "ban": ["0:01-0:02"]} and "C2" not in pr.picks(False)
    assert json.load(open(d / "picks.json"))["C1"]["ban"] == ["0:01-0:02"]


def test_profiles_mix_into_cuts_and_the_main_reel(tmp_path):
    folder = project(tmp_path, dict(day=dict(use=["no-night", "short"], formats=["landscape", "portrait"]),
                                    night=dict(use=["no-day", "short"], length=180, footage=["@portrait"])))
    raw = json.load(open(tmp_path / "focalcut" / "reel" / "reel.json"))
    raw["profiles"] = {"no-night": {"footage": ["!band"], "ban": ["C2_band"]},
                       "no-day": {"footage": ["!demos"], "plan": {"portrait": {"heroes": 1}}},
                       "short": {"length": 90, "plan": {"portrait": {"heroes": 2}}},
                       "loop": {"use": ["loop"]}}
    raw["use"] = ["short"]
    json.dump(raw, open(tmp_path / "focalcut" / "reel" / "reel.json", "w"))
    day = Project(folder, "day")
    assert day.footage == ["!band"] and day.ban == {"C2_band"} and day.config["length"] == 90
    assert not day.in_cut(unit("C2_band", "broll", "band", "landscape")) and day.in_cut(unit("C3", "demos", "demos", "landscape"))
    night = Project(folder, "night")
    assert night.footage == ["!demos", "@portrait"] and night.config["length"] == 180   # rules add up, own keys win
    assert night.config["plan"]["portrait"]["heroes"] == 2 and night.own_music          # later profile wins
    main = Project(folder)
    assert main.config["length"] == 90 and "profiles" not in main.config and "use" not in main.config
    with pytest.raises(RuntimeError):
        main.save_config()                                   # would flatten the profiles into reel.json
    raw["cuts"]["bad"] = dict(use=["loop"])
    json.dump(raw, open(tmp_path / "focalcut" / "reel" / "reel.json", "w"))
    with pytest.raises(ValueError):
        Project(folder, "bad")


def test_picks_go_to_the_reels_the_editor_ticked(tmp_path):
    folder = project(tmp_path, dict(short=dict(length=90)))
    d = tmp_path / "focalcut" / "reel"
    json.dump({"C3": {"add": ["0:01.00-0:02.00", "0:05.00-0:06.00", "0:08-0:09"], "end": "0:05.00-0:06.00",
                      "only": {"0:05-0:06": ["main"], "0:08.00-0:09.00": ["short"]}}}, open(d / "picks.json", "w"))
    main, short = Project(folder).picks()["C3"], Project(folder, "short").picks()["C3"]
    assert main["add"] == ["0:01.00-0:02.00", "0:05.00-0:06.00"] and main["end"] == "0:05.00-0:06.00"
    assert short["add"] == ["0:01.00-0:02.00", "0:08-0:09"] and "end" not in short and "only" not in short
    assert len(Project(folder).picks(reel=False)["C3"]["add"]) == 3


def test_bans_prefers_more_and_roles_go_to_their_reels_too(tmp_path):
    folder = project(tmp_path, dict(short=dict(length=90)))
    d = tmp_path / "focalcut" / "reel"
    json.dump({"C3": {"ban": ["all", "0:10-0:12", "0:20-0:22"], "prefer": ["0:30-0:31"], "more": 2,
                      "open": "0:01-0:02", "add": ["0:01-0:02"],
                      "only": {"ban:all": ["short"], "ban:0:10.00-0:12.00": ["main"], "prefer:0:30-0:31": ["short"],
                               "more:": ["short"], "open:0:01-0:02": ["main"]}}}, open(d / "picks.json", "w"))
    main, short = Project(folder).picks(rejects=False)["C3"], Project(folder, "short").picks(rejects=False)["C3"]
    assert main["ban"] == ["0:10-0:12", "0:20-0:22"] and "prefer" not in main and "more" not in main
    assert main["open"] == "0:01-0:02" and main["add"] == ["0:01-0:02"]
    assert short["ban"] == ["all", "0:20-0:22"] and short["prefer"] == ["0:30-0:31"] and short["more"] == 2
    assert short["open"] == "0:00-0:03" and short["add"] == ["0:01-0:02"]   # (reel.json's opening, as no pick is)


def test_a_clip_picked_only_for_other_reels_gives_this_one_nothing(tmp_path):
    folder = project(tmp_path, dict(short=dict(length=90)))
    d = tmp_path / "focalcut" / "reel"
    json.dump({"C1-01_couple": {"add": ["0:01-0:02"], "only": {"0:01-0:02": ["main"]}},
               "C3": {"add": ["0:05-0:06"], "only": {"0:05-0:06": ["main"]}}}, open(d / "picks.json", "w"))
    assert Project(folder, "short").picks()["C1-01_couple"]["ban"] == ["all"]
    assert "ban" not in Project(folder).picks()["C1-01_couple"]
    assert "ban" not in Project(folder, "short").picks()["C3"]          # (the reel's named opening: kept)
