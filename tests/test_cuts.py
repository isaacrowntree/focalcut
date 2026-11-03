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
    with pytest.raises(KeyError):
        Project(folder, "nope")
