"""The regions look (reel/regions.py): the job handed to color-grade-ai's match_camera.py and its report read back
as reference.looks() returns LUTs. No GPU, no footage, no match_camera."""
import os
import types

import pytest

from focalcut.reel import reference as R
from focalcut.reel import regions


def _project(tmp_path, **config):
    reel = tmp_path / "focalcut" / "reel"
    cfg = dict(clock={"C2*": -3389}, proxies={"C2210.MP4": ["export/baptisms/C2210.mp4", []]},
               match_per_clip=["marcos"], look_flat=["drone"], look_classes={"drone": ["sand", "black"]},
               spans={"DJI_0018": [0, None]}, **config)
    return types.SimpleNamespace(folder=str(tmp_path), reel_dir=str(reel), config=cfg, name="film",
                                 path=lambda *p: os.path.join(str(reel), *p))


CAMS = {"isaac": ["C2*"], "marcos": ["C0*"], "drone": ["DJI_*"]}
UNITS = [dict(clip="C2210", source="C2210.MP4"), dict(clip="C0678", source="Marcos/C0678.MP4"),
         dict(clip="C0681", source="Marcos/C0681.MP4"), dict(clip="DJI_0018", source="DJI_0018.MP4"),
         dict(clip="C2210", source="C2210.MP4")]


def test_job_has_the_hero_as_reference_and_the_rest_as_sources(tmp_path):
    j = regions.job(_project(tmp_path), UNITS, CAMS, "isaac")
    assert j["reference"]["name"] == "isaac" and j["reference"]["files"] == [str(tmp_path / "C2210.MP4")]
    marcos, drone = j["sources"]
    assert marcos["per_clip"] and marcos["cube"] == "marcos_regions_{clip}.cube" and len(marcos["files"]) == 2
    assert drone["flat"] and not drone["per_clip"] and drone["cube"] == "drone_regions.cube"
    assert drone["classes"] == ["sand", "black"] and "classes" not in marcos
    assert j["analyse"] == {"C2210": str(tmp_path / "export/baptisms/C2210.mp4")}
    assert j["spans"]["DJI_0018"] == [0, 1e9] and j["reference"]["clock"] == {"C2*": -3389}


def test_report_becomes_luts_per_camera_and_per_clip(tmp_path):
    p = _project(tmp_path)
    chk = dict(before=dict(all=10.0), after=dict(all=4.0))
    report = dict(sources={
        "drone": dict(clips={"DJI_0018": dict(cube="drone_regions.cube", check=chk, flag=[])}),
        "marcos": dict(clips={"C0678": dict(cube="marcos_regions_C0678.cube", check=chk, flag=[]),
                              "C0681": dict(cube="marcos_regions_C0681.cube", check={}, flag=["little seen"])})})
    luts = regions.luts(report, CAMS, "isaac", p.reel_dir, p.path("looks"))
    assert luts["isaac"]["cube"] is None                       # the hero stays as shot
    assert luts["drone"] == dict(match=["DJI_*"], cube="looks/drone_regions.cube", how="matched to isaac by region",
                                 check=dict(before=10.0, after=4.0))
    assert luts["marcos:C0678"]["cube"] == "looks/marcos_regions_C0678.cube" and luts["marcos:C0678"]["match"] == ["C0678"]
    assert luts["marcos:C0681"]["check"] is None and luts["marcos:C0681"]["flag"] == ["little seen"]
    assert "marcos" not in luts                                # no camera-wide cube when each clip has its own


def test_looks_hands_over_only_when_asked(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(regions, "looks", lambda project, units, cams, hero: called.append(hero) or {"ok": True})
    p = _project(tmp_path, look_method="regions", cameras=CAMS, look_hero="isaac", look_strength=0)
    assert R.looks(p, {}, UNITS, [0, 1, 2, 3, 4], [None] * 5, None) == {"ok": True} and called == ["isaac"]


def test_missing_match_camera_says_where_to_point(tmp_path):
    with pytest.raises(FileNotFoundError, match="match_camera"):
        regions.load(str(tmp_path / "nope.py"))
