"""Timeline construction and NLE export.

The point of these tests is that a segment's *time survives the round trip*: a cut the
judge placed at 12.5s must land on frame 625 of a 50fps source, in every format, or an
editor opens the timeline on the wrong frame.
"""
from __future__ import annotations

import os
from xml.etree import ElementTree as ET

import pytest

from focalcut import timeline as T

from .conftest import needs_ffmpeg

otio = pytest.importorskip("opentimelineio")


class TestWindowsPaths:
    def test_maps_wsl_mount_to_drive_letter(self):
        assert T.to_windows_path("/mnt/d/Footage/clip.mp4") == "D:/Footage/clip.mp4"

    def test_uppercases_the_drive(self):
        assert T.to_windows_path("/mnt/c/x.mp4").startswith("C:/")

    def test_leaves_non_mount_paths_alone(self):
        assert T.to_windows_path("/home/user/x.mp4") == "/home/user/x.mp4"

    def test_ignores_multi_letter_second_segment(self):
        # /mnt/media is a normal folder, not a drive mount
        assert T.to_windows_path("/mnt/media/x.mp4") == "/mnt/media/x.mp4"

    def test_preserves_spaces_for_the_url_layer(self):
        assert T.to_windows_path("/mnt/d/My Footage/a.mp4") == "D:/My Footage/a.mp4"


class TestBuildTimeline:
    @needs_ffmpeg
    def test_lays_segments_end_to_end(self, media, clips):
        tl = T.build_timeline(str(media), clips, "selects")
        track = tl.tracks[0]
        assert len(track) == 3
        # 12.5 + 17.5 + 9.0 == 39s at 50fps
        assert tl.duration().value == 1950
        assert tl.duration().rate == 50

    @needs_ffmpeg
    def test_source_in_points_are_frame_exact(self, media, clips):
        tl = T.build_timeline(str(media), clips, "s")
        track = tl.tracks[0]
        starts = [c.source_range.start_time.value for c in track]
        durations = [c.source_range.duration.value for c in track]
        assert starts == [0, 625, 100]        # 0s, 12.5s, 2.0s @ 50fps
        assert durations == [625, 875, 450]   # 12.5s, 17.5s, 9.0s

    @needs_ffmpeg
    def test_names_are_stem_and_index(self, media, clips):
        tl = T.build_timeline(str(media), clips, "s")
        assert [c.name for c in tl.tracks[0]] == ["a-01", "a-02", "b-01"]

    @needs_ffmpeg
    def test_skips_zero_and_negative_length_segments(self, media):
        bad = [{"name": "a.mp4", "segments": [[5.0, 5.0], [8.0, 4.0], [0.0, 3.0]]}]
        tl = T.build_timeline(str(media), bad, "s")
        assert len(tl.tracks[0]) == 1

    @needs_ffmpeg
    def test_windows_paths_reach_the_media_reference(self, media, clips, monkeypatch):
        monkeypatch.setattr(T, "to_windows_path", lambda p: "D:/fake/" + os.path.basename(p))
        tl = T.build_timeline(str(media), clips, "s", windows_paths=True)
        assert "D%3A/fake/a.mp4" in tl.tracks[0][0].media_reference.target_url

    def test_rejects_an_empty_library(self):
        with pytest.raises(T.TimelineError, match="no clips"):
            T.build_timeline("/nowhere", [], "s")

    def test_reports_missing_media_by_path(self, tmp_path):
        with pytest.raises(T.TimelineError, match="media missing"):
            T.build_timeline(str(tmp_path), [{"name": "gone.mp4", "segments": [[0, 1]]}], "s")

    @needs_ffmpeg
    def test_rejects_a_manifest_with_no_segments(self, media):
        # the skeleton manifest state: sheets built, judge has not run yet
        with pytest.raises(T.TimelineError, match="no segments"):
            T.build_timeline(str(media), [{"name": "a.mp4", "segments": []}], "s")


class TestWrite:
    @needs_ffmpeg
    @pytest.mark.parametrize("fmt", ["otio", "fcp7", "edl", "aaf"])
    def test_every_format_writes_a_nonempty_file(self, media, clips, tmp_path, fmt):
        if fmt not in T.available_formats():
            pytest.skip(f"{fmt} adapter not installed")
        out = tmp_path / ("selects" + T.FORMATS[fmt][1])
        T.write_timeline(T.build_timeline(str(media), clips, "s"), str(out), fmt)
        assert out.stat().st_size > 0

    @needs_ffmpeg
    @pytest.mark.parametrize("fmt", ["otio", "fcp7", "aaf"])
    def test_timing_survives_the_round_trip(self, media, clips, tmp_path, fmt):
        if fmt not in T.available_formats():
            pytest.skip(f"{fmt} adapter not installed")
        out = tmp_path / ("selects" + T.FORMATS[fmt][1])
        T.write_timeline(T.build_timeline(str(media), clips, "s"), str(out), fmt)
        back = otio.adapters.read_from_file(str(out))
        assert len(list(back.find_clips())) == 3
        assert back.duration().to_seconds() == pytest.approx(39.0, abs=0.05)

    @needs_ffmpeg
    def test_fcp7_xml_is_wellformed_xmeml_with_a_clip_per_segment(self, media, clips, tmp_path):
        out = tmp_path / "s.xml"
        T.write_timeline(T.build_timeline(str(media), clips, "dance"), str(out), "fcp7")
        root = ET.parse(out).getroot()
        assert root.tag == "xmeml"
        assert root.findtext(".//sequence/name") == "dance"
        assert root.findtext(".//sequence/rate/timebase") == "50"
        items = root.findall(".//video//clipitem")
        assert len(items) == 3
        # the second select must start on frame 625, not 624 or 626
        assert items[1].findtext("in") == "625"

    @needs_ffmpeg
    def test_edl_timecodes_are_frame_exact(self, media, clips, tmp_path):
        if "edl" not in T.available_formats():
            pytest.skip("cmx_3600 adapter not installed")
        out = tmp_path / "s.edl"
        T.write_timeline(T.build_timeline(str(media), clips, "s"), str(out), "edl")
        text = out.read_text()
        assert "00:00:12:25" in text   # 12.5s @ 50fps
        assert "00:00:39:00" in text   # end of the 39s timeline

    @needs_ffmpeg
    def test_creates_the_output_directory(self, media, clips, tmp_path):
        out = tmp_path / "nested" / "deeper" / "s.xml"
        T.write_timeline(T.build_timeline(str(media), clips, "s"), str(out), "fcp7")
        assert out.exists()

    @needs_ffmpeg
    def test_rejects_an_unknown_format(self, media, clips, tmp_path):
        with pytest.raises(T.TimelineError, match="unknown format"):
            T.write_timeline(T.build_timeline(str(media), clips, "s"),
                             str(tmp_path / "x.mov"), "prores")


class TestFormats:
    def test_fcp7_is_available_in_a_normal_install(self):
        assert "fcp7" in T.available_formats()

    def test_otio_is_always_available(self):
        assert "otio" in T.available_formats()

    def test_default_output_uses_the_format_suffix(self, tmp_path):
        p = T.default_output("/f", "edl", str(tmp_path))
        assert p.endswith("selects.edl")
