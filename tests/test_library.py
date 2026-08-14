"""Library scanning, manifest round-trips, and segment derivation."""
from __future__ import annotations

import itertools
import json

import pytest

from focalcut import library as lib

from .conftest import needs_ffmpeg


class TestDeriveSegments:
    def test_no_cuts_gives_one_whole_clip_segment(self):
        assert lib.derive_segments(30.0, []) == [[0.0, 30.0]]

    def test_one_cut_splits_in_two_covering_the_clip(self):
        assert lib.derive_segments(30.0, [12.5]) == [[0.0, 12.5], [12.5, 30.0]]

    def test_cuts_are_sorted_before_splitting(self):
        assert lib.derive_segments(30.0, [20.0, 5.0]) == [[0.0, 5.0], [5.0, 20.0], [20.0, 30.0]]

    def test_segments_are_contiguous_and_span_the_duration(self):
        segs = lib.derive_segments(97.3, [10.0, 40.5, 80.0])
        assert segs[0][0] == 0.0
        assert segs[-1][1] == 97.3
        for a, b in itertools.pairwise(segs):
            assert a[1] == b[0]

    def test_rounds_to_two_decimals(self):
        assert lib.derive_segments(10.0, [3.14159]) == [[0.0, 3.14], [3.14, 10.0]]


class TestManifest:
    def test_load_returns_empty_when_absent(self, tmp_path):
        assert lib.load(str(tmp_path)) == {}

    def test_save_then_load_round_trips_the_clips(self, tmp_path):
        clips = [{"name": "a.mp4", "cuts": [5.0], "segments": [[0, 5], [5, 10]]}]
        lib.save(str(tmp_path), clips, {"status": "judged"})
        got = lib.load(str(tmp_path))
        assert got["clips"] == clips
        assert got["status"] == "judged"

    def test_save_counts_multi_and_single_couple_clips(self, tmp_path):
        clips = [
            {"name": "a.mp4", "cuts": [5.0], "segments": [[0, 5], [5, 10]]},
            {"name": "b.mp4", "cuts": [], "segments": [[0, 10]]},
            {"name": "c.mp4", "cuts": [2.0, 7.0], "segments": [[0, 2], [2, 7], [7, 10]]},
        ]
        with open(lib.save(str(tmp_path), clips, {})) as fh:
            got = json.load(fh)
        assert got["clips_count"] == 3
        assert got["multi_couple"] == 2
        assert got["single_segment"] == 1
        assert got["total_cuts"] == 3

    def test_manifest_lands_inside_the_library_subfolder(self, tmp_path):
        path = lib.save(str(tmp_path), [], {})
        assert (tmp_path / lib.LIBDIR / lib.MANIFEST).samefile(path)


class TestListClips:
    @needs_ffmpeg
    def test_finds_video_files_sorted(self, media):
        assert [c["name"] for c in lib.list_clips(str(media))] == ["a.mp4", "b.mp4"]

    @needs_ffmpeg
    def test_reports_duration_and_display_dimensions(self, media):
        a = next(c for c in lib.list_clips(str(media)) if c["name"] == "a.mp4")
        assert a["duration"] == pytest.approx(30.0, abs=0.2)
        assert (a["width"], a["height"]) == (1080, 1920)
        assert a["fps"] == pytest.approx(50.0)

    @needs_ffmpeg
    def test_portrait_only_keeps_portrait_clips(self, media):
        assert len(lib.list_clips(str(media), portrait_only=True)) == 2

    def test_ignores_non_video_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hi")
        (tmp_path / "sheet.jpg").write_bytes(b"\xff\xd8\xff")
        assert lib.list_clips(str(tmp_path)) == []


class TestSheets:
    @needs_ffmpeg
    def test_builds_one_sheet_per_clip(self, media, tmp_path):
        jobs = lib.build_sheets(str(media), str(tmp_path / "sheets"), interval=10, cols=3)
        assert len(jobs) == 2
        assert all(j["sheet"] and j["error"] is None if "error" in j else j["sheet"] for j in jobs)

    @needs_ffmpeg
    def test_window_sheet_is_clamped_to_the_clip_start(self, media, tmp_path):
        # a cut at 3s with a 9s half-width would start at -6s; must clamp to 0
        out = lib.window_sheet(str(media), "a.mp4", 3.0, str(tmp_path / "w.jpg"),
                               half_width=9.0, interval=2.0)
        assert out and (tmp_path / "w.jpg").stat().st_size > 0
