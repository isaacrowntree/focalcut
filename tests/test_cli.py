"""End-to-end CLI behaviour.

These also guard the import graph: `library`, `sheet`, `export` and `formats` must run
with torch absent, because the recommended pipeline is GPU-free and most users will
never install the SAM extra.
"""
from __future__ import annotations

import json

import pytest

from focalcut import library as lib
from focalcut.cli import main

from .conftest import needs_ffmpeg


@pytest.fixture
def judged(media):
    """A library whose manifest has been filled in by the judge."""
    clips = [
        {"name": "a.mp4", "duration": 30.0, "cuts": [12.5],
         "segments": lib.derive_segments(30.0, [12.5])},
        {"name": "b.mp4", "duration": 20.0, "cuts": [], "segments": [[0.0, 20.0]]},
    ]
    lib.save(str(media), clips, {"status": "judged"})
    return media


class TestFormats:
    def test_lists_formats_and_exits_clean(self, capsys):
        assert main(["formats"]) == 0
        assert "fcp7" in capsys.readouterr().out


class TestLibrary:
    @needs_ffmpeg
    def test_builds_sheets_and_a_skeleton_manifest(self, media, capsys):
        assert main(["library", str(media), "--interval", "10", "--cols", "3"]) == 0
        manifest = lib.load(str(media))
        assert manifest["clips_count"] == 2
        assert manifest["status"] == "sheets_built"
        # the judge has not run, so every clip is one whole-length segment
        assert all(c["needs_review"] for c in manifest["clips"])


class TestExport:
    @needs_ffmpeg
    def test_writes_a_default_named_timeline(self, judged, capsys):
        assert main(["export", str(judged), "--no-windows-paths"]) == 0
        assert (judged / "focalcut" / "selects.xml").exists()

    @needs_ffmpeg
    @pytest.mark.parametrize("fmt,suffix", [("fcp7", ".xml"), ("edl", ".edl"), ("otio", ".otio")])
    def test_honours_the_format_flag(self, judged, tmp_path, fmt, suffix):
        out = tmp_path / ("t" + suffix)
        assert main(["export", str(judged), "-f", fmt, "-o", str(out), "--no-windows-paths"]) == 0
        assert out.stat().st_size > 0

    @needs_ffmpeg
    def test_errors_without_a_library(self, tmp_path, capsys):
        assert main(["export", str(tmp_path)]) == 1
        assert "run `focalcut library` first" in capsys.readouterr().err

    @needs_ffmpeg
    def test_skips_clips_the_judge_marked_non_couple(self, judged, tmp_path):
        manifest = lib.load(str(judged))
        manifest["clips"][1]["kind"] = "non-couple"
        lib.save(str(judged), manifest["clips"], {})
        out = tmp_path / "t.otio"
        main(["export", str(judged), "-f", "otio", "-o", str(out), "--no-windows-paths"])
        data = json.loads(out.read_text())
        names = [c["name"] for c in data["tracks"]["children"][0]["children"]]
        assert names == ["a-01", "a-02"]
