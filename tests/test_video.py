"""Probing and the scaling maths.

`probe` is the single place focalcut learns a clip's true display orientation, so the
rotation handling is worth pinning down: phone footage is routinely stored landscape
with a 90 degree rotation flag, and a timeline built on the stored dimensions comes out
sideways.
"""
from __future__ import annotations

import pytest

from focalcut import video as V

from .conftest import needs_ffmpeg


class TestProbe:
    @needs_ffmpeg
    def test_reads_dimensions_fps_and_duration(self, media):
        info = V.probe(str(media / "a.mp4"))
        assert (info.width, info.height) == (1080, 1920)
        assert info.fps == pytest.approx(50.0)
        assert info.duration == pytest.approx(30.0, abs=0.2)

    @needs_ffmpeg
    def test_rotated_source_reports_upright_display_dimensions(self, tmp_path):
        import subprocess
        flat, src = tmp_path / "flat.mp4", tmp_path / "rot.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=1920x1080:rate=25:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(flat)],
            check=True, capture_output=True)
        # a display matrix is the way phones actually flag rotation; the older
        # `-metadata rotate=` tag is no longer written by current ffmpeg
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90",
             "-i", str(flat), "-c", "copy", str(src)],
            check=True, capture_output=True)
        info = V.probe(str(src))
        # stored 1920x1080 landscape + a 90 degree flag == a portrait clip
        assert (info.width, info.height) == (1080, 1920)


class TestTargetSize:
    def test_landscape_scales_the_width_to_long_side(self):
        info = V.VideoInfo("x", 1920, 1080, 25, 10)
        assert V.target_size(info, 512) == (512, 288)

    def test_portrait_scales_the_height_to_long_side(self):
        info = V.VideoInfo("x", 1080, 1920, 25, 10)
        assert V.target_size(info, 512) == (288, 512)

    def test_dimensions_are_always_even(self):
        # odd dimensions make libx264 and several filters fail
        for w, h in [(1000, 563), (999, 1777), (1234, 567)]:
            tw, th = V.target_size(V.VideoInfo("x", w, h, 25, 10), 511)
            assert tw % 2 == 0 and th % 2 == 0

    def test_square_input_stays_square(self):
        assert V.target_size(V.VideoInfo("x", 800, 800, 25, 10), 256) == (256, 256)

    def test_aspect_ratio_is_preserved(self):
        w, h = V.target_size(V.VideoInfo("x", 1920, 1080, 25, 10), 640)
        assert w / h == pytest.approx(1920 / 1080, abs=0.01)
