"""Shared fixtures.

Everything here is synthesised with ffmpeg at session scope — the suite never needs a
GPU, network access, the gated SAM 3 weights, or any checked-in media.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _synth(path, seconds, w, h, rate, source="testsrc"):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"{source}=size={w}x{h}:rate={rate}:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    """A folder holding two portrait clips: a.mp4 (30s @50fps), b.mp4 (20s @50fps)."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    d = tmp_path_factory.mktemp("media")
    _synth(d / "a.mp4", 30, 1080, 1920, 50)
    _synth(d / "b.mp4", 20, 1080, 1920, 50, source="testsrc2")
    return d


@pytest.fixture
def clips():
    """Library records with a cut in the middle of a.mp4."""
    return [
        {"name": "a.mp4", "segments": [[0.0, 12.5], [12.5, 30.0]]},
        {"name": "b.mp4", "segments": [[2.0, 11.0]]},
    ]
