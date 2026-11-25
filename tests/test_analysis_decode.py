"""The analysis steps' decoding: which decoder reads a file (hw.readers), score's frames and their times through a
proxy, the reference pass's one-pass frame reads and the times its unit frames carry."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from focalcut.reel import hw, reference, score, shake
from focalcut.reel.project import Project

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def clip(path, seconds=6, fps=25, gop=25, size="96x54"):
    """A small H.264 file whose picture changes every frame (a moving test pattern), a keyframe every gop frames."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}:d={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", str(gop), "-keyint_min", str(gop),
                    "-sc_threshold", "0", "-bf", "0", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_readers_try_the_gpu_first_and_always_end_on_the_cpu(monkeypatch):
    monkeypatch.delenv("FOCALCUT_ANALYSIS_CPU", raising=False)
    monkeypatch.setattr(hw, "nvdec", lambda: True)
    monkeypatch.setattr(hw, "win_ffmpeg", lambda: None)
    fmt = {"a.mp4": "yuv420p", "b.MP4": "yuv422p10le", "c.MP4": "yuv420p10le"}
    monkeypatch.setattr(hw, "pix_fmt", lambda src: fmt[src])
    kinds = {f: [r[0] for r in hw.readers(f)] for f in fmt}
    assert kinds == {"a.mp4": ["cuda", "cpu"], "b.MP4": ["cpu"], "c.MP4": ["cuda", "cpu"]}   # NVDEC: 4:2:0 only
    cuda = hw.readers("c.MP4")[0]
    assert cuda[2] == ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] and cuda[4] == "hwdownload,format=p010le,"
    assert hw.readers("a.mp4")[0][4] == "hwdownload,format=nv12,"
    assert [r[0] for r in hw.readers("a.mp4", gpu=False)] == ["cpu"]
    monkeypatch.setenv("FOCALCUT_ANALYSIS_CPU", "1")
    assert [r[0] for r in hw.readers("a.mp4")] == ["cpu"]
    monkeypatch.delenv("FOCALCUT_ANALYSIS_CPU")
    monkeypatch.setattr(hw, "nvdec", lambda: False)
    assert [r[0] for r in hw.readers("a.mp4")] == ["cpu"]


def test_readers_send_a_422_camera_file_to_quick_sync(monkeypatch):
    monkeypatch.delenv("FOCALCUT_ANALYSIS_CPU", raising=False)
    monkeypatch.setattr(hw, "nvdec", lambda: True)
    monkeypatch.setattr(hw, "win_ffmpeg", lambda: "C:/ffmpeg.exe")
    monkeypatch.setattr(hw, "pix_fmt", lambda src: "yuv422p10le")
    monkeypatch.setattr(hw, "on_windows_drive", lambda p: True)
    monkeypatch.setattr(hw, "winpath", lambda p: "J:\\x.MP4")
    (kind, exe, dec, P, download), last = hw.readers("/mnt/j/x.MP4")
    assert kind == "qsv" and exe == "C:/ffmpeg.exe" and "qsv" in dec and download == "hwdownload,format=y210le,"
    assert P("/mnt/j/x.MP4") == "J:\\x.MP4" and last[0] == "cpu"


def test_source_time_is_the_inverse_of_decode_from(tmp_path):
    reel = tmp_path / "focalcut" / "reel"
    reel.mkdir(parents=True)
    for shifts in ([], [[3.0, -1.0]], [[2.0, 1.5]], [[3.0, -1.0], [10.0, -0.5], [20.0, 2.0]]):
        (reel / "reel.json").write_text(json.dumps({"proxies": {"C1.MP4": ["px.mp4", shifts]}}))
        pr = Project(str(tmp_path))
        for t in (0.0, 1.9, 3.0, 4.2, 9.99, 10.0, 15.5, 20.0, 31.0):      # a source time, to the proxy and back
            if any(at - 1.0 <= t < at for at, d in shifts if d < 0):
                continue                                                  # (inside a dropped stretch: not in the proxy)
            assert score.source_time(shifts, pr.decode_from("C1.MP4", t)[1]) == pytest.approx(t)


def test_score_frame_size_is_long_side_640_and_even():
    assert score.frame_size(3840, 2160) == (640, 360) and score.frame_size(2160, 3840) == (360, 640)
    w, h = score.frame_size(1920, 818)
    assert w == 640 and h % 2 == 0 and abs(h - 640 * 818 / 1920) <= 1


@needs_ffmpeg
def test_score_frames_come_at_the_analysis_rate_from_any_reader(tmp_path, monkeypatch):
    f = clip(tmp_path / "a.mp4", seconds=4)
    monkeypatch.setenv("FOCALCUT_ANALYSIS_CPU", "1")
    got = list(score.frames(f, 96, 54))
    assert len(got) == 4 * score.FPS and got[0].shape == (360, 640, 3)
    assert not np.array_equal(got[0], got[1])
    # a decoder that gives nothing is passed over for the next
    monkeypatch.setattr(hw, "readers", lambda src, gpu=True: [("cuda", "ffmpeg", ["-hwaccel", "nope"], str, ""),
                                                              ("cpu", "ffmpeg", [], str, "")])
    again = list(score.frames(f, 96, 54))
    assert len(again) == len(got) and all(np.array_equal(a, b) for a, b in zip(again, got, strict=True))


@needs_ffmpeg
def test_score_reads_the_proxy_and_writes_the_sources_own_times(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setenv("FOCALCUT_ANALYSIS_CPU", "1")
    reel = tmp_path / "focalcut" / "reel"
    reel.mkdir(parents=True)
    clip(tmp_path / "px.mp4", seconds=3)                     # the proxy: 3 s -- the source without its 2.0-3.0 s
    (reel / "reel.json").write_text(json.dumps({"proxies": {"C1.MP4": ["px.mp4", [[3.0, -1.0]]]}}))
    pr = Project(str(tmp_path))
    opened = []
    real = score.frames
    monkeypatch.setattr(score, "frames", lambda path, w, h, gpu=True: (opened.append(path), real(path, w, h, gpu))[1])
    monkeypatch.setattr(score, "people", lambda batch: ([(np.zeros((0, 4)), np.zeros(0)) for _ in batch],
                                                        [np.zeros((0, 17, 3), np.float32) for _ in batch]))
    out = tmp_path / "scores"
    msg = score.score(str(tmp_path), "C1.MP4", str(out), project=pr)
    assert opened == [str(tmp_path / "px.mp4")] and "proxy" in msg
    z = np.load(out / "C1.npz")
    assert len(z["t"]) == 3 * score.FPS and list(out.iterdir()) == [out / "C1.npz"]      # (no part file left)
    assert np.allclose(z["t"][:20], np.arange(20) / score.FPS)                            # before the join: as is
    assert np.allclose(z["t"][20:], np.arange(20, 30) / score.FPS + 1.0)                  # after it: a second on
    assert np.allclose(z["pose_t"], z["t"][::2])
    assert score.score(str(tmp_path), "C1.MP4", str(out), project=pr).endswith("cached")


@needs_ffmpeg
def test_shake_grey_frames_are_the_cpu_decodes_whatever_reader_fails_first(tmp_path, monkeypatch):
    f = clip(tmp_path / "a.mp4", seconds=2, fps=50)
    monkeypatch.setenv("FOCALCUT_ANALYSIS_CPU", "1")
    g = shake.grey_frames(f, 0.0, 2.0, 64, 36, 25)
    assert g.shape == (50, 36, 64)
    monkeypatch.setattr(hw, "readers", lambda src, gpu=True: [("cuda", "ffmpeg", ["-hwaccel", "nope"], str, ""),
                                                              ("cpu", "ffmpeg", [], str, "")])
    assert np.array_equal(shake.grey_frames(f, 0.0, 2.0, 64, 36, 25), g)


@needs_ffmpeg
def test_frames_at_gives_frame_by_frames_pixels_in_one_pass(tmp_path, monkeypatch):
    f = clip(tmp_path / "film.mp4", seconds=6)
    times = [0.0, 0.513, 1.0, 1.02, 2.96, 4.44, 5.9]         # (two of them inside one frame: the same frame twice)
    want = [reference.frame(f, t) for t in times]
    calls = []
    real = reference.frame
    monkeypatch.setattr(reference, "frame", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    got = list(reference.frames_at(f, times))
    assert len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))
    assert not calls                                          # no seek for any of them
    small = list(reference.frames_at(f, times, 48))
    assert all(np.array_equal(a, real(f, t, 48)) for a, t in zip(small, times, strict=True))
    assert list(reference.frames_at(f, [])) == []
    late = list(reference.frames_at(f, [1.0, 0.5]))           # not ascending: read one by one, still right
    assert np.array_equal(late[0], real(f, 1.0)) and np.array_equal(late[1], real(f, 0.5))


@needs_ffmpeg
def test_unit_frames_carry_their_own_times_from_keyframes_or_every_frame(tmp_path):
    often = clip(tmp_path / "often.mp4", seconds=9, gop=25)   # a keyframe a second: the keyframes are enough
    ts, fr = reference._grab(often, 1.0, 8.0, 2.0, 48)
    assert ts == [0.0, 2.0, 4.0, 6.0] and fr.shape == (4, 28, 48, 3)
    for t, f in zip(ts, fr, strict=True):                     # each is the frame at the time it says
        assert np.array_equal(f, reference.frame(often, 1.0 + t, 48))
    rare = clip(tmp_path / "rare.mp4", seconds=9, gop=250)    # one keyframe in all: every frame is decoded
    ts2, fr2 = reference._grab(rare, 1.0, 8.0, 2.0, 48)
    assert len(ts2) == 4 and np.allclose(ts2, [0.0, 2.0, 4.0, 6.0], atol=0.05)
    assert all(np.array_equal(f, reference.frame(rare, 1.0 + t, 48)) for t, f in zip(ts2, fr2, strict=True))


def test_embed_units_keeps_what_the_store_already_has(tmp_path):
    reel = tmp_path / "focalcut" / "reel"
    reel.mkdir(parents=True)
    (reel / "reel.json").write_text("{}")
    pr = Project(str(tmp_path))
    exp = tmp_path / "export" / "show"
    exp.mkdir(parents=True)
    for n in ("A", "B"):
        (exp / f"{n}.mp4").write_bytes(b"x")
    units = [dict(id="A", category="show", source="A.MP4"), dict(id="B", category="show", source="B.MP4")]

    class E:
        seen = 0

        def images(self, imgs, batch=32):
            E.seen += len(imgs)
            return np.full((len(imgs), 768), 2.0, np.float32)
    idx, times, imgs = [0, 0, 1], [0.0, 2.0, 0.0], [np.zeros((2, 2, 3), np.uint8)] * 3
    emb, kept = reference.embed_units(pr, E(), units, idx, times, imgs)
    assert kept == 0 and E.seen == 3                          # no store yet
    store = dict(unit=np.array(["A", "A"]), t=np.array([0.0, 2.0]), emb=np.full((2, 768), 7.0, np.float16))
    np.savez(pr.path("reference_frames.npz"), **store)        # an old store (its times were a step early): not used
    assert reference.embed_units(pr, E(), units, idx, times, imgs)[1] == 0
    np.savez(pr.path("reference_frames.npz"), store=np.array(reference.STORE), **store)
    E.seen = 0
    emb, kept = reference.embed_units(pr, E(), units, idx, times, imgs)
    assert kept == 2 and E.seen == 1 and emb[0, 0] == 7.0 and emb[1, 0] == 7.0 and emb[2, 0] == 2.0
    import os
    later = os.path.getmtime(pr.path("reference_frames.npz")) + 60
    os.utime(exp / "A.mp4", (later, later))                   # A exported again since: its frames are embedded again
    assert reference.embed_units(pr, E(), units, idx, times, imgs)[1] == 0
