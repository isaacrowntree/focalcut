"""The reel's size (reel.json "resolution") through the project, the shot filter and cache key, the MilkDrop layer,
the encoder arguments and the XML; and the render's streamed transition rewrite."""
import json
import subprocess

import numpy as np
import pytest

from focalcut.reel import hw, overlay, render
from focalcut.reel.project import Project, frame_size


def project(tmp_path, config=None, cut=None):
    reel = tmp_path / "focalcut" / "reel"
    reel.mkdir(parents=True, exist_ok=True)
    (reel / "reel.json").write_text(json.dumps(config or {}))
    return Project(str(tmp_path), cut)


@pytest.mark.parametrize("res,land,port", [
    (None, (1920, 1080), (1080, 1920)),
    ("1080p", (1920, 1080), (1080, 1920)),
    ("4k", (3840, 2160), (2160, 3840)),
    ("4K", (3840, 2160), (2160, 3840)),
    ("UHD", (3840, 2160), (2160, 3840)),
    ("2160p", (3840, 2160), (2160, 3840)),
    ("1440p", (2560, 1440), (1440, 2560)),
    ("720p", (1280, 720), (720, 1280)),
    ([3840, 2160], (3840, 2160), (2160, 3840)),
    ([2160, 3840], (3840, 2160), (2160, 3840)),                 # either way round: the format decides
    ({"landscape": "4k"}, (3840, 2160), (1080, 1920)),          # per format; the other stays 1080p
    ({"landscape": "4k", "portrait": [1216, 2160]}, (3840, 2160), (1216, 2160)),
])
def test_frame_size(res, land, port):
    assert frame_size(res, "landscape") == land and frame_size(res, "portrait") == port


@pytest.mark.parametrize("bad", ["huge", "p", [3840], [3840, 2160, 1], "0p", [0, 0]])
def test_frame_size_says_what_is_wrong(bad):
    with pytest.raises(ValueError, match="resolution"):
        frame_size(bad, "landscape")


def test_project_size_and_a_cut_of_its_own(tmp_path):
    cfg = {"resolution": "1080p", "cuts": {"short": {"resolution": "4k"}, "plain": {}}}
    assert project(tmp_path, cfg).size("portrait") == (1080, 1920)
    assert project(tmp_path, cfg, "short").size("portrait") == (2160, 3840)
    assert project(tmp_path, cfg, "plain").size("landscape") == (1920, 1080)
    assert project(tmp_path, {}).size("landscape") == (1920, 1080)
    assert render.size(project(tmp_path, {"resolution": "4k"}), "landscape") == (3840, 2160)
    assert render.size(None, "portrait") == (1080, 1920)        # no project: 1080p


def shot(**over):
    return dict(dict(unit=dict(id="C1", source="C1.MP4", rotate=None, role="dance"), src_in=1.0, length=2.0,
                     frames=100, fx=[], crop=dict(w=0.3164, h=1.0, z=1.0, path=[[0, 0.5, 0.5]]), peak=1.0,
                     kind="spin"), **over)


def test_shot_filter_scales_to_the_projects_size(tmp_path):
    s = shot()
    assert "scale=1080:1920:flags=lanczos" in render.shot_filter(s, "portrait", project(tmp_path))
    f = render.shot_filter(s, "portrait", project(tmp_path, {"resolution": "4k"}))
    assert "scale=2160:3840:flags=lanczos" in f and f.index("crop=") < f.index("scale=") and f.endswith("format=yuv420p")
    assert "scale=3840:2160" in render.shot_filter(s, "landscape", project(tmp_path, {"resolution": "4k"}))
    assert "scale=1920:1080" in render.shot_filter(s, "landscape")           # no project: 1080p, as before


def test_cache_key_follows_the_size_and_the_encode_but_not_the_seed(monkeypatch):
    s = shot()
    hd, uhd = render.key(s, 0.5, (1920, 1080)), render.key(s, 0.5, (3840, 2160))
    assert hd != uhd and hd == render.key(dict(s), 0.5, (1920, 1080))
    assert render.key(s, 0.5, (1920, 1080)) != render.key(shot(src_in=1.02), 0.5, (1920, 1080))
    seed = render.key(s)                                        # the effects' seed: the shot alone, at any size
    monkeypatch.setattr(hw, "ENC_VERSION", hw.ENC_VERSION + 1)
    assert render.key(s, 0.5, (1920, 1080)) != hd and render.key(s) == seed


def test_read_length_covers_a_paced_shot():
    assert render.read_length(shot()) == pytest.approx(2.1)
    assert render.read_length(shot(speed=1.25)) == pytest.approx(2.6)


def test_layer_size_keeps_a_1080p_canvas():
    opts = overlay.options({"engine": "milkdrop"})
    assert overlay.layer_size(1920, 1080, opts) == (1920, 1080)
    assert overlay.layer_size(3840, 2160, opts) == (1920, 1080)
    assert overlay.layer_size(2160, 3840, opts) == (1080, 1920)
    assert overlay.layer_size(1080, 1920, opts) == (1080, 1920)
    assert overlay.layer_size(3840, 2160, dict(opts, milkdrop_side=2560)) == (2560, 1440)
    assert overlay.layer_size(1280, 720, opts) == (1280, 720)                # never bigger than the reel
    assert "yuv444p" in overlay.layer_enc()


def test_effect_canvas_is_the_same_at_4k():
    assert overlay.effect_divisor(1920, 1080) == 2 and overlay.effect_divisor(1080, 1920) == 2
    assert overlay.effect_divisor(3840, 2160) == 4 and overlay.effect_divisor(2160, 3840) == 4
    assert overlay.effect_divisor(320, 180) == 2
    a = overlay.Show(1920, 1080, 50, 0.5, np.random.default_rng(0))
    b = overlay.Show(3840, 2160, 50, 0.5, np.random.default_rng(0))
    assert (a.w, a.h) == (b.w, b.h) == (960, 540)


def test_whites_follow_the_cuts_flashes():
    beat = 0.48
    plan = dict(beat=beat, shots=[dict(frames=50), dict(frames=50, transition="flash"), dict(frames=50, transition="swipe"),
                                  dict(frames=6, transition="dipwhite"), dict(frames=50, transition="dipwhite")])
    w = overlay.whites(plan, 206, 50)
    assert w.shape == (206,) and w[50] == pytest.approx(1.0) and w[49] > 0.5 and w[30] == 0 and w[70] == 0
    assert w[94:106].max() == 0                                 # a swipe is not a flash
    assert w[150] == pytest.approx(1.0) and w[146] == 0 and w[148] > 0      # no longer than the short shot after it
    assert w[156] == pytest.approx(1.0) and 0 < w[157] < 1 and w[170] == 0   # ... or the one before


def test_final_encode_has_a_keyframe_a_second_and_bt709(monkeypatch):
    monkeypatch.setattr(hw, "nvenc", lambda: False)
    e, f = hw.enc(), hw.enc(final=True)
    assert "-g" not in e and f[f.index("-g") + 1] == "50" and f[:len(e)] == e
    assert e[e.index("-pix_fmt") + 1] == "yuv420p" and e[e.index("-colorspace") + 1] == "bt709"


def test_run_enc_waits_for_an_encoder_session(monkeypatch):
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        busy = len(calls) < 3
        return subprocess.CompletedProcess(cmd, 1 if busy else 0, "", "OpenEncodeSessionEx failed" if busy else "")
    monkeypatch.setattr(hw.subprocess, "run", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    hw.run_enc(["ffmpeg", "x"])
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(hw.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "No such file"))
    with pytest.raises(subprocess.CalledProcessError):          # any other failure: at once
        hw.run_enc(["ffmpeg", "x"])


def test_splice_replaces_the_tail_and_streams_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "nvenc", lambda: False)
    W, H, n = 64, 36, 12
    src, out = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    frames = [np.full((H, W, 3), 20 + 15 * k, np.uint8) for k in range(n)]
    render.write_frames(frames, src)
    assert render.frame_count(src) == n and render.frame_size(src) == (W, H)
    tail = render.read_frames(src, n - 3)
    assert len(tail) == 3 and abs(int(tail[0].mean()) - (20 + 15 * 9)) <= 2
    head = render.read_frames(src, 0, 2)
    assert len(head) == 2 and abs(int(head[1].mean()) - 35) <= 2
    render.splice(src, out, n - 3, [np.full((H, W, 3), 250, np.uint8)] * 3)
    got = render.read_frames(out)
    assert len(got) == n
    assert [abs(int(g.mean()) - (20 + 15 * k)) <= 2 for k, g in enumerate(got[:n - 3])] == [True] * (n - 3)
    assert all(g.mean() > 245 for g in got[n - 3:])
    render.splice(src, out, 0, [np.full((H, W, 3), 250, np.uint8)] * 2)       # ... and a head
    got = render.read_frames(out)
    assert len(got) == n and got[0].mean() > 245 and got[1].mean() > 245 and abs(int(got[2].mean()) - 50) <= 2


def test_timer_adds_up_stages():
    t = render.Timer()
    with t("shots"):
        pass
    t.add("shots", 2.0)
    t.add("overlay", 1.26)
    r = t.report()
    assert r["shots"] >= 2.0 and r["overlay"] == 1.3 and "total" in r
