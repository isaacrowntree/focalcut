"""Speed curves: a 1x stretch plays the source's own frames (snap_steps), a curve streams its source a few frames
at a time and gives what it gives from a list, on RGB and on raw 4:2:0 frames; and a shot's cache key follows what
shapes its pixels and nothing else."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from focalcut import fx
from focalcut.fx import speed as SP
from focalcut.reel import render


def ctx(peak=1.4, beat=0.5, **params):
    return fx.Context(fps=50, beat=beat, peak=peak, params={"cap": 1.5, **params})


def test_snap_steps_puts_a_1x_stretch_on_a_whole_frame():
    step = np.r_[np.ones(10), np.full(20, 1.4), np.full(30, 0.33), np.ones(40)]
    assert abs(step[:60].sum() - round(step[:60].sum())) > 0.05           # the ramp ends part-way into a frame
    out = SP.snap_steps(step)
    assert abs(out[:60].sum() - round(out[:60].sum())) < 1e-9             # ... and now on one
    assert np.array_equal(out[:10], step[:10]) and np.array_equal(out[60:], step[60:])   # 1x stays 1x
    assert abs(out.sum() - step.sum()) <= 0.5 + 1e-9                      # half a source frame at most
    assert np.all(np.abs(out[10:60] / step[10:60] - 1) < 0.02)            # a per cent or so of the ramp's speed
    assert np.all(out[10:30] > 1) and np.all(out[30:60] < 1)                  # fast stays fast, slow stays slow


def test_snap_steps_leaves_what_it_cannot_nudge():
    flat = np.full(50, 0.8)                                               # no 1x stretch (a glide): nothing to do
    assert np.array_equal(SP.snap_steps(flat), flat)
    blip = np.r_[np.ones(5), [1.004], np.ones(5)]                         # a ramp too slight to carry the nudge
    assert np.array_equal(SP.snap_steps(blip), blip)
    assert np.array_equal(SP.snap_steps(np.ones(20)), np.ones(20))


@pytest.mark.parametrize("name", ["hero", "bullet", "montage", "jumpcut", "flashin", "fit"])
def test_every_frame_at_1x_is_a_whole_source_frame(name):
    c = ctx(src_len=5.0) if name == "fit" else ctx()
    _, idx, s = fx.EFFECTS[name]()._index(4.0, c)
    one = np.abs(s - 1) < 1e-9
    run = one[:-1] & one[1:]                                              # frames inside a 1x stretch
    assert run[-20:].all()                                                # the tail is one
    assert np.array_equal(idx[1:][run], np.rint(idx[1:][run]))
    assert np.all(np.diff(idx) > 0)                                       # time still only moves forward


def test_snapping_moves_the_source_by_half_a_frame_at_most():
    e = fx.EFFECTS["hero"]()
    for peak in (0.9, 1.4, 2.2):
        c = ctx(peak)
        _, src, s = e._map(4.0, c)
        plain = np.concatenate([[0], np.cumsum((s[1:] + s[:-1]) / 2 / 50)])
        assert np.abs(src - plain).max() <= 0.5 / 50 + 1e-9
        assert e.source_length(4.0, c) >= src[-1] + 1 / 50


def marked(n, h=36, w=64):
    """n frames, each a texture moved along and its own index written in a corner: told apart exactly."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    out = []
    for k in range(n):
        f = np.roll(base, 2 * k, axis=1).copy()
        f[0, 0], f[0, 1] = k % 256, k // 256
        out.append(f)
    return out


def test_a_1x_stretch_plays_the_source_frames_themselves():
    src = marked(260)
    c = ctx()
    c.out_frames = 200
    e = fx.EFFECTS["hero"]()
    out = e.apply(src, c)
    _, m, s = e._index(4.0, c)
    one = np.abs(s[:200] - 1) < 1e-9
    assert one.mean() > 0.4 and one[-20:].all()
    for k in np.nonzero(one)[0]:
        assert out[k] is src[int(m[k])] and m[k] == int(m[k])                     # not a copy, not an in-between
    assert any(not any(o is f for f in src) for o in out)                 # the slow stretch is made of in-betweens


@pytest.mark.parametrize("name,params", [("hero", {}), ("hero", {"face": True}), ("jumpcut", {"cap": 3.5}),
                                         ("glide", {}), ("hero", {"pace": 1.15})])
def test_streaming_gives_what_the_whole_list_gives_and_holds_a_few_frames(name, params):
    src = marked(150)
    c = ctx(1.0, **params)
    c.out_frames = 120
    e = fx.EFFECTS[name]()
    whole = e.apply(src, c)
    held, seen = [0], [0]

    def feed():
        for f in src:
            seen[0] += 1
            yield f

    class Spy(SP.Window):
        def need(self, i):
            r = super().need(i)
            held[0] = max(held[0], len(self.buf))
            return r
    real, SP.Window = SP.Window, Spy
    try:
        got = list(e.stream(feed(), c, 120))
    finally:
        SP.Window = real
    assert len(got) == len(whole) == 120
    assert all(np.array_equal(a, b) for a, b in zip(got, whole, strict=True))
    assert held[0] <= 8                                                   # never the whole shot


def test_a_curve_past_its_sources_end_holds_the_last_frame():
    src = marked(30)
    c = ctx(0.3)
    c.out_frames = 60
    out = fx.EFFECTS["hero"]().apply(src, c)
    assert len(out) == 60 and out[-1] is src[-1]


def yuv_of(rgb):
    h, w = rgb.shape[:2]
    return SP.Yuv(render.rgb_to_yuv420(rgb), w, h)


def test_raw_yuv_frames_pass_through_as_their_own_bytes_and_blend_like_rgb():
    rgb = marked(150)
    src = [yuv_of(f) for f in rgb]
    c = ctx(1.0)
    c.out_frames = 120
    e = fx.EFFECTS["hero"]()
    out = list(e.stream(iter(src), c, 120))
    _, m, s = e._index(120 / 50, c)
    one = np.abs(s[:120] - 1) < 1e-9
    for k in np.nonzero(one)[0]:
        assert out[k].raw is src[int(m[k])].raw                           # the decoded bytes go to the encoder
    made = [o for o in out if not any(o is f for f in src)]
    assert made and all(len(bytes(o.raw)) == 64 * 36 * 3 // 2 for o in made)
    a, b = np.full((36, 64, 3), 60, np.uint8), np.full((36, 64, 3), 180, np.uint8)   # two flat frames: the
    dis = SP.cv2.DISOpticalFlow_create(SP.cv2.DISOPTICAL_FLOW_PRESET_FAST)             # in-between is their mix
    mid_rgb = SP.Flows(dis, a, b).between(a, b, np.float32(0.25))
    mid_yuv = SP.Flows(dis, yuv_of(a), yuv_of(b)).between(yuv_of(a), yuv_of(b), np.float32(0.25))
    assert abs(int(mid_rgb.mean()) - 90) <= 1
    want = yuv_of(np.full((36, 64, 3), 90, np.uint8))
    assert abs(int(mid_yuv.y.mean()) - int(want.y.mean())) <= 1 and abs(int(mid_yuv.u.mean()) - 128) <= 1


def test_grid_is_made_once_per_size():
    assert SP._grid(64, 36) is SP._grid(64, 36)
    g = SP._grid(8, 4)
    assert g.shape == (4, 8, 2) and g.dtype == np.float32 and tuple(g[3, 5]) == (5.0, 3.0)


# -- the cache key ------------------------------------------------------------------------------------------------
def shot(**over):
    return dict(dict(unit=dict(id="C1", source="C1.MP4", rotate=None, role="dance", t0=0.0, t1=60.0), src_in=1.0,
                     length=2.0, frames=100, fx=[], t=2.0, peak=1.0, kind="spin",
                     crop=dict(w=0.5, h=1.0, z=1.0, path=[[0, 0.5, 0.5]], face=0.2, action=1.1, fill=0.9, churn=0.0)),
                **over)


def key(s, beat=0.5):
    return render.key(s, beat, (1920, 1080))


def test_a_plain_shots_key_ignores_the_planners_scores_and_the_effects_fields():
    s = shot()
    rescored = shot(crop=dict(s["crop"], face=0.9, action=0.2, fill=0.1, churn=0.4, low=0.3), peak=1.7, kind="closeup",
                    rank=3.0, act="finale", t=2.5)
    rescored["unit"] = dict(s["unit"], role="transition")
    assert key(rescored) == key(s)
    assert key(s, beat=0.61) == key(s) and key(shot(beat=0.47)) == key(s)   # a plain shot has no beat
    assert key(shot(crop=dict(s["crop"], path=[[0, 0.52, 0.5]]))) != key(s)   # the picture's own things do count
    assert key(shot(crop=dict(s["crop"], w=0.45))) != key(s)
    assert key(shot(src_in=1.02)) != key(s) and key(shot(frames=101)) != key(s)
    assert key(shot(gain=1.2)) != key(s) and key(shot(logo="in")) != key(s)


def test_an_effect_shots_key_follows_what_its_effects_read():
    s = shot(fx=["hero"])
    assert key(s) != key(shot())
    assert key(shot(fx=["hero"], crop=dict(s["crop"], action=0.1, fill=0.2))) == key(s)   # scores: still not
    assert key(shot(fx=["hero"], peak=1.2)) != key(s) and key(shot(fx=["hero"], t=2.2)) != key(s)
    assert key(shot(fx=["hero"], kind="closeup")) != key(s)
    assert key(shot(fx=["hero"], crop=dict(s["crop"], face=0.8))) != key(s)   # a face is kept sharp: other frames
    assert key(shot(fx=["hero"], group=True)) != key(s) and key(shot(fx=["hero"], add=True)) != key(s)
    # the beat fx_shot renders with is the shot's own where the plan gives it one, else the plan's
    assert key(s, beat=0.61) != key(s)
    own = shot(fx=["hero"], beat=0.47)
    assert key(own) != key(s) and key(own, beat=0.61) == key(own)
    assert key(shot(fx=["hero"], beat=0.5)) == key(s)


def test_shot_files_are_named_by_key_so_a_shot_added_earlier_re_renders_nothing(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(render, "cut_shot", lambda project, s, fmt, out, qsv=False: (made.append(s["src_in"]),
                                                                                    open(out, "w").close()))
    monkeypatch.setattr(render.hw, "win_ffmpeg", lambda: None)
    a, b, c = shot(src_in=1.0), shot(src_in=5.0), shot(src_in=9.0)
    first = render.render_shots(None, dict(beat=0.5, shots=[a, b]), "landscape", str(tmp_path))
    assert sorted(made) == [1.0, 5.0]
    again = render.render_shots(None, dict(beat=0.5, shots=[c, a, b, a]), "landscape", str(tmp_path))
    assert made[2:] == [9.0]                                              # only the new shot was rendered
    assert again[1:3] == first and again[3] == again[1]                   # in the plan's order; a repeat is one file
    assert all("shot_" + render.key(s, 0.5, (1920, 1080)) in f for s, f in zip([c, a, b, a], again, strict=True))


def test_file_id_is_the_content_not_the_time(tmp_path):
    f = tmp_path / "look.cube"
    f.write_text("LUT one")
    one = render.file_id(str(f))
    import os
    os.utime(f, (1, 1))                                                   # written again, the same look
    assert render.file_id(str(f)) == one
    f.write_text("LUT two, another look")
    assert render.file_id(str(f)) != one and render.file_id(str(tmp_path / "none.cube")) is None


def test_ahead_runs_an_iterator_on_its_own_thread_and_passes_its_error_on():
    assert list(render.Ahead(iter(range(50)), n=2)) == list(range(50))

    def bad():
        yield 1
        raise ValueError("decode failed")
    it = iter(render.Ahead(bad()))
    assert next(it) == 1
    with pytest.raises(ValueError, match="decode failed"):
        next(it)
    slow = render.Ahead(iter(range(10 ** 6)), n=2)                        # a reader that stops early lets it go
    assert next(iter(slow)) == 0
    slow.stop()
    slow.thread.join(2)
    assert not slow.thread.is_alive()
