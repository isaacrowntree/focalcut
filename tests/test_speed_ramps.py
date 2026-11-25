"""Speed ramps by what moves (fx.speed.RAMPS, docs/research/speed-ramps.md): the speed never jumps, slow motion
stops at what the source's frame rate carries, the action lands on the music, and a ramp is set from reel.json."""
import numpy as np
import pytest

from focalcut import fx
from focalcut.fx import speed as SP

BEAT = 0.49172                                             # 122 BPM: a bar of 1.97 s, a dunk shot three bars


def ctx(peak, src_fps=50, pace=1.0, **params):
    return fx.Context(fps=50, beat=BEAT, peak=peak, seed=1,
                      params={"cap": 1.5, "pace": pace, "src_fps": src_fps, **params})


def curve(name, L, c):
    t = np.arange(round(L * 50) + 1) / 50
    return t, fx.EFFECTS[name]().speed(t, L, c)


@pytest.mark.parametrize("name", SP.RAMP_FX)
@pytest.mark.parametrize("L,peak", [(5.9, 3.66), (3.93, 2.44), (2.95, 1.83), (7.87, 1.0), (1.97, 1.9)])
@pytest.mark.parametrize("src_fps,pace", [(50, 1.0), (25, 1.0), (25, 4.0)])
def test_speed_is_continuous_and_eased(name, L, peak, src_fps, pace):
    _, v = curve(name, L, ctx(peak, src_fps, pace))
    step = np.abs(np.diff(v))
    assert step.max() <= 0.12 * pace, (name, step.max())              # no jump from one frame to the next
    assert np.abs(np.diff(v, 2)).max() <= 0.03 * pace                  # and no corner: the change itself eases
    assert v.min() >= min(SP.slow_floor(src_fps), pace) - 1e-9 and v.max() <= max(SP.MAX_FAST, pace) + 1e-9


@pytest.mark.parametrize("name", ["dunk", "rise", "reveal", "none"])
def test_ends_at_its_pace(name):
    t, v = curve(name, 5.9, ctx(3.66))
    assert np.allclose(v[t >= 5.9 - SP.TAIL], 1.0, atol=0.02)


def test_slow_floor_follows_the_source_frame_rate():
    assert SP.slow_floor(50) == 0.5 and SP.slow_floor(25) == 1.0 and SP.slow_floor(100) == 0.5
    assert SP.slow_floor(None) == 0.5 and SP.slow_floor(50, keep=50) == 1.0
    _, v50 = curve("dunk", 5.9, ctx(3.66, 50))
    _, v25 = curve("dunk", 5.9, ctx(3.66, 25))
    assert v50.min() == pytest.approx(0.5) and v25.min() == pytest.approx(1.0)   # 25p: never slowed, still ramped
    assert v25.max() == pytest.approx(2.0, abs=0.01)
    _, deep = curve("dunk", 5.9, ctx(3.66, 50, ramps={"dunk": {"slow": 0.2}}))
    assert deep.min() == pytest.approx(0.5)                                      # asked for deeper: the floor holds


def test_clean_speeds():
    assert SP.clean_speeds(25) == [1.0, 2.0, 4.0] and SP.clean_speeds(50) == [1.0, 2.0, 3.0, 4.0]
    assert 1.25 not in SP.clean_speeds(25)


def test_dunk_shape_and_where_it_lands():
    """Real time, quicker going under, slow as the face clears the water -- on the bar line -- then real time."""
    e, L, p = fx.EFFECTS["dunk"](), 5.9, 0.62 * 5.9
    c = ctx(p)
    land = e.land(L, c)
    assert land == pytest.approx(8 * BEAT, abs=1e-6)                   # the second bar line of a three-bar shot
    t, v = curve("dunk", L, c)
    assert np.allclose(v[t < land - 4 * BEAT - 0.02], 1.0)             # real time until a bar before the landing
    assert v[(t > land - 3 * BEAT) & (t < land - 2 * BEAT)].min() == pytest.approx(2.0, abs=0.01)
    slow = (t >= land - BEAT) & (t <= land + 1.9 * BEAT)               # slow from a beat before the face is out
    assert np.allclose(v[slow], 0.5, atol=0.01)
    assert (v < 0.55).sum() / 50 >= 1.2                                # long enough to be seen (hero's was 0.5 s)
    # the shot's moment (source time s["t"]) plays at the landing: render puts src_in at t - source_peak
    tt, src, _ = e._map(L, c)
    assert e.source_peak(p, c, L) == pytest.approx(float(np.interp(land, tt, src)), abs=1e-6)
    pre, post, at = SP.ramp_window("dunk", L, p, BEAT, params=c.params)
    assert at == pytest.approx(land) and pre > land and post < L - land    # more source before it, less after


def test_peak_is_left_alone_when_fixed_or_unsnapped():
    e, L, p = fx.EFFECTS["dunk"](), 5.9, 3.66
    assert e.land(L, ctx(p, peak_fixed=True)) == p
    assert e.land(L, ctx(p, ramps={"dunk": {"snap": None}})) == p
    assert e.land(L, ctx(p, ramps={"dunk": {"snap": "beat"}})) == pytest.approx(7 * BEAT)
    assert e.land(L, ctx(5.6)) == 5.6                                  # no room for the slow motion after a beat


def test_short_shot_has_no_speed_up():
    _, v = curve("dunk", 2.95, ctx(1.83))
    assert v.max() == pytest.approx(1.0) and v.min() == pytest.approx(0.5)


def test_rise_never_speeds_up():
    _, v = curve("rise", 5.9, ctx(3.66))
    assert v.max() == pytest.approx(1.0) and v.min() == pytest.approx(0.5)


def test_reveal_and_drift_scale_the_pace():
    t, v = curve("reveal", 3.93, ctx(1.97, 25, 1.0))
    assert v[0] == pytest.approx(2.0) and np.allclose(v[t >= 1.97], 1.0) and np.all(np.diff(v) <= 1e-9)
    t, v = curve("drift", 5.9, ctx(5.9, 25, 4.0))
    assert v[0] == pytest.approx(2.0) and v[-1] == pytest.approx(2.0, abs=0.01)
    assert np.allclose(v[(t > 2 * BEAT) & (t < 5.9 - 2 * BEAT)], 4.0)


def test_reel_json_overrides_and_none():
    _, v = curve("dunk", 5.9, ctx(3.66, ramps={"dunk": {"fast": 1.5, "slow": 0.75, "hold": 1.0}}))
    assert v.max() == pytest.approx(1.5, abs=0.01) and v.min() == pytest.approx(0.75)
    _, v = curve("none", 5.9, ctx(3.66, pace=1.0))
    assert np.allclose(v, 1.0)
    e = fx.EFFECTS["none"]()
    assert e.source_length(5.9, ctx(3.66)) == pytest.approx(5.9 + 2 / 50)


def test_stream_keeps_motion_evenly_spaced():
    """A ramp never plays the nearest frame through a speed change (the 1-2-1-2 step judder of the old fast run
    on a face): the positions it reads advance smoothly."""
    e, c = fx.EFFECTS["dunk"](), ctx(3.66, face=True)
    _, idx, _ = e._index(5.9, c)
    steps = np.diff(idx)
    assert np.abs(np.diff(steps)).max() < 0.15
    n = 80
    src = (np.full((32, 32, 3), k, np.uint8) for k in range(400))
    out = list(e.stream(src, fx.Context(fps=50, beat=BEAT, peak=0.8, params={"src_fps": 50, "face": True}), n))
    assert len(out) == n


def test_a_dunk_renders_with_the_dunk_ramp_unless_reel_json_names_one():
    from types import SimpleNamespace

    from focalcut.reel import render
    s = dict(dunk=True, fx=["hero"], effect="hero")
    assert render.dunk_ramp(s, SimpleNamespace(config={}))["fx"] == ["dunk"]
    assert render.dunk_ramp(s, SimpleNamespace(config={"reference_ramp": "hero"}))["fx"] == ["hero"]
    assert render.dunk_ramp(dict(s, fx=["rise"]), SimpleNamespace(config={}))["fx"] == ["rise"]
    assert render.dunk_ramp(dict(fx=["hero"]), SimpleNamespace(config={}))["fx"] == ["hero"]      # a dance move
    assert set(SP.RAMP_FX) == set(render.RAMP_FX) and set(render.RAMP_FX) <= set(render.SPEED_FX)
