"""Effect modules: the registry, frame-count contracts, and the speed-curve source maths (no GPU, no SAM 3)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from focalcut import fx
from focalcut.fx.speed import Speed


def frames(n=40, h=72, w=128):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return [np.roll(base, i, axis=1) for i in range(n)]


def test_registry_has_the_documented_modules():
    assert {"montage", "hero", "bullet", "jumpcut", "flashin", "flashout", "spotlight", "freeze", "neon",
            "particles", "velocity", "echo", "strobe", "orbs", "aura", "punchin", "shake",
            "glitch"} <= set(fx.EFFECTS)
    assert {"flash", "fade", "zoom", "swipe", "glitch"} <= set(fx.TRANSITIONS)


@pytest.mark.parametrize("name", ["punchin", "shake", "glitch"])
def test_frame_effects_keep_length_and_shape(name):
    f = frames()
    out = fx.EFFECTS[name]().apply(f, fx.Context(fps=50, beat=0.6, peak=0.4))
    assert len(out) == len(f)
    assert out[0].shape == f[0].shape and out[0].dtype == np.uint8


@pytest.mark.parametrize("name", sorted(fx.TRANSITIONS))
def test_transitions_return_exactly_the_frames_they_replace(name):
    ctx = fx.Context(fps=50, beat=0.6)
    tr = fx.TRANSITIONS[name]()
    n = tr.frames(ctx) // 2 * 2
    a, b = frames(n // 2), frames(n // 2)
    out = tr.apply(a, b, ctx)
    assert len(out) == n
    assert all(o.shape == a[0].shape for o in out)


def test_hero_is_a_velocity_edit_fast_then_held_slow_on_the_landing_and_keeps_length():
    ctx = fx.Context(fps=50, beat=0.6, peak=1.2)
    hero = fx.EFFECTS["hero"]()
    assert isinstance(hero, Speed)
    t = np.linspace(0, 2.4, 241)
    v = hero.speed(t, 2.4, ctx)
    assert v.max() <= 1.5 + 1e-9                  # dancing never past 1.5x
    assert v[0] > 1.2                             # fast through the move
    hold = (t >= 1.2) & (t <= 1.2 + 0.6)          # arrives slow on the landing and holds a beat
    assert np.allclose(v[hold], hero.slow)
    assert v[-1] > 1.2                            # eases back up to speed
    assert hero.source_peak(1.2, ctx) > 0
    src = hero.source_length(2.4, ctx)
    ctx.out_frames = 120
    out = hero.apply(frames(int(src * 50) + 2), ctx)
    assert len(out) == 120


def test_speed_cap_and_closeup_hold_deeper():
    ctx = fx.Context(fps=50, beat=0.6, peak=1.2, params={"cap": 1.2})
    t = np.linspace(0, 2.4, 241)
    assert fx.EFFECTS["bullet"]().speed(t, 2.4, ctx).max() <= 1.2 + 1e-9
    plain = fx.EFFECTS["hero"]().speed(t, 2.4, fx.Context(fps=50, beat=0.6, peak=1.2))
    close = fx.EFFECTS["hero"]().speed(t, 2.4, fx.Context(fps=50, beat=0.6, peak=1.2, params={"closeup": True}))
    assert close.min() < plain.min() and (close < 0.5).sum() > (plain < 0.5).sum()


def test_montage_reads_more_source_than_it_outputs():
    ctx = fx.Context(fps=50, beat=0.6, peak=1.2)
    assert fx.EFFECTS["montage"]().source_length(2.4, ctx) > 2.4


def test_chain_applies_in_order():
    f = frames()
    ctx = fx.Context(fps=50, beat=0.6, peak=0.4)
    out = fx.chain(f, ["punchin", "glitch"], ctx)
    assert len(out) == len(f)


@pytest.mark.parametrize("name", ["dipwhite", "spin", "lightleak", "swipe", "fadein"])
def test_new_transitions_return_every_frame(name):
    a, b = frames(10), frames(10)
    out = fx.TRANSITIONS[name]().apply(a, b, fx.Context(fps=50, beat=0.6))
    assert len(out) == 20 and out[0].shape == a[0].shape and out[0].dtype == np.uint8


def test_swipe_follows_the_motion():
    from focalcut.fx.transitions import motion_dir
    f = frames(6)                                     # content rolls right by a pixel a frame (at full size)
    dx, _ = motion_dir([np.repeat(np.repeat(x, 4, 0), 4, 1) for x in f])
    assert dx > 0.3
