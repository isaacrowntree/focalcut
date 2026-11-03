"""Style libraries (fx/looks.py) and the carnival effects: no GPU, no models -- masks are synthesised."""
import numpy as np
import pytest

from focalcut.fx import EFFECTS, TRANSITIONS, Context, carnival, looks
from focalcut.reel import grammar as G


def test_look_names_are_unique_and_tagged():
    n = sum(len(d["looks"]) for d in looks.LIBRARIES.values())
    assert len(looks.LOOKS) == n
    assert looks.LOOKS["clean"]["library"] == "glitch" and looks.LOOKS["carnaval"]["library"] == "carnival"
    assert looks.LOOKS["mainstage"]["library"] == "edm"


def test_looks_name_only_registered_styles():
    for key, lk in looks.LOOKS.items():
        for name in lk["subject"] + lk["accents"] + list(lk.get("fx", {}).values()):
            assert name in EFFECTS, (key, name)
        for name in lk["transitions"] + list(lk.get("reasons", {}).values()):
            assert name in TRANSITIONS and name in G.HALF, (key, name)
        for _act, name, _ in lk.get("extra", []) + lk.get("hits", []):
            assert name in EFFECTS, (key, name)


def test_overlay_is_the_looks_under_reel_json():
    assert looks.overlay({}) is None
    assert looks.overlay({"overlay": {"engine": "milkdrop"}}) == {"engine": "milkdrop"}
    o = looks.overlay({"look": "carnaval"})
    assert o["engine"] == "milkdrop" and o["flicker"] is False and o["milkdrop"] == carnival.MILKDROP_FIESTA
    o = looks.overlay({"look": "carnaval", "overlay": {"milkdrop": ["x"], "flicker": True}})
    assert o["milkdrop"] == ["x"] and o["flicker"] is True and o["milkdrop_calm"] == carnival.MILKDROP_SUN
    assert looks.overlay({"look": "carnaval", "overlay": False}) is None
    assert looks.overlay({"look": "clean"}) is None                       # a glitch look has no overlay of its own
    cfg = {"look": "clean", "plan": {"portrait": {"look": "disco"}}}
    assert looks.overlay(cfg, "landscape") is None and looks.overlay(cfg, "portrait")["milkdrop"] == carnival.MILKDROP_DISCO


def frames(n=40, h=72, w=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 200, (h, w, 3)).astype(np.uint8)
    return [np.roll(base, k, 1) for k in range(n)]


def ctx_with_masks(fr):
    c = Context(fps=25, beat=0.48, peak=0.8, seed=1)
    h, w = fr[0].shape[:2]
    m = np.zeros((h, w), np.float32); m[h // 4:h - 4, w // 3:2 * w // 3] = 1
    c._masks, c._mask_frames = [m] * len(fr), len(fr)
    return c


@pytest.mark.parametrize("name", sorted(k for k in EFFECTS if k.startswith("c_")))
def test_carnival_effects_keep_length_and_shape(name):
    fr = frames()
    out = EFFECTS[name]().apply(fr, ctx_with_masks(fr))
    assert len(out) == len(fr)
    assert all(o.shape == fr[0].shape and o.dtype == np.uint8 for o in out)
    assert any(not np.array_equal(o, f) for o, f in zip(out, fr, strict=True))     # it did something


@pytest.mark.parametrize("name", sorted(k for k in TRANSITIONS if k.startswith("c_")))
def test_carnival_transitions_go_from_a_to_b(name):
    a, b = frames(12, seed=1), frames(12, seed=2)
    out = TRANSITIONS[name]().apply(a, b, Context(fps=25, beat=0.48, seed=3))
    assert len(out) == len(a) + len(b)
    assert all(o.shape == a[0].shape and o.dtype == np.uint8 for o in out)


def test_strobe_is_one_wash_a_beat_at_most():
    fr = [np.full((8, 8, 3), 100, np.uint8) for _ in range(60)]
    c = Context(fps=25, beat=0.48, peak=0.4, seed=0)
    out = EFFECTS["c_strobe"]().apply(fr, c)
    hit = [i for i, o in enumerate(out) if not np.array_equal(o, fr[i])]
    starts = [i for i in hit if i - 1 not in hit]
    assert len(starts) == 3 and min(np.diff(starts)) >= round(c.beat * c.fps)
