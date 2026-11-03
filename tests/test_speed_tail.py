"""Every speed curve returns to 1x and ends on fx.speed.TAIL seconds at 1x; the closing shot carries no speed
effect; velocity.speed_tails flags any shot that does not end at real speed."""
import numpy as np
import pytest

from focalcut import fx
from focalcut.fx import speed as SP
from focalcut.reel import plan, velocity


def ctx(peak, **params):
    return fx.Context(fps=50, beat=0.4567, peak=peak, seed=1, params={"cap": 1.5, **params})


@pytest.mark.parametrize("name", ["hero", "bullet", "montage", "jumpcut", "flashin", "fit"])
@pytest.mark.parametrize("L,peak", [(18.3, 17.4), (18.3, 6.0), (3.0, 1.4), (1.8, 1.5)])
def test_curves_end_at_1x(name, L, peak):
    e = fx.EFFECTS[name]()
    c = ctx(peak, src_len=L * 1.2)
    t = np.arange(round(L * 50) + 1) / 50
    v = e.speed(t, L, c)
    assert np.allclose(v[t >= L - SP.TAIL], 1.0, atol=0.02), name
    if name != "fit":                                      # (fit plays the editor's whole selection, by design)
        assert v.max() <= 1.5 + 1e-6 or name == "jumpcut"


def test_hero_long_shot_is_1x_outside_the_move():
    """The Fiesta ending: 18 s with a landing near the end used to run at 1.4x for 17 s."""
    e, L, p = fx.EFFECTS["hero"](), 18.344, 6.0
    t = np.arange(int(L * 50) + 1) / 50
    v = e.speed(t, L, ctx(p))
    assert np.allclose(v[t < p - 1.5], 1.0) and np.allclose(v[t > p + 1.2], 1.0)
    assert v.max() == pytest.approx(1.4, abs=0.01) and v.min() < 0.35


def test_fit_still_reads_the_whole_selection():
    e = fx.EFFECTS["fit"]()
    _, src, s = e._map(4.0, ctx(1.0, src_len=6.0))
    assert src[-1] == pytest.approx(6.0, abs=0.02) and s[-1] == pytest.approx(1.0, abs=0.01)


def shot(fx_, L=18.344, peak=17.4, **kw):
    return dict(unit={"id": "C1", "role": "show"}, fx=list(fx_), length=L, peak=peak, beat=0.4567, kind="energy", **kw)


def test_speed_tails_flags_a_shot_that_ends_fast(monkeypatch):
    assert velocity.speed_tails({"shots": [shot(["hero"]), shot([]), shot(["c_glow"])]}) == []
    class Old(SP.Speed):                                   # the old curve: fast again after the hold, to the end
        name = "old_hero"; fast, slow, hold, ease = 1.4, 0.3, 1.0, 0.4
        def speed(self, t, out_len, ctx):
            return np.full_like(t, 1.4)
    monkeypatch.setitem(fx.EFFECTS, "old_hero", Old)
    bad = velocity.speed_tails({"shots": [shot(["old_hero"])]})
    assert len(bad) == 1 and "1.40" in bad[0]
    assert velocity.speed_tails({"shots": [shot(["flashout"], velocity=True)]}) == []   # a velocity edit's out-half


def test_closing_and_ending_shots_play_at_real_speed():
    seq = [shot(["hero"]), shot(["bullet", "c_glow"], ending=True), shot(["flashin"]), shot(["hero"])]
    plan.real_speed_close(seq)
    assert seq[0]["fx"] == ["hero"]
    assert seq[1]["fx"] == ["c_glow"] and seq[1]["effect"] == "c_glow"
    assert seq[2]["fx"] == ["flashin"]
    assert seq[3]["fx"] == [] and seq[3]["effect"] is None
