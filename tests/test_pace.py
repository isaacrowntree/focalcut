"""A base pace under the dancing (plan "pace"): the shot reads more source, its ramps run on top, the cap holds."""
import numpy as np

from focalcut import fx
from focalcut.reel import plan as P


def shot(cat, fx_=(), **kw):
    u = dict(id="A", clip="A", role="dance", category=cat, t0=0.0, t1=60.0)
    return dict(unit=u, fx=list(fx_), length=2.0, t=10.7, peak=0.7, src_in=10.0, beat=0.5, kind="spin", **kw)


def test_segments_play_quicker_and_their_move_stays_on_the_beat():
    a, b = shot("segments"), shot("matinee")
    P.pace_shots([a, b], {"pace": {"segments": 1.15}})
    assert a["speed"] == 1.15 and "speed" not in b
    assert a["src_in"] == round(10.7 - 0.7 * 1.15, 3)                 # the peak's moment lands where it did
    assert P.played(a) == (a["src_in"], 2.0 * 1.15)


def test_an_editors_whole_selection_plays_as_chosen():
    s = shot("segments", ["fit"], span=[10.0, 12.0])
    P.pace_shots([s], {"pace": {"segments": 1.15}})
    assert "speed" not in s


def test_the_pace_sits_under_a_ramp_and_the_cap_still_holds():
    def curve(pace):
        ctx = fx.Context(fps=50, beat=0.5, peak=1.0, params={"cap": 1.5, "pace": pace})
        return fx.EFFECTS["hero"]()._map(2.0, ctx)
    _, src1, s1 = curve(1.0)
    _, src2, s2 = curve(1.15)
    assert src2[-1] > src1[-1]                                       # more of the move read
    assert np.all(s2 <= 1.5 + 1e-9)
    assert np.isclose(s2[-1], 1.15)                                  # it ends at the base pace, not 1x
    assert np.isclose(s2.min(), s1.min() * 1.15)                     # the slow landing is relative to the base


def test_one_clip_can_be_paced_by_its_id():
    a, b, c = shot("matinee"), shot("matinee"), shot("matinee")
    a["unit"].update(id="C1991_matinee", clip="C1991")
    b["unit"].update(id="C19910_matinee", clip="C19910")
    c["unit"].update(id="C1994-02", clip="C1994")
    P.pace_shots([a, b, c], {"pace": {"C1991": 1.2, "C1994": 1.2}})
    assert a["speed"] == 1.2 and "speed" not in b and c["speed"] == 1.2
