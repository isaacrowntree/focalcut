"""Scrub-first selection: moments come from windows that pass every rule; expression from the subject's faces."""
import numpy as np

from focalcut.reel import expression as EX
from focalcut.reel import windows as WI


def test_a_window_reads_the_worse_face_and_the_best_engaged_one():
    tr = dict(t=np.arange(6) * 0.5, e=np.array([[0.9, -0.6], [0.8, 0.7], [0.9, np.nan], [np.nan, np.nan],
                                                [-0.8, 0.5], [0.6, 0.6]], np.float32),
              size=np.ones((6, 2), np.float32))
    e, best = EX.window(tr, 0.0, 1.0)                         # samples 0..2: worse faces -0.6, 0.7, 0.9
    assert abs(e - (-0.6 + 0.7 + 0.9) / 3) < 1e-6 and abs(best - 0.9) < 1e-6
    assert np.isnan(EX.window(tr, 1.4, 1.6)[0])               # nobody's face seen
    assert np.isnan(EX.window(None, 0, 1)[0])


def ok(t0, s):
    return dict(t0=t0, t1=t0 + 2.4, ok=True, score=s, rank=0.5, expr=0.4, reasons=[])


def bad(t0, why):
    return dict(t0=t0, t1=t0 + 2.4, ok=False, score=1.0, rank=0.9, expr=-0.5, reasons=why)


def test_moments_come_only_from_passing_windows_and_a_unit_without_says_why():
    rec = dict(unit=dict(id="u", role="dance", t0=0.0, t1=10.0))
    wins = {"portrait": [ok(1.0, 0.8), ok(1.5, 0.9), bad(4.0, ["back"]), ok(6.0, 0.5)],
            "landscape": [ok(6.0, 0.7), bad(1.0, ["shake"])]}
    ms, fails = WI.moments_from(rec, None, None, wins)
    assert [m["win"][0] for m in ms] == [1.5, 6.0]            # the overlapping weaker window gives way
    assert ms[0]["fmt_ok"] == {"landscape": False, "portrait": True} and ms[0]["score_l"] == 0
    assert ms[1]["fmt_ok"] == {"landscape": True, "portrait": True}
    assert fails == {"back": 1, "shake": 1}
    none, why = WI.moments_from(rec, None, None, {"portrait": [bad(1.0, ["back", "soft"]), bad(3.0, ["back"])]})
    assert none == [] and why == {"back": 2, "soft": 1}


def test_the_grid_steps_every_length_through_the_unit():
    g = WI.grid(dict(t0=0.0, t1=5.0))
    assert (0.3, 1.5) in g and (0.3, 3.9) in g and all(b <= 4.7 + 1e-6 for _, b in g)
