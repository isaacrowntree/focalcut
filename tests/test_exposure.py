"""reel/exposure.py: the highlight pull's curve and when a shot gets one -- no footage, no ffmpeg."""
import math

import numpy as np
import pytest

from focalcut.reel import exposure as X


def apply(pull, x):
    """The lutyuv expression evaluated on legal-range luma x (0..1, superwhite above)."""
    e = X.curve(pull).replace("\\,", ",")
    e = e.replace("if(", "IF(").replace("lt(", "LT(").replace("pow(", "POW(").replace("sin(", "SIN(")
    val = X.BLACK + x * (X.WHITE - X.BLACK)
    y = eval(e, {"IF": lambda c, a, b: a if c else b, "LT": lambda a, b: a < b, "POW": pow, "SIN": math.sin,
                 "PI": math.pi, "val": val})
    return (y - X.BLACK) / (X.WHITE - X.BLACK)


@pytest.mark.parametrize("pull", [0.0, 0.05, X.MAX, 0.18])
def test_curve_keeps_shadows_lands_superwhite_on_white_and_never_reverses(pull):
    xs = np.linspace(0, X.TOP, 400)
    ys = np.array([apply(pull, x) for x in xs])
    assert all(apply(pull, x) == pytest.approx(x) for x in (0.0, 0.2, X.KNEE - 0.01))   # shadows, mids as shot
    assert ys[-1] == pytest.approx(1.0, abs=1e-3)                                        # white stays white
    assert (np.diff(ys) > 0).all()                                                       # no tone reversal
    assert (ys <= xs + 1e-9).all()


def test_pull_brings_skin_down_more_the_bigger_it_is():
    assert apply(0.1, 0.85) < apply(0.05, 0.85) < apply(0.0, 0.85) < 0.85


def test_amount_only_for_lit_blown_daytime_subjects():
    blown = dict(frame=0.6, face=0.84, face_clip=0.3, subject=0.2, top=1.09)
    assert 0 < X.amount(blown) <= X.MAX
    assert X.amount(dict(blown, frame=0.2)) == 0                    # night: never touched
    assert X.amount(dict(blown, face=0.3)) == 0                     # faces in shade / backlit: the sky's not ours
    assert X.amount(dict(blown, face=None, face_clip=None)) == 0    # no face to judge by
    assert X.amount(dict(frame=0.6, face=0.6, face_clip=0.02, subject=0.03, top=1.0)) == 0   # holds its detail
    assert X.amount(dict(blown, face_clip=0.6)) >= X.amount(blown)


def test_by_hand_pulls_match_a_unit_or_a_stretch():
    cfg = {"pull": {"C2148": 0.1, "C2150@0:10-0:20": 0.05}}
    def u(i):
        return {"id": i, "clip": i}
    assert X.by_hand(cfg, u("C2148"), {"src_in": 3.0, "length": 2.0}) == 0.1
    assert X.by_hand(cfg, u("C2150"), {"src_in": 18.0, "length": 4.0}) == 0.05
    assert X.by_hand(cfg, u("C2150"), {"src_in": 21.0, "length": 2.0}) is None
    assert X.filt(0) == "" and X.filt(0.05).startswith("format=yuv422p10le,lutyuv=y=")
