"""The night lift's window: the curve lifts a dark face toward the target within its stops, leaves black, the deep
shadows and white alone, and the ffmpeg expression is the same curve."""
import numpy as np

from focalcut.reel import window as W


def _eval(expr, val):
    e = expr.replace("\\,", ",").replace("between(val,64,940)", "(64<=val<=940)")
    e = e.replace("if(", "_if(").replace("clip(", "_clip(")
    return eval(e, {"_if": lambda c, a, b: a if c else b, "_clip": lambda x, lo, hi: min(max(x, lo), hi), "val": val})


def test_lift_reaches_toward_target_within_stops():
    for level in (0.08, 0.12, 0.2, 0.3):
        a, k = W.amount(level, 0.4, 0.6)
        got = W.lifted(level, a)
        assert got <= min(0.4, level * 2 ** 0.6) + 1e-3 and got > level * 1.15
        assert 1.0 < k < 1.5


def test_lift_leaves_bright_faces_silhouettes_and_black():
    assert W.amount(0.45) == (0.0, 1.0)
    assert W.amount(0.02) == (0.0, 1.0)
    assert W.amount(None) == (0.0, 1.0)
    assert W.lifted(0.0, 1.0) == 0.0 and W.lifted(0.015, 1.0) == 0.015 and abs(W.lifted(1.0, 1.0) - 1.0) < 1e-9


def test_ffmpeg_curve_is_the_same_curve():
    a, k = W.amount(0.12)
    y, c = W.curve(a, k)
    for lv in np.linspace(0, 1, 41):
        val = 64 + 876 * lv
        assert abs((_eval(y, val) - 64) / 876 - W.lifted(lv, a)) < 1e-5
    assert _eval(y, 30) == 30 and _eval(y, 1000) == 1000          # out of legal range: as shot
    assert abs(_eval(c, 612) - (512 + 100 * k)) < 1e-6 and _eval(c, 512) == 512


def test_window_is_soft_and_follows_the_face():
    m = W._window([(80, 40, 20, 24)], 160, 90)
    assert m[int(40 + W.DROP * 24), 80] == 1.0                  # full strength on the face's window
    assert m[5, 5] == 0.0 and m[85, 155] == 0.0                 # nothing far from it
    row = m[int(40 + W.DROP * 24), 80:]
    assert np.all(np.diff(row) <= 1e-6) and (row > 0).sum() > 15   # a long feather, falling off smoothly


def test_stage_coloured_skin_keeps_its_colour():
    assert W.amount(0.12, sat=0.05)[1] > 1.2
    assert W.amount(0.12, sat=0.25)[1] == 1.0
    assert W.amount(0.12, sat=0.25)[0] == W.amount(0.12, sat=0.05)[0]   # the face is lifted all the same


def test_each_thread_has_its_own_detector(monkeypatch):
    import sys
    import threading
    import types
    made = []
    cv2 = types.SimpleNamespace(FaceDetectorYN=types.SimpleNamespace(create=lambda *a: made.append(1) or object()))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr("focalcut.reel.people.models", lambda: {"det": "x"})
    W._local.__dict__.clear()
    got = []
    ts = [threading.Thread(target=lambda: got.append(W._detector())) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(made) == 3 and len({id(d) for d in got}) == 3
