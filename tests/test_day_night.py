"""Daytime footage never plays in a night act, whichever pass places it; a face stays sharp through a speed curve."""
import numpy as np

from focalcut import fx
from focalcut.reel import plan as P


def shot(uid, act, beats, span=None, day=None):
    u = dict(id=uid, clip=uid, role="dance", **({} if day is None else {"day": day}))
    return dict(unit=u, act=act, beats=beats, t=0.0, fx=[], effect=None, **({"span": span} if span else {}))


def test_quick_cuts_into_the_night_never_take_daytime_footage():
    seq = [shot("D", "climax", 2, [10.0, 20.0], day=True),     # a day pick the editor put late
           shot("A", "arrival", 2, [30.0, 40.0], day=True),
           shot("N", "night", 2, [50.0, 60.0], day=False),
           shot("K", "climax", 2), shot("L", "climax", 2), shot("F", "finale", 2)]
    P.quick_cuts(seq, 4, "climax", 0.5, at=4)
    cuts = [s for s in seq if s.get("reuse")]
    assert cuts and all(not c["unit"]["day"] for c in cuts)


def test_quick_cuts_into_the_day_may_take_it():
    seq = [shot("A", "arrival", 2, [30.0, 40.0], day=True), shot("B", "arrival", 2), shot("S", "shows", 2)]
    P.quick_cuts(seq, 2, "arrival", 0.5, at=1)
    assert any(s.get("reuse") and s["unit"]["id"] == "A" for s in seq)


def _speed_frames(face):
    f = [np.full((32, 48, 3), 2 * k, np.uint8) for k in range(120)]      # every frame its own grey
    ctx = fx.Context(fps=50, beat=0.6, peak=0.6, params={"face": face, "cap": 1.5})
    ctx.out_frames = 60
    e = fx.EFFECTS["hero"]()
    fast = e._map(60 / 50, ctx)[2][:60] > 1.3
    return {2 * k for k in range(120)}, [int(o[0, 0, 0]) for o, q in zip(e.apply(f, ctx), fast, strict=True) if q]


def test_a_face_is_never_averaged_over_frames_on_the_fast_stretch():
    greys, out = _speed_frames(face=True)
    assert out and all(v in greys for v in out)                          # each fast output frame a real frame
    greys, blended = _speed_frames(face=False)
    assert any(v not in greys for v in blended)                          # without a face it still blurs the motion


def test_quick_cuts_never_take_the_closing_shots_footage():
    end = dict(shot("H", "finale", 4, [130.9, 133.0], day=True), ending=True)     # the hands-in, the reel's last shot
    seq = [shot("H", "shows", 2, [128.0, 134.0], day=True),                      # the same moment, picked again
           shot("A", "shows", 2, [30.0, 40.0], day=True), shot("B", "arrival", 2), shot("C", "arrival", 2), end]
    P.quick_cuts(seq, 3, "arrival", 0.5, at=3)
    cuts = [s for s in seq if s.get("reuse")]
    assert cuts and all(c["unit"]["id"] != "H" for c in cuts)
    moved = shot("H", "finale", 4, [130.9, 133.0], day=True)                     # taken out of seq to move it
    seq = [shot("H", "shows", 2, [128.0, 134.0], day=True), shot("B", "arrival", 2), shot("C", "arrival", 2)]
    P.quick_cuts(seq, 2, "arrival", 0.5, at=2, avoid=[moved])
    assert not any(s.get("reuse") and s["unit"]["id"] == "H" for s in seq)
    P.quick_cuts(seq, 2, "arrival", 0.5, at=2)                                   # (without it, it would be taken)
    assert any(s.get("reuse") and s["unit"]["id"] == "H" for s in seq)
