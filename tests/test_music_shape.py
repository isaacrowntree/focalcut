"""music.arrange (a path through a song's bars to the film's length and quiet window) and render.shape_envelope (the
music's level and brightness built to the film)."""
import numpy as np

from focalcut.reel import music as MU
from focalcut.reel import render as R


def song(n=40, quiet=(20, 23)):
    db = np.zeros(n)
    db[quiet[0]:quiet[1] + 1] = -9.0                    # a quiet break
    J = {12: [4], 4: [12], 30: [26], 26: [30]}         # bars 4/12 alike, 26/30 alike
    return db, J


def runs(p):
    return [p[k] for k in range(len(p)) if k == 0 or p[k] != p[k - 1] + 1]


def test_arrange_length_ends_and_one_quiet_break():
    db, J = song()
    p = MU.arrange(db, J, 48, quiet=(28, 31), intro=0, outro=4)
    assert p is not None and len(p) == 48 and p[0] == 0 and p[-1] == len(db) - 1
    quiet = [t for t, b in enumerate(p) if db[b] < MU.QUIET_DB]
    assert quiet and min(quiet) >= 28 - 1 and max(quiet) <= 31 + 1      # the break in its window, once
    assert np.bincount(p).max() <= 2                                      # nothing played three times


def test_arrange_unreachable_length_is_none():
    db, J = song()
    assert MU.arrange(db, J, 43, quiet=None) is None                     # repeats come in 8 or 4 bars


def test_shape_envelope_hush_then_build():
    def shot(cat, at, L=4.0, **kw):
        return dict(unit=dict(category=cat), rec_in=at, length=L, **kw)
    plan = dict(beat=0.4, shots=[shot("arrival", 0, 20), shot("speakers", 20, 20)]
                + [shot("baptisms", 40 + 8 * k, 8, dunk=True) for k in range(5)] + [shot("celebration", 80, 8)],
                nat=[dict(at=44.0 + 8 * k) for k in range(5)])
    t = np.array([10.0, 30.0, 41.0, 60.0, 73.0, 85.0])
    db, fc = R.shape_envelope(plan, True, t)
    assert db[1] < db[0] and fc[1] < 5000                                 # the talk down and muffled
    assert db[1] < db[2] < db[3] < db[4] <= db[5] + 0.01                  # building through the event
    assert fc[2] < fc[3] < fc[4] and fc[5] > 19000                       # opening up to full
    rise = R.shape_envelope(plan, True, np.array([44.8, 46.0]))[0]
    assert rise[0] > rise[1] - 0.2                                        # a swell after a rise
