"""Portrait framing of an editor's pick: the strip goes to the faces first (plan.face_first)."""
from __future__ import annotations

import numpy as np

from focalcut.reel import plan as P

from .test_picks_mode import many_picks, picks_reel

CW, CH = 0.3164, 1.0                                             # a 9:16 strip out of a 16:9 frame


def scan(faces):
    """A face scan: [(t, cx, cy, h)] -> the persons step's faces_dense layout."""
    t, cx, cy, h = (np.array(v, np.float32) for v in zip(*faces, strict=True))
    return dict(t=t, cx=cx, cy=cy, h=h, score=np.full(len(t), 0.9, np.float32))


def strip(x, length):
    grid = np.arange(0, length + 0.01, 0.1)
    return grid, np.full(len(grid), x), np.full(len(grid), 0.5)


def test_a_picks_strip_leaves_a_back_for_the_face_beside_it():
    # the body tracker sits on someone at the frame's left edge (no face: a back); two friends face the camera
    yf = scan([(t, 0.70, 0.3, 0.2) for t in (10.0, 10.5, 11.0, 11.5)] + [(11.0, 0.42, 0.3, 0.2), (11.5, 0.45, 0.3, 0.2)])
    grid, x, y = strip(0.16, 1.5)
    ft = P.face_first(yf, 10.0, 1.5, grid, x, y, CW, CH, body=lambda t: (0.9, 0.05, 0.1, 0.9))
    assert ft is not None and ft["held"] == 1.0
    assert np.all(np.abs(ft["x"] - 0.70) < 0.03)                # on the face seen all through, not hopping to the other
    # the strip already on that face: left as it is
    grid, x, y = strip(0.70, 1.5)
    assert P.face_first(yf, 10.0, 1.5, grid, x, y, CW, CH) is None
    # no face at all: the body tracker's strip stays
    assert P.face_first(scan([(3.0, 0.5, 0.3, 0.2)]), 10.0, 1.5, grid, x, y, CW, CH) is None


def test_two_friends_who_face_the_camera_in_turn_get_one_pan_not_a_ping_pong():
    a = [(t, 0.68, 0.3, 0.2) for t in (10.0, 10.5, 12.5, 13.0, 13.5)]            # turns away 11.0-12.0
    b = [(t, 0.27, 0.25, 0.22) for t in (10.0, 10.5, 11.0, 11.5, 12.0)]           # faces the camera till then
    grid, x, y = strip(0.68, 3.5)
    ft = P.face_first(scan(a + b), 10.0, 3.5, grid, x, y, CW, CH, body=lambda t: (0.9, 0.05, 0.1, 0.9))
    assert ft is not None
    xs = ft["x"]
    assert xs[0] < 0.4 and xs[15] < 0.4 and xs[-1] > 0.6        # on b while b faces the camera, on a at the end
    steps = np.diff(xs)
    turns = np.sum(np.diff(np.sign(steps[np.abs(steps) > 1e-4])) != 0)
    assert turns == 0                                           # one move, one way
    assert ft["held"] > 0.7                                     # (a face in the strip but for the pan)


def test_a_face_far_behind_never_takes_the_strip_and_the_face_by_the_strip_wins_a_tie():
    grid, x, y = strip(0.30, 1.0)
    # the subject (0.30) shows a face once, held already; the other face is a small one far behind
    small = scan([(t, 0.8, 0.4, 0.05) for t in (10.0, 10.5, 11.0)] + [(10.5, 0.30, 0.3, 0.2)])
    assert P.face_first(small, 10.0, 1.0, grid, x, y, CW, CH) is None
    # two faces of a size: the one by the body tracker's strip keeps the shot (its subject, their face cut off)
    two = scan([(t, 0.35, 0.3, 0.19) for t in (10.0, 10.5, 11.0)] + [(t, 0.75, 0.3, 0.21) for t in (10.0, 10.5, 11.0)])
    grid, x, y = strip(0.20, 1.0)                               # (the strip sits off their face: on the torso)
    ft = P.face_first(two, 10.0, 1.0, grid, x, y, CW, CH, body=lambda t: (0.9, 0.05, 0.1, 0.9))
    assert ft is not None and np.all(np.abs(ft["x"] - 0.35) < 0.03)
    grid, x, y = strip(0.55, 1.0)                               # between the two: the bigger face
    ft = P.face_first(two, 10.0, 1.0, grid, x, y, CW, CH)
    assert ft is not None and np.all(np.abs(ft["x"] - 0.75) < 0.03)
    # a dancer whose face is hidden (hair, a mask) keeps the shot from a far-off face: too small for their build
    far = scan([(t, 0.8, 0.3, 0.14) for t in (10.0, 10.5, 11.0)])
    grid, x, y = strip(0.30, 1.0)
    assert P.face_first(far, 10.0, 1.0, grid, x, y, CW, CH, body=lambda t: (0.95, 0.03, 0.1, 0.5)) is None
    assert P.face_first(far, 10.0, 1.0, grid, x, y, CW, CH, body=lambda t: (0.5, 0.2, 0.15, 0.45)) is not None   # (a head's size there)
    # someone filmed from further off, their small face in the strip: a bigger face elsewhere doesn't take the shot
    own = scan([(t, 0.30, 0.25, 0.06) for t in (10.0, 10.5, 11.0)] + [(t, 0.10, 0.3, 0.14) for t in (10.0, 10.5, 11.0)])
    assert P.face_first(own, 10.0, 1.0, grid, x, y, CW, CH, body=lambda t: (0.5, 0.2, 0.15, 0.45)) is None
    # the strip holds its subject's face: a smaller, steadier face beside them never takes it, one that dwarfs
    # theirs does (the person right in front of them is who the pick is of)
    grid, x, y = strip(0.45, 1.0)
    mine = [(10.0, 0.50, 0.3, 0.2), (10.5, 0.42, 0.3, 0.2), (11.0, 0.48, 0.3, 0.2)]
    beside = scan(mine + [(t, 0.25, 0.3, 0.15) for t in (10.0, 10.5, 11.0)])
    ft = P.face_first(beside, 10.0, 1.0, grid, x, y, CW, CH)
    assert ft is None or np.all(np.abs(ft["x"] - 0.46) < 0.06)
    front = scan(mine + [(t, 0.25, 0.3, 0.4) for t in (10.0, 10.5, 11.0)])
    ft = P.face_first(front, 10.0, 1.0, grid, x, y, CW, CH)
    assert ft is not None and np.all(np.abs(ft["x"] - 0.25) < 0.03)
    tiny = scan([(t, 0.8, 0.3, 0.1) for t in (10.0, 10.5, 11.0)])                               # never a distant face
    assert P.face_first(tiny, 10.0, 1.0, grid, x, y, CW, CH, body=lambda t: (0.5, 0.2, 0.15, 0.45)) is None


def test_a_face_bigger_than_the_strip_is_centred_and_a_friend_who_fits_shares_it():
    big = scan([(10.0, 0.47, 0.45, 0.4), (10.5, 0.44, 0.48, 0.4)])
    grid, x, y = strip(0.67, 0.5)                               # the body tracker: on a shoulder
    ft = P.face_first(big, 10.0, 0.5, grid, x, y, CW, CH)
    assert ft is not None and abs(float(ft["x"][0]) - 0.47) < 0.03 and ft["held"] == 1.0
    pair = scan([(10.0, 0.47, 0.3, 0.15), (10.0, 0.57, 0.3, 0.14)])
    grid, x, y = strip(0.20, 0.5)
    ft = P.face_first(pair, 10.0, 0.5, grid, x, y, CW, CH)
    assert ft is not None and 0.48 < float(ft["x"][0]) < 0.56   # between the two, nearer the bigger face


def test_only_the_editors_picks_are_framed_faces_first():
    u = dict(id="C1", export_t0=10.0)
    picks = {"C1": {"add": ["0:05.00-0:07.00", "0:20"]}}
    assert P.in_pick(picks, u, 15.0, 2.0) and P.in_pick(picks, u, 15.5, 1.0)
    assert P.in_pick(picks, u, 14.5, 2.0)                       # most of it inside the pick
    assert not P.in_pick(picks, u, 17.5, 1.0) and not P.in_pick(picks, u, 30.0, 1.0)
    assert not P.in_pick({}, u, 15.0, 2.0)


def test_an_ending_marked_follow_keeps_the_derived_crop(tmp_path, monkeypatch):
    seen = []

    def fake(project, unit, t0, length, fmt, focus=True, pick=None):
        seen.append((unit["id"], pick))
        return dict(w=0.3, h=1.0, z=1.0, path=[[0, 0.62, 0.5]])
    monkeypatch.setattr(P, "reframe", fake)
    picks = many_picks(n_clips=2, per=2)
    for at, want in (("follow", 0.62), (0.4, 0.4), (None, 0.5)):
        pr, music, moments = picks_reel(tmp_path, picks, plan={"ending": ["U11", 50.0, 53.0] + ([at] if at else [])})
        seen.clear()
        last = P.build(pr, "portrait", music, moments)["shots"][-1]
        assert last["ending"] and last["crop"]["path"][0][1] == want
        assert ("U11", True if at == "follow" else None) in seen
