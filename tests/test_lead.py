"""Who a portrait strip is on and how it follows them (reel/lead.py), on made-up pose data: the cases the editor
corrected by hand on Electric Nation 2026 (the dancer over a bystander's clear face, no hand-over mid-shot, a mask
followed with no face, hands on a mixer, a floor move, two together) as rules, not clips."""
import numpy as np
import pytest

from focalcut.reel import lead as LD
from focalcut.reel import plan as PL

AW, AH = 640.0, 360.0
CW = 0.3164                                        # a 9:16 strip out of a 16:9 frame


def person(cx, top=0.1, h=0.8, wave=0.0, lying=False, head=True, w=None):
    """One pose person: (box, keypoints) in analysis pixels. wave: how far the wrists are swung out (a dancer's
    arms, frame to frame); lying: stretched out along the floor; head: the head's keypoints are seen."""
    w = w or 0.3 * h * AH / AW
    kp = np.zeros((17, 3))
    if lying:
        box = [cx - 0.3, 0.6, cx + 0.3, 0.8]
        pts = {0: (cx - 0.27, 0.68), 5: (cx - 0.18, 0.66), 6: (cx - 0.18, 0.74), 11: (cx + 0.1, 0.66), 12: (cx + 0.1, 0.74),
               7: (cx - 0.2, 0.78), 8: (cx - 0.2, 0.78), 9: (cx - 0.25, 0.79), 10: (cx - 0.25, 0.79)}
    else:
        box = [cx - w / 2, top, cx + w / 2, top + h]
        sh = top + 0.2 * h
        pts = {5: (cx - w / 3, sh), 6: (cx + w / 3, sh), 11: (cx - w / 4, top + 0.55 * h), 12: (cx + w / 4, top + 0.55 * h),
               7: (cx - w / 2, sh + 0.12 * h), 8: (cx + w / 2, sh + 0.12 * h),
               9: (cx - w / 2 - wave, sh + 0.2 * h - wave), 10: (cx + w / 2 + wave, sh + 0.2 * h - wave)}
        if head:
            pts.update({0: (cx, top + 0.08 * h), 1: (cx - 0.01, top + 0.07 * h), 2: (cx + 0.01, top + 0.07 * h),
                        3: (cx - 0.02, top + 0.08 * h), 4: (cx + 0.02, top + 0.08 * h)})
    for k, (x, y) in pts.items():
        kp[k] = [x * AW, y * AH, 0.9]
    return [box[0] * AW, box[1] * AH, box[2] * AW, box[3] * AH, 0.9], kp


def scene(frames, dt=0.2):
    """Pose scores (frame.load's dict) from frames: each a list of person() results, 5 a second."""
    P = max(len(f) for f in frames)
    boxes, kp = np.full((len(frames), P, 5), np.nan), np.zeros((len(frames), P, 17, 3))
    for i, f in enumerate(frames):
        for k, (b, q) in enumerate(f):
            boxes[i, k], kp[i, k] = b, q
    return dict(pose_t=np.arange(len(frames)) * dt, boxes=boxes, kp=kp, aw=AW, ah=AH,
                focus=np.where(np.isnan(boxes[..., 0]), np.nan, 10.0))


def faces(rows):
    """A face scan from (t, cx, cy, h) rows."""
    t, cx, cy, h = (np.array(v, float) for v in zip(*rows, strict=True))
    return dict(t=t, cx=cx, cy=cy, h=h, score=np.full(len(t), 0.9))


def test_people_are_linked_through_a_shot_and_a_weak_or_double_box_is_no_one():
    frames = []
    for i in range(10):
        f = [person(0.3 + 0.01 * i), person(0.7)]
        dup = person(0.7)
        dup[0][4] = 0.7                                          # the pose pass found the second person twice
        junk = person(0.5, h=0.2)
        junk[0][4] = 0.3                                         # a bag: not confident
        frames.append([*f, dup, junk])
    trs = LD.tracks(scene(frames), None, 0.0, 2.0)
    assert len(trs) == 2 and all(len(tr["t"]) == 10 for tr in trs)
    mover = min(trs, key=lambda tr: tr["head"][0, 0])
    assert mover["head"][-1, 0] == pytest.approx(0.39, abs=0.01)


def test_the_one_dancing_takes_the_shot_from_a_still_face_beside_them():
    """A bystander stands facing the camera; the dancer, bigger and head down (no face), throws their arms about:
    the shot is the dancer's, even when the body tracker's strip opened on the bystander."""
    frames = [[person(0.2, top=0.25, h=0.6), person(0.62, top=0.05, h=0.95, wave=0.06 * (i % 2), head=False)]
              for i in range(10)]
    d = scene(frames)
    yf = faces([(t, 0.2, 0.3, 0.09) for t in np.arange(0, 2.0, 0.5)])
    out = LD.strip(d, yf, None, 0.0, 1.8, CW, 1.0, body=(np.array([0.0, 1.8]), np.array([0.2, 0.2])))
    assert out["on"] == "person" and abs(out["x"].mean() - 0.62) < 0.05
    assert "busiest" in out["why"] or "dancing" in out["why"]


def test_a_shot_keeps_the_person_it_opens_on_when_nobody_is_clearly_more_its_subject():
    """Two friends of a size, both dancing; one turns away half way (no face) while the other faces the camera: the
    strip stays where the shot opened, no hand-over."""
    frames = [[person(0.17, wave=0.03 * (i % 2), head=i < 8), person(0.6, wave=0.03 * (i % 2))] for i in range(16)]
    d = scene(frames)
    yf = faces([(t, 0.17, 0.16, 0.12) for t in np.arange(0, 1.6, 0.5)] + [(t, 0.6, 0.16, 0.12) for t in np.arange(0, 3.2, 0.5)])
    out = LD.strip(d, yf, None, 0.0, 3.0, CW, 1.0, body=(np.array([0.0, 3.0]), np.array([0.17, 0.17])))
    assert np.abs(out["x"] - 0.17).max() < 0.06               # (0.158: the strip's left stop)


def test_the_strip_stays_on_a_masked_head_while_it_moves():
    """A mask shows no face to the scan; the head moves across a third of the frame in a second and the strip is on
    it the whole way, with no lag to speak of."""
    xs = np.r_[np.linspace(0.6, 0.3, 6), np.linspace(0.3, 0.5, 5)]
    d = scene([[person(x)] for x in xs])
    out = LD.strip(d, None, None, 0.0, 2.0, CW, 1.0)
    want = np.interp(out["t"], np.arange(len(xs)) * 0.2, xs)
    assert np.abs(out["x"] - want).max() < 0.05 and np.abs(out["x"] - want).mean() < 0.02


def test_hands_at_work_with_no_head_in_frame_are_framed_where_the_fingers_are():
    """A DJ's arms come down from the top of the frame onto the mixer: no head, two wrists -- the strip goes to the
    fingertips (past the wrists, along the forearms), not to the middle of the body."""
    frames = []
    for _ in range(6):
        box = [0.3 * AW, 0.0, 1.0 * AW, 0.7 * AH, 0.9]
        kp = np.zeros((17, 3))
        for k, (x, y) in {7: (0.52, 0.05), 8: (0.75, 0.05), 9: (0.47, 0.3), 10: (0.62, 0.3)}.items():
            kp[k] = [x * AW, y * AH, 0.9]
        frames.append([(box, kp)])
    out = LD.strip(scene(frames), None, None, 0.0, 1.0, CW, 1.0)
    assert out["on"] == "hands" and out["cut"] == 0.0
    assert abs(out["x"].mean() - 0.49) < 0.04 and out["y"].mean() > 0.4     # (the body's middle is at 0.65)


def test_someone_on_the_floor_is_framed_on_their_body_not_their_head():
    d = scene([[person(0.5, lying=True)] for _ in range(6)])
    out = LD.strip(d, None, None, 0.0, 1.0, CW, 1.0)
    head = 0.5 - 0.27
    assert out["on"] == "floor" and out["x"].mean() > head + 0.1          # more of the torso ...
    assert out["x"].mean() - CW / 2 < head                                # ... with the head still in the strip


def test_two_together_share_the_strip_and_two_apart_do_not():
    close = scene([[person(0.42, wave=0.02 * (i % 2)), person(0.58, wave=0.02 * (i % 2))] for i in range(6)])
    out = LD.strip(close, None, None, 0.0, 1.0, CW, 1.0)
    assert out["on"] == "pair" and abs(out["x"].mean() - 0.5) < 0.03
    apart = scene([[person(0.25, wave=0.02 * (i % 2)), person(0.75, wave=0.02 * (i % 2))] for i in range(6)])
    out = LD.strip(apart, None, None, 0.0, 1.0, CW, 1.0, body=(np.array([0.0, 1.0]), np.array([0.25, 0.25])))
    assert out["on"] == "person" and abs(out["x"].mean() - 0.25) < 0.04   # one whole, the other left out


def test_a_follow_click_takes_a_person_with_no_face_and_keeps_them_all_shot():
    """The editor clicks a costume (no face anywhere): the strip follows that body through the shot, whoever else is
    bigger or busier; a click on nobody holds the point."""
    frames = [[person(0.3 + 0.03 * i, head=False), person(0.75, top=0.02, h=0.98, wave=0.05 * (i % 2))] for i in range(8)]
    d = scene(frames)
    crop = dict(w=CW, h=1.0, z=1.0, path=[[0.0, 0.75, 0.5]], face=0.0, present=0.0, inside=0.0, head=0.0)
    c = PL.follow_override(crop, None, [(0.2, 0.31, 0.3)], 0.0, 1.4, d)
    assert c["follow"] == "person" and c["path"][0][1] < 0.36 and c["path"][-1][1] > 0.46
    c = PL.follow_override(crop, None, [(0.2, 0.05, 0.9)], 0.0, 1.4, scene([[person(0.75)] for _ in range(8)]))
    assert c["follow"] == "point" and c["path"][0][1] == pytest.approx(CW / 2, abs=1e-3)


def test_a_trimmed_picks_part_is_judged_by_cut_heads_halved_people_and_pairs():
    whole = dict(on="person", cut=0.0, halved=0.0)
    assert PL.part_cost(dict(on="person", cut=0.5, halved=0.0)) > PL.part_cost(whole) + PL.PART_GAIN
    assert PL.part_cost(dict(on="person", cut=0.0, halved=0.6)) > PL.part_cost(whole) + PL.PART_GAIN
    assert PL.part_cost(dict(on="pair", cut=0.0, halved=0.0)) < PL.part_cost(whole)
    assert PL.part_cost(dict(w=0.3, path=[])) is None          # nobody followed: nothing to judge by
