"""A DJ shot shows the DJ and their decks (Isaac, Electric Nation 2026: "for all the dj shots I want you to show more
of their turntables than just their bodies"): in portrait the strip sits on the DJ's hands and the gear past them,
their head kept in it, never zoomed in; the planner's moments favour hands down on the decks over arms in the air or
the DJ alone with the gear out of frame. The editor's own framing (focal points, a follow click) stands."""
from __future__ import annotations

import json
import os

import numpy as np

from focalcut.reel import lead as LD
from focalcut.reel import plan as PL
from focalcut.reel import validate as VA
from focalcut.reel.project import Project

AW, AH = 640.0, 360.0
CW = 0.3164                                        # a 9:16 strip out of a 16:9 frame


def dj(cx, reach=0.0, up=False, top=0.15, h=0.85, w=0.14):
    """A DJ behind the decks: (box, keypoints) in analysis pixels. reach: how far (frame widths) their hands are
    out to the side on the gear (+ right); up: both arms thrown in the air."""
    kp = np.zeros((17, 3))
    sh = top + 0.2 * h
    hand_y = sh + 0.3 * h
    pts = {0: (cx, top + 0.08 * h), 1: (cx - 0.01, top + 0.07 * h), 2: (cx + 0.01, top + 0.07 * h),
           3: (cx - 0.02, top + 0.08 * h), 4: (cx + 0.02, top + 0.08 * h),
           5: (cx - w / 3, sh), 6: (cx + w / 3, sh), 11: (cx - w / 4, top + 0.6 * h), 12: (cx + w / 4, top + 0.6 * h)}
    if up:
        pts.update({7: (cx - w / 2, sh - 0.1 * h), 8: (cx + w / 2, sh - 0.1 * h),
                    9: (cx - w / 2, top - 0.05), 10: (cx + w / 2, top - 0.05)})
    elif reach:
        pts.update({7: (cx + 0.4 * reach, sh + 0.2 * h), 8: (cx + 0.5 * reach, sh + 0.2 * h),
                    9: (cx + 0.8 * reach, hand_y), 10: (cx + reach, hand_y)})
    else:                                          # facing the camera over the decks, a hand either side
        pts.update({7: (cx - w / 2, sh + 0.15 * h), 8: (cx + w / 2, sh + 0.15 * h),
                    9: (cx - w / 2 - 0.02, hand_y), 10: (cx + w / 2 + 0.02, hand_y)})
    for k, (x, y) in pts.items():
        kp[k] = [x * AW, min(max(y, 0.0), 1.0) * AH, 0.9]
    box = [min(cx - w / 2, cx + reach) * AW, top * AH, max(cx + w / 2, cx + reach) * AW, min(top + h, 1.0) * AH, 0.9]
    return box, kp


def scene(frames, dt=0.2):
    P = max(len(f) for f in frames)
    boxes, kp = np.full((len(frames), P, 5), np.nan), np.zeros((len(frames), P, 17, 3))
    for i, f in enumerate(frames):
        for k, (b, q) in enumerate(f):
            boxes[i, k], kp[i, k] = b, q
    return dict(pose_t=np.arange(len(frames)) * dt, boxes=boxes, kp=kp, aw=AW, ah=AH,
                focus=np.where(np.isnan(boxes[..., 0]), np.nan, 10.0))


def test_a_dj_unit_is_known_by_a_whole_word_of_its_tag_id_or_category():
    assert PL.is_dj(dict(id="C2055_dj-klp-day", tag="dj-klp-day", category="broll"))
    assert PL.is_dj(dict(id="C2033_dj", tag="", category="broll"))
    assert PL.is_dj(dict(id="C9", tag="night", category="DJ"))
    assert PL.is_dj(dict(id="C1_dj-gio"))
    assert not PL.is_dj(dict(id="C5_djembe", tag="drum-circle"))
    assert not PL.is_dj(dict(id="C6_crowd-night", tag="crowd-night", category="broll"))


def test_decks_beside_the_dj_are_the_focal_point_with_the_near_side_of_their_face_kept():
    """The decks out to the DJ's side: the strip goes to the hands and the gear past them (Isaac: "part of the DJ's
    head cut off is fine"), as far as keeps their eyes and nose inside its edge -- never the whole head gone."""
    d = scene([[dj(0.35, reach=0.2)] for _ in range(8)])
    plain = LD.strip(d, None, None, 0.0, 1.4, CW, 1.0)
    out = LD.strip(d, None, None, 0.0, 1.4, CW, 1.0, dj=True)
    assert out["on"] == "decks" and plain["on"] != "decks"
    hw = 0.5 * float(np.median(out["lead"]["hs"])) * LD.ASPECT * 0.8
    left_edge = out["x"] - CW / 2
    assert np.all(0.35 - left_edge >= LD.FACE_IN * hw)                        # the face's near side in
    assert np.all(0.35 - left_edge < hw + 0.01)                               # ... and no more of the head than that
    tips = 0.35 + 0.2 + LD.FINGERS * 0.04                                     # the right hand's fingers on the gear
    assert np.all(out["x"] + CW / 2 > tips + 0.06)                            # the hands and plenty of gear past them
    assert np.all(np.abs(out["x"] - out["x"][0]) < 1e-9)                      # one offset, steady
    # the other way round: decks on the DJ's left
    left = LD.strip(scene([[dj(0.65, reach=-0.2)] for _ in range(8)]), None, None, 0.0, 1.4, CW, 1.0, dj=True)
    assert np.all(left["x"] + CW / 2 - 0.65 >= LD.FACE_IN * hw) and np.all(left["x"] + CW / 2 - 0.65 < hw + 0.01)


def test_decks_in_front_of_the_dj_keep_their_head_near_the_middle():
    """Facing the camera over the decks (hands a little to one side): the strip leans only a little, the decks
    across its full-height bottom."""
    front = LD.strip(scene([[dj(0.5)] for _ in range(8)]), None, None, 0.0, 1.4, CW, 1.0, dj=True)
    assert front["on"] == "decks" and abs(front["x"].mean() - 0.5) < 0.02
    lean = LD.strip(scene([[dj(0.5, reach=0.06)] for _ in range(8)]), None, None, 0.0, 1.4, CW, 1.0, dj=True)
    assert 0.5 < lean["x"].mean() <= 0.5 + LD.BELOW * CW + 1e-3


def test_arms_up_on_a_drop_dont_swing_a_djs_strip():
    frames = [[dj(0.35, reach=0.2)] for _ in range(5)] + [[dj(0.35, up=True)] for _ in range(5)]
    out = LD.strip(scene(frames), None, None, 0.0, 1.8, CW, 1.0, dj=True)
    assert np.ptp(out["x"]) < 0.02 and out["x"].mean() > 0.4


def test_the_dj_is_the_one_at_the_decks_not_a_bigger_hype_man_beside_them():
    frames = [[dj(0.3, reach=0.1), dj(0.75, up=True, top=0.05, h=0.95, w=0.18)] for _ in range(8)]
    out = LD.strip(scene(frames), None, None, 0.0, 1.4, CW, 1.0, dj=True)
    assert out["on"] == "decks" and abs(out["lead"]["head"][:, 0].mean() - 0.3) < 0.02


def test_hands_on_the_decks_rank_over_arms_up():
    down = scene([[dj(0.4, reach=0.12)] for _ in range(13)])
    up = scene([[dj(0.4, up=True)] for _ in range(13)])
    assert LD.decks(down, None, 0.0, 2.4) > 0.8
    assert LD.decks(up, None, 0.0, 2.4) < 0.1
    # hands only, no head (the mixer from above): the decks
    frames = []
    for _ in range(13):
        kp = np.zeros((17, 3))
        for k, (x, y) in {7: (0.52, 0.05), 8: (0.75, 0.05), 9: (0.47, 0.3), 10: (0.62, 0.3)}.items():
            kp[k] = [x * AW, y * AH, 0.9]
        frames.append([([0.3 * AW, 0.0, 1.0 * AW, 0.7 * AH, 0.9], kp)])
    assert LD.decks(scene(frames), None, 0.0, 2.4) > 0.8


def dj_project(tmp_path, frames, picks=None):
    pr = Project(str(tmp_path))
    os.makedirs(pr.path("scores"))
    d = scene(frames)
    np.savez(pr.path("scores", "C7.npz"), pose_t=d["pose_t"], boxes=d["boxes"], kp=d["kp"], size=np.array([3840, 2160]),
             t=d["pose_t"], subj=np.ones(len(frames)))
    if picks:
        json.dump(picks, open(pr.path("picks.json"), "w"))
    return pr, dict(id="C7_dj-test", tag="dj-test", category="broll", role="transition", score="C7", clip="C7",
                    orient="landscape", t0=0.0, t1=len(frames) * 0.2)


def test_a_dj_shot_is_never_zoomed_in_and_frames_the_decks_in_portrait(tmp_path):
    frames = [[dj(0.4, reach=0.2, top=0.35, h=0.65)] for _ in range(20)]      # small in frame: a zoom would come
    pr, u = dj_project(tmp_path, frames)
    c = PL.reframe(pr, u, 1.0, 1.4, "portrait")
    assert c["z"] == 1.0 and c["h"] == 1.0 and c["on"] == "decks"
    xs = np.array(c["path"])[:, 1]
    assert xs.mean() > 0.5 and xs.mean() - c["w"] / 2 < 0.4                  # on the decks, the face's near side in
    other = dict(u, id="C7_band-test", tag="band-test")                       # the same footage as a band's: zoomed
    assert PL.reframe(pr, other, 1.0, 1.4, "portrait").get("on") != "decks"
    # the validator measures it as it plays: the DJ in the strip, nothing broken
    shot = dict(unit=u, src_in=1.0, length=1.4, crop=c, act="night", beats=2, fx=[])
    m = VA.measure(pr, u, 1.0, 1.4, c, "portrait")
    assert m["sub"]["seen"] >= 0.9 and not [w for w in VA.check(m, u, VA.RULES) if w[0] in ("out", "headcut", "halved")]
    assert shot["crop"]["w"] < 0.5


def test_the_validator_lets_a_djs_head_sit_on_the_strips_edge_but_not_someone_else_halved():
    from .test_validate import person
    from .test_validate import shot as measured
    band = dict(id="C1_band", tag="band", role="transition", category="broll")
    deck = dict(id="C2_dj-x", tag="dj-x", role="transition", category="broll")
    edge = measured(person(edge=0.8, cut=0.8))
    assert {w[0] for w in VA.check(edge, band, VA.RULES)} >= {"out", "headcut"}
    on_decks = measured(person(edge=0.8, cut=0.8, deck=True))
    assert not {w[0] for w in VA.check(on_decks, deck, VA.RULES)} & {"out", "headcut"}
    gone = measured(person(seen=0.2, deck=True))                              # off the strip altogether: still out
    assert "out" in {w[0] for w in VA.check(gone, deck, VA.RULES)}
    halved = dict(on_decks, halved=0.9)                                       # the other DJ cut in half: still fails
    assert "halved" in {w[0] for w in VA.check(halved, deck, VA.RULES)}


def test_a_dj_shot_the_editor_framed_stays_as_they_framed_it(tmp_path):
    frames = [[dj(0.4, reach=0.2, top=0.35, h=0.65)] for _ in range(20)]
    pr, u = dj_project(tmp_path, frames, picks={"C7_dj-test": {"portrait": [["0:01.50", 0.3, 0.5]]}})
    c = PL.reframe(pr, u, 1.0, 1.4, "portrait")
    assert c.get("on") != "decks" and c.get("focus")                   # his focal point, on the derived crop
    assert all(abs(p[1] - max(0.3, c["w"] / 2)) < 1e-3 for p in c["path"])
    later = PL.reframe(pr, u, 3.0, 0.8, "portrait")                    # a shot of the clip he didn't frame: the DJ's
    assert later["on"] == "decks" and later["z"] == 1.0


def test_the_planner_ranks_a_djs_hands_on_the_decks_over_arms_up(tmp_path):
    frames = [[dj(0.4, reach=0.12)] for _ in range(30)] + [[dj(0.4, up=True)] for _ in range(30)]
    pr, u = dj_project(tmp_path, frames)
    working, hype = PL.dj_rank(pr, u, 2.0, 2.4), PL.dj_rank(pr, u, 8.0, 2.4)
    assert working > 1.1 and hype < 0.9                                 # a bias, both still playable
    assert PL.dj_rank(pr, dict(u, id="C7_crowd", tag="crowd"), 8.0, 2.4) == 1.0
    # a long pick trimmed to its best part: the part on the decks
    a = PL.best_part(None, 1.0, 11.0, 2.0, bias=lambda x, L: PL.dj_rank(pr, u, x, L))
    assert a + 2.0 <= 6.2
    # assess, both formats: the moment on the decks ranks over the arms-up one of the same score
    rec = dict(id=u["id"], unit=u, role="transition", moments=[dict(t=3.0, score=1.0, kind="broll"),
                                                                dict(t=9.0, score=1.0, kind="broll")])
    for fmt in ("portrait", "landscape"):
        PL.assess(pr, [rec], fmt, PL.config(pr, fmt))
        key = "score_p" if fmt == "portrait" else "score_l"
        assert rec["moments"][0][key] > 1.2 * rec["moments"][1][key]
