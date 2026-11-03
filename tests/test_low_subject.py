"""Portrait: a subject sitting low in the strip (a band under a tall lighting rig) -- lift() zooms so the crop can
move down and centre it; what still sits low ranks well down (plays)."""
import numpy as np

from focalcut.reel import frame as FR
from focalcut.reel import plan

CW0, CH0 = FR.crop_size(16 / 9, 9 / 16)


def box(y0, y1, x0=0.45, x1=0.55, n=10):
    return np.tile([x0, y0, x1, y1], (n, 1)).astype(float)


def pos(union, z):
    ch = CH0 / z
    cy = float(np.median((union[:, 1] + union[:, 3]) / 2))
    by = np.clip(cy, ch / 2, 1 - ch / 2)
    return (cy - (by - ch / 2)) / ch


def test_a_centred_subject_keeps_its_zoom():
    u = box(0.2, 0.8)
    assert plan.lift(u, 1.0, CW0, CH0, 1.125) == 1.0


def test_a_subject_low_in_the_frame_is_zoomed_up_and_centred():
    u = box(0.6, 0.85)
    assert pos(u, 1.0) > plan.R["low"]
    z = plan.lift(u, 1.0, CW0, CH0, 1.125)
    assert 1.0 < z <= plan.R["low_zmax"] and pos(u, z) <= plan.R["low"]


def test_a_player_in_the_bottom_quarter_is_not_zoomed_for_nothing():
    u = box(0.75, 0.97)                                   # under a tall rig: even 1.5x leaves them low
    assert plan.lift(u, 1.0, CW0, CH0, 1.125) == 1.0      # (ranked down instead: test_a_low_subject_plays_badly)


def test_lift_never_zooms_past_the_subject():
    u = box(0.35, 1.0)                                    # tall and low: zooming in would cut them
    z = plan.lift(u, 1.0, CW0, CH0, 1.125)
    assert (u[0, 3] - u[0, 1]) <= 0.9 * CH0 / z + 1e-9
    wide = box(0.75, 0.97, 0.2, 0.8)                      # a whole band wider than the strip: no zoom helps
    assert plan.lift(wide, 1.0, CW0, CH0, 1.125) == 1.0


def test_a_low_subject_plays_badly():
    good = dict(present=1, inside=1, action=1, w=0.32, split=0, body=1, low=0.5)
    assert plan.plays(dict(good, low=0.8), "show") < 0.2 * plan.plays(good, "show")
    assert plan.plays(dict(good, low=0.8, w=1.0), "show") == plan.plays(dict(good, w=1.0), "show")   # portrait only


# --- who the portrait strip is about: backs, a hidden partner, a headless performer, a partner torn by the strip ---

def people(boxes, F=10):
    """Pose boxes as frame fractions (F, P, 5), the same in every frame."""
    nb = np.full((F, len(boxes), 5), np.nan)
    for p, b in enumerate(boxes):
        nb[:, p, :4] = b
        nb[:, p, 4] = 0.9
    return nb


def kp_faces(F, P, nose):
    kp = np.zeros((F, P, 17, 3))
    kp[:, :, 0, 2] = nose
    return kp


def show(nb, kp, fseen, who=0, bx=0.5, cw=0.6, by=0.5, ch=1.0):
    F = nb.shape[0]
    return plan.shown(nb, kp, np.asarray(fseen, bool), np.full(F, who), np.ones(F, bool), np.full(F, bx), np.full(F, by),
                      cw, ch)


def test_a_back_hiding_the_partner_is_unusable_however_good_the_hair():
    nb = people([[0.40, 0.1, 0.62, 0.95], [0.45, 0.15, 0.60, 0.9]])     # the follow's back over the lead
    hide, _, back, _ = show(nb, kp_faces(10, 2, 0.0), np.zeros((10, 2)))
    assert hide == 1.0 and back == 1.0
    assert plan.unusable(dict(hide=hide), "dance") == "hide"
    hide, _, back, _ = show(nb, kp_faces(10, 2, 0.9), np.tile([False, True], (10, 1)))   # the lead's face over her
    assert hide == 0.0 and back == 0.0                                                  # shoulder: fine


def test_a_bystanders_face_does_not_rescue_a_back():
    nb = people([[0.40, 0.1, 0.60, 0.95], [0.75, 0.2, 0.80, 0.6]])     # the subject's back; a face at the edge
    _, _, back, _ = show(nb, kp_faces(10, 2, 0.0), np.tile([False, True], (10, 1)))
    assert back == 1.0 and plan.unusable(dict(back=back), "show") == "back"


def test_a_performers_head_cut_off_by_the_crop_top():
    nb = people([[0.40, 0.05, 0.60, 0.95]])
    _, head, _, _ = show(nb, kp_faces(10, 1, 0.9), np.ones((10, 1)), by=0.6, ch=0.7)   # crop top at 0.25
    assert head == 0.0 and plan.unusable(dict(head=head), "show") == "head"
    assert plan.unusable(dict(head=head), "dance") == "head"                           # dancers too
    nb[:, 0, 1] = 0.0                                                                  # cut by the camera's own top
    assert show(nb, kp_faces(10, 1, 0.9), np.ones((10, 1)))[1] == 0.0
    assert plan.unusable(dict(present=0.6), "dance") == "exit"                         # walks out during the shot


def test_a_partner_half_in_the_strip_or_in_and_out_is_torn():
    lead, follow = [0.45, 0.1, 0.58, 0.95], [0.62, 0.12, 0.75, 0.93]
    fs = np.ones((10, 2))
    _, _, _, torn = show(people([lead, follow]), kp_faces(10, 2, 0.9), fs, bx=0.53, cw=0.3)  # strip 0.38-0.68
    assert torn > 0.9 and plan.unusable(dict(torn=torn), "dance") == "torn"
    _, _, _, both = show(people([lead, follow]), kp_faces(10, 2, 0.9), fs, bx=0.6, cw=0.5)   # both in: framing
    assert both == 0.0
    nb = people([lead, follow])
    nb[5:, 1, :4] = [0.70, 0.12, 0.83, 0.93]                               # she steps out of the strip half the time
    _, _, _, flick = show(nb, kp_faces(10, 2, 0.9), fs, bx=0.53, cw=0.34)
    assert flick >= 0.5


def test_someone_sliced_by_the_cameras_own_edge_counts_as_split():
    F, aw, ah = 10, 640.0, 360.0
    boxes = np.full((F, 2, 5), np.nan, np.float32)
    boxes[:, 0, :4] = [0.30 * aw, 0.1 * ah, 0.45 * aw, 0.95 * ah]        # the subject
    boxes[:, 1, :4] = [0.0, 0.2 * ah, 0.06 * aw, 0.9 * ah]               # a sliver of the partner at the frame edge
    d = dict(boxes=boxes, kp=np.zeros((F, 2, 17, 3), np.float32), aw=aw, ah=ah, pose_t=np.arange(F) * 0.2)
    union = np.tile([0.30, 0.1, 0.45, 0.95], (F, 1))
    cut, _ = plan.sliced(d, np.arange(F), union, np.full(F, 0.25), np.full(F, 0.5), 0.5, 1.0)
    assert cut == 1.0


def test_portrait_rules_come_from_the_config():
    try:
        plan.portrait_rules({"portrait_rules": {"torn": 0.9, "nonsense": 1}})
        assert plan.R["torn"] == 0.9 and "nonsense" not in plan.R
        assert plan.unusable(dict(torn=0.5), "dance") is None
    finally:
        plan.portrait_rules({})
    assert plan.R == plan.RULES


def test_the_picker_passes_over_unusable_moments_while_it_has_others():
    r = {"id": "U1", "role": "transition", "unit": {"clip": "C1"}, "moments": [
        dict(t=1.0, kind="energy", score=0.9, score_p=0.9, unusable="shake"), dict(t=9.0, kind="energy", score=0.2,
                                                                                  score_p=0.2)]}
    P = plan.Picker("portrait", [r])
    assert P.best([r])[1]["t"] == 9.0
    r["moments"] = r["moments"][:1]
    assert P.best([r])[1]["t"] == 1.0                     # nothing else: still covered


def test_look_room_moves_the_strip_toward_a_profiles_gaze():
    F, aw, ah = 10, 640.0, 360.0
    boxes = np.full((F, 1, 5), np.nan, np.float32)
    boxes[:, 0, :4] = [0.45 * aw, 0.1 * ah, 0.55 * aw, 0.95 * ah]
    kp = np.zeros((F, 1, 17, 3), np.float32)
    kp[:, 0, 0] = [0.54 * aw, 0.15 * ah, 0.9]                  # nose to the right of the one visible ear: faces right
    kp[:, 0, 3] = [0.50 * aw, 0.15 * ah, 0.9]
    d = dict(boxes=boxes, kp=kp, aw=aw, ah=ah, pose_t=np.arange(F) * 0.2)
    union = np.tile([0.45, 0.1, 0.55, 0.95], (F, 1))
    g = plan.gaze(d, d["pose_t"], union, 1 / 6)
    assert np.allclose(g, 1 / 6, atol=0.02)                     # the face a third of the way in, space ahead
    kp[:, 0, 0, 0] = 0.46 * aw                                 # facing left
    assert (plan.gaze(d, d["pose_t"], union, 1 / 6) < -0.1).all()
    kp[:, 0, 4] = [0.54 * aw, 0.15 * ah, 0.9]; kp[:, 0, 0, 0] = 0.52 * aw   # both ears, nose between: to camera
    assert np.allclose(plan.gaze(d, d["pose_t"], union, 1 / 6), 0)


def test_band_broll_from_behind_is_framed_and_rejected_in_portrait_scenery_is_not():
    singer = {"id": "C2007_band-singer", "tag": "band-singer", "role": "transition"}
    assert plan.people_broll(singer)
    assert not plan.people_broll({"id": "C2160_venue-night", "tag": "venue-night", "role": "transition"})
    assert not plan.people_broll({"id": "C2148", "tag": "day", "role": "show"})      # (a subject already)
    nb = people([[0.45, 0.1, 0.70, 0.95], [0.10, 0.3, 0.20, 0.7]])                   # his back; a fan filming
    _, _, back, _ = show(nb, kp_faces(10, 2, 0.0), np.tile([False, True], (10, 1)))
    assert plan.unusable(dict(back=back, w=0.32), "transition") == "back"
    r = {"id": "C2007_band-singer", "role": "transition", "unit": singer, "moments": [dict(t=1.0, plays_p=0.05)]}
    assert not plan.framable(r, "portrait") and plan.framable(r, "landscape")
