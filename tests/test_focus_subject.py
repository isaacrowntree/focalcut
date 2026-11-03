"""The subject follows the camera's focus: someone big and blurry in front never takes the crop from the sharp one."""
import numpy as np

from focalcut.reel import frame as FR


def shot(soft_frames=None, F=20):
    """A 640x360 clip: a sharp man on the left (in profile, mid-size) and a big blurry one close to the lens on the
    right, whom the subject track (saliency + depth) picked."""
    aw, ah = 640.0, 360.0
    boxes = np.full((F, 2, 5), np.nan, np.float32)
    boxes[:, 0] = [0.08 * aw, 0.2 * ah, 0.3 * aw, 0.95 * ah, 0.9]     # sharp
    boxes[:, 1] = [0.55 * aw, 0.0, 0.98 * aw, 1.0 * ah, 0.9]          # blurry, near the lens
    focus = np.tile(np.array([40.0, 40.0], np.float32), (F, 1))
    for f in (range(F) if soft_frames is None else soft_frames):
        focus[f, 1] = 6.0
    t = np.arange(F) * 0.2
    return dict(pose_t=t, boxes=boxes, focus=focus, aw=aw, ah=ah, kp=np.zeros((F, 2, 17, 3)),
                sub_t=t[::5], sub_box=np.tile([0.55, 0.0, 0.98, 1.0], (len(t[::5]), 1)).astype(np.float32),
                sub_touch=np.tile([False, True, True, True], (len(t[::5]), 1))), t


def test_a_blurry_foreground_person_loses_the_subject_to_the_sharp_one():
    d, t = shot()
    union, touch, _ = FR.track(d, t)
    cx = (union[:, 0] + union[:, 2]) / 2
    assert np.all(np.abs(cx - 0.19) < 0.03)                   # centred on the sharp man
    assert not touch[:, 2].any()                              # and no longer cut off by the right edge
    _, z, ccx, _ = FR.framing(union, 3840, 2160, 1080, 1920)
    cw = FR.crop_size(16 / 9, 9 / 16)[0] / z
    assert np.all(ccx - cw / 2 <= 0.08 + 1e-6) and np.all(ccx + cw / 2 >= 0.3 - 1e-6)   # the strip holds him whole


def test_one_soft_frame_is_motion_blur_not_the_cameras_choice():
    d, t = shot(soft_frames=[3])
    union, _, _ = FR.track(d, t)
    assert np.allclose(union[:, 0], 0.55, atol=0.02)          # the track's subject stays


def test_the_thresholds_are_reel_json_keys():
    d, t = shot()
    try:
        FR.configure({"focus_blur": 0.1})                     # 6 / 40 is not under a tenth: nobody is blurry
        union, _, _ = FR.track(d, t)
        assert np.allclose(union[:, 0], 0.55, atol=0.02)
    finally:
        FR.configure({})
    assert FR.FOCUS["blur"] == 0.4


def test_a_window_with_nothing_in_focus_is_never_a_moment():
    from focalcut.reel import moments as MO
    n = 30
    sharp = np.r_[np.full(10, 400.0), np.full(10, 20.0), np.full(10, 20.0)]   # sharp, then soft frames
    top = np.r_[np.full(10, 40.0), np.full(10, 5.0), np.full(10, 10.0)]      # ...the last third: a sharp person
    cam = np.r_[np.zeros(20), np.full(10, 2.0)]
    s = MO.soft(sharp, top, cam)
    assert not s[:3].any() and s[12:18].all() and not s[26:].any()   # (and the second around it)
    assert MO.soft(np.full(n, 20.0), np.full(n, 10.0), np.full(n, 12.0)).all()   # moving: the floor rises 1.5x
    assert not MO.soft(np.full(n, 400.0), np.full(n, 5.0), np.zeros(n)).any()    # a sharp frame is not soft
    assert MO.soft(np.full(n, 100.0), np.full(n, 5.0), np.zeros(n), typical=800.0).all()   # a dip in a sharp clip
    try:
        MO.configure({"soft_focus": 4})
        assert not MO.soft(np.full(n, 20.0), np.full(n, 5.0), np.zeros(n)).any()
    finally:
        MO.configure({})
    u = dict(id="u", category="demos", role="show", orient="landscape", t0=0.0, t1=6.0)
    t = np.arange(n) / MO.HZ
    parts = {k: np.full(n, 0.8) for k in MO.W}
    S = dict(t=t, parts=parts, quality=np.ones(n), steady=np.ones(n), activity=np.ones(n), face=np.zeros(n),
             framing={"landscape": np.ones(n), "portrait": np.ones(n)}, soft=s)
    _, total = MO.curve(u, S)
    assert (total[s] == 0).all() and (total[~s] > 0).all()
