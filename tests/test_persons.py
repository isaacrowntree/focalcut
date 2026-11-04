"""The people a shot is about (persons.py) and the pick rules measured on them (windows.person_reasons)."""
import numpy as np

from focalcut.reel import persons as PS
from focalcut.reel import windows as WI


def _kp(cx, top, w):
    """COCO keypoints of a person facing the camera: nose, eyes, ears around (cx, top + w/4); shoulders w apart."""
    kp = np.zeros((17, 3), np.float32)
    y = top + 0.25 * w
    kp[0] = cx, y, 0.9
    kp[1], kp[2] = (cx + 0.08 * w, y - 0.05 * w, 0.9), (cx - 0.08 * w, y - 0.05 * w, 0.9)
    kp[3], kp[4] = (cx + 0.15 * w, y, 0.6), (cx - 0.15 * w, y, 0.6)
    kp[5], kp[6] = (cx + 0.5 * w, top + w, 0.9), (cx - 0.5 * w, top + w, 0.9)
    return kp


def _face(cx, cy, size, score=0.95):
    f = np.zeros(15, np.float32)
    f[:4] = cx - size / 2, cy - size / 2, size, size
    f[4:6], f[6:8], f[8:10] = (cx - 0.2 * size, cy - 0.1 * size), (cx + 0.2 * size, cy - 0.1 * size), (cx, cy + 0.05 * size)
    f[14] = score
    return f


def test_main_people_are_the_tallest_and_those_nearly_as_tall_beside_them():
    boxes = np.array([[100, 50, 160, 350, 1], [170, 60, 230, 340, 1], [400, 150, 420, 230, 1], [500, 70, 560, 350, 1]],
                     np.float32)
    assert PS.main_people(boxes) == [0, 1]           # the far one (small) and the one across the floor (tall) aren't


def test_a_face_is_only_its_owners():
    frame = np.full((360, 640, 3), 128, np.uint8)
    boxes = np.array([[100, 50, 160, 350, 1]], np.float32)
    kp = np.stack([_kp(130, 50, 60)])
    own = _face(130, 65, 30)                         # on their own face points, their size: theirs
    r = PS.measure(frame, np.stack([own]), kp, boxes, 1.0, 1.0)
    assert r[1][0]
    far = _face(131, 66, 6)                          # a small face further back, showing past their head: not theirs
    r = PS.measure(frame, np.stack([far]), kp, boxes, 1.0, 1.0)
    assert not r[1][0]
    kp_back = kp.copy()
    kp_back[0, :3, 2] = 0.0                          # seen from behind: no nose or eyes, only the ears
    elsewhere = _face(300, 65, 30)
    r = PS.measure(frame, np.stack([elsewhere]), kp_back, boxes, 1.0, 1.0)
    assert not r[1][0]


def _track(face, yaw=0.0, rel=0.5, n=6, slots=1):
    t = np.arange(n, dtype=np.float32) * 0.5
    box = np.full((n, 3, 4), np.nan, np.float32)
    box[:, :slots] = [0.2, 0.1, 0.4, 0.9]
    f = np.zeros((n, 3), bool)
    f[:, :slots] = np.asarray(face, bool).reshape(n, slots) if np.ndim(face) else face
    return dict(t=t, box=box, face=f, yaw=np.full((n, 3), yaw, np.float32), rel=np.full((n, 3), rel, np.float32))


def test_person_rules():
    assert WI.person_reasons(_track(True), 0, 2.5)[0] == []
    assert WI.person_reasons(_track(False), 0, 2.5)[0] == ["back"]               # nobody's face: a back
    turn = np.array([[0, 1], [0, 1], [0, 0], [0, 1], [1, 1], [0, 1]], bool)
    assert WI.person_reasons(_track(turn, slots=2), 0, 2.5)[0] == []             # a couple mid-turn, her face on
    assert WI.person_reasons(_track(True, rel=0.03), 0, 2.5)[0] == ["soft"]      # a face smeared, something sharp
    assert WI.person_reasons(_track(True, yaw=4.0), 0, 2.5)[0] == []                 # side-on is not a back
    assert WI.person_reasons(_track(True, yaw=4.0), 0, 2.5, {"yaw": 3.0})[0] == ["profile"]   # unless asked
    assert WI.person_reasons(_track(True, rel=0.03), 0, 2.5, {"rel": 0.01})[0] == []   # reel.json person_rules
    assert WI.person_reasons(None, 0, 2.5) == ([], {})
