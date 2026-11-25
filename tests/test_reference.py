"""Reference mode: learning an edit from a finished film (reference.py), cutting on it (template.py), the dunk and
cheer detectors (moments.py), and the render's split screens and stutters (render.py). No GPU, no footage."""
import numpy as np

from focalcut.reel import events as E
from focalcut.reel import moves as V
from focalcut.reel import reference as R
from focalcut.reel import render
from focalcut.reel import template as T


def test_shot_spans_join_flash_frames():
    assert R.shot_spans([1.0, 1.05, 3.0], 5.0) == [[0.0, 1.05], [1.05, 3.0], [3.0, 5.0]]


def test_stutters_need_three_short_shots():
    spans = [[0, 2], [2, 2.2], [2.2, 2.4], [2.4, 2.6], [2.6, 5], [5, 5.2], [5.2, 8]]
    assert R.stutters(spans) == [None, 0, 0, 0, None, None, None]


def _split(cols=2, rows=1, w=320, h=180):
    img = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(0)
    for c in range(cols):
        for r in range(rows):
            img[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols] = rng.integers(0, 255, 3)
    return img


def test_split_layout_finds_side_by_side_and_stacked():
    lay = R.split_layout(_split(2, 1))
    assert lay and len(lay) == 2 and abs(lay[0][2] - 0.5) < 0.02 and lay[1][0] > 0.45
    lay = R.split_layout(_split(1, 2))
    assert lay and len(lay) == 2 and abs(lay[0][3] - 0.5) < 0.02


def test_split_layout_leaves_a_plain_picture():
    img = np.tile(np.linspace(0, 255, 320, dtype=np.uint8)[None, :, None], (180, 1, 3))
    assert R.split_layout(img) is None


def test_viterbi_keeps_the_order_of_the_day():
    # two categories: A in the morning, B later; shot 2 looks a little more like A but B has begun -- it stays B
    sim = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.5], [0.1, 0.9]])
    spans = [(0.0, 100.0), (1000.0, 2000.0)]
    assert R.viterbi(sim, spans) == [0, 1, 1, 1]


def test_transfer_moves_stats_and_keeps_tone_monotone():
    src = dict(Lq=list(np.linspace(20, 70, 25)), a=[0.0, 5.0], b=[0.0, 5.0])
    dst = dict(Lq=list(np.linspace(10, 90, 25)), a=[2.0, 10.0], b=[-3.0, 10.0])
    f = R.transfer(src, dst, 1.0)
    lab = np.stack([np.linspace(0, 100, 50), np.zeros(50), np.zeros(50)], -1)
    out = f(lab)
    assert np.all(np.diff(out[:, 0]) >= 0)
    assert abs(out[25, 1] - 2.0) < 1e-6 and abs(out[25, 2] + 3.0) < 1e-6
    assert np.allclose(R.transfer(src, dst, 0.0)(lab), lab)


def test_lab_round_trip():
    rgb = np.array([[10, 20, 30], [200, 150, 100], [255, 255, 255]], float)
    assert np.allclose(R.lab2rgb(R.rgb2lab(rgb)), rgb, atol=0.6)


def _group(n_frames, dunk=None, hug=None, ah=360.0):
    """Three people standing (boxes 200 px tall, heads at y=100); a dunk takes one away while the others stoop."""
    t = np.arange(n_frames) * 0.2
    b = np.full((n_frames, 4, 5), np.nan)
    for k, x in enumerate((200, 300, 400)):
        b[:, k] = [x - 30, 100, x + 30, 300, 0.9]
    if dunk:
        m = (t >= dunk[0]) & (t < dunk[1])
        b[m, 1] = np.nan
        b[m, 0, 1] = b[m, 2, 1] = 170                     # the two still seen bend down
    if hug:
        m = (t >= hug[0]) & (t < hug[1])
        b[m, 1] = np.nan                                  # one hidden in a hug: nobody stoops
    return t, b, ah


def test_immersions_finds_a_dunk_and_not_a_hug():
    t, b, ah = _group(100, dunk=(8.0, 9.4))
    got = E.dip(t, b, ah)
    assert len(got) == 1 and abs(got[0][0] - 8.0) < 0.3 and abs(got[0][1] - 9.4) < 0.3
    t, b, ah = _group(100, hug=(8.0, 9.4))
    assert E.dip(t, b, ah) == []


def test_immersions_ignores_a_long_absence():
    t, b, ah = _group(150, dunk=(8.0, 20.0))
    assert E.dip(t, b, ah) == []


def test_stutter_graph_has_n_pieces():
    g = render.stutter_graph(4, 1.0, 2.0)
    assert g.startswith("split=4") and g.count("trim=") == 4 and g.endswith("concat=n=4:v=1:a=0")


def test_panel_crop_shape_and_bounds():
    w, h, x, _y = render.panel_crop([0, 0, 0.5, 1.0], 1920, 1080, cx=0.95)
    assert abs(w / h * 16 / 9 - (960 / 1080)) < 1e-6 and x + w <= 1.0 + 1e-9 and x >= 0


def test_slots_join_stutters_and_drop_the_tag():
    ref = dict(shots=[dict(i=0, t0=0, t1=2, length=2), dict(i=1, t0=2, t1=2.2, length=0.2, stutter=0),
                      dict(i=2, t0=2.2, t1=2.4, length=0.2, stutter=0), dict(i=3, t0=2.4, t1=4, length=1.6),
                      dict(i=4, t0=4, t1=9, length=5, after_music=True)])
    sl = T.slots(ref)
    assert [s["i"] for s in sl] == [0, 1, 3] and sl[1]["pieces"] == 2 and abs(sl[1]["length"] - 0.4) < 1e-9


def test_beat_grid_lands_cuts_on_beats():
    music = dict(beats=list(np.arange(0, 60, 0.5)), start=0.0, length=60.0)
    ref = dict(music_end=60.0, duration=60.0)
    sl = [dict(t0=0.0, t1=2.1), dict(t0=2.1, t1=2.2), dict(t0=2.2, t1=6.0)]
    beats, start = T.beat_grid(music, ref, sl)
    assert start == 0 and all(b >= 1 for b in beats) and sum(beats) == 12


def test_events_configure_defaults_and_old_key():
    ev = E.configure({"events": {"dunk": {"detect": ["dip", "cheer"], "in": ["baptisms"], "cheer": {"over": 25}}}})
    assert ev["dunk"]["cheer"]["over"] == 25 and ev["dunk"]["dip"]["least"] == 0.4
    assert "dunk" in E.configure({"immersion": ["baptisms"]})
    try:
        E.configure({"events": {"x": {"detect": ["nope"]}}})
        raise AssertionError("an unknown detector must be refused")
    except ValueError:
        pass


def test_cheer_finds_a_burst_over_the_murmur():
    db = np.full(200, -60.0)
    db[80:96] = -28.0                                        # 4 s of cheering, 32 dB over
    got = E.cheer(db)
    assert len(got) == 1 and 19.0 <= got[0][0] <= 20.5 and got[0][1] > 25


def test_moves_label():
    n = 30
    held = np.zeros((n, 4))
    spin = np.zeros((n, 4)); spin[:, 2] = 0.02               # 0.2 rad/s at 10 fps
    push = np.zeros((n, 4)); push[:, 3] = 0.01
    pan = np.zeros((n, 4)); pan[:, 0] = 0.01
    tilt = np.zeros((n, 4)); tilt[:, 1] = 0.01
    assert [V.label(x)[0] for x in (held, spin, push, -push, pan, tilt)] == ["held", "spin", "push", "pull", "pan", "tilt"]


def test_forward_never_walks_back_through_a_spread_category():
    # A morning (0-100), B later (1000-1100), C spread over all of it (0-3000: B-roll)
    spans = [(0.0, 100.0), (1000.0, 1100.0), (0.0, 3000.0)]
    sim = np.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9], [0.9, 0.1, 0.2]])
    assert R.forward([0, 1, 2, 0], sim, spans) == [0, 1, 2, 2]


def test_grid_fits_a_lineup_with_a_logo_cell():
    cells, logo = T.grid(14, "logo.png")
    assert len(cells) == 14 and logo is not None
    cols = round(1 / cells[0][2])
    rows = round(1 / cells[0][3])
    assert cols * rows == 15 and logo["rect"] not in cells


def test_columns_tile_the_frame():
    c = T.columns(3)
    assert len(c) == 3 and abs(sum(r[2] for r in c) - 1.0) < 1e-3 and all(r[3] == 1.0 for r in c)


def test_match_gains_bring_bright_shots_down_only():
    from focalcut.reel import exposure as X
    shots = [dict(unit=dict(id=f"u{i}", category="baptisms", clip=f"C22{i}"), src_in=0.0, length=2.0) for i in range(4)]
    cache = {X._key(s["unit"], s): dict(back=b) for s, b in zip(shots, (0.58, 0.60, 0.84, 0.62), strict=True)}
    g = X.match_gains({"clips": ["baptisms"]}, shots, cache)
    assert 2 in g and 0.65 < g[2] < 0.75 and 0 not in g


def test_dust_rolloff_and_filter():
    from focalcut.reel import dust as Dd
    assert Dd.rolloff(0.4) == 1.0 and Dd.rolloff(0.95) < 0.6
    assert "lut2" in Dd.FILTER and "{fmt}" in Dd.FILTER


def test_moves_frames_with_people_masked_ignores_people():
    rng = np.random.default_rng(1)
    bg = (rng.random((120, 160)) * 255).astype(np.uint8)
    import cv2
    bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
    g = np.stack([bg.copy() for _ in range(6)])
    for k in range(6):                                       # a "person" sliding across, the camera still
        g[k, 40:90, 20 + 8 * k:60 + 8 * k] = 255
    masks = np.full(g.shape, 255, np.uint8)
    masks[:, 30:100, :] = 0
    mv = V.frames_moves(g, masks)
    assert np.abs(mv[1:, 0]).max() < 0.01
