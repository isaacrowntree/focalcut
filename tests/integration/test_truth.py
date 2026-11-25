"""Integration tests against the truth, not against focalcut's own past output.

Each test builds an input whose right answer is known by construction -- footage synthesised with ffmpeg where every
cut, pan, speck and exposure is put there on purpose -- or, where marked, real footage whose answer was checked by
eye, and asserts focalcut finds that answer within a stated tolerance. A test failing here means focalcut is wrong
about the world, not merely different from yesterday.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest

from ..conftest import needs_ffmpeg

pytestmark = needs_ffmpeg

W, H = 640, 360
TAG = "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"   # a camera file's own tags


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def raw_frames(path, fmt="gray", size=(W, H), extra=()):
    """Every frame of a file as arrays (8-bit gray by default)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), *extra, "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                         capture_output=True, check=True).stdout
    per = size[0] * size[1] * (3 if fmt == "rgb24" else 1) * (2 if "10" in fmt or "16" in fmt else 1)
    n = len(out) // per
    dt = "<u2" if per % 2 == 0 and ("10" in fmt or "16" in fmt) else np.uint8
    return np.frombuffer(out[:n * per], dt).reshape(n, -1)


# -- camera moves: a still picture moved by known amounts -------------------------------------------------------
@pytest.fixture(scope="module")
def texture(tmp_path_factory):
    """A large still with detail everywhere (noise, blurred to a texture the tracker can hold)."""
    d = tmp_path_factory.mktemp("moves")
    p = d / "tex.png"
    rng = np.random.default_rng(3)
    import cv2
    img = cv2.GaussianBlur((rng.random((1440, 2560)) * 255).astype(np.uint8), (0, 0), 2.0)
    cv2.imwrite(str(p), cv2.merge([img, img, img]))
    return d, p


def _moved(d, tex, name, vf, seconds=3):
    out = d / f"{name}.mp4"
    ff("-loop", "1", "-i", str(tex), "-t", str(seconds), "-r", "25", "-vf", vf + f",scale={W}:{H}",
       "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out))
    return out


def test_moves_reads_known_camera_moves(texture):
    """A 1280-wide window over the still: held; panned right at 0.1 frame widths a second; zoomed in (the window
    growing 8% a second); turned at 0.2 rad a second. Each must be named so, at about its true rate."""
    from focalcut.reel import moves as V
    d, tex = texture
    cases = {
        "held": "crop=1280:720:400:300",
        "pan": "crop=1280:720:'200+t*128':300",                       # 0.1 of the 1280 window per second
        "push": "scale=w='2560*exp(0.08*t)':h=-2:eval=frame,crop=1280:720",   # the picture grows 8% a second
        "spin": "rotate=a='0.2*t':ow=1600:oh=900,crop=1280:720",
    }
    got = {}
    for name, vf in cases.items():
        path = _moved(d, tex, name, vf)
        mv = V.frames_moves(V.grey(str(path), 0.2, 2.6, gpu=False, rate=10))
        got[name] = V.label(mv, rate=10)
    assert got["held"][0] == "held"
    assert got["pan"][0] == "pan" and abs(got["pan"][1]["net"] - 0.1) < 0.025          # frame widths a second
    assert got["push"][0] == "push" and abs(got["push"][1]["zoom"] - 0.08) < 0.02
    assert got["spin"][0] == "spin" and abs(got["spin"][1]["rot"] - 0.2) < 0.04


def test_moves_ignores_people_when_masked(texture):
    """A held camera with a large object sweeping across it: tracked on the background (the object masked out, as
    the score step's people boxes do), the camera reads as held."""
    from focalcut.reel import moves as V
    d, tex = texture
    path = _moved(d, tex, "crossing", "crop=1280:720:400:300,drawbox=x='100+t*300':y=150:w=300:h=500:c=white:t=fill")
    g = V.grey(str(path), 0.2, 2.6, gpu=False, rate=10)
    masks = np.full(g.shape, 255, np.uint8)
    masks[:, 60:330, :] = 0                                       # where the object travels (rows)
    assert V.label(V.frames_moves(g, masks), rate=10)[0] == "held"


# -- dust: specks of known depth in known places -------------------------------------------------------------------
def _y4m(path, planes, w, h, sub="422"):
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C{sub}p10 XYSCSS={sub.upper()}P10 XCOLORRANGE=LIMITED\n"
                 "FRAME\n".encode())
        for p in planes:
            fh.write(np.ascontiguousarray(p, dtype="<u2").tobytes())


def test_dust_division_restores_the_picture_exactly(tmp_path):
    """A clean 10-bit 4:2:2 frame (tagged BT.709, limited range, as camera files are), then a speck taking 30% of
    the light in a disc: the dust filter (dust.FILTER, the render's own) given the speck's map must give the clean
    frame back inside the speck (within rounding) and leave every other pixel bit-exact -- superwhites included."""
    from focalcut.reel import dust
    w, h = 320, 180
    rng = np.random.default_rng(0)
    y = rng.integers(200, 1019, (h, w)).astype(float)                 # up to superwhite (> 940)
    u = rng.integers(400, 624, (h, w // 2)).astype(float)
    v = rng.integers(400, 624, (h, w // 2)).astype(float)
    yy, xx = np.mgrid[:h, :w]
    disc = (xx - 100) ** 2 + (yy - 90) ** 2 <= 12 ** 2
    disc_c = disc[:, ::2]
    t = 0.7
    dy, du, dv = y.copy(), u.copy(), v.copy()
    dy[disc] = (y[disc] - 64) * t + 64
    du[disc_c] = (u[disc_c] - 512) * t + 512
    dv[disc_c] = (v[disc_c] - 512) * t + 512
    clean, dusty, mp = tmp_path / "clean.y4m", tmp_path / "dusty.y4m", tmp_path / "map.y4m"
    _y4m(clean, [y, u, v], w, h)
    _y4m(dusty, [np.round(dy), np.round(du), np.round(dv)], w, h)
    loss = np.where(disc, (1 - t) * 1023, 0)
    _y4m(mp, [np.round(loss), np.round(loss[:, ::2]), np.round(loss[:, ::2])], w, h)
    tagged = tmp_path / "dusty.mkv"                                  # tagged like the camera's files (setparams
    ff("-i", str(dusty), "-vf", TAG, "-c:v", "ffv1", str(tagged))   # labels; -colorspace on output would convert)
    out = tmp_path / "out.y4m"
    ff("-i", str(tagged), "-i", str(mp), "-filter_complex", dust.FILTER.format(fmt="yuv422p10le"),
       "-frames:v", "1", "-pix_fmt", "yuv422p10le", "-strict", "-1", str(out))
    raw = open(out, "rb").read()
    body = np.frombuffer(raw[raw.index(b"FRAME\n") + 6:], "<u2")
    oy = body[:w * h].reshape(h, w).astype(float)
    assert np.array_equal(oy[~disc], y[~disc]), "pixels outside the speck changed"
    assert np.abs(oy[disc] - y[disc]).max() <= 2, "the speck was not divided out"


def test_dust_rolloff_shape():
    """Below the knee the whole measured darkening comes out; towards white less, never below the floor."""
    from focalcut.reel import dust
    assert dust.rolloff(0.2) == dust.rolloff(0.6) == 1.0
    assert 0.3 <= dust.rolloff(0.99) < dust.rolloff(0.8) < 1.0


# -- exposure: one stop too bright ----------------------------------------------------------------------------------
def test_exposure_match_takes_exactly_the_extra_stop(tmp_path):
    """Three shots of a scene at ISO 200 (background 0.60 of white) and one at ISO 400 -- a stop brighter: in
    gamma-encoded video 0.60 x 2^(1/2.2). The match must bring that one down by exactly 2^(-1/2.2) and leave the
    others alone; and the lutyuv it renders with must scale luma above black by that factor."""
    from focalcut.reel import exposure as X
    bright = 0.60 * 2 ** (1 / 2.2)
    shots = [dict(unit=dict(id=f"u{i}", category="water", clip=f"C{i}"), src_in=0.0, length=2.0) for i in range(4)]
    cache = {X._key(s["unit"], s): dict(back=b) for s, b in zip(shots, (0.60, 0.60, 0.60, bright), strict=True)}
    g = X.match_gains({"clips": ["water"]}, shots, cache)
    assert set(g) == {3} and abs(g[3] - 2 ** (-1 / 2.2)) < 0.01
    src = tmp_path / "flat.mkv"
    ff("-f", "lavfi", "-i", f"color=c=0x{0xC0:02x}{0x80:02x}{0x80:02x}:s=64x64:d=0.2,format=yuv422p10le",
       "-c:v", "ffv1", str(src))
    y_in = raw_frames(src, "yuv422p10le", (64, 64))[0][:64 * 64].astype(float).mean()
    y_out = raw_frames(src, "yuv422p10le", (64, 64), ["-vf", X.gain_filter(g[3])])[0][:64 * 64].astype(float).mean()
    assert abs((y_out - 64) - (y_in - 64) * g[3]) < 1.0


# -- the reference film: a film made to a known edit -------------------------------------------------------------
@pytest.fixture(scope="module")
def film(tmp_path_factory):
    """A 'finished film' at 25 fps with every edit decision known:
         0.0-3.0  a soft sky-blue shot; a title ("Malabar 2026") fades in from 0.8 s to 1.6 s at x 240, y 92
         3.0-5.0  textured green
         5.0-7.0  a split screen: two sources side by side, halves 320 px
         7.0-7.8  a stutter: four 0.2 s jump-cuts between two textured sources
         7.8-9.8  textured brown
         9.8-12.8 an end card: a small dark disc on a light plain ground"""
    d = tmp_path_factory.mktemp("film")
    parts = []

    def seg(name, lavfi, secs, vf="null"):
        p = d / f"{name}.mp4"
        sep = ":" if "=" in lavfi else "="
        ff("-f", "lavfi", "-i", f"{lavfi}{sep}s={W}x{H}:r=25:d={secs}", "-vf", vf + ",format=yuv420p",
           "-c:v", "libx264", "-crf", "14", "-g", "1", str(p))
        parts.append(p)

    seg("a", "color=c=0x8fb6dc", 3.0, "drawbox=x=240:y=90:w=160:h=40:c=0x202020@1:t=fill:"
        "enable='gte(t,0.8)',fade=t=in:st=0:d=0.01")
    # the title as a real fade: overlay a dark bar whose alpha ramps 0.8 -> 1.6 s
    a2 = d / "a2.mp4"                                       # real lettering, fading in 0.8 -> 1.6 s
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    ff("-f", "lavfi", "-i", f"color=c=0x8fb6dc:s={W}x{H}:r=25:d=3",    # over footage-like ground (soft
       "-vf", "noise=alls=40:allf=t,gblur=sigma=3,"                    # texture: a plain ground with lettering
       f"drawtext=fontfile={font}:text='Malabar 2026':x=240:y=92:fontsize=30:fontcolor=0x202020:"   # on it is a card)
       "alpha='clip((t-0.8)/0.8,0,1)',format=yuv420p", "-c:v", "libx264", "-crf", "14", "-g", "1", str(a2))
    parts[0] = a2
    tex = "noise=alls=60,gblur=sigma=2.5"                     # footage-like: texture, no straight edges (test
    seg("b", "color=c=0x2a4a1c", 2.0, tex)                       # cards' bars have seams a split screen would too)
    sp = d / "split.mp4"                                    # two camera-like panels: textured, no hard lines
    ff("-f", "lavfi", "-i", f"color=c=0x6a8fb0:s=320x{H}:r=25:d=2", "-f", "lavfi",
       "-i", f"color=c=0xb08a5a:s=320x{H}:r=25:d=2",
       "-filter_complex", "[0:v]noise=alls=70:allf=t,gblur=sigma=2.5[a];[1:v]noise=alls=70:allf=t,gblur=sigma=2.5[b];"
       "[a][b]hstack=inputs=2,format=yuv420p", "-t", "2", "-c:v", "libx264", "-crf", "14", "-g", "1", str(sp))
    parts.append(sp)
    for k in range(4):
        seg(f"st{k}", "color=c=0xe8c898" if k % 2 == 0 else "color=c=0x182850", 0.2, tex)
    seg("c", "color=c=0xb09078", 2.0, tex)
    seg("card", "color=c=0xd8e4f0", 3.0, "drawbox=x=300:y=160:w=40:h=40:c=0x101010@1:t=fill")
    lst = d / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = d / "film.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
       "-r", "25", str(out))
    return out


def test_reference_reads_a_known_edit(film):
    from focalcut.reel import reference as R
    ref = R.analyse(str(film))
    t0s = [round(s["t0"], 2) for s in ref["shots"]]
    want = [0.0, 3.0, 5.0, 7.0, 7.2, 7.4, 7.6, 7.8, 9.8]
    assert len(t0s) == len(want), t0s
    assert all(abs(a - b) <= 0.04 for a, b in zip(t0s, want, strict=True)), t0s     # a frame at 25 fps
    by = {round(s["t0"]): s for s in ref["shots"]}
    split = next(s for s in ref["shots"] if abs(s["t0"] - 5.0) < 0.05)
    assert split.get("split") and len(split["split"]) == 2 and abs(split["split"][0][2] - 0.5) < 0.02
    stut = [s for s in ref["shots"] if "stutter" in s]
    assert len(stut) == 4 and abs(stut[0]["t0"] - 7.0) < 0.05
    card = ref["shots"][-1]
    assert card.get("card") and card["card"]["box"]
    bx = card["card"]["box"]
    assert abs(bx[0] - 300 / W) < 0.03 and abs(bx[1] - 160 / H) < 0.04 and bx[2] < 0.12
    title = ref["shots"][0].get("title")
    assert title, "the fading title was not found"
    tb = title["box"]
    assert abs(tb[0] - 240 / W) < 0.03 and abs(tb[1] - 92 / H) < 0.05 and 0.2 < tb[2] < 0.45
    assert 0.4 <= title["fade"][0] <= 1.3 and 1.2 <= title["fade"][1] <= 2.2
    assert not any(s.get("split") for k, s in by.items() if k != 5)


# -- real footage, checked by eye -------------------------------------------------------------------------------------
MALABAR = "/mnt/j/Videos/2026/2026-11-21 Grace City Church Baptisms - Malabar Beach"


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/scores/C2211.npz")),
                    reason="the Malabar baptism footage is not on this machine")
def test_dunks_in_c2211_are_the_three_seen():
    """C2211 (Grace City baptisms, tripod): three baptisms, confirmed frame by frame by eye -- the candidates go
    under at about 81.6 s, 164.4 s and 289.6 s. The dip detector must find exactly those, and the crowd's cheer must
    follow each rise."""
    from focalcut.reel import events as E
    from focalcut.reel import frame as FR
    d = FR.load(os.path.join(MALABAR, "focalcut/reel/scores/C2211.npz"))
    got = E.dip(d["pose_t"], d["boxes"], d["ah"], d["t"], d["subj"], aw=d["aw"])
    assert [round(a) for a, _b, _s in got] == [82, 164, 290], got
    proxy = os.path.join(MALABAR, "export/baptisms/C2211.mp4")
    heard = [c for c, _ in E.cheer(E.level(proxy))]
    for _a, b, _s in got:
        assert any(0.0 <= c - b <= 3.0 for c in heard), (b, heard)


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/plan_landscape.json")),
                    reason="the Malabar baptism plan is not on this machine")
def test_malabar_plan_keeps_its_promises():
    """What the cut must do, whatever it chooses: all 13 baptisms in, in the order they happened; every speaker in
    the lineup on screen; the A7V never under a LUT; the drone shots of the baptisms flying (no landing stretch at
    the end of DJI 0018); the reel running to the song's end."""
    plan = json.load(open(os.path.join(MALABAR, "focalcut/reel/plan_landscape.json")))
    cfg = json.load(open(os.path.join(MALABAR, "focalcut/reel/reel.json")))
    dunks = [s for s in plan["shots"] if s.get("dunk")]
    assert len(dunks) == 13
    order = [(s["unit"]["clip"], s["t"]) for s in dunks]
    assert order == sorted(order)
    shown = {p["unit"]["id"] for s in plan["shots"] for p in (s.get("split") or {}).get("panels", [])}
    shown |= {s["unit"]["id"] for s in plan["shots"]}
    speakers = sorted(f[:-4] for f in os.listdir(os.path.join(MALABAR, "export/speakers")) if f.endswith(".mp4"))
    assert set(speakers) <= shown, set(speakers) - shown
    assert not any(k.startswith("C2") or k == "C*" for k in cfg.get("lut", {}))
    import numpy as np
    z = np.load(os.path.join(MALABAR, "focalcut/reel/reference_frames.npz"))
    units, times, high = z["unit"].astype(str), z["t"], z["high"].astype(float)
    floor = (cfg.get("steady") or {}).get("air_high", 0.4)
    for s in plan["shots"]:                                   # every frame of every drone shot reads as flying
        if s["unit"]["id"].startswith("DJI") and not s.get("card") and not s.get("pick"):   # (the editor's word wins)
            span = s["length"] * (s.get("speed") or 1.0)
            m = (units == s["unit"]["id"]) & (times >= s["src_in"] - 1) & (times <= s["src_in"] + span + 1)
            assert not m.any() or high[m].min() >= floor, (s["unit"]["id"], s["src_in"], high[m])
    music = json.load(open(os.path.join(MALABAR, "focalcut/reel/music.json")))
    beat = music["beat"]                                      # the film runs to the song's end, within a beat
    if plan["stinger_at"] < plan["video_length"] - 1.0:       # a film shortened (slots with nothing to play left
        assert plan["music_length"] == plan["video_length"]     # out): the song fades out over the card
    else:
        assert abs(plan["video_length"] + plan["music_start"] - (music["start"] + music["length"])) < beat + 0.1


# -- the render keeps colour: what goes in flat comes out flat ------------------------------------------------------
class _Project:
    """Just enough of a Project for render.split_shot_on: a folder of tagged sources, no config."""
    def __init__(self, folder):
        self.folder, self.config, self.reel_dir = str(folder), {}, str(folder)

    def decode_from(self, source, t):
        return os.path.join(self.folder, source), t

    def size(self, fmt):
        return (W, H)

    def path(self, *p):
        return os.path.join(self.folder, *p)


def test_split_screen_keeps_colour_and_layout(tmp_path):
    """Two flat-colour BT.709 sources (as camera files are tagged) in a side-by-side split: after the panels have
    arrived, each panel shows its source's exact colour (within 2 of 255 in RGB -- 8-bit 4:2:0 rounding), the
    gutter shows the border colour, and nothing is shifted by a colour-matrix conversion on the way out."""
    from focalcut.reel import render
    for name, col in (("left.mp4", "0x3c78c8"), ("right.mp4", "0xd25a3c")):
        ff("-f", "lavfi", "-i", f"color=c={col}:s=1280x720:r=50:d=2",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p," + TAG,
           "-c:v", "libx264", "-crf", "0", str(tmp_path / name))
    pr = _Project(tmp_path)
    unit = lambda n: dict(id=n, source=n, clip=n, category="x", role="show", t0=0.0, t1=2.0)   # noqa: E731
    s = dict(length=1.6, frames=80, unit=unit("left.mp4"), split=dict(
        rects=[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0]], style="slide",
        panels=[dict(unit=unit("left.mp4"), src_in=0.0, t=0.5), dict(unit=unit("right.mp4"), src_in=0.0, t=0.5)]))
    out = tmp_path / "split.mp4"
    render.split_shot_on(pr, s, "landscape", str(out), qsv=False)
    fr = raw_frames(out, "rgb24", (W, H), ["-vf", "select=eq(n\\,70)"])[0].reshape(H, W, 3).astype(int)
    left = raw_frames(tmp_path / "left.mp4", "rgb24", (1280, 720))[0][:3].astype(int)       # each source as it
    right = raw_frames(tmp_path / "right.mp4", "rgb24", (1280, 720))[0][:3].astype(int)     # decodes: the truth
    g = round(W * render.SPLIT["gutter"] / 2) * 2
    assert np.abs(fr[H // 2, W // 4] - left).max() <= 2, (fr[H // 2, W // 4], left)
    assert np.abs(fr[H // 2, 3 * W // 4] - right).max() <= 2, (fr[H // 2, 3 * W // 4], right)
    assert np.abs(fr[g // 4, W // 2] - [0xF2, 0xEF, 0xEA]).max() <= 6, fr[g // 4, W // 2]     # the border


@pytest.mark.xfail(strict=True, reason="known limitation: a dead-level, crisp horizon between two textureless fields "
                   "is indistinguishable, from one frame, from a stacked split screen")
def test_a_sea_horizon_is_not_a_split_screen(tmp_path):
    """A single camera picture with a crisp horizon running the whole width (sky over sea, as a long lens on a
    clear day shows it: soft over a couple of pixels, texture on both sides) must not be read as two panels."""
    import cv2

    from focalcut.reel import reference as R
    rng = np.random.default_rng(5)
    img = np.zeros((H, W, 3), np.float32)
    img[: H // 2] = [210, 180, 140]                                     # sky (BGR)
    img[H // 2:] = [120, 90, 40]                                        # sea
    img += cv2.GaussianBlur(rng.normal(0, 12, (H, W, 3)).astype(np.float32), (0, 0), 1.5)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)                            # a lens's softness
    assert R.split_layout(np.clip(img, 0, 255).astype(np.uint8)[..., ::-1]) is None


REFERENCE = os.path.join(MALABAR, "Reference - Baptism example (Malabar 2025).mp4")


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason="the 2025 reference film is not on this machine")
def test_reference_film_splits_are_the_five_seen():
    """The 2025 Malabar film, checked shot by shot by eye: five split screens (at 30.8 s, 33.5 s, 42.7 s, 71.9 s and
    75.7 s; the third is 2 over 1, the fourth stacked), its end card at 238 s, its clap stutter at 57 s -- and none
    of its many sea horizons read as a split."""
    from focalcut.reel import reference as R
    ref = R.analyse(REFERENCE)
    splits = [round(s["t0"], 1) for s in ref["shots"] if s.get("split")]
    assert splits == [30.8, 33.5, 42.7, 71.9, 75.7], splits
    cards = [round(s["t0"]) for s in ref["shots"] if s.get("card")]
    assert cards == [238], cards
    assert any(56.8 <= s["t0"] <= 57.8 for s in ref["shots"] if "stutter" in s)


# -- exposure: a raised ISO taken back off the scene, the faces spared -----------------------------------------------
def test_protected_gain_spares_skin_and_takes_the_water_down(tmp_path):
    """A 10-bit luma ramp through the render's protected gain (exposure.gain_filter): every level comes out where
    exposure.protected() says (within 4 of 1023 codes); levels below the protect point (dark skin) exactly as shot;
    above the top (the bright water) scaled by the full gain; the curve rising everywhere (no tone inversion), and
    the same for the strongest gain the match allows."""
    from focalcut.reel import exposure as X
    w, h = 1024, 8
    y = np.tile(np.linspace(64, 940, w).round(), (h, 1))
    for g in (0.685, 0.45):
        p = X.protect_span(g, X.PROTECT)
        src = tmp_path / "ramp.y4m"
        _y4m(src, [y, np.full((h, w // 2), 512), np.full((h, w // 2), 512)], w, h)
        out = tmp_path / "out.y4m"
        ff("-i", str(src), "-vf", X.gain_filter(g, p), "-pix_fmt", "yuv422p10le", "-strict", "-1", str(out))
        raw = open(out, "rb").read()
        o = np.frombuffer(raw[raw.index(b"FRAME\n") + 6:], "<u2")[:w * h].reshape(h, w)[4].astype(float)
        want = np.array([64 + 876 * X.protected((v - 64) / 876, g, p) for v in y[4]])
        assert np.abs(o - want).max() <= 5                                   # (an 8-bit key: a code or so)
        lv = (y[4] - 64) / 876
        low = lv < p[0]
        if low.any():
            assert np.abs(o[low] - y[4][low]).max() <= 2                     # skin and shadows as shot
        top = lv > p[1] + 0.01
        if top.any():
            assert np.abs(o[top] - (64 + (y[4][top] - 64) * g)).max() <= 4   # the bright scene: the full gain
        # rising: no tone inversion (the key is read from 8-bit luma, so steps of up to one 8-bit code -- 4 of
        # 1023 -- below what the 8-bit film carries)
        assert np.diff(want).min() >= 0 and np.diff(o).min() >= -4


def test_face_window_qualifier_leaves_bright_surroundings(tmp_path):
    """The face lift's luma qualifier (window.graph with knee()): through a window that is fully on, a dark level
    (a face at 0.15) is lifted and a bright one (water at 0.7) is left exactly as shot; the curve never inverts."""
    from focalcut.reel import window as V
    a, k = V.amount(0.15, 0.38, 0.7)
    kn = V.knee(0.15, a)
    ys = np.linspace(0, 1, 501)
    f = [y + V.key(y, kn) * (V.lifted(y, a) - y) for y in ys]
    assert np.diff(f).min() > 0
    w, h = 64, 16
    lv = np.r_[np.full(w // 2, 0.15), np.full(w // 2, 0.7)]
    yy = np.tile(np.round(64 + 876 * lv), (h, 1))
    src, mask, out = tmp_path / "pic.y4m", tmp_path / "mask.y4m", tmp_path / "out.y4m"
    _y4m(src, [yy, np.full((h, w // 2), 512), np.full((h, w // 2), 512)], w, h)
    with open(mask, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 Cmono\nFRAME\n".encode() + bytes([255]) * (w * h))
    g = V.graph(a, k, "1:v", kn=kn).rstrip(",")
    ff("-i", str(src), "-i", str(mask), "-filter_complex", f"[0:v]{g}", "-frames:v", "1", "-pix_fmt", "yuv422p10le",
       "-strict", "-1", str(out))
    raw = open(out, "rb").read()
    o = np.frombuffer(raw[raw.index(b"FRAME\n") + 6:], "<u2")[:w * h].reshape(h, w)[h // 2].astype(float)
    assert o[8] > yy[0, 8] + 30                                   # the face lifted
    assert abs(o[-8] - yy[0, -8]) <= 2                            # the water as shot


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/scores/C0771_moves.npz")),
                    reason="the Malabar baptism footage is not on this machine")
def test_camera_faults_are_the_ones_seen():
    """Camera faults (template.Steady.faults) on real footage, each checked frame by frame by eye:
       C0771 (handheld)  the camera dropped at the end of the take, from 10 s -- a fault; 5-9 s held steady -- clean
       C0698 (handheld)  the camera dropped onto a box and the grass from about 8 s -- a fault
       C2211 (tripod)    the lens hunting for focus at 289.1-289.5 s, just before a baptism -- a fault; the dunk of
                         81.6 s with the 3 s before it -- clean
       C2183 (handheld)  a walk with the horizon rolling (4-7 s) -- a fault under the default rules (a project may
                         call such clips "walking" -- steadied hard, their sway forgiven: steady "walking")"""
    from focalcut.reel import template as T
    from focalcut.reel.project import Project
    pr = Project(MALABAR)
    pr.config["steady"] = {k: v for k, v in (pr.config.get("steady") or {}).items() if k != "walking"}
    st = T.Steady(pr, T.Cameras(pr, pr.load("moments.json")))
    units = {u["id"]: u for u in pr.load("units.json")}
    assert not st.camera_ok(units["C0771"], 9.9, 10.9)
    assert st.camera_ok(units["C0771"], 5.0, 9.0)
    assert not st.camera_ok(units["C0698"], 8.0, 9.0)
    assert "blur" in st.camera_why(units["C2211"], 289.0, 289.6)
    assert st.camera_ok(units["C2211"], 78.6, 82.6)
    assert not st.camera_ok(units["C2183"], 4.0, 7.0)


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/plan_landscape.json")),
                    reason="the Malabar baptism plan is not on this machine")
def test_malabar_plan_plays_no_camera_fault():
    """Every shot and every split-screen panel from a ground camera plays clear of camera faults over the whole
    stretch of source it plays (a dunk the window its ramp reads; a stutter twice its length) -- the editor's own
    picks excepted, a grid's cells (a sixteenth of the frame) held only to the mic being held, and a shot the plan
    itself flags (no stretch of its moment keeps the rule: reported in the plan log, not hidden)."""
    from focalcut.reel import template as T
    from focalcut.reel.project import Project
    pr = Project(MALABAR)
    cams = T.Cameras(pr, pr.load("moments.json"))
    st = T.Steady(pr, cams)
    plan = pr.load("plan_landscape.json")
    faults = []
    for i, s in enumerate(plan["shots"]):
        if s.get("card") or s.get("pick") or s.get("photo") or s.get("photos"):
            continue                                                   # (a still has no camera to fault)
        L = s["length"]
        span = 2 * L if s.get("stutter") else L if s.get("dunk") else L * (s.get("speed") or 1.0)
        a0 = s["src_in"]
        if s.get("dunk") and s.get("fx"):                    # a ramp reads its own window around the moment
            from focalcut.fx.speed import ramp_window
            pre, post, _ = ramp_window(s["fx"][0], L, s["peak"], s.get("beat") or plan["beat"],
                                       params={"ramps": pr.config.get("ramps")})
            a0, span = s["t"] - pre, pre + post
        items = [(p["unit"], p["src_in"], L) for p in s["split"]["panels"]] if s.get("split") else \
            [(s["unit"], a0, span)]
        grid = s.get("split") and len(s["split"]["panels"]) > 4      # (a grid's tiny cells: the mic held, not the
        for u, a, n in items:                                          # camera rule)
            if grid:
                if not st.held_ok(u, a, n):
                    faults.append((i, u["id"], a, "mic not held"))
            elif not cams.aerial(u["category"]) and not st.camera_ok(u, a, a + n) and not s.get("hard_fail"):
                faults.append((i, u["id"], a, st.camera_why(u, a, a + n)))   # (a fault the plan names -- no stretch
    assert not faults, faults                                                  # of it keeps the rule -- is reported)


def test_lut_segments_pick_the_stretch_a_shot_plays(tmp_path):
    """A clip whose camera changed part way (looks/segments.json): a shot takes the cube of the stretch its middle
    falls in; before the first or after the last stretch, the nearest; a clip not listed, reel.json "lut"."""
    from focalcut.reel import render
    (tmp_path / "looks").mkdir()
    (tmp_path / "looks" / "seg.json").write_text(json.dumps(
        {"C1": [[0.0, 7.68, "c1_s01.cube", "start"], [7.68, 9.12, "c1_s02.cube", "step"]]}))

    Pr = type("Pr", (), {"config": {"lut": {"C*": "looks/any.cube"},
                                    "lut_segments": str(tmp_path / "looks" / "seg.json")}})

    u = dict(clip="C1", source="C1.MP4")
    pick = lambda at: os.path.basename(render.lut_cube(Pr, u, at))      # noqa: E731
    assert pick(3.0) == "c1_s01.cube" and pick(8.0) == "c1_s02.cube" and pick(20.0) == "c1_s02.cube"
    assert render.lut_cube(Pr, dict(clip="C2", source="C2.MP4"), 3.0) == "looks/any.cube"
    assert render.mid_of(dict(src_in=6.0, length=2.0, speed=1.5)) == 7.5


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/plan_landscape.json")),
                    reason="the Malabar baptism plan is not on this machine")
def test_malabar_plan_tells_the_baptisms_in_order():
    """The day as it happened: the 13 baptisms in order; no ground shot from the water shows a moment from after the
    next baptism still to play (no one after their baptism before it); none of the baptisms' ground footage before
    the first one; after the last, the look back at the people in the water runs in the order it was shot."""
    from focalcut.reel import template as T
    from focalcut.reel.project import Project
    pr = Project(MALABAR)
    plan = pr.load("plan_landscape.json")
    units = [s["unit"] for s in plan["shots"]]
    wall = T.clocks(pr, units)
    cams = T.Cameras(pr, pr.load("moments.json"))

    def when(s):
        w = wall.get(s["unit"]["id"])
        return None if w is None else w + s["src_in"] - s["unit"]["t0"]
    shots = [s for s in plan["shots"] if not s.get("card")]
    dunks = [(k, when(s)) for k, s in enumerate(shots) if s.get("dunk")]
    assert [w for _k, w in dunks] == sorted(w for _k, w in dunks)
    first, last = dunks[0][1], dunks[-1][1]
    tail = []
    for k, s in enumerate(shots):
        w = when(s)
        if w is None or s.get("dunk") or cams.aerial(s["unit"]["category"]) or not first - 60 <= w <= last + 60:
            continue
        nxt = next((t for j, t in dunks if j > k), None)
        assert nxt is None or w <= nxt + 10, (k, s["unit"]["id"], s["src_in"])
        if nxt is None and not s.get("split"):
            tail.append(w)
    assert tail == sorted(tail)


def test_the_mic_is_held_by_the_hand_at_it_not_the_face_behind_it():
    """A speaker (big, near) holds a mic at their mouth; someone behind them (smaller) has their face right behind the
    mic in the picture. The holder is the speaker -- their wrist is at the mic -- not the face that overlaps it."""
    from focalcut.reel.holding import holder
    mic = np.array([0.51, 0.37, 0.69, 0.52])
    speaker = [0.30, 0.03, 0.69, 0.99]
    behind = [0.55, 0.17, 0.77, 0.99]
    kp = np.zeros((2, 17, 3))
    kp[0, 0] = [0.53, 0.26, 0.98]                   # speaker: nose above the mic, a wrist just under it
    kp[0, 10] = [0.57, 0.58, 0.92]
    kp[1, 0] = [0.62, 0.40, 0.91]                   # behind: nose inside the mic's box, wrists at the hips
    kp[1, 9] = [0.72, 1.00, 0.60]
    got = holder(mic, np.array([speaker, behind], float), kp)
    assert list(got) == speaker
    assert holder(None, np.array([speaker]), kp[:1]) is None


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/scores/C2207_holding.npz")),
                    reason="the Malabar baptism footage is not on this machine")
def test_speakers_are_whoever_holds_the_mic():
    """Checked by eye: in C2207 at 10.5 s and C2203 at 17.5 s the speaker -- left of frame, holding the mic -- is the
    holder, not the person standing just behind the mic; every speaker clip has its speaker holding the mic for most
    of it."""
    from focalcut.reel.holding import Holding
    from focalcut.reel.project import Project
    pr = Project(MALABAR)
    units = {u["id"]: u for u in pr.load("units.json")}
    h = Holding(pr)
    for uid, t, x0 in (("C2207", 10.5, 0.30), ("C2203", 17.5, 0.27)):
        b = h.at(units[uid], t - 0.25, t + 0.25)
        assert b is not None and abs(b[0] - x0) < 0.05, (uid, b)
    assert all(h.held(u, u["t0"], u["t1"]) >= 0.6 for u in units.values() if u["category"] == "speakers")


@pytest.mark.skipif(not os.path.exists(os.path.join(MALABAR, "focalcut/reel/scores/C2214.npz")),
                    reason="the Malabar baptism footage is not on this machine")
def test_tripod_stretches_without_a_subject_are_refused():
    """Checked by eye on the tripod (reel.json "subject": baptisms need two whole people): C2210 at 83.5 s is empty
    water with someone cut by the frame's edge, C2214 at 80 s and C2213 at 61 s one man waiting alone, C2217 at
    161 s people leaving the frame -- all refused; C2216 at 88.6 s (three people, a baptism) passes."""
    from focalcut.reel import template as T
    from focalcut.reel.project import Project
    pr = Project(MALABAR)
    pr.config["subject"] = {"baptisms": {"people": 2, "share": 0.9}}
    st = T.Steady(pr, T.Cameras(pr, pr.load("moments.json")))
    units = {u["id"]: u for u in pr.load("units.json")}
    for uid, a, n in (("C2210", 83.5, 1.5), ("C2214", 80.0, 2.8), ("C2213", 60.9, 1.5), ("C2217", 161.2, 1.5)):
        assert not st.people_ok(units[uid], a, n), uid
    assert st.people_ok(units["C2216"], 88.6, 2.0)
