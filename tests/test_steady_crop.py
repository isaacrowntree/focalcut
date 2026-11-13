"""Light steadying in the crop: the crop rides the camera's jitter, only where that measurably steadies the shot."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from focalcut.reel import render  # noqa: E402
from focalcut.reel.shake import frames_motion, hf  # noqa: E402


def _frames(n=60, w=320, h=180, amp=3.0, seed=1):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h * 2, w * 2)) * 255).astype(np.uint8), (0, 0), 1.5)
    jx = amp * np.sin(np.arange(n) * 2.1) + rng.normal(0, amp / 2, n)       # a handheld wobble, no pan
    jy = amp * np.cos(np.arange(n) * 1.7)
    ox, oy = np.rint(jx).astype(int), np.rint(jy).astype(int)
    g = np.stack([base[h // 2 + y:h // 2 + y + h, w // 2 + x:w // 2 + x + w] for x, y in zip(ox, oy, strict=True)])
    return g


def test_steadies_a_wobbling_shot_and_leaves_a_steady_one(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "STEADY", str(tmp_path))
    for amp, want in ((3.0, True), (0.0, False)):
        g = _frames(amp=amp)
        mv = frames_motion(g)
        monkeypatch.setattr(render, "shot_motion",
                            lambda *a, frames=False, g=g, mv=mv: ((np.arange(len(mv)) / 50, mv, g) if frames
                                                                  else (np.arange(len(mv)) / 50, mv)))
        s = dict(unit=dict(source=f"x{amp}.mp4", id="u"), src_in=0.0, crop=None)
        c = render.steady_crop(None, s, len(mv) / 50)
        assert (c is not None) == want
        if c:
            assert c["w"] < 1 and len(c["path"]) > 10 and c.get("steadied")
            xs = np.array([p[1] for p in c["path"]])
            assert np.abs(xs - 0.5).max() <= render.STAB_CAP + 1e-6        # a small amount, never more
            assert hf(mv).mean() > 0


def test_long_filter_graph_moves_to_a_file(tmp_path):
    base = str(tmp_path / "shot.mp4")
    short = ["ffmpeg", "-vf", "scale=1:1", base]
    assert render.from_file(short, str, base) == short
    long = ["ffmpeg", "-i", "x", "-filter_complex", "a" * (render.LONG_ARGS + 10), base]
    out = render.from_file(long, str, base)
    assert out[3] == "-/filter_complex" and open(out[4]).read() == "a" * (render.LONG_ARGS + 10)
