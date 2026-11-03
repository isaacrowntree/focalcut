"""render.crop_expr: the moving crop's ffmpeg expression -- right values, and shallow enough for ffmpeg on long shots."""
import shutil
import subprocess

import numpy as np
import pytest

from focalcut.reel.render import crop_expr


def evaluate(expr, t):
    py = expr.replace("\\,", ",").replace("if(", "IF(").replace("lt(", "LT(")
    return eval(py, {"IF": lambda c, a, b: a if c else b, "LT": lambda a, b: a < b, "t": t})


def path(n, seed=0):
    rng = np.random.default_rng(seed)
    return [[round(i * 0.2, 3), float(rng.uniform(0.2, 0.8)), 0.5] for i in range(n + 1)]


@pytest.mark.parametrize("n", [1, 2, 7, 125])
def test_piecewise_linear_through_the_points(n):
    p = path(n)
    e = crop_expr(p, 1)
    ts = [q[0] for q in p]
    for t in np.linspace(-0.3, ts[-1] + 1, 97):
        want = np.interp(t, ts, [q[1] for q in p]) if t >= 0 else p[0][1] + (p[1][1] - p[0][1]) * t / 0.2
        assert evaluate(e, float(t)) == pytest.approx(want, abs=1e-6), t


def test_long_shot_stays_shallow():
    e = crop_expr(path(125), 1)
    depth = peak = 0
    for ch in e:
        depth += ch == "("
        depth -= ch == ")"
        peak = max(peak, depth)
    assert peak < 40


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg")
def test_ffmpeg_takes_a_25s_crop():
    e = crop_expr(path(125), 1)
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:d=0.1", "-vf",
                        f"crop=w=50:h=50:x=({e})*iw-ow/2", "-f", "null", "-"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
