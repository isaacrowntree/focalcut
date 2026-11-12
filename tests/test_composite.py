"""composite.Compositor (the overlay on the GPU) against overlay.frame (numpy): the same picture to within rounding,
through kicks, splits, drops (kaleidoscope, duotone, confetti, sparks), flickers and the MilkDrop layer; and its
BT.709 conversions. Runs on torch's CPU device, so it needs torch but no GPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)                                       # (small frames: more threads only get in the way)

from focalcut.reel import composite, overlay  # noqa: E402

FPS, W, H = 50, 320, 180


def scene(n=60, milkdrop=True, seed=0):
    """A smooth moving test picture, a MilkDrop-like layer, and a grid with a downbeat, a drop and flickers."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = [np.stack([127 + 100 * np.sin(xx / 23 + k / 9), 127 + 100 * np.cos(yy / 17 - k / 7),
                        127 + 90 * np.sin((xx + yy) / 31 + k / 5)], -1).astype(np.uint8) for k in range(n)]
    lw, lh = (W, H) if milkdrop is True else milkdrop or (W, H)
    y2, x2 = np.mgrid[0:lh, 0:lw].astype(np.float32)
    layer = [np.stack([255 * (np.sin(x2 * 9 / lw + k / 4) > 0.3), 200 * (np.cos(y2 * 7 / lh - k / 6) > 0.2),
                       120 + 100 * np.sin((x2 - y2) * 5 / lw)], -1).astype(np.uint8) for k in range(n)] \
        if milkdrop else None
    beat = 0.4
    t = np.arange(n) / FPS
    kb = (t // beat).astype(int)
    G = dict(since=t - kb * beat, down=kb % 4 == 0, drop=np.where((kb >= 1) & (kb < 3), kb - 1, -1), bar=kb // 4, k=kb,
             beats=np.arange(8) * beat, energy=np.ones(8), drops=[1])
    F = dict(bands=rng.random((n, overlay.NB)).astype(np.float32), bass=rng.random(n).astype(np.float32) * 1.4,
             wave=np.zeros((n, 256), np.float32))
    S = np.full(n, 0.9, np.float32)
    return frames, layer, beat, G, F, S


def shows(beat, opts, milkdrop, n):
    out = []
    for _ in range(2):
        s = overlay.Show(W, H, FPS, beat, np.random.default_rng(7))
        s.edge = overlay.edge_mask(W, H, opts["clear"])
        s.flick = {0: overlay.WHITE, 2: overlay.DARK}
        if milkdrop:
            s.zoom = np.linspace(1, 2, n).astype(np.float32)
            s.white = np.clip(1 - np.abs(np.arange(n) - 30) / 8, 0, 1).astype(np.float32)   # a flash in the cut
        out.append(s)
    return out


@pytest.mark.parametrize("engine,layer", [("milkdrop", True), ("builtin", False), ("milkdrop", (160, 90))])
def test_gpu_frame_matches_numpy(engine, layer):
    n = 60
    frames, md, beat, G, F, S = scene(n, layer)
    opts = overlay.options({"engine": engine})
    preset = ([], "neon") if engine == "milkdrop" else (["tunnel", "kaleido", "plasma", "sparks"], "acid")
    a, b = shows(beat, opts, md is not None, n)
    comp = composite.Compositor(b, F, G, S, opts, torch.device("cpu"))
    worst = []
    for k in range(n):
        want = overlay.frame(frames[k], k, a, F, G, S, preset, opts, md[k] if md else None).astype(np.float32)
        f = torch.from_numpy(frames[k].astype(np.float32)).permute(2, 0, 1)
        m = torch.from_numpy(md[k].astype(np.float32)).permute(2, 0, 1) if md else None
        y, cb, cr = composite.rgb_to_yuv(f)                    # (chroma at full size: the planes carry all of it)
        y, uv = comp.frame(y, torch.stack([cb, cr]), k, preset, m)
        got = composite.yuv_to_rgb(y, uv[0], uv[1]).clamp(0, 255).permute(1, 2, 0).numpy()
        worst.append(float(np.abs(got - want).mean()))
        # numpy truncates and resamples in 8 bits; a layer smaller than the frame is keyed and blended at its own
        # size here and then scaled up, in numpy scaled up first: the same picture, a little apart on its hard edges
        assert np.percentile(np.abs(got - want), 99) < (4 if layer in (True, False) else 25), (k, worst[-1])
    assert max(worst) < 1.0, max(worst)
    assert a.weight == b.weight and abs((a.cover or 0) - (b.cover or 0)) < 0.01   # the shows' state stays in step


def test_yuv_roundtrip_and_matrix():
    for rgb, yuv in (((255, 0, 0), (63, 102, 240)), ((0, 255, 0), (173, 42, 26)), ((0, 0, 255), (32, 240, 118)),
                     ((255, 255, 255), (235, 128, 128)), ((0, 0, 0), (16, 128, 128))):       # BT.709, limited range
        x = torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1)
        got = [round(float(p)) for p in composite.rgb_to_yuv(x)]
        assert got == list(yuv), (rgb, got)
        back = composite.yuv_to_rgb(*(torch.tensor(float(v)).view(1, 1) for v in yuv)).flatten()
        assert torch.allclose(back, torch.tensor(rgb, dtype=torch.float32), atol=1.5)


def test_raw_frames_in_and_out():
    frames, _, beat, G, F, S = scene(1, False)
    opts = overlay.options({"engine": "builtin"})
    show = shows(beat, opts, False, 1)[0]
    comp = composite.Compositor(show, F, G, S, opts, torch.device("cpu"))
    x = torch.from_numpy(frames[0].astype(np.float32)).permute(2, 0, 1)
    y, cb, cr = composite.rgb_to_yuv(x)
    raw = comp.to_yuv420(y, torch.nn.functional.avg_pool2d(torch.stack([cb, cr])[None], 2)[0])
    assert raw.dtype == np.uint8 and raw.size == W * H * 3 // 2
    y2, uv2 = comp.from_yuv420(bytearray(raw.tobytes()))
    assert y2.shape == (H, W) and uv2.shape == (2, H // 2, W // 2)
    assert float((y2 - y).abs().max()) <= 1.0                  # rounding and the dither: a code value at most
    again = comp.to_yuv420(y2, uv2)                            # whole code values pass the dither untouched
    assert np.array_equal(again, raw)
    flat = torch.full((H, W), 100.3)                           # dithered: the mean survives the rounding
    out = comp.to_yuv420(flat, uv2)[:W * H].astype(np.float32)
    assert abs(out.mean() - 100.3) < 0.02 and len(np.unique(out)) == 2


def test_untouched_frame_passes_through():
    """No MilkDrop, off the beat, no drop: the planes come back as they went in (no RGB round trip, no resample)."""
    _, _, beat, G, F, S = scene(40, False)
    G["drop"][:] = -1
    opts = overlay.options({"engine": "builtin", "presets": [[[], "neon"]], "flicker": False})
    show = shows(beat, opts, False, 40)[0]
    show.flick = {}
    comp = composite.Compositor(show, F, G, S, opts, torch.device("cpu"))
    k = 39                                                     # late in a beat: the kick's zoom and blink are gone
    assert G["since"][k] > 0.35
    y = torch.rand(H, W) * 200 + 20
    uv = torch.rand(2, H // 2, W // 2) * 200 + 20
    y2, uv2 = comp.frame(y, uv, k, ([], "neon"))
    assert float((y2 - y).abs().max()) < 0.5 and float((uv2 - uv).abs().max()) < 0.5


def test_device_off(monkeypatch):
    monkeypatch.setenv("FOCALCUT_OVERLAY", "cpu")
    assert composite.device() is None
