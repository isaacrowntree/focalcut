"""The overlay's compositor on the GPU (torch): overlay.frame's arithmetic, frame for frame, at any output size.

overlay.frame works in numpy on one CPU core: about 4 frames a second at 1080p, a quarter of that at 4K -- most of
a render's time. Here the same sums run on the GPU, arranged so a frame costs a few passes over it:

  - The footage stays in the Y'CbCr 4:2:0 it is decoded to and encoded from (BT.709, limited range). Nearly all
    of the show is `footage * a + b` per pixel (the vignette, MilkDrop over it, blinks, flickers, a cut's flash to
    white), and that carries straight over to Y'CbCr, so there is no conversion to RGB and back: the clear centre
    of the frame passes through untouched, and the chroma is not resampled a second time.
  - `a` and `b` are built at the MilkDrop layer's own size (1080p even in a 4K reel: its presets are drawn for
    that canvas) and scaled up once, instead of every step running at the output's size.
  - Only frames that fold or recolour the footage itself (the RGB split on a bar, a drop's kaleidoscope, duotone
    and light bursts) go through RGB, as a few frames in a hundred do.
  - The beat's zoom punch resamples the picture bicubically (overlay.frame: bilinear), so those frames stay crisp.
  - The last step dithers before rounding to 8 bits, so the synthetic gradients (the vignette, glows, MilkDrop's
    fields) come out as fine grain, not bands.

The small stateful effects (confetti, sparks, the built-in presets) are still drawn by overlay.Show on the CPU at
their own size and blended here. Used when torch has a CUDA device (FOCALCUT_OVERLAY=cpu forces overlay.frame). The
two agree to within rounding; tests/test_composite.py holds them together.
"""
from __future__ import annotations

import math
import os

import numpy as np

KR, KB = 0.2126, 0.0722                             # BT.709
KG = 1 - KR - KB
SY, SC = 219 / 255, 224 / 255                       # limited range: luma and chroma excursions per unit of RGB


def device():
    """The torch device to composite on, or None: no torch, no CUDA, or FOCALCUT_OVERLAY=cpu."""
    if os.environ.get("FOCALCUT_OVERLAY", "").lower() in ("cpu", "numpy"):
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def yuv_to_rgb(y, u, v):
    """Limited-range BT.709 planes (float, 8-bit code values; chroma at luma size) -> RGB 0..255, (3, H, W)."""
    import torch
    y = (y - 16) / SY
    cb, cr = (u - 128) / SC, (v - 128) / SC
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return torch.stack([r, g, b])


def rgb_to_yuv(x, offset=True):
    """RGB 0..255 (3, H, W) -> limited-range BT.709 planes as float code values (Y, Cb, Cr), chroma at luma size.
    offset=False: the linear part alone (what an RGB amount adds to each plane)."""
    r, g, b = x[0], x[1], x[2]
    y = KR * r + KG * g + KB * b
    cb, cr = (b - y) / (2 * (1 - KB)), (r - y) / (2 * (1 - KR))
    o = (16, 128) if offset else (0, 0)
    return o[0] + y * SY, o[1] + cb * SC, o[1] + cr * SC


class Compositor:
    """overlay.frame on a torch device. show, F, G, S and opts are overlay.apply's. A frame is its luma plane
    (H, W) and its chroma planes (2, h, w) -- half size for 4:2:0, any size works -- as float code values; the
    MilkDrop layer is float RGB (3, h, w) at its own size."""

    def __init__(self, show, F, G, S, opts, dev):
        import torch
        self.t, self.dev = torch, dev
        self.show, self.F, self.G, self.S, self.opts = show, F, G, S, opts
        W, H = show.W, show.H
        self.W, self.H = W, H
        self.E = torch.from_numpy(np.ascontiguousarray(show.edge[..., 0])).to(dev)        # (H, W)
        self.edges = {}
        g = torch.exp(-torch.arange(-4, 5, dtype=torch.float32, device=dev) ** 2 / 2)     # cv2's sigma 1 kernel
        self.gk = (g / g.sum())
        self.grids, self.polar, self.ramps = {}, None, {}
        self.noise, self.out, self.turn = None, None, 0              # the dither's noise (to_yuv420)

    # pieces --------------------------------------------------------------------------------------------------
    def grid(self, h, w):
        """The identity sampling grid of an (h, w) plane: pixel centres in -1..1, (1, h, w, 2)."""
        if (h, w) not in self.grids:
            torch = self.t
            ny = (torch.arange(h, device=self.dev, dtype=torch.float32) * 2 + 1) / h - 1
            nx = (torch.arange(w, device=self.dev, dtype=torch.float32) * 2 + 1) / w - 1
            self.grids[(h, w)] = torch.stack([nx[None, :].expand(h, w), ny[:, None].expand(h, w)], -1)[None]
        return self.grids[(h, w)]

    def edge(self, h, w):
        """show.edge (where the effects may show) at another size."""
        if (h, w) == (self.H, self.W):
            return self.E
        if (h, w) not in self.edges:
            if len(self.edges) > 64:                                 # (a zoom easing: one size per frame)
                self.edges.clear()
            self.edges[(h, w)] = self.fit(self.E[None], h, w)[0]
        return self.edges[(h, w)]

    def zoom(self, a, z, mode="bilinear"):
        """overlay.zoom of planes (C, h, w): scaled by z about the centre (z >= 1 never reads past the edge)."""
        h, w = a.shape[-2:]
        grid = self.grid(h, w) / z + self.t.tensor([(1 - 1 / z) / w, (1 - 1 / z) / h], device=self.dev)
        return self.t.nn.functional.grid_sample(a[None], grid, mode=mode, padding_mode="reflection",
                                                align_corners=False)[0]

    def fit(self, a, h, w, mode="bilinear"):
        """(C, ., .) -> (C, h, w): `mode` going up, averaged going down."""
        ah, aw = a.shape[-2:]
        if (ah, aw) == (h, w):
            return a
        Fn = self.t.nn.functional
        if h < ah or w < aw:
            if ah == 2 * h and aw == 2 * w:
                return Fn.avg_pool2d(a[None], 2)[0]
            return Fn.interpolate(a[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0]
        return Fn.interpolate(a[None], size=(h, w), mode=mode, align_corners=False)[0]

    def kaleido(self, f, t, n=8):
        """Show.kaleido, sampled from the full frame at full size (the CPU one folds a small copy)."""
        torch = self.t
        H, W = f.shape[-2:]
        if self.polar is None:
            g = self.grid(H, W)[0]
            dx, dy = g[..., 0] * (W / 2), g[..., 1] * (H / 2)
            self.polar = torch.hypot(dx, dy), torch.atan2(dy, dx)
        r, th = self.polar
        wedge = 2 * math.pi / n
        a = torch.remainder(th + t * 0.35, wedge)
        a = torch.where(a > wedge / 2, wedge - a, a) - math.pi / 2
        r = r * 0.8
        grid = torch.stack([r * torch.cos(a) / (W / 2), r * torch.sin(a) / (H / 2)], -1)[None]
        return torch.nn.functional.grid_sample(f[None], grid, mode="bilinear", padding_mode="reflection",
                                               align_corners=False)[0]

    def duotone(self, f):
        """Show.duotone: the frame's luma through the palette's ramp."""
        torch = self.t
        name = self.show.pal.name
        if name not in self.ramps:
            self.ramps[name] = torch.from_numpy(self.show.ramp).to(self.dev)              # (256, 3)
        g = (0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]).round().clamp(0, 255).long()
        return self.ramps[name][g].permute(2, 0, 1)

    def key(self, md):
        """overlay.key at the layer's own size: (h, w)."""
        torch = self.t
        m = md / 255
        lum = m.mean(0)
        sat = m.amax(0) - m.amin(0)
        a = ((lum - 0.2) / 0.35).clamp(0, 1) ** 1.5 * (0.3 + 0.7 * ((sat - 0.12) / 0.35).clamp(0, 1))
        pad = torch.nn.functional.pad                            # cv2's blur, sigma 1 (9 taps each way; as sums of
        h, w = a.shape                                           # shifted copies: no cuDNN, which some installs break)
        p = pad(a[None, None], (4, 4, 0, 0), mode="reflect")[0, 0]
        a = sum(self.gk[i] * p[:, i:i + w] for i in range(9))
        p = pad(a[None, None], (0, 0, 4, 4), mode="reflect")[0, 0]
        return sum(self.gk[i] * p[i:i + h] for i in range(9))

    # the frame -------------------------------------------------------------------------------------------------
    def frame(self, y, uv, k, preset, md=None):
        """One output frame as (luma, chroma) planes: overlay.frame, step for step. Not clipped: to_yuv420 does."""
        from . import overlay as O
        torch = self.t
        show, F, G, S, opts = self.show, self.F, self.G, self.S, self.opts
        H, W = y.shape
        s, beat, fps = float(S[k]), show.beat, show.fps
        since, down, drop = float(G["since"][k]), bool(G["down"][k]), int(G["drop"][k])
        kick = float(np.exp(-since / (0.22 * beat)))
        bass = min(float(F["bass"][k]), 1.3)
        effects, pal = preset
        show.palette(pal)
        active = set(effects) | ({"confetti", "sparks"} if drop >= 0 or show.flakes else set())
        ease = 1 / max(1, beat * fps)
        for e in set(show.weight) | active:
            show.weight[e] = min(1.0, show.weight.get(e, 0) + ease) if e in active else \
                max(0.0, show.weight.get(e, 0) - ease)
        c = dict(t=k / fps, bass=bass, bands=F["bands"][k], kick=kick, since=since, down=down, drop=drop,
                 bar=int(G["bar"][k]), beat_now=since < 1 / fps, frame=None)
        B = opts["beat"]
        if "kick" in B:
            z = 1 + (0.035 if down else 0.018) * s * kick
            if (z - 1) * max(W, H) / 2 > 0.25:                       # (under a quarter of a pixel at the edge: as it is)
                y, uv = self.zoom(y[None], z, "bicubic")[0], self.zoom(uv, z)
        # what folds or recolours the footage itself: through RGB, on the few frames that have any --------------
        sh = int((4 - since * fps) * 4 * s * show.W / 1920) if "split" in B and down and since < 4 / fps else 0
        kw = show.weight.get("kaleido", 0)
        a = (0.08 + 0.25 * bass ** 2) * kw
        if down and kw:
            a = max(a, 0.7 * kw * float(np.exp(-since / (0.5 * beat))))
        if drop >= 0:
            a = max(a, (0.85 if drop == 0 else 0.55) * float(np.exp(-since / (0.6 * beat)))
                    * (0.35 if md is not None else 1))
        a *= s
        d = 0.0
        if drop >= 0:
            d = 0.6 * s * float(np.exp(-since / (0.8 * beat))) * (1.0 if drop == 0 else 0.6) * (0.3 if md is not None else 1)
        light = None
        for e in O.LIGHT:
            wgt = show.weight.get(e, 0)
            if wgt <= 0:
                continue
            img = getattr(show, e)(c) * wgt
            light = img if light is None else light + img
        if sh or a > 0.02 or d > 0.02 or light is not None:
            E = self.E
            c2 = self.fit(uv, H, W)
            f = yuv_to_rgb(y, c2[0], c2[1])
            if sh:
                f = torch.stack([torch.roll(f[0], sh, 1), f[1], torch.roll(f[2], -sh, 1)])
            x = f
            if a > 0.02:
                x = x * (1 - a * E) + self.kaleido(f, c["t"]) * (a * E)
            if d > 0.02:
                x = x * (1 - d * E) + self.duotone(f) * (d * E)
            if light is not None:
                dark = 1.15 - x.mean(0) / 255
                lt = torch.from_numpy(np.ascontiguousarray(np.clip(light, 0, 255), np.float32)).to(self.dev)
                lt = self.fit(lt.permute(2, 0, 1), H, W)
                x = 255 - (255 - x) * (255 - lt * (s * dark).clamp(max=1.0) * E) / 255
            y, cb, cr = rgb_to_yuv(x)
            uv = self.fit(torch.stack([cb, cr]), *uv.shape[-2:])
        # everything else is footage * ka + kb: built at the MilkDrop layer's size, in RGB ----------------------
        ka, kb = 1.0, 0.0                                            # (scalars until MilkDrop makes them pictures)
        if md is not None:
            z = float(show.zoom[k]) if show.zoom is not None else 1.0
            if z > 1.01:
                h, w = md.shape[-2:]
                cw, ch = int(w / z), int(h / z)
                md = md[:, (h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
            h, w = md.shape[-2:]
            if h > H or w > W:                                       # (a layer bigger than the frame: at the frame's)
                md, (h, w) = self.fit(md, H, W), (H, W)
            E = self.edge(h, w)
            k_ = self.key(md)
            st = max(1, round(8 * w / W))
            cov = float((k_[::st, ::st] * E[::st, ::st]).sum() / E[::st, ::st].sum())
            show.cover = cov if show.cover is None else show.cover + (cov - show.cover) * min(1.0, 1 / (beat * fps))
            bare = max(float(np.clip(1 - show.cover / O.EDGE_LIT, 0, 1)), show.bare[k] if show.bare is not None else 0.0)
            v = opts["vignette"] + (1 - opts["vignette"]) * opts["bare"] * bare
            ee = (E * (1 + 0.5 * opts["bare"] * bare)).clamp(max=1.0) if bare > 0.01 else E
            dk = (1 - v * ee) ** 2
            wh = float(show.white[k]) if show.white is not None else 0.0
            g = min(1.0, s * (0.75 + 0.25 * kick))
            am = (k_ + (1 - k_) * E * opts["edge_fill"]) * E * (g + (1 - g) * E)
            am = 1 - (1 - am) ** 2
            mdu = md * (k_ + (1 - k_) * O.DULL)
            grey = mdu.mean(0, keepdim=True)
            mdu = (grey + (mdu - grey) * 1.4).clamp(0, 255) * 1.1
            if wh > 0:
                mdu = mdu * (1 - wh) + 255 * wh
            ka = dk * (1 - am)                                       # (h, w)
            kb = mdu * am                                            # (3, h, w)
            if wh > 0:
                kb = kb + (255 * wh) * ((1 - dk) * (1 - am))
        blink = (0.14 if down else 0.07) * s * kick if "kick" in B else 0.0
        if "drops" in B and drop >= 0:
            blink = max(blink, (0.6 if drop == 0 else 0.25) * float(np.exp(-since / (0.12 * beat))))
        if blink:
            ka, kb = ka * (1 - blink), kb * (1 - blink) + 255 * blink
        fl = show.flick.get(int(G["k"][k])) if show.flick else None
        if fl:
            hold = 2 / fps                                           # (overlay.flicker)
            fa = 0.92 * (1.0 if since < hold else float(np.exp(-(since - hold) / (0.07 * beat))))
            if fa >= 0.01:
                ka, kb = ka * (1 - fa), kb * (1 - fa) + (255 * fa if fl == O.WHITE else 0.0)
        # ... and applied to the planes ---------------------------------------------------------------------------
        if not torch.is_tensor(ka):                                  # no MilkDrop: one gain and one grey level
            if ka != 1.0 or kb != 0.0:
                y, uv = 16 + (y - 16) * ka + kb * SY, 128 + (uv - 128) * ka
            return y, uv
        if not torch.is_tensor(kb):
            kb = torch.full((3, 1, 1), float(kb), device=self.dev).expand(3, *ka.shape)
        by, bcb, bcr = rgb_to_yuv(kb, offset=False)
        mode = show.md_filter
        ay = self.fit(torch.stack([ka, by]), H, W, mode)
        y = torch.addcmul(ay[1].add_(16), y - 16, ay[0].clamp_(0, 1))
        ac = self.fit(torch.stack([ka, bcb, bcr]), *uv.shape[-2:], mode)
        uv = torch.addcmul(ac[1:] + 128, uv - 128, ac[:1].clamp(0, 1))
        return y, uv

    # raw 4:2:0 in and out ----------------------------------------------------------------------------------------
    def from_yuv420(self, buf, W=None, H=None):
        """A raw yuv420p frame (a writable buffer) -> its luma (H, W) and chroma (2, H/2, W/2) as floats."""
        torch = self.t
        W, H = W or self.W, H or self.H
        a = torch.frombuffer(buf, dtype=torch.uint8).to(self.dev).float()
        return a[:W * H].view(H, W), a[W * H:].view(2, H // 2, W // 2)

    def from_yuv444(self, buf, w, h):
        """A raw yuv444p frame (the MilkDrop layer's file) -> float RGB (3, h, w)."""
        torch = self.t
        a = torch.frombuffer(buf, dtype=torch.uint8).to(self.dev).float().view(3, h, w)
        return yuv_to_rgb(a[0], a[1], a[2])

    def to_yuv420(self, y, uv):
        """The planes as a raw yuv420p frame (uint8 array), dithered by half a code value of noise before rounding:
        a plane the show left alone comes out as it went in, a smooth synthetic gradient as fine grain, not bands."""
        torch = self.t
        n = y.numel() + uv.numel()
        if self.noise is None or self.noise.shape[1] != n:           # a few frames of noise, made once: each frame
            self.noise = (torch.rand(8, n, device=self.dev) - 0.5) * 0.996   # takes one, rolled by a random offset
            self.out = torch.empty(n, device=self.dev)               # (a hair under a whole step: a whole value
        self.turn += 1                                               # can never tip over)
        d = torch.roll(self.noise[self.turn % 8], int(torch.randint(n, (1,))))
        o = self.out
        torch.add(y.reshape(-1), d[:y.numel()], out=o[:y.numel()])
        torch.add(uv.reshape(-1), d[y.numel():], out=o[y.numel():])
        o.round_()
        o[:y.numel()].clamp_(16, 235)
        o[y.numel():].clamp_(16, 240)
        return o.to(torch.uint8).cpu().numpy()
