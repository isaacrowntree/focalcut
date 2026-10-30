"""Camera-style effects locked to the beat: punch-in zooms, shake, RGB-split glitch."""
import cv2
import numpy as np

from .base import Effect, register


def zoom(f, z, cx=0.5, cy=0.5):
    H, W = f.shape[:2]
    M = cv2.getRotationMatrix2D((W * cx, H * cy), 0, z)
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@register
class PunchIn(Effect):
    """A zoom kick on every beat of the shot (shots start on a beat), easing back within half a beat."""
    name = "punchin"

    def apply(self, frames, ctx):
        bf, out = ctx.beat * ctx.fps, []
        amt = self.p.get("amount", 0.07)
        for i, f in enumerate(frames):
            ph = (i % bf) / (0.5 * bf)
            out.append(zoom(f, 1 + amt * max(0.0, 1 - ph) ** 2))
        return out


@register
class Shake(Effect):
    """Handheld-style jolt at the peak, decaying over a beat."""
    name = "shake"

    def apply(self, frames, ctx):
        n, H, W = len(frames), *frames[0].shape[:2]
        p, out = ctx.peak_frame(n), []
        for i, f in enumerate(frames):
            k = np.exp(-max(0, i - p) / (0.35 * ctx.beat * ctx.fps)) if i >= p else 0
            dx, dy = ctx.rng.normal(0, 14 * k * W / 1080, 2)
            M = np.float32([[1.03, 0, dx - 0.015 * W], [0, 1.03, dy - 0.015 * H]])
            out.append(cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT))
        return out


def rgb_split(f, k, rng):
    H, W = f.shape[:2]
    s = int(k * 18 * W / 1080)
    g = f.copy()
    g[..., 0] = np.roll(f[..., 0], s, 1); g[..., 2] = np.roll(f[..., 2], -s, 1)
    for _ in range(int(k * 6)):                   # a few displaced horizontal slices
        y = rng.integers(0, H - 20); h = rng.integers(6, max(7, H // 25))
        g[y:y + h] = np.roll(g[y:y + h], rng.integers(-W // 12, W // 12), 1)
    return g


@register
class Glitch(Effect):
    """RGB split and slice displacement for a few frames around the peak."""
    name = "glitch"

    def apply(self, frames, ctx):
        n = len(frames); p = ctx.peak_frame(n); w = 0.25 * ctx.beat * ctx.fps
        return [rgb_split(f, np.exp(-0.5 * ((i - p) / w) ** 2), ctx.rng) if abs(i - p) < 3 * w else f
                for i, f in enumerate(frames)]
