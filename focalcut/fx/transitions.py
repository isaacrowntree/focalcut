"""Transitions across a cut: flash, fade, dipwhite, fadein, zoom, swipe (whip), spin, lightleak, glitch.

apply(a_tail, b_head, ctx) gets the last n/2 frames of shot A and the first n/2 of shot B and returns n frames that
replace them, so the cut stays on its beat and the reel's length doesn't change.
"""
import cv2
import numpy as np

from .base import Transition, register, to_u8
from .camera import rgb_split, zoom


def ease(u):
    return u * u * (3 - 2 * u)


@register
class Flash(Transition):
    name = "flash"; duration_beats = 0.5

    @staticmethod
    def white(i, n):
        """How white frame i of the n is (0..1): the overlay floods its own layers by the same amount."""
        return (1 - abs(i - n / 2) / (n / 2)) ** 1.5            # 0 -> 1 at the cut -> 0

    def apply(self, a, b, ctx):
        n = len(a) + len(b); out = []
        for i, f in enumerate(list(a) + list(b)):
            k = self.white(i, n)
            out.append(to_u8(f.astype(np.float32) * (1 - k) + 255 * k))
        return out


@register
class Fade(Transition):
    name = "fade"; duration_beats = 1.0

    def apply(self, a, b, ctx):
        n = len(a) + len(b); out = []
        for i in range(n):
            u = ease(i / (n - 1))
            fa, fb = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            out.append(cv2.addWeighted(fa, 1 - u, fb, u, 0))
        return out


def radial_blur(f, k):
    """f smeared outwards from its centre over 8% of its size at k = 1: copies a couple of pixels apart at the
    frame's edge (a handful of copies shows as ghost outlines, more so on a 4K frame), averaged."""
    reach = k * 0.08
    taps = int(np.clip(np.ceil(reach * max(f.shape[:2]) / 2 / 2.5), 4, 48))
    acc = f.astype(np.float32)
    for j in range(1, taps + 1):
        acc += zoom(f, 1 + reach * j / taps)
    return to_u8(acc / (taps + 1))


@register
class Zoom(Transition):
    name = "zoom"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        out = []
        for i, f in enumerate(a):                                  # A pushes in and blurs
            u = (i + 1) / len(a)
            out.append(radial_blur(zoom(f, 1 + 0.35 * u ** 2), u))
        for i, f in enumerate(b):                                  # B lands from a zoom, sharpening
            u = 1 - i / len(b)
            out.append(radial_blur(zoom(f, 1 + 0.35 * u ** 2), u))
        return out


def motion_dir(frames):
    """Mean motion across the last frames (dx, dy in pixels per frame, at quarter size): which way things move."""
    if len(frames) < 2:
        return 0.0, 0.0
    g = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)), cv2.COLOR_RGB2GRAY) for f in frames[-3:]]
    fl = np.mean([cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0) for a, b in zip(g, g[1:])], 0)
    return float(np.median(fl[..., 0])), float(np.median(fl[..., 1]))


@register
class Swipe(Transition):
    """Whip pan: A is flung out and B swings in the same way, motion blur strongest at the cut. The direction follows
    the movement at the end of A (the camera's pan, or the dancers travelling), so the whip continues the motion."""
    name = "swipe"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        n = len(a) + len(b); H, W = a[0].shape[:2]; out = []
        dx, dy = motion_dir(a)                              # mostly the background: the camera's pan
        horiz = abs(dx) >= abs(dy) or W > H
        v = dx if horiz else dy
        sgn = (1 if v > 0 else -1) if abs(v) >= 0.3 else (1 if ctx.rng.random() < 0.5 else -1)
        L, ax = (W, 1) if horiz else (H, 0)
        for i in range(n):
            u = ease(i / (n - 1)); shift = int(round(L * u))
            fa, fb = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            if sgn > 0:                                     # content moving right/down: A leaves that way
                canvas, o = np.concatenate([fb, fa], ax), L - shift
            else:
                canvas, o = np.concatenate([fa, fb], ax), shift
            frame = canvas[:, o:o + W] if horiz else canvas[o:o + H]
            k = max(1, int(110 * (1 - abs(2 * u - 1)) ** 1.5 * L / 1080))
            out.append(cv2.blur(frame, (k, 1) if horiz else (1, k)))
        return out


@register
class Spin(Transition):
    """Spin cut: A whirls with rotational blur into B, which comes out of the same rotation -- for cutting from one
    turn to the next."""
    name = "spin"; duration_beats = 0.67

    def apply(self, a, b, ctx):
        n = len(a) + len(b); H, W = a[0].shape[:2]; out = []
        sgn = 1 if ctx.rng.random() < 0.5 else -1
        for i in range(n):
            u = i / (n - 1)
            if i < len(a):
                f, ang = a[i], 90 * (u / 0.5) ** 2
            else:
                f = b[i - len(a)]; ang = -90 * ((1 - u) / 0.5) ** 2
            k = 1 - abs(2 * u - 1)                                   # 0 -> 1 at the cut -> 0
            z = 1 + 0.5 * k                                          # zoomed in enough to hide the corners
            acc = np.zeros((H, W, 3), np.float32)
            arc = np.radians(9 * k) * np.hypot(W, H) / 2             # the blur's sweep at the corners, in pixels
            steps = 1 + int(np.clip(np.ceil(arc / 3), 6 * k, 64))    # copies ~3 px apart there: a smear, no ghosts
            for j in range(steps):                                   # rotational motion blur
                M = cv2.getRotationMatrix2D((W / 2, H / 2), sgn * (ang - 9 * k * j / max(steps, 1)), z)
                acc += cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            out.append(to_u8(acc / steps))
        return out


@register
class LightLeak(Transition):
    """Warm light washes across the frame and dissolves A into B -- the soft change of scene."""
    name = "lightleak"; duration_beats = 2.0

    def apply(self, a, b, ctx):
        n = len(a) + len(b); H, W = a[0].shape[:2]; out = []
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = max(H, W)
        for i in range(n):
            u = i / (n - 1)
            fa, fb = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            mix = cv2.addWeighted(fa, 1 - ease(np.clip((u - 0.3) / 0.4, 0, 1)), fb, ease(np.clip((u - 0.3) / 0.4, 0, 1)), 0)
            cx, cy = W * (-0.2 + 1.4 * u), H * (0.3 + 0.2 * np.sin(u * 3))
            r = np.exp(-(((xx - cx) / (0.45 * d)) ** 2 + ((yy - cy) / (0.35 * d)) ** 2))
            k = np.sin(np.pi * u) ** 1.2
            leak = (r * k)[..., None] * np.array([255, 150, 60], np.float32)
            g = mix.astype(np.float32)
            out.append(to_u8(255 - (255 - g) * (255 - leak * 0.95) / 255))     # screen blend
        return out


@register
class DipWhite(Transition):
    """A slower flash: A burns out to white, B develops out of it."""
    name = "dipwhite"; duration_beats = 1.5

    @staticmethod
    def white(i, n):
        return ease(1 - abs(i - n / 2) / (n / 2))

    def apply(self, a, b, ctx):
        n = len(a) + len(b); out = []
        for i, f in enumerate(list(a) + list(b)):
            k = self.white(i, n)
            out.append(to_u8(f.astype(np.float32) * (1 - k) + 255 * k))
        return out


@register
class FadeIn(Transition):
    """From black: for the very first shot (A is ignored)."""
    name = "fadein"; duration_beats = 2.0

    def apply(self, a, b, ctx):
        n = len(a) + len(b); out = []
        for i, f in enumerate(list(a) + list(b)):
            out.append(to_u8(f.astype(np.float32) * ease(i / (n - 1))))
        return out


@register
class GlitchCut(Transition):
    name = "glitch"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        n = len(a) + len(b); out = []
        for i, f in enumerate(list(a) + list(b)):
            k = 1 - abs(i - n / 2) / (n / 2)
            out.append(rgb_split(f, k, ctx.rng))
        return out
