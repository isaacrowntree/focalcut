"""Speed curves, after CapCut's Speed > Curve presets: montage, hero, bullet (ramp - hold - ramp), jumpcut, flashin,
flashout; and fit, which plays an editor's whole in-out in the slot it was given.

Each preset is a speed multiplier over output time, centred on the move's peak. Slow stretches get optical-flow
in-between frames (smooth slow-mo); fast stretches average the frames they skip (motion blur). The source window is
longer or shorter than the output, so the renderer asks source_length() / source_peak() before decoding.
"""
import cv2
import numpy as np

from .base import Effect, register


class Speed(Effect):
    """A velocity curve the way CapCut's custom curve and a time-remap ramp are built: fast through the move, easing
    down to arrive slow exactly as the dancer lands (the shot's peak: frame dwell detected by the planner), holding
    there over the rest so the hair and the face play out, then easing back up to speed. Dancing never runs faster than
    CAP (any faster and people look sped up, not energetic); a close-up holds deeper and longer -- a hair flip in
    near-still motion."""
    fast, slow, hold, ease = 1.0, 0.35, 1.0, 0.4    # speed through the move, during the hold; hold and ease in beats
    CAP = 1.5

    def curve(self, t, ctx):
        p = ctx.peak if ctx.peak is not None else 0.0
        b = ctx.beat
        close = ctx.params.get("closeup")
        slow = self.slow * (0.6 if close else 1.0)
        hold = (self.hold + (0.5 if close else 0.0)) * b
        ease = self.ease * b
        fast = min(self.fast, ctx.params.get("cap", self.CAP))
        def smooth(x):                                                        # ease in and out, no corners
            return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
        down = smooth((t - (p - ease)) / ease)                               # 0 before the ease, 1 at the landing
        up = smooth((t - (p + hold)) / ease)                                 # 0 during the hold, 1 after it
        return fast + (slow - fast) * down * (1 - up)

    def speed(self, t, out_len, ctx):
        return np.minimum(self.curve(t, ctx), ctx.params.get("cap", self.CAP))

    def _map(self, out_len, ctx):
        t = np.arange(int(round(out_len * ctx.fps)) + 1) / ctx.fps
        s = self.speed(t, out_len, ctx)
        src = np.concatenate([[0], np.cumsum((s[1:] + s[:-1]) / 2 / ctx.fps)])
        return t, src, s

    def source_length(self, out_len, ctx):
        return float(self._map(out_len, ctx)[1][-1]) + 2 / ctx.fps

    def source_peak(self, out_peak, ctx, out_len=None):
        """Source time under output time out_peak (out_len: the shot's length, for curves that depend on it)."""
        t, src, _ = self._map(out_len or max(out_peak * 2, 1.0), ctx)
        return float(np.interp(out_peak, t, src))

    def out_time(self, src_time, out_len, ctx):
        """The inverse: the output time at which source time src_time plays."""
        t, src, _ = self._map(out_len, ctx)
        return float(np.interp(src_time, src, t))

    def apply(self, frames, ctx):
        n_out = len(frames) if not hasattr(ctx, "out_frames") else ctx.out_frames
        t, src, s = self._map(n_out / ctx.fps, ctx)
        idx = src * ctx.fps
        out, flow_cache = [], {}
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        last = len(frames) - 1
        for k in range(n_out):
            x = min(idx[k], last)
            if s[k] > 1.3 and k > 0:              # fast: blend the frames this output frame spans (motion blur)
                a, b = int(idx[k - 1]), int(min(x, last))
                out.append(np.mean(np.stack(frames[a:b + 1]).astype(np.float32), 0).astype(np.uint8)
                           if b > a else frames[b])
                continue
            i, a = int(x), np.float32(x - int(x))          # float32 keeps the remap maps float32
            if a < 0.05 or i >= last:
                out.append(frames[i]); continue
            if i not in flow_cache:                # slow: flow-warp both neighbours to the in-between time
                g0 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY); g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                flow_cache = {i: (dis.calc(g0, g1, None), dis.calc(g1, g0, None))}
            f01, f10 = flow_cache[i]
            h, w = f01.shape[:2]
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            w0 = cv2.remap(frames[i], gx + f10[..., 0] * a, gy + f10[..., 1] * a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(frames[i + 1], gx + f01[..., 0] * (1 - a), gy + f01[..., 1] * (1 - a), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            out.append(cv2.addWeighted(w0, 1 - a, w1, a, 0))
        return out


@register
class Montage(Speed):
    name = "montage"; fast, slow, hold, ease = 1.5, 0.5, 0.75, 0.25      # quick in, a short hold, quick out


@register
class Hero(Speed):
    name = "hero"; fast, slow, hold, ease = 1.4, 0.3, 1.0, 0.4           # a move held on its landing


@register
class Bullet(Speed):
    name = "bullet"; fast, slow, hold, ease = 1.5, 0.15, 2.0, 0.6        # the reel's peak: a long, deep hold


@register
class JumpCut(Speed):
    name = "jumpcut"

    def speed(self, t, out_len, ctx):                         # a quick speed-up just before the peak
        p = ctx.peak if ctx.peak is not None else out_len / 2
        return np.where((t > p - 0.6 * ctx.beat) & (t < p - 0.1 * ctx.beat), ctx.params.get("cap", 3.5), 1.0)


@register
class FlashIn(Speed):
    name = "flashin"

    def speed(self, t, out_len, ctx):                         # rushes in fast, settles to normal
        return np.minimum(1 + 2.5 * np.exp(-t / (0.6 * ctx.beat)), ctx.params.get("cap", 3.5))


@register
class FlashOut(Speed):
    name = "flashout"

    def speed(self, t, out_len, ctx):                         # normal, then rushes out
        return np.minimum(1 + 2.5 * np.exp(-(out_len - t) / (0.6 * ctx.beat)), ctx.params.get("cap", 3.5))


@register
class Fit(Speed):
    """The editor's whole selection in its slot: quicker at the ends, easing to its slowest over the middle (a bell
    across the shot), scaled so the selection's every frame plays -- params "src_len": the selection's length."""
    name = "fit"
    dip, width = 0.5, 0.35                                    # how much slower the middle is; the bell's width (of L)

    def shape(self, t, out_len):
        return 1 - self.dip * np.exp(-((t - out_len / 2) / (self.width * out_len)) ** 2)

    def speed(self, t, out_len, ctx):
        dense = np.linspace(0, out_len, 2001)
        area = np.trapezoid(self.shape(dense, out_len), dense)
        return self.shape(t, out_len) * ctx.params.get("src_len", out_len) / max(area, 1e-6)

    def source_length(self, out_len, ctx):
        return ctx.params.get("src_len", out_len) + 2 / ctx.fps
