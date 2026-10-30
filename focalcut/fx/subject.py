"""Subject effects driven by SAM 3 performer masks: spotlight, freeze, neon outline, particles, velocity lines, echo."""
import colorsys

import cv2
import numpy as np

from .base import Effect, dim, register, to_u8


def edge(m, width=3):
    hard = (m > 0.5).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(hard, k, iterations=width) - cv2.erode(hard, k, iterations=1)).astype(np.float32)


@register
class Spotlight(Effect):
    """Background desaturated and dimmed, performers in full colour; eases in over the first beat."""
    name = "spotlight"; needs_masks = True

    def apply(self, frames, ctx):
        masks, ramp, out = ctx.masks(frames), max(1, int(ctx.beat * ctx.fps)), []
        for i, (f, m) in enumerate(zip(frames, masks)):
            f = f.astype(np.float32); k = min(1.0, i / ramp); a = m[..., None]
            out.append(to_u8(f * a + (f * (1 - k) + dim(f) * k) * (1 - a)))
        return out


@register
class Freeze(Effect):
    """Play to the peak, hold it for one beat with the performers cut out and outlined over a dim background
    (gentle push-in), then carry on from the peak. Output length is unchanged."""
    name = "freeze"; needs_masks = True

    def apply(self, frames, ctx):
        masks = ctx.masks(frames)
        n, H, W = len(frames), *frames[0].shape[:2]
        p, hold = ctx.peak_frame(n), int(round(ctx.beat * ctx.fps))
        f, m = frames[p].astype(np.float32), masks[p]
        ring = cv2.GaussianBlur(edge(m, 4), (0, 0), 1.5)[..., None]
        held = []
        for j in range(hold):
            g = f * m[..., None] + dim(f, 0.35, 0.15) * (1 - m[..., None])
            g = g * (1 - ring) + 255 * ring
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + 0.045 * j / max(hold - 1, 1))
            held.append(cv2.warpAffine(to_u8(g), M, (W, H), borderMode=cv2.BORDER_REFLECT))
        return (list(frames[:p]) + held + list(frames[p:p + n - p - hold]))[:n]


@register
class Neon(Effect):
    """Glowing outline tracing the performers, hue drifting over time; background slightly dimmed."""
    name = "neon"; needs_masks = True

    def apply(self, frames, ctx):
        masks, out = ctx.masks(frames), []
        for i, (f, m) in enumerate(zip(frames, masks)):
            e = edge(m, 2)
            r, g, b = colorsys.hsv_to_rgb((0.52 + i / (ctx.fps * 3)) % 1, 0.85, 1.0)
            col = np.array([r, g, b], np.float32) * 255
            glow = cv2.GaussianBlur(e, (0, 0), 6)[..., None] * 1.6 + cv2.GaussianBlur(e, (0, 0), 2)[..., None]
            base = f.astype(np.float32) * (0.78 + 0.22 * m[..., None])
            out.append(to_u8(base + glow * col * 0.9 + e[..., None] * 255 * 0.6))
        return out


@register
class Particles(Effect):
    """Sparks thrown off the performers' outline, busiest around the peak."""
    name = "particles"; needs_masks = True

    def apply(self, frames, ctx):
        masks = ctx.masks(frames)
        n, H, W = len(frames), *frames[0].shape[:2]
        p = ctx.peak_frame(n)
        parts, out = [], []                        # [x, y, vx, vy, life]
        for i, (f, m) in enumerate(zip(frames, masks)):
            ys, xs = np.nonzero(edge(m, 1))
            rate = int(40 * np.exp(-0.5 * ((i - p) / (0.6 * ctx.beat * ctx.fps)) ** 2)) + 2
            if len(xs):
                for j in ctx.rng.integers(0, len(xs), rate):
                    a = ctx.rng.uniform(0, 2 * np.pi); sp = ctx.rng.uniform(2, 9) * W / 1080
                    parts.append([xs[j], ys[j], np.cos(a) * sp, np.sin(a) * sp - 2, ctx.rng.uniform(0.3, 0.8) * ctx.fps])
            layer = np.zeros((H, W), np.float32)
            alive = []
            for q in parts:
                q[0] += q[2]; q[1] += q[3]; q[3] += 0.25; q[4] -= 1
                if q[4] > 0 and 0 <= q[0] < W and 0 <= q[1] < H:
                    cv2.circle(layer, (int(q[0]), int(q[1])), max(1, W // 540), min(1.0, q[4] / 10), -1)
                    alive.append(q)
            parts = alive
            glow = (cv2.GaussianBlur(layer, (0, 0), 3) * 2 + layer)[..., None]
            out.append(to_u8(f.astype(np.float32) + glow * np.array([255, 220, 150], np.float32)))
        return out


@register
class Velocity(Effect):
    """Light streaks trailing the performers: their outline smeared back along their motion."""
    name = "velocity"; needs_masks = True

    def apply(self, frames, ctx):
        masks, out, trail = ctx.masks(frames), [], None
        for f, m in zip(frames, masks):
            e = cv2.GaussianBlur(edge(m, 2), (0, 0), 1.5)
            trail = e if trail is None else np.maximum(trail * 0.82, e)
            streak = cv2.GaussianBlur(trail, (0, 0), 4)[..., None]
            out.append(to_u8(f.astype(np.float32) + streak * np.array([170, 210, 255], np.float32) * 1.4))
        return out


@register
class Echo(Effect):
    """Afterimage: fading copies of the performers from the previous frames."""
    name = "echo"; needs_masks = True

    def apply(self, frames, ctx):
        masks, out = ctx.masks(frames), []
        lag = max(1, int(0.08 * ctx.fps))
        for i, f in enumerate(frames):
            acc = f.astype(np.float32)
            for k, a in ((3, 0.18), (2, 0.28), (1, 0.4)):
                j = i - k * lag
                if j >= 0:
                    m = masks[j][..., None] * a * (1 - masks[i][..., None])
                    acc = acc * (1 - m) + frames[j].astype(np.float32) * m
            out.append(to_u8(acc))
        return out


@register
class Strobe(Effect):
    """Stage strobe on the beat: performers flash bright, the background drops to near-black between flashes."""
    name = "strobe"; needs_masks = True

    def apply(self, frames, ctx):
        masks, out = ctx.masks(frames), []
        per = max(2, int(ctx.beat * ctx.fps / self.p.get("per_beat", 2)))
        for i, (f, m) in enumerate(zip(frames, masks)):
            on = (i % per) < max(1, per // 3)
            f = f.astype(np.float32); a = m[..., None]
            subj = f * (1.45 if on else 1.0) + (40 if on else 0)
            bg = f * (0.55 if on else 0.12)
            out.append(to_u8(subj * a + bg * (1 - a)))
        return out


@register
class Orbs(Effect):
    """Glowing orbs circling the performers (orbit follows the mask's centre and size)."""
    name = "orbs"; needs_masks = True

    def apply(self, frames, ctx):
        masks, out = ctx.masks(frames), []
        H, W = frames[0].shape[:2]
        n_orbs = self.p.get("count", 5)
        hues = [(0.55 + k / n_orbs * 0.5) % 1 for k in range(n_orbs)]
        cx = cy = rx = ry = None
        for i, (f, m) in enumerate(zip(frames, masks)):
            ys, xs = np.nonzero(m > 0.5)
            if len(xs):                                   # smooth the orbit's centre and size
                c = (xs.mean(), ys.mean(), (xs.max() - xs.min()) * 0.65 + W * 0.04, (ys.max() - ys.min()) * 0.35)
                cx, cy, rx, ry = c if cx is None else [0.85 * o + 0.15 * v for o, v in zip((cx, cy, rx, ry), c)]
            layer = np.zeros((H, W, 3), np.float32)
            if cx is not None:
                for k in range(n_orbs):
                    ang = 2 * np.pi * (k / n_orbs + i / (ctx.fps * 1.6))
                    x, y = cx + rx * np.cos(ang), cy + ry * np.sin(ang) * 0.6 - ry * 0.2
                    front = np.sin(ang) > 0                 # orbs pass behind the dancers on the far side
                    r, g, b = colorsys.hsv_to_rgb(hues[k], 0.7, 1.0)
                    dot = np.zeros((H, W), np.float32)
                    cv2.circle(dot, (int(x), int(y)), max(3, W // 150), 1.0, -1)
                    glow = cv2.GaussianBlur(dot, (0, 0), W / 90) * 3 + dot
                    if not front:
                        glow *= (1 - m)
                    layer += glow[..., None] * np.array([r, g, b], np.float32) * 255
            out.append(to_u8(f.astype(np.float32) * 0.9 + layer))
        return out


@register
class Aura(Effect):
    """Coloured fire around the silhouette: flame tongues licking upward off the outline (scrolling noise shaped by a
    halo that is pushed upward), with a hot core and a saturated edge."""
    name = "aura"; needs_masks = True

    def apply(self, frames, ctx):
        masks, out = ctx.masks(frames), []
        H, W = frames[0].shape[:2]
        hue = self.p.get("hue", 0.58)                     # 0.58 blue-violet; 0.05 orange fire; 0.85 magenta
        cell = max(4, W // 90)
        seed = ctx.rng.random((H // cell + 4, W // cell + 4)).astype(np.float32)
        seed2 = ctx.rng.random((H // (cell * 2) + 4, W // (cell * 2) + 4)).astype(np.float32)
        lift = H // 40                                    # flames rise above the body
        for i, (f, m) in enumerate(zip(frames, masks)):
            hard = (m > 0.4).astype(np.uint8)
            grow = cv2.dilate(hard, np.ones((3, 3), np.uint8), iterations=max(4, W // 80)).astype(np.float32)
            halo = cv2.GaussianBlur(grow, (0, 0), W / 140)
            halo = np.concatenate([halo[lift:], np.zeros((lift, W), np.float32)], 0) * (1 - m)   # push up, no wrap
            nz = 0.65 * cv2.resize(np.roll(seed, -(i * 2) % seed.shape[0], 0), (W, H), interpolation=cv2.INTER_CUBIC) \
                + 0.35 * cv2.resize(np.roll(seed2, -(i * 3) % seed2.shape[0], 0), (W, H), interpolation=cv2.INTER_CUBIC)
            nz = cv2.GaussianBlur(nz, (0, 0), sigmaX=W / 500, sigmaY=W / 90)          # tall, thin tongues
            flame = np.clip((halo * (0.1 + 1.5 * nz) - 0.5) * 2.0, 0, 1)
            flame = cv2.GaussianBlur(flame, (0, 0), 1.5)
            core = np.array(colorsys.hsv_to_rgb(hue, 0.55, 1.0), np.float32)
            rim = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0), np.float32)
            col = (rim * (1 - flame[..., None]) + core * flame[..., None]) * 255
            glow = cv2.GaussianBlur(flame, (0, 0), W / 70)[..., None] * rim * 255 * 0.5
            out.append(to_u8(f.astype(np.float32) * (0.85 + 0.15 * m[..., None]) + flame[..., None] * col * 0.95 + glow * 0.7))
        return out
