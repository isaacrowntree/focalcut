"""Carnival styles: a Latin street-party / tropical-disco family of looks for a whole film, as subject effects,
transitions and accents -- the colour, confetti, sequins and mirror-ball light of a carnival or a 90s merenhouse
video, where glitch.py is digital damage and edm.py is a main stage's flicker.

Subject (on the dancer only, through the SAM 3 masks; the room stays clean):
  c_glow       fiesta glow: a soft halo round the dancer, changing colour on every beat, swelling on the peak
  c_sparkle    sequins: star glints twinkle on the bright parts of the dancer, a shower of them on the peak
  c_confetti   confetti pop: a burst of tumbling paper confetti thrown up from the dancer on the peak, falling away
  c_streamers  serpentina: echoes of the dancer a few frames behind in flat carnival colours, under them
  c_pop        pop-art freeze: the peak holds a moment, the dancer cut out with a white sticker edge over a halftone
               room in two carnival colours, then it plays on
Transitions (apply(a_tail, b_head): the cut stays on its beat):
  c_whip       whip pan: A flies out sideways in a motion blur, B flies in from the other side
  c_burst      confetti cannon: two cannons fire across the frame and B is there when the paper clears
  c_flash      colour flash: a bloom into gold, magenta or turquoise (never white: that is the EDM look) and out on B
  c_stripes    serpentina wipe: slanted ribbons of carnival colour sweep across, B behind them
  c_sun        sun iris: a twelve-ray sun opens from the middle with a gold rim, B inside it
Accents (whole frame, at the shot's peak):
  c_zoom       zoom punch: a 9 % push and a saturation pop on the hit, settling in a few frames
  c_strobe     colour strobe: a wash of carnival colour on three beats from the hit, one a beat at most
  c_flare      sun flare: a warm flare and anamorphic streak sweep across the frame through the hit
  c_disco      mirror ball: spots of coloured light wheel across the room, brighter on every beat, kept off faces
  c_halftone   halftone pop: a few frames of 90s print dots in two carnival colours on the hit

LOOKS group them into ways to dress a whole film (lookbook.py renders them all), each with the MilkDrop presets
that suit it ("overlay": defaults under reel.json's own "overlay"). Taste (skills/focalcut/references/research.md,
"Carnival"): colour, not damage; keep it a party, not a slot machine -- mostly hard cuts, one effect at a time,
confetti saved for the big moments, faces always clean.
"""
import cv2
import numpy as np

from .base import Effect, Transition, register, to_u8
from .glitch import _seq, beats, feather, luma, masks_of, over, pulse

# carnival colours (RGB): gold, magenta, turquoise, orange, lime, purple, sky blue, red
PALETTE = np.array([[255, 196, 0], [255, 45, 149], [0, 210, 200], [255, 107, 26], [155, 226, 45], [142, 60, 203],
                    [40, 170, 255], [235, 30, 60]], np.float32)


def screen(base, light):
    """Screen blend (light 0..255 float): adds light without clipping colour into white blocks."""
    b = base.astype(np.float32)
    return to_u8(255 - (255 - b) * (255 - np.clip(light, 0, 255)) / 255)


def beat_index(n, ctx):
    """Which beat each frame is in, counted from the peak (the peak's beat is 0)."""
    t = (np.arange(n) - ctx.peak_frame(n)) / ctx.fps
    return np.floor(t / ctx.beat).astype(int)


def colours(ctx, k=2):
    """k different palette colours, chosen per shot."""
    return PALETTE[ctx.rng.choice(len(PALETTE), k, replace=False)]


def centroid(m):
    ys, xs = np.nonzero(m > 0.5)
    return (float(xs.mean()), float(ys.min() + 0.3 * (ys.max() - ys.min()))) if len(xs) else None


def halftone(f, c1, c2, cell):
    """The frame as print dots: c2 dots on a c1 ground, bigger where it is darker."""
    H, W = f.shape[:2]
    small = cv2.resize(luma(f.astype(np.float32)), (W // cell + 1, H // cell + 1), interpolation=cv2.INTER_AREA) / 255
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = yy // cell, xx // cell
    d = np.hypot(xx - (cx + 0.5) * cell, yy - (cy + 0.5) * cell)
    r = 0.62 * cell * np.sqrt(np.clip(1 - small[cy, cx], 0, 1))
    dot = np.clip(r - d + 0.5, 0, 1)[..., None]
    return to_u8(c1 * (1 - dot) + c2 * dot)


class Confetti:
    """Paper confetti: tumbling rectangles under gravity and drag, drawn anti-aliased."""

    def __init__(self, rng, W, H, fps):
        self.rng, self.W, self.H, self.fps = rng, W, H, fps
        self.p = np.zeros((0, 9), np.float32)       # x, y, vx, vy, angle, spin, size, colour, tumble phase

    def burst(self, x, y, n, speed, spread=np.pi, up=-np.pi / 2, size=0.009):
        a = up + self.rng.uniform(-spread / 2, spread / 2, n)
        v = self.rng.uniform(0.45, 1.0, n) * speed * self.H
        p = np.column_stack([np.full(n, x), np.full(n, y), np.cos(a) * v, np.sin(a) * v,
                             self.rng.uniform(0, np.pi, n), self.rng.uniform(-12, 12, n),
                             self.rng.uniform(0.6, 1.4, n) * size * self.W, self.rng.integers(0, len(PALETTE), n),
                             self.rng.uniform(0, 2 * np.pi, n)]).astype(np.float32)
        self.p = np.vstack([self.p, p])

    def step(self):
        dt = 1 / self.fps
        p = self.p
        p[:, 2] *= 0.985; p[:, 3] = p[:, 3] * 0.985 + 0.9 * self.H * dt     # drag, gravity
        p[:, 0] += p[:, 2] * dt + np.sin(p[:, 8]) * 0.6; p[:, 1] += p[:, 3] * dt
        p[:, 4] += p[:, 5] * dt; p[:, 8] += 7 * dt
        self.p = p[(p[:, 1] < self.H + 40) & (p[:, 0] > -60) & (p[:, 0] < self.W + 60)]

    def draw(self, f):
        img = f.copy()
        for x, y, _, _, a, _, s, c, ph in self.p:
            w, h = s, s * 1.7 * max(abs(np.cos(ph)), 0.15)                     # tumbling: the paper turns edge-on
            ca, sa = np.cos(a), np.sin(a)
            pts = np.array([[x + ca * dx - sa * dy, y + sa * dx + ca * dy]
                            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))])
            col = PALETTE[int(c)] * (0.75 + 0.25 * abs(np.cos(ph)))              # the paper catches the light
            cv2.fillConvexPoly(img, np.round(pts * 4).astype(np.int32), tuple(float(v) for v in col),
                               cv2.LINE_AA, shift=2)
        return img


# --- subject ---------------------------------------------------------------------------------------------------------

@register
class Glow(Effect):
    """Fiesta glow: a soft halo outside the dancer's edge, a new carnival colour on every beat, swelling on the peak."""
    name = "c_glow"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx)
        k = np.maximum(pulse(n, ctx, tau=0.3, attack=0.1), 0.45 * beats(n, ctx, 0.15)) * 0.65 + 0.35
        bi = beat_index(n, ctx); off = int(ctx.rng.integers(len(PALETTE)))
        out = []
        for i, f in enumerate(frames):
            H, W = f.shape[:2]
            m = cv2.resize((M[i] > 0.5).astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            r = max(3, W // 160)
            halo = cv2.GaussianBlur(cv2.dilate(m, np.ones((2 * r + 1,) * 2, np.uint8)), (0, 0), r * 1.2)
            halo = cv2.resize(np.clip(halo * 1.6, 0, 1) * (1 - np.clip(m * 1.5, 0, 1)), (W, H))
            col = PALETTE[(bi[i] + off) % len(PALETTE)]
            out.append(screen(f, halo[..., None] * col * k[i]))
        return out


@register
class Sparkle(Effect):
    """Sequins: four-point star glints twinkle on the bright parts of the dancer (each lives a few frames), a shower of
    them on the peak; gold and white."""
    name = "c_sparkle"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx); k = pulse(n, ctx, tau=0.35, attack=0.12)
        life = max(4, int(0.14 * ctx.fps)); live = []
        out = []
        for i, f in enumerate(frames):
            H, W = f.shape[:2]
            lum = luma(f.astype(np.float32))
            m = M[i] > 0.5
            thr = np.percentile(lum[m], 60) if m.any() else 0
            sel = np.argwhere(m & (lum > thr)) if m.any() else np.zeros((0, 2), int)
            spawn = int(ctx.rng.poisson(1.0 + 9.0 * k[i])) if len(sel) else 0
            for y, x in sel[ctx.rng.integers(0, len(sel), spawn)] if spawn else []:
                live.append([x, y, 0, ctx.rng.uniform(0.6, 1.3) * W / 70, ctx.rng.random() < 0.6])
            light = np.zeros((H, W, 3), np.float32)
            for s in live:
                x, y, age, size, gold = s
                e = np.sin(np.pi * (age + 1) / (life + 1)); r = int(size * e)
                if r < 2:
                    continue
                col = (255, 214, 120) if gold else (255, 255, 255)
                cv2.line(light, (int(x - r), int(y)), (int(x + r), int(y)), col, 2, cv2.LINE_AA)
                cv2.line(light, (int(x), int(y - r)), (int(x), int(y + r)), col, 2, cv2.LINE_AA)
                cv2.circle(light, (int(x), int(y)), max(2, r // 5), col, -1, cv2.LINE_AA)
            for s in live:
                s[2] += 1
            live = [s for s in live if s[2] < life]
            if light.any():
                light = light + cv2.GaussianBlur(light, (0, 0), 3) * 1.5
                out.append(screen(f, light))
            else:
                out.append(f)
        return out


@register
class ConfettiPop(Effect):
    """Confetti pop: on the peak, a burst of tumbling paper confetti is thrown up from the dancer's chest and falls away
    across the rest of the shot (in front of everything, like real confetti in the air)."""
    name = "c_confetti"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); p = ctx.peak_frame(n); H, W = frames[0].shape[:2]
        M = masks_of(frames, ctx)
        at = centroid(M[p]) or (W / 2, H * 0.4)
        c = Confetti(ctx.rng, W, H, ctx.fps)
        out = list(frames[:p])
        for i in range(p, n):
            if i == p:
                c.burst(*at, 260, speed=1.5, spread=np.pi * 0.9)
            if i == p + int(0.2 * ctx.fps):
                c.burst(*at, 120, speed=1.1, spread=np.pi * 1.2)
            out.append(c.draw(frames[i])); c.step()
        return out


@register
class Streamers(Effect):
    """Serpentina: echoes of the dancer 4, 8, 12 and 16 frames back, each a flat carnival colour (with a little of its
    own light), under the dancer -- a ribbon of colour behind every move, strongest round the peak."""
    name = "c_streamers"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx)
        k = np.clip(pulse(n, ctx, tau=0.6, attack=0.4) * 1.3, 0, 1)
        cols = colours(ctx, 4)
        out = []
        for i, f in enumerate(frames):
            if k[i] < 0.05:
                out.append(f); continue
            acc = f.astype(np.float32)
            for j, lag in enumerate((16, 12, 8, 4)):
                s = i - lag
                if s < 0:
                    continue
                tint = cols[j] * 0.75 + luma(frames[s].astype(np.float32))[..., None] * 0.25
                w = feather(M[s], 2, 7) * (0.35 + 0.15 * j) * k[i]
                acc = acc * (1 - w[..., None]) + tint * w[..., None]
            out.append(over(to_u8(acc), f, feather(M[i], 0, 5)))
        return out


@register
class Pop(Effect):
    """Pop-art freeze: the peak frame holds for half a beat with the dancer cut out -- a white sticker edge and a
    coloured one outside it -- over the room as halftone dots in two carnival colours; then the shot plays on from
    where it would have been."""
    name = "c_pop"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); p = ctx.peak_frame(n); M = masks_of(frames, ctx)
        h = max(6, int(0.5 * ctx.beat * ctx.fps))
        f = frames[p]; H, W = f.shape[:2]
        c1, c2, c3 = colours(ctx, 3)
        m = (M[p] > 0.5).astype(np.uint8)
        r1, r2 = max(4, W // 150), max(8, W // 70)
        white = cv2.dilate(m, np.ones((2 * r1 + 1,) * 2, np.uint8)).astype(np.float32)
        outer = cv2.dilate(m, np.ones((2 * r2 + 1,) * 2, np.uint8)).astype(np.float32)
        room = halftone(f, c1 * 0.35 + 20, c2, max(8, W // 110)).astype(np.float32)
        img = room * (1 - outer[..., None]) + c3 * outer[..., None]
        img = img * (1 - white[..., None]) + 255 * white[..., None]
        img = to_u8(img * (1 - m[..., None]) + f.astype(np.float32) * m[..., None])
        out = list(frames)
        for j in range(h):
            if p + j < n:
                z = 1 + 0.04 * (1 - j / h)
                M2 = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
                out[p + j] = cv2.warpAffine(img, M2, (W, H), borderMode=cv2.BORDER_REFLECT)
        return out


# --- transitions -----------------------------------------------------------------------------------------------------

def motion_blur(f, length):
    """Horizontal motion blur `length` px long (done at half size)."""
    L = int(length / 2)
    if L < 2:
        return f
    H, W = f.shape[:2]
    s = cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    s = cv2.filter2D(s, -1, np.full((1, L), 1 / L, np.float32), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


def shift(f, dx):
    H, W = f.shape[:2]
    return cv2.warpAffine(f, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)


@register
class Whip(Transition):
    """Whip pan: A accelerates out sideways in a growing motion blur, B arrives from the other side and settles -- the
    camera swung from one dancer to the next."""
    name = "c_whip"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        out = []; W = a[0].shape[1]; d = -1 if ctx.rng.random() < 0.5 else 1
        for j, f in enumerate(a):
            t = (j + 1) / len(a)
            out.append(motion_blur(shift(f, d * W * 0.45 * t ** 3), W * 0.16 * t ** 2))
        for j, f in enumerate(b):
            t = 1 - j / len(b)
            out.append(motion_blur(shift(f, -d * W * 0.45 * t ** 3), W * 0.16 * t ** 2))
        return out


@register
class Burst(Transition):
    """Confetti cannon: two cannons fire from the bottom corners across the frame; at the cut, behind the thickest of
    the paper, A becomes B, and the confetti falls away over B."""
    name = "c_burst"; duration_beats = 1.0

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        cf = Confetti(ctx.rng, W, H, ctx.fps)
        fire = max(0, c - int(0.45 * ctx.fps))
        k = max(1, int(round(0.1 * ctx.fps)))
        out = []
        for i in range(n):
            if fire <= i < fire + k:                          # each cannon fires ~380 pieces over a tenth of a second
                for _ in range(4):
                    cf.burst(ctx.rng.uniform(0, W * 0.08), H * ctx.rng.uniform(0.85, 1.0), 95 // k, speed=2.6,
                             spread=1.1, up=-np.pi / 3.4, size=0.018)
                    cf.burst(W - ctx.rng.uniform(0, W * 0.08), H * ctx.rng.uniform(0.85, 1.0), 95 // k, speed=2.6,
                             spread=1.1, up=-np.pi + np.pi / 3.4, size=0.018)
            fade = min(1.0, (n - 1 - i) / max(n * 0.35, 1))   # the paper thins out before B plays on clean
            d = cf.draw(fr[i]); cf.step()
            out.append(d if fade >= 1 else to_u8(fr[i] * (1 - fade) + d.astype(np.float32) * fade))
        return out


@register
class Flash(Transition):
    """Colour flash: A blooms up into one carnival colour (gold, magenta or turquoise), peaking on the cut, and B comes
    out of it -- a warm dip, never white."""
    name = "c_flash"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr)
        col = PALETTE[int(ctx.rng.choice([0, 1, 2]))]
        out = []
        for i, f in enumerate(fr):
            w = max(0.0, 1 - abs(i - c + 0.5) / (n / 2)) ** 1.5 * 0.9
            bloom = cv2.GaussianBlur(f, (0, 0), 6 + 20 * w).astype(np.float32)
            out.append(screen(f, (bloom * 0.6 + col * 0.8) * w))
        return out


@register
class Stripes(Transition):
    """Serpentina wipe: a band of slanted ribbons in carnival colours sweeps across the frame; A ahead of it, B behind."""
    name = "c_stripes"; duration_beats = 0.75

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        cols = colours(ctx, 5); K = len(cols)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s = xx + yy * np.tan(np.radians(22))
        band, span = W * 0.32, W + H * np.tan(np.radians(22))
        out = []
        for i in range(n):
            A, B = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            front = -band + (span + 2 * band) * (i + 0.5) / n        # the band's middle crosses the frame's at the cut
            inband = (s < front) & (s >= front - band)
            k = np.clip(((s - (front - band)) / band * K).astype(int), 0, K - 1)
            img = np.where((s < front - band)[..., None], B, A).astype(np.float32)
            img[inband] = cols[k[inband]]
            out.append(to_u8(img))
        return out


@register
class Sun(Transition):
    """Sun iris: a twelve-ray sun opens from the middle of the frame, turning as it grows, with a gold rim; B inside."""
    name = "c_sun"; duration_beats = 0.75

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot(xx - W / 2, yy - H / 2); th = np.arctan2(yy - H / 2, xx - W / 2)
        R = np.hypot(W, H) / 2 * 1.25
        gold = PALETTE[0]; rim = max(4, W // 120)
        out = []
        for i in range(n):
            A, B = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            u = ((i + 0.5) / n) ** 1.6
            edge = R * u * (1 + 0.16 * np.cos(12 * th + 1.5 * u * np.pi))
            inside = np.clip(edge - r, 0, 1)[..., None]
            ring = (np.clip(edge - r + rim, 0, 1) - inside[..., 0])[..., None]
            img = A * (1 - inside) + B * inside
            out.append(to_u8(img * (1 - ring) + gold * ring))
        return out


# --- accents ---------------------------------------------------------------------------------------------------------

@register
class Zoom(Effect):
    """Zoom punch: a 9 % push and a saturation pop on the hit, settling back in a few frames."""
    name = "c_zoom"

    def apply(self, frames, ctx):
        k = pulse(len(frames), ctx, 0.1); out = []
        for i, f in enumerate(frames):
            if k[i] < 0.04:
                out.append(f); continue
            H, W = f.shape[:2]
            g = cv2.warpAffine(f, cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + 0.09 * k[i]), (W, H),
                               borderMode=cv2.BORDER_REFLECT)
            hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.45 * k[i]), 0, 255)
            out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        return out


@register
class Strobe(Effect):
    """Colour strobe: on the hit and the two beats after it, two frames washed in a carnival colour (a new one each
    beat) -- one wash a beat at most."""
    name = "c_strobe"

    def apply(self, frames, ctx):
        n = len(frames); p = ctx.peak_frame(n); out = list(frames)
        cols = colours(ctx, 3); step = int(round(ctx.beat * ctx.fps))
        for j in range(3):
            for d in range(2):
                i = p + j * step + d
                if i < n:
                    f = frames[i].astype(np.float32)
                    w = 0.55 if d == 0 else 0.3
                    out[i] = to_u8(f * (1 - w) + (cols[j] * 0.6 + f * cols[j] / 255 * 0.8) * w)
        return out


@register
class Flare(Effect):
    """Sun flare: a warm flare (orange into magenta) and an anamorphic streak sweep across the frame through the hit,
    in the dark parts of the frame most."""
    name = "c_flare"

    def apply(self, frames, ctx):
        n = len(frames); p = ctx.peak_frame(n); H, W = frames[0].shape[:2]
        k = pulse(n, ctx, tau=0.35, attack=0.2)
        span = max(1, int(1.2 * ctx.beat * ctx.fps)); y0 = H * ctx.rng.uniform(0.2, 0.4)
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
        warm, pink = PALETTE[3], PALETTE[1]
        out = []
        for i, f in enumerate(frames):
            if k[i] < 0.04:
                out.append(f); continue
            x0 = W * (0.1 + 0.8 * np.clip((i - p + span / 2) / span, 0, 1))
            d = np.hypot(xx - x0, yy - y0) / W
            core = np.exp(-(d / 0.09) ** 2)[..., None]; halo = np.exp(-(d / 0.3) ** 2)[..., None]
            streak = np.exp(-((yy - y0) / (H * 0.012)) ** 2)[..., None] * np.exp(-((xx - x0) / (W * 0.45)) ** 2)[..., None]
            light = (core * 255 + halo * warm * 0.8 + streak * pink * 0.9) * k[i]
            light = cv2.resize(light, (W, H))
            dark = 1 - 0.6 * luma(f.astype(np.float32))[..., None] / 255
            out.append(screen(f, light * dark))
        return out


@register
class Disco(Effect):
    """Mirror ball: forty spots of coloured and white light wheel across the room, brighter on every beat and on the
    hit, in the dark parts of the frame (faces stay clean)."""
    name = "c_disco"

    def apply(self, frames, ctx):
        n = len(frames); H, W = frames[0].shape[:2]
        k = np.maximum(beats(n, ctx, 0.18) * 0.6, pulse(n, ctx, 0.3)) * 0.6 + 0.4
        N = 40; phi = ctx.rng.uniform(0, 2 * np.pi, N); lat = ctx.rng.uniform(-1, 1, N)
        col = np.where(ctx.rng.random(N)[:, None] < 0.35, 255.0, PALETTE[ctx.rng.integers(0, len(PALETTE), N)])
        h, w = H // 4, W // 4; rad = max(3, w // 70)
        out = []
        for i, f in enumerate(frames):
            light = np.zeros((h, w, 3), np.float32)
            a = phi + 0.9 * i / ctx.fps
            vis = np.cos(a) > -0.2                                   # spots on the near side of the ball
            xs = w / 2 + np.sin(a) * w * 0.65; ys = h / 2 + lat * h * 0.55
            for x, y, c, v in zip(xs, ys, col, vis):
                if v:
                    cv2.circle(light, (int(x), int(y)), rad, tuple(float(t) for t in c), -1, cv2.LINE_AA)
            light = cv2.resize(cv2.GaussianBlur(light, (0, 0), rad * 0.6), (W, H))
            dark = 1 - 0.75 * luma(f.astype(np.float32))[..., None] / 255
            out.append(screen(f, light * dark * k[i] * 0.85))
        return out


@register
class Halftone(Effect):
    """Halftone pop: four frames on the hit printed as 90s halftone dots in two carnival colours, then the shot."""
    name = "c_halftone"

    def apply(self, frames, ctx):
        p = ctx.peak_frame(len(frames)); out = list(frames)
        c1, c2 = colours(ctx, 2); W = frames[0].shape[1]
        for i in range(p, min(len(frames), p + 4)):
            out[i] = halftone(frames[i], c1 * 0.3 + 230 * 0.7, c2 * 0.85, max(8, W // 110))
        return out


# --- looks -----------------------------------------------------------------------------------------------------------

# MilkDrop presets for the carnival looks (Butterchurn's packs): flowers, fireworks, disco light, glowsticks and fire,
# warm and saturated, where edm.py keeps the space tunnels, lasers and ice
MILKDROP_FIESTA = [
    "martin - into the fireworks", "Martin - disco mix 6",
    "stahlregen & geiss + rovastar - fields of flowers (mashup 9 - space flower rmx) - mash0001 - pack em in, we got a "
    "long haul + flashlight",
    "Cope - The Neverending Explosion of Red Liquid Fire", "Stahlregen + martin + others - Psychedelic Metal Flower",
    "Martin - disco mix 3 -fast", "goody + martin - crystal palace - schizotoxin - the wild iris bloom - 16 iterations",
    "Eo.S. - glowsticks v2 03 music", "ech0 - liquid firesticks I", "Cope - Passage (mandala mix)",
    "flexi - jelly fish mandala"]
MILKDROP_SUN = ["Geiss - Desert Rose 4", MILKDROP_FIESTA[2], "martin - into the fireworks", MILKDROP_FIESTA[6]]
MILKDROP_DISCO = ["Martin - disco mix 6", "Martin - disco mix 3 -fast", "Eo.S. - glowsticks v2 03 music",
                  "martin - into the fireworks", "flexi - jelly fish mandala", "Cope - Passage (mandala mix)",
                  "Stahlregen + martin + others - Psychedelic Metal Flower"]
ZOOM = {"Cope - Passage (mandala mix)": 2, "Stahlregen + martin + others - Psychedelic Metal Flower": 2,
        "Cope - The Neverending Explosion of Red Liquid Fire": 2, "ech0 - liquid firesticks I": 2,
        "flexi - jelly fish mandala": 2}

# how the planner says the stock transition reasons in a carnival look (plan.glitch_look / transitions())
REASONS = {"spin": "c_whip", "swipe": "c_whip", "flash": "c_flash", "zoom": "c_burst", "fade": "c_stripes",
           "glitch": "c_stripes", "dipwhite": "c_flash", "lightleak": "c_sun"}

EDGE = dict(vignette=0.35, clear=0.9, bare=0.1)   # a light edge fade: a sunny daytime film, not a tunnel (overlay.py's default
                                        # 0.85 takes the corners to near black; this keeps them at ~40%)

LOOKS = {
    "carnaval": dict(
        title="Carnaval", mood="a street carnival in full colour: confetti, sequins and serpentina, every colour of the "
        "parade, MilkDrop fireworks, flowers and fire round the edges",
        base=[], subject=["c_confetti", "c_sparkle", "c_streamers", "c_glow"],
        transitions=["c_stripes", "c_whip", "c_burst", "c_flash"], accents=["c_zoom", "c_strobe", "c_flare"],
        use="a carnival cut on the cut nearest every 4-bar phrase, plus the cuts with a reason; sequins and serpentina "
            "on the moves, confetti on the climax's biggest move; zoom punches and colour strobes on the hits",
        reasons=REASONS, fx={"spotlight": "c_glow", "neon": "c_sparkle", "aura": "c_confetti"}, pulse=16,
        extra=[("arrival", "c_glow", ("spin", "style", "pattern", "energy")),
               ("shows", "c_sparkle", ("spin", "style", "pattern")),
               ("shows", "c_streamers", ("spin", "trick", "dip", "style")),
               ("night", "c_streamers", ("shake", "energy", "style", "spin")),
               ("climax", "c_confetti", ("spin", "trick", "dip", "energy"))],
        hits=[("arrival", "c_zoom", 1), ("shows", "c_zoom", 2), ("shows", "c_flare", 1), ("night", "c_strobe", 1),
              ("night", "c_zoom", 2), ("climax", "c_zoom", 3), ("climax", "c_strobe", 2), ("climax", "c_flare", 1)],
        overlay=dict(engine="milkdrop", milkdrop=MILKDROP_FIESTA, milkdrop_calm=MILKDROP_SUN, milkdrop_zoom=ZOOM,
                     flicker=False, **EDGE)),
    "tropical": dict(
        title="Tropical Sun", mood="a summer's day by the water: warm sun flares, a glow round the dancers, whip pans; "
        "the lightest touch, MilkDrop flowers soft at the edges",
        base=[], subject=["c_glow", "c_sparkle"], transitions=["c_whip", "c_sun", "c_flash"],
        accents=["c_flare", "c_zoom"],
        use="a carnival cut every 8 bars at most; glow on the moves, a sun flare on the biggest hit of each act",
        reasons=dict(REASONS, zoom="c_sun", fade="c_sun", glitch="c_whip"),
        fx={"spotlight": "c_glow", "neon": "c_sparkle", "aura": "c_glow"}, pulse=32,
        extra=[("arrival", "c_glow", ("spin", "style", "pattern", "energy")),
               ("shows", "c_glow", ("spin", "style", "pattern")),
               ("night", "c_sparkle", ("shake", "energy", "style", "spin"))],
        hits=[("arrival", "c_flare", 1), ("shows", "c_flare", 1), ("night", "c_zoom", 1), ("climax", "c_zoom", 2),
              ("climax", "c_flare", 1)],
        overlay=dict(engine="milkdrop", milkdrop=MILKDROP_SUN + MILKDROP_FIESTA[:3], milkdrop_calm=MILKDROP_SUN,
                     milkdrop_zoom=ZOOM, flicker=False, **EDGE, strength=dict(arrival=0.35, shows=0.5, night=0.7,
                                                                      climax=0.8, finale=0.7))),
    "disco": dict(
        title="Tropical Disco", mood="a Latin club night, 70s to 90s: mirror-ball light, colour strobes and pop-art "
        "freezes, MilkDrop disco and glowsticks",
        base=[], subject=["c_pop", "c_streamers", "c_glow"], transitions=["c_sun", "c_stripes", "c_flash", "c_whip"],
        accents=["c_disco", "c_strobe", "c_halftone"],
        use="mirror ball on the night's hits, a pop-art freeze on the biggest move of the night and the climax, colour "
            "strobes and halftone on the hardest hits",
        reasons=dict(REASONS, swipe="c_stripes", zoom="c_sun"),
        fx={"spotlight": "c_glow", "neon": "c_streamers", "aura": "c_pop"}, pulse=16,
        extra=[("shows", "c_glow", ("spin", "style", "pattern")),
               ("night", "c_pop", ("spin", "trick", "dip", "style")),
               ("night", "c_streamers", ("shake", "energy", "style", "spin")),
               ("climax", "c_pop", ("spin", "trick", "dip", "energy"))],
        hits=[("shows", "c_disco", 1), ("night", "c_disco", 2), ("night", "c_strobe", 1), ("night", "c_halftone", 1),
              ("climax", "c_strobe", 2), ("climax", "c_halftone", 1), ("climax", "c_disco", 1)],
        overlay=dict(engine="milkdrop", milkdrop=MILKDROP_DISCO, milkdrop_calm=MILKDROP_SUN, milkdrop_zoom=ZOOM,
                     flicker=False, **EDGE)),
}
