"""Glitch styles: one family of digital-damage looks for a whole film, as subject effects, transitions and accents.

Subject (on the dancer only, through the SAM 3 masks; the room stays clean):
  g_outline    RGB-split fringes round the dancer, thrown along their motion, pulsing on the peak
  g_mosh       datamosh bloom: the dancer's pixels held and dragged along their own motion, then released
  g_sort       pixel-sorted silhouette: streaks of sorted colour pour off the dancer
  g_ghost      scanline ghost trails: tinted echoes a few frames behind, drawn in lines
  g_dissolve   digital disintegrate: the dancer breaks into blocks that drift off, then fly back together
  g_slitscan   slit-scan: each row of the dancer from a different moment, so a turn bends through the body
  g_damage     block damage: macroblocks inside the dancer shift and colour-break, on hits
Transitions (apply(a_tail, b_head): the cut stays on its beat):
  g_blocks     macroblocks of B punch through A          g_moshcut    datamosh: B's motion drags A's pixels
  g_slices     horizontal bands slide, B shows in them    g_tear       RGB channels fly apart, punch, land on B
  g_vhsroll    VHS tracking roll with a noise bar         g_signal     a frame or two of "signal lost" snow / bars
  g_sortwipe   a pixel-sort wipe from A into B            g_stutter    noise warp and doubled frames hide the cut
Accents (whole frame; at the shot's peak unless they are a base texture):
  g_scanlines  scanlines (base) with a bump on the beat   g_crt        CRT curvature, bloom, vignette (base)
  g_vhs        VHS chroma bleed, noise, head-switch band  g_jpeg       JPEG macroblock crunch on the hit
  g_swap       a frame of swapped colour channels         g_interlace  odd and even lines tear apart
  g_freeze     freeze-stutter on the hit, then catch up   g_diff       a frame of what moved (difference)
  g_punch      a jolt and RGB punch on the beat

LOOKS group them into four ways to dress a whole film (lookbook.py renders them all). Built from the research in
skills/focalcut/references/research.md ("Glitch"): Sapphire / Red Giant docs, glitch-art writing, CapCut guides.
"""
import cv2
import numpy as np

from .base import Effect, Transition, register, to_u8

# --- shared pieces -------------------------------------------------------------------------------------------------

def pulse(n, ctx, tau=0.12, attack=0.03):
    """0..1 per frame: a sharp rise into the peak and a decay after it (tau, attack in seconds)."""
    p = ctx.peak_frame(n)
    t = (np.arange(n) - p) / ctx.fps
    return np.where(t >= 0, np.exp(-t / tau), np.exp(t / max(attack, 1e-3)))


def beats(n, ctx, tau=0.1):
    """0..1 per frame: a pulse on every beat, phased so one lands on the peak."""
    t = (np.arange(n) - ctx.peak_frame(n)) / ctx.fps
    return np.exp(-np.mod(t, ctx.beat) / tau)


def rgb(f, dx, dy=0):
    """R shifted by +dx, B by -dx (and dy)."""
    out = f.copy()
    out[..., 0] = np.roll(f[..., 0], (dy, dx), (0, 1))
    out[..., 2] = np.roll(f[..., 2], (-dy, -dx), (0, 1))
    return out


def feather(m, grow=4, soft=9):
    m = (m > 0.5).astype(np.uint8)
    if grow:
        m = cv2.dilate(m, np.ones((grow * 2 + 1,) * 2, np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (soft | 1, soft | 1), 0)


def over(base, top, w):
    w = w[..., None] if w.ndim == 2 else w
    return to_u8(base.astype(np.float32) * (1 - w) + top.astype(np.float32) * w)


def flow(a, b, scale=0.25):
    """Optical flow a -> b (pixels, full size), from a quarter-size Farneback."""
    ga = cv2.cvtColor(cv2.resize(a, None, fx=scale, fy=scale), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, None, fx=scale, fy=scale), cv2.COLOR_RGB2GRAY)
    f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (a.shape[1], a.shape[0])) / scale


def warp(img, fl, gain=1.0):
    """Drag img along a flow field (backward map: each pixel takes what was behind it)."""
    H, W = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img, gx - gain * fl[..., 0], gy - gain * fl[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def luma(f):
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def sort_rows(f, sel):
    """Sort each run of selected pixels along its row by brightness (pixel sorting)."""
    out = f.copy()
    L = luma(f.astype(np.float32))
    for y in np.nonzero(sel.any(1))[0]:
        row = sel[y]
        e = np.diff(np.r_[0, row.astype(np.int8), 0])
        for x0, x1 in zip(np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]):
            if x1 - x0 > 2:
                out[y, x0:x1] = f[y, x0:x1][np.argsort(L[y, x0:x1])]
    return out


def bands(H, rng, lo=2, hi=60):
    """Random horizontal bands covering the frame: [(y0, y1)]."""
    out, y = [], 0
    while y < H:
        h = int(rng.integers(lo, hi)); out.append((y, min(H, y + h))); y += h
    return out


def noise_field(H, W, rng, cell=48):
    """Smooth 2D noise in -1..1 (upsampled random grid)."""
    g = rng.uniform(-1, 1, (H // cell + 2, W // cell + 2)).astype(np.float32)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)


def masks_of(frames, ctx):
    return [np.asarray(m, np.float32) for m in ctx.masks(frames)]


# --- subject ---------------------------------------------------------------------------------------------------------

@register
class Outline(Effect):
    """RGB-split fringes round the dancer, thrown along their motion: 3 px at rest, up to ~60 px on the peak."""
    name = "g_outline"; needs_masks = True

    def apply(self, frames, ctx):
        M = masks_of(frames, ctx)
        k = pulse(len(frames), ctx, tau=0.18)
        cx = [np.nonzero(m.any(0))[0].mean() if m.any() else np.nan for m in (x > 0.5 for x in M)]
        v = np.nan_to_num(np.gradient(np.array(cx, float))) if len(cx) > 1 else np.zeros(len(frames))
        out = []
        for i, f in enumerate(frames):
            W = f.shape[1]
            dx = int(round((3 + 0.03 * W * k[i]) * (1 if v[i] >= 0 else -1)))
            m = (M[i] > 0.5).astype(np.float32)
            reach = np.maximum.reduce([m, np.roll(m, dx, 1), np.roll(m, -dx, 1)])
            out.append(over(f, rgb(f, dx, int(dx * 0.3)), feather(reach, 2, 5)))
        return out


@register
class Mosh(Effect):
    """Datamosh bloom on the dancer: from just before the peak their pixels stop refreshing and are dragged along
    their own motion (twice over), then they snap back over a few frames. The room stays clean."""
    name = "g_mosh"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx)
        p = ctx.peak_frame(n)
        a = max(1, p - int(0.15 * ctx.fps)); b = min(n - 1, p + int(0.45 * ctx.fps)); rel = int(0.12 * ctx.fps) + 1
        out = list(frames)
        buf = frames[a - 1].copy()
        for i in range(a, min(n, b + rel)):
            fl = flow(frames[i - 1], frames[i])
            buf = warp(buf, fl, gain=2.2)
            w = feather(np.maximum(M[i], M[a - 1]), 6, 15)
            if i > b:
                w *= 1 - (i - b) / rel
            out[i] = over(frames[i], buf, w)
        return out


@register
class Sort(Effect):
    """Pixel-sorted silhouette: the dancer's columns sorted by brightness, the streaks pouring below them further on
    the peak."""
    name = "g_sort"; needs_masks = True

    def apply(self, frames, ctx):
        M = masks_of(frames, ctx); k = pulse(len(frames), ctx, tau=0.2, attack=0.15)
        out = []
        for i, f in enumerate(frames):
            if k[i] < 0.08:
                out.append(f); continue
            m = (M[i] > 0.5).astype(np.uint8)
            reach = int(k[i] * 0.15 * f.shape[0])
            if reach:
                kern = np.zeros((2 * reach + 1, 1), np.uint8); kern[reach:] = 1      # grow downwards only
                m = cv2.dilate(m, kern)
            L = luma(f.astype(np.float32)) / 255
            sel = (m > 0) & (L > 0.12) & (L < 0.92)
            s = sort_rows(f.transpose(1, 0, 2).copy(), sel.T).transpose(1, 0, 2)
            out.append(over(f, s, feather(m, 0, 5) * min(1.0, k[i] * 1.6)))
        return out


@register
class Ghost(Effect):
    """Scanline ghost trails: echoes of the dancer 3, 6 and 9 frames back, tinted cyan and magenta, drawn in lines,
    under the dancer."""
    name = "g_ghost"; needs_masks = True
    TINTS = (np.array([0.3, 1.0, 1.0]), np.array([1.0, 0.3, 1.0]), np.array([0.4, 0.6, 1.0]))

    def apply(self, frames, ctx):
        M = masks_of(frames, ctx); k = np.clip(pulse(len(frames), ctx, tau=0.6, attack=0.4) * 1.3, 0, 1)
        H = frames[0].shape[0]
        lines = (np.arange(H) % 3 != 0).astype(np.float32)[:, None, None]
        out = []
        for i, f in enumerate(frames):
            acc = f.astype(np.float32)
            for j, lag in enumerate((9, 6, 3)):
                s = i - lag
                if s < 0 or k[i] < 0.05:
                    continue
                w = feather(M[s], 0, 5) * (1 - feather(M[i], 2, 5)) * (0.35 + 0.15 * j) * k[i]
                g = frames[s].astype(np.float32) * self.TINTS[j] * lines
                acc = acc * (1 - w[..., None]) + g * w[..., None]
            out.append(to_u8(acc))
        return out


@register
class Dissolve(Effect):
    """Digital disintegrate and back: from the peak the dancer breaks into 10 px blocks, left to right with some
    noise, each drifting up and away as it fades; a beat's hold of the room (blurred where they stood); then the blocks
    fly back the way they went and the dancer is whole again before the shot ends."""
    name = "g_dissolve"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx); p = ctx.peak_frame(n)
        left = max(6, n - 1 - p)                                  # frames after the peak, all of it round trip
        dur = max(3, min(int(0.5 * ctx.fps), int(left / 2.6)))    # out, a short hold, back
        hold = max(1, min(int(0.25 * ctx.beat * ctx.fps), left - 2 * dur))
        H, W = frames[0].shape[:2]; b = max(6, W // 96)
        gh, gw = H // b + 1, W // b + 1
        thr = np.clip(np.linspace(0, 1, gw)[None, :] * 0.7 + ctx.rng.uniform(0, 0.3, (gh, gw)), 0, 1)
        vel = ctx.rng.uniform([-0.5, -2.5], [1.5, -0.5], (gh, gw, 2)) * b
        out = []
        for i, f in enumerate(frames):
            k = i - p
            if k <= 0 or k >= 2 * dur + hold:
                out.append(f); continue
            prog = k / dur if k <= dur else 1.0 if k <= dur + hold else 1 - (k - dur - hold) / dur
            prog *= 1.3                                           # (every block gone at the widest)
            m = M[i] > 0.5
            gone = np.kron(thr < prog, np.ones((b, b), bool))[:H, :W] & m
            bg = cv2.blur(f, (b * 4 + 1, b * 4 + 1))
            img = f.copy(); img[gone] = bg[gone]
            age = np.clip(prog - thr, 0, None)
            for (gy, gx) in zip(*np.nonzero((age > 0) & (age < 0.6))):
                y0, x0 = gy * b, gx * b
                blk = f[y0:y0 + b, x0:x0 + b]; mb = m[y0:y0 + b, x0:x0 + b]
                if not mb.any():
                    continue
                t = age[gy, gx] * dur
                y1, x1 = int(y0 + vel[gy, gx, 1] * t), int(x0 + vel[gy, gx, 0] * t)
                if 0 <= y1 < H - b and 0 <= x1 < W - b and blk.shape[:2] == (b, b):
                    a = (1 - age[gy, gx] / 0.6) * mb[..., None]
                    img[y1:y1 + b, x1:x1 + b] = to_u8(img[y1:y1 + b, x1:x1 + b] * (1 - a) + blk * a)
            out.append(img)
        return out


@register
class SlitScan(Effect):
    """Slit-scan dancer: row y of the dancer comes from up to 14 frames back, the top row now, the feet last; the
    room stays live. Strongest around the peak."""
    name = "g_slitscan"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx)
        k = np.clip(pulse(n, ctx, tau=0.5, attack=0.4) * 1.2, 0, 1)
        H = frames[0].shape[0]; T = 14
        out = []
        for i, f in enumerate(frames):
            lag = np.clip((np.arange(H) / H * T * k[i]).astype(int), 0, i)
            src = np.stack([frames[i - lg][y] for y, lg in enumerate(lag)])
            mm = np.stack([M[i - lg][y] for y, lg in enumerate(lag)])
            out.append(over(f, src, feather(np.maximum(mm, M[i]), 2, 7)))
        return out


@register
class Damage(Effect):
    """Block damage inside the dancer on the hits: up to two fifths of the 32 px blocks shift and swap colour
    channels, for two or three frames on the peak and a lighter flicker on the beats."""
    name = "g_damage"; needs_masks = True

    def apply(self, frames, ctx):
        n = len(frames); M = masks_of(frames, ctx)
        k = np.maximum(pulse(n, ctx, tau=0.07, attack=0.02), 0.6 * beats(n, ctx, tau=0.05))
        out = []
        for i, f in enumerate(frames):
            if k[i] < 0.25:
                out.append(f); continue
            H, W = f.shape[:2]; b = max(8, W // 60)
            img = f.copy(); m = M[i] > 0.5
            ys, xs = np.nonzero(m[::b, ::b])
            pick = ctx.rng.random(len(ys)) < 0.4 * k[i]
            for gy, gx in zip(ys[pick], xs[pick]):
                y0, x0 = gy * b, gx * b
                dy, dx = ctx.rng.integers(-b, b + 1, 2)
                y1, x1 = np.clip([y0 + dy, x0 + dx], 0, [H - b, W - b])
                blk = f[y1:y1 + b, x1:x1 + b][..., ctx.rng.permutation(3)]
                img[y0:y0 + b, x0:x0 + b] = blk[:img[y0:y0 + b, x0:x0 + b].shape[0], :img[y0:y0 + b, x0:x0 + b].shape[1]]
            out.append(img)
        return out


# --- transitions -----------------------------------------------------------------------------------------------------

def _seq(a, b):
    return list(a) + list(b), len(a)


@register
class Blocks(Transition):
    """Macroblocks of B punch through A: mixed 48-160 px blocks (at 1080p) flip to B on a ramp centred on the cut, some thrown
    sideways with a colour offset."""
    name = "g_blocks"; duration_beats = 0.75

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        out = []
        for i in range(n):
            u = np.clip((i - c) / max(n * 0.35, 1) + 0.5, 0, 1)
            A, B = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            img = (A if i < c else B).copy()
            for _ in range(int(170 * (1 - abs(2 * u - 1)) + 10)):
                s = int(ctx.rng.choice([48, 64, 96, 128, 160]) * W / 1920) + 4
                y, x = ctx.rng.integers(0, H - s), ctx.rng.integers(0, W - s)
                src = B if ctx.rng.random() < u else A
                dx = int(ctx.rng.integers(-120, 121) * W / 1920)
                xs = int(np.clip(x + dx, 0, W - s))
                img[y:y + s, x:x + s] = rgb(src[y:y + s, xs:xs + s], int(ctx.rng.integers(-14, 15)))
            out.append(img)
        return out


@register
class MoshCut(Transition):
    """Datamosh cut: A's last frame is never refreshed -- B's own motion drags it until B resolves through in blocks.
    Needs motion in B (a spin, a whip)."""
    name = "g_moshcut"; duration_beats = 1.0

    def apply(self, a, b, ctx):
        out = list(a); buf = a[-1].copy(); H, W = buf.shape[:2]; s = max(8, W // 60)
        nb = len(b); thr = ctx.rng.random((H // s + 1, W // s + 1))
        for j in range(nb):
            if j:
                buf = warp(buf, flow(b[j - 1], b[j]), gain=1.6)
            u = max(0.0, (j / max(nb - 1, 1) - 0.45) / 0.55)            # hold, then resolve block by block
            m = np.kron(thr < u, np.ones((s, s)))[:H, :W].astype(np.float32)
            out.append(over(buf, b[j], m))
        return out


@register
class Slices(Transition):
    """Horizontal bands slide sideways, most at the cut, and B shows in more of them as it passes; the moving bands
    split their colour."""
    name = "g_slices"; duration_beats = 0.75

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        bs = bands(H, ctx.rng, max(2, H // 200), max(6, H // 14))
        off = ctx.rng.uniform(-1, 1, len(bs)); flip = ctx.rng.random(len(bs))
        out = []
        for i in range(n):
            k = 1 - abs(i - c + 0.5) / (n / 2)
            u = np.clip((i - c) / max(n * 0.4, 1) + 0.5, 0, 1)
            A, B = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            img = np.empty_like(A)
            for (y0, y1), o, fl in zip(bs, off, flip):
                src = B if fl < u else A
                dx = int(o * 0.2 * W * k ** 1.5)
                band = np.roll(src[y0:y1], dx, 1)
                img[y0:y1] = rgb(band, int(np.sign(dx) * min(abs(dx) // 5, 16))) if abs(dx) > 2 else band
            out.append(img)
        return out


@register
class Tear(Transition):
    """RGB tear: A's channels fly apart with a zoom punch and a sideways smear, B arrives at the widest split and the
    channels snap back together on it."""
    name = "g_tear"; duration_beats = 0.75

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr)
        out = []
        for i, f in enumerate(fr):
            k = max(0.0, 1 - abs(i - c + 0.5) / (n / 2)) ** 1.3
            H, W = f.shape[:2]
            dx = int(0.06 * W * k)
            z = 1 + 0.16 * k
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
            g = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            if dx:
                g = cv2.blur(g, (max(1, dx // 3) * 2 + 1, 1))
            out.append(rgb(g, dx))
        return out


@register
class VhsRoll(Transition):
    """VHS tracking roll: A rolls down out of the frame and B rolls in from the top behind a noisy tracking bar;
    rows near the bar skew, colour slips."""
    name = "g_vhsroll"; duration_beats = 1.0

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        out = []
        for i in range(n):
            u = i / (n - 1)
            e = u * u * (3 - 2 * u)
            y = int(e * H)                                              # where the seam is
            A, B = a[min(i, len(a) - 1)], b[max(0, i - len(a))]
            img = np.concatenate([B[H - y:], A[:H - y]]) if y else A.copy()
            bar = max(8, H // 30)
            for r in range(max(0, y - 3 * bar), min(H, y + 3 * bar)):
                img[r] = np.roll(img[r], int(14 * W / 1920 * np.sin(r / 7 + i) * (1 - abs(r - y) / (3 * bar))), 0)
            y0, y1 = max(0, y - bar // 2), min(H, y + bar // 2)
            if y1 > y0:
                nz = ctx.rng.integers(0, 255, (y1 - y0, W, 1), dtype=np.uint8)
                img[y0:y1] = to_u8(img[y0:y1] * 0.3 + nz * 0.7)
            out.append(rgb(img, max(1, W // 400)))
        return out


@register
class Signal(Transition):
    """Signal lost: a beat of snow (or colour bars, or a blue NO SIGNAL screen: kind=) over the cut, with a jolt of
    colour slip either side."""
    name = "g_signal"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        kind = self.p.get("kind", "snow")
        hole = range(max(0, c - max(1, n // 6)), min(n, c + max(1, n // 6)))
        out = []
        for i, f in enumerate(fr):
            if i in hole:
                if kind == "bars":
                    cols = np.array([[192, 192, 192], [192, 192, 0], [0, 192, 192], [0, 192, 0], [192, 0, 192],
                                     [192, 0, 0], [0, 0, 192]], np.uint8)
                    img = cols[(np.arange(W) * 7 // W)][None].repeat(H, 0)
                elif kind == "blue":
                    img = np.zeros((H, W, 3), np.uint8); img[..., 2] = 170
                    cv2.putText(img, "NO SIGNAL", (W // 2 - W // 7, H // 2), cv2.FONT_HERSHEY_SIMPLEX, W / 900,
                                (255, 255, 255), max(1, W // 600), cv2.LINE_AA)
                else:
                    g = ctx.rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
                    img = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)[..., None].repeat(3, 2)
                img = to_u8(img * (0.75 + 0.25 * (np.arange(H) % 2))[:, None, None])
            else:
                d = min(abs(i - hole.start), abs(i - hole.stop + 1))
                img = rgb(f, int(W / 120 / (1 + d))) if d < 3 else f
            out.append(img)
        return out


@register
class SortWipe(Transition):
    """Pixel-sort wipe: sorted streaks spread across A from the left, B takes over under them at the cut, and the
    streaks drain off to the right."""
    name = "g_sortwipe"; duration_beats = 1.0

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        out = []
        for i, f in enumerate(fr):
            r = i / (n - 1)
            s = 1 - abs(2 * r - 1)
            L = luma(f.astype(np.float32)) / 255
            band = np.zeros((H, W), bool)
            x0, x1 = int(max(0, r - 0.5) * 2 * W), int(min(1, r * 2) * W)
            band[:, x0:x1] = True
            sel = band & (np.abs(L - 0.5) < 0.5 * s)
            out.append(sort_rows(f, sel))
        return out


@register
class Stutter(Transition):
    """Stutter cut: frames around the cut go A A B A B B, warped by a noise field that peaks at the cut."""
    name = "g_stutter"; duration_beats = 0.5

    def apply(self, a, b, ctx):
        fr, c = _seq(a, b); n = len(fr); H, W = fr[0].shape[:2]
        order = list(range(n))
        for j, o in zip(range(c - 2, c + 2), (c - 2, c, c - 1, c + 1)):
            if 0 <= j < n and 0 <= o < n:
                order[j] = o
        nx, ny = noise_field(H, W, ctx.rng), noise_field(H, W, ctx.rng)
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        out = []
        for i in range(n):
            k = max(0.0, 1 - abs(i - c + 0.5) / 3)
            f = fr[order[i]]
            amp = 0.02 * W * k
            out.append(cv2.remap(f, gx + amp * nx, gy + amp * ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                       if amp > 0.5 else f)
        return out


# --- accents ---------------------------------------------------------------------------------------------------------

@register
class Scanlines(Effect):
    """Scanlines over the whole shot (strength=0.15), a little darker on every beat."""
    name = "g_scanlines"

    def apply(self, frames, ctx):
        H = frames[0].shape[0]; p = max(2, H // 360)
        line = ((np.arange(H) % (2 * p)) < p).astype(np.float32)[:, None, None]
        k = beats(len(frames), ctx, 0.08); s = self.p.get("strength", 0.15)
        return [to_u8(f * (1 - (s + 0.15 * k[i]) * line)) for i, f in enumerate(frames)]


@register
class Crt(Effect):
    """CRT: barrel curvature, highlight bloom and a vignette; the bow punches a little on the peak."""
    name = "g_crt"

    def apply(self, frames, ctx):
        H, W = frames[0].shape[:2]
        gx, gy = np.meshgrid(np.linspace(-1, 1, W, dtype=np.float32), np.linspace(-1, 1, H, dtype=np.float32))
        r2 = gx ** 2 + gy ** 2
        vig = np.clip(1.15 - 0.45 * r2, 0, 1)[..., None]
        k = pulse(len(frames), ctx, tau=0.15)
        out = []
        for i, f in enumerate(frames):
            c = 0.06 + 0.05 * k[i]
            mx, my = (gx * (1 + c * r2) + 1) * (W - 1) / 2, (gy * (1 + c * r2) + 1) * (H - 1) / 2
            g = cv2.remap(f, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            hi = np.clip(g - 170, 0, None)
            g = g + 0.5 * cv2.GaussianBlur(hi, (0, 0), W / 150)
            out.append(to_u8(g * vig))
        return out


@register
class Vhs(Effect):
    """VHS: chroma smeared and slipped right, luma noise, a torn head-switching band at the bottom; the tracking
    jitters on the beats (strength=1)."""
    name = "g_vhs"

    def apply(self, frames, ctx):
        H, W = frames[0].shape[:2]; s = self.p.get("strength", 1.0)
        k = beats(len(frames), ctx, 0.06)
        out = []
        for i, f in enumerate(frames):
            y = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32)
            ch = cv2.resize(cv2.resize(y[..., 1:], (W // 8, H)), (W, H))
            y[..., 1:] = np.roll(ch, int(W / 480 * (1 + 2 * s)), 1)
            y[..., 0] += ctx.rng.normal(0, 4 * s, (H, W))
            img = cv2.cvtColor(to_u8(y), cv2.COLOR_YCrCb2RGB)
            hb = max(4, H // 70)
            for r in range(H - hb, H):
                img[r] = np.roll(img[r], int(ctx.rng.integers(10, 40) * W / 1920), 0)
            if k[i] > 0.5:                                                   # a tracking wobble on the beat
                r0 = int(ctx.rng.integers(0, H - H // 20))
                img[r0:r0 + H // 20] = np.roll(img[r0:r0 + H // 20], int(W / 100 * k[i]), 1)
            out.append(img)
        return out


@register
class Jpeg(Effect):
    """JPEG crunch: three frames on the peak go through a quality-4 JPEG, blocks and all."""
    name = "g_jpeg"

    def apply(self, frames, ctx):
        p = ctx.peak_frame(len(frames)); out = list(frames)
        for i in range(p, min(len(frames), p + 3)):
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 4])
            out[i] = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return out


@register
class Swap(Effect):
    """Channel swap: one frame on the peak in the wrong colours (RGB -> BRG)."""
    name = "g_swap"

    def apply(self, frames, ctx):
        p = ctx.peak_frame(len(frames)); out = list(frames)
        out[p] = np.ascontiguousarray(frames[p][..., [2, 0, 1]])
        return out


@register
class Interlace(Effect):
    """Interlace jitter: odd lines slip right and even lines left on the peak and the beats, odd lines a frame late."""
    name = "g_interlace"

    def apply(self, frames, ctx):
        n = len(frames); k = np.maximum(pulse(n, ctx, 0.08), 0.5 * beats(n, ctx, 0.05))
        out = []
        for i, f in enumerate(frames):
            d = int(f.shape[1] / 160 * k[i])
            if d < 1:
                out.append(f); continue
            img = f.copy()
            img[1::2] = np.roll(frames[max(0, i - 1)][1::2], d, 1)
            img[0::2] = np.roll(f[0::2], -d, 1)
            out.append(img)
        return out


@register
class Freeze(Effect):
    """Freeze-stutter on the hit: the peak frame sticks for a sixth of a beat-ish (f, f+1, f, f+1, f) with a colour slip
    as it catches, then the shot jumps on to where it would have been."""
    name = "g_freeze"

    def apply(self, frames, ctx):
        n = len(frames); p = ctx.peak_frame(n); h = max(4, int(0.4 * ctx.beat * ctx.fps))
        out = list(frames)
        for j in range(h):
            if p + j < n:
                out[p + j] = frames[min(n - 1, p + (j % 2 if j < h - 1 else 0))]
        if p < n:
            out[p] = rgb(out[p], max(3, frames[0].shape[1] // 90))
        return out


@register
class Diff(Effect):
    """Difference flash: two frames on the peak show only what moved (|f - f two frames back|, boosted) -- footwork and
    hair light up."""
    name = "g_diff"

    def apply(self, frames, ctx):
        p = ctx.peak_frame(len(frames)); out = list(frames)
        for i in range(max(2, p), min(len(frames), p + 2)):
            out[i] = to_u8(np.abs(frames[i].astype(np.float32) - frames[i - 2]) * 3.5)
        return out


@register
class Punch(Effect):
    """Beat punch: a jolt, a 7 % zoom and an RGB split on the peak, decaying in a few frames."""
    name = "g_punch"

    def apply(self, frames, ctx):
        k = pulse(len(frames), ctx, 0.09); out = []
        for i, f in enumerate(frames):
            if k[i] < 0.05:
                out.append(f); continue
            H, W = f.shape[:2]
            dx, dy = ctx.rng.uniform(-1, 1, 2) * W * 0.02 * k[i]
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + 0.07 * k[i]); M[:, 2] += (dx, dy)
            g = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            out.append(rgb(g, int(W / 120 * k[i])))
        return out


# --- looks -----------------------------------------------------------------------------------------------------------

LOOKS = {
    "clean": dict(title="Clean Signal", mood="modern, premium: natural footage, glitches only on hits, blocky "
                  "digital damage and colour chunks on the peaks",
                  base=[], subject=["g_outline", "g_damage", "g_dissolve"],
                  transitions=["g_tear", "g_slices", "g_sortwipe", "g_blocks"],
                  accents=["g_freeze", "g_punch", "g_jpeg"],
                  use="a glitch transition on the cut nearest every 4-bar phrase, plus the cuts with a reason; outline and "
                      "damage on the moves; freeze, punch and JPEG crunch on the hits",
                  # how the planner says it (plan.glitch_look): each transition reason in this look's terms, its
                  # subject effects in place of the stock ones, one extra subject effect per act on the move it
                  # suits, and accents on the act's hardest hits
                  reasons={"spin": "g_blocks", "swipe": "g_slices", "flash": "g_tear", "zoom": "g_tear",
                           "fade": "g_sortwipe", "glitch": "g_blocks", "dipwhite": "g_tear",
                           "lightleak": "g_sortwipe"},
                  fx={"spotlight": "g_outline", "neon": "g_damage", "aura": "g_dissolve"},
                  pulse=16,                         # beats: a glitch cut on the cut nearest each 4-bar phrase
                  extra=[("arrival", "g_outline", ("spin", "style", "pattern", "energy")),
                         ("shows", "g_outline", ("spin", "style", "pattern")),
                         ("shows", "g_damage", ("spin", "trick", "dip", "style")),
                         ("night", "g_damage", ("shake", "energy", "style", "spin")),
                         ("climax", "g_damage", ("shake", "energy", "style", "spin"))],
                  hits=[("arrival", "g_punch", 1), ("shows", "g_freeze", 2), ("shows", "g_jpeg", 1), ("night", "g_freeze", 1),
                        ("night", "g_jpeg", 2), ("night", "g_punch", 1),
                        ("climax", "g_punch", 4), ("climax", "g_jpeg", 2), ("climax", "g_freeze", 1)]),
    "tape": dict(title="Tape from the Congress", mood="VHS retro, warm: a light tape texture throughout",
                 base=[("g_vhs", dict(strength=0.4))], subject=["g_ghost"], transitions=["g_vhsroll", "g_signal"],
                 accents=["g_interlace", "g_scanlines"],
                 use="texture all film; a tracking roll every 45-60 s; signal lost at most twice; keep close-ups light"),
    "mosh": dict(title="Moshpit Mambo", mood="harsh digital, high energy: corruption saved for the peaks",
                 base=[], subject=["g_mosh", "g_damage"], transitions=["g_moshcut", "g_blocks", "g_stutter"],
                 accents=["g_jpeg", "g_diff", "g_swap"],
                 use="climax third only; one datamosh per act at most; opening and social floor stay clean"),
    "neon": dict(title="Neon Grid", mood="cyber, stylised: scanlines and bloom, sorted light",
                 base=[("g_scanlines", dict(strength=0.1))], subject=["g_sort", "g_slitscan", "g_dissolve"],
                 transitions=["g_sortwipe", "g_slices"], accents=["g_swap", "g_freeze"],
                 use="subject effect at most once per 16 bars; pixel sort 3-4 times a film; dissolve for the exit"),
}
