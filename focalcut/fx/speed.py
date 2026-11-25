"""Speed curves, after CapCut's Speed > Curve presets: montage, hero, bullet (ramp - hold - ramp), jumpcut, flashin,
flashout; and fit, which plays an editor's whole in-out in the slot it was given.

Each preset is a speed multiplier over output time, centred on the move's peak. Slow stretches get optical-flow
in-between frames (smooth slow-mo); fast stretches average the frames they skip (motion blur). The source window is
longer or shorter than the output, so the renderer asks source_length() / source_peak() before decoding.

Ramps by what moves (docs/research/speed-ramps.md): dunk, rise, reveal, drift, none -- RAMPS, the Ramp class and
ramp_window() at the end of this file. Each is a few speeds joined by eased transitions timed in beats, keyed to
the action (the shot's peak, landed on a bar line or a beat), never slower than the source's frame rate carries
(slow_floor) and set from reel.json "ramps".

A stretch at exactly 1x plays the source's own frames, untouched (snap_steps: the ramps before it are nudged so it
starts on a whole frame), and a curve reads its source as it goes (Speed.stream), holding only the few frames it is
working on -- as RGB arrays or as the raw 4:2:0 a shot is decoded to and encoded from (Yuv), so a 4K shot through a
speed curve alone is never held in memory whole and never takes a trip through RGB.
"""
import cv2
import numpy as np

from .base import Effect, register

TAIL = 0.5                                          # seconds at 1x at the end of every shot


GHOST = 0.02             # flow beyond this share of the frame width (90th percentile) ghosts when warped


def settle_tail(v, t, out_len, ctx, tail=TAIL):
    """Whatever a curve does, a shot's last tail seconds play at 1x, eased into over 0.4 beats before them."""
    ease = max(0.4 * ctx.beat, 1 / ctx.fps)
    x = np.clip((t - (out_len - tail - ease)) / ease, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * x)
    return v + (1 - v) * w


FLOW_W = 960             # optical flow is found on frames this wide and scaled up (4K flow: 16x the work, same warp)


class Yuv:
    """A raw yuv420p frame, as a shot's filter chain hands it over and the encoder takes it: `raw` (its bytes, or
    an array of them) and the three planes as views of it. A frame a speed curve passes through goes on as the very
    bytes that were decoded."""
    __slots__ = ("raw", "u", "v", "y")

    def __init__(self, raw, w, h):
        a = raw if isinstance(raw, np.ndarray) else np.frombuffer(raw, np.uint8)
        n = w * h
        self.raw = raw
        self.y, self.u, self.v = a[:n].reshape(h, w), a[n:n + n // 4].reshape(h // 2, w // 2), \
            a[n + n // 4:n + n // 2].reshape(h // 2, w // 2)

    @classmethod
    def join(cls, y, u, v):
        return cls(np.concatenate([y.ravel(), u.ravel(), v.ravel()]), y.shape[1], y.shape[0])


def _planes(f):
    return (f.y, f.u, f.v) if isinstance(f, Yuv) else (f,)


def _like(f, planes):
    return Yuv.join(*planes) if isinstance(f, Yuv) else planes[0]


def _grey(f, size=None):
    """A frame's grey picture for the optical flow, full range, at size (w, h) when given: an RGB frame's grey, or
    a Yuv frame's luma stretched from video range (16-235), so both find the same flow."""
    g = f.y if isinstance(f, Yuv) else cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    if size and size != g.shape[1::-1]:
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return cv2.convertScaleAbs(g, alpha=255 / 219, beta=-16 * 255 / 219) if isinstance(f, Yuv) else g


_grids = {}


def _grid(w, h):
    """Every pixel's own (x, y), float32, two channels: what a flow is added to for cv2.remap. Kept per size (a 4K
    one is 66 MB and took 25 ms to build for every in-between frame)."""
    if (w, h) not in _grids:
        if len(_grids) >= 6:
            _grids.clear()
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        _grids[(w, h)] = np.ascontiguousarray(np.dstack([gx, gy]))
    return _grids[(w, h)]


class Flows:
    """The optical flow between two neighbouring frames, both ways, found once at FLOW_W wide (a 4K shot's
    in-betweens cost what a 1080p one's do) and scaled to each plane's size as it is first warped."""

    def __init__(self, dis, a, b):
        h, w = _planes(a)[0].shape[:2]
        k = min(1.0, FLOW_W / w)
        self.size = (int(round(w * k)), int(round(h * k))) if k < 1.0 else (w, h)
        g0, g1 = _grey(a, self.size), _grey(b, self.size)
        self.small = dis.calc(g0, g1, None), dis.calc(g1, g0, None)        # a -> b, b -> a
        self.at = {}
        self._ghost = None

    def ghost(self):
        """The motion's 90th percentile as a share of the frame width (on the small flow: 3 ms, not the 50 a 4K
        one took)."""
        if self._ghost is None:
            f = self.small[0]
            self._ghost = float(np.percentile(np.hypot(f[..., 0], f[..., 1]), 90)) / self.size[0]
        return self._ghost

    def sized(self, w, h):
        """Both flows in the pixels of a w x h plane (a chroma plane's are half the luma's)."""
        if (w, h) not in self.at:
            k = w / self.size[0]
            self.at[(w, h)] = self.small if k == 1.0 else tuple(
                cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * np.float32(k) for f in self.small)
        return self.at[(w, h)]

    def between(self, f0, f1, a):
        """The frame a (0..1) of the way from f0 to f1: both warped along the flow to that time, cross-faded."""
        out = []
        for p0, p1 in zip(_planes(f0), _planes(f1)):
            h, w = p0.shape[:2]
            f01, f10 = self.sized(w, h)
            grid = _grid(w, h)
            w0 = cv2.remap(p0, cv2.scaleAdd(f10, float(a), grid), None, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(p1, cv2.scaleAdd(f01, float(1 - a), grid), None, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            out.append(cv2.addWeighted(w0, float(1 - a), w1, float(a), 0))
        return _like(f0, out)


class Window:
    """The stretch of a shot's source a speed curve is working on: frames read from an iterator as they are first
    asked for, and let go once the curve has moved past them. `last` is the final frame's index once the source
    has ended (None until then)."""

    def __init__(self, source):
        self.it, self.buf, self.base, self.last = iter(source), [], 0, None

    def need(self, i):
        """Read on until frame i is here, or the source ends. Returns the last index known to exist."""
        while self.last is None and self.base + len(self.buf) <= i:
            f = next(self.it, None)
            if f is None:
                self.last = self.base + len(self.buf) - 1
            else:
                self.buf.append(f)
        return self.last if self.last is not None else self.base + len(self.buf) - 1

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.buf[max(i.start - self.base, 0):max(i.stop - self.base, 0)]
        return self.buf[i - self.base]

    def drop(self, below):
        """Let go of the frames before `below` (never the newest: a curve past the source's end holds on it)."""
        k = min(below - self.base, len(self.buf) - 1)
        if k > 0:
            del self.buf[:k]
            self.base += k


def _mean(frames):
    """Frames averaged (motion blur over the frames a fast output frame spans)."""
    return _like(frames[0], [np.mean(np.stack(ps).astype(np.float32), 0).astype(np.uint8)
                             for ps in zip(*map(_planes, frames))])


SNAP_SHARE = 0.5         # snap_steps: a ramp is nudged by at most this share of what it differs from 1x by


def snap_steps(step):
    """step: the source frames each output frame advances by. Wherever the curve plays at exactly 1x, the stretch
    of changing speed before it is nudged (each frame in proportion to how far from 1x it runs, half a source frame
    at most over the whole stretch -- a per cent or so of a ramp's speed) so the 1x stretch starts on a whole source
    frame and so plays the source's own frames. Left alone, a ramp ends a fraction of a frame in, and every frame
    of the 1x stretch after it -- most of a shot -- is an optical-flow in-between of two real frames: softer than
    the footage, and the slowest thing a render computes. Returns the nudged copy."""
    step = np.array(step, float)
    one = np.abs(step - 1.0) < 1e-9
    k, n, start = 0, len(step), 0                  # start: where the stretch of changing speed began
    while k < n:
        if not one[k]:
            k += 1
            continue
        if k > start:                              # a 1x run begins at step k, after steps start..k-1 off 1x
            pos = step[:k].sum()
            d = np.rint(pos) - pos
            w = np.abs(step[start:k] - 1.0)
            if abs(d) > 1e-9 and abs(d) <= SNAP_SHARE * w.sum():
                step[start:k] += d * w / w.sum()
        while k < n and one[k]:
            k += 1
        start = k
    return step


class Speed(Effect):
    """A velocity curve the way CapCut's custom curve and a time-remap ramp are built: fast through the move, easing
    down to arrive slow exactly as the dancer lands (the shot's peak: frame dwell detected by the planner), holding
    there over the rest so the hair and the face play out, then easing back to real speed (1x) and staying there. The
    fast part is only the RUN beats into the move (a long shot plays at 1x before it), and every shot ends on TAIL
    seconds at 1x however late its landing. Dancing never runs faster than CAP (any faster and people look sped up,
    not energetic); a close-up holds deeper and longer -- a hair flip in near-still motion."""
    fast, slow, hold, ease = 1.0, 0.35, 1.0, 0.4    # speed through the move, during the hold; hold and ease in beats
    CAP = 1.5
    RUN = 2.0                                       # beats of fast run-up into the move

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
        run = smooth((t - (p - ease - self.RUN * b)) / ease)                # 0 long before the move, 1 in its run
        down = smooth((t - (p - ease)) / ease)                               # 0 before the ease, 1 at the landing
        up = smooth((t - (p + hold)) / ease)                                 # 0 during the hold, 1 after it
        v = 1 + (fast - 1) * run * (1 - down)                                # 1x, then fast through the move
        return v + (slow - v) * down * (1 - up)                              # slow on the landing, then 1x

    def speed(self, t, out_len, ctx):
        v = np.minimum(self.curve(t, ctx), ctx.params.get("cap", self.CAP))
        return settle_tail(v, t, out_len, ctx)

    paced = True                                    # the shot's base pace (params "pace") scales the curve
    even = False                                    # (Ramp) motion stays evenly spaced through a speed change

    def _index(self, out_len, ctx):
        """(output times, the source frame each plays -- fractional between two, whole on a 1x stretch -- and the
        speed there), per output frame."""
        t = np.arange(int(round(out_len * ctx.fps)) + 1) / ctx.fps
        s = self.speed(t, out_len, ctx)
        p = ctx.params.get("pace", 1.0) if self.paced else 1.0
        if p != 1.0:                                # dancing a touch quicker throughout, never past the cap
            s = np.minimum(s * p, max(ctx.params.get("cap", self.CAP), p))
        idx = np.concatenate([[0], np.cumsum(snap_steps((s[1:] + s[:-1]) / 2))])   # in source frames
        whole = np.rint(idx)
        return t, np.where(np.abs(idx - whole) < 1e-6, whole, idx), s   # (1x: whole frames, no rounding dust)

    def _map(self, out_len, ctx):
        """(output times, the source time each plays, the speed there)."""
        t, idx, s = self._index(out_len, ctx)
        return t, idx / ctx.fps, s

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
        return list(self.stream(frames, ctx, n_out))

    def stream(self, source, ctx, n_out):
        """The curve's n_out output frames, one at a time, from an iterator of source frames (RGB arrays or Yuv):
        read as the curve reaches them, a handful held at once."""
        t, idx, s = self._index(n_out / ctx.fps, ctx)
        frames, flows = Window(source), (None, None)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        hold = self.even and self.shape(ctx).get("synth") == "hold"
        face = bool(ctx.params.get("face"))        # a face in the shot stays sharp: no frame averaging over it,
        for k in range(n_out):                     # and no flow-warp where the motion is too big to warp cleanly
            last = frames.need(int(idx[k]) + 1)    # (the source's last frame, or as far as it has been read)
            x = min(idx[k], last)
            whole = abs(x - round(x)) < 0.05      # (an even curve: only where the fast frame is a real one)
            if face and s[k] > 1.3 and (whole or not self.even):
                yield frames[int(round(x))] if round(x) <= last else frames[last]
            elif s[k] > 1.3 and k > 0 and not (face and self.even):   # fast: blend the frames this output frame
                                                                      # spans (motion blur)
                a, b = int(idx[k - 1]), int(x)
                yield _mean(frames[a:b + 1]) if b > a else frames[b]
            else:
                i, a = int(x), np.float32(x - int(x))
                if a < 0.05 or i >= last:          # a whole source frame (every frame of a 1x stretch): as it is
                    yield frames[i]
                else:                              # slow: flow-warp both neighbours to the in-between time
                    if flows[0] != i:
                        flows = (i, Flows(dis, frames[i], frames[i + 1]))
                    if hold and s[k] < 1.0:      # (reel.json ramps "synth": "hold") a slowed frame held, not made
                        yield frames[i]
                    elif face and flows[1].ghost() > GHOST:
                        yield frames[i] if a < 0.5 else frames[i + 1]   # warping it would ghost the face
                    else:
                        yield flows[1].between(frames[i], frames[i + 1], a)
            frames.drop(int(idx[k]))               # (the next frame's blur starts from here)


@register
class Montage(Speed):
    name = "montage"; fast, slow, hold, ease = 1.5, 0.5, 0.75, 0.25      # quick in, a short hold, quick out


@register
class Glide(Speed):
    """Gentle slow motion for a calm passage (an intro, a breakdown): the whole shot at GLIDE x, no ramp, no hold --
    optical-flow in-betweens, so it floats rather than stutters, and the face guard keeps faces sharp."""
    name = "glide"
    GLIDE = 0.8

    def speed(self, t, out_len, ctx):
        return np.full_like(t, ctx.params.get("glide", self.GLIDE), dtype=float)


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
        v = np.where((t > p - 0.6 * ctx.beat) & (t < p - 0.1 * ctx.beat), ctx.params.get("cap", 3.5), 1.0)
        return settle_tail(v, t, out_len, ctx)


@register
class FlashIn(Speed):
    name = "flashin"

    def speed(self, t, out_len, ctx):                         # rushes in fast, settles to normal
        v = np.minimum(1 + 2.5 * np.exp(-t / (0.6 * ctx.beat)), ctx.params.get("cap", 3.5))
        return settle_tail(v, t, out_len, ctx)


@register
class FlashOut(Speed):
    name = "flashout"

    velocity = True                                           # the one curve that ends fast: the out-half of a velocity
                                                              # edit, cut mid-rush into its swipe (never a shot's own end)

    def speed(self, t, out_len, ctx):                         # normal, then rushes out
        return np.minimum(1 + 2.5 * np.exp(-(out_len - t) / (0.6 * ctx.beat)), ctx.params.get("cap", 3.5))


@register
class Fit(Speed):
    """The editor's whole selection in its slot: quicker at the ends, easing to its slowest over the middle (a bell
    across the shot), scaled so the selection's every frame plays -- params "src_len": the selection's length."""
    name = "fit"
    paced = False                                             # the editor's selection plays whole, as chosen
    dip, width = 0.5, 0.35                                    # how much slower the middle is; the bell's width (of L)

    def shape(self, t, out_len):
        return 1 - self.dip * np.exp(-((t - out_len / 2) / (self.width * out_len)) ** 2)

    def speed(self, t, out_len, ctx):                        # scaled so the curve, its 1x tail included, reads src_len
        if ctx.params.get("even"):                           # (params "even": one steady pace across the slot -- a
            v = ctx.params.get("src_len", out_len) / max(out_len, 1e-6)   # pan along a row, every face its moment)
            return np.full_like(np.asarray(t, dtype=float), v)
        dense = np.linspace(0, out_len, 2001)
        w = settle_tail(np.zeros_like(dense), dense, out_len, ctx)          # the tail's weight
        k = (ctx.params.get("src_len", out_len) - np.trapezoid(w, dense)) / max(
            np.trapezoid(self.shape(dense, out_len) * (1 - w), dense), 1e-6)
        return settle_tail(self.shape(t, out_len) * max(k, 0.05), t, out_len, ctx)

    def source_length(self, out_len, ctx):
        return ctx.params.get("src_len", out_len) + 2 / ctx.fps


# -- ramps by what moves ----------------------------------------------------------------------------------------
# docs/research/speed-ramps.md. Times are in beats (a shot is cut on bars, so a speed change a whole number of
# beats from the landing falls on the music); speeds multiply the shot's pace. reel.json "ramps": {"<shape>": {...}}
# overrides any of it for a project; {"ramps": {"fps_floor": 25}} is read by every shape.
RAMPS = {
    # a person lowered into the water and lifted out: real time while they stand, quicker as they go back and
    # under, slowing as they are lifted so the face breaks the surface in slow motion (the peak: the emergence),
    # held through the first breath, then eased back to real time for what follows
    "dunk": dict(fast=2.0, slow=0.5, up=1.0, run=1.0, down=1.0, lead=1.0, hold=3.0, out=1.0, snap="bar"),
    # an emergence, a reveal of a person, a reaction: no speed-up, real time easing into slow motion on the moment
    "rise": dict(fast=1.0, slow=0.5, up=0.0, run=0.0, down=2.0, lead=1.0, hold=4.0, out=2.0, snap="bar"),
    # a drone push-in, fly-over or reveal: quick through the travel, easing down to settle on the subject
    "reveal": dict(fast=2.0, down=4.0),
    # a held drone horizon as a time-lapse: its pace through the middle, eased from and to `edge` at the ends
    "drift": dict(edge=0.5, up=2.0, out=2.0),
    "none": {},
}
FPS_FLOOR = 25.0         # real frames a second a slowed stretch still shows: 50p slows to 0.5x, 25p not at all
MAX_FAST = 4.0
BAR = 4                  # beats in a bar


def slow_floor(src_fps, fps=50, keep=FPS_FLOOR):
    """The slowest a source of src_fps may play on an fps timeline and still show `keep` real frames a second:
    0.5 for 50p (every other frame an in-between of two real ones), 1.0 for 25p (which a 50 fps timeline already
    shows each frame of twice: any slower stutters, and an in-between of a frame and its own repeat is no
    in-between at all)."""
    real = min(float(src_fps or fps), float(fps))
    return float(min(1.0, max(keep / real, 0.0)))


def clean_speeds(src_fps, fps=50, most=MAX_FAST):
    """The constant speeds (1x and up) a source of src_fps plays at on an fps timeline with every frame the same
    length on screen, nearest frame, nothing synthesised: 25p at 50 fps -> [1, 2, 4]. (1.25x there shows its
    frames for two timeline frames and one in turn: judder on any camera move.)"""
    r = float(fps) / float(src_fps or fps)                   # timeline frames a source frame fills at 1x
    out = {k * r for k in range(1, int(most / r) + 1)}       # a whole number of source frames each timeline frame
    out |= {r / m for m in range(1, int(r) + 1)}             # or each source frame a whole number of timeline frames
    return sorted(round(v, 4) for v in out if 1.0 <= v <= most)


def _eased(t, knots):
    """The speed at times t of a curve through knots [(time, speed), ...]: eased from each knot to the next (a
    half cosine: the speed never jumps, and has no corner at either end), flat before the first and after the
    last."""
    t = np.asarray(t, float)
    v = np.full_like(t, knots[0][1], dtype=float)
    for (t0, v0), (t1, v1) in zip(knots, knots[1:]):
        if t1 - t0 < 1e-9:
            v = np.where(t >= t1, v1, v)
            continue
        x = np.clip((t - t0) / (t1 - t0), 0, 1)
        v = np.where(t >= t0, v0 + (v1 - v0) * (0.5 - 0.5 * np.cos(np.pi * x)), v)
    return v


class Ramp(Speed):
    """A speed ramp for one kind of motion: speeds joined by eased transitions (half cosines: the speed never
    jumps and never turns a corner), timed in beats round the shot's peak. See RAMPS for the shapes."""
    name = "none"
    paced = False                                   # the pace is inside the curve (its floor and cap hold after it)
    even = True

    def shape(self, ctx):
        """The shape's numbers: RAMPS, then reel.json "ramps" (ctx.params "ramps": shared keys, and the shape's
        own under its name)."""
        cfg = ctx.params.get("ramps") or {}
        d = dict(tail=TAIL, bar=BAR, fps_floor=FPS_FLOOR, max_fast=MAX_FAST, synth="flow", **RAMPS[self.name])
        d.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
        d.update(cfg.get(self.name) or {})
        return d

    def floor(self, ctx, d):
        return slow_floor(ctx.params.get("src_fps"), ctx.fps, d["fps_floor"])

    def land(self, out_len, ctx, d=None):
        """Where in the shot the action lands (output seconds): the shot's peak, moved to the bar line within a
        beat of it (snap "bar"), else to the nearest beat ("bar", "beat"), when the shot keeps room for the slow
        motion after it. Left where it is when the renderer had to move the source (params "peak_fixed")."""
        d = d or self.shape(ctx)
        p = ctx.peak if ctx.peak is not None else 0.6 * out_len
        snap, b = d.get("snap"), ctx.beat
        if not snap or ctx.params.get("peak_fixed") or not b:
            return float(p)
        lo = (d.get("lead", 0) + d.get("down", 0)) * b
        hi = out_len - d["tail"] - (d.get("out", 0) + 1) * b
        def ok(x):
            return lo - 1e-6 <= x <= hi + 1e-6
        bar = round(p / (d["bar"] * b)) * d["bar"] * b
        if snap == "bar" and abs(bar - p) <= b + 1e-6 and ok(bar):
            return float(bar)
        beat = round(p / b) * b
        return float(beat if ok(beat) else p)

    def knots(self, out_len, ctx, d):
        return [(0.0, 1.0)]

    def speed(self, t, out_len, ctx):
        d = self.shape(ctx)
        pace = float(ctx.params.get("pace", 1.0) or 1.0)
        v = _eased(t, self.knots(out_len, ctx, d)) * pace
        v = np.clip(v, min(self.floor(ctx, d), pace), max(d["max_fast"], pace))
        return self.settle(v, t, out_len, ctx, d, pace)

    def settle(self, v, t, out_len, ctx, d, pace):
        """The shot's last `tail` seconds at its pace (1x for footage of people), eased into."""
        ease = max(0.4 * ctx.beat, 1 / ctx.fps)
        x = np.clip((t - (out_len - d["tail"] - ease)) / ease, 0, 1)
        return v + (pace - v) * (0.5 - 0.5 * np.cos(np.pi * x))

    def source_peak(self, out_peak, ctx, out_len=None):
        """Source time under the landing (land(): the peak, on its bar line or beat) -- the renderer puts the
        shot's moment there. Any other output time: the source time under it."""
        L = out_len or max(out_peak * 2, 1.0)
        t, src, _ = self._map(L, ctx)
        at = self.land(L, ctx) if ctx.peak is not None and abs(out_peak - ctx.peak) < 1e-6 else out_peak
        return float(np.interp(at, t, src))


class _Slowing(Ramp):
    """Quicker into the action, slow motion on it: `up` beats easing to `fast`, `run` beats there, `down` beats
    easing to `slow`, which arrives `lead` beats before the landing and holds `hold` beats after it (less in a
    short shot), then `out` beats back to real time. fast 1: no speed-up (rise). A source too slow in frame rate
    to slow down (slow_floor) plays the landing at real time: quick, then settling -- the ramp still reads."""

    def knots(self, out_len, ctx, d):
        b = ctx.beat
        P = self.land(out_len, ctx, d)
        slow = max(d["slow"], self.floor(ctx, d))
        fast = min(max(d["fast"], 1.0), d["max_fast"])
        pre = [d["up"] * b, d["run"] * b, d["down"] * b, d["lead"] * b]
        if sum(pre) > P + 1e-6:                                # no room for the run-up (a short shot, an early
            fast = 1.0                                         # landing): no speed-up -- real time into the slow
        if fast <= 1.0:                                        # motion, not the whole ramp squeezed into a bar
            pre[0] = pre[1] = 0.0
        if sum(pre) > P:                                       # (still none: the ease down shorter)
            k = max(P, 0.0) / sum(pre)
            pre = [x * k for x in pre]
        up, run, down, lead = pre
        t0 = P - lead - down - run - up
        end = out_len - d["tail"]                              # back at real time by here
        out = min(d["out"] * b, max(end - P, 0.0))
        hold = min(d["hold"] * b, max(end - out - P, 0.0))
        ks = [(0.0, 1.0), (t0, 1.0)]
        if fast > 1.0:
            ks += [(t0 + up, fast), (t0 + up + run, fast)]
        ks += [(P - lead, slow), (P + hold, slow), (P + hold + out, 1.0)]
        return ks


@register
class Dunk(_Slowing):
    name = "dunk"


@register
class Rise(_Slowing):
    name = "rise"


@register
class Reveal(Ramp):
    """A camera move that arrives somewhere (a drone push-in, a fly-over, a reveal): `fast` times the shot's pace
    from the cut, easing over `down` beats to the pace as it reaches the peak, and staying there."""
    name = "reveal"

    def knots(self, out_len, ctx, d):
        P = self.land(out_len, ctx, d)
        down = min(d["down"] * ctx.beat, max(P, 0.0))
        return [(0.0, d["fast"]), (P - down, d["fast"]), (P, 1.0)]


@register
class Drift(Ramp):
    """A held horizon as a time-lapse: the shot's pace through the middle, eased up from `edge` of it over `up`
    beats at the start and back down to it over `out` beats at the end (a time-lapse that starts and stops at
    full speed reads as a cut-in loop; eased, as time gathering and letting go)."""
    name = "drift"

    def knots(self, out_len, ctx, d):
        b, e = ctx.beat, d["edge"]
        up, out = (min(x * b, out_len / 3) for x in (d["up"], d["out"]))
        return [(0.0, e), (up, 1.0), (out_len - out, 1.0), (out_len, e)]

    def settle(self, v, t, out_len, ctx, d, pace):
        return v                                               # (its own eased end; a time-lapse has no 1x tail)


@register
class NoRamp(Ramp):
    """No ramp: the shot at its pace throughout (reel.json "reference_ramp": "none")."""
    name = "none"

    def knots(self, out_len, ctx, d):
        return [(0.0, 1.0)]


RAMP_FX = tuple(RAMPS)                                         # the registered names (reel.render, reel.plan)


def ramp_window(name, out_len, peak, beat, fps=50, params=None):
    """(seconds of source a ramp reads before the shot's moment, seconds after it, where in the shot the moment
    plays) for a shot of out_len seconds whose peak is `peak` seconds in -- what a planner checks the footage over
    (a camera nudge just before a dunk) instead of the shot's own length either side. params: the renderer's
    ("ramps", "src_fps", "pace")."""
    from .base import EFFECTS, Context
    e = EFFECTS[name]()
    ctx = Context(fps=fps, beat=beat, peak=peak, params=dict(params or {}))
    if not isinstance(e, Speed):
        return peak, out_len - peak, peak
    pre = e.source_peak(peak, ctx, out_len)
    at = e.land(out_len, ctx) if isinstance(e, Ramp) else peak
    return pre, e.source_length(out_len, ctx) - 2 / fps - pre, at
