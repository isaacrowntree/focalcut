"""Effect modules: one interface, a shared context, a registry.

An Effect turns a shot's frames into new frames:
    class MyEffect(Effect):
        name = "my"; needs_masks = False
        def apply(self, frames, ctx): return frames
Frames are a list of HxWx3 uint8 RGB arrays at ctx.fps. Effects that change timing (speed curves) consume more or
less source than they output: they say how much via source_length() and where the peak sits via source_peak().

A Transition blends the tail of shot A into the head of shot B over its own duration (frames from both are given).
Everything that needs the performers' masks asks ctx.masks(frames) -- SAM 3 runs once per shot and is shared.
"""
import numpy as np

EFFECTS, TRANSITIONS = {}, {}


def register(cls):
    (TRANSITIONS if issubclass(cls, Transition) else EFFECTS)[cls.name] = cls
    return cls


class Context:
    def __init__(self, fps=50, beat=0.6, peak=None, params=None, seed=0):
        self.fps, self.beat, self.peak = fps, beat, peak       # peak: seconds into the shot where the move peaks
        self.params = params or {}
        self.rng = np.random.default_rng(seed)
        self._masks = None
        self._mask_frames = None

    def masks(self, frames):
        """Performer masks (float 0..1) for these frames; computed with SAM 3 once and cached."""
        if self._masks is None or self._mask_frames != len(frames):
            from .masks import performer_masks
            self._masks = performer_masks(np.stack(frames), group=bool(self.params.get("group")))
            self._mask_frames = len(frames)
        return self._masks

    def peak_frame(self, n):
        return int(np.clip(round((self.peak if self.peak is not None else n / self.fps / 2) * self.fps), 0, n - 1))


class Effect:
    name = "effect"
    needs_masks = False

    def __init__(self, **params):
        self.p = params

    def source_length(self, out_len, ctx):
        return out_len                                         # seconds of source needed for out_len of output

    def source_peak(self, out_peak, ctx):
        return out_peak                                        # where the peak sits in the source window

    def apply(self, frames, ctx):
        raise NotImplementedError


class Transition:
    name = "transition"
    duration_beats = 0.5                                       # length of the blend, in beats

    def __init__(self, **params):
        self.p = params

    def frames(self, ctx):
        return max(2, int(round(self.duration_beats * ctx.beat * ctx.fps)))

    def apply(self, a_tail, b_head, ctx):
        raise NotImplementedError


def chain(frames, names, ctx):
    """Apply effects by name, in order (timing effects should come first)."""
    for n in names:
        frames = EFFECTS[n]().apply(frames, ctx)
    return frames


# helpers shared by several modules
def dim(f, amount=0.42, sat=0.25):
    g = f.mean(-1, keepdims=True)
    return (g + (f - g) * sat) * amount


def to_u8(f):
    return np.clip(f, 0, 255).astype(np.uint8)
