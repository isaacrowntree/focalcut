"""Effect modules (CapCut-style), each applicable on its own: importing registers every effect and transition.

    from focalcut.fx import EFFECTS, TRANSITIONS, Context, chain
    frames = chain(frames, ["hero", "neon"], Context(fps=50, beat=0.6, peak=1.2))
"""
from . import camera, carnival, edm, glitch, speed, subject, transitions
from .base import EFFECTS, TRANSITIONS, Context, Effect, Transition, chain, register
