"""Cheap baseline: detect cuts by global frame difference (no SAM, no permanence).

This is the strawman focalcut should beat. It has no notion of object permanence, so a
dancer turning their back or a couple crossing the frame spikes the diff and produces a
false cut. Comparing its false-split rate to focalcut's is the headline benchmark result.

Usage:
    python -m bench.baselines.framediff <video> [--fps 6] [--thresh 0.35] > pred.json
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from focalcut import video as V


def framediff_cuts(path: str, fps: float = 6.0, long_side: int = 256,
                   thresh: float = 0.35, min_gap: float = 1.5) -> list[float]:
    info = V.probe(path)
    size = V.target_size(info, long_side)
    frames = V.decode(path, fps, size).astype(np.float32) / 255.0
    if len(frames) < 2:
        return []
    diffs = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2, 3))
    diffs = diffs / (diffs.max() + 1e-9)
    cuts, last = [], -1e9
    for i, d in enumerate(diffs):
        t = (i + 1) / fps
        if d >= thresh and t - last >= min_gap:
            cuts.append(round(t, 2))
            last = t
    return cuts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--thresh", type=float, default=0.35)
    a = ap.parse_args(argv)
    cuts = framediff_cuts(a.video, fps=a.fps, thresh=a.thresh)
    json.dump({"video": a.video, "cuts": cuts}, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
