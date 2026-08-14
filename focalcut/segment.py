"""Turn per-frame tracks into focal-subject segments.

Policy (single focal subject/couple at a time): at each frame the *focal set* is the
small group of large, central tracks the camera is actually framing — foreground
dancers, not the crowd behind them. A cut is declared when the focal set turns over
(the camera has settled on different people) and stays turned over for `min_dwell`
seconds. Object permanence in the tracker keeps ids stable through spins / brief
occlusion, so turning your back or a passing couple does NOT trigger a false cut.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import Detection


@dataclass
class Segment:
    start: float        # seconds
    end: float          # seconds
    obj_ids: list[int]  # focal track ids dominating this segment

    @property
    def duration(self) -> float:
        return self.end - self.start


def _focal_set(dets: list[Detection], frame_wh: tuple[int, int],
               area_frac: float, salience_ratio: float) -> set[int]:
    """The foreground dancers in one frame: large + central, near the top of salience."""
    if not dets:
        return set()
    W, H = frame_wh
    cx, cy = W / 2, H / 2
    diag = (W ** 2 + H ** 2) ** 0.5
    scored = []
    for d in dets:
        if d.area < area_frac * W * H:           # drop background/small dancers
            continue
        bx = (d.box[0] + d.box[2]) / 2
        by = (d.box[1] + d.box[3]) / 2
        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / diag
        center_w = float(np.exp(-(dist ** 2) / (2 * 0.35 ** 2)))  # central -> ~1
        scored.append((d.area * center_w, d.obj_id))
    if not scored:
        return set()
    top = max(s for s, _ in scored)
    return {oid for s, oid in scored if s >= salience_ratio * top}


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
          + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter)
    return inter / ua if ua > 0 else 0.0


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def segment(per_frame: dict[int, list[Detection]], fps: float, frame_wh: tuple[int, int],
            area_frac: float = 0.015, salience_ratio: float = 0.45,
            turnover: float = 0.2, min_dwell: float = 0.6,
            min_segment: float = 1.5, t_offset: float = 0.0) -> list[Segment]:
    """Detect focal-subject segments. `t_offset` shifts times into source timecode."""
    frames = sorted(per_frame)
    if not frames:
        return []
    focal = {f: _focal_set(per_frame[f], frame_wh, area_frac, salience_ratio) for f in frames}

    dwell_frames = max(1, round(min_dwell * fps))
    segs: list[Segment] = []
    cur_ids = focal[frames[0]]
    seg_start = frames[0]
    low_run = 0          # consecutive frames whose focal set disagrees with cur_ids

    for f in frames[1:]:
        fs = focal[f]
        if fs and _jaccard(fs, cur_ids) < turnover:
            low_run += 1
            if low_run >= dwell_frames:
                # cut: the camera has been on new people long enough
                cut_at = f - dwell_frames + 1
                segs.append(Segment(seg_start / fps + t_offset, cut_at / fps + t_offset, sorted(cur_ids)))
                seg_start = cut_at
                cur_ids = fs
                low_run = 0
        else:
            if fs:
                cur_ids = cur_ids | fs if _jaccard(fs, cur_ids) >= turnover else fs
            low_run = 0
    segs.append(Segment(seg_start / fps + t_offset, (frames[-1] + 1) / fps + t_offset, sorted(cur_ids)))

    return _merge_short(segs, min_segment)


def _chroma_hist(crop: np.ndarray, bins: int = 6):
    """Flare-robust appearance descriptor: 2D chromaticity histogram of the focal crop.

    Pure-white (flare) and near-black pixels carry no colour and are excluded; if too few
    coloured pixels remain (the crop is washed out), returns None — which the segmenter
    treats as a 'degraded, coast' frame rather than evidence of a new dancer.
    """
    c = crop.reshape(-1, 3).astype(np.float32)
    s = c.sum(1)
    s[s == 0] = 1.0
    bright = c.mean(1) / 255.0
    mask = (bright > 0.12) & (bright < 0.85)
    if mask.sum() < 25:
        return None
    rg = c[mask, 0] / s[mask]
    gg = c[mask, 1] / s[mask]
    h, _, _ = np.histogram2d(rg, gg, bins=bins, range=[[0, 1], [0, 1]])
    h = h / (h.sum() + 1e-9)
    return h.ravel()


def _hist_sim(a, b) -> float:
    return float(np.minimum(a, b).sum())


def _focal(dets: list[Detection], W: int, H: int, area_frac: float):
    """Largest, most-central detection above the area floor (the couple the camera frames)."""
    cx, cy = W / 2, H / 2
    diag = (W ** 2 + H ** 2) ** 0.5
    best, best_s = None, -1.0
    for d in dets:
        if d.area < area_frac * W * H:
            continue
        bx = (d.box[0] + d.box[2]) / 2
        by = (d.box[1] + d.box[3]) / 2
        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / diag
        s = d.area * float(np.exp(-(dist ** 2) / (2 * 0.35 ** 2)))
        if s > best_s:
            best, best_s = d, s
    return best


def segment_focal(per_frame: dict[int, list[Detection]], frames: np.ndarray, times: list[float],
                  frame_wh: tuple[int, int], area_frac: float = 0.02, min_score: float = 0.5,
                  sim_thresh: float = 0.45, iou_keep: float = 0.2, min_dwell: float = 3.0,
                  min_segment: float = 4.0) -> list[Segment]:
    """Track ONE focal subject by appearance, coasting through flare/occlusion.

    `times[f]` is the real timestamp (seconds) of frame index `f`, so this works on
    irregularly-spaced keyframes. A cut is emitted only on *positive, sustained* evidence
    of a different focal subject — a confident frame whose focal crop is both spatially
    disjoint from and colour-different to the one we're following. Degraded frames (washed
    out, no clear focal) coast.
    """
    W, H = frame_wh
    order = sorted(per_frame)
    if not order:
        return []
    if len(order) > 1:
        diffs = sorted(times[order[i + 1]] - times[order[i]] for i in range(len(order) - 1))
        dt = diffs[len(diffs) // 2] or 0.5          # median frame spacing
    else:
        dt = 0.5
    dwell = max(1, round(min_dwell / dt))

    segs: list[Segment] = []
    seg_start = order[0]
    cur_hist = None          # appearance of the subject we're currently following
    cur_box = None
    challenge = 0            # consecutive confident frames disagreeing with current
    challenge_start = order[0]

    for f in order:
        foc = _focal(per_frame[f], W, H, area_frac)
        hist = None
        if foc is not None and foc.score >= min_score:
            x0, y0, x1, y1 = [int(max(0, v)) for v in foc.box]
            crop = frames[f][y0:y1, x0:x1]
            if crop.size:
                hist = _chroma_hist(crop)

        if hist is None:                 # degraded / no clear focal -> coast
            continue
        if cur_hist is None:             # first confident frame establishes the subject
            cur_hist, cur_box = hist, foc.box
            challenge = 0
            continue

        same = (_hist_sim(hist, cur_hist) >= sim_thresh) or (_iou(foc.box, cur_box) >= iou_keep)
        if same:
            cur_hist = 0.7 * cur_hist + 0.3 * hist     # slow EMA so the subject can drift
            cur_box = foc.box
            challenge = 0
        else:
            if challenge == 0:
                challenge_start = f                     # first frame of the new subject
            challenge += 1
            if challenge >= dwell:                      # sustained different subject -> cut
                segs.append(Segment(times[seg_start], times[challenge_start], []))
                seg_start = challenge_start
                cur_hist, cur_box, challenge = hist, foc.box, 0

    segs.append(Segment(times[seg_start], times[order[-1]] + dt, []))
    return _merge_short(segs, min_segment)


def _merge_short(segs: list[Segment], min_segment: float) -> list[Segment]:
    """Fold sub-threshold shards into the neighbour they share the most ids with."""
    if not segs:
        return segs
    out = [segs[0]]
    for s in segs[1:]:
        if s.duration < min_segment:
            prev = out[-1]
            prev.end = s.end
            prev.obj_ids = sorted(set(prev.obj_ids) | set(s.obj_ids))
        else:
            out.append(s)
    # a leading shard may survive; clean one more pass
    merged = [out[0]]
    for s in out[1:]:
        if merged[-1].duration < min_segment:
            merged[-1].end = s.end
            merged[-1].obj_ids = sorted(set(merged[-1].obj_ids) | set(s.obj_ids))
        else:
            merged.append(s)
    return merged
