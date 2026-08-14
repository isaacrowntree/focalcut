"""focalcut benchmark — score predicted focal cuts against hand-labeled ground truth.

This benchmarks the thing you actually care about: per-dancer cut quality. It treats
segmentation as boundary detection plus over/under-segmentation, and reports a
"manual-correction count" — the merges + splits an editor would still have to do —
which is the operational form of "we don't want to have to recut".

Ground-truth file (JSON), one per source clip:
    {
      "video": "C1776_1_1.mp4",
      "cuts": [12.4, 31.0, 58.7, ...]          # seconds where the focal dancer changes
    }
Optionally add "duration" (seconds) for over/under-seg rates.

Usage:
    python -m bench.eval --gt path/to/gt.json --pred path/to/pred.json [--tol 0.5]
where pred.json is {"cuts": [...]} or focalcut's segment list (see load_pred()).
"""
from __future__ import annotations

import argparse
import json


def load_pred(path: str) -> list[float]:
    with open(path) as fh:
        d = json.load(fh)
    if "cuts" in d:
        return sorted(float(c) for c in d["cuts"])
    # focalcut segment dump: [{start,end},...] -> interior boundaries
    segs = d.get("segments", d)
    starts = sorted(float(s["start"]) for s in segs)
    return starts[1:]  # drop the very first (clip start is not a cut)


def match(pred: list[float], gt: list[float], tol: float):
    """Greedy 1-1 match of predicted to GT boundaries within `tol` seconds."""
    used = [False] * len(gt)
    tp = 0
    for p in pred:
        best, bd = -1, tol + 1e-9
        for i, g in enumerate(gt):
            if not used[i] and abs(p - g) <= tol and abs(p - g) < bd:
                best, bd = i, abs(p - g)
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(pred) - tp          # predicted cuts with no GT match -> false splits
    fn = len(gt) - tp            # missed GT cuts            -> false merges
    return tp, fp, fn


def evaluate(pred: list[float], gt: list[float], tol: float, duration: float | None):
    tp, fp, fn = match(pred, gt, tol)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "tol_s": tol,
        "gt_cuts": len(gt), "pred_cuts": len(pred),
        "matched": tp, "false_splits": fp, "false_merges": fn,
        "precision": round(prec, 3), "recall": round(rec, 3), "boundary_f1": round(f1, 3),
        # the "no-recut" number: every false split or merge is one manual fix
        "manual_corrections": fp + fn,
    }
    if duration:
        mins = duration / 60.0
        out["false_splits_per_min"] = round(fp / mins, 2)
        out["false_merges_per_min"] = round(fn / mins, 2)
        out["manual_corrections_per_min"] = round((fp + fn) / mins, 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tol", type=float, default=0.5, help="boundary match tolerance (s)")
    a = ap.parse_args(argv)
    with open(a.gt) as fh:
        gt = json.load(fh)
    res = evaluate(load_pred(a.pred), sorted(float(c) for c in gt["cuts"]),
                   a.tol, gt.get("duration"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
