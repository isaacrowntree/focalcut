"""Fast engine: sparse SAM 3 *image* detection + IoU association.

The video tracker's memory bank makes time and VRAM scale with frame count (minutes
per clip, ~12 GB). When the focal subject is continuously visible — as on a dance
floor where the camera simply pans to the next couple — we don't need per-frame mask
propagation. Sampling a frame every couple of seconds, running SAM 3's image detector
(flat ~3 GB, batchable, ~7 fps on a 4070 Ti), and stitching ids across samples by box
IoU gives the same "don't re-cut on a back-turn" permanence in a few seconds per clip.
"""
from __future__ import annotations
import numpy as np
from .track import Detection


class Sam3ImageDetector:
    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda", dtype=None):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.model = Sam3Model.from_pretrained(model_id).to(device, dtype=self.dtype).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def detect(self, frames: np.ndarray, prompt: str = "person",
               min_score: float = 0.5, chunk: int = 4) -> list[list[Detection]]:
        """Return per-frame [Detection,...] (obj_id is a placeholder -1 until associated)."""
        torch = self.torch
        H, W = frames.shape[1], frames.shape[2]
        out_per_frame: list[list[Detection]] = []
        with torch.inference_mode():
            for i in range(0, len(frames), chunk):
                imgs = [frames[j] for j in range(i, min(i + chunk, len(frames)))]
                inp = self.processor(images=imgs, text=["" + prompt] * len(imgs),
                                     return_tensors="pt").to(self.device)
                out = self.model(**inp)
                res = self.processor.post_process_object_detection(
                    out, threshold=min_score, target_sizes=[(H, W)] * len(imgs))
                for r in res:
                    boxes = r["boxes"].float().cpu().numpy()
                    scores = r["scores"].float().cpu().numpy()
                    dets = []
                    for b, s in zip(boxes, scores):
                        x0, y0, x1, y1 = [float(v) for v in b]
                        dets.append(Detection(-1, (x0, y0, x1, y1),
                                              abs((x1 - x0) * (y1 - y0)), float(s)))
                    out_per_frame.append(dets)
        return out_per_frame


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def associate(frames_dets: list[list[Detection]], iou_thresh: float = 0.3) -> dict[int, list[Detection]]:
    """Greedy frame-to-frame IoU matching -> persistent obj ids.

    A subject staying in frame keeps its id (continuous visibility => stable focal id);
    when the camera moves to new people, their boxes don't overlap the old ones, so they
    get fresh ids -> the focal set turns over -> segment.py emits a cut.
    """
    per_frame: dict[int, list[Detection]] = {}
    prev: list[Detection] = []
    next_id = 0
    for f, dets in enumerate(frames_dets):
        assigned = []
        used_prev = set()
        # match each new det to the best unused previous det
        for d in sorted(dets, key=lambda x: -x.area):
            best, best_iou = None, iou_thresh
            for k, p in enumerate(prev):
                if k in used_prev:
                    continue
                v = _iou(d.box, p.box)
                if v >= best_iou:
                    best, best_iou = k, v
            if best is not None:
                used_prev.add(best)
                assigned.append(Detection(prev[best].obj_id, d.box, d.area, d.score))
            else:
                assigned.append(Detection(next_id, d.box, d.area, d.score))
                next_id += 1
        per_frame[f] = assigned
        prev = assigned
    return per_frame
