"""SAM 3 concept-prompted video tracking.

Wraps transformers' Sam3VideoModel: a text/concept prompt detects and tracks every
matching instance through the clip, holding stable object ids across occlusion and
re-appearance (object permanence). We keep only lightweight per-detection geometry
(id, box, area, score) — never the full masks — so memory stays flat over long clips.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    obj_id: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels
    area: float                             # mask area in pixels (precise)
    score: float


class Sam3Tracker:
    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda", dtype=None):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.model = Sam3VideoModel.from_pretrained(model_id).to(device, dtype=self.dtype).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)

    def track(self, frames: np.ndarray, prompt: str = "person",
              min_score: float = 0.0) -> dict[int, list[Detection]]:
        """Return {frame_idx: [Detection, ...]} for every frame of `frames` (N,H,W,3)."""
        torch = self.torch
        frame_list = [frames[i] for i in range(len(frames))]
        sess = self.processor.init_video_session(
            video=frame_list, inference_device=self.device,
            processing_device="cpu", video_storage_device="cpu", dtype=self.dtype)
        sess = self.processor.add_text_prompt(inference_session=sess, text=prompt)

        per_frame: dict[int, list[Detection]] = {}
        with torch.inference_mode():
            for out in self.model.propagate_in_video_iterator(
                    inference_session=sess, max_frame_num_to_track=len(frame_list)):
                p = self.processor.postprocess_outputs(sess, out)
                per_frame[out.frame_idx] = self._to_detections(p, min_score)
        return per_frame

    @staticmethod
    def _to_detections(p: dict, min_score: float) -> list[Detection]:
        ids = p.get("object_ids")
        if ids is None or len(ids) == 0:
            return []
        ids = [int(x) for x in ids.tolist()]
        scores = [float(x) for x in p["scores"].tolist()]
        boxes = p["boxes"].float().cpu().numpy()
        masks = p.get("masks")
        areas = (masks.float().sum(dim=(1, 2)).cpu().numpy()
                 if masks is not None else
                 np.abs((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        dets = []
        for j, oid in enumerate(ids):
            if scores[j] < min_score:
                continue
            dets.append(Detection(oid, tuple(boxes[j].tolist()), float(areas[j]), scores[j]))
        return dets
