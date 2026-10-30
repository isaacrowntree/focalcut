"""Subject masks with SAM 3 video: track every "person", keep the subject -- the salient, near, large person the shot
is about (focalcut.reel.subject_engine's rule) and whoever dances with them: their partner on a social floor, or the
rest of the line on a stage (group=True) -- union them per frame. The edge comes from BiRefNet's matte on a crop
around the subject (SAM 3's masks are coarse once scaled up), the frames between tracked ones are blended, not held,
and a little memory keeps the edge from shimmering."""
import os

import cv2
import numpy as np
import torch
from PIL import Image

if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
    torch.backends.cudnn.enabled = False          # some CUDA installs ship broken cuDNN engines
STEP = 2                                          # track every 2nd frame; blend masks for the frame between
MAX_SIDE = 960                                    # SAM 3 runs at this size; masks are scaled back up
MAX_OBJECTS = 6                                   # crowds: only the most confident people are tracked
_model = _proc = None


def performer_masks(frames, prompt="person", group=False):
    global _model, _proc
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    if _model is None:
        from transformers import Sam3VideoConfig
        cfg = Sam3VideoConfig.from_pretrained("facebook/sam3")
        cfg.max_num_objects, cfg.new_det_thresh = MAX_OBJECTS, 0.8
        _model = Sam3VideoModel.from_pretrained("facebook/sam3", config=cfg, torch_dtype=torch.bfloat16).to("cuda").eval()
        _proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    H0, W0 = frames.shape[1:3]
    sc = min(1.0, MAX_SIDE / max(H0, W0))
    small = [Image.fromarray(f).resize((int(W0 * sc), int(H0 * sc))) for f in frames[::STEP]]
    sess = _proc.init_video_session(video=small, inference_device="cuda",
                                    processing_device="cpu", video_storage_device="cpu", dtype=torch.bfloat16)
    _proc.add_text_prompt(sess, prompt)
    per = {}
    with torch.inference_mode():
        for out in _model.propagate_in_video_iterator(inference_session=sess):
            r = _proc.postprocess_outputs(sess, out)
            per[out.frame_idx] = {int(o): m.cpu().numpy().astype(bool) for o, m in zip(r["object_ids"], r["masks"])}
    H, W = small[0].height, small[0].width
    top = pick_subject(per, frames, H, W, group)
    last = max(per) if per else 0

    def union(k):
        objs, m = per.get(min(k, last), {}), np.zeros((H, W), bool)
        for o in top:
            if o in objs:
                m |= objs[o]
        return m.astype(np.float32)
    masks = np.zeros((len(frames), H0, W0), np.float32)
    prev = None
    for i in range(len(frames)):
        k, r = divmod(i, STEP)                              # between tracked frames: blend, don't hold
        m = union(k) if r == 0 or k >= last else (1 - r / STEP) * union(k) + r / STEP * union(k + 1)
        a = refine(frames[i], cv2.resize(m, (W0, H0), interpolation=cv2.INTER_LINEAR))
        prev = a if prev is None else 0.7 * a + 0.3 * prev  # a little memory against edge shimmer
        masks[i] = prev
    del sess
    torch.cuda.empty_cache()
    return masks


def refine(frame, coarse, matte=None):
    """Clean edges: BiRefNet's matte on a crop around the subject (so it sees them at high resolution), kept to the
    subject by SAM 3's mask grown a little; SAM's eroded core fills in anyone the matte half-misses (a partner)."""
    if matte is None:
        from ..reel.subject_engine import matte
    H, W = coarse.shape
    ys, xs = np.nonzero(coarse > 0.5)
    if not len(xs):
        return cv2.GaussianBlur(coarse, (0, 0), 2.5)
    pad = int(0.08 * max(H, W))
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad)
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = matte(frame[y0:y1, x0:x1])
    r = max(3, int(0.012 * max(H, W)))
    hard = (coarse > 0.5).astype(np.uint8)

    def ell(n):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    gate = cv2.GaussianBlur(cv2.dilate(hard, ell(2 * r)).astype(np.float32), (0, 0), r)
    core = cv2.GaussianBlur(cv2.erode(hard, ell(2 * r)).astype(np.float32), (0, 0), r)
    return np.maximum(out * gate, core)


def pick_subject(per, frames, H, W, group):
    """Track ids of the subject (subject_engine's rule: salient, near, large -- not merely sharp, since dancing blurs
    and bystanders stand still), scored on a few frames across the shot."""
    from ..reel import subject_engine as E
    if not per:
        return []
    keys = sorted(per)
    tot = {}
    probe = [keys[int(len(keys) * q)] for q in (0.2, 0.5, 0.8)]
    for fi in probe:
        objs = per[fi]
        if not objs:
            continue
        rgb = cv2.resize(frames[min(fi * STEP, len(frames) - 1)], (W, H), interpolation=cv2.INTER_AREA)
        sal, dep = E.maps(rgb)
        ids = list(objs)
        sc = E.score(np.stack([objs[o] for o in ids]), sal, dep)
        for o, v in zip(ids, sc):
            tot[o] = tot.get(o, 0) + v
    return E.keep_subject({o: tot[o] / len(probe) for o in tot}, group)   # absent from a probe counts as zero there

