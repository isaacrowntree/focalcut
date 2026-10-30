"""Who the shot is about: SAM 3 finds every person (exact masks), BiRefNet says what stands out as the foreground
subject (salient-object segmentation), Depth Anything V2 says who is nearest the camera. The subject is the
salient, near, large person plus anyone nearly as strong -- the couple, or a line of performers.

Focus alone picks the wrong people on stage footage: dancers in motion blur read as soft and still bystanders as
sharp. Saliency + depth does not have that failure.
"""
from __future__ import annotations

import os

import numpy as np

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
KEEP = 0.45                                       # someone counts as subject at this fraction of the best score
SAL_ID, DEPTH_ID, SAM_ID = "ZhengPeng7/BiRefNet", "depth-anything/Depth-Anything-V2-Small-hf", "facebook/sam3"
_m = {}


def _models(sam=True):
    """Loaded on first use; the effect renderer only needs the saliency and depth models (it has SAM 3 video)."""
    import torch
    from transformers import AutoModelForImageSegmentation, Sam3Model, Sam3Processor, pipeline
    if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
    if "bir" not in _m:
        _m["bir"] = AutoModelForImageSegmentation.from_pretrained(SAL_ID, trust_remote_code=True).to("cuda").eval().half()
        _m["depth"] = pipeline("depth-estimation", model=DEPTH_ID, device=0)
    if sam and "sam" not in _m:
        _m["proc"] = Sam3Processor.from_pretrained(SAM_ID)
        _m["sam"] = Sam3Model.from_pretrained(SAM_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
    return _m


def maps(rgb):
    """Saliency (0..1) and nearness (0..1, 1 = closest) maps at the image's size."""
    import cv2
    import torch
    from PIL import Image
    m = _models(sam=False)
    h, w = rgb.shape[:2]
    x = (cv2.resize(rgb, (1024, 1024)).astype(np.float32) / 255 - MEAN) / STD
    with torch.inference_mode():
        sal = torch.sigmoid(m["bir"](torch.from_numpy(x.transpose(2, 0, 1)[None]).half().cuda())[-1]).float()
        dep = np.asarray(m["depth"](Image.fromarray(rgb))["predicted_depth"], np.float32).squeeze()
    sal = cv2.resize(sal.cpu().numpy()[0, 0], (w, h))
    dep = cv2.resize(dep, (w, h)); dep = (dep - dep.min()) / (np.ptp(dep) + 1e-6)
    return sal, dep


def matte(rgb):
    """BiRefNet's foreground matte (0..1) at the image's size: clean, hair-level edges."""
    import cv2
    import torch
    m = _models(sam=False)
    h, w = rgb.shape[:2]
    x = (cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
    with torch.inference_mode():
        a = torch.sigmoid(m["bir"](torch.from_numpy(x.transpose(2, 0, 1)[None]).half().cuda())[-1]).float()
    return cv2.resize(a.cpu().numpy()[0, 0], (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def people(rgb, threshold=0.4):
    """SAM 3 person instances: boolean masks (n, h, w) at the image's size."""
    import torch
    from PIL import Image
    m = _models()
    h, w = rgb.shape[:2]
    with torch.inference_mode():
        inp = m["proc"](images=Image.fromarray(rgb), text="person", return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        r = m["proc"].post_process_instance_segmentation(m["sam"](**inp), threshold=threshold, mask_threshold=0.5,
                                                         target_sizes=[(h, w)])[0]
    return r["masks"].cpu().numpy().astype(bool)


def score(masks, sal, dep):
    """Subject score per mask: salient, near, large."""
    if not len(masks):
        return np.zeros(0)
    area = np.array([m.mean() for m in masks])
    S = np.array([sal[m].mean() if m.any() else 0 for m in masks])
    D = np.array([np.median(dep[m]) if m.any() else 0 for m in masks])
    return (0.25 + S) * (D / (D.max() + 1e-6)) ** 2 * np.sqrt(area / (area.max() + 1e-9))


def choose(sc):
    """Indices of the subject among the scored masks."""
    if not len(sc):
        return []
    best = sc.max()
    return [int(k) for k in np.argsort(-sc) if sc[k] >= KEEP * best][:6]


def keep_subject(avg, group):
    """From track id -> average subject score: the best, plus partners (2 in all) or line-mates (up to 6). A partner
    or a stage line-mate is often darker or half hidden, so they are kept at a lower bar than a stranger."""
    if not avg:
        return []
    best = max(avg.values())
    order = sorted((o for o in avg if avg[o] >= (0.2 if group else 0.3) * best), key=lambda o: -avg[o])
    return order[:6 if group else 2]


def describe(masks, sc):
    """The subject among scored masks: union box and edge contact as fractions, member count, strength, mask."""
    keep = choose(sc)
    sub = np.any(masks[keep], 0)
    ys, xs = np.nonzero(sub)
    h, w = sub.shape
    pad = 2
    return dict(box=(xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h),
                touch=(xs.min() < pad, ys.min() < pad, xs.max() >= w - pad, ys.max() >= h - pad),
                n=len(keep), strength=float(sc[keep].sum()), mask=sub)


def subject(rgb):
    """Everything about one frame's subject (see describe), or None if nobody is there."""
    masks = people(rgb)
    if not len(masks):
        return None
    sal, dep = maps(rgb)
    return describe(masks, score(masks, sal, dep))
