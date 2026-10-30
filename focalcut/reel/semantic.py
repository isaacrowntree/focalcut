"""What a shot shows, from the subject's crop: SigLIP 2 zero-shot on every keyframe of the subject track.
  hips  hips and butt working -- shakes, rolls, the view from behind that animations and shines are about
  face  a face with expression, close enough to read
against wide / empty / blurry / walking shots. Writes <reel>/scores/<name>_semantic.npz (t, hips, face).
Resumable; only sources whose subject track exists.
"""
from __future__ import annotations

import os
import queue
import threading

import numpy as np

from .subject import by_score, keyframes

MODEL = "google/siglip2-base-patch16-512"
GROUPS = {
    "hips": ["a dancer shaking her hips, seen from behind", "a close-up of a dancer's hips and butt while dancing",
             "a woman twerking", "a dancer rolling her hips", "a dancer's body from behind, hips swaying"],
    "face": ["a close-up of a dancer's smiling face", "a dancer laughing", "a dancer with an intense, expressive face",
             "a dancer looking straight at the camera"],
    "none": ["a wide shot of a stage with small dancers far away", "people standing around talking", "a blurry photo",
             "a person walking", "an empty dance floor", "a dark photo"],
}


def crop(rgb, box):
    """The subject's box with a little margin, or None if it is too small to read."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box * [w, h, w, h]
    mx, my = 0.08 * (x1 - x0), 0.05 * (y1 - y0)
    c = rgb[int(max(0, y0 - my)):int(min(h, y1 + my)), int(max(0, x0 - mx)):int(min(w, x1 + mx))]
    return c if c.shape[0] >= 16 and c.shape[1] >= 16 else None


def run(project) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL).to(dev, dt).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    prompts = [p for g in GROUPS.values() for p in g]
    which = np.array([k for k, g in GROUPS.items() for _ in g])
    with torch.inference_mode():
        tin = proc(text=prompts, return_tensors="pt", padding="max_length", max_length=64).to(dev)
        t = model.get_text_features(**tin)
        txt = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1)
    for name, units in by_score(project.load("units.json")).items():
        out = project.path("scores", name + "_semantic.npz")
        sub = project.path("scores", name + "_subject.npz")
        if os.path.exists(out) or not os.path.exists(sub):
            continue
        s = np.load(sub)
        q = queue.Queue(maxsize=32)
        threading.Thread(target=keyframes, args=(project, units[0], [(u["t0"], u["t1"]) for u in units], q),
                         daemon=True).start()
        T, H, F = [], [], []
        while (item := q.get()) is not None:
            tt, rgb = item
            j = int(np.argmin(np.abs(s["t"] - tt)))
            T.append(tt)
            c = None if np.isnan(s["box"][j][0]) or abs(s["t"][j] - tt) > 0.3 else crop(rgb, s["box"][j])
            if c is None:
                H.append(0.0); F.append(0.0); continue
            with torch.inference_mode():
                inp = proc(images=[Image.fromarray(c)], return_tensors="pt").to(dev)
                inp["pixel_values"] = inp["pixel_values"].to(dt)
                im = model.get_image_features(**inp)
                im = torch.nn.functional.normalize(getattr(im, "pooler_output", im).float(), dim=-1)
                p = torch.softmax(100 * im @ txt.T, -1)[0].cpu().numpy()
            H.append(float(p[which == "hips"].sum())); F.append(float(p[which == "face"].sum()))
        np.savez_compressed(out, t=np.array(T), hips=np.array(H, np.float32), face=np.array(F, np.float32))
        p90 = lambda v: np.percentile(v, 90) if v else 0.0
        print(f"[reel] semantic {name}: {len(T)} frames, hips p90 {p90(H):.2f}, face p90 {p90(F):.2f}", flush=True)
