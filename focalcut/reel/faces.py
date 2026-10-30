"""Second pass on the shortlisted moments: a full-resolution frame, face crops from the pose keypoints, and a
facial-expression score from SigLIP 2 zero-shot (joy / laughter / excitement / intensity against neutral / turned
away / blurry). Adds "face" (0-1) to every moment in moments.json and saves thumbs/<unit>_<t>.jpg.
Scores are cached in faces_cache.json by unit and time, so re-planning never re-extracts frames.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MODEL = "google/siglip2-base-patch16-512"
POS = ["a photo of a person smiling happily", "a photo of a person laughing", "a photo of an excited, joyful face",
       "a photo of a dancer with a passionate, intense expression"]
NEG = ["a photo of a person with a neutral, bored expression", "a photo of the back of a person's head",
       "a blurry photo of a face", "a photo of a person looking down"]
_m = None


def model():
    global _m
    if _m is None:
        import torch
        from transformers import AutoModel, AutoProcessor
        if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
            torch.backends.cudnn.enabled = False
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        mdl = AutoModel.from_pretrained(MODEL).to(dev, dt).eval()
        proc = AutoProcessor.from_pretrained(MODEL)
        with torch.inference_mode():
            tin = proc(text=POS + NEG, return_tensors="pt", padding="max_length", max_length=64).to(dev)
            t = mdl.get_text_features(**tin)
            txt = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1)
        _m = dict(model=mdl, proc=proc, txt=txt, dev=dev, dt=dt)
    return _m


def grab(folder, unit, t):
    """Full-resolution RGB frame at t (camera rotation and any unit rotation applied)."""
    from PIL import Image
    vf = ["-vf", unit["rotate"]] if unit.get("rotate") else []
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", os.path.join(folder, unit["source"]), *vf,
                        "-frames:v", "1", "-f", "image2", "-c:v", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(b)).convert("RGB")


def face_boxes(kp, scale, W, H):
    """Square face crops from nose/eyes/ears (analysis-res keypoints -> full-res pixels)."""
    out = []
    for p in kp:
        if np.isnan(p[0, 0]) or p[0, 2] < 0.5:
            continue                               # no confident nose: face not visible
        pts = [p[i, :2] for i in range(5) if p[i, 2] >= 0.4]
        if len(pts) < 3:
            continue
        pts = np.array(pts) * scale
        c = pts.mean(0)
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 8 * scale)
        s = span * 2.6
        x0, y0 = max(0, c[0] - s / 2), max(0, c[1] - s / 2 - 0.1 * s)
        out.append((int(x0), int(y0), int(min(W, x0 + s)), int(min(H, y0 + s)), s))
    return out


def expression(crops):
    import torch
    M = model()
    with torch.inference_mode():
        inp = M["proc"](images=crops, return_tensors="pt").to(M["dev"])
        e = M["model"].get_image_features(pixel_values=inp["pixel_values"].to(M["dt"]))
        e = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
        p = (e @ M["txt"].T * 100).softmax(-1)
    return p[:, :len(POS)].sum(-1).cpu().numpy()


def run(project) -> int:
    mo = project.load("moments.json")
    os.makedirs(project.path("thumbs"), exist_ok=True)
    cpath = project.path("faces_cache.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    jobs = []
    for c in mo:
        for m in c["moments"]:
            key = f'{c["id"]}@{m["t"]:.1f}'
            if key in cache:                      # scored on an earlier run
                m["face"], m["thumb"] = cache[key]
            else:
                jobs.append((c, m))

    def load(job):
        c, m = job
        thumb = project.path("thumbs", f'{c["id"].replace("#", "-")}_{m["t"]:.1f}.jpg')
        im = grab(project.folder, c["unit"], m["t"])
        if not os.path.exists(thumb):
            im.resize((480, round(480 * im.height / im.width))).save(thumb, quality=85)
        return job, im, thumb

    with ThreadPoolExecutor(6) as ex:          # ffmpeg seeks dominate; run them in parallel
        for (c, m), im, thumb in ex.map(load, jobs):
            m["thumb"] = os.path.relpath(thumb, project.reel_dir)
            if c["unit"].get("rotate"):
                m["face"] = 0.0                    # keypoints were taken unrotated; skip
                continue
            d = np.load(project.path("scores", c["unit"]["score"] + ".npz"))
            j = int(np.argmin(np.abs(d["pose_t"] - m["t"])))
            boxes = face_boxes(d["kp"][j], max(im.width, im.height) / 640, im.width, im.height)
            if not boxes:
                m["face"] = 0.0
                continue
            ex_scores = expression([im.crop(b[:4]) for b in boxes])
            size = np.array([min(1.0, b[4] / 160) for b in boxes])      # small, distant faces count for less
            m["face"] = round(float(np.max(ex_scores * size)), 3)
    for c in mo:
        for m in c["moments"]:
            if "face" in m:
                cache[f'{c["id"]}@{m["t"]:.1f}'] = [m["face"], m.get("thumb")]
    json.dump(cache, open(cpath, "w"))
    project.dump("moments.json", mo)
    print(f"[reel] {len(jobs)} moments scored for expression ({sum(len(c['moments']) for c in mo) - len(jobs)} cached)")
    return len(jobs)
