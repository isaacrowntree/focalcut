"""Expression track for every unit with people: twice a second, the faces of the subject and their partner (the two
largest whose nose sits in the subject box, frame.track) scored by SigLIP 2 zero-shot (faces.expression). Expression
only, never looks: engaged (smiling, laughing, joyful, eyes open on the partner) minus disengaged (eyes closed
mid-blink, a grimace, a blank or bored look, a frown, looking down), -1..1 per face.

Frames come from the review step's 1080p preview copies when there are any (else the export file), so this is one
quick GPU pass per shoot; the window scrub (windows.py) then reads the track without any model. Writes
<reel>/scores/expr_<unit>.npz: t (K,) source seconds, e (K, 2) engagement of up to two faces (NaN: none), size (K, 2)
face size 0..1 (small, distant faces count for less); and scores/taste_<unit>.npz: t, emb (K, 768) the whole frame's
SigLIP 2 embedding, for the taste model (taste.py). Resumable: existing tracks are kept.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

STEP = 0.5                                        # seconds between samples
BATCH = 48                                        # face crops per model call


def video_of(project, u):
    """The unit's export (its preview copy when there is one), and its time offset: video time = source time - x0."""
    base = u["id"].split("#")[0]
    prev = project.path("previews", base + ".mp4")
    exp = os.path.join(project.export_dir, u.get("category", ""), base + ".mp4")
    f = prev if os.path.exists(prev) else exp
    return (f if os.path.exists(f) else None), float(u.get("export_t0", 0.0))


def frames(f, t0, t1):
    """RGB frames every STEP seconds from t0 to t1 (video time): [(t, PIL image), ...]."""
    from PIL import Image
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", f], capture_output=True, text=True).stdout.split(",")
    w, h = int(probe[0]), int(probe[1])
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.2f}", "-i", f, "-t", f"{t1 - t0:.2f}",
                        "-vf", f"fps={1 / STEP}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = w * h * 3
    return [(max(0.0, t0) + k * STEP, Image.frombuffer("RGB", (w, h), p[k * n:(k + 1) * n]))
            for k in range(len(p) // n)]


def embed(images):
    """Whole-frame SigLIP 2 embeddings (normalised, float16) -- the taste step's input (taste.py)."""
    import torch

    from . import faces as FA
    M = FA.model()
    with torch.inference_mode():
        inp = M["proc"](images=images, return_tensors="pt").to(M["dev"])
        e = M["model"].get_image_features(pixel_values=inp["pixel_values"].to(M["dt"]))
        e = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
    return e.cpu().numpy().astype(np.float16)


def track_one(project, u):
    from . import faces as FA
    from . import frame as FR
    key = u["id"].replace("#", "-")
    out, tout = project.path("scores", f"expr_{key}.npz"), project.path("scores", f"taste_{key}.npz")
    if os.path.exists(out) and os.path.exists(tout):
        return f'{u["id"]}: cached'
    f, x0 = video_of(project, u)
    if f is None:
        return f'{u["id"]}: skipped'
    fr = frames(f, u["t0"] - x0, u["t1"] - x0)
    if not fr:
        return f'{u["id"]}: no frames'
    ts = np.array([t + x0 for t, _ in fr])
    if not os.path.exists(tout):                    # the whole frame, for the taste model
        emb = np.concatenate([embed([im for _, im in fr[i:i + BATCH]]) for i in range(0, len(fr), BATCH)])
        np.savez_compressed(tout, t=ts, emb=emb)
    if os.path.exists(out) or u.get("rotate"):      # (keypoints were taken unrotated: no face crops)
        return f'{u["id"]}: {len(ts)} frames embedded'
    d = FR.load(project.path("scores", u["score"] + ".npz"))
    union, _, _ = FR.track(d, ts)
    E, Z = np.full((len(ts), 2), np.nan, np.float32), np.zeros((len(ts), 2), np.float32)
    jobs = []                                       # (frame, slot, crop, size)
    for k, (t, (_, im)) in enumerate(zip(ts, fr)):
        j = int(np.argmin(np.abs(d["pose_t"] - t)))
        if abs(d["pose_t"][j] - t) > 0.3:
            continue
        boxes = FA.subject_faces(d, j, union[k], FA.face_boxes_at(d["kp"][j], max(im.width, im.height) / 640,
                                                                  im.width, im.height))
        for s, b in enumerate(boxes):
            jobs.append((k, s, im.crop(b[:4]), min(1.0, b[4] / 110)))   # (1080p: a face this wide is a close one)
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        g, b = FA.expression([c for _, _, c, _ in chunk])
        for (k, s, _, z), gg, bb in zip(chunk, g, b):
            E[k, s], Z[k, s] = gg - bb, z
    np.savez_compressed(out, t=ts, e=E, size=Z)
    return f'{u["id"]}: {len(ts)} samples, {len(jobs)} faces'


def load(project, u):
    """A unit's expression track (t, e, size), or None."""
    f = project.path("scores", f'expr_{u["id"].replace("#", "-")}.npz')
    return dict(np.load(f)) if os.path.exists(f) else None


def window(tr, t0, t1):
    """Engagement over a window: per sample the worse of the subject's and partner's faces, averaged (NaN: no face
    seen in it); and the best engaged face (0..1, size-weighted) for the plan's face term."""
    if tr is None:
        return np.nan, 0.0
    m = (tr["t"] >= t0) & (tr["t"] <= t1)
    e = tr["e"][m]
    seen = ~np.isnan(e).all(1)
    if not seen.any():
        return np.nan, 0.0
    worst = np.nanmin(e[seen], 1)
    best = np.nanmax(np.clip(np.nan_to_num(e[seen], nan=-1.0), 0, 1) * tr["size"][m][seen])
    return float(worst.mean()), float(best)


def run(project) -> None:
    os.makedirs(project.path("scores"), exist_ok=True)
    for u in project.load("units.json"):
        if os.path.exists(project.path("scores", u["score"] + ".npz")):
            print("[reel] expression", track_one(project, u), flush=True)
