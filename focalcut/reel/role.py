"""Who is who, by SigLIP 2 zero-shot at 2 frames a second of each unit's export.

Lead and follow: in a partner dance the follow (usually the woman) is the one to feature, and a couple should share
the frame (frame.follow). Each person the score pass found: how much they look like a woman. Writes
<reel>/scores/role_<unit>.npz (t: source time, box: the person's box as fractions of the frame, p: woman, 0..1),
matched back to the pose boxes by time and overlap -- not by box index, so a rescore never makes it stale.

Musicians: band footage (B-roll tagged "band", or reel.json plan members / member_clips) is often the camera roaming
the crowd, a back, a blur or empty stage light between the real shots of the players. The whole frame against
prompts for a player performing and for those: <reel>/scores/band_<unit>.npz (t, p: a musician playing, 0..1), so
the plan cuts to the band only where the band is (plan.assess).

Resumable; units with an export file.
"""
from __future__ import annotations

import os

import numpy as np

from .people import frames
from .semantic import MODEL, crop

WOMAN = ["a photo of a woman dancing", "a woman", "a female dancer in a dress or skirt"]
MAN = ["a photo of a man dancing", "a man", "a male dancer in a shirt and trousers"]
PLAYING = ["a singer singing into a microphone on stage", "a violinist playing the violin", "a trumpet player playing",
           "a musician playing congas or timbales", "a close-up of a musician performing on stage",
           "a band of musicians playing on stage"]
NOT_PLAYING = ["a crowd of people dancing", "people filming with their phones", "the back of a man's head",
               "a blurry out of focus photo", "an empty stage with coloured lights", "faces of people in a crowd"]
WIDTH = 1280                # analysis width (portrait clips: this height)
MIN_H = 0.12                # people smaller than this (fraction of frame height) are left out: too small to read


def boxes_at(d, t):
    """The pose boxes nearest t (within 0.15 s) as fractions of the frame, (P, 4); empty when none is close."""
    pt = d["pose_t"]
    if not len(pt):
        return np.zeros((0, 4))
    i = int(np.clip(np.searchsorted(pt, t), 0, len(pt) - 1))
    if i and abs(pt[i - 1] - t) < abs(pt[i] - t):
        i -= 1
    if abs(pt[i] - t) > 0.15:
        return np.zeros((0, 4))
    b = d["boxes"][i, :, :4]
    b = b[~np.isnan(b[:, 0])] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    return b[(b[:, 3] - b[:, 1]) >= MIN_H]


def scan(project, u, model, proc, txt, dev, dt):
    import subprocess

    import torch
    from PIL import Image

    from . import frame as FR
    from .shake import export_of
    out = project.path("scores", f"role_{u['id']}.npz")
    path, src = export_of(project, u), project.path("scores", u["score"] + ".npz")
    if os.path.exists(out) or not path or not os.path.exists(src):
        return None
    d = FR.load(src)
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    if abs(W / H - d["size"][0] / d["size"][1]) > 0.02:
        return f"{u['id']}: export and source differ in shape, skipped"
    s = WIDTH / max(W, H)
    w, h = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    x0 = u.get("export_t0", 0.0)
    T, B, P = [], [], []
    for t, f in frames(path, u["t0"] - x0, u["t1"] - x0, w, h):
        bs = boxes_at(d, t + x0)
        cs = [(b, c) for b in bs if (c := crop(f[..., ::-1], b)) is not None]
        if not cs:
            continue
        with torch.inference_mode():
            inp = proc(images=[Image.fromarray(np.ascontiguousarray(c)) for _, c in cs], return_tensors="pt").to(dev)
            inp["pixel_values"] = inp["pixel_values"].to(dt)
            im = model.get_image_features(**inp)
            im = torch.nn.functional.normalize(getattr(im, "pooler_output", im).float(), dim=-1)
            p = torch.softmax(100 * im @ txt.T, -1).cpu().numpy()
        for (b, _), q in zip(cs, p):
            T.append(t + x0); B.append(b); P.append(float(q[:len(WOMAN)].sum()))
    np.savez_compressed(out, t=np.array(T, np.float32), box=np.array(B, np.float32).reshape(-1, 4),
                        p=np.array(P, np.float32))
    return f"{u['id']}: {len(T)} people seen, {np.mean(np.array(P) > 0.5) if P else 0:.0%} read as women"


def band_footage(project, u):
    """Is this unit footage of the band (where players are cut in)?"""
    cfg = {}
    for f in ("landscape", "portrait"):
        cfg.update((project.config.get("plan") or {}).get(f) or {})
    tag = u.get("tag", "")
    return u["role"] == "transition" and ("band" in tag or any(k in tag for k in cfg.get("members", []))
                                          or u["clip"] in cfg.get("member_clips", []))


def scan_band(project, u, model, proc, txt, dev, dt):
    """A musician playing, frame by frame (2 a second), for one unit of band footage."""
    import subprocess

    import torch
    from PIL import Image

    from .shake import export_of
    out = project.path("scores", f"band_{u['id']}.npz")
    path = export_of(project, u)
    if os.path.exists(out) or not path:
        return None
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    s = 640 / max(W, H)
    x0 = u.get("export_t0", 0.0)
    import cv2
    T, P, S = [], [], []
    for t, f in frames(path, u["t0"] - x0, u["t1"] - x0, int(W * s) // 2 * 2, int(H * s) // 2 * 2):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = g.shape                                            # sharpness of the middle (SigLIP looks past blur)
        S.append(float(cv2.Laplacian(g[h // 6:5 * h // 6, w // 6:5 * w // 6], cv2.CV_32F).var()))
        with torch.inference_mode():
            inp = proc(images=[Image.fromarray(np.ascontiguousarray(f[..., ::-1]))], return_tensors="pt").to(dev)
            inp["pixel_values"] = inp["pixel_values"].to(dt)
            im = model.get_image_features(**inp)
            im = torch.nn.functional.normalize(getattr(im, "pooler_output", im).float(), dim=-1)
            p = torch.softmax(100 * im @ txt.T, -1)[0].cpu().numpy()
        T.append(t + x0); P.append(float(p[:len(PLAYING)].sum()))
    np.savez_compressed(out, t=np.array(T, np.float32), p=np.array(P, np.float32), sharp=np.array(S, np.float32))
    ok = (np.array(P) >= BAND_P) & (np.array(S) >= BAND_SHARP) if P else np.zeros(0, bool)
    return f"{u['id']}: {len(T)} frames, {ok.mean() if P else 0:.0%} show the band playing, in focus"


BAND_P, BAND_SHARP, SHARP_FULL = 0.45, 120.0, 400.0   # a musician playing: SigLIP share, centre sharpness (640 px)


def musician(project, u, a, b):
    """The band on screen over a stretch of a unit's source time: (how surely a player is performing 0..1, how sharp
    the frame is), or None when the unit wasn't scanned. Below BAND_P or BAND_SHARP it isn't a band shot (a crowd, a
    back, a blur); the plan ranks the rest by p x min(sharp / SHARP_FULL, 1)."""
    f = project.path("scores", f"band_{u['id']}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    m = (z["t"] >= a) & (z["t"] <= b)
    if not m.any():
        return None
    return float(z["p"][m].mean()), float(np.median(z["sharp"][m])) if "sharp" in z else SHARP_FULL


def run(project) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor
    if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL).to(dev, dt).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    with torch.inference_mode():
        tin = proc(text=WOMAN + MAN, return_tensors="pt", padding="max_length", max_length=64).to(dev)
        t = model.get_text_features(**tin)
        txt = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1)
        tin = proc(text=PLAYING + NOT_PLAYING, return_tensors="pt", padding="max_length", max_length=64).to(dev)
        t = model.get_text_features(**tin)
        btxt = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1)
    for u in project.load("units.json"):
        if band_footage(project, u):
            msg = scan_band(project, u, model, proc, btxt, dev, dt)
            if msg:
                print("[reel] band", msg, flush=True)
    for u in project.load("units.json"):
        if u["role"] in ("dance", "show") and not u.get("rotate"):
            msg = scan(project, u, model, proc, txt, dev, dt)
            if msg:
                print("[reel] role", msg, flush=True)


def woman(d, times, boxes):
    """How much each pose person (F, P) looks like a woman, from the role scan (d["role"]): the mean over role
    detections within 0.3 s whose box overlaps theirs well; NaN where there is none."""
    out = np.full(boxes.shape[:2], np.nan)
    if "role" not in d or not len(d["role"]["t"]):
        return out
    r = d["role"]
    nb = boxes[..., :4] / [d["aw"], d["ah"], d["aw"], d["ah"]]
    for f, t in enumerate(times):
        near = np.nonzero(np.abs(r["t"] - t) <= 0.3)[0]
        if not len(near):
            continue
        rb = r["box"][near]
        for k in range(boxes.shape[1]):
            b = nb[f, k]
            if np.isnan(b[0]):
                continue
            iw = np.clip(np.minimum(rb[:, 2], b[2]) - np.maximum(rb[:, 0], b[0]), 0, None)
            ih = np.clip(np.minimum(rb[:, 3], b[3]) - np.maximum(rb[:, 1], b[1]), 0, None)
            inter = iw * ih
            iou = inter / ((rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            if (iou >= 0.5).any():
                out[f, k] = float(r["p"][near][iou >= 0.5].mean())
    return out
