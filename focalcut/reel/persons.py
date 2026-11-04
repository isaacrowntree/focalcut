"""Per-person measurements for the pick rules (concepts §19), twice a second over every unit with people.

The rules are about the people a shot is about -- not any face somewhere in the subject box (in a crowd or on a
stage that box is nearly the whole frame, and a bystander's face would vouch for a back-turned dancer). So each
sample names its main people: the tallest person in the frame and anyone nearly as tall (MAIN_H) standing right
beside them (a couple, a duo), at most three. For each of them, measured, not judged:

- face: a YuNet face of theirs -- its centre inside their own head (from the pose's face and ear keypoints, else
  the top of their box) and its size a head's for their build (FACE_FIT x their shoulder width), so a face further
  back that happens to sit behind their head is not theirs. No face: a back, or a head turned away.
- yaw: how far the face is turned from the camera, from YuNet's landmarks: the nose's offset from between the
  eyes, in eye distances (0 facing the lens, about +-0.5 in three-quarter, larger in profile).
- sharp: the variance of the Laplacian over their face box at SHARP_PX square, and rel -- that over the sharpest
  SHARP_PX tile of the same frame (the frame's own noise and grain cancel out): a soft face in a frame with
  something sharp in it scores low however noisy the footage is.

Writes scores/persons_<unit>.npz: t (K,), box (K, 3, 4) frame fractions (NaN: nobody), face (K, 3) bool,
yaw (K, 3), sharp (K, 3), rel (K, 3), frame (K,) the sharpest tile; and scores/faces_dense_<unit>.npz, every face
found (t, cx, cy, h, score: the people step's faces_<unit>.npz layout, at every sample rather than about one a
second), for frame.face_seen. CPU only (OpenCV), resumable.
"""
from __future__ import annotations

import os

import numpy as np

STEP = 0.5
MAIN_H, MAIN_GAP, MAIN_MAX = 0.75, 0.5, 3         # nearly as tall, within half a width of the tallest
FACE_FIT = (0.25, 1.8)                            # a face's width for its owner: this x their build (shoulders,
                                                  # else their box)
SHARP_PX = 64
SCALES = (1920, 640)                              # YuNet misses big close faces at full size, small far ones when
                                                  # shrunk: both, merged


def detect(dets, frame, small):
    """Faces in a frame at every scale (frame pixels), duplicates dropped (the more confident kept)."""
    out = []
    for det, fr in zip(dets, (frame, small)):
        _, f = det.detect(fr)
        if f is not None:
            f = f.copy()
            k = frame.shape[1] / fr.shape[1]
            f[:, :14] *= k
            out += list(f)
    out.sort(key=lambda f: -f[14])
    keep = []
    for f in out:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        if all(abs(cx - g[0] - g[2] / 2) > 0.5 * g[2] or abs(cy - g[1] - g[3] / 2) > 0.5 * g[3] for g in keep):
            keep.append(f)
    return np.array(keep, np.float32).reshape(-1, 15)


def main_people(boxes):
    """Indices of a pose frame's main people: the tallest and those nearly as tall right beside them."""
    h = np.nan_to_num(boxes[:, 3] - boxes[:, 1])
    if h.max() <= 0:
        return []
    k = int(np.argmax(h))
    w = boxes[k, 2] - boxes[k, 0]
    gap = np.nan_to_num(np.maximum(boxes[:, 0] - boxes[k, 2], boxes[k, 0] - boxes[:, 2]), nan=9e9)
    main = (h >= MAIN_H * h[k]) & (gap < MAIN_GAP * w)
    main[k] = True
    return [int(i) for i in np.argsort(-h) if main[i]][:MAIN_MAX]


def head_region(box, kp):
    """(x0, y0, x1, y1, shoulder width) of a person's head, in the pose's pixels."""
    pts = np.array([kp[i, :2] for i in range(5) if kp[i, 2] >= 0.2])
    sh = abs(kp[5, 0] - kp[6, 0]) if min(kp[5, 2], kp[6, 2]) >= 0.3 else 0.0
    bw, bh = box[2] - box[0], box[3] - box[1]
    sh = max(sh, 0.45 * bw, 0.12 * bh)              # (side-on, the shoulders overlap: the body's own size)
    if len(pts) >= 2:
        c, r = pts.mean(0), max(1.2 * np.ptp(pts[:, 0]), 0.45 * sh)
    else:
        r = min(0.3 * bw, 0.12 * bh)
        c = np.array([(box[0] + box[2]) / 2, box[1] + r])
    return c[0] - r, c[1] - 1.2 * r, c[0] + r, c[1] + r, sh


def lap(g):
    import cv2
    return float(cv2.Laplacian(cv2.resize(g, (SHARP_PX, SHARP_PX), interpolation=cv2.INTER_AREA), cv2.CV_32F).var())


def frame_sharp(g):
    """The sharpest SHARP_PX tile of a grey frame (variance of the Laplacian)."""
    import cv2
    L = cv2.Laplacian(g, cv2.CV_32F)
    H, W = g.shape
    v = [L[y:y + SHARP_PX, x:x + SHARP_PX].var() for y in range(0, H - SHARP_PX + 1, SHARP_PX)
         for x in range(0, W - SHARP_PX + 1, SHARP_PX)]
    return float(np.percentile(v, 99)) if v else np.nan


def measure(frame, faces, kp, boxes, sx, sy):
    """One sample: (box (3, 4) fractions, face, yaw, sharp, rel, frame sharpness). frame: BGR; faces: YuNet rows
    in frame pixels; kp / boxes: the pose frame (pose pixels); sx, sy: pose pixels -> frame pixels."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    fs = frame_sharp(g)
    B = np.full((MAIN_MAX, 4), np.nan, np.float32)
    F = np.zeros(MAIN_MAX, bool)
    Y, S, R = (np.full(MAIN_MAX, np.nan, np.float32) for _ in range(3))
    used = set()
    for s, i in enumerate(main_people(boxes)):
        b = boxes[i]
        B[s] = b[0] * sx / W, b[1] * sy / H, b[2] * sx / W, b[3] * sy / H
        _, _, _, _, sh = head_region(b, kp[i])
        own = np.array([kp[i][q, :2] for q in range(5) if kp[i][q, 2] >= 0.3])   # their face and ear points
        front = np.array([kp[i][q, :2] for q in range(3) if kp[i][q, 2] >= 0.3])  # nose, eyes: the face side
        best, hits = None, 0
        for n, f in enumerate(faces):
            if n in used or not len(front):         # (only ears seen: from behind, whatever face shows past them)
                continue
            gr = 0.6 * f[2]                         # a face of theirs holds their own face points (grown: the pose is
            #                                         5 a second, a spinning head moves between) -- a nose or an eye
            fx0, fy0, fx1, fy1 = (f[0] - gr) / sx, (f[1] - gr) / sy, (f[0] + f[2] + gr) / sx, (f[1] + f[3] + gr) / sy

            def held(p):
                return (p[:, 0] >= fx0) & (p[:, 0] <= fx1) & (p[:, 1] >= fy0) & (p[:, 1] <= fy1)   # noqa: B023
            k = int(held(own).sum()) if held(front).any() else 0
            if k >= 1 and FACE_FIT[0] * sh <= f[2] / sx <= FACE_FIT[1] * sh and (k, f[14]) > (hits, faces[best][14] if best is not None else 0):
                best, hits = n, k
        if best is None:
            continue
        used.add(best)
        f = faces[best]
        F[s] = True
        eye = np.array([(f[4] + f[6]) / 2, (f[5] + f[7]) / 2])
        Y[s] = (f[8] - eye[0]) / max(abs(f[4] - f[6]), 1.0)
        fx0, fy0 = int(max(0, f[0])), int(max(0, f[1]))
        crop = g[fy0:int(min(H, f[1] + f[3])), fx0:int(min(W, f[0] + f[2]))]
        if min(crop.shape) >= 16:
            S[s] = lap(crop)
            R[s] = S[s] / fs if fs and fs > 0 else np.nan
    return B, F, Y, S, R, fs


def track_one(args):
    folder, cut, u = args
    import cv2

    from . import frame as FR
    from . import people as PE
    from .project import Project
    project = Project(folder, cut)
    out = project.path("scores", f'persons_{u["id"].replace("#", "-")}.npz')
    if os.path.exists(out):
        return f'{u["id"]}: cached'
    base = u["id"].split("#")[0]                    # the export, not the 720p review preview: faces need the pixels
    f = os.path.join(project.export_dir, u.get("category", ""), base + ".mp4")
    f, x0 = (f if os.path.exists(f) else None), float(u.get("export_t0", 0.0))
    p = project.path("scores", u["score"] + ".npz")
    if f is None or u.get("rotate") or not os.path.exists(p):
        return f'{u["id"]}: skipped'
    import subprocess
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", f]).decode().split(",")[:2])
    s = min(1.0, SCALES[0] / max(W, H))
    w, h = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    ws, hs = int(w * SCALES[1] / max(w, h)) // 2 * 2, int(h * SCALES[1] / max(w, h)) // 2 * 2
    dets = [cv2.FaceDetectorYN.create(PE.models()["det"], "", size, 0.6, 0.3, 5000) for size in ((w, h), (ws, hs))]
    d = FR.load(p)
    sx, sy = w / d["aw"], h / d["ah"]
    T, B, F, Y, S, R, FS = [], [], [], [], [], [], []
    A = []                                          # every face: t, cx, cy, h (frame fractions), score
    PE.RATE, rate = 1 / STEP, PE.RATE
    try:
        for t, fr in PE.frames(f, u["t0"] - x0, u["t1"] - x0, w, h):
            ts = t + x0
            j = int(np.argmin(np.abs(d["pose_t"] - ts)))
            if abs(d["pose_t"][j] - ts) > 0.3:
                continue
            faces = detect(dets, fr, cv2.resize(fr, (ws, hs), interpolation=cv2.INTER_AREA))
            for fc in faces:
                A.append((ts, (fc[0] + fc[2] / 2) / w, (fc[1] + fc[3] / 2) / h, fc[3] / h, fc[14]))
            r = measure(fr, faces, d["kp"][j], d["boxes"][j], sx, sy)
            T.append(ts)
            for L, v in zip((B, F, Y, S, R, FS), r):
                L.append(v)
    finally:
        PE.RATE = rate
    np.savez_compressed(out, t=np.array(T, np.float32), box=np.array(B).reshape(-1, MAIN_MAX, 4),
                        face=np.array(F).reshape(-1, MAIN_MAX), yaw=np.array(Y).reshape(-1, MAIN_MAX),
                        sharp=np.array(S).reshape(-1, MAIN_MAX), rel=np.array(R).reshape(-1, MAIN_MAX),
                        frame=np.array(FS, np.float32))
    A = np.array(A, np.float32).reshape(-1, 5)
    np.savez_compressed(project.path("scores", f'faces_dense_{u["id"]}.npz'), t=A[:, 0], cx=A[:, 1], cy=A[:, 2],
                        h=A[:, 3], score=A[:, 4])
    return f'{u["id"]}: {len(T)} samples'


def load(project, u):
    f = project.path("scores", f'persons_{u["id"].replace("#", "-")}.npz')
    return dict(np.load(f)) if os.path.exists(f) else None


def run(project, workers=8) -> None:
    from concurrent.futures import ProcessPoolExecutor
    us = [u for u in project.load("units.json") if os.path.exists(project.path("scores", u["score"] + ".npz"))]
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for msg in ex.map(track_one, [(project.folder, project.cut, u) for u in us]):
            print("[reel] persons", msg, flush=True)
