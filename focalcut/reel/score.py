"""First-pass shot scoring over every source file the units use (reduced resolution).

Per sampled frame (10 fps): global camera motion and subject motion (Farneback flow), sharpness and exposure in the
main subject box, plus the box's contrast and the brightness of a ring around it (lit, silhouetted or murky: see
moments.clarity). Every 2nd frame (5 fps): people (RT-DETR) and their 17-point poses (ViTPose).
Writes <reel>/scores/<name>.npz. Resumable: sources with an existing npz are skipped.

Decoding 4K 10-bit 4:2:2 HEVC is CPU-bound, so sources are shared across a few worker processes; more than two
GPU workers on a 12 GB card overflows VRAM (and on WSL pages to system RAM, which is far slower than one worker).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")               # workers share the CPU with their ffmpeg decoders

FPS, LONG = 10, 640                              # analysis rate and long-side resolution
MAXP = 4                                         # people kept per frame (largest boxes)
DET_ID = "PekingU/rtdetr_r50vd_coco_o365"
POSE_ID = "usyd-community/vitpose-base-simple"
_models = None


def models():
    global _models
    if _models is None:
        import cv2
        import torch
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
            torch.backends.cudnn.enabled = False
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        _models = dict(dev=dev, dt=dt,
                       det_proc=AutoProcessor.from_pretrained(DET_ID),
                       det=RTDetrForObjectDetection.from_pretrained(DET_ID).to(dev, dt).eval(),
                       pose_proc=AutoProcessor.from_pretrained(POSE_ID),
                       pose=VitPoseForPoseEstimation.from_pretrained(POSE_ID).to(dev, dt).eval())
    return _models


def probe(path):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height:stream_side_data=rotation:format=duration",
                                            "-of", "json", path]))
    s = j["streams"][0]
    rot = abs(int(next((d.get("rotation", 0) for d in s.get("side_data_list", [])), 0)))
    w, h = (s["height"], s["width"]) if rot in (90, 270) else (s["width"], s["height"])
    return float(j["format"]["duration"]), w, h


def frames(path, w, h):
    import numpy as np
    sw, sh = (LONG, round(LONG * h / w / 2) * 2) if w >= h else (round(LONG * w / h / 2) * 2, LONG)
    # analysis only: skip non-reference frames and deblocking (about 20% faster to decode), no audio
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "5", "-skip_frame", "noref", "-skip_loop_filter", "all",
                          "-i", path, "-an", "-vf", f"fps={FPS},scale={sw}:{sh}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = sw * sh * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(sh, sw, 3)
    p.wait()


def people(batch):
    """Person boxes (xyxy, score) and 17 keypoints (x, y, conf) for up to MAXP people per frame."""
    import numpy as np
    import torch
    M = models()
    with torch.inference_mode():
        inp = M["det_proc"](images=batch, return_tensors="pt").to(M["dev"])
        inp["pixel_values"] = inp["pixel_values"].to(M["dt"])
        res = M["det_proc"].post_process_object_detection(M["det"](**inp), target_sizes=[im.shape[:2] for im in batch],
                                                          threshold=0.45)
        boxes = []
        for r in res:
            keep = r["labels"] == 0               # COCO person
            b, s = r["boxes"][keep].float().cpu().numpy(), r["scores"][keep].float().cpu().numpy()
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)
            o = np.argsort(-area)[:MAXP]
            boxes.append((b[o], s[o]))
        kps = [np.zeros((0, 17, 3), np.float32) for _ in batch]
        todo = [i for i, (b, _) in enumerate(boxes) if len(b)]
        if todo:
            bx = [np.c_[boxes[i][0][:, :2], boxes[i][0][:, 2:] - boxes[i][0][:, :2]] for i in todo]   # COCO xywh
            pin = M["pose_proc"]([batch[i] for i in todo], boxes=bx, return_tensors="pt").to(M["dev"])
            pin["pixel_values"] = pin["pixel_values"].to(M["dt"])
            out = M["pose"](**pin)
            out.heatmaps = out.heatmaps.float()   # the decoder's scipy filters reject fp16
            pr = M["pose_proc"].post_process_pose_estimation(out, boxes=bx)
            for i, r in zip(todo, pr):
                kps[i] = np.stack([np.c_[p["keypoints"].float().cpu().numpy(), p["scores"].float().cpu().numpy()]
                                   for p in r]).astype(np.float32)
    return boxes, kps


def score(folder: str, source: str, out_dir: str) -> str:
    import cv2
    import numpy as np

    from .project import Project
    out = os.path.join(out_dir, Project.score_name(source) + ".npz")
    if os.path.exists(out):
        return f"{source}: cached"
    path = os.path.join(folder, source)
    dur, w, h = probe(path)
    t0 = time.time()
    T, CAM, SUBJ, SHARP, LUMA, AREA, CX, NP = [], [], [], [], [], [], [], []
    CONTR, RING = [], []
    PT, KP, BX = [], [], []
    prev, pending = None, []                      # pending: (frame index, frame, flow magnitude) awaiting people()

    def flush():
        b, k = people([f for _, f, _ in pending])
        for (fi, f, mag), (boxes, sc), kk in zip(pending, b, k):
            PT.append(fi / FPS)
            kp = np.full((MAXP, 17, 3), np.nan, np.float32); kp[:len(kk)] = kk[:MAXP]; KP.append(kp)
            bx = np.full((MAXP, 5), np.nan, np.float32)
            if len(boxes):
                bx[:len(boxes)] = np.c_[boxes, sc][:MAXP]
            BX.append(bx)
            NP[fi] = len(boxes)
            if not len(boxes):
                continue
            x0, y0, x1, y1 = boxes[0].astype(int)
            crop = cv2.cvtColor(f[max(y0, 0):y1, max(x0, 0):x1], cv2.COLOR_RGB2GRAY)
            if crop.size >= 100:
                SHARP[fi] = cv2.Laplacian(crop, cv2.CV_32F).var()
                LUMA[fi] = crop.mean()
                CONTR[fi] = crop.std()
                bw, bh = x1 - x0, y1 - y0                  # a ring half the box's size around it: what it stands against
                X0, Y0 = max(x0 - bw // 2, 0), max(y0 - bh // 2, 0)
                X1, Y1 = min(x1 + bw // 2, f.shape[1]), min(y1 + bh // 2, f.shape[0])
                outer = cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_RGB2GRAY).astype(np.float64)
                n_ring = outer.size - crop.size
                if n_ring > 100:
                    RING[fi] = (outer.sum() - crop.astype(np.float64).sum()) / n_ring
            AREA[fi] = (x1 - x0) * (y1 - y0) / (f.shape[0] * f.shape[1])
            CX[fi] = (x0 + x1) / 2 / f.shape[1]
            if mag is not None:                    # flow inside the main person's box, minus the camera's own motion
                m = mag[max(y0 // 2, 0):y1 // 2, max(x0 // 2, 0):x1 // 2]
                if m.size:
                    SUBJ[fi] = max(float(np.percentile(m, 75)) - CAM[fi], 0)
        pending.clear()

    for i, f in enumerate(frames(path, w, h)):
        g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)), cv2.COLOR_RGB2GRAY)
        mag = None
        if prev is not None:
            mag = np.linalg.norm(cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2) * 2
        prev = g
        T.append(i / FPS); CAM.append(float(np.median(mag)) if mag is not None else 0.0)
        SUBJ.append(0.0); SHARP.append(0.0); LUMA.append(0.0); AREA.append(0.0); CX.append(0.5); NP.append(0)
        CONTR.append(0.0); RING.append(0.0)
        if i % 2 == 0:                             # people + poses at 5 fps
            pending.append((i, f, mag))
            if len(pending) == 16:
                flush()
    if pending:
        flush()
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out, t=np.array(T), cam=np.array(CAM), subj=np.array(SUBJ), sharp=np.array(SHARP),
                        luma=np.array(LUMA), contrast=np.array(CONTR), ring=np.array(RING), area=np.array(AREA),
                        cx=np.array(CX), npeople=np.array(NP),
                        pose_t=np.array(PT), kp=np.array(KP), boxes=np.array(BX), size=np.array([w, h]), dur=dur)
    return f"{source}: {dur:.0f}s in {time.time() - t0:.0f}s"


def shards(sources: list[tuple[str, float]], n: int) -> list[list[str]]:
    """Split (source, duration) pairs into n lists of roughly equal total duration, longest first."""
    out, tot = [[] for _ in range(n)], [0.0] * n
    for s, d in sorted(sources, key=lambda x: -x[1]):
        k = tot.index(min(tot)); out[k].append(s); tot[k] += d
    return [x for x in out if x]


def run(project, workers: int = 2) -> None:
    from .project import Project
    assert isinstance(project, Project)
    units = project.load("units.json")
    srcs = sorted({u["source"] for u in units})
    out_dir = project.path("scores")
    todo = [s for s in srcs if not os.path.exists(os.path.join(out_dir, project.score_name(s) + ".npz"))]
    if not todo:
        print("[reel] scores: all cached"); return
    durs = [(s, probe(os.path.join(project.folder, s))[0]) for s in todo]
    procs = [subprocess.Popen([sys.executable, "-m", "focalcut.reel.score", project.folder, out_dir, *sh])
             for sh in shards(durs, max(1, workers))]
    for p in procs:
        p.wait()
    missing = [s for s in srcs if not os.path.exists(os.path.join(out_dir, project.score_name(s) + ".npz"))]
    print(f"[reel] scored {len(srcs) - len(missing)}/{len(srcs)} sources" + (f"; failed: {missing}" if missing else ""))


if __name__ == "__main__":                        # worker: python -m focalcut.reel.score <folder> <out_dir> <src...>
    folder, out_dir, *names = sys.argv[1:]
    for n in names:
        try:
            print(score(folder, n, out_dir), flush=True)
        except Exception as e:                     # keep going; a rerun retries only the missing sources
            print(f"{n}: FAILED {e!r}", flush=True)
