"""First-pass shot scoring over every source file the units use (reduced resolution).

Per sampled frame (10 fps): global camera motion and subject motion (Farneback flow), sharpness and exposure in the
main subject box, plus the box's contrast and the brightness of a ring around it (lit, silhouetted or murky: see
moments.clarity). Every 2nd frame (5 fps): people (RT-DETR) and their 17-point poses (ViTPose).
Writes <reel>/scores/<name>.npz. Resumable: sources with an existing npz are skipped.

Each source is read from its proxy when reel.json has one (project.decode_from: a 1080p H.264 copy decodes on
NVDEC at about 15x real time, where the 4K 10-bit 4:2:2 HEVC camera file takes five CPU threads to reach 1.7x), on
the fastest decoder that reads it (hw.readers). The frames are scaled on the CPU either way, so a proxy decoded on
the GPU scores exactly as the same proxy decoded on the CPU would; against the camera file the numbers move a
little (a 1080p picture scaled to 640 px, not a 4K one: see docs/hardware.md). Sources are shared across a few
worker processes (WORKERS); each holds the two models, and past its memory a 12 GB card on WSL pages to system
RAM, which is far slower than one worker.
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


class DecodeError(RuntimeError):
    """A hardware decode that stopped part-way through a file (the worker reads that source again on the CPU)."""


def probe(path):
    return _probe(path)[:3]


def _probe(path):
    """(duration, width, height, rotation): the size as displayed (a phone's rotated clip is its upright size)."""
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height:stream_side_data=rotation:format=duration",
                                            "-of", "json", path]))
    s = j["streams"][0]
    rot = abs(int(next((d.get("rotation", 0) for d in s.get("side_data_list", [])), 0)))
    w, h = (s["height"], s["width"]) if rot in (90, 270) else (s["width"], s["height"])
    return float(j["format"]["duration"]), w, h, rot


def source_time(shifts, t):
    """A proxy's time t in its source's own time: the inverse of Project.decode_from for reel.json's proxy shifts
    [[at, d], ...] (from source time `at` on, the proxy's time is the source's + d, each shift on top of those
    before it)."""
    total, out = 0.0, t
    for at, d in sorted(shifts or ()):
        total += d
        if t - total >= at - 1e-9:
            out = t - total
    return out


def frame_size(w, h):
    """The analysis frame for a w x h picture: LONG on its long side, even."""
    return (LONG, round(LONG * h / w / 2) * 2) if w >= h else (round(LONG * w / h / 2) * 2, LONG)


def frames(path, w, h, gpu=True):
    """path's frames at FPS, RGB, at frame_size(w, h), from the first of hw.readers that decodes it: NVDEC (a
    proxy), Quick Sync (a 4:2:2 camera file), the CPU. On a GPU the frames are thinned to FPS before they are
    copied to memory and scaled on the CPU after (hw.readers: why not on the GPU). A rotated file (a phone's) goes
    to the CPU, where ffmpeg turns it upright. Raises DecodeError when a GPU decode stops part-way."""
    import numpy as np

    from . import hw
    sw, sh = frame_size(w, h)
    n = sw * sh * 3
    for kind, exe, dec, P, download in hw.readers(path, gpu and not _probe(path)[3]):
        # (every frame is decoded, whole, on every decoder, so a source scores the same wherever it is read:
        # skipping non-reference frames, 20% faster on the CPU, left fps= picking frames 20-40 ms off their time --
        # uneven steps that Farneback read as 15-20% less subject motion on 50p camera files)
        cpu = ["-threads", "5"] if kind == "cpu" else []
        p = subprocess.Popen([exe, "-v", "error", *cpu, *dec, "-i", P(path), "-an",
                              "-vf", f"fps={FPS},{download}scale={sw}:{sh}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             stdout=subprocess.PIPE, stderr=None if kind == "cpu" else subprocess.DEVNULL)
        got = 0
        try:
            while True:
                b = p.stdout.read(n)
                if len(b) < n:
                    break
                got += 1
                yield np.frombuffer(b, np.uint8).reshape(sh, sw, 3)
        finally:
            p.stdout.close()
            rc = p.wait()
        if got:
            if rc and kind != "cpu":
                raise DecodeError(f"{kind} decode of {os.path.basename(path)} stopped after {got} frames")
            return


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


def score(folder: str, source: str, out_dir: str, gpu: bool = True, project=None) -> str:
    """Score one source into out_dir. Decoded from its proxy when the project has one (the times written are the
    source's own: source_time), its size and length the camera file's."""
    import cv2
    import numpy as np

    from .project import Project
    out = os.path.join(out_dir, Project.score_name(source) + ".npz")
    if os.path.exists(out):
        return f"{source}: cached"
    project = project or Project(folder)
    path = project.decode_from(source, 0.0)[0]
    orig = os.path.join(folder, source)
    dur, w, h = probe(orig if os.path.exists(orig) else path)

    px = (project.config.get("proxies") or {}).get(source) or ()
    shifts = px[1] if len(px) > 1 else ()

    def at(i):                                     # frame i's time in the source (a proxy may have dropped a join)
        return source_time(shifts, i / FPS)
    t0 = time.time()
    T, CAM, SUBJ, SHARP, LUMA, AREA, CX, NP = [], [], [], [], [], [], [], []
    CONTR, RING = [], []
    PT, KP, BX = [], [], []
    prev, pending = None, []                      # pending: (frame index, frame, flow magnitude) awaiting people()

    def flush():
        b, k = people([f for _, f, _ in pending])
        for (fi, f, mag), (boxes, sc), kk in zip(pending, b, k):
            PT.append(at(fi))
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

    for i, f in enumerate(frames(path, w, h, gpu)):
        g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)), cv2.COLOR_RGB2GRAY)
        mag = None
        if prev is not None:
            mag = np.linalg.norm(cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2) * 2
        prev = g
        T.append(at(i)); CAM.append(float(np.median(mag)) if mag is not None else 0.0)
        SUBJ.append(0.0); SHARP.append(0.0); LUMA.append(0.0); AREA.append(0.0); CX.append(0.5); NP.append(0)
        CONTR.append(0.0); RING.append(0.0)
        if i % 2 == 0:                             # people + poses at 5 fps
            pending.append((i, f, mag))
            if len(pending) == 16:
                flush()
    if pending:
        flush()
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{out}.{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, t=np.array(T), cam=np.array(CAM), subj=np.array(SUBJ), sharp=np.array(SHARP),
                            luma=np.array(LUMA), contrast=np.array(CONTR), ring=np.array(RING), area=np.array(AREA),
                            cx=np.array(CX), npeople=np.array(NP),
                            pose_t=np.array(PT), kp=np.array(KP), boxes=np.array(BX), size=np.array([w, h]), dur=dur)
    os.replace(tmp, out)                           # (a worker stopped mid-write leaves no half a file to "resume")
    return f"{source}: {dur:.0f}s in {time.time() - t0:.0f}s" + ("" if path == orig else " (from its proxy)")


def shards(sources: list[tuple[str, float]], n: int) -> list[list[str]]:
    """Split (source, duration) pairs into n lists of roughly equal total duration, longest first."""
    out, tot = [[] for _ in range(n)], [0.0] * n
    for s, d in sorted(sources, key=lambda x: -x[1]):
        k = tot.index(min(tot)); out[k].append(s); tot[k] += d
    return [x for x in out if x]


def run(project, workers: int = 2) -> None:
    """Score every source the units use that has no scores yet, shared over `workers` processes by length."""
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
    from .project import Project
    proj = Project(folder)
    for n in names:
        try:
            try:
                print(score(folder, n, out_dir, project=proj), flush=True)
            except DecodeError as e:               # the GPU gave up on this file: all of it again on the CPU
                print(f"{n}: {e}; on the CPU instead", flush=True)
                print(score(folder, n, out_dir, gpu=False, project=proj), flush=True)
        except Exception as e:                     # keep going; a rerun retries only the missing sources
            print(f"{n}: FAILED {e!r}", flush=True)
