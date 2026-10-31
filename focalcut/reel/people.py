"""Everyone in the footage, by face: so the reel can show every person (at an event where people asked to be
filmed, each of them should be in it).

Frames sampled from each unit's export (2 a second) go through OpenCV's YuNet face detector and SFace recogniser
(both built into cv2; MIT and Apache 2.0 models, downloaded once to ~/.cache/focalcut/faces). Faces are grouped
into people in two levels -- within a unit (average linkage on the embeddings), then those groups across units --
so one person seen in five clips is one person. A person counts once they are seen clearly at least twice with a
face of a usable size; their appearances (unit, source time, face centre and height as fractions of the frame) let
the planner check who a shot shows and put in the best shot of anyone missing (plan.cover_people).

Kisses are found per unit too (SigLIP 2 at a frame a second; face pairs touching without torch): the plan puts one of each couple's in.

Writes <reel>/people.json ({"people": [...], "kisses": {unit: [times]}}) and review/people.jpg (one face per person, numbered, to check nobody was missed).
Per-unit detections are cached in scores/faces_<unit>.npz.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request

import numpy as np

MODELS = {"det": "face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
          "rec": "face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"}
RATE = 2.0                  # frames sampled per second
WIDTH = 1280                # analysis width (landscape; portrait clips: this height)
MIN_FACE = 0.035            # smallest face kept, as a fraction of frame height
PROMINENT = 0.06            # a face this big (fraction of frame height) is someone the reel can feature
SAME_UNIT, SAME_PERSON = 0.38, 0.42   # cosine similarity: same person inside a unit / across units (SFace: 0.363)
CONFIDENT = 0.88            # detector score for a face that counts towards someone being a person


def models() -> dict:
    d = os.path.expanduser("~/.cache/focalcut/faces")
    os.makedirs(d, exist_ok=True)
    out = {}
    for k, m in MODELS.items():
        f = os.path.join(d, os.path.basename(m))
        if not os.path.exists(f):
            urllib.request.urlretrieve("https://huggingface.co/opencv/" + m, f + ".part")
            os.replace(f + ".part", f)
        out[k] = f
    return out


def frames(path, t0, t1, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{max(t1 - t0, 0.1):.3f}", "-i", path,
                          "-vf", f"fps={RATE},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    k = 0
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            break
        yield t0 + k / RATE, np.frombuffer(b, np.uint8).reshape(h, w, 3)
        k += 1
    p.wait()


def detect(project, u, M) -> dict:
    """Faces in one unit: source time, centre x/y and height (fractions of the frame), detector score, sharpness and
    the unit-normalised SFace embedding."""
    import cv2
    out = project.path("scores", f"faces_{u['id']}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    path = os.path.join(project.folder, "export", u["category"], u["id"] + ".mp4")
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[:2])
    s = WIDTH / max(W, H)
    w, h = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    det = cv2.FaceDetectorYN.create(M["det"], "", (w, h), 0.75, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(M["rec"], "")
    x0 = u.get("export_t0", 0.0)
    T, CX, CY, FH, SC, SH, E = [], [], [], [], [], [], []
    for t, f in frames(path, u["t0"] - x0, u["t1"] - x0, w, h):
        _, faces = det.detect(f)
        for fc in faces if faces is not None else []:
            if fc[3] < MIN_FACE * h:
                continue
            crop = rec.alignCrop(f, fc)
            e = rec.feature(crop).ravel()
            T.append(t + x0); CX.append((fc[0] + fc[2] / 2) / w); CY.append((fc[1] + fc[3] / 2) / h)
            FH.append(fc[3] / h); SC.append(fc[14])
            SH.append(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
            E.append(e / (np.linalg.norm(e) + 1e-9))
    d = dict(t=np.array(T, np.float32), cx=np.array(CX, np.float32), cy=np.array(CY, np.float32),
             h=np.array(FH, np.float32), score=np.array(SC, np.float32), sharp=np.array(SH, np.float32),
             emb=np.array(E, np.float32).reshape(-1, 128))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **d)
    return d


def groups(emb: np.ndarray, thr: float) -> np.ndarray:
    """Average-linkage clusters at cosine similarity thr: a label per row."""
    if len(emb) == 0:
        return np.zeros(0, int)
    if len(emb) == 1:
        return np.zeros(1, int)
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(emb, method="average", metric="cosine")
    return fcluster(Z, t=1 - thr, criterion="distance") - 1


def tracks_in(d) -> list:
    """Faces linked over time inside one unit: a face in the next sample (half a second on) close to where this one
    was and about its size is the same person, whatever the lighting does to the embedding. Then tracks that look
    alike (average linkage on their mean embeddings) join: someone who left the frame and came back."""
    n = len(d["t"])
    if not n:
        return []
    order = np.argsort(d["t"], kind="stable")
    track = -np.ones(n, int)
    last = {}                                                    # track id -> index of its latest face
    nt = 0
    times = np.unique(d["t"])
    for t in times:
        here = [i for i in order if d["t"][i] == t]
        cand = [(j, k) for j in here for k, i in last.items()
                if 0 < t - d["t"][i] <= 1.2 / RATE
                and np.hypot(d["cx"][j] - d["cx"][i], (d["cy"][j] - d["cy"][i]) * 0.6) < 0.6 * max(d["h"][i], 0.05)
                and 0.66 < d["h"][j] / max(d["h"][i], 1e-6) < 1.5]
        cand.sort(key=lambda jk: np.hypot(d["cx"][jk[0]] - d["cx"][last[jk[1]]], d["cy"][jk[0]] - d["cy"][last[jk[1]]]))
        used_j, used_k = set(), set()
        for j, k in cand:
            if j in used_j or k in used_k:
                continue
            track[j] = k; last[k] = j; used_j.add(j); used_k.add(k)
        for j in here:
            if track[j] < 0:
                track[j] = nt; last[nt] = j; nt += 1
    ids = np.unique(track)
    cent = np.stack([d["emb"][track == k].mean(0) for k in ids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    lab = groups(cent, SAME_UNIT)
    return [np.nonzero(np.isin(track, ids[lab == g]))[0] for g in np.unique(lab)]


def kisses(d) -> list:
    """Source times in a unit where two faces touch: about the same size, side by side, closer than a face width
    (a kiss, or cheek to cheek)."""
    out = []
    for t in np.unique(d["t"]):
        ix = np.nonzero(d["t"] == t)[0]
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                hi, hj = d["h"][i], d["h"][j]
                if min(hi, hj) < PROMINENT * 0.8 or not 0.6 < hi / hj < 1.66:
                    continue
                if abs(d["cx"][i] - d["cx"][j]) < 0.9 * max(hi, hj) * 0.5625 and abs(d["cy"][i] - d["cy"][j]) < 0.5 * max(hi, hj):
                    out.append(round(float(t), 2))
    return sorted(set(out))


KISS = ["a photo of two people kissing", "a photo of a couple kissing at a party"]
NOT_KISS = ["a photo of friends posing cheek to cheek for a photo", "a photo of people dancing in a crowd",
            "a photo of two people talking", "a photo of a person singing"]


def scan_kisses(project, units, cand) -> dict:
    """Kisses, by SigLIP 2 zero-shot on every unit at one frame a second (kiss against posing, dancing, talking):
    kissing faces are often in profile, where the face detector misses one of them, so the face-pair candidates
    alone would miss kisses. A second's run of hits is one moment; its best frame is kept. Without torch the
    face-pair candidates (cand) stand as they are."""
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        from .faces import MODEL
    except ImportError:
        return cand
    if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    mdl = AutoModel.from_pretrained(MODEL).to(dev, dt).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    out = {}
    with torch.inference_mode():
        tin = proc(text=KISS + NOT_KISS, return_tensors="pt", padding="max_length", max_length=64).to(dev)
        t = mdl.get_text_features(**tin)
        txt = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1)

        def score(imgs):
            pin = proc(images=imgs, return_tensors="pt").to(dev)
            e = mdl.get_image_features(pixel_values=pin["pixel_values"].to(dt))
            im = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
            return (im @ txt.T * 100).softmax(-1)[:, :len(KISS)].sum(-1).cpu().numpy()
        for u in units:
            path = os.path.join(project.folder, "export", u["category"], u["id"] + ".mp4")
            x0 = u.get("export_t0", 0.0)
            T, P, batch = [], [], []
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{u['t0'] - x0:.3f}", "-t",
                                  f"{u['t1'] - u['t0']:.3f}", "-i", path, "-vf", "fps=1,scale=512:-2", "-f",
                                  "image2pipe", "-vcodec", "bmp", "-"], stdout=subprocess.PIPE)
            import io
            data = p.stdout.read()
            p.wait()
            k = 0
            while data[k:k + 2] == b"BM":                        # split the BMP stream
                size = int.from_bytes(data[k + 2:k + 6], "little")
                batch.append(Image.open(io.BytesIO(data[k:k + size])).convert("RGB"))
                T.append(u["t0"] + len(T)); k += size
                if len(batch) == 32:
                    P.extend(score(batch)); batch = []
            if batch:
                P.extend(score(batch))
            hits = [(t, q) for t, q in zip(T, P) if q > 0.5]
            moments = []
            for t, q in hits:
                if moments and t - moments[-1][-1][0] <= 1.5:
                    moments[-1].append((t, q))
                else:
                    moments.append([(t, q)])
            keep = [[round(float(t), 2), round(float(q), 3)] for t, q in (max(m, key=lambda x: x[1]) for m in moments)]
            if keep:
                out[u["id"]] = keep
    return out


def cluster(units: list, D: dict) -> list:
    """People: faces tracked and grouped inside each unit, then the groups across units."""
    tracks = []                                                  # (unit id, indices into its faces)
    for u in units:
        for ix in tracks_in(D[u["id"]]):
            tracks.append((u["id"], ix))
    if not tracks:
        return []
    cent = np.stack([D[uid]["emb"][ix].mean(0) for uid, ix in tracks])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    lab = groups(cent, SAME_PERSON)
    people = []
    for g in np.unique(lab):
        apps = []
        for j in np.nonzero(lab == g)[0]:
            uid, ix = tracks[j]
            d = D[uid]
            for i in ix:
                apps.append(dict(unit=uid, t=round(float(d["t"][i]), 2), cx=round(float(d["cx"][i]), 3),
                                 cy=round(float(d["cy"][i]), 3), h=round(float(d["h"][i]), 3),
                                 q=round(float(d["score"][i]) * min(float(d["sharp"][i]) / 150, 1.0), 3),
                                 score=round(float(d["score"][i]), 3)))
        clear = [a for a in apps if a["h"] >= PROMINENT and a["score"] >= CONFIDENT and a["q"] > 0.3]
        if len(clear) < 2:                                       # glimpsed, or a false detection
            continue
        best = max(clear, key=lambda a: a["h"] * a["q"] * (1.2 - abs(a["cx"] - 0.5)))
        people.append(dict(units=sorted({a["unit"] for a in clear}), seen=len(apps), best=best,
                           appearances=sorted(apps, key=lambda a: (a["unit"], a["t"]))))
    people.sort(key=lambda p: (-p["best"]["h"], p["best"]["unit"]))
    for k, p in enumerate(people):
        p["id"] = k + 1
    return people


def sheet(project, people, units) -> str:
    """review/people.jpg: every person's best face, numbered."""
    import cv2
    by = {u["id"]: u for u in units}
    tiles = []
    for p in people:
        b, u = p["best"], by[p["best"]["unit"]]
        path = os.path.join(project.folder, "export", u["category"], u["id"] + ".mp4")
        f = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{b['t'] - u.get('export_t0', 0.0):.2f}", "-i", path,
                            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) if f else None
        tile = np.zeros((160, 160, 3), np.uint8)
        if img is not None:
            H, W = img.shape[:2]
            s = int(b["h"] * H * 1.6)
            x, y = int(b["cx"] * W), int(b["cy"] * H)
            c = img[max(y - s // 2, 0):y + s // 2, max(x - s // 2, 0):x + s // 2]
            if c.size:
                tile = cv2.resize(c, (160, 160))
        cv2.putText(tile, str(p["id"]), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(tile, str(p["id"]), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        tiles.append(tile)
    cols = 12
    while len(tiles) % cols:
        tiles.append(np.zeros((160, 160, 3), np.uint8))
    out = project.path("review", "people.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if tiles:
        cv2.imwrite(out, np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def run(project) -> None:
    from concurrent.futures import ThreadPoolExecutor
    units = project.load("units.json")
    M = models()
    with ThreadPoolExecutor(4) as ex:
        D = dict(zip([u["id"] for u in units], ex.map(lambda u: detect(project, u, M), units)))
    people = cluster(units, D)
    kiss = scan_kisses(project, units, {u["id"]: k for u in units if (k := kisses(D[u["id"]]))})
    with open(project.path("people.json"), "w") as f:
        json.dump(dict(people=people, kisses=kiss), f)
    n = sum(len(d["t"]) for d in D.values())
    print(f"[reel] people: {len(people)} people from {n} faces in {len(units)} units, kisses in "
          f"{sum(len(v) for v in kiss.values())} moments in {len(kiss)} units -> {sheet(project, people, units)}")
