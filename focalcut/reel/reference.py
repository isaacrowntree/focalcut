"""Learn an edit from a finished film: `focalcut reel reference <folder> <video>`.

A client often hands over last year's film and asks for "the same again". This reads that film and writes
reference.json, a template the planner fills from this project's own footage (reel.json "mode": "reference"):

  shots     every cut (ffmpeg's scene score), each shot's length and scale -- aerial, wide, medium, close (SigLIP 2
            zero-shot) -- and what it is made of: a split screen (its panels, found from the seams that run the whole
            height or width of the frame), a card (a logo on a plain ground: low detail, held), a stutter (a run of
            jump-cuts too short to read, one slot), a title (text fading in over a still shot) and whether it plays
            after the music has ended (a tag that needs its own sound)
  category  which of this project's export categories each shot is like: the reference's frame against frames of
            every unit (SigLIP 2 image embeddings), kept in the order things happened (a Viterbi path through the
            categories' wall-clock spans, so the walk to the beach never cuts back to the car park)
  look      the film's colour, per group of shots (aerial and ground): Lab statistics and a tone curve, made into a
            .cube LUT per camera group of this project that moves its footage to it (reel.json "lut")

The planner (template.py) then cuts this project's footage on the reference's own cut points, shot for shot. With
the same song the cuts land where the editor put them; with another, the reference's timeline is scaled to it.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess

import numpy as np

MODEL = "google/siglip2-base-patch16-512"
SCENE = 0.25             # ffmpeg scene score that makes a cut
STUTTER = 0.35           # shots shorter than this, three or more in a row, are one stutter (jump-cuts on a hit)
SEAM = 0.55              # a split screen's seam: this share of the rows (columns) jump in brightness at one column
CARD_DETAIL = 4.0        # mean gradient (0-255) under which a held frame is a card, not footage
SAMPLE = 2.0             # seconds between the unit frames the reference is matched against
SLACK = 60.0             # a category may come back after another that started up to this many seconds later
SCALES = {
    "aerial": ["an aerial drone photo looking down on a beach and the sea", "a drone view from high above the coast"],
    "wide": ["a wide shot of a crowd of people on a beach", "a wide landscape photo of a bay"],
    "medium": ["a photo of a few people standing together", "people waist deep in the sea"],
    "close": ["a close-up portrait of a person's face", "a close-up of two people smiling"],
}


# -- the film itself ---------------------------------------------------------------------------------------------
def probe(video):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate:format=duration", "-of", "json", video]))
    s = j["streams"][0]
    n, d = (int(x) for x in s["r_frame_rate"].split("/"))
    return float(j["format"]["duration"]), int(s["width"]), int(s["height"]), n / d


def cuts(video, scene=SCENE):
    """Cut times (s) by ffmpeg's scene score."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-an", "-vf",
                          f"scale=320:-2,select='gt(scene,{scene})',showinfo", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return [float(x) for x in re.findall(r"pts_time:([0-9.]+)", err)]


def shot_spans(times, duration, least=0.1):
    """[t0, t1] per shot from cut times; a 'shot' under `least` s (a flash frame) joins the one before."""
    edges = [0.0] + [t for t in sorted(times) if 0 < t < duration] + [duration]
    out = []
    for a, b in zip(edges, edges[1:]):
        if out and b - a < least:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def stutters(spans, short=STUTTER, run=3):
    """Group ids for runs of `run` or more consecutive shots each under `short` s (plus the one they lead into
    when it is short too); None for a shot outside one."""
    g, out, i = 0, [None] * len(spans), 0
    while i < len(spans):
        j = i
        while j < len(spans) and spans[j][1] - spans[j][0] < short:
            j += 1
        if j - i >= run:
            for x in range(i, j):
                out[x] = g
            g += 1
            i = j
        else:
            i += 1
    return out


def frame(video, t, w=None):
    """One RGB frame at t (s), full size or w pixels wide."""
    d, W, H, _ = probe(video) if w is None else (0, w, None, 0)
    vf = f"scale={w}:-2" if w else "null"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", vf,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    if w is None:
        return np.frombuffer(raw, np.uint8).reshape(H, W, 3)
    h = len(raw) // (w * 3)
    return np.frombuffer(raw, np.uint8)[: h * w * 3].reshape(h, w, 3)


def frame_times(video):
    """Every video frame's time (s), in the order shown: read from the packets, nothing decoded."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout
    ts = []
    for line in out.split():
        try:
            ts.append(float(line.strip(",")))
        except ValueError:
            pass
    return np.sort(np.array(ts))


def frames_at(video, times, w=None):
    """The frames at `times` (s), full size or w wide, in one pass over the film: each the first frame at or after
    its time, as frame() gives -- the same frame through the same scaler, so the same pixels -- without a seek and
    a decoder start for every one (a 90-shot film asks for 270: a minute of seeking, against one 8 s decode). The
    frames are picked by number, from the film's own frame times, so a variable frame rate is read right too.
    Yields RGB arrays in the order asked, one in memory at a time; whatever the pass could not give is read
    frame() by frame()."""
    times = [round(float(t), 3) for t in times]              # (frame()'s own precision: the same frame)
    done = 0
    pts = frame_times(video) if times else []
    if len(pts) and all(b >= a for a, b in zip(times, times[1:])):
        _, W, H, _ = probe(video)
        if w:
            H = len(frame(video, times[0], w))               # (the height ffmpeg's scale=w:-2 gives)
        W = w or W
        num = np.searchsorted(pts, np.array(times) - 5e-7)   # (frame() drops frames before t, to the microsecond)
        if num[-1] < len(pts):
            vf = "select='" + "+".join(f"eq(n\\,{k})" for k in sorted(set(num.tolist()))) + "'" \
                + (f",scale={w}:-2" if w else "")
            n = W * H * 3
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-an", "-vf", vf, "-fps_mode", "passthrough",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
            try:
                last, img = None, None
                for k in num:
                    if k != last:
                        b = p.stdout.read(n)
                        if len(b) < n:
                            break
                        last, img = k, np.frombuffer(b, np.uint8).reshape(H, W, 3)
                    done += 1
                    yield img
            finally:
                p.stdout.close()
                p.kill()
                p.wait()
    for t in times[done:]:
        yield frame(video, t, w)


def _seams(img, axis, share=SEAM, edge=0.04, stand=3.0):
    """Positions (0..1) where a seam runs across the whole frame: between columns (axis=1) or rows (axis=0), the
    picture jumps -- in any colour channel, so two panels alike in brightness but not in colour still part -- in at
    least `share` of the lines, and `stand` times as often as the frame's own texture jumps nearby (busy footage
    alone never makes a seam). A gutter's two edges count once."""
    g = img.astype(np.float32)
    if g.ndim == 2:
        g = g[..., None]
    d = (np.abs(np.diff(g, axis=axis)) > 18).any(axis=2)      # a jump between neighbouring pixels, any channel
    frac = d.mean(axis=1 - axis)                              # per column (row) boundary: share of lines that jump
    n = len(frac)
    k = max(n // 40, 3)
    near = np.array([np.median(np.r_[frac[max(i - k, 0):max(i - 1, 0)], frac[i + 2:i + k + 1]] if n > 4 else frac)
                     for i in range(n)])
    found = []
    for i in np.argsort(-frac):
        if frac[i] < share:
            break
        x = (i + 1) / (n + 1)
        if frac[i] < stand * near[i]:
            continue
        if edge < x < 1 - edge and all(abs(x - y) > 0.03 for y in found):
            found.append(x)
    return sorted(found)


def split_layout(img, depth=2):
    """A split screen's panels as [x, y, w, h] (0..1), or None for a single picture. Guillotine cuts: the seams
    running the whole height split it into columns, each column's own whole-width seams into rows, and so on."""
    gray = img if img.ndim == 3 else img[..., None]

    def part(x0, y0, x1, y1, d):
        H, W = gray.shape[:2]
        sub = gray[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        if d == 0 or min(sub.shape[:2]) < 24:
            return [[x0, y0, x1 - x0, y1 - y0]]
        v = _seams(sub, 1)
        if v:
            xs = [x0] + [x0 + s * (x1 - x0) for s in v] + [x1]
            return [r for a, b in zip(xs, xs[1:]) for r in part(a, y0, b, y1, d - 1)]
        h = _seams(sub, 0)
        if h:
            ys = [y0] + [y0 + s * (y1 - y0) for s in h] + [y1]
            return [r for a, b in zip(ys, ys[1:]) for r in part(x0, a, x1, b, d - 1)]
        return [[x0, y0, x1 - x0, y1 - y0]]
    rects = part(0.0, 0.0, 1.0, 1.0, depth)
    rects = [[round(v, 3) for v in r] for r in rects if r[2] > 0.12 and r[3] > 0.12]
    return rects if len(rects) > 1 else None


def detail(img):
    """Mean absolute gradient (0-255): footage is busy, a card is not."""
    g = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2


def card_spec(img):
    """A card's ground (top and bottom colour) and the box of what sits on it (the logo), 0..1."""
    top, bot = img[: max(img.shape[0] // 20, 1)].reshape(-1, 3).mean(0), img[-max(img.shape[0] // 20, 1):].reshape(-1, 3).mean(0)
    from scipy.ndimage import median_filter, zoom
    small = img[::8, ::8].astype(float)                      # the ground: a heavy median smooths the logo away
    ground = median_filter(small, size=(15, 15, 1))
    ground = zoom(ground, (img.shape[0] / small.shape[0], img.shape[1] / small.shape[1], 1), order=1)
    ground = ground[: img.shape[0], : img.shape[1]]
    ink = np.abs(img[: ground.shape[0], : ground.shape[1]].astype(float) - ground).sum(2) > 60
    ys, xs = np.nonzero(ink)
    box = None
    if len(xs):
        H, W = ink.shape
        box = [round(xs.min() / W, 3), round(ys.min() / H, 3), round((xs.max() - xs.min()) / W, 3),
               round((ys.max() - ys.min()) / H, 3)]
    return dict(top=[int(v) for v in top], bottom=[int(v) for v in bot], box=box)


def _grad(img):
    g = img.mean(2) if img.ndim == 3 else img
    gy, gx = np.gradient(g.astype(float))
    return np.hypot(gx, gy)


def _new_ink(a, b):
    """Where b has sharp strokes on ground that was smooth in a: text (or a logo) laid over the picture, which a
    drifting camera alone doesn't make (its edges were there a few pixels away already)."""
    from scipy.ndimage import maximum_filter
    return (_grad(b) > 25) & (maximum_filter(_grad(a), 7) < 10)


def _band(m, rows_min=0.02, gap=12):
    """The densest band of rows of a mask (runs closer than `gap` rows joined), and its columns: (y0, y1, x0, x1)."""
    H = m.shape[0]
    on = np.nonzero(m.mean(1) > rows_min)[0]
    if not len(on):
        return None
    runs, start = [], on[0]
    for x, y in zip(on, on[1:]):
        if y - x > gap:
            runs.append((start, x + 1)); start = y
    runs.append((start, on[-1] + 1))
    y0, y1 = max(runs, key=lambda r: m[r[0]:r[1]].sum())
    cols = np.nonzero(m[y0:y1].mean(0) > rows_min)[0]
    if not len(cols) or (y1 - y0) / H > 0.45:
        return None
    return int(y0), int(y1), int(cols.min()), int(cols.max()) + 1


def card_fades(video, t0, t1, plain, step=0.25):
    """How long a card takes to come in (a dissolve from the shot before: detail still above the card's own) and to
    go out, in seconds."""
    def busy(t):
        return detail(frame(video, t, 480)) > max(plain * 1.6, plain + 0.6)
    n = int(min(3.0, (t1 - t0) / 3) / step)
    fin = next((k * step for k in range(n) if not busy(t0 + 0.04 + k * step)), 0.0)
    fout = next((k * step for k in range(n) if not busy(t1 - 0.04 - k * step)), 0.0)
    return [round(fin, 2), round(fout, 2)]


def title_in(video, t0, t1, w=640):
    """Text fading in over a shot: the box (0..1) of the strokes that appear on smooth ground between the shot's
    first frame and later in it, and the fade (start, end) in seconds from the shot's start. None without one."""
    a = frame(video, t0 + 0.05, w).astype(float)
    b = frame(video, t0 + min(2.5, (t1 - t0) * 0.75), w).astype(float)
    m = _new_ink(a, b)
    band = _band(m)
    if band is None or m.mean() < 0.002:
        return None
    y0, y1, x0, x1 = band
    H, W = m.shape
    box = [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]
    region = np.zeros_like(m); region[y0:y1, x0:x1] = m[y0:y1, x0:x1]
    ts = np.linspace(0.05, min(3.0, t1 - t0 - 0.05), 12)
    seen = np.array([float(_new_ink(a, frame(video, t0 + t, w).astype(float))[region].mean()) for t in ts])
    seen = seen / max(seen.max(), 1e-6)
    start = float(ts[np.argmax(seen > 0.1)]) if (seen > 0.1).any() else 0.0
    end = float(ts[np.argmax(seen > 0.9)]) if (seen > 0.9).any() else start + 1.0
    return dict(box=[round(float(v), 3) for v in box], fade=[round(start, 2), round(max(end, start + 0.3), 2)],
                lines=_lines(m[y0:y1, x0:x1], y0 / H, H))


def _lines(m, top, H, gap=3):
    """The text's lines: [y (0..1, centre), height (0..1)] per run of inked rows."""
    on = m.mean(1) > 0.02
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j:j + gap].any():
                j += 1
            if j - i >= 3:
                out.append([round(top + (i + j) / 2 / H, 4), round((j - i) / H, 4)])
            i = j
        i += 1
    return out


def music_end(video):
    """Where the film's music stops (s): the start of the last silence (under -50 dB for 0.25 s or more) after which
    the sound comes back, if it does -- a tag playing its own sound after the score -- else the film's end."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    if not len(a):
        return None
    hop = 400                                                # 50 ms
    rms = np.sqrt(np.add.reduceat(a[: len(a) // hop * hop] ** 2, np.arange(0, len(a) // hop * hop, hop)) / hop)
    db = 20 * np.log10(rms + 1e-9)
    quiet = db < -50
    best = None
    i = int(len(db) * 0.6)
    while i < len(db):
        if quiet[i]:
            j = i
            while j < len(db) and quiet[j]:
                j += 1
            if j - i >= 5 and j < len(db) - 10:
                best = i * hop / 8000
            i = j
        i += 1
    return best if best is not None else len(a) / 8000


# -- colour -------------------------------------------------------------------------------------------------------
def _lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055) * 255.0


M_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    xyz = _lin(np.asarray(rgb, float)) @ M_XYZ.T / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * WHITE
    return np.clip(_gam(xyz @ np.linalg.inv(M_XYZ).T), 0, 255)


QS = np.linspace(1, 99, 25)


def stats(imgs):
    """A group of frames' colour: L quantiles (the tone curve) and the a/b means and spreads."""
    lab = np.concatenate([rgb2lab(im[::4, ::4].reshape(-1, 3)) for im in imgs])
    return dict(Lq=[round(float(v), 3) for v in np.percentile(lab[:, 0], QS)],
                a=[round(float(lab[:, 1].mean()), 3), round(float(lab[:, 1].std()), 3)],
                b=[round(float(lab[:, 2].mean()), 3), round(float(lab[:, 2].std()), 3)])


def transfer(src, dst, strength=1.0, spread=(0.5, 3.0)):
    """f(lab) -> lab moving footage with colour `src` (stats) to `dst`: L by its quantiles (a monotone tone curve,
    ends held), a and b by mean and spread (the spread ratio clipped, so a flat log picture may be stretched but no
    channel is blown up). strength: 0 is the footage untouched, 1 the full match."""
    sq, dq = np.maximum.accumulate(np.array(src["Lq"])), np.maximum.accumulate(np.array(dst["Lq"]))
    xs = np.r_[0.0, sq + np.arange(len(sq)) * 1e-6, 100.0]
    ys = np.r_[0.0, dq, 100.0]
    ys = np.maximum.accumulate(ys)

    def f(lab):
        out = lab.copy()
        out[..., 0] = np.interp(lab[..., 0], xs, ys)
        for k, c in ((1, "a"), (2, "b")):
            r = np.clip(dst[c][1] / max(src[c][1], 1e-3), *spread)
            out[..., k] = (lab[..., k] - src[c][0]) * r + dst[c][0]
        return lab + strength * (out - lab)
    return f


def write_cube(path, f, n=33):
    """A .cube 3D LUT for f (lab -> lab) over Rec.709 RGB, red fastest."""
    g = np.linspace(0, 255, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    out = lab2rgb(f(rgb2lab(rgb))) / 255.0
    with open(path, "w") as fh:
        fh.write(f"TITLE \"focalcut reference look\"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        fh.writelines(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in out)
    return path


# -- matching to this project's footage --------------------------------------------------------------------------
class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
            torch.backends.cudnn.enabled = False
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(MODEL).to(self.dev, self.dt).eval()
        self.proc = AutoProcessor.from_pretrained(MODEL)

    def images(self, imgs, batch=32):
        from PIL import Image
        out = []
        for i in range(0, len(imgs), batch):
            with self.torch.inference_mode():
                inp = self.proc(images=[Image.fromarray(x) for x in imgs[i:i + batch]], return_tensors="pt").to(self.dev)
                inp["pixel_values"] = inp["pixel_values"].to(self.dt)
                e = self.model.get_image_features(**inp)
                e = getattr(e, "pooler_output", e).float()
                out.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def texts(self, prompts):
        with self.torch.inference_mode():
            tin = self.proc(text=prompts, return_tensors="pt", padding="max_length", max_length=64).to(self.dev)
            e = self.model.get_text_features(**tin)
            e = getattr(e, "pooler_output", e).float()
            return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


HIGH = ["an aerial drone photo from high above a beach and the sea", "a drone view high over the ocean and coast",
        "a top-down aerial view of people in the sea from high above"]
LOW = ["a photo of sand on the ground close up", "a blurry photo of a beach from just above the sand",
       "a person's hand reaching for a drone", "a view from ground level of a beach"]


def aerial_high(E, emb, scale=25.0):
    """Per frame (0..1): how much it reads as flown high above the beach and sea rather than low over the sand (a
    drone taking off, landing or being caught). SigLIP 2 zero-shot, the two prompt sets' best against each other."""
    hi, lo = E.texts(HIGH), E.texts(LOW)
    d = (emb @ hi.T).max(1) - (emb @ lo.T).max(1)
    return 1 / (1 + np.exp(-scale * d))


def scale_of(emb, text_emb, names):
    """The shot's scale: the zero-shot class whose prompts sit nearest (mean over each class's prompts)."""
    sims = emb @ text_emb.T
    per = {}
    for n, s in zip(names, sims):
        per.setdefault(n, []).append(s)
    return max(per, key=lambda n: np.mean(per[n]))


def unit_export(project, u):
    """The export file a unit came from (its proxy here), and the source time its first frame shows."""
    name = u["id"].split("#")[0]
    p = os.path.join(project.export_dir, u["category"], name + ".mp4")
    return (p, u.get("export_t0", 0.0)) if os.path.exists(p) else (os.path.join(project.folder, u["source"]), 0.0)


STORE = 2                # reference_frames.npz's layout: 2 -- "t" is each frame's own time (1: up to a step early)
GRAB_JOBS = 6            # units read at once by unit_frames (each a keyframe-only decode: light, and mostly reading)
SPARSE = 1.6             # a file whose keyframes are further apart than this many sample steps is decoded whole


def _grab(path, a, dur, every, w):
    """(times from a, frames n x h x w x 3) of a stretch of a file, about every `every` s: its keyframes alone when
    they come often enough (an export has one every 0.5-1 s: reading them is a twentieth of the CPU time of decoding
    every frame to keep one in a hundred), else every frame decoded. The times are the frames' own (showinfo)."""
    _, W, H, _ = probe(path)
    h = int(round(H * w / W / 2) * 2)
    vf = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{every - 0.02:.3f})',showinfo,scale={w}:-2"
    ts, fr = [], np.zeros((0, h, w, 3), np.uint8)
    for keys in (["-skip_frame", "nokey"], []):             # (not "-discard nokey" too: quicker still, and the
        #                                                     times it reports are wrong by up to a second)
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", *keys, "-ss", f"{a:.3f}",
                            "-t", f"{dur:.3f}", "-i", path, "-an", "-vf", vf, "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        ts = [float(x) for x in re.findall(rb"Parsed_showinfo.*?pts_time:\s*(-?[0-9.]+)", r.stderr)]
        n = min(len(r.stdout) // (w * h * 3), len(ts))
        ts, fr = ts[:n], np.frombuffer(r.stdout, np.uint8)[: n * w * h * 3].reshape(n, h, w, 3)
        gaps = np.diff([*ts, dur]) if n else np.array([dur])
        if n and gaps.max() <= SPARSE * every:               # (keyframes far apart: every frame then)
            break
    return ts, fr


def unit_frames(project, units, every=SAMPLE, w=384):
    """Frames about every `every` seconds of each unit, from its export: (unit index, source time, RGB) lists, the
    times the frames' own. GRAB_JOBS units are read at once."""
    from concurrent.futures import ThreadPoolExecutor

    def one(u):
        path, x0 = unit_export(project, u)
        a = max(u["t0"] - x0, 0.0)
        ts, fr = _grab(path, a, u["t1"] - x0 - a, every, w)
        return [x0 + a + t for t in ts], fr
    idx, times, imgs = [], [], []
    with ThreadPoolExecutor(GRAB_JOBS) as ex:
        for k, (ts, fr) in enumerate(ex.map(one, units)):
            for t, f in zip(ts, fr):
                idx.append(k); times.append(round(t, 3)); imgs.append(f)
    return idx, times, imgs


def embed_units(project, E, units, idx, times, imgs):
    """SigLIP 2 embeddings of unit_frames' frames, taking from reference_frames.npz those it already holds: the
    same unit at the same time (to 0.1 s), stored after the unit's export file was last written. Only a store
    whose times are the frames' own (STORE 2) is used: an older one's frames were up to a sample step later than
    the times written beside them. Returns (the embeddings, how many were reused)."""
    out = np.zeros((len(imgs), 768), np.float32)
    todo = list(range(len(imgs)))
    store = project.path("reference_frames.npz")
    if os.path.exists(store):
        z = np.load(store)
        if "store" in z and int(z["store"]) == STORE and z["emb"].shape[1:] == out.shape[1:]:
            have = {(str(u), round(float(t), 1)): k for k, (u, t) in enumerate(zip(z["unit"], z["t"]))}
            since = os.path.getmtime(store)
            fresh = {u["id"]: os.path.getmtime(p) <= since for u in units
                     for p in [unit_export(project, u)[0]] if os.path.exists(p)}
            emb, todo = z["emb"], []
            for j, (k, t) in enumerate(zip(idx, times)):
                row = have.get((units[k]["id"], round(t, 1))) if fresh.get(units[k]["id"]) else None
                if row is None:
                    todo.append(j)
                else:
                    out[j] = emb[row]
    if todo:
        new = E.images([imgs[j] for j in todo])
        if new.shape[1] != out.shape[1]:                      # (another model's width: nothing stored fits it)
            return E.images(imgs), 0
        out[todo] = new
    return out, len(imgs) - len(todo)


def wall_clock(project, u):
    """When a unit's footage was shot (epoch seconds): the file's creation time plus reel.json "clock" offsets by
    clip-name pattern ({"C*": -3600}: a camera set to the wrong zone). None when unknown."""
    src = os.path.join(project.folder, u["source"])
    try:
        tag = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
                                                  "-of", "json", src]))["format"]["tags"]["creation_time"]
    except Exception:
        return None
    from datetime import datetime
    t = datetime.fromisoformat(tag.replace("Z", "+00:00")).timestamp()
    for pat, off in (project.config.get("clock") or {}).items():
        if fnmatch.fnmatch(u["clip"], pat):
            t += off
    return t + u["t0"]


def viterbi(sim, spans, slack=SLACK, stay=0.02):
    """The category per shot: best total similarity over a path that keeps to the order things happened. sim:
    (shots, categories); spans: per category (start, end) wall clock or None (floats anywhere). A shot may follow
    another of a category that had not ended before this one's started (they overlap in time, or this one began
    later, or within `slack` s); staying in a category is worth `stay` (no needless flicker between two)."""
    n, k = sim.shape
    ok = np.ones((k, k), bool)
    for a in range(k):
        for b in range(k):
            if spans[a] is not None and spans[b] is not None and a != b:
                ok[a, b] = spans[b][1] >= spans[a][0] - slack
    score, back = sim[0].copy(), np.zeros((n, k), int)
    for i in range(1, n):
        cand = score[:, None] + np.where(ok, 0.0, -1e9) + np.eye(k) * stay
        back[i] = cand.argmax(0)
        score = cand.max(0) + sim[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


def forward(path, sim, spans, slack=SLACK, wide=2700.0):
    """The path with time only moving forward: a cursor at the latest start of the categories used so far (only
    those of a part of the day -- a category spread over more than `wide` s, B-roll or a drone flying all
    morning, never moves it), and a shot whose category had ended before the cursor (by more than `slack` s) takes
    the most similar category that hadn't. viterbi() alone can be walked back through such a spread category."""
    cur, out = None, []
    for i, c in enumerate(path):
        def ok(k, cur=cur):
            return spans[k] is None or cur is None or spans[k][1] >= cur - slack
        if not ok(c):
            c = max((k for k in range(sim.shape[1]) if ok(k)), key=lambda k: sim[i, k], default=c)
        out.append(c)
        sp = spans[c]
        if sp is not None and sp[1] - sp[0] <= wide:
            cur = sp[0] if cur is None else max(cur, sp[0])
    return out


# -- the whole pass -----------------------------------------------------------------------------------------------
def analyse(video):
    """The film's shots, each with what can be read from the picture alone."""
    dur, W, H, fps = probe(video)
    spans = shot_spans(cuts(video), dur)
    st = stutters(spans)
    shots = []
    from concurrent.futures import ThreadPoolExecutor

    from . import moves

    def move(span):                                         # the camera's move over the shot (moves.py)
        a, b = span
        return moves.label(moves.frames_moves(moves.grey(video, a + 0.05, b - a - 0.1, gpu=False)))[0]
    with ThreadPoolExecutor(GRAB_JOBS) as ex:               # (each shot its own short decode: several at once)
        moved = dict(zip([i for i, (a, b) in enumerate(spans) if b - a >= 0.6],
                         ex.map(move, [sp for sp in spans if sp[1] - sp[0] >= 0.6])))
    mids = frames_at(video, [(a + b) / 2 for a, b in spans])   # each shot's middle frame, one pass over the film
    for i, ((a, b), mid) in enumerate(zip(spans, mids)):
        s = dict(i=i, t0=round(a, 3), t1=round(b, 3), length=round(b - a, 3))
        if st[i] is not None:
            s["stutter"] = st[i]
        if i in moved:
            s["move"] = moved[i]
        lay = split_layout(mid)
        if lay:
            s["split"] = lay
        if b - a >= 2.0 and detail(mid) < CARD_DETAIL:
            c = card_spec(mid)                               # a plain ground with something small on it (a logo);
            if c["box"] and c["box"][2] * c["box"][3] < 0.4:  # soft-focus footage is plain too, but fills the frame
                c["fade"] = card_fades(video, a, b, detail(mid))
                s["card"] = c
        shots.append(s)
    if shots:                                                # lettering fading in on the film's first shot is
        t = title_in(video, *spans[0])                       # its title, plain ground or not (a card is the
        if t and len(shots) > 1:                             # logo held at the close)
            shots[0]["title"] = t
            shots[0].pop("card", None)
    return dict(video=os.path.abspath(video), duration=round(dur, 3), size=[W, H], fps=round(fps, 3),
                music_end=music_end(video), shots=shots)


def match(project, ref, units, unit_ids, remb, uemb):
    """Each reference shot's category: its embedding against the frames of each category's units (their best few),
    along the Viterbi path that keeps to the order things happened (the categories' wall-clock spans)."""
    shots = ref["shots"]
    cats = sorted({u["category"] for u in units})
    clocks = {}
    for u in units:
        w = wall_clock(project, u)
        if w is not None:
            c = clocks.setdefault(u["category"], [w, w + u["t1"] - u["t0"]])
            c[0], c[1] = min(c[0], w), max(c[1], w + u["t1"] - u["t0"])
    sim = np.full((len(shots), len(cats)), -1.0)
    cat_of_unit = {u["id"]: u["category"] for u in units}
    cat_of = np.array([cats.index(cat_of_unit[i]) if i in cat_of_unit else -1 for i in unit_ids])
    for c in range(len(cats)):
        m = cat_of == c
        if m.any():
            s = remb @ uemb[m].T                               # each reference shot against the category's frames:
            top = np.sort(s, axis=1)[:, -min(5, m.sum()):]     # its best few matches (one good frame is luck)
            sim[:, c] = top.mean(1)
    playable = [i for i, s in enumerate(shots) if not s.get("card")]
    spans = [clocks.get(c) for c in cats]
    path = forward(viterbi(sim[playable], spans), sim[playable], spans)
    for i, c in zip(playable, path):
        shots[i]["category"] = cats[c]
        shots[i]["sim"] = round(float(sim[i, c]), 3)
    ref["categories"] = {c: dict(clock=clocks.get(c)) for c in cats}
    return ref


def rematch(project, looks_too=False, strength=0.5):
    """Match the reference to the footage again from the stored embeddings (reference_frames.npz): after the
    export, the clock offsets or SLACK changed. Units added since (a second shooter's clips) are embedded now;
    nothing else is read again. looks_too: the camera LUTs are rebuilt as well (frames of every camera, sampled)."""
    ref = project.load("reference.json")
    z = dict(np.load(project.path("reference_frames.npz")))
    units = project.load("units.json")
    have = {str(u) for u in z["unit"]}
    new = [u for u in units if u["id"] not in have]
    E = None
    if new:
        E = Embedder()
        idx, times, imgs = unit_frames(project, new)
        emb = E.images(imgs)
        z["unit"] = np.concatenate([z["unit"].astype(str), np.array([new[k]["id"] for k in idx], dtype=str)])
        z["t"] = np.concatenate([z["t"], np.array(times)])
        z["emb"] = np.concatenate([z["emb"], emb.astype(np.float16)])
        np.savez_compressed(project.path("reference_frames.npz"), **z)
        print(f"[reel] reference: embedded {len(new)} new units ({len(imgs)} frames)")
    if "high" not in z or new:                                # each frame: flying high, or low over the ground?
        E = E or Embedder()
        z["high"] = aerial_high(E, z["emb"].astype(np.float32)).astype(np.float16)
        np.savez_compressed(project.path("reference_frames.npz"), **z)
    ids = [str(u) for u in z["unit"]]
    match(project, ref, units, ids, z["ref"].astype(np.float32), z["emb"].astype(np.float32))
    if looks_too:
        sample = []
        for u in units:                                        # a frame every 6 s is plenty for a camera's colour
            sample.append(u)
        idx, times, imgs = unit_frames(project, sample, every=6.0)
        E = E or Embedder()
        uemb = E.images(imgs)
        ref["luts"] = looks(project, ref, sample, idx, imgs, uemb, strength)
    project.dump("reference.json", ref)
    return ref


def run(project, video, strength=0.5):
    """Analyse the film, match it to this project's units, and write reference.json and the look LUTs."""
    ref = analyse(video)
    me = ref["music_end"]
    for s in ref["shots"]:
        if me is not None and s["t0"] >= me - 0.5:
            s["after_music"] = True
    E = Embedder()
    shots = ref["shots"]
    at = [(s["t0"] + s["t1"]) / 2 for s in shots]
    mids, full = list(frames_at(video, at, 384)), list(frames_at(video, at, 640))   # (a pass over the film each)
    remb = E.images(mids)
    names = [n for n, ps in SCALES.items() for _ in ps]
    temb = E.texts([p for ps in SCALES.values() for p in ps])
    for s, e in zip(shots, remb):
        s["scale"] = "card" if s.get("card") else "split" if s.get("split") else scale_of(e, temb, names)
    units = project.load("units.json")
    idx, times, imgs = unit_frames(project, units)
    uemb, kept = embed_units(project, E, units, idx, times, imgs)
    if kept:
        print(f"[reel] reference: {kept} of {len(imgs)} unit frames were embedded already (reference_frames.npz)")
    np.savez_compressed(project.path("reference_frames.npz"), unit=np.array([units[k]["id"] for k in idx]),
                        t=np.array(times), emb=uemb.astype(np.float16), ref=remb.astype(np.float16),
                        store=np.array(STORE))
    match(project, ref, units, [units[k]["id"] for k in idx], remb, uemb)
    # the look: the film's ground and aerial shots' colour, and this project's cameras matched to each other first
    groups = {"aerial": [f for f, s in zip(full, shots) if s.get("scale") == "aerial"],
              "ground": [f for f, s in zip(full, shots) if s.get("scale") in ("wide", "medium", "close")]}
    ref["look"] = {g: stats(v) for g, v in groups.items() if v}
    luts = looks(project, ref, units, idx, imgs, uemb, strength)
    ref["luts"] = luts
    project.dump("reference.json", ref)
    return ref


def looks(project, ref, units, idx, imgs, uemb, strength=0.5, hero=None):
    """A .cube per camera: every camera matched to the hero camera's colour (reel.json "look_hero", a clip pattern;
    default the camera with the most footage) on frames of the same kind of scene -- for each of its frames, the hero's
    nearest few by SigLIP 2, so a drone's sea is matched to the hero's sea, not to its faces -- then the whole film moved
    `strength` of the way from the hero's colour to the reference's ground shots. The hero's own LUT is that look alone.
    reel.json "look_strength" overrides strength; 0 makes the hero's LUT the identity (the others still match it).
    reel.json "look_method": "regions" matches by semantic region instead (regions.py)."""
    strength = float(project.config.get("look_strength", strength))
    cams = camera_groups(units, project.config.get("cameras"))
    of = {c: np.array([any(fnmatch.fnmatch(units[k]["clip"], p) for p in pats) for k in idx]) for c, pats in cams.items()}
    hero = hero or project.config.get("look_hero")
    if hero:
        hero = next((c for c, pats in cams.items() if hero == c or hero in pats
                     or any(fnmatch.fnmatch(hero.rstrip("*") + "0", p) for p in pats)), hero)
    if hero not in cams:
        hero = max(cams, key=lambda c: of[c].sum())
    if project.config.get("look_method") == "regions":         # matched on the same things in both cameras (SAM 3
        from . import regions

        return regions.looks(project, units, cams, hero)       # regions): the hero as shot, no reference look on top
    H = np.nonzero(of[hero])[0]
    hero_stats = stats([imgs[j] for j in H[::3]])
    target = ref.get("look", {}).get("ground")
    look = transfer(hero_stats, target, strength) if target else (lambda lab: lab)
    look_dir = project.path("looks")
    os.makedirs(look_dir, exist_ok=True)
    luts = {}
    for cam, pats in cams.items():
        mine = np.nonzero(of[cam])[0]
        if not len(mine):
            continue
        if cam == hero:
            if strength <= 0:                                  # the hero camera as shot: no LUT at all
                luts[cam] = dict(match=pats, cube=None, how="as shot (the hero camera)")
                continue
            f, how = look, "the reference's look"
        else:
            sim = uemb[mine] @ uemb[H].T                        # the hero's frames of the same kind of scene
            near = np.unique(H[np.argsort(-sim, axis=1)[:, :3].ravel()])
            fit, test = mine[0::2], mine[1::2]                  # matched on half its frames, checked on the rest
            target = stats([imgs[j] for j in near])
            match = transfer(stats([imgs[j] for j in fit]), target, 1.0)
            f = (lambda lab, m=match: look(m(lab))) if strength > 0 else match
            how = f"matched to {hero}" + (", then the reference's look" if strength > 0 else "")
            test = test if len(test) else fit
            held = [imgs[j] for j in test]
            near_t = np.unique(H[np.argsort(-(uemb[test] @ uemb[H].T), axis=1)[:, :3].ravel()])
            target = stats([imgs[j] for j in near_t])           # the hero on the same scenes as the held-out frames
            after = [lab2rgb(f(rgb2lab(im[::4, ::4].reshape(-1, 3).astype(float)))).reshape(im[::4, ::4].shape)
                     for im in held]
            check = dict(before=distance(stats(held), target), after=distance(stats(after), target))
        if cam != hero and cam in (project.config.get("match_per_clip") or ()):   # a camera whose clips vary
            for clip in sorted({units[idx[j]]["clip"] for j in mine}):           # too much for one LUT: one each
                cj = np.array([j for j in mine if units[idx[j]]["clip"] == clip])
                nj = np.unique(H[np.argsort(-(uemb[cj] @ uemb[H].T), axis=1)[:, :4].ravel()])
                tgt = stats([imgs[j] for j in nj])
                fc = transfer(stats([imgs[j] for j in cj]), tgt, 1.0)
                after = [lab2rgb(fc(rgb2lab(imgs[j][::4, ::4].reshape(-1, 3).astype(float)))).reshape(
                    imgs[j][::4, ::4].shape) for j in cj]
                ck = dict(before=distance(stats([imgs[j] for j in cj]), tgt), after=distance(stats(after), tgt))
                pc = write_cube(os.path.join(look_dir, f"{cam}_{clip}.cube"), fc)
                luts[f"{cam}:{clip}"] = dict(match=[clip], cube=os.path.relpath(pc, project.reel_dir),
                                             how=f"{clip} matched to {hero}", check=ck)
                print(f"[reel] look: {clip} matched to {hero}: {ck['before']:.1f} -> {ck['after']:.1f}")
        path = write_cube(os.path.join(look_dir, f"{cam}.cube"), f)
        luts[cam] = dict(match=pats, cube=os.path.relpath(path, project.reel_dir), how=how)
        if cam != hero:
            luts[cam]["check"] = check
            print(f"[reel] look: {cam} {how}: distance to {hero} {check['before']:.1f} -> {check['after']:.1f} "
                  f"(Lab, on frames it wasn't matched on)")
    return luts


def distance(a, b):
    """How far apart two groups' colour is (stats()): the mean gap in their tone curves (L quantiles), plus the gaps
    in a/b means and spreads -- in Lab units, roughly delta-E. 0 is a perfect match; under ~3 reads as one camera."""
    tone = float(np.mean(np.abs(np.array(a["Lq"]) - np.array(b["Lq"]))))
    col = sum(abs(a[c][0] - b[c][0]) + 0.5 * abs(a[c][1] - b[c][1]) for c in ("a", "b"))
    return round(tone + col, 2)


def camera_groups(units, cameras=None):
    """Footage grouped by camera: reel.json "cameras" ({"isaac": ["C2*"], "marcos": ["C0*"]}), else from the clip
    names' letters: {"C": ["C*"], "DJI": ["DJI_*"]}."""
    if cameras:
        return {k: list(v) for k, v in cameras.items()}
    out = {}
    for u in units:
        m = re.match(r"[A-Za-z]+", u["clip"])
        p = m.group(0) if m else "other"
        out.setdefault(p, [f"{p}*" if p != "DJI" else "DJI_*"])
    return out


def summary(ref):
    """Lines describing the reference: its sections by category, the special shots."""
    lines = [f"{len(ref['shots'])} shots over {ref['duration']:.1f} s, music to {ref.get('music_end') or 0:.1f} s"]
    run_, start = None, 0.0
    for s in ref["shots"] + [dict(category=None, t0=ref["duration"])]:
        c = s.get("category") or ("card" if s.get("card") else None)
        if c != run_:
            if run_ is not None:
                lines.append(f"  {start:6.1f}-{s['t0']:6.1f}  {run_}")
            run_, start = c, s["t0"]
    for s in ref["shots"]:
        tags = [k for k in ("title", "split", "card", "after_music") if s.get(k)] + (["stutter"] if "stutter" in s else [])
        if tags:
            lines.append(f"  shot {s['i']:3d} @ {s['t0']:6.1f}: {', '.join(tags)}")
    return lines
